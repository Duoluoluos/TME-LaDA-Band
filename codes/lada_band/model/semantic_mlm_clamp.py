import hashlib

import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, BertConfig, BertModel, AutoConfig
import omegaconf
from tqdm import tqdm
try:
    from ..utils.sampling import top_k_sampling, top_p_sampling, temperature_sampling
    from ..utils.mask import scalar_to_batch_tensor, random_mask, apply_mask, _gamma, mask_by_mink
    from ..utils.remask_schedule import build_remask_ratios
    try:
        from .llama import LlamaModel
    except:
        from .llama_codeclm_env import LlamaModel
    # from .conformer import Conformer (need gateloop)
except:
    import sys
    sys.path.append('../')
    from utils.sampling import top_k_sampling, top_p_sampling, temperature_sampling
    from utils.mask import scalar_to_batch_tensor, random_mask, apply_mask, _gamma, mask_by_mink
    from utils.remask_schedule import build_remask_ratios

from ..utils.precision_utils import precision_to_dtype


class SemanticMLMClampModel(nn.Module):
    @staticmethod
    def _cfg_node_to_dict(node):
        if node is None:
            return {}
        if hasattr(node, 'to_dict'):
            return node.to_dict()
        return dict(node)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_name = cfg.model.name
        # Convert to dict if it's a Config object to allow ** unpacking
        model_cfg = cfg['model'][self.model_name]
        if hasattr(model_cfg, 'to_dict'):
            model_cfg_dict = model_cfg.to_dict()
        else:
            model_cfg_dict = model_cfg

        if self.model_name == 'llama':
            lm_dtype = precision_to_dtype(cfg.train.get('precision', '32-true'))
            llama_cfg_dict = dict(model_cfg_dict)
            llama_cfg_dict.pop('json_path', None)
            pretrained_cfg = self._cfg_node_to_dict(model_cfg.get('pretrained', {}))
            llama_cfg_dict.pop('pretrained', None)

            model_hg_cfg = LlamaConfig(**llama_cfg_dict)
            model_hg_cfg.torch_dtype = lm_dtype
            pretrained_enabled = bool(pretrained_cfg.get('enabled', False))
            pretrained_path = pretrained_cfg.get('path') or model_cfg.get('json_path')

            if pretrained_enabled:
                if not pretrained_path:
                    raise ValueError("model.llama.pretrained.enabled=true but no pretrained path/json_path was provided")

                load_kwargs = {
                    'config': model_hg_cfg,
                    'ignore_mismatched_sizes': bool(pretrained_cfg.get('ignore_mismatched_sizes', True)),
                    'local_files_only': bool(pretrained_cfg.get('local_files_only', True)),
                    'torch_dtype': lm_dtype,
                }
                use_safetensors = pretrained_cfg.get('use_safetensors', None)
                if use_safetensors is not None:
                    load_kwargs['use_safetensors'] = bool(use_safetensors)

                self.lm = LlamaModel.from_pretrained(pretrained_path, **load_kwargs)
                print(
                    "[SemanticMLMClampModel] loaded pretrained backbone "
                    f"from {pretrained_path} "
                    f"(ignore_mismatched_sizes={load_kwargs['ignore_mismatched_sizes']})"
                )
            else:
                self.lm = LlamaModel(model_hg_cfg)

            self.lm.embed_tokens.requires_grad_(False) # freeze for ddp
        elif self.model_name == 'bert':
            model_hg_cfg = BertConfig(**model_cfg_dict)
            self.lm = BertModel(model_hg_cfg, add_pooling_layer=False)
            self.lm.embeddings.word_embeddings.requires_grad_(False)  # freeze for ddp

        hidden_size = model_cfg.hidden_size if model_cfg.get('hidden_size') else model_cfg.dim
        assert hidden_size % 2 == 0
        self.voc_embed = nn.Embedding(cfg.data.voc.size + 1, hidden_size//2)
        self.acc_embed = nn.Embedding(cfg.data.acc.size + 2, hidden_size//2)

        self.special_tokens = cfg.data.special_tokens

        self.to_logits = nn.Linear(hidden_size, cfg.data.acc.size, bias=False)  # TODO bias=False
        self.rtd_enabled = bool(cfg.loss.get('rtd', {}).get('enabled', False))
        if self.rtd_enabled:
            self.rtd_head = nn.Linear(hidden_size, 1)

        # Clamp3 embedding dimension is 768, need to handle dimension matching
        clamp_dim = 768  # From clamp3.py, it returns [B, 1, 768]

        # Handle dimension mismatch between clamp3 and model
        if not self.cfg.get('clamp_cond') or self.cfg.clamp_cond == 'add':
            if clamp_dim != hidden_size // 2:
                self.clamp_proj = nn.Linear(clamp_dim, hidden_size//2)
        elif self.cfg.clamp_cond == 'prefix':
            if clamp_dim != hidden_size:
                self.clamp_proj = nn.Linear(clamp_dim, hidden_size)

    def cal_parameter_num(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def device(self):
        return next(self.parameters()).device

    def _build_causal_attention_mask(self, attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if attention_mask.dim() != 2:
            raise ValueError(
                f"causal_attention expects a 2D attention mask, got shape={tuple(attention_mask.shape)}"
            )

        batch_size, seq_len = attention_mask.shape
        device = attention_mask.device
        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full((seq_len, seq_len), fill_value=min_dtype, dtype=dtype, device=device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len).clone()

        key_padding_mask = attention_mask[:, None, None, :].eq(0)
        causal_mask = causal_mask.masked_fill(key_padding_mask, min_dtype)
        return causal_mask

    def _prepare_conditioning(self, acc_clamp_embeds, seq_len: int, target_dtype: torch.dtype):
        prefix_embeds = None
        add_cond = None

        if not self.cfg.get('clamp_cond') or self.cfg.clamp_cond == 'add':
            if acc_clamp_embeds is not None:
                if acc_clamp_embeds.dim() == 3:
                    if acc_clamp_embeds.size(1) == 1:
                        acc_clamp_embeds = acc_clamp_embeds.squeeze(1)
                    else:
                        acc_clamp_embeds = acc_clamp_embeds.mean(dim=1)
                if hasattr(self, 'clamp_proj'):
                    acc_clamp_embeds = self.clamp_proj(acc_clamp_embeds)
                add_cond = acc_clamp_embeds.to(target_dtype).unsqueeze(1).expand(-1, seq_len, -1)
        elif (self.cfg.get('clamp_cond') == 'prefix') and (acc_clamp_embeds is not None):
            if acc_clamp_embeds.dim() == 2:
                acc_clamp_embeds = acc_clamp_embeds.unsqueeze(1)
            if hasattr(self, 'clamp_proj'):
                acc_clamp_embeds = self.clamp_proj(acc_clamp_embeds)
            prefix_embeds = acc_clamp_embeds.to(target_dtype)

        return prefix_embeds, add_cond

    def forward(
        self,
        voc_ids,
        acc_ids,
        attention_mask=None,
        acc_clamp_embeds=None,
        return_hidden_states=False,
        causal_attention: bool = False,
    ):
        device = voc_ids.device
        if attention_mask is None:
            attention_mask = (voc_ids != self.special_tokens['voc_pad']).to(device)  # [B, T]

        voc_embeds = self.voc_embed(voc_ids)   # [B, T, dv]
        acc_embeds = self.acc_embed(acc_ids)   # [B, T, da]
        prefix_embeds, add_cond = self._prepare_conditioning(
            acc_clamp_embeds,
            seq_len=voc_ids.size(1),
            target_dtype=voc_embeds.dtype,
        )
        if add_cond is not None:
            acc_embeds = acc_embeds + add_cond
            voc_embeds = voc_embeds + add_cond

        input_embeds = torch.cat([voc_embeds, acc_embeds], dim=-1)
        if prefix_embeds is not None:
            prefix_len = prefix_embeds.size(1)
            input_embeds = torch.cat([prefix_embeds, input_embeds], dim=1)
            if attention_mask.dim() != 2:
                raise ValueError("prefix conditioning currently expects a 2D attention mask")
            attention_mask = F.pad(attention_mask, (prefix_len, 0), value=True)

        if causal_attention:
            attention_mask = self._build_causal_attention_mask(attention_mask, input_embeds.dtype)
        else:
            assert attention_mask.size(1) == input_embeds.size(1), \
                f"mask {attention_mask.size()} vs embeds {input_embeds.size()}"

        # Feed to language model
        if self.model_name == 'conformer':
            out = self.lm(x=input_embeds, mask=attention_mask)
        else:
            out = self.lm(inputs_embeds=input_embeds, attention_mask=attention_mask).last_hidden_state

        logits = self.to_logits(out)
        if return_hidden_states:
            return logits, out
        return logits

    def compute_rtd_logits(self, hidden_states):
        if not self.rtd_enabled:
            raise RuntimeError("RTD head is disabled in config")
        return self.rtd_head(hidden_states).squeeze(-1)

    def _prepare_generation_inputs(self, voc_ids, acc_clamp_embeds=None):
        B, T = voc_ids.shape[0], voc_ids.shape[1]
        if B != 1:
            raise ValueError(f"generate currently only supports batch_size=1, got {B}")

        voc_ids = voc_ids.squeeze(0)
        prefix_len = 0
        if acc_clamp_embeds is not None:
            if acc_clamp_embeds.dim() == 2:
                acc_clamp_embeds = acc_clamp_embeds.unsqueeze(1)
            prefix_len = acc_clamp_embeds.shape[1]
            target_dtype = self.clamp_proj.weight.dtype if hasattr(self, 'clamp_proj') else self.voc_embed.weight.dtype
            acc_clamp_embeds = acc_clamp_embeds.to(target_dtype)

        print(f"prefix_len: {prefix_len}")
        return voc_ids, T, prefix_len, acc_clamp_embeds

    def _prepare_generation_embeddings(self, voc_ids, acc_clamp_embeds=None):
        voc_ids, T, prefix_len, acc_clamp_embeds = self._prepare_generation_inputs(
            voc_ids,
            acc_clamp_embeds=acc_clamp_embeds,
        )
        voc_embeds = self.voc_embed(voc_ids[None])
        prefix_embeds, add_cond = self._prepare_conditioning(
            acc_clamp_embeds,
            seq_len=T,
            target_dtype=voc_embeds.dtype,
        )
        if add_cond is not None:
            voc_embeds = voc_embeds + add_cond
        return voc_ids, T, prefix_len, voc_embeds, prefix_embeds, add_cond

    def _sample_from_logits(self, logits, top_p=None, top_k=None, temperature=1.0):
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        logits = logits.clone()
        if top_p is not None:
            logits = top_p_sampling(logits, top_p=top_p)
        if top_k is not None:
            logits = top_k_sampling(logits, top_k=top_k)

        probs = temperature_sampling(logits=logits, temp=temperature)
        sampled = torch.multinomial(probs, 1).long().squeeze(-1)
        sel_probs = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        return sampled, sel_probs

    @staticmethod
    def _normalize_remask_strategy(strategy):
        normalized = str(strategy or 'low_confidence').strip().lower().replace('-', '_')
        if normalized == 'no_remask':
            normalized = 'one_shot'
        valid = {'low_confidence', 'random', 'fixed_position', 'one_shot'}
        if normalized not in valid:
            raise ValueError(
                f"Unsupported remask_strategy={strategy}. "
                "Expected one of: low_confidence, random, fixed_position, one_shot."
            )
        return normalized

    @staticmethod
    def _stable_seed(seed, sequence_id, step_index, salt):
        base_seed = 0 if seed is None else int(seed)
        text = f"{base_seed}|{sequence_id or 'unknown'}|{int(step_index)}|{salt}"
        digest = hashlib.blake2b(text.encode('utf-8'), digest_size=8).digest()
        return int.from_bytes(digest, 'big') % (2**63 - 1)

    def _gumbel_noise_like(self, tensor, seed, sequence_id, step_index, salt):
        generator = torch.Generator(device='cpu')
        generator.manual_seed(self._stable_seed(seed, sequence_id, step_index, salt))
        noise = torch.empty(tensor.shape, dtype=torch.float32, device='cpu')
        noise.uniform_(1e-20, 1.0, generator=generator)
        noise = noise.to(device=tensor.device, dtype=tensor.dtype)
        return -torch.log(-torch.log(noise))

    def _random_scores(self, length, device, seed, sequence_id, step_index):
        generator = torch.Generator(device='cpu')
        generator.manual_seed(self._stable_seed(seed, sequence_id, step_index, 'random_remask'))
        scores = torch.rand(length, generator=generator, dtype=torch.float32, device='cpu')
        return scores.to(device=device)

    def _fixed_position_scores(self, length, device, seed, sequence_id):
        base_seed = 0 if seed is None else int(seed)
        seq_key = sequence_id or 'unknown'
        denom = float(2**64 - 1)
        values = []
        for pos in range(length):
            text = f"{base_seed}|{seq_key}|{pos}|fixed_position"
            digest = hashlib.blake2b(text.encode('utf-8'), digest_size=8).digest()
            values.append(int.from_bytes(digest, 'big') / denom)
        return torch.tensor(values, dtype=torch.float32, device=device)

    def _select_remask_mask(
        self,
        cur_mask,
        sel_probs,
        num_to_mask,
        strategy,
        seed,
        sequence_id,
        step_index,
        selection_temperature,
        fixed_position_scores=None,
    ):
        candidate_idx = torch.nonzero(cur_mask, as_tuple=False).squeeze(-1)
        candidate_count = int(candidate_idx.numel())
        n = int(num_to_mask.reshape(-1)[0].item()) if torch.is_tensor(num_to_mask) else int(num_to_mask)
        n = max(0, min(n, candidate_count))

        new_mask = torch.zeros_like(cur_mask, dtype=torch.bool)
        if n == 0 or candidate_count == 0:
            return new_mask
        if n == candidate_count:
            new_mask[candidate_idx] = True
            return new_mask

        if strategy == 'low_confidence':
            scores = torch.log(sel_probs.detach().clamp_min(1e-20))
            if selection_temperature > 0:
                scores = scores + selection_temperature * self._gumbel_noise_like(
                    sel_probs.detach(),
                    seed=seed,
                    sequence_id=sequence_id,
                    step_index=step_index,
                    salt='low_confidence_remask',
                )
        elif strategy == 'random':
            scores = self._random_scores(
                length=cur_mask.numel(),
                device=cur_mask.device,
                seed=seed,
                sequence_id=sequence_id,
                step_index=step_index,
            )
        elif strategy == 'fixed_position':
            scores = fixed_position_scores
            if scores is None:
                scores = self._fixed_position_scores(
                    length=cur_mask.numel(),
                    device=cur_mask.device,
                    seed=seed,
                    sequence_id=sequence_id,
                )
        else:
            raise ValueError(f"Unsupported remask_strategy={strategy}")

        chosen_rel = torch.topk(scores[candidate_idx], k=n, largest=False).indices
        new_mask[candidate_idx[chosen_rel]] = True
        return new_mask

    @staticmethod
    def _append_remask_trace(
        remask_trace,
        sequence_id,
        strategy,
        step_index,
        sampling_steps,
        remask_ratio,
        cur_mask,
        sel_probs,
        num_to_mask,
        new_mask,
        selection_temperature,
    ):
        if remask_trace is None:
            return

        mask_conf = sel_probs.detach()[cur_mask]
        if mask_conf.numel() > 0:
            mean_conf = float(mask_conf.mean().item())
            min_conf = float(mask_conf.min().item())
            max_conf = float(mask_conf.max().item())
        else:
            mean_conf = min_conf = max_conf = None

        n = int(num_to_mask.reshape(-1)[0].item()) if torch.is_tensor(num_to_mask) else int(num_to_mask)
        remask_trace.append({
            'sequence_id': sequence_id,
            'strategy': strategy,
            'step_index': int(step_index),
            'step': int(step_index) + 1,
            'sampling_steps': int(sampling_steps),
            'remask_ratio': float(remask_ratio),
            'mask_count_start': int(cur_mask.sum().item()),
            'num_to_mask': int(n),
            'actual_remask_count': int(new_mask.sum().item()),
            'mean_confidence': mean_conf,
            'min_confidence': min_conf,
            'max_confidence': max_conf,
            'selection_temperature': float(selection_temperature),
        })

    def _generate_diffusion(
        self,
        voc_ids,
        acc_clamp_embeds=None,
        sampling_steps=36,
        top_p=None,
        top_k=None,
        temperature=1.0,
        mask_temperature=10.5,
        remask_schedule='cosine',
        remask_ratios=None,
        remask_strategy='low_confidence',
        remask_seed=0,
        remask_sequence_id=None,
        remask_trace=None,
    ):
        voc_ids, T, prefix_len, acc_clamp_embeds = self._prepare_generation_inputs(
            voc_ids,
            acc_clamp_embeds=acc_clamp_embeds,
        )
        remask_strategy = self._normalize_remask_strategy(remask_strategy)
        scheduled_ratios = build_remask_ratios(
            sampling_steps=sampling_steps,
            schedule_spec=remask_schedule,
            remask_ratios=remask_ratios,
        )

        attn_mask = torch.ones(1, T, dtype=torch.bool, device=voc_ids.device)
        acc_ids_masked = torch.full_like(voc_ids, self.special_tokens['acc_mask'])
        fixed_position_scores = None
        if remask_strategy == 'fixed_position':
            fixed_position_scores = self._fixed_position_scores(
                length=T,
                device=voc_ids.device,
                seed=remask_seed,
                sequence_id=remask_sequence_id,
            )

        for i in range(sampling_steps):
            logits = self.forward(
                voc_ids=voc_ids[None],
                acc_ids=acc_ids_masked[None],
                attention_mask=attn_mask,
                acc_clamp_embeds=acc_clamp_embeds
            ).squeeze(0)

            logits_acc = logits[prefix_len:] if prefix_len > 0 else logits
            sampled, sel_probs = self._sample_from_logits(
                logits_acc,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
            )

            cur_mask = (acc_ids_masked == self.special_tokens['acc_mask'])
            tokens = torch.where(cur_mask, sampled, acc_ids_masked)
            sel_probs = torch.where(
                cur_mask,
                sel_probs,
                torch.full_like(sel_probs, float('inf')),
            )

            r = torch.tensor([(i + 1) / sampling_steps], device=voc_ids.device)
            remask_ratio = scheduled_ratios[i]
            num_to_mask = torch.floor(
                torch.tensor([remask_ratio * T], device=voc_ids.device)
            ).long()

            if remask_strategy == 'one_shot':
                empty_mask = torch.zeros_like(cur_mask, dtype=torch.bool)
                self._append_remask_trace(
                    remask_trace=remask_trace,
                    sequence_id=remask_sequence_id,
                    strategy=remask_strategy,
                    step_index=i,
                    sampling_steps=sampling_steps,
                    remask_ratio=0.0,
                    cur_mask=cur_mask,
                    sel_probs=sel_probs,
                    num_to_mask=torch.zeros_like(num_to_mask),
                    new_mask=empty_mask,
                    selection_temperature=0.0,
                )
                return tokens

            if i != sampling_steps - 1:
                num_to_mask = torch.max(
                    torch.tensor(1, device=voc_ids.device),
                    torch.min(cur_mask.sum().sub(1), num_to_mask)
                )

            selection_temperature = mask_temperature * (1 - r.item())
            new_mask = self._select_remask_mask(
                cur_mask=cur_mask,
                sel_probs=sel_probs,
                num_to_mask=num_to_mask,
                strategy=remask_strategy,
                seed=remask_seed,
                sequence_id=remask_sequence_id,
                step_index=i,
                selection_temperature=selection_temperature,
                fixed_position_scores=fixed_position_scores,
            ).to(tokens.device)
            self._append_remask_trace(
                remask_trace=remask_trace,
                sequence_id=remask_sequence_id,
                strategy=remask_strategy,
                step_index=i,
                sampling_steps=sampling_steps,
                remask_ratio=remask_ratio,
                cur_mask=cur_mask,
                sel_probs=sel_probs,
                num_to_mask=num_to_mask,
                new_mask=new_mask,
                selection_temperature=selection_temperature,
            )

            acc_ids_masked = torch.where(
                new_mask,
                self.special_tokens['acc_mask'],
                tokens,
            )

        return tokens

    def _generate_autoregressive(
        self,
        voc_ids,
        acc_clamp_embeds=None,
        top_p=None,
        top_k=None,
        temperature=1.0,
    ):
        voc_ids, T, prefix_len, voc_embeds, prefix_embeds, add_cond = self._prepare_generation_embeddings(
            voc_ids,
            acc_clamp_embeds=acc_clamp_embeds,
        )

        past_key_values = None
        total_prefix_len = prefix_len
        if prefix_embeds is not None:
            prefix_attention_mask = torch.ones(
                1,
                prefix_embeds.size(1),
                dtype=torch.bool,
                device=voc_ids.device,
            )
            prefix_outputs = self.lm(
                inputs_embeds=prefix_embeds,
                attention_mask=prefix_attention_mask,
                past_key_values=None,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = prefix_outputs.past_key_values

        generated_ids = torch.full(
            (1, T),
            self.special_tokens['acc_pad'],
            dtype=voc_ids.dtype,
            device=voc_ids.device,
        )

        for step in range(T):
            if step == 0:
                acc_input_ids = torch.full(
                    (1, 1),
                    self.special_tokens['acc_mask'],
                    dtype=voc_ids.dtype,
                    device=voc_ids.device,
                )
            else:
                acc_input_ids = generated_ids[:, step - 1:step]

            acc_embeds = self.acc_embed(acc_input_ids)
            if add_cond is not None:
                acc_embeds = acc_embeds + add_cond[:, step:step + 1]
            step_input_embeds = torch.cat([voc_embeds[:, step:step + 1], acc_embeds], dim=-1)

            attention_mask = torch.ones(
                1,
                total_prefix_len + step + 1,
                dtype=torch.bool,
                device=voc_ids.device,
            )
            outputs = self.lm(
                inputs_embeds=step_input_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            step_logits = self.to_logits(outputs.last_hidden_state[:, -1, :]).squeeze(0)
            next_token, _ = self._sample_from_logits(
                step_logits,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
            )
            generated_ids[:, step] = next_token[0]

        return generated_ids.squeeze(0)

    def generate(self,
             voc_ids,
             acc_clamp_embeds=None,
             sampling_steps=36,
             top_p=None,
             top_k=None,
             temperature=1.0,
             mask=None,                 # Keep signature but don't use this name
             mask_temperature=10.5,
             remask_schedule='cosine',
             remask_ratios=None,
             remask_strategy='low_confidence',
             remask_seed=0,
             remask_sequence_id=None,
             remask_trace=None,
             decode_mode='diffusion'):
        decode_mode = str(decode_mode).lower()
        if decode_mode in {'diffusion', 'mlm'}:
            return self._generate_diffusion(
                voc_ids=voc_ids,
                acc_clamp_embeds=acc_clamp_embeds,
                sampling_steps=sampling_steps,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                mask_temperature=mask_temperature,
                remask_schedule=remask_schedule,
                remask_ratios=remask_ratios,
                remask_strategy=remask_strategy,
                remask_seed=remask_seed,
                remask_sequence_id=remask_sequence_id,
                remask_trace=remask_trace,
            )
        if decode_mode in {'autoregressive', 'ar', 'causal'}:
            return self._generate_autoregressive(
                voc_ids=voc_ids,
                acc_clamp_embeds=acc_clamp_embeds,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
            )
        raise ValueError(
            f"Unsupported decode_mode={decode_mode}. "
            "Expected one of: diffusion, maskgit, mlm, autoregressive, ar, causal."
        )
