import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, BertConfig, BertModel, AutoConfig
import omegaconf
from tqdm import tqdm
try:
    from ..utils.sampling import top_k_sampling, top_p_sampling, temperature_sampling
    from ..utils.mask import scalar_to_batch_tensor, random_mask, apply_mask, _gamma, mask_by_mink
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

from ..utils.precision_utils import precision_to_dtype


class AccompLLaDA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_name = cfg.model.name
        model_cfg = cfg['model'][self.model_name]

        # Convert to dict if it's a Config object
        if hasattr(model_cfg, 'to_dict'):
            model_cfg_dict = model_cfg.to_dict()
        else:
            model_cfg_dict = model_cfg

        if self.model_name == 'llama':
            lm_dtype = precision_to_dtype(cfg.train.get('precision', '32-true'))
            model_hg_cfg = LlamaConfig(**model_cfg_dict)
            model_hg_cfg.torch_dtype = lm_dtype
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

    def forward(self, voc_ids, acc_ids, attention_mask=None, acc_clamp_embeds=None):
        device = voc_ids.device
        if attention_mask is None:
            attention_mask = (voc_ids != self.special_tokens['voc_pad']).to(device)  # [B, T]

        voc_embeds = self.voc_embed(voc_ids)   # [B, T, dv]
        acc_embeds = self.acc_embed(acc_ids)   # [B, T, da]

        # Process clamp embeddings based on condition type
        if acc_clamp_embeds is not None:
            # Remove the extra dimension [B, 1, 768] -> [B, 768]
            acc_clamp_embeds = acc_clamp_embeds.squeeze(1)

        if not self.cfg.get('clamp_cond') or self.cfg.clamp_cond == 'add':
            if acc_clamp_embeds is not None:
                # Project clamp embeddings if needed
                if hasattr(self, 'clamp_proj'):
                    acc_clamp_embeds = self.clamp_proj(acc_clamp_embeds)
                # Expand to match sequence length
                acc_clamp_embeds = acc_clamp_embeds.unsqueeze(1).expand(-1, voc_ids.size(1), -1)
                acc_embeds = acc_embeds + acc_clamp_embeds
                voc_embeds = voc_embeds + acc_clamp_embeds  # Add to both voc and acc embeddings
            input_embeds = torch.cat([voc_embeds, acc_embeds], dim=-1)
        elif (self.cfg.get('clamp_cond') == 'prefix') and (acc_clamp_embeds is not None):
            # Project clamp embeddings if needed
            if hasattr(self, 'clamp_proj'):
                acc_clamp_embeds = self.clamp_proj(acc_clamp_embeds)
            input_embeds = torch.cat([voc_embeds, acc_embeds], dim=-1)
            P = acc_clamp_embeds.size(1)
            # Concatenate prefix to input embeddings
            input_embeds = torch.cat([acc_clamp_embeds, input_embeds], dim=1)  # [B,P+T,dv+da]

            # Pad attention mask to match new input length
            attention_mask = F.pad(attention_mask, (P, 0), value=True)

        # Sanity check: length consistency
        assert attention_mask.size(1) == input_embeds.size(1), \
            f"mask {attention_mask.size()} vs embeds {input_embeds.size()}"

        # Feed to language model
        if self.model_name == 'conformer':
            out = self.lm(x=input_embeds, mask=attention_mask)
        else:
            out = self.lm(inputs_embeds=input_embeds, attention_mask=attention_mask).last_hidden_state

        logits = self.to_logits(out)
        return logits

    def generate(self,
                 voc_ids,
                 acc_clamp_embeds=None,
                 sampling_steps=64,
                 top_p=0.8,         # 建议保留，过滤低概率噪声
                 top_k=50,          # 建议保留
                 temperature=1.0,
                 mask_temperature=10.5, # 引入 MaskGIT 的 Gumbel 噪声 Trick
                 schedule='cosine'):    # 引入非线性调度
        """
        融合 LLaDA 的 Pure Diffusion 理论与 MaskGIT 的听感优化 Trick。
        """
        """
        batch_size: 1
        """
        B, T = voc_ids.shape[0], voc_ids.shape[1]
        assert B == 1
        voc_ids = voc_ids.squeeze(0)

        # ---- 1) Calculate prefix length and normalize acc_clamp_embeds shape ----
        prefix_len = acc_clamp_embeds.shape[1]
        device = voc_ids.device
        acc_clamp_embeds = acc_clamp_embeds.to(self.clamp_proj.weight.dtype)
        print(f"prefix_len: {prefix_len}")


        # 1. 初始化：全 Mask
        acc_ids_curr = torch.full_like(voc_ids, self.special_tokens['acc_mask'])
        attn_mask = torch.ones(1, T, dtype=torch.bool, device=device)

        # 2. 调度函数选择：解决“杂音”的关键
        def get_mask_ratio(step):
            r = step / sampling_steps
            if schedule == 'cosine':
                return torch.cos(torch.tensor(r * 3.14159 / 2)).item()
            return 1 - r # Linear

        for i in range(sampling_steps):
            # 当前 step 的掩码比例 t
            t = get_mask_ratio(i)
            # 下一个 step 的掩码比例 s
            s = get_mask_ratio(i + 1)

            # ----- Forward -----
            logits = self.forward(
                voc_ids=voc_ids[None],
                acc_ids=acc_ids_curr[None],
                attention_mask=attn_mask,
                acc_clamp_embeds=acc_clamp_embeds
            ).squeeze(0)

            logits_acc = logits[prefix_len:] if prefix_len > 0 else logits

            # ----- 采样预测 x0 (带 Trick) -----
            # 使用 top_k/top_p 过滤掉那些导致“不协和音”的低概率候选
            if top_p is not None:
                logits_acc = top_p_sampling(logits_acc, top_p=top_p)
            if top_k is not None:
                logits_acc = top_k_sampling(logits_acc, top_k=top_k)

            probs = torch.softmax(logits_acc / temperature, dim=-1)
            pred_ids = torch.multinomial(probs, 1).squeeze(-1)

            # 计算原始置信度
            confidences = torch.gather(probs, -1, pred_ids.unsqueeze(-1)).squeeze(-1)

            # 根据论文逻辑，confidence 决定了哪些 token 被 remask
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(confidences) + 1e-9) + 1e-9)
            # 随采样进度退火的噪声
            temp = mask_temperature * t
            fuzzed_confidences = confidences + temp * gumbel_noise

            # ----- 重掩码比例计算 -----
            n_unmasked = int(T * (1.0 - s))

            if i == sampling_steps - 1 or n_unmasked >= T:
                acc_ids_curr = pred_ids
                break

            # 选取置信度最高（受扰动后）的 n_unmasked 个位置保留
            _, topk_indices = torch.topk(fuzzed_confidences, k=n_unmasked)

            # 构造下一轮的输入：Pure Diffusion 模式（不锁定，全部重刷） [cite: 316]
            new_ids = torch.full_like(pred_ids, self.special_tokens['acc_mask'])
            new_ids[topk_indices] = pred_ids[topk_indices]
            acc_ids_curr = new_ids

        return acc_ids_curr
