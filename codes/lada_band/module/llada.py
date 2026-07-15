import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import functools
import omegaconf
from omegaconf import OmegaConf
import lightning as L
from transformers import get_scheduler
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification import MulticlassAccuracy
from ..utils.mask import sample_mask, apply_mask
from ..model.clamp3 import Clamp3
import os
import sys
import re
_MUCODEC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'MuCodec')
_MUCODEC_DIR = os.path.normpath(os.path.abspath(_MUCODEC_DIR))
if _MUCODEC_DIR not in sys.path:
    sys.path.insert(0, _MUCODEC_DIR)
from beartype.typing import List, Optional, Tuple, Union
from ..model.semantic_mlm_clamp import SemanticMLMClampModel
from ..utils.condition import ClassifierFreeGuidanceDropout
from ..utils.audio_utils import _av_read, _av_info, AudioFileInfo, convert_audio
from generate import MuCodec


class MoALinear(nn.Module):
    """Mixture-of-Adapters linear wrapper (LoRA-style low-rank experts)."""
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        num_experts: int = 4,
        alpha: float = 16.0,
        router_temperature: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"MoALinear expects nn.Linear, got {type(base_layer)}")
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")
        if num_experts <= 0:
            raise ValueError(f"num_experts must be > 0, got {num_experts}")

        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

        self.rank = int(rank)
        self.num_experts = int(num_experts)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.router_temperature = float(router_temperature)
        self.dropout = nn.Dropout(float(dropout))

        in_features = self.base_layer.in_features
        out_features = self.base_layer.out_features

        self.router = nn.Linear(in_features, self.num_experts, bias=False)
        self.lora_A = nn.Parameter(torch.empty(self.num_experts, in_features, self.rank))
        self.lora_B = nn.Parameter(torch.empty(self.num_experts, self.rank, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.router.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)

        x_dropped = self.dropout(x)
        router_logits = self.router(x_dropped)
        if self.router_temperature != 1.0:
            router_logits = router_logits / self.router_temperature
        router_weights = torch.softmax(router_logits, dim=-1)

        # x -> each expert low-rank branch -> weighted sum over experts
        projected = torch.einsum('...i,eir->...er', x_dropped, self.lora_A)
        expert_out = torch.einsum('...er,ero->...eo', projected, self.lora_B)
        mixed_out = (expert_out * router_weights.unsqueeze(-1)).sum(dim=-2)
        return base_out + self.scaling * mixed_out


def _get_parent_module(root: nn.Module, module_name: str):
    if '.' not in module_name:
        return root, module_name
    parent_name, child_name = module_name.rsplit('.', 1)
    parent = root.get_submodule(parent_name)
    return parent, child_name


class ModelModule(L.LightningModule):
    def __init__(self, cfg, mode='train'):
        super().__init__()
        # self.save_hyperparameters(ignore=['model'])

        # 1. 初始化主模型 (先不急着转 float16，最后统一转)
        self.model = SemanticMLMClampModel(cfg)

        self.cfg = cfg
        self.lr = cfg.optim.lr
        self._moa_wrapped_module_names = set()
        self.moa_enabled = bool(cfg.get('moa', {}).get('enabled', False))
        self._moa_trainable_extra = set(cfg.get('moa', {}).get('trainable_extra', []))
        progressive_cfg = self._cfg_node_to_dict(cfg.train.get('progressive_unfreeze', {}))
        default_progressive_modules = ['voc_embed', 'acc_embed', 'clamp_proj', 'to_logits']
        if getattr(self.model, 'rtd_enabled', False):
            default_progressive_modules.append('rtd_head')
        self.progressive_unfreeze_enabled = bool(progressive_cfg.get('enabled', False))
        self._progressive_trainable_modules = list(
            progressive_cfg.get('trainable_modules', default_progressive_modules)
        )
        self._progressive_initial_unfrozen_layers = self._normalize_unfrozen_layers(
            progressive_cfg.get('initial_unfrozen_layers', 0)
        )
        self._progressive_milestones = self._build_progressive_schedule(progressive_cfg.get('milestones', []))
        self._current_unfrozen_layers = None
        rtd_cfg = self._cfg_node_to_dict(cfg.loss.get('rtd', {}))
        self.rtd_enabled = bool(rtd_cfg.get('enabled', False))
        self.rtd_weight = float(rtd_cfg.get('weight', 0.2))
        self.rtd_sampling = str(rtd_cfg.get('sampling', 'argmax')).lower()
        self.rtd_temperature = float(rtd_cfg.get('temperature', 1.0))
        self.rtd_backbone_grad = bool(rtd_cfg.get('backbone_grad', False))
        self.backbone_gradient_checkpointing_enabled = bool(cfg.train.get("gradient_checkpointing", False))
        if self.rtd_enabled and self.rtd_backbone_grad and self.backbone_gradient_checkpointing_enabled:
            print(
                "[RTD] gradient_checkpointing=True with a second backbone forward is incompatible "
                "with DDP reentrant backward; forcing loss.rtd.backbone_grad=false."
            )
            self.rtd_backbone_grad = False

        if self.moa_enabled and self.progressive_unfreeze_enabled:
            print("[trainability] moa.enabled=true; progressive_unfreeze will be ignored.")
            self.progressive_unfreeze_enabled = False

        if self.moa_enabled:
            self._enable_moa_peft()
        elif self.progressive_unfreeze_enabled:
            print(
                "[trainability] progressive unfreeze enabled "
                f"with initial_unfrozen_layers={self._progressive_initial_unfrozen_layers}"
            )

        if cfg.loss.name == 'ce':
            self.loss_func = F.cross_entropy

        self.top1_accuracy = MulticlassAccuracy(
            cfg.data.acc.size,
            top_k=1,
            average='micro',
            multidim_average='global',
            ignore_index=self.model.special_tokens['acc_pad']
        )
        self.top10_accuracy = MulticlassAccuracy(
            cfg.data.acc.size,
            top_k=10,
            average="micro",
            multidim_average="global",
            ignore_index=self.model.special_tokens['acc_pad']
        )
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True, seed=8)
        self._runtime_debug_printed = False
        self._ar_rtd_warning_printed = False

        # 2. 初始化 Clamp3 (它内部可能有自己的加载逻辑)
        self.clamp3 = Clamp3(precision=cfg.train.precision, ckpt_path=cfg.clamp3_path)
        self.clamp3.eval()
        self.clamp3.requires_grad_(False)

        self.clamp_drop = ClassifierFreeGuidanceDropout(p=cfg.clamp_dropout)

        # 3. 初始化 Codec 并加载权重
        if 'codec_path' in cfg:
            # New MuCodec initialization
            self.codec = MuCodec(
                model_path=cfg.codec_path,
                layer_num=cfg.get('codec_layer_num', 7), # Default to 7 if not specified
                load_main_model=True,
                device='cuda' # Initial device, will be moved by Lightning or manually
            )

            # Register codec submodules for Lightning to track/move them
            # MuCodec wrapper has .model (PromptCondAudioDiffusion), .vae, .stft
            self.codec_model = self.codec.model
            self.codec_vae = self.codec.vae
            self.codec_stft = self.codec.stft

            self.codec.model.eval()
            self.codec.vae.eval()
            self.codec.model.requires_grad_(False)
            self.codec.vae.requires_grad_(False)
            # stft usually has no parameters but good to be safe
            if hasattr(self.codec.stft, 'eval'):
                self.codec.stft.eval()
            if isinstance(self.codec.stft, nn.Module):
                self.codec.stft.requires_grad_(False)



        # 再次确保处于 eval 模式的模块不被重置
        self.clamp3.eval()
        if hasattr(self, 'codec'):
            self.codec.model.eval()
            self.codec.model.requires_grad_(False)
            self.codec.vae.requires_grad_(False)
            if isinstance(self.codec.stft, nn.Module):
                self.codec.stft.requires_grad_(False)

        # VRAM optimization:
        # - Don't force-cast the whole LightningModule (would upcast frozen large submodules to fp32).
        # - Optionally enable gradient checkpointing on the backbone to reduce activations.
        if self.backbone_gradient_checkpointing_enabled and hasattr(self.model, "lm"):
            lm = self.model.lm
            if hasattr(lm, "gradient_checkpointing_enable"):
                try:
                    lm.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                except TypeError:
                    lm.gradient_checkpointing_enable()
            elif hasattr(lm, "gradient_checkpointing"):
                lm.gradient_checkpointing = True
            # DDP is much less fragile with non-reentrant activation checkpointing.
            lm._gradient_checkpointing_func = functools.partial(
                torch.utils.checkpoint.checkpoint,
                use_reentrant=False,
            )
            if hasattr(lm, "gradient_checkpointing"):
                lm.gradient_checkpointing = True

        # Final guard: enforce PEFT trainability after all init/to() calls.
        self._apply_trainability_policy(current_step=0, force_log=True)

    @staticmethod
    def _cfg_node_to_dict(node):
        if node is None:
            return {}
        if hasattr(node, 'to_dict'):
            return node.to_dict()
        return dict(node)

    def _get_total_backbone_layers(self) -> int:
        layers = getattr(getattr(self.model, 'lm', None), 'layers', None)
        return len(layers) if layers is not None else 0

    def _normalize_unfrozen_layers(self, value) -> int:
        total_layers = self._get_total_backbone_layers()
        if isinstance(value, str):
            if value.lower() == 'all':
                return total_layers
            value = int(value)
        elif value is None:
            value = 0
        else:
            value = int(value)

        if value < 0:
            return total_layers
        return min(value, total_layers)

    def _build_progressive_schedule(self, milestones):
        schedule = []
        for item in milestones or []:
            item_dict = self._cfg_node_to_dict(item)
            if 'step' not in item_dict:
                raise ValueError("train.progressive_unfreeze.milestones items must contain `step`")
            layers_value = item_dict.get('unfrozen_layers', item_dict.get('backbone_trainable_layers', 0))
            schedule.append({
                'step': int(item_dict['step']),
                'unfrozen_layers': self._normalize_unfrozen_layers(layers_value),
            })
        schedule.sort(key=lambda x: x['step'])
        return schedule

    def _freeze_auxiliary_modules(self) -> None:
        if hasattr(self, "clamp3") and isinstance(self.clamp3, nn.Module):
            self.clamp3.requires_grad_(False)
        if hasattr(self, "codec_model") and isinstance(self.codec_model, nn.Module):
            self.codec_model.requires_grad_(False)
        if hasattr(self, "codec_vae") and isinstance(self.codec_vae, nn.Module):
            self.codec_vae.requires_grad_(False)
        if hasattr(self, "codec_stft") and isinstance(self.codec_stft, nn.Module):
            self.codec_stft.requires_grad_(False)

    def _matches_progressive_target(self, param_name: str) -> bool:
        for target in self._progressive_trainable_modules:
            if param_name == target or param_name.startswith(target + '.'):
                return True
        return False

    def _resolve_unfrozen_layers(self, current_step: int) -> int:
        unfrozen_layers = self._progressive_initial_unfrozen_layers
        for item in self._progressive_milestones:
            if current_step >= item['step']:
                unfrozen_layers = item['unfrozen_layers']
            else:
                break
        return unfrozen_layers

    def _apply_progressive_unfreeze(self, current_step: int, force_log: bool = False) -> None:
        self._freeze_auxiliary_modules()
        self.model.requires_grad_(False)

        if hasattr(self.model, 'lm') and hasattr(self.model.lm, 'embed_tokens'):
            self.model.lm.embed_tokens.requires_grad_(False)

        for name, param in self.model.named_parameters():
            if self._matches_progressive_target(name):
                param.requires_grad = True

        total_layers = self._get_total_backbone_layers()
        unfrozen_layers = self._resolve_unfrozen_layers(current_step)
        if total_layers > 0 and unfrozen_layers > 0:
            start_idx = max(0, total_layers - unfrozen_layers)
            for layer in self.model.lm.layers[start_idx:]:
                layer.requires_grad_(True)
            if hasattr(self.model.lm, 'norm'):
                self.model.lm.norm.requires_grad_(True)

        if hasattr(self.model, 'lm') and hasattr(self.model.lm, 'embed_tokens'):
            self.model.lm.embed_tokens.requires_grad_(False)

        if force_log or unfrozen_layers != self._current_unfrozen_layers:
            total = sum(p.numel() for p in self.model.parameters())
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            ratio = 100.0 * trainable / total if total > 0 else 0.0
            print(
                "[progressive_unfreeze] "
                f"step={current_step} unfrozen_layers={unfrozen_layers}/{total_layers} "
                f"trainable={trainable}/{total} ({ratio:.4f}%)"
            )
            self._current_unfrozen_layers = unfrozen_layers

    def _apply_trainability_policy(self, current_step: int = 0, force_log: bool = False) -> None:
        if self.moa_enabled:
            self._apply_peft_freeze()
            return
        if self.progressive_unfreeze_enabled:
            self._apply_progressive_unfreeze(current_step=current_step, force_log=force_log)
            return
        self._freeze_auxiliary_modules()

    def _resolve_masking_cfg(self):
        mask_cfg = self._cfg_node_to_dict(self.cfg.train.get('masking', {}))
        curriculum_cfg = self._cfg_node_to_dict(mask_cfg.pop('curriculum', {}))
        if not bool(curriculum_cfg.get('enabled', False)):
            return mask_cfg

        current_step = int(self.global_step)
        milestones = curriculum_cfg.get('milestones', []) or []
        milestones = sorted(
            (self._cfg_node_to_dict(item) for item in milestones),
            key=lambda item: int(item.get('step', 0))
        )
        for item_dict in milestones:
            if current_step < int(item_dict.get('step', 0)):
                break
            for key, value in item_dict.items():
                if key != 'step':
                    mask_cfg[key] = value
        return mask_cfg

    @staticmethod
    def _is_autoregressive_masking(mask_cfg) -> bool:
        strategy = str(mask_cfg.get('strategy', 'bernoulli')).lower()
        return strategy in {'autoregressive', 'autoregressive_lm', 'ar', 'causal'}

    def _build_autoregressive_inputs(self, acc_ids: torch.Tensor, acc_pad_mask: torch.Tensor) -> torch.Tensor:
        acc_input = torch.full_like(acc_ids, self.model.special_tokens['acc_pad'])
        if acc_ids.size(1) == 0:
            return acc_input

        acc_input[:, 0] = self.model.special_tokens['acc_mask']
        if acc_ids.size(1) > 1:
            acc_input[:, 1:] = acc_ids[:, :-1]

        acc_input = torch.where(
            acc_pad_mask.bool(),
            acc_input,
            torch.full_like(acc_input, self.model.special_tokens['acc_pad']),
        )
        return acc_input

    def _mark_invalid_audio_samples(self, batch) -> None:
        if not isinstance(batch, dict):
            return

        valid_mask = None
        invalid_by_key = {}
        for key in ("voc_audio", "acc_audio", "ref_audio"):
            value = batch.get(key)
            if not isinstance(value, torch.Tensor) or value.ndim < 2:
                continue

            flat_value = value.reshape(value.shape[0], -1)
            key_valid_mask = torch.isfinite(flat_value).all(dim=1)
            if not key_valid_mask.all():
                invalid_by_key[key] = (~key_valid_mask).nonzero(as_tuple=False).flatten().cpu().tolist()
            valid_mask = key_valid_mask if valid_mask is None else (valid_mask & key_valid_mask)

        if valid_mask is None or valid_mask.all():
            return

        existing_valid = batch.get('valid')
        if isinstance(existing_valid, torch.Tensor):
            batch['valid'] = existing_valid.to(device=valid_mask.device, dtype=torch.bool) & valid_mask
        else:
            batch['valid'] = valid_mask

        invalid_indices = (~batch['valid']).nonzero(as_tuple=False).flatten().cpu().tolist()
        songids = batch.get('songid')
        if isinstance(songids, list):
            invalid_songids = [songids[idx] for idx in invalid_indices if idx < len(songids)]
        else:
            invalid_songids = invalid_indices

        reasons = ", ".join(f"{key}={indices}" for key, indices in invalid_by_key.items())
        print(
            "[llada] dropping invalid audio samples before MuCodec "
            f"indices={invalid_indices} songids={invalid_songids} reasons={reasons}",
            flush=True,
        )

    def _sample_rtd_tokens(self, logits: torch.Tensor) -> torch.Tensor:
        if self.rtd_sampling == 'argmax':
            return logits.argmax(dim=-1)
        if self.rtd_sampling in {'sample', 'multinomial'}:
            temperature = max(self.rtd_temperature, 1e-5)
            probs = torch.softmax(logits.float() / temperature, dim=-1)
            flat_probs = probs.reshape(-1, probs.size(-1))
            sampled = torch.multinomial(flat_probs, num_samples=1)
            return sampled.view(*logits.shape[:-1])
        raise ValueError(f"Unsupported loss.rtd.sampling: {self.rtd_sampling}")

    def _force_codec_fp32(self) -> None:
        """
        MuCodec contains BatchNorm/Conv stacks and feature extraction that may produce fp32 tensors even when the input
        waveform is cast. Under `bf16-true`, Lightning will cast module weights to bf16, which can break BN with fp32
        inputs. Keep MuCodec in fp32 for stability.
        """
        if hasattr(self, "codec_model") and isinstance(self.codec_model, nn.Module):
            self.codec_model.to(dtype=torch.float32)
        if hasattr(self, "codec_vae") and isinstance(self.codec_vae, nn.Module):
            self.codec_vae.to(dtype=torch.float32)
        if hasattr(self, "codec_stft") and isinstance(self.codec_stft, nn.Module):
            self.codec_stft.to(dtype=torch.float32)

    def on_fit_start(self) -> None:
        # Lightning prints model summary around fit start; make sure trainable flags are correct.
        self._apply_trainability_policy(current_step=int(self.global_step), force_log=True)
        self._force_codec_fp32()
        if getattr(self.trainer, "is_global_zero", True):
            world_size = getattr(self.trainer, "world_size", 1)
            accumulate = getattr(self.trainer, "accumulate_grad_batches", self.cfg.train.acc_grad)
            per_device_batch = self.cfg.data.batch_size
            print(
                "[llada] runtime config "
                f"world_size={world_size} "
                f"per_device_batch={per_device_batch} "
                f"accumulate_grad_batches={accumulate} "
                f"effective_global_batch={per_device_batch * world_size * accumulate}"
            )

    def on_validation_start(self) -> None:
        self._force_codec_fp32()

    def on_train_batch_start(self, batch, batch_idx) -> None:
        if self.progressive_unfreeze_enabled:
            self._apply_trainability_policy(current_step=int(self.global_step))

    def load_model(self, model_fp):
        ckpt = torch.load(model_fp, map_location='cpu', weights_only=False)
        state_dict = {k[6:]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
        optional_missing_prefixes = []
        optional_unexpected_prefixes = []
        if getattr(self.model, 'rtd_enabled', False):
            optional_missing_prefixes.append('rtd_head.')
        else:
            optional_unexpected_prefixes.append('rtd_head.')

        def _is_optional_missing(key: str) -> bool:
            return any(key.startswith(prefix) for prefix in optional_missing_prefixes)

        def _is_optional_unexpected(key: str) -> bool:
            return any(key.startswith(prefix) for prefix in optional_unexpected_prefixes)

        if self.moa_enabled and len(self._moa_wrapped_module_names) > 0:
            remapped = {}
            for key, value in state_dict.items():
                new_key = key
                for module_name in self._moa_wrapped_module_names:
                    prefix = module_name + "."
                    if key.startswith(prefix) and ".base_layer." not in key:
                        suffix = key[len(prefix):]
                        if suffix in ("weight", "bias"):
                            new_key = prefix + "base_layer." + suffix
                            break
                remapped[new_key] = value
            missing_keys, unexpected_keys = self.model.load_state_dict(remapped, strict=False)
            print(f"[MoA] load ckpt with strict=False. missing={len(missing_keys)}, unexpected={len(unexpected_keys)}")
        else:
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            filtered_missing_keys = [key for key in missing_keys if not _is_optional_missing(key)]
            filtered_unexpected_keys = [key for key in unexpected_keys if not _is_optional_unexpected(key)]
            if filtered_missing_keys or filtered_unexpected_keys:
                raise RuntimeError(
                    "Error(s) in loading state_dict for SemanticMLMClampModel: "
                    f"missing={filtered_missing_keys}, unexpected={filtered_unexpected_keys}"
                )
            if missing_keys or unexpected_keys:
                print(
                    "[load_model] load ckpt with strict=False. "
                    f"missing={missing_keys}, unexpected={unexpected_keys}"
                )
        del ckpt
        # Some training setups may re-enable grads after loading; enforce again.
        self._apply_trainability_policy(current_step=int(self.global_step), force_log=True)

    def _apply_peft_freeze(self):
        """Force-freeze backbone and only unfreeze PEFT params (MoA + optional extras)."""
        if not self.moa_enabled:
            return

        # Always freeze non-trainable big modules. Note: eval() does NOT freeze params.
        self._freeze_auxiliary_modules()

        # Freeze everything inside llada model.
        self.model.requires_grad_(False)

        # Unfreeze only MoA params (router + low-rank experts). Base weights stay frozen.
        for m in self.model.lm.modules():
            if isinstance(m, MoALinear):
                m.router.weight.requires_grad = True
                m.lora_A.requires_grad = True
                m.lora_B.requires_grad = True

        # Optional trainable modules in self.model, e.g. ["to_logits", "voc_embed", "acc_embed", "clamp_proj"].
        if len(self._moa_trainable_extra) > 0:
            for name, param in self.model.named_parameters():
                if any(key in name for key in self._moa_trainable_extra):
                    param.requires_grad = True

    def _enable_moa_peft(self):
        moa_cfg = self.cfg.get('moa', {})
        target_keywords = list(moa_cfg.get(
            'target_modules',
            ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
        ))
        rank = int(moa_cfg.get('rank', 8))
        num_experts = int(moa_cfg.get('num_experts', 4))
        alpha = float(moa_cfg.get('alpha', 16.0))
        router_temperature = float(moa_cfg.get('router_temperature', 1.0))
        dropout = float(moa_cfg.get('dropout', 0.05))
        trainable_extra = set(moa_cfg.get('trainable_extra', []))
        self._moa_trainable_extra = trainable_extra
        last_n_layers = int(moa_cfg.get('last_n_layers', 0))
        layers_allowlist = moa_cfg.get('layers', None)
        if layers_allowlist is not None:
            layers_allowlist = set(int(x) for x in layers_allowlist)
        elif last_n_layers > 0:
            try:
                total_layers = int(self.cfg.model.llama.num_hidden_layers)
            except Exception:
                total_layers = None
            if total_layers is not None and last_n_layers > 0:
                start = max(0, total_layers - last_n_layers)
                layers_allowlist = set(range(start, total_layers))

        replaced = 0
        for module_name, module in list(self.model.lm.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if not any(key in module_name for key in target_keywords):
                continue
            if layers_allowlist is not None:
                m = re.search(r"layers\\.(\\d+)\\.", module_name)
                if m is not None:
                    layer_idx = int(m.group(1))
                    if layer_idx not in layers_allowlist:
                        continue

            parent, child_name = _get_parent_module(self.model.lm, module_name)
            setattr(
                parent,
                child_name,
                MoALinear(
                    base_layer=module,
                    rank=rank,
                    num_experts=num_experts,
                    alpha=alpha,
                    router_temperature=router_temperature,
                    dropout=dropout,
                ),
            )
            self._moa_wrapped_module_names.add(f"lm.{module_name}")
            replaced += 1

        if replaced == 0:
            raise RuntimeError(
                f"[MoA] no target linear layers were replaced. target_modules={target_keywords}"
            )

        # Apply freeze/unfreeze policy now.
        self._apply_peft_freeze()

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        ratio = 100.0 * trainable / total if total > 0 else 0.0
        print(f"[MoA] replaced_linear={replaced}, trainable={trainable}/{total} ({ratio:.4f}%)")

    @torch.no_grad()
    def generate_with_clamp_and_codec(self,
                                            voc_audio: torch.Tensor,
                                            text: Union[Optional[List[str]], Optional[str]] = None,
                                            ref_audio: torch.Tensor=None,
                                            sampling_steps: int=36,
                                            top_p: float=None,
                                            top_k: float=None,
                                            temperature: float=1.0,
                                            mask_temperature: float=10.5,
                                            remask_schedule: str='cosine',
                                            remask_ratios: Optional[List[float]]=None,
                                            remask_strategy: str='low_confidence',
                                            remask_seed: int=0,
                                            remask_sequence_id: Optional[str]=None,
                                            remask_trace: Optional[list]=None,
                                            decode_mode: str='diffusion'):


        # MuCodec encoding
        # Ensure codec has correct device
        self.codec.device = self.device

        # sound2code expects [2, T] or [1, 2, T]
        audio_in = voc_audio.squeeze(0)
        codec_dtype = None
        if hasattr(self, "codec_model") and isinstance(self.codec_model, nn.Module):
            try:
                codec_dtype = next(self.codec_model.parameters()).dtype
            except StopIteration:
                codec_dtype = None
        if codec_dtype is not None and audio_in.dtype != codec_dtype:
            audio_in = audio_in.to(dtype=codec_dtype)
        voc_ids_full = self.codec.sound2code(audio_in) # Returns [1, 1, Total]
        voc_ids = voc_ids_full[0, 0, :] # [Total]
        voc_ids = voc_ids.unsqueeze(0)  # [1, Total]

        text_clamp = None
        ref_audio_clamp = None

        if text is not None:
            if not isinstance(text, list):
                text = [text]
            text_clamp = self.clamp3(texts=text)
        if ref_audio is not None:
            ref_audio_clamp = self.clamp3(audio=ref_audio)

        # Keep prefix layout consistent with training: missing modality becomes a zero prefix slot.
        if self.cfg.get('clamp_cond') == 'prefix':
            if text_clamp is not None and ref_audio_clamp is None:
                ref_audio_clamp = torch.zeros_like(text_clamp[:, :1, :])
            elif ref_audio_clamp is not None and text_clamp is None:
                text_clamp = torch.zeros_like(ref_audio_clamp[:, :1, :])

        # Combine embeddings if both are provided
        acc_clamp_embeds = None
        if text_clamp is None:
            acc_clamp_embeds = ref_audio_clamp
        elif ref_audio_clamp is None:
            acc_clamp_embeds = text_clamp
        else:
            # For clamp3, both text and audio embeddings are [B, 1, 768], so we can concatenate them
            acc_clamp_embeds = torch.cat([ref_audio_clamp, text_clamp], dim=1)


        codes = self.model.generate(
                            voc_ids=voc_ids,
                            sampling_steps=sampling_steps,
                            acc_clamp_embeds=acc_clamp_embeds,
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
                            decode_mode=decode_mode)

        codes = codes.reshape(1, 1, -1)
        print("✅The Mask LM has generated codes:", codes.shape)

        # MuCodec decoding
        # codes: [1, 1, T]
        audio_48k = self.codec.code2sound(codes=codes)
        # Convert back to 24k mono
        # audio = convert_audio(audio_48k, from_rate=48000, to_rate=24000, to_channels=1)

        return audio_48k

    def _process_batch(self, batch):
        self._mark_invalid_audio_samples(batch)
        if 'valid' in batch:
            valid_mask = batch['valid']
            if not valid_mask.any():
                device = batch['voc_audio'].device if 'voc_audio' in batch else self.device
                dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)
                dummy_acc = torch.tensor(0.0, device=device)
                return {
                    'loss': dummy_loss,
                    'mlm_loss': dummy_loss.detach(),
                    'top1_accuracy': dummy_acc,
                    'top10_accuracy': dummy_acc,
                    'rtd_loss': dummy_acc,
                    'rtd_acc': dummy_acc,
                }

            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and v.shape[0] == valid_mask.shape[0]:
                    batch[k] = v[valid_mask]

                elif isinstance(v, list) and len(v) == valid_mask.shape[0]:
                    batch[k] = [item for i, item in enumerate(v) if valid_mask[i]]

        if 'voc_ids' not in batch:
            voc_audio = batch['voc_audio']
            acc_audio = batch['acc_audio']
            ref_audio = batch['ref_audio']
            codec_encode_batch_size = self.cfg.get('codec_encode_batch_size', None)
            if codec_encode_batch_size is not None:
                codec_encode_batch_size = int(codec_encode_batch_size)
            with torch.no_grad():
                codec_dtype = None
                if hasattr(self, "codec_model") and isinstance(self.codec_model, nn.Module):
                    try:
                        codec_dtype = next(self.codec_model.parameters()).dtype
                    except StopIteration:
                        codec_dtype = None
                self.codec.device = self.device
                if codec_dtype is not None:
                    if voc_audio.dtype != codec_dtype:
                        voc_audio = voc_audio.to(dtype=codec_dtype)
                    if acc_audio.dtype != codec_dtype:
                        acc_audio = acc_audio.to(dtype=codec_dtype)
                voc_ids = self.codec.sound2code_batch(
                    voc_audio,
                    batch_size=codec_encode_batch_size,
                ).squeeze(1)
                acc_ids = self.codec.sound2code_batch(
                    acc_audio,
                    batch_size=codec_encode_batch_size,
                ).squeeze(1)

            # voc_ids, _ = self.codec.encode(voc_audio)
            # acc_ids, _ = self.codec.encode(acc_audio)
            # voc_ids = voc_ids.squeeze(1)
            # acc_ids = acc_ids.squeeze(1)

            with torch.no_grad():
                text_clamp = self.clamp3(texts=batch['text'])  # [B, 1, 768]
                audio_clamp = self.clamp3(audio=ref_audio)     # [B, 1, 768]

            if text_clamp is not None:
                text_clamp = self.clamp_drop(text_clamp)
            if audio_clamp is not None:
                audio_clamp = self.clamp_drop(audio_clamp)

            prefix_list = []
            if self.cfg.get('clamp_cond') == 'prefix':
                if text_clamp is not None:
                    prefix_list.append(text_clamp)  # [B, 1, 768]
                if audio_clamp is not None:
                    prefix_list.append(audio_clamp)  # [B, 1, 768]

            # [B, P, 768] 或 None
            acc_clamp_embeds = None
            if len(prefix_list) > 0:
                # Clamp embeddings are already [B, 1, 768], so we can stack them directly
                acc_clamp_embeds = torch.cat(prefix_list, dim=1)  # [B, P, 768]
        else:
            voc_ids = batch['voc_ids']
            acc_ids = batch['acc_ids']
            acc_clamp_embeds = batch.get('acc_clamp_embeds', None)

        # 仅基于 acc_ids 计算 pad 掩码（长度 = T），这里不要 pad 前缀
        acc_pad_mask = (acc_ids != self.model.special_tokens['acc_pad']).to(acc_ids.device)  # [B, T]

        mask_cfg = self._resolve_masking_cfg()
        use_autoregressive_masking = self._is_autoregressive_masking(mask_cfg)
        if use_autoregressive_masking:
            acc_random_mask = None
            acc_ids_masked = self._build_autoregressive_inputs(acc_ids, acc_pad_mask)
        else:
            # Some PyTorch builds don't implement Sobol drawing for bf16; always draw in fp32.
            r = self.rng.draw(acc_ids.shape[0], dtype=torch.float32)[:, 0].to(acc_ids.device)
            acc_random_mask = sample_mask(
                acc_ids,
                r,
                strategy=mask_cfg.get('strategy', 'bernoulli'),
                valid_mask=acc_pad_mask,
                min_mask_ratio=mask_cfg.get('min_mask_ratio', None),
                max_mask_ratio=mask_cfg.get('max_mask_ratio', None),
                span_prob=mask_cfg.get('span_prob', 0.5),
                min_spans=mask_cfg.get('min_spans', 1),
                max_spans=mask_cfg.get('max_spans', 3),
                preserve_first_token=mask_cfg.get('preserve_first_token', False),
            )
            acc_ids_masked = apply_mask(acc_ids, acc_random_mask, self.model.special_tokens['acc_mask'])

        # 前缀个数（用于 labels/metrics 对齐用）
        prefix_len = 0
        if self.cfg.get('clamp_cond') == 'prefix' and acc_clamp_embeds is not None:
            if acc_clamp_embeds.dim() == 3:
                prefix_len = acc_clamp_embeds.size(1)  # P
            else:
                prefix_len = 1
        # print(f'prefix_len: {prefix_len}')
        # 目标 labels：把没被 mask 的位置填 pad；prefix_len>0 时在左侧 pad P 位
        if use_autoregressive_masking:
            labels = acc_ids.clone()
        else:
            labels = acc_ids.clone().masked_fill(~acc_random_mask.bool(), self.model.special_tokens['acc_pad'])
        if prefix_len > 0:
            labels = F.pad(labels, (prefix_len, 0), value=self.model.special_tokens['acc_pad'])  # [B, P+T]
        #     acc_pad_mask = F.pad(acc_pad_mask, (prefix_len, 0), value=True)  # [B, P+T]

        # 前向（把未 pad 的 mask 交给 forward，由 forward 来 pad 到 P+T）
        logits = self.model(
            voc_ids, acc_ids_masked,
            attention_mask=acc_pad_mask,              # [B, T] 这里不 pad
            acc_clamp_embeds=acc_clamp_embeds,        # [B,P,768] 或 [B,1,768]
            causal_attention=use_autoregressive_masking,
        )

        # metrics：跳过前缀位
        if prefix_len > 0:
            logits_for_metric = logits[:, prefix_len:]
            labels_for_metric = labels[:, prefix_len:]
        else:
            logits_for_metric = logits
            labels_for_metric = labels

        mlm_loss = self.loss_func(
            logits.transpose(-1, -2), labels,
            ignore_index=self.model.special_tokens['acc_pad'],
            reduction='mean',
            label_smoothing=self.cfg.loss.label_smoothing
        )
        top10_accuracy = self.top10_accuracy(logits_for_metric.transpose(-1, -2).detach(), labels_for_metric)
        top1_accuracy  = self.top1_accuracy(logits_for_metric.transpose(-1, -2).detach(), labels_for_metric)
        total_loss = mlm_loss
        rtd_loss = torch.tensor(0.0, device=mlm_loss.device)
        rtd_acc = torch.tensor(0.0, device=mlm_loss.device)

        if self.rtd_enabled and not use_autoregressive_masking:
            logits_for_sampling = logits_for_metric.detach()
            with torch.no_grad():
                sampled_tokens = self._sample_rtd_tokens(logits_for_sampling)
                replaced_acc_ids = torch.where(acc_random_mask.bool(), sampled_tokens, acc_ids)
                rtd_labels = ((replaced_acc_ids != acc_ids) & acc_pad_mask.bool()).float()

            if self.rtd_backbone_grad:
                _, corrupted_hidden_states = self.model(
                    voc_ids,
                    replaced_acc_ids,
                    attention_mask=acc_pad_mask,
                    acc_clamp_embeds=acc_clamp_embeds,
                    return_hidden_states=True,
                )
            else:
                with torch.no_grad():
                    _, corrupted_hidden_states = self.model(
                        voc_ids,
                        replaced_acc_ids,
                        attention_mask=acc_pad_mask,
                        acc_clamp_embeds=acc_clamp_embeds,
                        return_hidden_states=True,
                    )
                corrupted_hidden_states = corrupted_hidden_states.detach()
            if prefix_len > 0:
                corrupted_hidden_states = corrupted_hidden_states[:, prefix_len:]
            rtd_logits = self.model.compute_rtd_logits(corrupted_hidden_states)

            valid_rtd_mask = acc_pad_mask.bool()
            if valid_rtd_mask.any():
                rtd_loss = F.binary_cross_entropy_with_logits(
                    rtd_logits[valid_rtd_mask],
                    rtd_labels[valid_rtd_mask],
                    reduction='mean',
                )
                rtd_predictions = (rtd_logits[valid_rtd_mask] > 0).float()
                rtd_acc = (rtd_predictions == rtd_labels[valid_rtd_mask]).float().mean()
                total_loss = total_loss + self.rtd_weight * rtd_loss
        elif self.rtd_enabled and use_autoregressive_masking and not self._ar_rtd_warning_printed:
            print("[RTD] skipping RTD loss because train.masking.strategy=autoregressive")
            self._ar_rtd_warning_printed = True

        return {
            'loss': total_loss,
            'mlm_loss': mlm_loss.detach(),
            'top1_accuracy': top1_accuracy,
            'top10_accuracy': top10_accuracy,
            'rtd_loss': rtd_loss.detach(),
            'rtd_acc': rtd_acc.detach(),
        }


    def _process_prefix_embeddings(self, acc_pad_mask, acc_clamp_embeds):
        """根据前缀个数动态 pad 注意力掩码"""
        if self.cfg.get('clamp_cond') == 'prefix' and acc_clamp_embeds is not None:
            # acc_clamp_embeds 可能是 [B, 1, 768] 或 [B, P, 768]
            if acc_clamp_embeds.dim() == 3:
                prefix_len = acc_clamp_embeds.size(1)  # P
            else:
                prefix_len = 1
            acc_pad_mask = F.pad(acc_pad_mask, (prefix_len, 0), value=True)
        return acc_pad_mask

    def training_step(self, batch, batch_idx):
        scheduler = self.lr_schedulers()
        log_runtime = batch_idx == 0 and not self._runtime_debug_printed
        batch_size = None
        if isinstance(batch, dict):
            for key in ("voc_audio", "acc_audio", "ref_audio", "voc_ids", "acc_ids"):
                value = batch.get(key)
                if isinstance(value, torch.Tensor):
                    batch_size = value.shape[0]
                    break

        mem_before = None
        reserved_before = None
        peak_alloc = None
        peak_reserved = None
        if log_runtime and torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            mem_before = torch.cuda.memory_allocated(self.device) / 1024 ** 3
            reserved_before = torch.cuda.memory_reserved(self.device) / 1024 ** 3

        metrics = self._process_batch(batch)
        loss = metrics['loss']

        if log_runtime:
            self._runtime_debug_printed = True
            if torch.cuda.is_available() and self.device.type == "cuda":
                peak_alloc = torch.cuda.max_memory_allocated(self.device) / 1024 ** 3
                peak_reserved = torch.cuda.max_memory_reserved(self.device) / 1024 ** 3
            if getattr(self.trainer, "is_global_zero", True):
                msg = (
                    "[llada] first train step "
                    f"actual_batch={batch_size} "
                    f"cfg_batch={self.cfg.data.batch_size} "
                    f"accumulate_grad_batches={getattr(self.trainer, 'accumulate_grad_batches', self.cfg.train.acc_grad)}"
                )
                if mem_before is not None:
                    msg += (
                        f" cuda_alloc_before_gb={mem_before:.2f} "
                        f"cuda_reserved_before_gb={reserved_before:.2f} "
                        f"cuda_peak_alloc_gb={peak_alloc:.2f} "
                        f"cuda_peak_reserved_gb={peak_reserved:.2f}"
                    )
                print(msg)

        self.log("train/total_loss", loss,
                 on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/mlm_loss", metrics['mlm_loss'],
                 on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/lr", scheduler.get_last_lr()[0],
                 on_step=True, prog_bar=True, sync_dist=True)
        self.log("train/top_1_acc", metrics['top1_accuracy'],
                 on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/top_10_acc", metrics['top10_accuracy'],
                 on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.rtd_enabled:
            self.log("train/rtd_loss", metrics['rtd_loss'],
                     on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("train/rtd_acc", metrics['rtd_acc'],
                     on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        if self.progressive_unfreeze_enabled:
            self.log(
                "train/unfrozen_backbone_layers",
                float(self._resolve_unfrozen_layers(int(self.global_step))),
                on_step=True,
                prog_bar=False,
                sync_dist=True,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        self.current_epoch
        metrics = self._process_batch(batch)

        self.log("val/total_loss", metrics['loss'],
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/mlm_loss", metrics['mlm_loss'],
                 on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val/top_1_acc", metrics['top1_accuracy'],
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/top_10_acc", metrics['top10_accuracy'],
                 on_epoch=True, prog_bar=True, sync_dist=True)
        if self.rtd_enabled:
            self.log("val/rtd_loss", metrics['rtd_loss'],
                     on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("val/rtd_acc", metrics['rtd_acc'],
                     on_epoch=True, prog_bar=False, sync_dist=True)

    def configure_optimizers(self):
        optim_cfg = self.cfg.optim
        # Ensure trainable flags are correct before building optimizer param groups.
        self._apply_trainability_policy(current_step=0, force_log=True)

        # 分组参数，为投影层设置不同的学习率
        params = []
        # if hasattr(self.model, 'clamp_proj'):
        #     # 为投影层设置不同的学习率（可选）
        #     params.append({
        #         'params': self.model.clamp_proj.parameters(),
        #         'lr': self.lr * 0.5  # 可调整的学习率乘数
        #     })

        # 添加其他模型参数
        if self.progressive_unfreeze_enabled:
            candidate_params = [
                p for n, p in self.model.named_parameters()
                if not n.startswith('lm.embed_tokens')
            ]
        else:
            candidate_params = [p for _, p in self.model.named_parameters() if p.requires_grad]

        params.append({'params': candidate_params})
        if len(params[0]['params']) == 0:
            raise RuntimeError("No trainable parameters found in self.model. Check MoA config/targets.")

        if optim_cfg.name == 'adamw':
            optimizer = torch.optim.AdamW(params,
                                        lr=float(optim_cfg.lr),
                                        betas=[float(b) for b in optim_cfg.betas],
                                        eps=float(optim_cfg.eps),
                                        weight_decay=float(optim_cfg.weight_decay))

        total_steps = int(self.trainer.estimated_stepping_batches)
        scheduler_name = optim_cfg.scheduler.name

        if scheduler_name == 'cos':
            scheduler = get_scheduler(name="cosine",
                                      optimizer=optimizer,
                                      num_warmup_steps=optim_cfg.scheduler.warmup,
                                      num_training_steps=total_steps)
        elif scheduler_name == 'cos_with_hold':
            warmup_steps = int(optim_cfg.scheduler.get('warmup', 0))
            hold_steps = int(optim_cfg.scheduler.get('hold_steps', 0))

            def lr_lambda(current_step: int):
                if warmup_steps > 0 and current_step < warmup_steps:
                    return max(float(current_step) / float(max(1, warmup_steps)), 1e-8)
                if current_step < warmup_steps + hold_steps:
                    return 1.0

                decay_steps = max(1, total_steps - warmup_steps - hold_steps)
                progress = min(
                    1.0,
                    max(0.0, float(current_step - warmup_steps - hold_steps) / float(decay_steps))
                )
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

        lr_scheduler_config = {
            'scheduler': scheduler,
            'interval': 'step'
        }
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}


    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """
        恢复 checkpoint 时，默认保留其中的 optimizer/scheduler/global_step。
        如需救急降学习率，可通过 `optim.resume_lr` 显式覆写。
        """
        resumed_step = int(checkpoint.get("global_step", 0))
        self._apply_trainability_policy(current_step=resumed_step, force_log=True)

        resume_lr = self.cfg.optim.get("resume_lr", None)
        if resume_lr is None:
            print(f"[checkpoint] restore full training state from global_step={resumed_step}")
            return

        new_lr = float(resume_lr)

        # 1. 修改优化器状态中的学习率
        if "optimizer_states" in checkpoint:
            for opt_state in checkpoint["optimizer_states"]:
                for param_group in opt_state["param_groups"]:
                    param_group["lr"] = new_lr
                    print(f"已手动将 Checkpoint 中的优化器学习率修改为: {new_lr}")

        # 2. 如果你有 LR Scheduler，也建议一并处理
        if "lr_schedulers" in checkpoint:
            for scheduler in checkpoint["lr_schedulers"]:
                # 注意：不同的 scheduler 内部属性名不同，通常修改 base_lrs 或 _last_lr
                if "base_lrs" in scheduler:
                    scheduler["base_lrs"] = [new_lr] * len(scheduler["base_lrs"])
                if "_last_lr" in scheduler:
                    scheduler["_last_lr"] = [new_lr] * len(scheduler["_last_lr"])
                print(f"已手动重置 Scheduler 学习率基准")
