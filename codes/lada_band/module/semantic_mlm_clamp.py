import torch
import torch.nn as nn
import torch.nn.functional as F
import omegaconf
from omegaconf import OmegaConf
import lightning as L
from transformers import get_scheduler
from torchmetrics.classification import MulticlassAccuracy
from ..utils.mask import sample_mask, apply_mask
from ..model.clamp3 import Clamp3
import os
import sys
from beartype.typing import List, Optional, Tuple, Union
from ..model.semantic_mlm_clamp import SemanticMLMClampModel
from ..utils.condition import ClassifierFreeGuidanceDropout
from hydra.utils import to_absolute_path
from ..utils.audio_utils import _av_read, _av_info, AudioFileInfo, convert_audio

# Ensure MuCodec is in path
mucodec_dir = "./MuCodec"
sys.path.append(mucodec_dir)

try:
    from generate import MuCodec
except ImportError:
    print(f"Warning: Could not import MuCodec from {mucodec_dir}")

class ModelModule(L.LightningModule):
    def __init__(self, cfg: omegaconf.DictConfig, mode='train'):
        super().__init__()
        # self.save_hyperparameters(ignore=['model'])

        # 1. 初始化主模型 (先不急着转 float16，最后统一转)
        self.model = SemanticMLMClampModel(cfg)

        self.cfg = cfg
        self.lr = cfg.optim.lr

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

        # 2. 初始化 Clamp3 (它内部可能有自己的加载逻辑)
        self.clamp3 = Clamp3(precision=cfg.train.precision, ckpt_path=cfg.clamp3_path)
        self.clamp3.eval()

        self.clamp_drop = ClassifierFreeGuidanceDropout(p=cfg.clamp_dropout)

        # 3. 初始化 Codec 并加载权重
        if 'codec_path' in cfg:
            # New MuCodec initialization
            self.codec = MuCodec(
                model_path=cfg.codec_path,
                layer_num=cfg.get('codec_layer_num', 7), # Default to 7 if not specified
                load_main_model=True,
                device='cpu' # Initial device, will be moved by Lightning or manually
            )

            # Register codec submodules for Lightning to track/move them
            # MuCodec wrapper has .model (PromptCondAudioDiffusion), .vae, .stft
            self.codec_model = self.codec.model
            self.codec_vae = self.codec.vae
            self.codec_stft = self.codec.stft

            self.codec.model.eval()
            self.codec.vae.eval()
            # stft usually has no parameters but good to be safe
            if hasattr(self.codec.stft, 'eval'):
                self.codec.stft.eval()



        # 再次确保处于 eval 模式的模块不被重置
        self.clamp3.eval()
        if hasattr(self, 'codec'):
            self.codec.model.eval()

        self.to(torch.float32)

    def load_model(self, model_fp):
        ckpt = torch.load(model_fp, map_location='cpu', weights_only=False)
        state_dict = {k[6:]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
        self.model.load_state_dict(state_dict)
        del ckpt

    @torch.no_grad()
    def generate_with_clamp_and_codec(self,
                                            voc_audio: torch.Tensor,
                                            text: Union[Optional[List[str]], Optional[str]] = None,
                                            ref_audio: torch.Tensor=None,
                                            sampling_steps: int=36,
                                            top_p: float=None,
                                            top_k: float=None,
                                            temperature: float=1.0,
                                            mask_temperature: float=10.5):
        assert (len(voc_audio.shape) == 2) and (voc_audio.shape[0] == 1)

        if ref_audio is not None:
            ref_audio = convert_audio(ref_audio, from_rate=24000, to_rate=24000, to_channels=1)

        # MuCodec encoding
        # Ensure codec has correct device
        self.codec.device = self.device

        # voc_audio: [1, T] (24k mono) -> [1, 2, T_48k] (48k stereo)
        voc_audio_48k = convert_audio(voc_audio.unsqueeze(1), from_rate=24000, to_rate=48000, to_channels=2)
        # sound2code expects [2, T] or [1, 2, T]
        voc_ids_full = self.codec.sound2code(voc_audio_48k.squeeze(0)) # Returns [1, 1, Total]
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
                            mask_temperature=mask_temperature)

        codes = codes.reshape(1, 1, -1)
        print("✅The Mask LM has generated codes:", codes.shape)

        # MuCodec decoding
        # codes: [1, 1, T]
        audio_48k = self.codec.code2sound(codes=codes)
        # Convert back to 24k mono
        audio = convert_audio(audio_48k, from_rate=48000, to_rate=24000, to_channels=1)

        return audio

    def _process_batch(self, batch):
        if 'valid' in batch:
            valid_mask = batch['valid']
            if not valid_mask.any():
                device = batch['voc_audio'].device if 'voc_audio' in batch else self.device
                dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)
                dummy_acc = torch.tensor(0.0, device=device)
                return dummy_loss, dummy_acc, dummy_acc

            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and v.shape[0] == valid_mask.shape[0]:
                    batch[k] = v[valid_mask]

                elif isinstance(v, list) and len(v) == valid_mask.shape[0]:
                    batch[k] = [item for i, item in enumerate(v) if valid_mask[i]]

        if 'voc_ids' not in batch:
            voc_audio = batch['voc_audio']
            acc_audio = batch['acc_audio']

            # Encode voc_audio using MuCodec
            voc_ids_list = []
            for i in range(voc_audio.shape[0]):
                # 24k Mono -> 48k Stereo
                va = convert_audio(voc_audio[i].unsqueeze(0).unsqueeze(0), from_rate=24000, to_rate=48000, to_channels=2)
                self.codec.device = self.device
                # sound2code expects [2, T]
                codes = self.codec.sound2code(va.squeeze(0)) # [1, 1, T]
                voc_ids_list.append(codes[0, 0, :])
            voc_ids = torch.stack(voc_ids_list) # [B, T]

            # Encode acc_audio using MuCodec
            acc_ids_list = []
            for i in range(acc_audio.shape[0]):
                aa = convert_audio(acc_audio[i].unsqueeze(0).unsqueeze(0), from_rate=24000, to_rate=48000, to_channels=2)
                self.codec.device = self.device
                codes = self.codec.sound2code(aa.squeeze(0)) # [1, 1, T]
                acc_ids_list.append(codes[0, 0, :])
            acc_ids = torch.stack(acc_ids_list) # [B, T]

            # voc_ids, _ = self.codec.encode(voc_audio)
            # acc_ids, _ = self.codec.encode(acc_audio)
            # voc_ids = voc_ids.squeeze(1)
            # acc_ids = acc_ids.squeeze(1)

            text_clamp = self.clamp3(texts=batch['text'])  # [B, 1, 768]
            audio_clamp = self.clamp3(audio=acc_audio)     # [B, 1, 768]

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

        # 随机 mask：只对 T 个 acc token 做，不要切 [:, prefix_len:]
        # Some PyTorch builds don't implement Sobol drawing for bf16; always draw in fp32.
        r = self.rng.draw(acc_ids.shape[0], dtype=torch.float32)[:, 0].to(acc_ids.device)
        mask_cfg = self.cfg.train.get('masking', {})
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
        )                                        # [B, T]
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
        labels = acc_ids.clone().masked_fill(~acc_random_mask.bool(), self.model.special_tokens['acc_pad'])
        if prefix_len > 0:
            labels = F.pad(labels, (prefix_len, 0), value=self.model.special_tokens['acc_pad'])  # [B, P+T]
        #     acc_pad_mask = F.pad(acc_pad_mask, (prefix_len, 0), value=True)  # [B, P+T]

        # 前向（把未 pad 的 mask 交给 forward，由 forward 来 pad 到 P+T）
        logits = self.model(
            voc_ids, acc_ids_masked,
            attention_mask=acc_pad_mask,              # [B, T] 这里不 pad
            acc_clamp_embeds=acc_clamp_embeds         # [B,P,768] 或 [B,1,768]
        )

        # metrics：跳过前缀位
        if prefix_len > 0:
            logits_for_metric = logits[:, prefix_len:]
            labels_for_metric = labels[:, prefix_len:]
        else:
            logits_for_metric = logits
            labels_for_metric = labels

        loss = self.loss_func(
            logits.transpose(-1, -2), labels,
            ignore_index=self.model.special_tokens['acc_pad'],
            reduction='mean',
            label_smoothing=self.cfg.loss.label_smoothing
        )
        top10_accuracy = self.top10_accuracy(logits_for_metric.transpose(-1, -2).detach(), labels_for_metric)
        top1_accuracy  = self.top1_accuracy(logits_for_metric.transpose(-1, -2).detach(), labels_for_metric)
        return loss, top1_accuracy, top10_accuracy


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

        loss, top1_accuracy, top10_accuracy = self._process_batch(batch)

        self.log("train/total_loss", loss,
                 on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/lr", scheduler.get_last_lr()[0],
                 on_step=True, prog_bar=True, sync_dist=True)
        self.log("train/top_1_acc", top1_accuracy,
                 on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/top_10_acc", top10_accuracy,
                 on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.current_epoch
        loss, top1_accuracy, top10_accuracy = self._process_batch(batch)

        self.log("val/total_loss", loss,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/top_1_acc", top1_accuracy,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/top_10_acc", top10_accuracy,
                 on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optim_cfg = self.cfg.optim

        # 分组参数，为投影层设置不同的学习率
        params = []
        # if hasattr(self.model, 'clamp_proj'):
        #     # 为投影层设置不同的学习率（可选）
        #     params.append({
        #         'params': self.model.clamp_proj.parameters(),
        #         'lr': self.lr * 0.5  # 可调整的学习率乘数
        #     })

        # 添加其他模型参数
        params.append({
            'params': [p for n, p in self.model.named_parameters()
                    if p.requires_grad and not ('clamp_proj' in n)]
        })

        if optim_cfg.name == 'adamw':
            optimizer = torch.optim.AdamW(params,
                                        lr=self.lr,
                                        betas=optim_cfg.betas,
                                        eps=optim_cfg.eps,
                                        weight_decay=optim_cfg.weight_decay)


        if optim_cfg.scheduler.name == 'cos':
            scheduler = get_scheduler(name="cosine",
                                      optimizer=optimizer,
                                      num_warmup_steps=optim_cfg.scheduler.warmup,
                                      num_training_steps=self.trainer.estimated_stepping_batches)

        lr_scheduler_config = {
            'scheduler': scheduler,
            'interval': 'step'
        }
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}


    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """
        在从 ckpt 恢复状态时手动修改学习率
        """
        new_lr = 5e-6  # 设置为你想要救急的更小的学习率

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
