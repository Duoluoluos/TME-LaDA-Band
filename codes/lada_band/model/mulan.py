import torch
import torch.nn as nn
from .musiclm_pytorch.musiclm_pytorch import MuLaNEmbedder
from .musiclm_pytorch.utils import create_MuLaN_from_config
import omegaconf
import os
import sys
import numpy as np
from transformers import BertConfig, AutoTokenizer


class MuLanInfer(nn.Module):
    '''
    cfg: cfg.mulan
    '''
    def __init__(self, cfg: omegaconf.DictConfig, precision, ckpt_path=None):
        super().__init__()
        mulan = create_MuLaN_from_config(cfg)
        self.mulan_embedder = MuLaNEmbedder(mulan, sr=24000, clip_secs=10, checkpoint_path=ckpt_path)
        for param in self.mulan_embedder.parameters():
            param.requires_grad = False
        self.mulan_embedder.eval()

        self.precision = precision
        self.trans_precision = False

    @torch.no_grad()
    def _get_wav_embedding(self, audio) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            # Inference Check
            if self.precision.startswith('bf16'):
                audio = audio.to(torch.bfloat16)

            if audio.ndim == 3 and audio.shape[1] == 1:
                audio = audio.squeeze(1)

        # B, T
        if self.precision == 'bf16-mixed' and not self.trans_precision:
        # if self.precision == 'bf16-mixed' and not self.trans_precision and self.training:
        # if self.precision.startswith('bf16-mixed'):
            self.mulan_embedder = self.mulan_embedder.to(torch.float32)
            # self.mulan_embedder.cpu()
            # self.precision = 'fp32'
            self.trans_precision = True

        if self.trans_precision:
            with torch.cuda.amp.autocast(enabled=False):
                # Inference Check
                audio_embeds = self.mulan_embedder(wavs=audio.float())
            audio_embeds = audio_embeds.to(torch.bfloat16)
        else:
            audio_embeds = self.mulan_embedder(wavs=audio)  # [B, T] -> [B, 512]

        audio_embeds = audio_embeds.unsqueeze(1)
        return audio_embeds

    @torch.no_grad()
    def _get_text_embedding(self, texts) -> torch.Tensor:
        """Get the wav embedding from the WavCondition.
        The conditioner will either extract the embedding on-the-fly computing it from the condition wav directly
        or will rely on the embedding cache to load the pre-computed embedding if relevant.
        """
        # with torch.cuda.amp.autocast(enabled=False):
        text_embeds = self.mulan_embedder(texts = texts)
        text_embeds = text_embeds.unsqueeze(1)
        # import pdb; pdb.set_trace()
        return text_embeds

    def forward(self, audio=None, texts=None):
        assert (audio is not None) ^ (texts is not None)

        if texts is None:
            with torch.no_grad():
                embeds = self._get_wav_embedding(audio)
        else:
            with torch.no_grad():
                embeds = self._get_text_embedding(texts)

        # TODO: audio mask
        return embeds

# You can import Clamp3 from the separate module if needed
# from .clamp3 import Clamp3
