import yaml
import random
import inspect
import os
import numpy as np
from tqdm import tqdm
import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from einops import repeat
from lada_band.model.semantic_extractor.tango_descript.tools.torch_tools import wav_to_fbank

import diffusers
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DDPMScheduler
from lada_band.model.semantic_extractor.tango_descript.models.transformer_2d import Transformer2DModel
from lada_band.model.semantic_extractor.tango_descript.get_mulan import get_mulan
# from libs.rvq2 import RVQEmbedding
from lada_band.model.semantic_extractor.tango_descript.libs.descript_quantize2 import ResidualVectorQuantize

class SampleProcessor(torch.nn.Module):
    def project_sample(self, x: torch.Tensor):
        """Project the original sample to the 'space' where the diffusion will happen."""
        return x

    def return_sample(self, z: torch.Tensor):
        """Project back from diffusion space to the actual sample space."""
        return z

class Feature2DProcessor(SampleProcessor):
    def __init__(self, dim: int = 8, power_std: tp.Union[float, tp.List[float], torch.Tensor] = 1., \
                 num_samples: int = 100_000):
        super().__init__()
        self.num_samples = num_samples
        self.dim = dim
        self.power_std = power_std
        self.register_buffer('counts', torch.zeros(1))
        self.register_buffer('sum_x', torch.zeros(dim, 32))
        self.register_buffer('sum_x2', torch.zeros(dim, 32))
        self.register_buffer('sum_target_x2', torch.zeros(dim, 32))
        self.counts: torch.Tensor
        self.sum_x: torch.Tensor
        self.sum_x2: torch.Tensor

    @property
    def mean(self):
        mean = self.sum_x / self.counts
        return mean

    @property
    def std(self):
        std = (self.sum_x2 / self.counts - self.mean**2).clamp(min=0).sqrt()
        return std

    @property
    def target_std(self):
        return 1

    def project_sample(self, x: torch.Tensor):
        assert x.dim() == 4
        if self.counts.item() < self.num_samples:
            self.counts += len(x)
            self.sum_x += x.mean(dim=(2,)).sum(dim=0)
            self.sum_x2 += x.pow(2).mean(dim=(2,)).sum(dim=0)
        rescale = (self.target_std / self.std.clamp(min=1e-12)) ** self.power_std  # same output size
        x = (x - self.mean.view(1, -1, 1, 32).contiguous()) * rescale.view(1, -1, 1, 32).contiguous()
        return x

    def return_sample(self, x: torch.Tensor):
        assert x.dim() == 4
        rescale = (self.std / self.target_std) ** self.power_std
        x = x * rescale.view(1, -1, 1, 32).contiguous() + self.mean.view(1, -1, 1, 32).contiguous()
        return x

class PromptCondAudioDiffusion(nn.Module):
    def __init__(
        self,
        num_channels,
        scheduler_name,
        unet_model_name=None,
        unet_model_config_path=None,
        snr_gamma=None,
        uncondition=True,
        out_paint=False,
    ):
        super().__init__()

        assert unet_model_name is not None or unet_model_config_path is not None, "Either UNet pretrain model name or a config file path is required"

        self.scheduler_name = scheduler_name
        self.unet_model_name = unet_model_name
        self.unet_model_config_path = unet_model_config_path
        self.snr_gamma = snr_gamma
        self.uncondition = uncondition

        # https://huggingface.co/docs/diffusers/v0.14.0/en/api/schedulers/overview
        print(self.scheduler_name)
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.scheduler_name, subfolder="scheduler")
        self.inference_scheduler = DDPMScheduler.from_pretrained(self.scheduler_name, subfolder="scheduler")
        self.normfeat = Feature2DProcessor(dim=num_channels)

        self.rsp48toclap = torchaudio.transforms.Resample(48000, 24000)
        _package_dir = os.path.dirname(os.path.abspath(__file__))
        _default_mulan_config = os.path.normpath(os.path.join(_package_dir, '../../conf/mulan.yaml'))
        self.clap_embd_extractor = get_mulan(config=_default_mulan_config).eval()
        self.rvq = ResidualVectorQuantize(input_dim = 1024, n_codebooks = 1, codebook_size = 10_000, codebook_dim = 32, quantizer_dropout = 0.0)
        self.cond_embedding = nn.Linear(1024, 16*32)


    def forward(self, input_audios):
        prompt_stride = 3
        input_audios = self.rsp48toclap(input_audios)
        inputs = self.clap_embd_extractor.mulan.audio.processor(input_audios, sampling_rate=self.clap_embd_extractor.mulan.audio.sr, return_tensors="pt")
        input_values = inputs['input_values'].squeeze(0).to(input_audios.device, dtype = input_audios.dtype)
        prompt_embeds = self.clap_embd_extractor.mulan.audio.model(input_values, output_hidden_states=True).hidden_states[13].detach() # batch_size, Time steps, 1024 feature_dim
        prompt_embeds = torch.nn.functional.avg_pool1d(prompt_embeds.permute(0,2,1), prompt_stride, prompt_stride) # b e t
        prompt_embeds = prompt_embeds.detach() # b e t
        quantized_prompt_embeds, codes, _, commitment_loss, codebook_loss = self.rvq(prompt_embeds) # b,d,t
        return codes, quantized_prompt_embeds


    @torch.no_grad()
    def fetch_codes(self, input_audios):
        prompt_stride = 3
        # input_audios = self.rsp48toclap(input_audios)
        inputs = self.clap_embd_extractor.mulan.audio.processor(input_audios, sampling_rate=self.clap_embd_extractor.mulan.audio.sr, return_tensors="pt")
        input_values = inputs['input_values'].squeeze(0).to(input_audios.device, dtype = input_audios.dtype)
        prompt_embeds = self.clap_embd_extractor.mulan.audio.model(input_values, output_hidden_states=True).hidden_states[13].detach() # batch_size, Time steps, 1024 feature_dim
        prompt_embeds = torch.nn.functional.avg_pool1d(prompt_embeds.permute(0,2,1), prompt_stride, prompt_stride) # b e t
        prompt_embeds = prompt_embeds.detach() # b e t
        quantized_prompt_embeds, codes, _, commitment_loss, codebook_loss = self.rvq(prompt_embeds) # b,d,t
        return codes, quantized_prompt_embeds
