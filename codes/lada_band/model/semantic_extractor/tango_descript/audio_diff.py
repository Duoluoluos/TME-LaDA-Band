import yaml
import random
import inspect
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
        diff_mulan_config=None,
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
        self.clap_embd_extractor = get_mulan(config=diff_mulan_config).eval()
        self.rvq = ResidualVectorQuantize(input_dim = 1024, n_codebooks = 1, codebook_size = 10_000, codebook_dim = 32, quantizer_dropout = 0.0)
        self.cond_embedding = nn.Linear(1024, 16*32)

        if unet_model_config_path:
            self.unet = Transformer2DModel.from_config(
                unet_model_config_path,
            )
            self.set_from = "random"
            print("Transformer initialized from pretrain.")
        # self.unet.set_attn_processor(AttnProcessor2_0())
        # self.unet.set_use_memory_efficient_attention_xformers(True)

        # self.start_embedding = nn.Parameter(torch.randn(1,1024))
        # self.end_embedding = nn.Parameter(torch.randn(1,1024))

    def compute_snr(self, timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    def forward(self, input_audios, latents, is_start, is_end, validation_mode=False, train_rvq=True):
        if not hasattr(self,"device"):
            self.device = input_audios.device
        if not hasattr(self,"dtype"):
            self.dtype = input_audios.dtype
        device = self.device

        with torch.no_grad():
            prompt_stride = 3
            # input_audios = self.rsp48toclap(input_audios)
            inputs = self.clap_embd_extractor.mulan.audio.processor(input_audios, sampling_rate=self.clap_embd_extractor.mulan.audio.sr, return_tensors="pt")
            input_values = inputs['input_values'].squeeze(0).to(input_audios.device, dtype = input_audios.dtype)
            prompt_embeds = self.clap_embd_extractor.mulan.audio.model(input_values, output_hidden_states=True).hidden_states[13].detach() # batch_size, Time steps, 1024 feature_dim
            prompt_embeds = torch.nn.functional.avg_pool1d(prompt_embeds.permute(0,2,1), prompt_stride, prompt_stride) # b e t
            prompt_embeds = prompt_embeds.detach() # b e t

        if(train_rvq):
            quantized_prompt_embeds, _, _, commitment_loss, codebook_loss = self.rvq(prompt_embeds) # b,d,t
        else:
            self.rvq.eval()
            quantized_prompt_embeds, _, _, commitment_loss, codebook_loss = self.rvq(prompt_embeds) # b,d,t
            commitment_loss = commitment_loss.detach()
            codebook_loss = codebook_loss.detach()
            quantized_prompt_embeds = quantized_prompt_embeds.detach()
        # print(quantized_prompt_embeds.shape, embed_ind.shape, commit_loss.shape)
        quantized_prompt_embeds = self.cond_embedding(quantized_prompt_embeds.permute(0,2,1)) # b t 16*32
        quantized_prompt_embeds = quantized_prompt_embeds.reshape(quantized_prompt_embeds.shape[0], quantized_prompt_embeds.shape[1]//2, 2, 16, 32).reshape(quantized_prompt_embeds.shape[0], quantized_prompt_embeds.shape[1]//2, 2*16, 32).permute(0,2,1,3).contiguous() # b 32 t f

        num_train_timesteps = self.noise_scheduler.num_train_timesteps
        self.noise_scheduler.set_timesteps(num_train_timesteps, device=device)

        bsz, _, height, width = latents.shape
        resolution = torch.tensor([height, width]).repeat(bsz, 1)
        aspect_ratio = torch.tensor([float(height / width)]).repeat(bsz, 1)
        resolution = resolution.to(dtype=prompt_embeds.dtype, device=device)
        aspect_ratio = aspect_ratio.to(dtype=prompt_embeds.dtype, device=device)
        added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

        if self.uncondition:
            mask_indices = [k for k in range(quantized_prompt_embeds.shape[0]) if random.random() < 0.1]
            if len(mask_indices) > 0:
                quantized_prompt_embeds[mask_indices] = 0

        if validation_mode:
            timesteps = (self.noise_scheduler.num_train_timesteps//2) * torch.ones((bsz,), dtype=torch.int64, device=device)
        else:
            # Sample a random timestep for each instance
            timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=device)
        timesteps = timesteps.long()
        latents = self.normfeat.project_sample(latents)
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps) # b c t f

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        elif self.noise_scheduler.config.prediction_type == "sample":
            target = latents
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        if self.set_from == "random":
            model_pred = self.unet(
                torch.cat([noisy_latents, quantized_prompt_embeds],1),
                timestep = timesteps,
                added_cond_kwargs=added_cond_kwargs,
            ).sample
        else:
            raise ValueError(self.set_from)

        if self.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Adaptef from huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
            snr = self.compute_snr(timesteps)
            mse_loss_weights = (
                torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        return loss, commitment_loss.mean(), codebook_loss.mean()

    def init_device_dtype(self, device, dtype):
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def fetch_codes(self, input_audios):
        prompt_stride = 3
        # input_audios = self.rsp48toclap(input_audios)
        if torch.is_tensor(input_audios):
        # 如果 input_audios 在 GPU 上且是 bf16，必须转为 float32
        # Processor 内部会转 numpy，所以这里给它 float32 类型的 Tensor 或 numpy 都可以
            raw_audio = input_audios.detach().to(dtype=torch.float32).cpu().numpy()
        else:
            raw_audio = input_audios
        inputs = self.clap_embd_extractor.mulan.audio.processor(raw_audio, sampling_rate=self.clap_embd_extractor.mulan.audio.sr, return_tensors="pt")
        input_values = inputs['input_values'].to(input_audios.device, dtype=input_audios.dtype)
        # 获取模型权重的数据类型
        prompt_embeds = self.clap_embd_extractor.mulan.audio.model(
            input_values,
            output_hidden_states=True
        ).hidden_states[13].detach()
        prompt_embeds = torch.nn.functional.avg_pool1d(prompt_embeds.permute(0,2,1), prompt_stride, prompt_stride)

        prompt_embeds = prompt_embeds.detach()
        quantized_prompt_embeds, codes, _, commitment_loss, codebook_loss = self.rvq(prompt_embeds)
        return codes, quantized_prompt_embeds

    @torch.no_grad()
    def inference_codes(self, codes, inference_scheduler, guidance_scale=2, num_steps=20,
                  disable_progress=True):
        # import pdb; pdb.set_trace()
        classifier_free_guidance = guidance_scale > 1.0
        if not hasattr(self,"device"):
            self.device = codes.device
        device = self.device
        dtype = torch.float32

        inference_scheduler.set_timesteps(num_steps, device=device)
        timesteps = inference_scheduler.timesteps
        batch_size = codes.shape[0]

        quantized_prompt_embeds, _, _ = self.rvq.from_codes(codes)
        # import pdb; pdb.set_trace()
        # print(codes.shape, codes.max())

        quantized_prompt_embeds = self.cond_embedding(quantized_prompt_embeds.permute(0,2,1)) # b t 16*32
        quantized_prompt_embeds = quantized_prompt_embeds.reshape(quantized_prompt_embeds.shape[0], quantized_prompt_embeds.shape[1]//2, 2, 16, 32).reshape(quantized_prompt_embeds.shape[0], quantized_prompt_embeds.shape[1]//2, 2*16, 32).permute(0,2,1,3).contiguous() # b 32 t f
        num_frames = quantized_prompt_embeds.shape[-2]

        num_channels_latents = self.unet.config.in_channels - 32
        latents = self.prepare_latents(batch_size, num_frames, inference_scheduler, num_channels_latents, dtype, device)

        bsz, _, height, width = latents.shape
        resolution = torch.tensor([height, width]).repeat(bsz, 1)
        aspect_ratio = torch.tensor([float(height / width)]).repeat(bsz, 1)
        resolution = resolution.to(dtype=quantized_prompt_embeds.dtype, device=device)
        aspect_ratio = aspect_ratio.to(dtype=quantized_prompt_embeds.dtype, device=device)
        if classifier_free_guidance:
            resolution = torch.cat([resolution, resolution], 0)
            aspect_ratio = torch.cat([aspect_ratio, aspect_ratio], 0)
        added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

        if classifier_free_guidance:
            quantized_prompt_embeds = torch.cat([torch.zeros_like(quantized_prompt_embeds), quantized_prompt_embeds],0)

        num_warmup_steps = len(timesteps) - num_steps * inference_scheduler.order
        progress_bar = tqdm(range(num_steps), disable=disable_progress)

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2, 0) if classifier_free_guidance else latents
            latent_model_input = inference_scheduler.scale_model_input(latent_model_input, t)
            # print(i,t);latent_model_input = torch.from_numpy(np.load('../MusicLDM_huggingface/latent_model_input.npy')).to(self.device)[[0]]
            # import pdb;pdb.set_trace()
            noise_pred = self.unet(
                torch.cat([latent_model_input, quantized_prompt_embeds],1),
                timestep = torch.tensor([t,]*(1+classifier_free_guidance)*bsz).to(latent_model_input.device),
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            # print(noise_pred);exit()

            if classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = inference_scheduler.step(noise_pred, t, latents).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % inference_scheduler.order == 0):
                progress_bar.update(1)

        latents = self.normfeat.return_sample(latents)
        return latents

    @torch.no_grad()
    def inference(self, input_audios, inference_scheduler, guidance_scale=2, num_steps=20,
                  disable_progress=True):
        codes, _ = self.fetch_codes(input_audios)
        # import pdb; pdb.set_trace()AA

        latents = self.inference_codes(codes, inference_scheduler=inference_scheduler, \
            guidance_scale=guidance_scale, num_steps=num_steps, \
            disable_progress=disable_progress)
        return latents

    def prepare_latents(self, batch_size, num_frames, inference_scheduler, num_channels_latents, dtype, device):
        divisor = 4
        shape = (batch_size, num_channels_latents, num_frames, 32)
        if(num_frames%divisor>0):
            num_frames = round(num_frames/float(divisor))*divisor
            shape = (batch_size, num_channels_latents, num_frames, 32)
        latents = randn_tensor(shape, generator=None, device=device, dtype=dtype)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * inference_scheduler.init_noise_sigma
        return latents



class PromptCondAudioDiffusion_ENC(nn.Module):
    def __init__(self, sample_rate=48000, diff_mulan_config=None):
        super().__init__()
        # self.rsp48toclap = torchaudio.transforms.Resample(sample_rate, 24000)
        self.clap_embd_extractor = get_mulan(config=diff_mulan_config).eval()
        self.rvq = ResidualVectorQuantize(input_dim = 1024, n_codebooks = 1, codebook_size = 10_000, codebook_dim = 32, quantizer_dropout = 0.0)


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
