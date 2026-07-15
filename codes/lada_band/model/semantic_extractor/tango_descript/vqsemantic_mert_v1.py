import json
import torch
from tqdm import tqdm
from models_transformercond_winorm_ch16_everything_1024_trainvq import PromptCondAudioDiffusion_ENC
from diffusers import DDIMScheduler, DDPMScheduler
import torchaudio
import librosa
import os
import math
import numpy as np
from get_mulan import get_mulan

# We do not need melvae in code extractor
# from get_melvaehifigan48k import build_pretrained_models

import tools.torch_tools as torch_tools
import torch.nn as nn


class Mert_VQ_Semantic(nn.Module):
    def __init__(self,
        model_path,
        sample_rate=48000,
        device="cpu"):
        super().__init__()

        self.model = PromptCondAudioDiffusion_ENC().to(device)
        main_weights = torch.load(model_path, map_location=device)
        self.model.load_state_dict(main_weights, strict=False)
        print ("Successfully loaded checkpoint from:", model_path)
        self.model.eval()

        # Some hyper-parameters
        self.sample_rate = sample_rate
        self.channels = 1
        self.frame_rate = 25

    def forward(self, audio):
        codes = self.model.fetch_codes(audio)
        return codes

    @torch.no_grad()
    def encode(self, audio):
        if len(audio.shape)==3 and audio.shape[1] == 1:
            audio = audio.squeeze(1)
        codes = self.model.fetch_codes(audio)
        return codes


if __name__=="__main__":
    import torchaudio
    wav, sr = torchaudio.load("wav/instrument_testset_ukulele_diff_L_1.wav")
    audio_tokenizer = Mert_VQ_Semantic(model_path="model_cards/TrainVQ-Highquality-40kiter-20240206/pytorch_model_2.bin.net")
    audio_tokenizer.cuda()

    codes = audio_tokenizer(wav.cuda())
    import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
