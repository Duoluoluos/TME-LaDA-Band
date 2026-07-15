from transformers import EncodecModel, AutoProcessor, EncodecConfig
import omegaconf
import torch.nn as nn
import torch


class Codec(nn.Module):
    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = EncodecModel.from_pretrained(cfg.model_path)
        # self.processor = AutoProcessor.from_pretrained(cfg.model_path)

    @torch.no_grad()
    def encode(self, audio: torch.Tensor):  # TODO padding_mask没用
        assert audio.ndim == 3
        # assert padding_mask.ndim == 2

        encoder_outputs = self.model.encode(audio, bandwidth=self.cfg.bw)
        audio_codes = encoder_outputs.audio_codes[0]  # 第一维是frame, 这里默认设置是不分chunk所以纬度是1
        audio_scales = encoder_outputs.audio_scales
        # print('audio_codes', encoder_outputs.audio_codes.shape)
        return audio_codes, audio_scales

    @torch.no_grad()
    def decode(self, codes, scales):
        if codes.ndim == 3:  # decoder输入codes第一维得是frame
            codes = codes[None]
        # assert scales is None

        audio = self.model.decode(codes, scales)[0]  # (B, 1, T)

        return audio

    def forward(self, audio):
        assert audio.ndim == 3
        # assert padding_mask.ndim == 2

        codes, scales = self.model.encode(audio, bandwidth=self.cfg.bw)
        audio = self.model.decode(codes, scales)

        return audio
