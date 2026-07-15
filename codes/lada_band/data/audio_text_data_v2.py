import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import omegaconf
from omegaconf import OmegaConf
import numpy as np
from ..utils.audio_utils import _av_read, _av_info, AudioFileInfo, convert_audio, normalize_loudness
from pathlib import Path
import os
import pickle
import typing as tp
import random
from .fault_tolerant_audio_dataset import FaultTolerantAudioDatasetMixin

class ArrangeDataset(FaultTolerantAudioDatasetMixin, Dataset):
    '''
    送入dataset前需要确保:
    1. voc和acc音频长度相等, 采样率相等
    // 2. 音频为目标采样率和channel?
    3. 提前计算好meta信息, 方便选

    dataset功能
    1. 随机切duration长度音频片段
    2. 加高斯噪声
    3. 简单vad?
    '''
    def __init__(self, cfg: omegaconf.DictConfig, mode, debug=False):
        self.debug = debug
        self.duration = cfg.data.duration
        self.ref_duration = cfg.ref_duration
        self.sr = cfg.data.sr
        self.channels = cfg.data.channels
        self.voc_dir = cfg.data.voc_dir
        self.acc_dir = cfg.data.acc_dir
        self.voc_suffix = cfg.data.voc_suffix
        self.acc_suffix = cfg.data.acc_suffix
        # 添加是否有子目录的参数，默认为False
        self.have_subdir = getattr(cfg.data, 'have_subdir', False)

        self.noise_cfg = cfg.data.noise

        with open(cfg.data.meta_fp, 'rb') as f:
            self.metas: tp.Dict[str, AudioFileInfo] = pickle.load(f)
        self.mode = mode
        self.freq_balance_cfg = self._build_frequency_balance_cfg(cfg.data.get('frequency_balance'))
        if mode == 'train':
            songid_fp = cfg.data.train_fp
        elif mode == 'val':
            songid_fp = cfg.data.val_fp
        elif mode == 'test':
            songid_fp = cfg.data.val_fp
        with open(songid_fp, 'r') as f:
            self.songid_list = f.read().splitlines()
        self.target_frames = int(self.sr * self.duration)
        self.ref_target_frames = int(self.sr * self.ref_duration)
        self._init_fault_tolerance(cfg, mode)

    def __len__(self):
        return len(self.songid_list)

    def _build_frequency_balance_cfg(self, cfg: tp.Optional[omegaconf.DictConfig]) -> tp.Dict[str, tp.Any]:
        cfg = cfg if cfg is not None else OmegaConf.create({})
        bass_cfg = cfg.get('bass')
        if bass_cfg is None:
            bass_cfg = OmegaConf.create({})
        mid_cfg = cfg.get('mid')
        if mid_cfg is None:
            mid_cfg = OmegaConf.create({})
        treble_cfg = cfg.get('treble')
        if treble_cfg is None:
            treble_cfg = OmegaConf.create({})

        return {
            'enabled': self.mode == 'train' and bool(cfg.get('enabled', False)),
            'prob': float(cfg.get('prob', 0.0)),
            'apply_to_voc': bool(cfg.get('apply_to_voc', False)),
            'apply_to_acc': bool(cfg.get('apply_to_acc', True)),
            'apply_to_ref': bool(cfg.get('apply_to_ref', True)),
            'bass_min_cut_db': float(bass_cfg.get('min_cut_db', 2.0)),
            'bass_max_cut_db': float(bass_cfg.get('max_cut_db', 5.0)),
            'bass_freq_min': float(bass_cfg.get('freq_min', 80.0)),
            'bass_freq_max': float(bass_cfg.get('freq_max', 180.0)),
            'bass_q_min': float(bass_cfg.get('q_min', 0.6)),
            'bass_q_max': float(bass_cfg.get('q_max', 1.1)),
            'mid_min_gain_db': float(mid_cfg.get('min_gain_db', 1.0)),
            'mid_max_gain_db': float(mid_cfg.get('max_gain_db', 3.5)),
            'mid_freq_min': float(mid_cfg.get('freq_min', 1200.0)),
            'mid_freq_max': float(mid_cfg.get('freq_max', 2800.0)),
            'mid_q_min': float(mid_cfg.get('q_min', 0.6)),
            'mid_q_max': float(mid_cfg.get('q_max', 1.2)),
            'treble_min_gain_db': float(treble_cfg.get('min_gain_db', 1.0)),
            'treble_max_gain_db': float(treble_cfg.get('max_gain_db', 4.0)),
            'treble_freq_min': float(treble_cfg.get('freq_min', 3200.0)),
            'treble_freq_max': float(treble_cfg.get('freq_max', 6000.0)),
            'treble_q_min': float(treble_cfg.get('q_min', 0.6)),
            'treble_q_max': float(treble_cfg.get('q_max', 1.0)),
        }

    @staticmethod
    def _sample_range(min_value: float, max_value: float) -> float:
        min_value = float(min_value)
        max_value = float(max_value)
        if max_value <= min_value:
            return min_value
        return random.uniform(min_value, max_value)

    def _apply_frequency_balance(self, wav: torch.Tensor) -> torch.Tensor:
        cfg = self.freq_balance_cfg
        if (not cfg['enabled']) or random.random() > cfg['prob']:
            return wav

        bass_gain = -self._sample_range(cfg['bass_min_cut_db'], cfg['bass_max_cut_db'])
        bass_freq = self._sample_range(cfg['bass_freq_min'], cfg['bass_freq_max'])
        bass_q = self._sample_range(cfg['bass_q_min'], cfg['bass_q_max'])

        mid_gain = self._sample_range(cfg['mid_min_gain_db'], cfg['mid_max_gain_db'])
        mid_freq = self._sample_range(cfg['mid_freq_min'], cfg['mid_freq_max'])
        mid_q = self._sample_range(cfg['mid_q_min'], cfg['mid_q_max'])

        treble_gain = self._sample_range(cfg['treble_min_gain_db'], cfg['treble_max_gain_db'])
        treble_freq = self._sample_range(cfg['treble_freq_min'], cfg['treble_freq_max'])
        treble_q = self._sample_range(cfg['treble_q_min'], cfg['treble_q_max'])

        wav = torchaudio.functional.bass_biquad(
            wav, self.sr, gain=bass_gain, central_freq=bass_freq, Q=bass_q
        )
        wav = torchaudio.functional.equalizer_biquad(
            wav, self.sr, center_freq=mid_freq, gain=mid_gain, Q=mid_q
        )
        wav = torchaudio.functional.treble_biquad(
            wav, self.sr, gain=treble_gain, central_freq=treble_freq, Q=treble_q
        )
        return wav

    @staticmethod
    def _validate_audio_tensor(name: str, wav: torch.Tensor, songid: str) -> None:
        if wav.numel() == 0:
            raise RuntimeError(f"{name} is empty for songid={songid}")
        if not torch.isfinite(wav).all():
            raise RuntimeError(f"{name} contains non-finite values for songid={songid}")

    def __getitem__(self, index):
        return self._safe_getitem(index, self._load_item)

    def _load_item(self, index):
        songid = self.songid_list[index]
        # 当have_subdir=True时，在子目录中查找文件
        if self.have_subdir:
            song_relative_dir = self.metas[songid].song_relative_dir
            voc_fp = os.path.join(self.voc_dir, song_relative_dir, songid+'.'+self.voc_suffix)
            acc_fp = os.path.join(self.acc_dir, song_relative_dir, songid+'.'+self.acc_suffix)
        else:
            voc_fp = os.path.join(self.voc_dir, songid+'.'+self.voc_suffix)
            acc_fp = os.path.join(self.acc_dir, songid+'.'+self.acc_suffix)

        ## seek audio
        meta = self.metas[songid]
        max_start = max(0, meta.duration - self.duration)
        start = np.random.rand() * max_start
        ref_start = np.random.rand() * max(0, meta.duration - self.ref_duration)
        # 捕获音频加载/转换过程中的异常
        voc, sr = _av_read(voc_fp, seek_time=start, duration=self.duration)

        if voc.shape[-1] == 0:
            raise RuntimeError("vocals音频长度为0")
        voc = convert_audio(voc, from_rate=sr, to_rate=self.sr, to_channels=self.channels).squeeze(0)
        if self.freq_balance_cfg['apply_to_voc']:
            voc = self._apply_frequency_balance(voc)
        voc = normalize_loudness(voc)

        acc, sr = _av_read(acc_fp, seek_time=start, duration=self.duration)

        if acc.shape[-1] == 0:
            raise RuntimeError("accompaniment音频长度为0")
        acc = convert_audio(acc, from_rate=sr, to_rate=self.sr, to_channels=self.channels).squeeze(0)
        if self.freq_balance_cfg['apply_to_acc']:
            acc = self._apply_frequency_balance(acc)
        acc = normalize_loudness(acc)

        ref, sr = _av_read(acc_fp, seek_time=ref_start, duration=self.ref_duration)
        if ref.shape[-1] == 0:
            raise RuntimeError("reference音频长度为0")
        ref = convert_audio(ref, from_rate=sr, to_rate=self.sr, to_channels=1).squeeze(0)
        if self.freq_balance_cfg['apply_to_ref']:
            ref = self._apply_frequency_balance(ref)
        ref = normalize_loudness(ref)


        text = meta.text
        ## add noise
        if self.noise_cfg.name == 'gaussian':
            noise = torch.randn_like(voc) * self.noise_cfg.sigma
            voc = voc + noise

        ## pad
        voc = F.pad(voc, (0, self.target_frames - voc.shape[-1]), value=0.0)
        acc = F.pad(acc, (0, self.target_frames - acc.shape[-1]), value=0.0)
        ref = F.pad(ref, (0, self.ref_target_frames - ref.shape[-1]), value=0.0)

        self._validate_audio_tensor('voc_audio', voc, songid)
        self._validate_audio_tensor('acc_audio', acc, songid)
        self._validate_audio_tensor('ref_audio', ref, songid)

        if self.debug:
            print(songid, start)

        # 正常样本：valid=True
        result = {
            'voc_audio': voc, 'acc_audio': acc, 'ref_audio': ref,
            'start': start, 'songid': songid, 'text': text
        }
        return result
