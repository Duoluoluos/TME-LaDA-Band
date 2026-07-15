import av
import julius
from dataclasses import dataclass
import typing as tp
from pathlib import Path
import logging
import torch
import subprocess
import numpy as np


def f32_pcm(wav: torch.Tensor) -> torch.Tensor:
    """Convert audio to float 32 bits PCM format.
    """
    if wav.dtype.is_floating_point:
        return wav
    elif wav.dtype == torch.int16:
        return wav.float() / 2**15
    elif wav.dtype == torch.int32:
        return wav.float() / 2**31
    raise ValueError(f"Unsupported wav dtype: {wav.dtype}")


_av_initialized = False
def _init_av():
    global _av_initialized
    if _av_initialized:
        return
    logger = logging.getLogger('libav.mp3')
    logger.setLevel(logging.ERROR)
    _av_initialized = True


@dataclass
class AudioFileInfo:
    sample_rate: int
    duration: float
    channels: int
    text: str = None
    song_relative_dir: str = None


def convert_audio_channels(wav: torch.Tensor, channels: int = 2) -> torch.Tensor:
    """Convert audio to the given number of channels.

    Args:
        wav (torch.Tensor): Audio wave of shape [B, C, T].
        channels (int): Expected number of channels as output.
    Returns:
        torch.Tensor: Downmixed or unchanged audio wave [B, C, T].
    """
    *shape, src_channels, length = wav.shape
    # print(src_channels, length)
    if src_channels == channels:
        pass
    elif channels == 1:
        # Case 1:
        # The caller asked 1-channel audio, and the stream has multiple
        # channels, downmix all channels.
        wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        # Case 2:
        # The caller asked for multiple channels, but the input file has
        # a single channel, replicate the audio over all channels.
        wav = wav.expand(*shape, channels, length)
    elif src_channels >= channels:
        # Case 3:
        # The caller asked for multiple channels, and the input file has
        # more channels than requested. In that case return the first channels.
        wav = wav[..., :channels, :]
    else:
        # Case 4: What is a reasonable choice here?
        raise ValueError('The audio file has less channels than requested but is not mono.')
    return wav

def convert_audio(wav: torch.Tensor, from_rate: float,
                  to_rate: float, to_channels: int) -> torch.Tensor:
    """Convert audio to new sample rate and number of audio channels."""
    wav = julius.resample_frac(wav, int(from_rate), int(to_rate))
    wav = convert_audio_channels(wav, to_channels)
    return wav

def normalize_loudness(wav: torch.Tensor, target_db: float = -16.0, max_peak_db: float = -1.0) -> torch.Tensor:
    """
    Normalize audio loudness to a target dB RMS, ensuring peak does not exceed max_peak_db.

    Args:
        wav (torch.Tensor): Audio wave of shape [C, T] or [T].
        target_db (float): Target RMS loudness in dB.
        max_peak_db (float): Maximum allowed peak in dB.
    Returns:
        torch.Tensor: Normalized audio.
    """
    # Calculate current RMS
    rms = torch.sqrt(torch.mean(wav**2))
    if rms < 1e-9:
        return wav

    current_db = 20 * torch.log10(rms)
    gain_db = target_db - current_db
    gain = 10 ** (gain_db / 20)

    # Apply gain
    wav_norm = wav * gain

    # Check peak
    peak = torch.max(torch.abs(wav_norm))
    if peak > 0:
        peak_db = 20 * torch.log10(peak)
        if peak_db > max_peak_db:
            # Scale down to max_peak_db
            peak_gain_db = max_peak_db - peak_db
            peak_gain = 10 ** (peak_gain_db / 20)
            wav_norm = wav_norm * peak_gain

    return wav_norm

def _av_info(filepath: tp.Union[str, Path]) -> AudioFileInfo:
    _init_av()
    with av.open(str(filepath)) as af:
        stream = af.streams.audio[0]
        sample_rate = stream.codec_context.sample_rate
        duration = float(stream.duration * stream.time_base)
        channels = stream.channels
        return AudioFileInfo(sample_rate, duration, channels, None)

def _av_read(filepath: tp.Union[str, Path, tp.BinaryIO], seek_time: float = 0, duration: float = -1., format=None) -> tp.Tuple[torch.Tensor, int]:
    """FFMPEG-based audio file reading using PyAV bindings.
    Soundfile cannot read mp3 and av_read is more efficient than torchaudio.

    Args:
        filepath (str or Path): Path to audio file to read.
        seek_time (float): Time at which to start reading in the file.
        duration (float): Duration to read from the file. If set to -1, the whole file is read.
    Returns:
        tuple of torch.Tensor, int: Tuple containing audio data and sample rate
    """
    _init_av()
    with av.open(filepath, format=format) as af:
        stream = af.streams.audio[0]
        sr = stream.codec_context.sample_rate
        num_frames = int(sr * duration) if duration >= 0 else -1
        frame_offset = int(sr * seek_time)
        # we need a small negative offset otherwise we get some edge artifact
        # from the mp3 decoder.
        af.seek(int(max(0, (seek_time - 0.1)) / stream.time_base), stream=stream)
        frames = []
        length = 0
        for frame in af.decode(streams=stream.index):
            current_offset = int(frame.rate * frame.pts * frame.time_base)
            strip = max(0, frame_offset - current_offset)
            buf = torch.from_numpy(frame.to_ndarray())
            if buf.shape[0] != stream.channels:
                buf = buf.view(-1, stream.channels).t()
            buf = buf[:, strip:]
            frames.append(buf)
            length += buf.shape[1]
            if num_frames > 0 and length >= num_frames:
                break
        assert frames
        # If the above assert fails, it is likely because we seeked past the end of file point,
        # in which case ffmpeg returns a single frame with only zeros, and a weird timestamp.
        # This will need proper debugging, in due time.
        wav = torch.cat(frames, dim=1)
        assert wav.shape[0] == stream.channels
        if num_frames > 0:
            wav = wav[:, :num_frames]
        return f32_pcm(wav), sr

def ffmpeg_load_local(fp, sr, channel=1):
    command = ["ffmpeg", "-i", fp,
               "-f", "f32le", "-acodec", "pcm_f32le",
               "-ar", str(sr), "-loglevel", "panic", "-ac", str(channel), "-"]
    try:
        pipe = subprocess.Popen(command, stdout=subprocess.PIPE, startupinfo=None)
        raw_audio = pipe.stdout.read()
        # print(raw_audio)
        x = np.frombuffer(raw_audio, dtype="float32")
    except Exception as e:
        error_info = "ffmpeg_load: {} exception is {}".format(fp, e)
        print(error_info)
        x = None
    finally:
        pipe.stdout.close()

    if x is None:
        return False, None
    else:
        return True, x

def ffmpeg_load_object(ob, sr, channel=1):
    command = ["ffmpeg", "-i", '-',
               "-f", "f32le", "-acodec", "libmp3lame",
               "-ar", str(sr), "-loglevel", "panic", "-ac", str(channel), "-"]
    try:
        pipe = subprocess.Popen(command, stdout=subprocess.PIPE, stdin=ob)
        raw_audio = pipe.stdout.read()
        # print(raw_audio)
        x = np.frombuffer(raw_audio, dtype="float32")
    except Exception as e:
        error_info = "ffmpeg_load: {} exception is {}".format(e)
        print(error_info)
        x = None
    finally:
        pipe.stdout.close()

    if x is None:
        return False, None
    else:
        return True, x
