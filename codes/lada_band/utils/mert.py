from .audio_utils import _av_read, convert_audio
import os
from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
from torch import nn
import torchaudio.transforms as T
from tqdm import tqdm
import pickle
import glob
import typing as tp


def _resolve_mert_pretrained_path():
    env_path = os.environ.get('LADA_PRETRAINED_ROOT')
    if env_path:
        candidate = os.path.join(env_path, 'MERT-v1-95M')
        if os.path.isdir(candidate):
            return candidate
    return 'm-a-p/MERT-v1-95M'


_MERT_PRETRAINED_PATH = _resolve_mert_pretrained_path()


def mert_init(mert_name='MERT-v1') -> tp.Tuple[AutoModel, Wav2Vec2FeatureExtractor]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModel.from_pretrained(_MERT_PRETRAINED_PATH, trust_remote_code=True).to(device)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(_MERT_PRETRAINED_PATH, trust_remote_code=True)

    return model, processor


def mert_predict(processor: Wav2Vec2FeatureExtractor, model: AutoModel, audio: torch.Tensor):
    mert_sr = 24000
    chunk_len_s = 60

    mert_embedds = []
    for start in range(0, len(audio), mert_sr * chunk_len_s):
        end = min(start + mert_sr * chunk_len_s, len(audio))
        input_x = audio[start:end]

        input_x_processed = input_x.detach().float().cpu().numpy()

        inputs = processor(input_x_processed, sampling_rate=mert_sr, return_tensors="pt")

        device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        inputs['input_values'] = inputs['input_values'].to(device=device, dtype=model_dtype)
        inputs['attention_mask'] = inputs['attention_mask'].to(device=device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
        outputs = all_layer_hidden_states.mean(-3)

        mert_embedds.append(outputs)

    mert_embedds = torch.cat(mert_embedds, dim=0)

    return mert_embedds
