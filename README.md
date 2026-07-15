<div align="center">
  <h1>LaDA-Band: Language Diffusion Models for Vocal-to-Accompaniment Generation</h1>
  <p>
    <a href="https://arxiv.org/abs/2604.11052">
      <img src="https://img.shields.io/badge/arXiv-2604.11052-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv">
    </a>
    <a href="https://huggingface.co/sDuoluoluos/LaDA-Band">
      <img src="https://img.shields.io/badge/Hugging%20Face-Models-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face">
    </a>
    <a href="https://duoluoluos.github.io/TME-LaDA-Band/">
      <img src="https://img.shields.io/badge/Project%20Page-Demos-4c1?style=for-the-badge&logo=github&logoColor=white" alt="Project Page">
    </a>
  </p>
</div>

LaDA-Band generates an accompaniment from a dry vocal and an optional text description. It frames vocal-to-accompaniment generation as discrete masked diffusion with dual-track prefix conditioning and a two-stage training curriculum, targeting musical coherence and dynamic orchestration alongside acoustic quality.

> **ACM Multimedia 2026 (ACM MM '26).** See the [paper](https://arxiv.org/abs/2604.11052) and listen on the [interactive project page](https://duoluoluos.github.io/TME-LaDA-Band/).

## News

- **2026-07-11:** Released the public training/inference code, a reproducible direct-inference configuration, environment specifications, and a command-line demo launcher.
- **2026-07-10:** LaDA-Band was accepted to ACM Multimedia 2026.


## Repository layout

```text
.
├── codes/
│   ├── infer.py                 # primary inference entry point
│   ├── train.py                 # research training entry point
│   ├── lada_band/               # LaDA-Band models, data code, and configurations
│   ├── MuCodec/                 # vendored codec component
│   └── clamp3/                  # vendored CLaMP3 component
├── demo/                        # instructions for user-supplied vocal inputs
├── scripts/run_demo.sh          # concise inference launcher
├── requirements.txt             # validated inference environment
├── requirements-train.txt       # inference environment plus experiment logging
└── THIRD_PARTY_NOTICES.md
```

## Quick start

### 1. Create an environment


```bash
git clone https://github.com/Duoluoluos/TME-LaDA-Band.git
cd TME-LaDA-Band

conda create -n lada-band python=3.12 -y
conda activate lada-band

# Install the PyTorch build matching your CUDA runtime first.
pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0

# The Tsinghua mirror is convenient for the remaining Python packages.
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

`requirements.txt` intentionally pins `fairseq-fixed==0.12.3.1`; do not replace it with the unrelated upstream `fairseq` package. Confirm the environment before downloading large assets:

```bash
python - <<'PY'
import fairseq, numpy, scipy, torch
print('fairseq:', fairseq.__version__)
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
print('numpy:', numpy.__version__, 'scipy:', scipy.__version__)
PY
pip check
```

### 2. Obtain model assets

Request access to the [LaDA-Band Hugging Face repository](https://huggingface.co/sDuoluoluos/LaDA-Band) and follow its model-card terms. Keep the downloaded assets outside this Git checkout. The direct-inference configuration expects this layout under `LADA_PRETRAINED_ROOT`:

```text
pretrained/
├── Llama-3.2-1B/
├── MERT-v1-95M/
├── clamp3/
│   └── weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth
├── hub/
│   ├── models--m-a-p--MERT-v1-330M/
│   └── models--xlm-roberta-base/
├── mucodec/
│   ├── audioldm_48k.pth
│   ├── mucodec.pt
│   └── muq.pt
```

Pass the main LaDA-Band checkpoint separately with `LADA_CHECKPOINT_PATH`. The repository ignores common audio and weight extensions so that local assets are not committed accidentally.

### 3. Generate an accompaniment

Use a vocal recording that you are authorized to process. The launcher defaults to a 30-second segment, WAV output, and 60 diffusion sampling steps.

```bash
export LADA_PRETRAINED_ROOT=/absolute/path/to/pretrained
export LADA_CHECKPOINT_PATH=/absolute/path/to/lada_band_lm1B_total3B.ckpt
export CUDA_VISIBLE_DEVICES=0

bash scripts/run_demo.sh \
  demo/dry_vocal/102270106.m4a \
  "A powerful and symphonic metal track featuring intense guitar riffs, dramatic orchestral arrangements, and emotive vocals. This track has a intense, dramatic, emotional mood." \
  outputs/example
```

The generated accompaniment is written below `outputs/example/generated_acc/`. Set `LADA_SEGMENT_START`, `LADA_SEGMENT_END`, `LADA_SAMPLING_STEPS`, or `LADA_SAVE_FORMAT` before invoking the launcher to override its defaults. MP3 I/O requires a working FFmpeg installation; WAV avoids that extra system dependency.

For full control, call the entry point directly from `codes/`:

```bash
cd codes
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --model_fp "$LADA_CHECKPOINT_PATH" \
  --config lada_band/conf/infer_1B.yaml \
  --output_dir ../outputs/manual-run \
  --seg_time 0 30 \
  --vocal_fps demo/dry_vocal/102270106.m4a \
  --texts "A powerful and symphonic metal track featuring intense guitar riffs, dramatic orchestral arrangements, and emotive vocals. This track has a intense, dramatic, emotional mood." \
  --sampling_steps 60 \
  --save_format wav \
```

The model runs on logical `cuda:0`; use `CUDA_VISIBLE_DEVICES` to choose the physical GPU. CPU inference is not supported by this release.


## Training

The repository includes the research training entry point and sanitized configuration templates. Configure your own licensed data paths before using `codes/train.py`.

## Responsible use

- Use only vocals, text prompts, and reference material that you have the right to process.
- Generated music may resemble stylistic traits present in the input or model data; do not present generated material as an identifiable artist's work.
- Respect the gated model repository's terms and every third-party component's license.

## License and acknowledgements

First-party LaDA-Band source and model-weight terms must be specified by the project owners before redistribution or commercial use. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the license files retained with vendored components for their respective terms.

## Citation

```bibtex
@article{wang2026ladaband,
  title={LaDA-Band: Language Diffusion Models for Vocal-to-Accompaniment Generation},
  author={Wang, Qi and Shen, Zhexu and Chen, Meng and Yu, Guoxin and Pang, Chaoxu and Zhao, Weifeng and Zhou, Wenjiang},
  journal={arXiv preprint arXiv:2604.11052},
  year={2026}
}
```
