import os

# Respect externally-provided env vars (e.g. when launching DDP).
os.environ.setdefault('WANDB_MODE', 'online')
# Set LADA_TMPDIR when a shared or high-capacity temporary directory is needed.
_lada_tmpdir = os.environ.get('LADA_TMPDIR')
if _lada_tmpdir:
    os.environ.setdefault('TMPDIR', _lada_tmpdir)
    os.environ.setdefault('TEMP', _lada_tmpdir)
    os.environ.setdefault('TMP', _lada_tmpdir)
# Helps reduce CUDA allocator fragmentation on long runs (often lowers OOM risk).
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import lightning as L
import importlib
from lada_band.module.data_module import DataModule
# from omegaconf import OmegaConf

from lightning.pytorch.loggers import WandbLogger
import hashlib

from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import functools
# Ensure MuCodec is in path
import sys
_CODES_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _CODES_ROOT)
sys.path.insert(0, os.path.join(_CODES_ROOT, 'MuCodec'))
from generate import MuCodec
import yaml
from lada_band.utils.config_utils import parse_args_and_load_config
from lada_band.utils.precision_utils import resolve_training_precision

torch.set_float32_matmul_precision('high')

def _resolve_model_scale(cfg) -> str:
    explicit_scale = cfg.get("model_scale")
    if explicit_scale is not None:
        scale = str(explicit_scale).strip().upper()
        if scale in {"0.5B", "0_5B", "500M"}:
            return "0.5B"
        if scale in {"1B", "3B"}:
            return scale
        raise ValueError(
            f"unsupported model_scale={explicit_scale!r}, expected '0.5B', '1B' or '3B'"
        )

    llama_cfg = cfg.model.llama
    hidden_size = llama_cfg.get("hidden_size")
    num_hidden_layers = llama_cfg.get("num_hidden_layers")

    if hidden_size == 2048 and num_hidden_layers == 8:
        return "0.5B"
    if hidden_size == 2048 and num_hidden_layers == 16:
        return "1B"
    if hidden_size == 3072 and num_hidden_layers == 28:
        return "3B"

    raise ValueError(
        "unable to infer model scale from cfg.model.llama; "
        "please set model_scale to '0.5B', '1B' or '3B'"
    )

def train(cfg):
    L.seed_everything(seed=cfg.seed)
    requested_precision = str(cfg.train.get("precision", "32-true"))
    precision = resolve_training_precision(cfg)
    cfg.train.precision = precision
    if precision == "32-true":
        cfg.train.optimize_vram = False

    ## handle args and logger
    md5 = hashlib.md5()
    md5.update(str(cfg).encode('utf-8'))
    exp_name = md5.hexdigest()
    exp_dir = os.path.join('exp', exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    model_scale = _resolve_model_scale(cfg)
    cfg.model_scale = model_scale
    # OmegaConf.save(cfg, os.path.join(exp_dir, 'config.yaml'))
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg.to_dict(), f)

    module = importlib.import_module('lada_band.module.' + cfg.project)
    model_module = module.ModelModule(cfg)
    data_module = DataModule(cfg)

    if cfg.wandb:
        wandb_logger = WandbLogger(project=cfg.wandb_name, name=exp_name)
        # Watching full graphs/params across all ranks is memory-heavy; keep it to rank0 only.
        if int(os.environ.get("RANK", "0")) == 0 and bool(cfg.get("wandb_watch", True)):
            wandb_logger.watch(model_module, log=cfg.get("wandb_watch_log", "all"))
    else:
        wandb_logger = None

    ckpt_dir = cfg.get('ckpt_dir')
    if ckpt_dir is None:
        ckpt_dir = 'checkpoints'
    else:
        print(f"Checkpoints will be saved to: {ckpt_dir}")

    ckpt_callback = ModelCheckpoint(
    dirpath=ckpt_dir,
    filename=f'lada_step_{{step}}_{model_scale}',
    every_n_train_steps=500,  # 跑完第 1 个 step 就保存！
    save_top_k=1,
    save_last=False
    )

    gradient_checkpointing_enabled = bool(cfg.train.get("gradient_checkpointing", False))
    find_unused = bool(cfg.train.get("find_unused_parameters", True))
    if gradient_checkpointing_enabled and find_unused:
        print(
            "[train.py] gradient_checkpointing=True with DDP find_unused_parameters=True "
            "can trigger 'Expected to mark a variable ready only once'; "
            "forcing find_unused_parameters=False."
        )
        find_unused = False
        cfg.train.find_unused_parameters = False
    strategy = 'ddp_find_unused_parameters_true' if find_unused else 'ddp'

    ## train settings
    if cfg.get('debug'):
        devices = -1
        if cfg.get('small_dataset'):
            limit_train_batches = None
            limit_val_batches = None
        else:
            limit_train_batches = 0.003
            limit_val_batches = 0.01
        torch.set_printoptions(profile="full")
        wandb_logger = None
    else:
        devices = -1
        limit_train_batches = None
        limit_val_batches = None

    enable_val = bool(cfg.train.get("enable_val", True))
    if not enable_val:
        limit_val_batches = 0

    print(
        "[train.py] effective config "
        f"project={cfg.project} "
        f"model_scale={model_scale} "
        f"data.batch_size={cfg.data.batch_size} "
        f"train.acc_grad={cfg.train.acc_grad} "
        f"precision={precision} "
        f"enable_val={enable_val} "
        f"requested_precision={requested_precision} "
        f"force_fp32={bool(cfg.train.get('force_fp32', True))} "
        f"optimize_vram={bool(cfg.train.get('optimize_vram', False))}"
    )
    if precision != requested_precision:
        print(f"[train.py] precision override: {requested_precision} -> {precision}")

    trainer_kwargs = dict(
        accelerator='gpu',
        devices=devices,
        strategy=strategy,
        accumulate_grad_batches=cfg.train.acc_grad,
        gradient_clip_val=cfg.train.grad_clip,
        gradient_clip_algorithm="norm",
        logger=wandb_logger,
        default_root_dir=exp_dir,
        callbacks=[ckpt_callback],
        precision=precision,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        num_sanity_val_steps=0 if not enable_val else 2,
        max_epochs=-1,
        max_steps=cfg.train.max_steps
    )
    if enable_val:
        trainer_kwargs["val_check_interval"] = cfg.train.check_val_every_n_steps

    trainer = L.Trainer(**trainer_kwargs)

    # 2. Decide whether we are doing a full-state resume or weight-only init.
    fit_ckpt_path = cfg.get('resume_ckpt_fp')
    if fit_ckpt_path is None and bool(cfg.get('resume_training', False)):
        fit_ckpt_path = cfg.get('l_ckpt_fp')

    if fit_ckpt_path:
        if not os.path.exists(fit_ckpt_path):
            raise FileNotFoundError(f"resume checkpoint not found: {fit_ckpt_path}")
        if cfg.get('l_ckpt_fp'):
            print(
                f"[train.py] full-state resume requested from {fit_ckpt_path}; "
                "skipping weight-only load path."
            )
    elif cfg.get('l_ckpt_fp'):
        if not os.path.exists(cfg.l_ckpt_fp):
            raise FileNotFoundError(f"weight checkpoint not found: {cfg.l_ckpt_fp}")
        print(f"🔥🔥🔥正在加载预训练权重用于微调: {cfg.l_ckpt_fp}")
        model_module.load_model(cfg.l_ckpt_fp)

    print("trainable total", sum(p.numel() for p in model_module.parameters() if p.requires_grad))
    print("trainable lm", sum(p.numel() for p in model_module.model.lm.parameters() if p.requires_grad))
    print("trainable moa", sum(p.numel() for n,p in model_module.model.named_parameters() if p.requires_grad and ("lora_" in n or "router" in n)))
    for n,p in model_module.model.lm.named_parameters():
        if p.requires_grad and ("router" not in n and "lora_" not in n):
            print("BAD_LM_TRAINABLE", n)
            break

    # 3. 启动训练
    if fit_ckpt_path:
        print(f">>> 从 checkpoint 恢复训练（保留 optimizer / scheduler / global_step）: {fit_ckpt_path}")
    else:
        print(">>> 开始新的训练阶段（重置优化器和Step）...")
    trainer.fit(model=model_module, datamodule=data_module, ckpt_path=fit_ckpt_path)

if __name__ == '__main__':
    cfg = parse_args_and_load_config()
    train(cfg)
