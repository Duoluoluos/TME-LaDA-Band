import torch


def normalize_precision(precision) -> str:
    if precision is None:
        return "32-true"

    value = str(precision).strip().lower()
    aliases = {
        "32": "32-true",
        "fp32": "32-true",
        "float32": "32-true",
        "32-true": "32-true",
        "16": "16-mixed",
        "fp16": "16-mixed",
        "float16": "16-mixed",
        "16-mixed": "16-mixed",
        "16-true": "16-true",
        "bf16": "bf16-mixed",
        "bfloat16": "bf16-mixed",
        "bf16-mixed": "bf16-mixed",
        "bf16-true": "bf16-true",
    }
    return aliases.get(value, value)


def precision_to_dtype(precision) -> torch.dtype:
    normalized = normalize_precision(precision)
    if normalized.startswith("bf16"):
        return torch.bfloat16
    if normalized.startswith("16"):
        return torch.float16
    return torch.float32


def resolve_training_precision(cfg) -> str:
    requested = normalize_precision(cfg.train.get("precision", "32-true"))
    force_fp32 = bool(cfg.train.get("force_fp32", True))
    optimize_vram = bool(cfg.train.get("optimize_vram", False))

    if force_fp32:
        return "32-true"

    if optimize_vram and requested.endswith("-mixed"):
        return requested.replace("-mixed", "-true")

    return requested
