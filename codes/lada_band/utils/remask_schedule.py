import math
from typing import Iterable, List, Optional, Tuple


DEFAULT_REMASK_SCHEDULE_SPECS = (
    "cosine",
    "linear",
    "sqrt",
    "square",
    "cubic",
)


def _clamp_ratio(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _parse_power_value(raw_value: str) -> float:
    try:
        power = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"Invalid power schedule spec: {raw_value}") from exc
    if power <= 0:
        raise ValueError(f"power must be > 0, got {power}")
    return power


def normalize_remask_schedule_spec(spec: str) -> Tuple[str, Optional[float]]:
    if spec is None:
        raise ValueError("remask schedule spec must not be None")

    normalized = str(spec).strip().lower()
    if not normalized:
        raise ValueError("remask schedule spec must not be empty")

    aliases = {
        "cos": "cosine",
        "lin": "linear",
        "poly": "power",
        "pow": "power",
    }

    if ":" in normalized:
        name, raw_value = normalized.split(":", 1)
        name = aliases.get(name, name)
        if name != "power":
            raise ValueError(
                f"Unsupported remask schedule spec: {spec}. "
                "Only power schedules accept the name:value form."
            )
        return name, _parse_power_value(raw_value)

    normalized = aliases.get(normalized, normalized)
    named_powers = {
        "sqrt": 0.5,
        "square": 2.0,
        "cubic": 3.0,
    }
    if normalized in named_powers:
        return "power", named_powers[normalized]

    if normalized in {"cosine", "linear"}:
        return normalized, None

    raise ValueError(
        f"Unsupported remask schedule spec: {spec}. "
        "Expected one of cosine, linear, sqrt, square, cubic, power:<float>."
    )


def remask_schedule_to_slug(spec: str) -> str:
    name, power = normalize_remask_schedule_spec(spec)
    if name != "power":
        return name

    power_text = f"{power:g}".replace("-", "neg").replace(".", "p")
    return f"power_{power_text}"


def _compute_ratio(progress: float, schedule_name: str, power: Optional[float]) -> float:
    progress = _clamp_ratio(progress)
    if schedule_name == "cosine":
        return _clamp_ratio(math.cos(progress * math.pi / 2.0))
    if schedule_name == "linear":
        return _clamp_ratio(1.0 - progress)
    if schedule_name == "power":
        if power is None:
            raise ValueError("power schedule requires a power value")
        return _clamp_ratio((1.0 - progress) ** power)
    raise ValueError(f"Unsupported schedule name: {schedule_name}")


def validate_remask_ratios(remask_ratios: Iterable[float], sampling_steps: int) -> List[float]:
    ratios = [_clamp_ratio(value) for value in remask_ratios]
    if len(ratios) != sampling_steps:
        raise ValueError(
            f"Expected {sampling_steps} remask ratios, got {len(ratios)}"
        )

    for idx in range(1, len(ratios)):
        if ratios[idx] > ratios[idx - 1] + 1e-8:
            raise ValueError(
                "remask ratios must be monotonically non-increasing across denoising steps"
            )
    return ratios


def build_remask_ratios(
    sampling_steps: int,
    schedule_spec: str = "cosine",
    remask_ratios: Optional[Iterable[float]] = None,
) -> List[float]:
    if sampling_steps <= 0:
        raise ValueError(f"sampling_steps must be > 0, got {sampling_steps}")

    if remask_ratios is not None:
        return validate_remask_ratios(remask_ratios, sampling_steps)

    schedule_name, power = normalize_remask_schedule_spec(schedule_spec)
    ratios = []
    for step_idx in range(1, sampling_steps + 1):
        progress = step_idx / sampling_steps
        ratios.append(_compute_ratio(progress, schedule_name, power))
    return ratios


def build_remask_schedule_trace(
    sampling_steps: int,
    schedule_spec: str = "cosine",
    remask_ratios: Optional[Iterable[float]] = None,
) -> List[dict]:
    ratios = build_remask_ratios(
        sampling_steps=sampling_steps,
        schedule_spec=schedule_spec,
        remask_ratios=remask_ratios,
    )
    trace = []
    for step_idx, ratio in enumerate(ratios, start=1):
        trace.append(
            {
                "step": step_idx,
                "progress": step_idx / sampling_steps,
                "remask_ratio": ratio,
            }
        )
    return trace
