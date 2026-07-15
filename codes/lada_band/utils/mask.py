import torch
import random

def span_mask(x, r, min_spans=1, max_spans=3):
    """
    基于区间的Mask生成 (Span Masking)
    x: [B, T]
    r: [B] or scalar, 这里的 r 是 progress (0~1)，需要通过 _gamma 转为 mask_prob
    """
    B, T = x.shape
    device = x.device

    # 兼容处理 r
    if not isinstance(r, torch.Tensor):
        r = scalar_to_batch_tensor(r, B).to(device)
    r = r.to(device)

    # 计算 mask 概率 (注意：r=0 -> prob=1.0; r=1 -> prob=0.0)
    probs = _gamma(r) # [B] or [B, 1]
    if probs.ndim > 1: probs = probs.squeeze(-1)

    mask = torch.zeros_like(x, dtype=torch.long)

    for b in range(B):
        p = probs[b].item()
        total_masked_len = int(p * T)

        if total_masked_len < 1:
            continue

        # 如果需要Mask的比例很高 (>70%)，强制使用单一段落(1 span)，增加难度
        # 模拟 "整个前奏都丢了" 的情况
        if p > 0.7:
            curr_spans = 1
        else:
            curr_spans = random.randint(min_spans, max_spans)

        # 防止片段太多导致每个片段太短
        curr_spans = min(curr_spans, total_masked_len)

        # 简单的切分逻辑：将需要Mask的总长度分配给几个段
        base_len = total_masked_len // curr_spans
        rem = total_masked_len % curr_spans

        # 随机尝试放置 (简单的Monte Carlo放置，允许少量重叠以保持代码极简)
        # 也可以用更严格的无重叠逻辑，但在训练中随机性反而增加了鲁棒性
        for i in range(curr_spans):
            current_len = base_len + (1 if i < rem else 0)
            # 随机起点
            if T - current_len > 0:
                start = random.randint(0, T - current_len)
                mask[b, start : start + current_len] = 1
            else:
                mask[b, :] = 1 # 全盖住
    mask[:, 0] = 0
    return mask

def generate_mask_with_prob(shape, mask_prob, device):
    seq = shape[-1]
    rand = torch.randn(shape, device = device)
    rand[:, 0] = -torch.finfo(rand.dtype).max
    num_mask = min(int(seq * mask_prob), seq - 1)
    indices = rand.topk(num_mask, dim = -1).indices
    mask = ~torch.zeros(shape, device = device).scatter(1, indices, 1.).bool()
    return mask

def gumbel_noise_like(t):
    noise = torch.zeros_like(t).uniform_(1e-20, 1)
    return -torch.log(-torch.log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise_like(t)).argmax(dim=dim)

def mask_by_mink(num_to_mask, probs, temperature):
    noise = gumbel_noise_like(probs)
    confidence = torch.log(probs) + temperature * noise
    sorted_confidence, sorted_idx = confidence.sort(dim=-1)
    cut_off = torch.take_along_dim(sorted_confidence, num_to_mask, axis=-1)
    mask = confidence < cut_off
    return mask

def scalar_to_batch_tensor(x, batch_size):
    return torch.tensor(x).repeat(batch_size)

def _gamma(r):
    return (r * torch.pi / 2).cos().clamp(1e-10, 1.0)

def random_mask(
    x: torch.Tensor,
    r: torch.Tensor
):
    assert x.ndim == 2, "x must be (B, seq_len)"
    if not isinstance(r, torch.Tensor):
        r = scalar_to_batch_tensor(r, x.shape[0]).to(x.device)

    r = r.to(x.device)
    r = _gamma(r)[:, None]
    probs = torch.ones_like(x) * r

    mask = torch.bernoulli(probs)
    mask = mask.round().long()

    return mask


def _resolve_mask_ratios(
    r: torch.Tensor,
    batch_size: int,
    device: torch.device,
    min_mask_ratio: float = None,
    max_mask_ratio: float = None,
):
    if not isinstance(r, torch.Tensor):
        r = scalar_to_batch_tensor(r, batch_size).to(device)

    ratios = _gamma(r.to(device)).reshape(batch_size)
    if min_mask_ratio is not None or max_mask_ratio is not None:
        min_ratio = 0.0 if min_mask_ratio is None else float(min_mask_ratio)
        max_ratio = 1.0 if max_mask_ratio is None else float(max_mask_ratio)
        if min_ratio > max_ratio:
            raise ValueError(f"min_mask_ratio ({min_ratio}) must be <= max_mask_ratio ({max_ratio})")
        ratios = ratios.clamp(min=min_ratio, max=max_ratio)
    return ratios


def _candidate_indices(valid_mask_row: torch.Tensor, preserve_first_token: bool):
    candidate_idx = torch.nonzero(valid_mask_row, as_tuple=False).squeeze(-1)
    if preserve_first_token and candidate_idx.numel() > 0 and candidate_idx[0].item() == 0:
        candidate_idx = candidate_idx[1:]
    return candidate_idx


def _num_to_mask_from_ratio(mask_ratio: torch.Tensor, num_candidates: int):
    if num_candidates <= 0:
        return 0

    num_to_mask = int(torch.round(mask_ratio * num_candidates).item())
    if mask_ratio.item() > 0 and num_to_mask == 0:
        num_to_mask = 1
    return max(0, min(num_candidates, num_to_mask))


def _sample_positive_partition(total: int, parts: int, device: torch.device):
    if parts <= 0:
        raise ValueError(f"parts must be > 0, got {parts}")
    if total < parts:
        raise ValueError(f"total ({total}) must be >= parts ({parts})")
    if parts == 1:
        return torch.tensor([total], device=device, dtype=torch.long)

    cut_points = torch.randperm(total - 1, device=device)[: parts - 1] + 1
    cut_points, _ = cut_points.sort()
    boundaries = torch.cat([
        torch.zeros(1, device=device, dtype=torch.long),
        cut_points,
        torch.tensor([total], device=device, dtype=torch.long),
    ])
    return boundaries[1:] - boundaries[:-1]


def _sample_nonnegative_partition(total: int, parts: int, device: torch.device):
    if parts <= 0:
        raise ValueError(f"parts must be > 0, got {parts}")
    if total <= 0:
        return torch.zeros(parts, device=device, dtype=torch.long)

    draws = torch.multinomial(torch.ones(parts, device=device), total, replacement=True)
    return torch.bincount(draws, minlength=parts)


def exact_random_mask(
    x: torch.Tensor,
    r: torch.Tensor,
    valid_mask: torch.Tensor = None,
    min_mask_ratio: float = None,
    max_mask_ratio: float = None,
    preserve_first_token: bool = False,
):
    assert x.ndim == 2, "x must be (B, seq_len)"
    B, _ = x.shape
    device = x.device
    if valid_mask is None:
        valid_mask = torch.ones_like(x, dtype=torch.bool)
    else:
        valid_mask = valid_mask.to(device=device, dtype=torch.bool)

    ratios = _resolve_mask_ratios(r, B, device, min_mask_ratio, max_mask_ratio)
    mask = torch.zeros_like(x, dtype=torch.long)

    for b in range(B):
        candidate_idx = _candidate_indices(valid_mask[b], preserve_first_token)
        num_to_mask = _num_to_mask_from_ratio(ratios[b], candidate_idx.numel())
        if num_to_mask == 0:
            continue
        perm = torch.randperm(candidate_idx.numel(), device=device)[:num_to_mask]
        mask[b, candidate_idx[perm]] = 1

    return mask


def exact_span_mask(
    x: torch.Tensor,
    r: torch.Tensor,
    valid_mask: torch.Tensor = None,
    min_mask_ratio: float = None,
    max_mask_ratio: float = None,
    min_spans: int = 1,
    max_spans: int = 3,
    preserve_first_token: bool = False,
):
    assert x.ndim == 2, "x must be (B, seq_len)"
    B, _ = x.shape
    device = x.device
    if valid_mask is None:
        valid_mask = torch.ones_like(x, dtype=torch.bool)
    else:
        valid_mask = valid_mask.to(device=device, dtype=torch.bool)

    min_spans = max(1, int(min_spans))
    max_spans = max(min_spans, int(max_spans))

    ratios = _resolve_mask_ratios(r, B, device, min_mask_ratio, max_mask_ratio)
    mask = torch.zeros_like(x, dtype=torch.long)

    for b in range(B):
        candidate_idx = _candidate_indices(valid_mask[b], preserve_first_token)
        num_candidates = candidate_idx.numel()
        num_to_mask = _num_to_mask_from_ratio(ratios[b], num_candidates)
        if num_to_mask == 0:
            continue

        max_span_count = min(max_spans, num_to_mask)
        min_span_count = min(min_spans, max_span_count)
        if min_span_count == max_span_count:
            span_count = min_span_count
        else:
            span_count = int(torch.randint(min_span_count, max_span_count + 1, (1,), device=device).item())

        span_lengths = _sample_positive_partition(num_to_mask, span_count, device)
        gap_lengths = _sample_nonnegative_partition(num_candidates - num_to_mask, span_count + 1, device)

        compressed_mask = torch.zeros(num_candidates, device=device, dtype=torch.bool)
        cursor = 0
        for span_idx in range(span_count):
            cursor += int(gap_lengths[span_idx].item())
            span_len = int(span_lengths[span_idx].item())
            compressed_mask[cursor: cursor + span_len] = True
            cursor += span_len

        mask[b, candidate_idx[compressed_mask]] = 1

    return mask


def sample_mask(
    x: torch.Tensor,
    r: torch.Tensor,
    strategy: str = "bernoulli",
    valid_mask: torch.Tensor = None,
    min_mask_ratio: float = None,
    max_mask_ratio: float = None,
    span_prob: float = 0.5,
    min_spans: int = 1,
    max_spans: int = 3,
    preserve_first_token: bool = False,
):
    strategy = str(strategy).lower()
    if valid_mask is None:
        valid_mask = torch.ones_like(x, dtype=torch.bool)
    else:
        valid_mask = valid_mask.to(device=x.device, dtype=torch.bool)

    if strategy == "bernoulli":
        return random_mask(x, r) * valid_mask.long()

    if strategy in {"exact_random", "random_exact"}:
        return exact_random_mask(
            x,
            r,
            valid_mask=valid_mask,
            min_mask_ratio=min_mask_ratio,
            max_mask_ratio=max_mask_ratio,
            preserve_first_token=preserve_first_token,
        )

    if strategy in {"span", "exact_span"}:
        return exact_span_mask(
            x,
            r,
            valid_mask=valid_mask,
            min_mask_ratio=min_mask_ratio,
            max_mask_ratio=max_mask_ratio,
            min_spans=min_spans,
            max_spans=max_spans,
            preserve_first_token=preserve_first_token,
        )

    if strategy in {"mixed", "mixed_exact"}:
        B, _ = x.shape
        mask = torch.zeros_like(x, dtype=torch.long)
        span_prob = float(span_prob)
        if span_prob < 0.0 or span_prob > 1.0:
            raise ValueError(f"span_prob must be in [0, 1], got {span_prob}")

        use_span = torch.rand(B, device=x.device) < span_prob
        for b in range(B):
            row_kwargs = dict(
                valid_mask=valid_mask[b: b + 1],
                min_mask_ratio=min_mask_ratio,
                max_mask_ratio=max_mask_ratio,
                preserve_first_token=preserve_first_token,
            )
            if use_span[b]:
                row_mask = exact_span_mask(
                    x[b: b + 1],
                    r[b: b + 1] if isinstance(r, torch.Tensor) else r,
                    min_spans=min_spans,
                    max_spans=max_spans,
                    **row_kwargs,
                )
            else:
                row_mask = exact_random_mask(
                    x[b: b + 1],
                    r[b: b + 1] if isinstance(r, torch.Tensor) else r,
                    **row_kwargs,
                )
            mask[b: b + 1] = row_mask
        return mask

    raise ValueError(
        f"Unsupported mask strategy: {strategy}. "
        "Expected one of: bernoulli, exact_random, span, mixed."
    )

def random_mask_on_q(x, r, q):
    assert x.ndim == 3, "x (B, n_q, T)"
    B, n_q, T = x.shape

    q_mask = random_mask(x[torch.arange(B), q], r).bool()
    # print(q_mask)
    mask = torch.arange(0, n_q).reshape(1, -1, 1).repeat(B, 1, T)
    mask = (mask > q.reshape(-1, 1, 1).repeat(1, n_q, T))
    mask = mask.to(x.device)
    # print(mask)
    mask[torch.arange(B), q] = q_mask

    return mask

def apply_mask(
        x: torch.Tensor,
        mask: torch.Tensor,
        mask_token: int
    ):
    assert mask.ndim == 2, "mask must be (batch, seq_len), but got {mask.ndim}"
    assert mask.shape == x.shape, f"mask must be same shape as x, but got {mask.shape} and {x.shape}"
    assert mask.dtype == torch.long, "mask must be long dtype, but got {mask.dtype}"
    assert ~torch.any(mask > 1), "mask must be binary"
    assert ~torch.any(mask < 0), "mask must be binary"

    fill_x = torch.full_like(x, mask_token)
    x = x * (1 - mask) + fill_x * mask

    return x


if __name__ == '__main__':
    # a = generate_mask_with_prob((2, 1500), 0.15, 'cpu')
    # print(a, a.shape)
    rng = torch.quasirandom.SobolEngine(1, scramble=True, seed=8)
    def _gamma(r):
        return (r * torch.pi / 2).cos().clamp(1e-10, 1.0)

    # acc_ids = torch.rand(2, 1024)
    # r = rng.draw(acc_ids.shape[0])[:, 0].to(acc_ids.device)
    # print(r)
    # acc_random_mask = random_mask(acc_ids, r)

    acc_ids = torch.rand(5, 4, 2048)
    r = rng.draw(acc_ids.shape[0])[:, 0].to(acc_ids.device)
    print('r', r)
    q = torch.tensor([1, 2, 0, 3, 0])
    acc_random_mask = random_mask_on_q(acc_ids, r, q)

    print(acc_random_mask, acc_random_mask.sum(dim=-1))
