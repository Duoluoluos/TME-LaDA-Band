import numpy as np
from scipy.special import softmax
import torch.nn.functional as F
import torch


def top_p_sampling(logits, top_p=0.9, filter_value=-float('Inf'), min_tokens_to_keep=1):
    """
    针对 Batch/Sequence 维度的鲁棒 Top-p (Nucleus) Sampling 实现。
    支持输入形状: [Vocab], [Batch, Vocab], [Seq, Vocab], [Batch, Seq, Vocab]
    """
    # 1. 兼容性处理：如果是 NumPy 数组，转为 Tensor
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)

    # 记录原始设备，确保最后返回一致
    original_device = logits.device

    # 2. 确保在最后一个维度（Vocab）操作，不管前面有多少个维度（Batch/Seq）
    # sorted_logits: 排序后的概率值
    # sorted_indices: 排序后的原索引 ID
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

    # 3. 计算累积概率
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # 4. 生成移除掩码 (Remove Mask)
    # 我们需要移除那些累积概率超过 top_p 的 token
    sorted_indices_to_remove = cumulative_probs > top_p

    # 5. 掩码平移 (Shift)
    # 逻辑：如果第 k 个 token 让累积概率超过了 top_p，我们需要保留这第 k 个，
    # 也就是把掩码向右平移一位。
    # [..., 1:] 表示最后维度的切片，兼容任意形状
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0 # 永远保留概率最大的那个 token

    # (可选) 强制保留前 min_tokens_to_keep 个 token，防止分布过于平坦时全部被切掉
    if min_tokens_to_keep > 1:
        sorted_indices_to_remove[..., :min_tokens_to_keep] = 0

    # 6. 将掩码映射回原始 logits 的位置 (Scatter)
    # sorted_indices_to_remove 是对应 sorted_logits 的，需要映射回原始 logits 的顺序
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1,
        index=sorted_indices,
        src=sorted_indices_to_remove
    )

    # 7. 将被移除的 token 的 logit 设为负无穷
    logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits

def top_k_sampling(logits, top_k):
    assert logits.dim() == 2
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits[logits < v[:, -1:]] = -float("Inf")
    return logits

def temperature_sampling(logits, temp):
    probs = F.softmax(logits / temp, dim=-1)
    return probs
