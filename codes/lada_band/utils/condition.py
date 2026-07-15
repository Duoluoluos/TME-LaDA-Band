import torch
import torch.nn as nn
import typing as tp
from copy import deepcopy


class ClassifierFreeGuidanceDropout(nn.Module):
    """Classifier Free Guidance dropout.
    All attributes are dropped with the same probability.

    Args:
        p (float): Probability to apply condition dropout during training.
        seed (int): Random seed.
    """
    def __init__(self, p: float, seed=8):
        super().__init__()
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)
        self.p = p

    def forward(self, samples: tp.List[torch.Tensor]):
        """
        Args:
            samples (list[ConditioningAttributes]): List of conditions.
        Returns:
            list[ConditioningAttributes]: List of conditions after all attributes were set to None.
        """
        if not self.training:
            return samples

        B = samples.shape[0]

        probs = torch.zeros(B, 1, 1).float().uniform_(0, 1, generator=self.rng)
        probs = probs.to(samples)
        samples = (probs >= self.p) * samples

        samples = deepcopy(samples)
        return samples

    def __repr__(self):
        return f"ClassifierFreeGuidanceDropout(p={self.p})"