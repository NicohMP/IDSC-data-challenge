import torch


def normalize_patch(x0: torch.Tensor) -> torch.Tensor:
    return (x0 - x0.mean()) / (x0.std(correction=0) + 1e-8)
