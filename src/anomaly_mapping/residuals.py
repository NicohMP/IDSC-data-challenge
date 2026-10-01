"""
All residual mapping methods taking as input x0 and x0_hat
"""

from collections.abc import Callable
import torch
import torch.nn.functional as F

# x0          : (B, 1, H, W)
# x0_hat      : (B, 1, H, W)
# anomaly_map : (B, 1, H, W)
AnomalyMapper = Callable[
    [torch.Tensor, torch.Tensor],
    torch.Tensor,
]


def compute_residual_map(x0: torch.Tensor, x0_hat: torch.Tensor, metric: str = "pdf_relative") -> torch.Tensor:
    """
    Calcule la carte d'anomalie 2D (B, 1, H, W) selon la métrique choisie.
    """
    diff = x0 - x0_hat
    abs_diff = torch.abs(diff)
    
    if metric == "pdf_relative":
        # Metric from the paper: "Anomaly Detection with Diffusion Models"
        return abs_diff / (1.0 + torch.abs(x0_hat))
    
    elif metric == "squared":
        return diff.pow(2)
    
    elif metric == "absolute":
        return abs_diff
    
    elif metric == "signed_positive":
        # Keep only anomaly pixels that are brighter than the background
        # Ne garde que les pixels d'anomalie plus lumineux que le fond
        return torch.clamp(diff, min=0.0)
    
    elif metric == "z_score_local":
        # Pondération de l'erreur par la variance locale (fenêtre 3x3)
        std_local = F.avg_pool2d(x0_hat.pow(2), kernel_size=3, stride=1, padding=1) - \
                    F.avg_pool2d(x0_hat, kernel_size=3, stride=1, padding=1).pow(2)
        return abs_diff / (torch.sqrt(torch.abs(std_local)) + 1e-5)
    
    else:
        raise ValueError(f"residual metric unknown : {metric}")


def evaluate_residual_contrast(residual_map: torch.Tensor) -> torch.Tensor:
    """
    Evaluate the contrast of the residual map for each image in the batch
    use the difference (99th percentile - median) to isolate strong signal from background
    Args:
        residual_map: Tensor format (B, 1, H, W)
    Returns:
        Tensor format (B,) with the contrast score for each image in the batch
    """
    B = residual_map.shape[0]
    # Aplatir les dimensions spatiales pour obtenir (B, H*W)
    res_flat = residual_map.view(B, -1)
    
    # Calcul par élément du batch
    p99 = torch.quantile(res_flat, 0.99, dim=1)
    median = torch.median(res_flat, dim=1).values
    
    return p99 - median