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
    Compute the 2D anomaly map (B, 1, H, W) according to the chosen metric
    """
    diff = x0 - x0_hat
    abs_diff = torch.abs(diff)
    
    if metric == "pdf_relative":
        # Metric from the paper: "Anomaly Detection with Diffusion Models"
        return abs_diff / (1.0 + torch.abs(x0_hat))
    
    elif metric == "absolute":
            return abs_diff
    
    elif metric == "squared":
        return diff.pow(2)
    
    elif metric == "signed_positive":
        return torch.clamp(diff, min=0.0)
    
    elif metric == "z_score_local":
        # Compute local positive mean to ponderate the residuals
        diff = x0 - x0_hat
        pos_diff = torch.clamp(diff, min=0.0)
        local_density = F.avg_pool2d(pos_diff, kernel_size=3, stride=1, padding=1)
        
        return pos_diff * local_density
    
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