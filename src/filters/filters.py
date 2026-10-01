"""Orientation filters converting anomaly maps into angle-response profiles.

The public filters in this module implement the pipeline contract

    (B, 1, H, W) torch.Tensor -> (B, N_angles) torch.Tensor.

Radon and Hough reuse NumPy implementations internally and transfer their
small response profiles back to the input device. Gabor is evaluated directly
with batched PyTorch convolutions and therefore stays on the input device.
"""

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F


ANGLES = np.arange(0.0, 180.0, 5.0)

# Batch anomaly maps (B, 1, H, W) -> angle responses (B, N_angles).
Filter = Callable[[torch.Tensor], torch.Tensor]


def _validate_maps(maps: torch.Tensor) -> None:
    """Validate the tensor accepted by every public filter."""
    if not isinstance(maps, torch.Tensor):
        raise TypeError("maps must be a torch.Tensor")
    if maps.ndim != 4 or maps.shape[1] != 1:
        raise ValueError(
            "maps must have shape (B, 1, H, W), "
            f"got {tuple(maps.shape)}"
        )
    if not maps.is_floating_point():
        raise TypeError("maps must use a floating-point dtype")


def _line_coordinates(shape: tuple[int, int], theta_deg: float) -> np.ndarray:
    """Return each pixel's signed distance to a centered oriented line."""
    height, width = shape
    rows, columns = np.mgrid[0:height, 0:width].astype(np.float64)
    rows -= (height - 1) / 2
    columns -= (width - 1) / 2
    theta = np.deg2rad(theta_deg)
    return -columns * np.sin(theta) + rows * np.cos(theta)


def _accumulate(
    anomaly_map: np.ndarray,
    theta_deg: float,
    min_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum values along parallel lines and return their pixel counts."""
    rho = _line_coordinates(anomaly_map.shape, theta_deg)
    indices = np.round(rho - rho.min()).astype(int).ravel()
    line_count = indices.max() + 1
    totals = np.bincount(
        indices,
        weights=anomaly_map.ravel(),
        minlength=line_count,
    )
    counts = np.bincount(indices, minlength=line_count)
    keep = counts >= min_len

    if not np.any(keep):
        raise ValueError(
            f"min_len={min_len} removes every line for map shape "
            f"{anomaly_map.shape}"
        )

    return totals[keep], counts[keep]


def radon_profile(
    anomaly_map: np.ndarray,
    angles: np.ndarray = ANGLES,
    min_len: int = 24,
) -> np.ndarray:
    """Compute one Radon-style maximum line response per angle."""
    profile = []
    for theta in angles:
        totals, counts = _accumulate(anomaly_map, theta, min_len)
        profile.append((totals / counts).max())
    return np.asarray(profile)


def hough_profile(
    anomaly_map: np.ndarray,
    angles: np.ndarray = ANGLES,
    quantile: float = 0.9,
    min_len: int = 24,
) -> np.ndarray:
    """Compute the maximum fraction of active pixels per angle."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")

    binary_map = (
        anomaly_map >= np.quantile(anomaly_map, quantile)
    ).astype(np.float64)
    profile = []
    for theta in angles:
        votes, counts = _accumulate(binary_map, theta, min_len)
        profile.append((votes / counts).max())
    return np.asarray(profile)


def gabor_kernel(
    theta_deg: float,
    wavelength: float = 6.0,
    sigma_along: float = 6.0,
    sigma_across: float = 2.0,
    size: int = 25,
) -> np.ndarray:
    """Build a zero-mean Gabor kernel responding to a bright oriented line."""
    if size <= 0 or size % 2 == 0:
        raise ValueError("size must be a positive odd integer")

    half_size = size // 2
    rows, columns = np.mgrid[
        -half_size : half_size + 1,
        -half_size : half_size + 1,
    ].astype(np.float64)
    theta = np.deg2rad(theta_deg)
    along = columns * np.cos(theta) + rows * np.sin(theta)
    across = -columns * np.sin(theta) + rows * np.cos(theta)
    kernel = np.exp(
        -(
            along**2 / (2 * sigma_along**2)
            + across**2 / (2 * sigma_across**2)
        )
    )
    kernel *= np.cos(2 * np.pi * across / wavelength)
    kernel -= kernel.mean()
    return kernel / np.abs(kernel).sum()


def _numpy_profile_filter(
    maps: torch.Tensor,
    profile_function: Callable[[np.ndarray], np.ndarray],
) -> torch.Tensor:
    """Apply a NumPy single-map detector to a complete Torch batch."""
    _validate_maps(maps)
    maps_numpy = maps.detach().cpu().numpy()[:, 0]
    profiles = np.stack([profile_function(item) for item in maps_numpy])
    return torch.as_tensor(
        profiles,
        dtype=maps.dtype,
        device=maps.device,
    )


def radon_filter(
    maps: torch.Tensor,
    angles: np.ndarray = ANGLES,
    min_len: int = 24,
) -> torch.Tensor:
    """Return batched Radon profiles with shape ``(B, N_angles)``."""
    return _numpy_profile_filter(
        maps,
        lambda item: radon_profile(item, angles=angles, min_len=min_len),
    )


def hough_filter(
    maps: torch.Tensor,
    angles: np.ndarray = ANGLES,
    quantile: float = 0.9,
    min_len: int = 24,
) -> torch.Tensor:
    """Return batched Hough profiles with shape ``(B, N_angles)``."""
    return _numpy_profile_filter(
        maps,
        lambda item: hough_profile(
            item,
            angles=angles,
            quantile=quantile,
            min_len=min_len,
        ),
    )


def gabor_filter(
    maps: torch.Tensor,
    angles: np.ndarray = ANGLES,
    wavelength: float = 6.0,
    sigma_along: float = 6.0,
    sigma_across: float = 2.0,
    size: int = 25,
) -> torch.Tensor:
    """Return batched Gabor profiles with shape ``(B, N_angles)``."""
    _validate_maps(maps)
    padding = size // 2
    if padding >= maps.shape[-2] or padding >= maps.shape[-1]:
        raise ValueError(
            "Gabor kernel is too large for reflect padding: "
            f"kernel size {size}, map shape {tuple(maps.shape[-2:])}"
        )

    kernels_numpy = np.stack(
        [
            gabor_kernel(
                theta,
                wavelength=wavelength,
                sigma_along=sigma_along,
                sigma_across=sigma_across,
                size=size,
            )
            for theta in angles
        ]
    )
    kernels = torch.as_tensor(
        kernels_numpy[:, None],
        dtype=maps.dtype,
        device=maps.device,
    )
    padded_maps = F.pad(
        maps,
        (padding, padding, padding, padding),
        mode="reflect",
    )
    responses = F.conv2d(padded_maps, kernels)
    return responses.amax(dim=(2, 3))


FILTERS: dict[str, Filter] = {
    "radon": radon_filter,
    "hough": hough_filter,
    "gabor": gabor_filter,
}
