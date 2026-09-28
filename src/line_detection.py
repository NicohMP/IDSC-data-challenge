"""
Line detectors working on an anomaly map e (H, W): Radon, Hough and Gabor.

Convention: the first axis of e is frequency (rows), the second is time
(columns). The orientation theta (degrees, in [0, 180)) is the angle of the
line with the time axis: 0 = horizontal, 90 = vertical.

Each detector returns a profile R(theta): one evidence value per angle.
`scores_from_profile` turns a profile into two scores, one for horizontal
lines and one for other (oblique) lines, vertical lines being ignored.
No threshold is chosen here: calibrate it separately (e.g. on normal patches).
"""

import numpy as np
import torch
import torch.nn.functional as F

ANGLES = np.arange(0, 180, 5.0)


def _line_coordinates(shape: tuple[int, int], theta_deg: float) -> np.ndarray:
    """
    Signed distance rho of every pixel to the line through the patch center
    with orientation theta. Pixels on a common line share the same rho.
    """
    H, W = shape
    r, c = np.mgrid[0:H, 0:W].astype(np.float64)
    r -= (H - 1) / 2
    c -= (W - 1) / 2
    th = np.deg2rad(theta_deg)
    return -c * np.sin(th) + r * np.cos(th)


def _accumulate(
    e: np.ndarray, theta_deg: float, min_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sum and pixel count of e along all parallel lines of orientation theta,
    keeping only the lines that contain at least `min_len` pixels.
    """
    rho = _line_coordinates(e.shape, theta_deg)
    idx = np.round(rho - rho.min()).astype(int).ravel()
    n = idx.max() + 1
    total = np.bincount(idx, weights=e.ravel(), minlength=n)
    count = np.bincount(idx, minlength=n)
    keep = count >= min_len
    return total[keep], count[keep]


def radon_profile(
    e: np.ndarray, angles: np.ndarray = ANGLES, min_len: int = 24
) -> np.ndarray:
    """
    Radon transform: mean of e along each line, then max over the lines.
    Using the mean (not the sum) makes lines of different lengths comparable;
    `min_len` discards short corner lines whose mean is too noisy.
    """
    prof = []
    for th in angles:
        total, count = _accumulate(e, th, min_len)
        prof.append((total / count).max())
    return np.array(prof)


def hough_profile(
    e: np.ndarray,
    angles: np.ndarray = ANGLES,
    quantile: float = 0.9,
    min_len: int = 24,
) -> np.ndarray:
    """
    Hough transform: keep the pixels above a quantile of e (binary map), let
    each vote for the lines through it, and return the max vote fraction over
    the lines (fraction of the line's pixels that are 'on').
    """
    binary = (e >= np.quantile(e, quantile)).astype(np.float64)
    prof = []
    for th in angles:
        votes, count = _accumulate(binary, th, min_len)
        prof.append((votes / count).max())
    return np.array(prof)


def gabor_kernel(
    theta_deg: float, wavelength: float = 6.0, sigma_along: float = 6.0,
    sigma_across: float = 2.0, size: int = 25,
) -> np.ndarray:
    """
    Even Gabor kernel responding to a bright line of orientation theta:
    a cosine across the line, under a Gaussian elongated along the line.
    The kernel is made zero-mean so that flat areas give no response.
    """
    half = size // 2
    r, c = np.mgrid[-half : half + 1, -half : half + 1].astype(np.float64)
    th = np.deg2rad(theta_deg)
    u = c * np.cos(th) + r * np.sin(th)
    v = -c * np.sin(th) + r * np.cos(th)
    g = np.exp(-(u**2 / (2 * sigma_along**2) + v**2 / (2 * sigma_across**2)))
    g = g * np.cos(2 * np.pi * v / wavelength)
    g -= g.mean()
    return g / np.abs(g).sum()


def gabor_profile(
    e: np.ndarray, angles: np.ndarray = ANGLES, **kernel_kwargs
) -> np.ndarray:
    """
    Gabor filter bank: for each orientation, max of the (positive) response
    over the patch. Reflect padding avoids fake edges at the patch border.
    """
    kernels = np.stack([gabor_kernel(th, **kernel_kwargs) for th in angles])
    k = torch.from_numpy(kernels[:, None]).float()
    pad = k.shape[-1] // 2
    x = torch.from_numpy(e[None, None].astype(np.float32))
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    resp = F.conv2d(x, k)[0]
    return resp.amax(dim=(1, 2)).numpy()


DETECTORS = {"radon": radon_profile, "hough": hough_profile, "gabor": gabor_profile}


def scores_from_profile(
    profile: np.ndarray,
    angles: np.ndarray = ANGLES,
    h_tol: float = 10.0,
    v_tol: float = 10.0,
) -> tuple[float, float]:
    """
    Score for horizontal lines (theta within h_tol of 0 or 180) and for other
    lines (all remaining angles except the vertical band 90 +/- v_tol).
    """
    dist_h = np.minimum(angles, 180 - angles)
    horizontal = dist_h <= h_tol
    vertical = np.abs(angles - 90) <= v_tol
    other = ~horizontal & ~vertical
    return float(profile[horizontal].max()), float(profile[other].max())


def scores_from_profiles(
    profiles: np.ndarray,
    angles: np.ndarray = ANGLES,
    h_tol: float = 10.0,
    v_tol: float = 10.0,
) -> np.ndarray:
    """
    (N, 2) array of the horizontal and other scores of N profiles.
    """
    return np.array([scores_from_profile(p, angles, h_tol, v_tol) for p in profiles])
