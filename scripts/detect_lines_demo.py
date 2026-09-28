"""
Run the Radon, Hough and Gabor line detectors on the residual map of patches.

Steps:
1. Purify each patch with the diffusion model and compute the residual (Eq. 10)
2. Compute the orientation profile of the residual with each detector
3. Print the horizontal / other scores and plot residual + profiles

Usage: set the parameters in the "Parameters" section below, then run
    python scripts/detect_lines_demo.py
"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "src"))

import numpy as np
import torch
import matplotlib.pyplot as plt

from inference import load_diffusion_model
from line_detection import ANGLES, DETECTORS, scores_from_profile
from purification import residual_maps


if __name__ == "__main__":

    MODEL = "models/unet32.pt"
    TEST_DIR = "dataset/test"
    IMAGES = {
        "horizontal": "patch_9107_22.npy",
        "other": "patch_9150_13.npy",
        "normal": "patch_9099_13.npy",
    }
    T_START = 600
    DDIM_STEPS = 6
    SEED = 0
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model, schedule = load_diffusion_model(os.path.join(ROOT, MODEL), DEVICE)

    fig, axes = plt.subplots(
        len(IMAGES), 2, figsize=(11, 3.2 * len(IMAGES)), squeeze=False
    )
    for row, (name, file) in enumerate(IMAGES.items()):
        patch = np.load(os.path.join(ROOT, TEST_DIR, file)).astype(np.float32).squeeze()
        e = residual_maps(
            patch[None], model, schedule, T_START, DDIM_STEPS, SEED, DEVICE
        )[0]

        axes[row, 0].imshow(e, origin="lower", cmap="magma")
        axes[row, 0].set_title(f"{name} ({file}): residual")
        axes[row, 0].axis("off")

        print(f"\n{name} ({file})")
        for det_name, detector in DETECTORS.items():
            prof = detector(e)
            s_h, s_o = scores_from_profile(prof)
            print(f"  {det_name:6s} horizontal={s_h:.3f}  other={s_o:.3f}")
            axes[row, 1].plot(ANGLES, prof / prof.max(), label=det_name)
        axes[row, 1].set_xlabel("theta (deg)")
        axes[row, 1].set_title("orientation profile (each normalized by its max)")
        axes[row, 1].legend()

    plt.tight_layout()
    plt.show()
