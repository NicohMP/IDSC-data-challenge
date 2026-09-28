"""
Annotate ALL test patches with the line detectors (Radon, Hough, Gabor).

Steps:
1. Compute the residual maps of the whole test set (cached in outputs/)
2. Compute the orientation profile of each patch with each detector
3. Turn the profiles into horizontal / other scores (angular tolerances)
4. Calibrate one threshold per detector and label WITHOUT using the labels:
   the (1 - ALPHA) quantile of the scores of the unannotated test patches,
   which are mostly background (ALPHA is the tolerated false-alarm rate)
5. Save in outputs/: the predictions (format of annotations.json), the
   profiles, and the run configuration (parameters and thresholds) that
   scripts/evaluate_annotated.py reads back

Usage: set the parameters in the "Parameters" section below, then run
    python scripts/annotate_test_set.py
"""

import sys
import os
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "src"))

import numpy as np
import torch

from inference import load_diffusion_model
from line_detection import DETECTORS, scores_from_profiles
from purification import residual_maps

LABELS = ("horizontal", "other")
OUT_DIR = os.path.join(ROOT, "outputs")


def load_test_set(test_dir: str) -> tuple[list[str], np.ndarray]:
    files = sorted(f for f in os.listdir(test_dir) if f.endswith(".npy"))
    patches = np.stack(
        [np.load(os.path.join(test_dir, f)).astype(np.float32).squeeze() for f in files]
    )
    return files, patches


def get_residuals(patches, model_name, t_start, ddim_steps, seed, device) -> np.ndarray:
    cache = os.path.join(
        OUT_DIR, f"residuals_{model_name}_t{t_start}_k{ddim_steps}_s{seed}.npy"
    )
    if os.path.exists(cache):
        print(f"[INFO] Residuals loaded from {cache}")
        return np.load(cache)
    model, schedule = load_diffusion_model(
        os.path.join(ROOT, "models", f"{model_name}.pt"), device
    )
    print(f"[INFO] Computing residuals of {len(patches)} patches with {model_name}...")
    res = residual_maps(patches, model, schedule, t_start, ddim_steps, seed, device)
    np.save(cache, res)
    return res


if __name__ == "__main__":

    # Purification (Algorithm 1)
    MODEL = "unet32"
    T_START = 600
    DDIM_STEPS = 6
    SEED = 0
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Angular tolerance (degrees): a line is horizontal if its angle is within
    # H_TOL of 0; vertical lines (90 +/- V_TOL) are ignored.
    H_TOL = 10.0
    V_TOL = 10.0

    # Tolerated false-alarm rate on the reference (unannotated) patches
    ALPHA = 0.10

    os.makedirs(OUT_DIR, exist_ok=True)
    files, patches = load_test_set(os.path.join(ROOT, "dataset", "test"))
    with open(os.path.join(ROOT, "dataset", "annotations.json")) as f:
        annotations = json.load(f)
    is_annotated = np.array([f in annotations for f in files])
    print(f"[INFO] {len(files)} test patches, {is_annotated.sum()} annotated")

    residuals = get_residuals(patches, MODEL, T_START, DDIM_STEPS, SEED, DEVICE)
    profiles = {
        name: np.stack([det(e) for e in residuals]) for name, det in DETECTORS.items()
    }

    thresholds = {}
    for name in DETECTORS:
        scores = scores_from_profiles(profiles[name], h_tol=H_TOL, v_tol=V_TOL)
        thr = np.quantile(scores[~is_annotated], 1 - ALPHA, axis=0)
        thresholds[name] = thr.tolist()
        pred = scores > thr
        predictions = {
            f: {lab: bool(pred[i, j]) for j, lab in enumerate(LABELS)}
            for i, f in enumerate(files)
        }
        with open(os.path.join(OUT_DIR, f"predictions_{name}.json"), "w") as fh:
            json.dump(predictions, fh, indent=1)
        print(
            f"[INFO] {name}: flagged {pred[:, 0].mean():.1%} horizontal, "
            f"{pred[:, 1].mean():.1%} other over the {len(files)} patches"
        )

    np.savez(
        os.path.join(OUT_DIR, "profiles.npz"),
        files=np.array(files),
        **{f"profile_{name}": p for name, p in profiles.items()},
    )
    config = dict(
        model=MODEL, t_start=T_START, ddim_steps=DDIM_STEPS, seed=SEED,
        h_tol=H_TOL, v_tol=V_TOL, alpha=ALPHA, thresholds=thresholds,
    )
    with open(os.path.join(OUT_DIR, "run_config.json"), "w") as fh:
        json.dump(config, fh, indent=1)
    print(f"[INFO] Saved predictions, profiles and run_config.json in {OUT_DIR}")
