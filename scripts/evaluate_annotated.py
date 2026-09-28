"""
Evaluate the detectors on the 502 annotated test patches.

Reads what scripts/annotate_test_set.py saved in outputs/ (profiles, run
configuration with its parameters and thresholds): nothing is recomputed by
the diffusion model and no parameter is set here, except the tolerance sweep.

Steps:
1. Keep the annotated patches and their labels (dataset/annotations.json)
2. AUC and average precision (threshold free), then precision / recall / F1
   at the thresholds calibrated by annotate_test_set.py
3. Show how the AUC depends on the angular tolerance H_TOL (sensitivity
   analysis only: choosing H_TOL from it would tune the method on the labels)

Usage: run scripts/annotate_test_set.py first, then
    python scripts/evaluate_annotated.py
"""

import sys
import os
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "src"))

import numpy as np

from line_detection import DETECTORS, scores_from_profiles
from metrics import roc_auc, average_precision, binary_metrics

LABELS = ("horizontal", "other")
OUT_DIR = os.path.join(ROOT, "outputs")


if __name__ == "__main__":

    TOL_SWEEP = (5.0, 10.0, 15.0, 20.0, 30.0)

    with open(os.path.join(OUT_DIR, "run_config.json")) as f:
        cfg = json.load(f)
    data = np.load(os.path.join(OUT_DIR, "profiles.npz"))
    with open(os.path.join(ROOT, "dataset", "annotations.json")) as f:
        annotations = json.load(f)

    files = [str(f) for f in data["files"]]
    idx = np.array([i for i, f in enumerate(files) if f in annotations])
    Y = {
        lab: np.array([annotations[files[i]][lab] for i in idx], dtype=bool)
        for lab in LABELS
    }
    print(
        f"[INFO] {len(idx)} annotated patches "
        f"(horizontal: {Y['horizontal'].sum()}, other: {Y['other'].sum()})"
    )
    print(
        f"=== {cfg['model']}, t_start={cfg['t_start']}, K={cfg['ddim_steps']}, "
        f"seed={cfg['seed']}, H_TOL={cfg['h_tol']}, V_TOL={cfg['v_tol']}, "
        f"ALPHA={cfg['alpha']} ==="
    )
    print(
        f"{'detector':8s} {'label':11s} {'AUC':>5s} {'AP':>5s} | "
        f"{'prec':>5s} {'rec':>5s} {'F1':>5s} {'FP':>4s} {'FN':>4s}"
    )

    for name in DETECTORS:
        profiles = data[f"profile_{name}"][idx]
        scores = scores_from_profiles(profiles, h_tol=cfg["h_tol"], v_tol=cfg["v_tol"])
        pred = scores > np.array(cfg["thresholds"][name])
        f1s = []
        for j, lab in enumerate(LABELS):
            y, s = Y[lab], scores[:, j]
            m = binary_metrics(y, pred[:, j])
            f1s.append(m["f1"])
            print(
                f"{name:8s} {lab:11s} {roc_auc(y, s):5.3f} {average_precision(y, s):5.3f} | "
                f"{m['precision']:5.2f} {m['recall']:5.2f} {m['f1']:5.2f} "
                f"{m['fp']:4d} {m['fn']:4d}"
            )
        print(f"{name:8s} {'macro-F1':11s} {'':11s} | {'':17s}{np.mean(f1s):5.2f}")

    print("\n=== AUC vs angular tolerance H_TOL (columns) ===")
    print(f"{'detector':8s} {'label':11s} " + " ".join(f"{t:5.0f}" for t in TOL_SWEEP))
    for name in DETECTORS:
        profiles = data[f"profile_{name}"][idx]
        aucs = {lab: [] for lab in LABELS}
        for tol in TOL_SWEEP:
            scores = scores_from_profiles(profiles, h_tol=tol, v_tol=cfg["v_tol"])
            for j, lab in enumerate(LABELS):
                aucs[lab].append(roc_auc(Y[lab], scores[:, j]))
        for lab in LABELS:
            print(f"{name:8s} {lab:11s} " + " ".join(f"{a:5.3f}" for a in aucs[lab]))
