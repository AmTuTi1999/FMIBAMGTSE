"""Experiment Set 1 — Running example, C2 (marginal/perturbation methods).

E1.1: Score on X1 under f1 (should be 0; expect >0 spuriously) and f2 (>0).

Usage:
    python experiments/exp1_running_example.py [--config configs/default.yaml]
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import torch

from dagfaith.config import load_config, seed_everything, results_dir
from data.running_example import (
    sample_c2_data, make_f1, make_f2, make_f3, adversarial_gamma, torch_model_from_numpy_fn
)
from methods.shap_methods import KernelSHAPMethod, TimeSHAPMethod, ShapTimeMethod
from methods.lime_methods import TSMULEMethod
from methods.perturbation_methods import WinITMethod, DynamaskMethod, FITMethod



def run_e1_1(cfg: dict, rng: np.random.Generator, out_dir: str) -> pd.DataFrame:
    """E1.1: Compute each method's score on X1 under f1 and f2."""
    print("\n=== E1.1: Running example, C2 ===")
    re_cfg = cfg["running_example"]
    beta  = re_cfg["beta"]
    delta = re_cfg["delta"]
    n     = re_cfg["n_samples"]

    X = sample_c2_data(n=n, delta=delta, T=1, rng=rng)  # [n, 2, 1]
    X_t = torch.tensor(X, dtype=torch.float32)

    f1_np = make_f1(beta=beta)
    f2_np = make_f2(beta=beta, gamma=0.5)
    f3_np = make_f3(beta=beta, gamma=0.5, delta=delta)  # for reference, not used in this experiment
    f1 = torch_model_from_numpy_fn(f1_np)
    f2 = torch_model_from_numpy_fn(f2_np)
    f3 = torch_model_from_numpy_fn(f3_np)

    methods = [
        ("KernelSHAP(marginal)", KernelSHAPMethod(n_background=50, variant="marginal")),
        ("TimeSHAP",             TimeSHAPMethod(n_background=30)),
        ("ShapTime",             ShapTimeMethod(n_background=30)),
        ("TS-MULE(uniform)",     TSMULEMethod(variant="uniform", n_segments=1, n_neighbors=100)),
        ("WinIT",                WinITMethod(window_size=1, num_samples=20, metric="pd")),
        ("Dynamask",             DynamaskMethod(num_epoch=150, area_list=list(np.arange(0.25, 0.35, 0.01)))),
        ("FIT",                  FITMethod(num_samples=20)),
    ]

    rows = []
    for method_name, method in methods:
        print(f"  Running {method_name} …", flush=True)
        scores_f1 = method.attribute(f1, X_t).detach().cpu().numpy()
        scores_f2 = method.attribute(f2, X_t).detach().cpu().numpy()
        scores_f3 = method.attribute(f3, X_t).detach().cpu().numpy()  # for reference

        # Score on X1 (variable index 0), averaged over batch and time
        score_x1_f1 = float(np.abs(scores_f1[:, 0, :]).mean())
        score_x1_f2 = float(np.abs(scores_f2[:, 0, :]).mean())
        score_x1_f3 = float(np.abs(scores_f3[:, 0, :]).mean())  # for reference

        correct_f1 = score_x1_f1 < 0.0000001  # should be ~0 under f1 (x1 not causal)
        correct_f2 = score_x1_f2 > 0.0000001  # should be >0 under f2 (x1 is causal)
        correct_f3 = score_x1_f3 < 0.0000001  # should be ~0 under f3 (x1 not causal on-manifold)
        print(f"  {method_name:<28} | score(X1|f1)={score_x1_f1:.4f} {'✓' if correct_f1 else '✗'}"
              f"  score(X1|f2)={score_x1_f2:.4f} {'✓' if correct_f2 else '✗'}"
              f"  score(X1|f3)={score_x1_f3:.4f} {'✓' if correct_f3 else '✗'}"
              f"  {'C2 VIOLATED' if not correct_f1 else 'C2 OK'}"
              f"  {'C1 VIOLATED' if not correct_f3 else 'C1 OK'}")
        rows.append({
            "method":       method_name,
            "score_X1_f1":  score_x1_f1,
            "score_X1_f2":  score_x1_f2,
            "correct_f1":   correct_f1,
            "correct_f2":   correct_f2,
            "correct_f3":   correct_f3,
            "C2_violation": not correct_f1,
            "C1_violation": not correct_f3,
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "e1_1_running_example.csv"), index=False)
    print(f"  Saved → {out_dir}/e1_1_running_example.csv")
    return df



def main():
    """Entry point: load config, seed, and run all Experiment Set 1 sub-experiments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    out_dir = str(results_dir(cfg))

    run_e1_1(cfg, rng, out_dir)

    print("\nExperiment Set 1 complete.")


if __name__ == "__main__":
    main()
