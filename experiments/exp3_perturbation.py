"""Experiment Set 3 — WinIT, Dynamask, FIT (§2.1 temporal perturbation family).

E3.1: Score on X1 under f1 (should be ~0) and f2 (>0).
      At T=1 these methods do NOT suffer the C2 marginal-SHAP failure because
      they replace only the perturbed feature, leaving correlated features intact.
      Dynamask may give non-zero scores because its sparsity objective can
      "absorb" variance from correlated features.
E3.2: Adversarial collapse sweep — same γ grid as E1.2.
E3.3: VAR benchmark — here temporal correlation induces C2-style violations:
      removing x_{d,t} changes the realistic future trajectory, so non-causal
      features accumulate spurious window/mask scores.
      FIT's KL comparison partially controls for this; check whether its SHD
      improves over WinIT/Dynamask.

Usage:
    python experiments/exp3_perturbation.py [--config configs/default.yaml]
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
    sample_c2_data, make_f1, make_f2, adversarial_gamma, torch_model_from_numpy_fn,
)
from methods.perturbation_methods import WinITMethod, DynamaskMethod, FITMethod


def _methods(fast: bool = False):
    """Return the three methods; use reduced settings when fast=True."""
    if fast:
        return [
            ("WinIT",    WinITMethod(window_sizes=[1, 2], n_samples=10)),
            ("Dynamask", DynamaskMethod(n_epochs=50, lr=0.05, lambda_area=0.01)),
            ("FIT",      FITMethod(n_samples=10)),
        ]
    return [
        ("WinIT",    WinITMethod(window_sizes=[1, 2, 4], n_samples=20)),
        ("Dynamask", DynamaskMethod(n_epochs=150, lr=0.05, lambda_area=0.01)),
        ("FIT",      FITMethod(n_samples=30)),
    ]


def run_e3_1(cfg: dict, rng: np.random.Generator, out_dir: str) -> pd.DataFrame:
    """E3.1: Running-example C2 check for the temporal perturbation family."""
    print("\n=== E3.1: Running example, C2 (WinIT / Dynamask / FIT) ===")
    re_cfg = cfg["running_example"]
    beta  = re_cfg["beta"]
    delta = re_cfg["delta"]
    n     = re_cfg["n_samples"]

    X = sample_c2_data(n=n, delta=delta, T=1, rng=rng)
    X_t = torch.tensor(X, dtype=torch.float32)

    f1 = torch_model_from_numpy_fn(make_f1(beta=beta))
    f2 = torch_model_from_numpy_fn(make_f2(beta=beta, gamma=0.5))

    rows = []
    for method_name, method in _methods():
        print(f"  Running {method_name} …", flush=True)
        scores_f1 = method.attribute(f1, X_t).detach().cpu().numpy()
        scores_f2 = method.attribute(f2, X_t).detach().cpu().numpy()

        score_x1_f1 = float(np.abs(scores_f1[:, 0, :]).mean())
        score_x1_f2 = float(np.abs(scores_f2[:, 0, :]).mean())
        correct_f1  = score_x1_f1 < 0.01
        correct_f2  = score_x1_f2 > 0.01

        print(f"  {method_name:<10} | score(X1|f1)={score_x1_f1:.4f} {'✓' if correct_f1 else '✗'}"
              f"  score(X1|f2)={score_x1_f2:.4f} {'✓' if correct_f2 else '✗'}")
        rows.append({
            "method":       method_name,
            "score_X1_f1":  score_x1_f1,
            "score_X1_f2":  score_x1_f2,
            "correct_f1":   correct_f1,
            "correct_f2":   correct_f2,
            "C2_violation": not correct_f1,
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "e3_1_running_example.csv"), index=False)
    print(f"  Saved → {out_dir}/e3_1_running_example.csv")
    return df


def run_e3_2(cfg: dict, rng: np.random.Generator, out_dir: str) -> pd.DataFrame:
    """E3.2: Adversarial collapse sweep for the temporal perturbation family."""
    print("\n=== E3.2: Adversarial collapse sweep (WinIT / Dynamask / FIT) ===")
    re_cfg = cfg["running_example"]
    beta  = re_cfg["beta"]
    delta = re_cfg["delta"]
    n     = re_cfg["n_samples"]
    gamma_sweep = re_cfg.get("gamma_sweep", [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0])
    gamma_star  = adversarial_gamma(beta, delta)
    print(f"  Adversarial γ* = -2βδ = {gamma_star:.3f}")

    X    = sample_c2_data(n=n, delta=delta, T=1, rng=rng)
    X_t  = torch.tensor(X, dtype=torch.float32)

    rows = []
    for method_name, method in _methods():
        print(f"  {method_name} …", flush=True)
        for gamma in gamma_sweep:
            f1 = torch_model_from_numpy_fn(make_f1(beta=beta))
            f2 = torch_model_from_numpy_fn(make_f2(beta=beta, gamma=gamma))

            s_f1 = float(method.attribute(f1, X_t).detach().cpu().numpy()[:, 0, :].mean())
            s_f2 = float(method.attribute(f2, X_t).detach().cpu().numpy()[:, 0, :].mean())
            diff = abs(s_f1 - s_f2)
            near_collapse = diff < 0.02

            rows.append({
                "method":        method_name,
                "gamma":         gamma,
                "gamma_star":    gamma_star,
                "score_f1":      s_f1,
                "score_f2":      s_f2,
                "diff":          diff,
                "near_collapse": near_collapse,
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "e3_2_adversarial_collapse.csv"), index=False)
    print(f"  Saved → {out_dir}/e3_2_adversarial_collapse.csv")
    return df


def main():
    """Entry point: load config, seed, and run all Experiment Set 3 sub-experiments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg     = load_config(args.config)
    seed_everything(cfg["seed"])
    rng     = np.random.default_rng(cfg["seed"])
    out_dir = str(results_dir(cfg))

    run_e3_1(cfg, rng, out_dir)
    run_e3_2(cfg, rng, out_dir)
    print("\nExperiment Set 3 complete.")


if __name__ == "__main__":
    main()
