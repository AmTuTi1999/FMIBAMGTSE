"""Experiment Set 2 — Gradient-based methods, C1 (off-manifold violation).

E2.1: Off-manifold sensitivity: evaluate each method on f1 vs f3 (equal on supp(p)).
      Expect *different* attributions despite G_f1 = G_f3.
E2.2: Verify ∂f3/∂x1 = -γδ ≠ 0 everywhere on supp(p) yet edge is absent.
E2.3: Manifold vs off-manifold baselines: contrast IG baseline choices.
E2.4: VAR benchmark — SHD/P/R/F1 vs G_f for the gradient family.

Usage:
    python experiments/exp2_gradient.py [--config configs/default.yaml]
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from dagfaith.config import load_config, seed_everything, results_dir
from data.running_example import (
    sample_c1_data,
    make_f1, make_f3,
    make_f1_torch, make_f3_torch,
)
from methods.gradient_methods import (
    SaliencyMethod, IntegratedGradientsMethod,
    TemporalIGMethod, SmoothGradMethod, ExpectedAbsGradientMethod
)


def run_e2_1(cfg: dict, rng: np.random.Generator, out_dir: str) -> pd.DataFrame:
    """E2.1: Off-manifold sensitivity on f1 vs f3 under the C1 manifold case."""
    print("\n=== E2.1: Off-manifold sensitivity ===")
    re_cfg = cfg["running_example"]
    beta  = re_cfg["beta"]
    delta = re_cfg["delta"]
    n     = re_cfg["n_samples"]
    gamma = 0.5

    X = sample_c1_data(n=n, delta=delta, T=1, rng=rng)
    X_t = torch.tensor(X, dtype=torch.float32)

    f1 = make_f1_torch(beta=beta)
    f3 = make_f3_torch(beta=beta, gamma=gamma, delta=delta)

    # Verify f1 == f3 on the manifold data (use numpy versions for the check)
    f1_np = make_f1(beta=beta)
    f3_np = make_f3(beta=beta, gamma=gamma, delta=delta)
    y1 = f1_np(X)
    y3 = f3_np(X)
    max_diff = float(np.max(np.abs(y1 - y3)))
    print(f"  max|f1(x) - f3(x)| on supp(p) = {max_diff:.2e}  (should be ≈ 0)")

    methods = [
        ("Saliency",    SaliencyMethod()),
        ("IG(zero)",    IntegratedGradientsMethod(n_steps=50, baseline="zero")),
        ("IG(mean)",    IntegratedGradientsMethod(n_steps=50, baseline="mean")),
        ("TIG",         TemporalIGMethod(n_steps=20)),
        ("SmoothGrad",  SmoothGradMethod(n_samples=30, noise_std=0.1)),
        ("EAbsGrad",    ExpectedAbsGradientMethod()),
    ]

    rows = []
    for method_name, method in methods:
        s_f1 = method.attribute(f1, X_t).detach().cpu().numpy()  # [B, D, T]
        s_f3 = method.attribute(f3, X_t).detach().cpu().numpy()

        # Score on X1 (index 0) under each model
        score_x1_f1 = float(s_f1[:, 0, :].mean())
        score_x1_f3 = float(s_f3[:, 0, :].mean())
        diff_x1     = abs(score_x1_f3 - score_x1_f1)
        c1_violation = diff_x1 > 0.01  # f1==f3 on supp(p), so scores should match

        print(f"  {method_name:<14}: score(X1|f1)={score_x1_f1:.4f}  "
              f"score(X1|f3)={score_x1_f3:.4f}  diff={diff_x1:.4f}  "
              f"{'C1 VIOLATED ✗' if c1_violation else 'ok ✓'}")
        rows.append({
            "method":       method_name,
            "score_X1_f1":  score_x1_f1,
            "score_X1_f3":  score_x1_f3,
            "diff_X1":      diff_x1,
            "C1_violation": c1_violation,
            "max_f1_f3_diff_on_manifold": max_diff,
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "e2_1_offmanifold.csv"), index=False)
    print(f"  Saved → {out_dir}/e2_1_offmanifold.csv")
    return df


def run_e2_2(cfg: dict, rng: np.random.Generator, out_dir: str) -> pd.DataFrame:
    """E2.2: Verify ∂f3/∂x1 = -γδ ≠ 0 on supp(p) yet edge absent."""
    print("\n=== E2.2: Analytic gradient check ∂f3/∂x1 ===")
    re_cfg = cfg["running_example"]
    beta  = re_cfg["beta"]
    delta = re_cfg["delta"]
    gamma = 0.5
    n     = 100

    # Analytic: f3(x) = β·x2 + γ·(x2 - δ·x1) = (β+γ)·x2 - γδ·x1
    # ∂f3/∂x1 = -γδ  (constant, everywhere including on supp(p))
    expected_grad_x1 = -gamma * delta
    print(f"  Analytic ∂f3/∂x1 = -γδ = -{gamma}·{delta} = {expected_grad_x1:.4f}")

    X = sample_c1_data(n=n, delta=delta, T=1, rng=rng)
    X_t = torch.tensor(X, dtype=torch.float32).requires_grad_(True)

    f3 = make_f3_torch(beta=beta, gamma=gamma, delta=delta)

    # Numerical gradient ∂f3/∂x1
    out = f3(X_t).sum()
    out.backward()
    numerical_grad_x1 = float(X_t.grad[:, 0, :].mean())

    error = abs(numerical_grad_x1 - expected_grad_x1)
    print(f"  Numerical ∂f3/∂x1 = {numerical_grad_x1:.4f}")
    print(f"  Error (analytic vs numeric) = {error:.2e}")
    print(f"  → Gradient is nonzero ({numerical_grad_x1:.4f}) despite edge being absent.")
    print(f"    This confirms the C1 violation: gradient-based methods will flag X1→y.")

    rows = [{
        "beta":               beta,
        "gamma":              gamma,
        "delta":              delta,
        "expected_grad_x1":   expected_grad_x1,
        "numerical_grad_x1":  numerical_grad_x1,
        "error":              error,
        "edge_absent":        True,
        "gradient_nonzero":   abs(numerical_grad_x1) > 1e-4,
    }]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "e2_2_gradient_check.csv"), index=False)
    print(f"  Saved → {out_dir}/e2_2_gradient_check.csv")
    return df


def run_e2_3(cfg: dict, rng: np.random.Generator, out_dir: str) -> pd.DataFrame:
    """E2.3: Manifold vs off-manifold baselines for IG."""
    print("\n=== E2.3: Manifold vs off-manifold IG baselines ===")
    re_cfg = cfg["running_example"]
    beta  = re_cfg["beta"]
    delta = re_cfg["delta"]
    n     = re_cfg["n_samples"]
    gamma = 0.5

    X = sample_c1_data(n=n, delta=delta, T=1, rng=rng)
    X_t = torch.tensor(X, dtype=torch.float32)

    f3 = make_f3_torch(beta=beta, gamma=gamma, delta=delta)

    # Off-manifold baseline: straight line to 0 — traverses off-manifold space
    ig_offman = IntegratedGradientsMethod(n_steps=50, baseline="zero")
    s_offman  = ig_offman.attribute(f3, X_t).detach().cpu().numpy()
    score_x1_offman = float(s_offman[:, 0, :].mean())

    # On-manifold baseline: interpolate within supp(p) — use mean along the manifold
    # On the line x2=δx1, the baseline is x'=(0,0) and every point on the path stays
    # on the manifold iff x2=δx1 everywhere.  We simulate by scaling x uniformly.
    class OnManifoldIG(torch.nn.Module):
        """IG with baseline on the manifold: x'=0 and path stays on x2=δx1."""
        def __init__(self, n_steps=50):
            super().__init__()
            self.n_steps = n_steps

        def attribute(self, f, X):
            """Integrate gradients along α·X (uniform scaling keeps x2=δx1 intact)."""
            X = X.float().detach()
            alphas = torch.linspace(0, 1, self.n_steps + 1)
            grads = []
            for alpha in alphas:
                # On the manifold: scale x uniformly (x2=δx1 is preserved)
                x_alpha = alpha * X
                x_alpha = x_alpha.detach().requires_grad_(True)
                out = f(x_alpha)
                if isinstance(out, Tensor) and out.ndim > 0:
                    out = out.sum()
                out.backward()
                grads.append(x_alpha.grad.detach())
            grads = torch.stack(grads)
            integrated = (grads[:-1] + grads[1:]).mean(0) * 0.5
            return (integrated * X).abs()

    ig_onman  = OnManifoldIG(n_steps=50)
    s_onman   = ig_onman.attribute(f3, X_t).detach().cpu().numpy()
    score_x1_onman = float(s_onman[:, 0, :].mean())

    artifact = score_x1_offman - score_x1_onman
    print(f"  IG(off-manifold, zero baseline): score(X1|f3) = {score_x1_offman:.4f}")
    print(f"  IG(on-manifold, scale baseline): score(X1|f3) = {score_x1_onman:.4f}")
    print(f"  Difference (artifact) = {artifact:.4f}  ← due to path leaving supp(p)")

    rows = [{
        "ig_offmanifold_X1": score_x1_offman,
        "ig_onmanifold_X1":  score_x1_onman,
        "artifact":          artifact,
        "artifact_nonzero":  abs(artifact) > 0.01,
    }]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "e2_3_manifold_baseline.csv"), index=False)
    print(f"  Saved → {out_dir}/e2_3_manifold_baseline.csv")
    return df

def main():
    """Entry point: load config, seed, and run all Experiment Set 2 sub-experiments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    out_dir = str(results_dir(cfg))

    run_e2_1(cfg, rng, out_dir)
    run_e2_2(cfg, rng, out_dir)
    run_e2_3(cfg, rng, out_dir)

    print("\nExperiment Set 2 complete.")


if __name__ == "__main__":
    main()
