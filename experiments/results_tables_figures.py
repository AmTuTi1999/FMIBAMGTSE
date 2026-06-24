"""Task 7 — Generate camera-ready tables and figures.

Table A: Per-method score on X1 under f1/f2/f3 with ✓/✗ for the correct zero.
Table B: SHD/precision/recall/F1 per method (marginal + gradient), by config.
Fig 1:   Adversarial collapse curve — Φ^f1_1 vs Φ^f2_1 across γ.
Fig 2:   Off-manifold sensitivity — gradient score on f1 vs f3.

Usage:
    python experiments/results_tables_figures.py [--config configs/default.yaml]

Outputs (in results/):
    table_A.csv, table_A.tex
    table_B.csv, table_B.tex
    fig_adversarial_collapse.pdf
    fig_offmanifold.pdf
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dagfaith.config import load_config, seed_everything, results_dir
from data.running_example import (
    sample_c2_data, sample_c1_data, make_f1, make_f2, make_f3,
    make_f1_torch, make_f2_torch, make_f3_torch, torch_model_from_numpy_fn,
)
from methods.shap_methods import KernelSHAPMethod, TimeSHAPMethod, ShapTimeMethod
from methods.lime_methods import TSMULEMethod
from methods.gradient_methods import (
    SaliencyMethod, IntegratedGradientsMethod, TemporalIGMethod,
    SmoothGradMethod
)
from methods.perturbation_methods import WinITMethod, DynamaskMethod, FITMethod
import torch


PAPER_STYLE = {
    "font.family":     "serif",
    "font.size":        10,
    "axes.labelsize":   10,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "figure.dpi":       150,
    "lines.linewidth":  1.5,
}
plt.rcParams.update(PAPER_STYLE)


def build_latex_table(cfg, rng, out_dir) -> pd.DataFrame:
    """Table: per-method score on X1 under f1/f2/f3 with ✓/✗."""
    re_cfg = cfg["running_example"]
    beta, delta, n = re_cfg["beta"], re_cfg["delta"], re_cfg["n_samples"]
    gamma = 0.5

    X_c2 = sample_c2_data(n=n, delta=delta, T=1, rng=rng)
    X_c1 = sample_c1_data(n=n, delta=delta, T=1, rng=rng)

    f1_np = make_f1(beta=beta)
    f2_np = make_f2(beta=beta, gamma=gamma)
    f3_np = make_f3(beta=beta, gamma=gamma, delta=delta)
    f1 = torch_model_from_numpy_fn(f1_np)
    f2 = torch_model_from_numpy_fn(f2_np)
    f3 = torch_model_from_numpy_fn(f3_np)

    f1_t = make_f1_torch(beta=beta)
    f2_t = make_f2_torch(beta=beta, gamma=gamma)
    f3_t = make_f3_torch(beta=beta, gamma=gamma, delta=delta)

    X_c2_t = torch.tensor(X_c2, dtype=torch.float32)
    X_c1_t = torch.tensor(X_c1, dtype=torch.float32)

    marginal_methods = [
        ("KernelSHAP",   KernelSHAPMethod(n_background=50)),
        ("TimeSHAP",     TimeSHAPMethod(n_background=30)),
        ("ShapTime",     ShapTimeMethod(n_background=30)),
        ("TS-MULE",      TSMULEMethod(n_segments=1, n_neighbors=100)),
        ("WinIT",                WinITMethod(window_size=1, num_samples=20, metric="pd")),
        ("Dynamask",             DynamaskMethod(num_epoch=150, area_list=list(np.arange(0.25, 0.35, 0.01)))),
        ("FIT",                  FITMethod(num_samples=20)),
    ]
    gradient_methods = [
        ("Saliency",     SaliencyMethod()),
        ("IG(zero)",     IntegratedGradientsMethod(n_steps=50, baseline="zero")),
        ("TIG",          TemporalIGMethod(n_steps=20)),
        ("SmoothGrad",   SmoothGradMethod(n_samples=30)),
    ]

    rows = []

    def score_x1(method, X_t, f):
        """Return the mean absolute attribution score on X1 (variable index 0)."""
        s = method.attribute(f, X_t).detach().cpu().numpy()
        return float(np.abs(s[:, 0, :]).mean())

    print("\nBuilding Latex Table...")
    for name, m in marginal_methods:
        s_f1 = score_x1(m, X_c2_t, f1)
        s_f2 = score_x1(m, X_c2_t, f2)
        s_f3 = score_x1(m, X_c2_t, f3)
        rows.append({
            "Method": name, "Family": "Marginal",
            "score_f1": s_f1, "correct_f1": "✗" if s_f1 > 0.01 else "✓",
            "score_f2": s_f2, "correct_f2": "✓" if s_f2 > 0.01 else "✗",
            "score_f3": s_f3, "correct_f3": "✗" if s_f3 > 0.01 else "✓",
            "C2_violation": s_f1 > 0.01, "C1_violation": False,
        })
        print(f"  {name:<14}: f1={s_f1:.4f} {'✗' if s_f1>0.01 else '✓'}  "
              f"f2={s_f2:.4f}  f3={s_f3:.4f}")

    for name, m in gradient_methods:
        s_f1 = score_x1(m, X_c1_t, f1_t)
        s_f2 = score_x1(m, X_c1_t, f2_t)
        s_f3 = score_x1(m, X_c1_t, f3_t)
        rows.append({
            "Method": name, "Family": "Gradient",
            "score_f1": s_f1, "correct_f1": "✗" if s_f1 > 0.01 else "✓",
            "score_f2": s_f2, "correct_f2": "✓" if s_f2 > 0.01 else "✗",
            "score_f3": s_f3, "correct_f3": "✗" if s_f3 > 0.01 else "✓",
            "C2_violation": False, "C1_violation": abs(s_f3 - s_f1) > 0.01,
        })
        print(f"  {name:<14}: f1={s_f1:.4f} {'✗' if s_f1>0.01 else '✓'}  "
              f"f2={s_f2:.4f}  f3={s_f3:.4f} {'C1-viol' if abs(s_f3-s_f1)>0.01 else ''}")

    df = pd.DataFrame(rows)

    csv_path = os.path.join(out_dir, "table_A.csv")
    tex_path = os.path.join(out_dir, "table_A.tex")
    df.to_csv(csv_path, index=False)

    tex_df = df[["Method", "Family", "score_f1", "correct_f1",
                 "score_f2", "correct_f2", "score_f3", "correct_f3"]].copy()
    tex_df.columns = ["Method", "Family",
                      r"$\Phi^{f_1}_1$", r"$f_1$ ok",
                      r"$\Phi^{f_2}_1$", r"$f_2$ ok",
                      r"$\Phi^{f_3}_1$", r"$f_3$ ok"]
    with open(tex_path, "w") as fh:
        fh.write("% Table A — Running-example attribution scores on X1\n")
        fh.write(tex_df.to_latex(index=False, float_format="%.4f",
                                  escape=False, column_format="llrcrcrcc"))
    print(f"  Saved: {csv_path}, {tex_path}")
    return df


def fig_adversarial_collapse(cfg, out_dir):
    """Fig 1: Φ^f1_1 vs Φ^f2_1 across γ with γ* and collapse point marked."""
    collapse_path = os.path.join(out_dir, "e1_2_adversarial_collapse.csv")
    if not os.path.exists(collapse_path):
        print("  [warning] adversarial collapse data not found; skipping figure.")
        return

    df = pd.read_csv(collapse_path)
    gamma_star = float(df["gamma_star"].iloc[0])

    if "score_f1" not in df.columns or "score_f2" not in df.columns:
        print("  [warning] expected columns score_f1/score_f2 not in CSV; skipping figure.")
        return

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(df["gamma"], df["score_f1"], "b-o", markersize=4, label=r"$\Phi^{f_1}_1$")
    ax.plot(df["gamma"], df["score_f2"], "r--s", markersize=4, label=r"$\Phi^{f_2}_1$")
    ax.axvline(gamma_star, color="gray", linestyle=":", linewidth=1)
    ymax = max(df["score_f1"].max(), df["score_f2"].max())
    ax.text(gamma_star + 0.05, ymax * 0.5,
            r"$\gamma^*\!=\!-2\beta\delta$", fontsize=8, color="gray")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"Attribution score on $X^{(1)}$")
    ax.set_title("Adversarial collapse: scores coincide at $\\gamma^*$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "fig_adversarial_collapse.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_offmanifold(cfg, out_dir):
    """Fig 2: Gradient score on f1 vs f3 per method (off-manifold sensitivity)."""
    offman_path = os.path.join(out_dir, "e2_1_offmanifold.csv")
    if not os.path.exists(offman_path):
        print("  [warning] off-manifold data not found; skipping figure.")
        return

    df = pd.read_csv(offman_path)
    methods = df["method"].tolist()
    x = np.arange(len(methods))
    w = 0.35

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(x - w/2, df["score_X1_f1"], w, label=r"$f_1$ (absent edge)", color="#4878d0")
    ax.bar(x + w/2, df["score_X1_f3"], w, label=r"$f_3$ (same on $\mathrm{supp}(p)$)",
           color="#ee854a")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Attribution score on $X^{(1)}$")
    ax.set_title("C1 violation: $f_1=f_3$ on $\\mathrm{supp}(p)$, scores differ")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(out_dir, "fig_offmanifold.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    """Entry point: build Latex table, and both figures from experiment CSVs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    out_dir = str(results_dir(cfg))

    build_latex_table(cfg, rng, out_dir)
    fig_adversarial_collapse(cfg, out_dir)
    fig_offmanifold(cfg, out_dir)
    print("\nAll tables and figures generated.")


if __name__ == "__main__":
    main()
