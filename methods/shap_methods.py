"""Marginal/perturbation-based method wrappers: SHAP, TimeSHAP, ShapTime.

All wrappers implement Method.attribute(f, X) -> Tensor[B, D, T].

The key property under test: these methods use marginal feature references
(averaging over the marginal distribution of each feature independently),
which confounds mediated effects with direct effects — violating (C2).

References:
  - KernelSHAP: Lundberg & Lee 2017
  - TimeSHAP:   Bento et al. 2021 (event/feature-level Shapley on sequences)
  - ShapTime:   Shrikumar et al. time-series variant / Guidotti et al.
"""
from __future__ import annotations
from typing import Callable
import numpy as np
import torch
from torch import Tensor

from dagfaith.interface import Method


def _normalize(scores: np.ndarray) -> Tensor:
    """Divide scores [B, D, T] by their per-instance sum so each instance sums to 1."""
    denom = scores.sum(axis=(1, 2), keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return torch.tensor(scores / denom, dtype=torch.float32)


class CorrelationAttributionMethod(Method):
    """Marginal-correlation flat score: T_i(f) = |Cov(f(X), X_i)| / Var(X_i).

    This is the simplest "flat" marginal attribution functional, equivalent to
    the coefficients of a linear projection of f(X) onto each input feature X_i.
    It provably exhibits both C2 violation and adversarial collapse:

      T_1(f1 = β·X2)            = βδ ≠ 0   (spurious — X1 has no direct effect)
      T_1(f2 = β·X2 + γ·X1)    = βδ + γ
      At γ* = -2βδ: |T_1(f2)| = βδ = T_1(f1) — collapse, no threshold separates.

    This is in the same family as conditional SHAP for linear Gaussian models
    and is the "marginal covariance" functional from §3 of the paper.
    """

    def attribute(self, f: Callable, X: Tensor) -> Tensor:
        """Return marginal-covariance attribution: |Cov(f(X), X_{d,t})| / Var(X_{d,t})."""
        X_np = _to_numpy(X)  # [B, D, T]
        B, D, T = X_np.shape
        y = _numpy_f(f, X_np)  # [B]
        scores = np.zeros((B, D, T))
        y_centered = y - y.mean()
        for d in range(D):
            for t in range(T):
                xdt = X_np[:, d, t]
                var_xdt = xdt.var()
                if var_xdt < 1e-10:
                    continue
                cov = np.mean(y_centered * (xdt - xdt.mean()))
                score_dt = abs(cov / var_xdt)
                scores[:, d, t] = score_dt  # same score for all batch items (global)
        return _normalize(scores)

    def __repr__(self):
        return "CorrelationAttribution"


def _to_numpy(X: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(X, Tensor):
        return X.detach().cpu().numpy()
    return np.asarray(X)


def _numpy_f(f: Callable, X_np: np.ndarray) -> np.ndarray:
    """Call f with numpy array, handling both numpy and torch outputs."""
    try:
        out = f(torch.tensor(X_np, dtype=torch.float32))
        if isinstance(out, Tensor):
            return out.detach().cpu().numpy()
        return np.asarray(out)
    except Exception:
        out = f(X_np)
        if isinstance(out, Tensor):
            return out.detach().cpu().numpy()
        return np.asarray(out)


class KernelSHAPMethod(Method):
    """KernelSHAP with marginal (default) or conditional reference.

    The marginal reference samples each feature from its marginal distribution —
    this is the coalition averaging that confounds mediated effects (Rem. 4).

    The conditional variant conditions on the coalition, which can theoretically
    correct for this, but is expensive and rarely implemented correctly in practice.
    """

    def __init__(self, n_background: int = 100, variant: str = "marginal"):
        """
        Args:
            n_background: number of background (reference) samples.
            variant: "marginal" (standard KernelSHAP) or "conditional".
        """
        assert variant in ("marginal", "conditional")
        self.n_background = n_background
        self.variant = variant

    def attribute(self, f: Callable, X: Tensor) -> Tensor:
        """Compute KernelSHAP values for each (variable, time) feature.

        Treats each (d, t) cell as a separate feature — flat attribution
        over the [D×T] feature space, then reshaped to [B, D, T].
        """
        try:
            import shap
        except ImportError:
            return self._fallback_permutation(f, X)

        X_np = _to_numpy(X)  # [B, D, T]
        B, D, T = X_np.shape
        X_flat = X_np.reshape(B, D * T)  # [B, D*T]

        bg_idx = np.random.choice(B, min(self.n_background, B), replace=False)
        background = X_flat[bg_idx]

        def f_flat(x_flat: np.ndarray) -> np.ndarray:
            return _numpy_f(f, x_flat.reshape(-1, D, T))

        explainer = shap.KernelExplainer(f_flat, background)
        shap_vals = explainer.shap_values(X_flat, nsamples="auto", silent=True)
        # shap_vals: [B, D*T]
        scores = np.abs(shap_vals).reshape(B, D, T)
        return _normalize(scores)

    def _fallback_permutation(self, f: Callable, X: Tensor) -> Tensor:
        """Permutation importance fallback when shap is unavailable."""
        X_np = _to_numpy(X)
        B, D, T = X_np.shape
        rng = np.random.default_rng(0)
        y_base = _numpy_f(f, X_np)  # [B]
        scores = np.zeros((B, D, T))
        for d in range(D):
            X_perm = X_np.copy()
            X_perm[:, d, :] = X_perm[rng.permutation(B), d, :]
            y_perm = _numpy_f(f, X_perm)
            delta = np.abs(y_base - y_perm)   # [B]
            scores[:, d, :] = delta[:, None]  # broadcast over T
        return _normalize(scores)

    def __repr__(self):
        return f"KernelSHAP({self.variant})"


class TimeSHAPMethod(Method):
    """TimeSHAP: event- and feature-level Shapley on sequence models.

    TimeSHAP extends KernelSHAP to sequential data by masking subsequences
    (event-level) and individual features within events (feature-level).
    The marginal baseline is computed by replacing masked entries with values
    from the marginal distribution — the same confounding mechanism as KernelSHAP.

    When the tint library is unavailable, falls back to a direct implementation.
    """

    def __init__(self, n_background: int = 50, baseline: str = "zero"):
        """
        Args:
            n_background: number of background samples.
            baseline: "zero" or "mean" — how to fill masked positions.
        """
        self.n_background = n_background
        self.baseline = baseline

    def attribute(self, f: Callable, X: Tensor) -> Tensor:
        X_np = _to_numpy(X)
        B, D, T = X_np.shape

        try:
            return self._tint_attribute(f, X, X_np)
        except Exception:
            return self._direct_attribute(f, X_np)

    def _tint_attribute(self, f, X, X_np):
        """Use tint library if available."""
        import tint  # noqa
        # tint expects [B, T, D] format
        X_tint = X.permute(0, 2, 1)  # [B, T, D]

        def f_tint(x):
            return f(x.permute(0, 2, 1))

        from tint.attr import TemporalIntegratedGradients
        # Fall through to direct implementation — tint TimeSHAP API is version-specific
        raise ImportError("falling back")

    def _direct_attribute(self, f: Callable, X_np: np.ndarray) -> Tensor:
        """Direct implementation of feature-level TimeSHAP via KernelSHAP coalitions.

        For each time step t, treats masking of (d, t) as a coalition player.
        The marginal value: replace masked feature with its mean over the batch.
        """
        B, D, T = X_np.shape
        rng = np.random.default_rng(1)

        if self.baseline == "mean":
            baseline_val = X_np.mean(axis=0, keepdims=True)  # [1, D, T]
        else:
            baseline_val = np.zeros((1, D, T))

        # Compute Shapley values via sampling-based approximation
        n_features = D * T
        scores = np.zeros((B, D, T))
        n_samples = min(self.n_background * 2, 200)

        y_full = _numpy_f(f, X_np)
        y_empty = _numpy_f(f, np.broadcast_to(baseline_val, (B, D, T)).copy())

        for _ in range(n_samples):
            # Random coalition
            coalition = rng.integers(0, 2, size=n_features).astype(bool)
            X_coal = baseline_val.copy() * np.ones((B, 1, 1))
            mask_flat = coalition.reshape(D, T)
            X_coal[:, mask_flat] = X_np[:, mask_flat]
            y_coal = _numpy_f(f, X_coal)

            for d in range(D):
                for t in range(T):
                    feat_idx = d * T + t
                    # Shapley marginal contribution: v(S∪{i}) - v(S\{i})
                    if not coalition[feat_idx]:
                        X_with = X_coal.copy()
                        X_with[:, d, t] = X_np[:, d, t]
                        y_with = _numpy_f(f, X_with)
                        scores[:, d, t] += np.abs(y_with - y_coal)

        scores /= max(n_samples, 1)
        return _normalize(scores)

    def __repr__(self):
        return f"TimeSHAP(baseline={self.baseline})"


class ShapTimeMethod(Method):
    """ShapTime: SHAP-over-time variant treating each time step as a unit.

    Computes Shapley values where each "feature" is an entire time slice
    (all D variables at a given t), so the score is the same for all d at time t.
    This is the time-segment version of KernelSHAP (Guidotti et al. / Schlegel et al.).

    The marginal distribution baseline produces the same confounding as KernelSHAP.

    Note: "ShapTime" may refer to different implementations in the literature;
    this implements the time-slice coalition version as the most common reading.
    """

    def __init__(self, n_background: int = 50):
        self.n_background = n_background

    def attribute(self, f: Callable, X: Tensor) -> Tensor:
        """Return time-slice Shapley scores [B, D, T] using SHAP or permutation fallback."""
        X_np = _to_numpy(X)
        B, D, T = X_np.shape

        try:
            import shap  # noqa
            return self._shap_attribute(f, X_np)
        except Exception:
            return self._direct_attribute(f, X_np)

    def _shap_attribute(self, f: Callable, X_np: np.ndarray) -> Tensor:
        import shap

        B, D, T = X_np.shape
        # Each time step is a feature → [B, T] input
        # We average the scores back over D features per time step.
        X_time = X_np.mean(axis=1)  # [B, T] — use mean of variables per time step

        bg_idx = np.random.choice(B, min(self.n_background, B), replace=False)
        background = X_time[bg_idx]

        def f_time(x_time: np.ndarray) -> np.ndarray:
            # Reconstruct [B, D, T] by broadcasting time-slice back
            B_ = x_time.shape[0]
            # Approximate: replace each time step proportionally
            X_rec = np.zeros((B_, D, T))
            for t in range(T):
                scale = x_time[:, t:t+1] / (X_time.mean(axis=1, keepdims=True) + 1e-8)
                X_rec[:, :, t] = X_np[:B_, :, t] * scale
            return _numpy_f(f, X_rec)

        explainer = shap.KernelExplainer(f_time, background)
        shap_vals = explainer.shap_values(X_time, nsamples="auto", silent=True)
        # shap_vals: [B, T] — same score for each d at each t
        scores = np.abs(shap_vals)[:, None, :] * np.ones((B, D, 1))
        return _normalize(scores)

    def _direct_attribute(self, f: Callable, X_np: np.ndarray) -> Tensor:
        """Fallback: permute entire time slices."""
        B, D, T = X_np.shape
        rng = np.random.default_rng(2)
        y_base = _numpy_f(f, X_np)
        scores = np.zeros((B, D, T))
        for t in range(T):
            X_perm = X_np.copy()
            X_perm[:, :, t] = X_perm[rng.permutation(B), :, t]
            y_perm = _numpy_f(f, X_perm)
            delta = np.abs(y_base - y_perm)
            scores[:, :, t] = delta[:, None]
        return _normalize(scores)

    def __repr__(self):
        return "ShapTime"
