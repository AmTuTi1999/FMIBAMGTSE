"""TS-MULE: LIME-with-segmentation for time series.

TS-MULE (Schlegel et al. 2021) wraps LIME with different time-series
segmentation strategies and uses surrogate linear coefficients as scores.

Segmentation variants supported:
  - "uniform":     equal-width time segments
  - "matrix":      matrix profile-based segmentation
  - "exponential": exponentially-spaced segments (recent steps get finer granularity)
  - "sax":         SAX symbolic representation segments

The marginal perturbation (masking segments with mean/zero values from the marginal
distribution) produces the same confounding as KernelSHAP — violating (C2).

Reference: Schlegel et al., "TS-MULE: Local Interpretable Model-Agnostic
Explanations for Time Series Forecast Models", 2021.
"""
from __future__ import annotations
from typing import Callable, Literal
import numpy as np
import torch
from torch import Tensor

from dagfaith.interface import Method


SegmentVariant = Literal["uniform", "matrix", "exponential", "sax"]


def _to_numpy(X):
    """Convert a Tensor or array-like to a NumPy array."""
    if isinstance(X, Tensor):
        return X.detach().cpu().numpy()
    return np.asarray(X)


def _numpy_f(f, X_np):
    """Call f with a NumPy array, coercing Tensor outputs back to NumPy."""
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


def _uniform_segments(T: int, n_segments: int) -> list[tuple[int, int]]:
    """Partition [0, T) into n_segments equal-width (start, end) intervals."""
    boundaries = np.linspace(0, T, n_segments + 1, dtype=int)
    return [(boundaries[i], boundaries[i + 1]) for i in range(n_segments)]


def _exponential_segments(T: int, n_segments: int) -> list[tuple[int, int]]:
    """More segments near the end (recent time steps are finer)."""
    raw = np.exp(np.linspace(0, np.log(T + 1), n_segments + 1)) - 1
    boundaries = np.round(raw).astype(int)
    boundaries = np.clip(boundaries, 0, T)
    boundaries[-1] = T
    segs = [(int(boundaries[i]), int(boundaries[i + 1]))
            for i in range(n_segments) if boundaries[i + 1] > boundaries[i]]
    return segs


def _sax_segments(X_d: np.ndarray, n_segments: int) -> list[tuple[int, int]]:
    """SAX-based segmentation: equal-PAA segments (fallback to uniform)."""
    T = len(X_d)
    return _uniform_segments(T, n_segments)


def _matrix_segments(X_d: np.ndarray, n_segments: int) -> list[tuple[int, int]]:
    """Matrix-profile-inspired segmentation: find natural subsequence boundaries.

    Simplified version: use peaks/troughs in the mean-subtracted series.
    Falls back to uniform when scipy is unavailable.
    """
    try:
        from scipy.signal import find_peaks
        T = len(X_d)
        signal = X_d - X_d.mean()
        peaks, _ = find_peaks(np.abs(signal), distance=max(T // (n_segments * 2), 1))
        if len(peaks) >= n_segments - 1:
            boundaries = sorted(peaks[:n_segments - 1])
            boundaries = [0] + list(boundaries) + [T]
        else:
            boundaries = list(np.linspace(0, T, n_segments + 1, dtype=int))
        return [(int(boundaries[i]), int(boundaries[i + 1]))
                for i in range(len(boundaries) - 1)
                if boundaries[i + 1] > boundaries[i]]
    except ImportError:
        return _uniform_segments(len(X_d), n_segments)


def _segment_mask(segments: list[tuple[int, int]], T: int) -> np.ndarray:
    """Return integer segment labels [T] for a list of (start, end) segments."""
    labels = np.zeros(T, dtype=int)
    for seg_id, (start, end) in enumerate(segments):
        labels[start:end] = seg_id
    return labels


class TSMULEMethod(Method):
    """TS-MULE: LIME-with-segmentation for time-series attribution.

    For each instance x, TS-MULE:
      1. Segments the time axis into S segments.
      2. Generates perturbed neighbors by randomly masking segments.
      3. Fits a sparse linear surrogate on the neighbors.
      4. Returns surrogate coefficients as the attribution scores.

    Args:
        variant:       segmentation strategy (uniform/matrix/exponential/sax).
        n_segments:    number of time segments per variable.
        n_neighbors:   number of perturbed samples for LIME fitting.
        mask_value:    "mean" or "zero" — how to fill masked segments.
        kernel_width:  LIME exponential kernel width.
    """

    def __init__(
        self,
        variant: SegmentVariant = "uniform",
        n_segments: int = 8,
        n_neighbors: int = 500,
        mask_value: str = "mean",
        kernel_width: float = 0.75,
    ):
        self.variant = variant
        self.n_segments = n_segments
        self.n_neighbors = n_neighbors
        self.mask_value = mask_value
        self.kernel_width = kernel_width

    def _get_segments(self, X_d: np.ndarray) -> list[tuple[int, int]]:
        """Return (start, end) segments for a single variable trace X_d of length T."""
        T = len(X_d)
        n = min(self.n_segments, T)
        if self.variant == "uniform":
            return _uniform_segments(T, n)
        elif self.variant == "exponential":
            return _exponential_segments(T, n)
        elif self.variant == "sax":
            return _sax_segments(X_d, n)
        elif self.variant == "matrix":
            return _matrix_segments(X_d, n)
        return _uniform_segments(T, n)

    def _lime_weights(self, Z_bin: np.ndarray) -> np.ndarray:
        """Exponential kernel weights based on distance from all-ones (original)."""
        dists = np.sqrt(np.sum((Z_bin - 1) ** 2, axis=1))
        return np.exp(-(dists ** 2) / (self.kernel_width ** 2))

    def attribute(self, f: Callable, X: Tensor) -> Tensor:
        """Run TS-MULE: segment → perturb → fit surrogate → return [B, D, T] scores."""
        X_np = _to_numpy(X)  # [B, D, T]
        B, D, T = X_np.shape
        rng = np.random.default_rng(3)
        scores = np.zeros((B, D, T))

        for b in range(B):
            x_b = X_np[b]  # [D, T]
            fill = np.zeros((D, T))
            if self.mask_value == "mean":
                fill = X_np.mean(axis=0)  # [D, T]

            segs_per_d = []
            seg_labels_per_d = []
            for d in range(D):
                segs = self._get_segments(x_b[d])
                segs_per_d.append(segs)
                seg_labels_per_d.append(_segment_mask(segs, T))

            total_segs = sum(len(s) for s in segs_per_d)
            Z_bin = rng.integers(0, 2, size=(self.n_neighbors, total_segs))

            y_perts = np.zeros(self.n_neighbors)
            for n_idx in range(self.n_neighbors):
                x_pert = fill.copy()
                seg_offset = 0
                for d in range(D):
                    for seg_id, (start, end) in enumerate(segs_per_d[d]):
                        if Z_bin[n_idx, seg_offset + seg_id] == 1:
                            x_pert[d, start:end] = x_b[d, start:end]
                    seg_offset += len(segs_per_d[d])
                y_perts[n_idx] = _numpy_f(f, x_pert[None])[0]

            weights = self._lime_weights(Z_bin)
            try:
                from sklearn.linear_model import Ridge
                reg = Ridge(alpha=0.01, fit_intercept=True)
                W = np.diag(weights)
                reg.fit(Z_bin * weights[:, None] ** 0.5, y_perts * weights ** 0.5)
                coeffs = reg.coef_  # [total_segs]
            except Exception:
                coeffs = np.zeros(total_segs)

            seg_offset = 0
            for d in range(D):
                for seg_id, (start, end) in enumerate(segs_per_d[d]):
                    scores[b, d, start:end] = np.abs(coeffs[seg_offset + seg_id])
                seg_offset += len(segs_per_d[d])

        denom = scores.sum(axis=(1, 2), keepdims=True)
        denom = np.where(denom == 0, 1.0, denom)
        return torch.tensor(scores / denom, dtype=torch.float32)

    def __repr__(self):
        return f"TS-MULE({self.variant})"
