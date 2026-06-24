"""Gradient-based attribution method wrappers: Saliency, IG, TIG, SmoothGrad.

All wrappers implement Method.attribute(f, X) -> Tensor[B, D, T].

The key property under test: these methods compute gradients of f w.r.t. X,
which means their scores depend on how f behaves off the data manifold supp(p).
Two functions f1 and f3 that agree on supp(p) but differ off it will receive
different attribution scores — violating (C1) on-manifold evaluation.

References:
  - Saliency:             Simonyan et al. 2013
  - Integrated Gradients: Sundararajan et al. 2017
  - Temporal IG (TIG):    Leino et al. 2018 / Tonekaboni et al.
  - SmoothGrad:           Smilkov et al. 2017
"""
from __future__ import annotations
from typing import Callable
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from dagfaith.interface import Method


def _normalize(x: Tensor) -> Tensor:
    """Divide [B, D, T] tensor by per-instance sum so each instance sums to 1."""
    denom = x.sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)
    return x / denom


def _require_grad_f(f: Callable, X: Tensor) -> Tensor:
    """Run f(X) ensuring X has gradient tracking."""
    X = X.requires_grad_(True)
    out = f(X)
    if isinstance(out, Tensor) and out.ndim > 0:
        out = out.sum()
    return out, X


class SaliencyMethod(Method):
    """Vanilla saliency: |∂f/∂x|.

    The gradient is evaluated at the input point x, which includes sensitivity
    to directions orthogonal to the data manifold — violating (C1).
    """

    def attribute(self, f: Callable, X: Tensor) -> Tensor:
        """Return |∂f/∂X| normalised to sum 1 per instance."""
        X = X.float().detach()
        X.requires_grad_(True)
        out = f(X)
        if isinstance(out, Tensor) and out.ndim > 0:
            out = out.sum()
        out.backward()
        return _normalize(X.grad.abs().detach())  # [B, D, T]

    def __repr__(self):
        return "Saliency"


class IntegratedGradientsMethod(Method):
    """Integrated Gradients (IG): ∫₀¹ ∂f(x'+α(x−x'))/∂x · (x−x') dα.

    Uses a straight-line interpolation path from baseline x' to input x.
    The path traverses off-manifold regions (the line x' + α(x-x') may leave supp(p)),
    so the resulting scores depend on f off the manifold — violating (C1).

    Args:
        n_steps:   number of quadrature steps along the interpolation path.
        baseline:  "zero" or "mean" — the reference point x'.
    """

    def __init__(self, n_steps: int = 50, baseline: str = "zero"):
        self.n_steps = n_steps
        self.baseline = baseline

    def _get_baseline(self, X: Tensor) -> Tensor:
        """Return the baseline tensor x' (zeros or batch mean) matching X's shape."""
        if self.baseline == "mean":
            return X.mean(dim=0, keepdim=True).expand_as(X)
        return torch.zeros_like(X)

    def attribute(self, f: Callable, X: Tensor) -> Tensor:
        """Integrate gradients along the straight-line path from baseline to X."""
        X = X.float().detach()
        x0 = self._get_baseline(X)
        alphas = torch.linspace(0, 1, self.n_steps + 1, device=X.device)

        grads = []
        for alpha in alphas:
            x_interp = x0 + alpha * (X - x0)
            x_interp = x_interp.detach().requires_grad_(True)
            out = f(x_interp)
            if isinstance(out, Tensor) and out.ndim > 0:
                out = out.sum()
            out.backward()
            grads.append(x_interp.grad.detach())

        grads = torch.stack(grads, dim=0)  # [n_steps+1, B, D, T]
        integrated = (grads[:-1] + grads[1:]).mean(dim=0) * 0.5  # [B, D, T]
        ig = integrated * (X - x0)
        return _normalize(ig.abs())  # [B, D, T]

    def __repr__(self):
        return f"IG(n_steps={self.n_steps}, baseline={self.baseline})"


class TemporalIGMethod(Method):
    """Temporal Integrated Gradients (TIG): sequential IG extension.

    Integrates along the temporal axis by progressively revealing time steps,
    treating each new time step as its own integral path. This measures the
    marginal contribution of each time step t given steps 0..t-1.

    Still traverses off-manifold regions (partially-revealed sequences are
    unlikely under the joint distribution p) — violating (C1).

    Args:
        n_steps:    quadrature steps per temporal integration.
        baseline:   fill value for masked (future) time steps.
    """

    def __init__(self, n_steps: int = 20, baseline: str = "zero"):
        self.n_steps = n_steps
        self.baseline = baseline

    def attribute(self, f: Callable, X: Tensor) -> Tensor:
        """Compute per-time-step IG scores by integrating gradients as each step is revealed."""
        X = X.float().detach()
        B, D, T = X.shape

        if self.baseline == "mean":
            fill = X.mean(dim=0, keepdim=True).expand_as(X).clone()
        else:
            fill = torch.zeros_like(X)

        scores = torch.zeros_like(X)

        for t in range(T):
            x0 = X.clone()
            x0[:, :, t] = fill[:, :, t]

            alphas = torch.linspace(0, 1, self.n_steps + 1, device=X.device)
            grads_t = []
            for alpha in alphas:
                x_interp = X.clone()
                x_interp[:, :, t] = x0[:, :, t] + alpha * (X[:, :, t] - x0[:, :, t])
                x_interp = x_interp.detach().requires_grad_(True)
                out = f(x_interp)
                if isinstance(out, Tensor) and out.ndim > 0:
                    out = out.sum()
                out.backward()
                grads_t.append(x_interp.grad[:, :, t].detach())

            grads_t = torch.stack(grads_t, dim=0)  # [n_steps+1, B, D]
            integrated = (grads_t[:-1] + grads_t[1:]).mean(dim=0) * 0.5
            scores[:, :, t] = (integrated * (X[:, :, t] - x0[:, :, t])).abs()

        return _normalize(scores)  # [B, D, T]

    def __repr__(self):
        return f"TIG(n_steps={self.n_steps})"


class SmoothGradMethod(Method):
    """SmoothGrad: E[|∂f(x+η)/∂x|] with η ~ N(0, σ²I).

    Averages gradients over Gaussian-noisy inputs to reduce sensitivity noise.
    Adding noise perturbs x off the manifold — the expected gradient still depends
    on f's off-manifold behavior — violating (C1).

    Args:
        n_samples:  number of noisy samples.
        noise_std:  standard deviation of additive Gaussian noise σ.
    """

    def __init__(self, n_samples: int = 50, noise_std: float = 0.1):
        self.n_samples = n_samples
        self.noise_std = noise_std

    def attribute(self, f: Callable, X: Tensor) -> Tensor:
        """Average |∂f/∂(X+η)| over Gaussian noise samples and normalise."""
        X = X.float().detach()
        grad_sum = torch.zeros_like(X)

        for _ in range(self.n_samples):
            noise = torch.randn_like(X) * self.noise_std
            x_noisy = (X + noise).detach().requires_grad_(True)
            out = f(x_noisy)
            if isinstance(out, Tensor) and out.ndim > 0:
                out = out.sum()
            out.backward()
            grad_sum += x_noisy.grad.abs().detach()

        return _normalize(grad_sum / self.n_samples)  # [B, D, T]

    def __repr__(self):
        return f"SmoothGrad(n={self.n_samples}, σ={self.noise_std})"


class ExpectedAbsGradientMethod(Method):
    """Expected absolute gradient: E_p[|∂f/∂x|] estimated over the data distribution.

    This is the edge detection criterion from Rem. 2: computes E[|∂f/∂x|] over
    the empirical distribution of X (rather than at a single point).
    Violates (C1) when the support of p is a lower-dimensional manifold.
    """

    def attribute(self, f: Callable, X: Tensor) -> Tensor:
        """Compute E_p[|∂f/∂X|] over the batch and normalise."""
        X = X.float().detach()
        X_req = X.detach().requires_grad_(True)
        out = f(X_req)
        if isinstance(out, Tensor) and out.ndim > 0:
            out = out.sum()
        out.backward()
        avg_grad = X_req.grad.abs().detach()  # [B, D, T]
        return _normalize(avg_grad)

    def __repr__(self):
        return "ExpectedAbsGrad"
