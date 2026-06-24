"""Running-example data generators for the canonical failure cases.

C2 case (§3 Part 1): X2 = δ·X1 + ε, ε ~ N(0,1), ε ⊥ X1.
  Full support on R² → marginal methods assign nonzero score to X1 under f1.

C1 case (§3 Part 2): X2 = δ·X1 exactly.
  Manifold is a line in R² → gradient methods are sensitive off supp(p).

Models:
  f1(x) = β·x2                          (x1 has no direct effect)        Eq.(2)
  f2(x) = β·x2 + γ·x1                   (x1 has direct effect)           Eq.(3)
  f3(x) = β·x2 + γ·(x2 - δ·x1)         (equals f1 on supp(p))           §3 Part 2
"""
from __future__ import annotations
import numpy as np
import torch
from torch import Tensor


def sample_c2_data(
    n: int = 2000,
    delta: float = 0.5,
    T: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return X of shape [n, 2, T] for the C2 (mediation) case.

    X1 ~ N(0,1), X2 = δ·X1 + ε with ε ~ N(0,1) independent.
    When T>1 each time step is i.i.d. (the structure holds at every t).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    X1 = rng.standard_normal((n, T))
    eps = rng.standard_normal((n, T))
    X2 = delta * X1 + eps
    X = np.stack([X1, X2], axis=1)  # [n, 2, T]
    return X


def sample_c1_data(
    n: int = 2000,
    delta: float = 0.5,
    T: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return X of shape [n, 2, T] for the C1 (manifold) case.

    X1 ~ N(0,1), X2 = δ·X1 exactly — manifold is a line.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    X1 = rng.standard_normal((n, T))
    X2 = delta * X1
    X = np.stack([X1, X2], axis=1)  # [n, 2, T]
    return X


def make_f1(beta: float = 1.0):
    """f1(x) = β·x2  — x1 has no direct effect (the absent-edge model)."""
    def f1(X: np.ndarray) -> np.ndarray:
        # X: [..., 2, T]
        return beta * X[..., 1, :].mean(axis=-1)
    return f1


def make_f2(beta: float = 1.0, gamma: float = 0.5):
    """f2(x) = β·x2 + γ·x1  — x1 has a direct effect."""
    def f2(X: np.ndarray) -> np.ndarray:
        return beta * X[..., 1, :].mean(axis=-1) + gamma * X[..., 0, :].mean(axis=-1)
    return f2


def make_f3(beta: float = 1.0, gamma: float = 0.5, delta: float = 0.5):
    """f3(x) = β·x2 + γ·(x2 - δ·x1)  — equals f1 on supp(p), differs off-manifold.

    On supp(p): x2 = δ·x1, so x2 - δ·x1 = 0 → f3 = β·x2 = f1.
    Gradient ∂f3/∂x1 = -γδ ≠ 0, despite x1 being absent from f3 on supp(p).
    """
    def f3(X: np.ndarray) -> np.ndarray:
        x1 = X[..., 0, :].mean(axis=-1)
        x2 = X[..., 1, :].mean(axis=-1)
        return beta * x2 + gamma * (x2 - delta * x1)
    return f3


def adversarial_gamma(beta: float, delta: float) -> float:
    """Return γ* = -2βδ, the adversarial collapse point (§3).

    At this γ the flat score Φ^{f1}_1 = Φ^{f2}_1 and no threshold separates G_f1, G_f2.
    """
    return -2.0 * beta * delta


def torch_model_from_numpy_fn(fn):
    """Wrap a numpy function so it accepts [B,D,T] torch tensors.

    NOTE: this breaks autograd (detach is required for numpy).
    For gradient-based attribution methods, use make_f1_torch et al. instead.
    """
    def wrapper(X: Tensor) -> Tensor:
        X_np = X.detach().cpu().numpy()
        out = fn(X_np)
        return torch.tensor(out, dtype=X.dtype, device=X.device)
    return wrapper


def make_f1_torch(beta: float = 1.0):
    """Torch-differentiable f1(X) = β · mean(X[:,1,:]) over time."""
    def f1(X: Tensor) -> Tensor:
        return beta * X[:, 1, :].mean(dim=-1)
    return f1


def make_f2_torch(beta: float = 1.0, gamma: float = 0.5):
    """Torch-differentiable f2(X) = β·X2 + γ·X1."""
    def f2(X: Tensor) -> Tensor:
        return beta * X[:, 1, :].mean(dim=-1) + gamma * X[:, 0, :].mean(dim=-1)
    return f2


def make_f3_torch(beta: float = 1.0, gamma: float = 0.5, delta: float = 0.5):
    """Torch-differentiable f3(X) = β·X2 + γ·(X2 - δ·X1)."""
    def f3(X: Tensor) -> Tensor:
        x1 = X[:, 0, :].mean(dim=-1)
        x2 = X[:, 1, :].mean(dim=-1)
        return beta * x2 + gamma * (x2 - delta * x1)
    return f3
