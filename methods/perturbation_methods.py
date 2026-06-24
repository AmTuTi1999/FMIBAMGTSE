"""Perturbation-based attribution methods: WinIT, Dynamask, FIT.

Wrappers around the implementations in the WinIT repository
(../WinIT/winit/explainer/).  Each class adapts the WinIT explainer API to
the project's Method.attribute(f, X) -> Tensor[B, D, T] interface.

Key adapter: _FModel(TorchModel) wraps a plain callable f:[B,D,T]->[B] into
the TorchModel interface expected by the WinIT explainers.  sigmoid activation
produces pseudo-probabilities in (0,1) for WinIT's pd/kl/js metrics and
FIT's KL divergence term.

WinIT  — data-distribution counterfactuals (no generator training).
Dynamask — regression-adapted (MSE loss instead of log-loss).
FIT    — marginal-resampling counterfactuals (no generator training).
         Note: FIT is designed for sequential/RNN models; for t in [1,T) it
         compares p(y|x_{0:t}) vs p(y|x_{0:t-1}).  At T=1 (running example)
         all scores are zero by construction.
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
from typing import Callable

import numpy as np
import torch
from torch import Tensor

from dagfaith.interface import Method

_WINIT_ROOT = pathlib.Path(__file__).resolve().parent.parent / "WinIT"
if str(_WINIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_WINIT_ROOT))

from winit.models import TorchModel
from winit.explainer.winitexplainers import WinITExplainer
from winit.explainer.dynamaskexplainer import DynamaskExplainer
from winit.explainer.fitexplainers import FITExplainer
from winit.explainer.attribution.mask_group import MaskGroup

class _FModel(TorchModel):
    """Wraps f:[B,D,T]→[B] as a TorchModel with Sigmoid activation (num_states=1).

    forward() returns f(x) in "logit" space so that predict() = sigmoid(forward())
    lies in (0,1), satisfying the probability requirements of WinIT and FIT.

    When called with a shorter sequence (e.g. x[:, :, :t+1] from FIT) the model
    receives whatever shape is passed; if f cannot handle it the score for that
    (feature, time) pair is silently set to 0 via the except clause in FIT.
    """

    def __init__(self, f: Callable, D: int, device: str = "cpu"):
        super().__init__(feature_size=D, num_states=1, hidden_size=0, device=device)
        self._f = f

    def forward(self, x: Tensor, return_all: bool = True) -> Tensor:
        # x: [B, D, T_in]
        B, D, T_in = x.shape
        try:
            with torch.no_grad():
                out = self._f(x)
                if isinstance(out, Tensor):
                    out = out.detach().float()
                    if out.ndim > 1:
                        out = out.squeeze(-1)
                else:
                    out = torch.tensor(out, dtype=torch.float32)
        except Exception:
            out = torch.zeros(B, dtype=torch.float32)

        if return_all:
            return out.view(B, 1, 1).expand(B, 1, T_in).contiguous()  # [B, 1, T]
        return out.view(B, 1)  # [B, 1]


class WinITMethod(Method):
    """WinIT (Leung et al., AAAI 2023) via WinITExplainer.

    Counterfactuals are sampled from the empirical marginal of the input batch X
    (data-distribution mode — no generator training required).  The raw output
    has shape [B, D, T, window_size]; it is reduced to [B, D, T] by summing
    absolute values over the window dimension.

    Args:
        window_size: maximum lookback window W (paper default: 10).
        num_samples: MC samples per (feature, window) pair.
        metric:      comparison metric — "pd" (absolute diff), "kl", or "js".
    """

    def __init__(
        self,
        window_size: int = 4,
        num_samples: int = 10,
        metric: str = "pd",
    ):
        self.window_size = window_size
        self.num_samples = num_samples
        self.metric = metric
        self._tmpdir = tempfile.mkdtemp()

    def attribute(self, f: Callable, X: Tensor) -> Tensor:
        """Run WinITExplainer with marginal sampling and return [B, D, T] scores."""
        X = X.float().detach()
        B, D, T = X.shape

        model = _FModel(f, D)
        explainer = WinITExplainer(
            device="cpu",
            num_features=D,
            data_name="adapt",
            path=pathlib.Path(self._tmpdir),
            window_size=self.window_size,
            num_samples=self.num_samples,
            metric=self.metric,
        )

        explainer.data_distribution = X.cpu().numpy()
        explainer.set_model(model)

        scores = explainer.attribute(X)          # numpy [B, D, T, window_size]
        flat = np.abs(scores).sum(axis=-1)       # [B, D, T]
        denom = flat.sum()
        return torch.tensor(flat / (denom if denom != 0 else 1.0), dtype=torch.float32)

    def __repr__(self) -> str:
        return f"WinIT(W={self.window_size}, n={self.num_samples}, metric={self.metric})"



def _mse_loss_multiple(y_pred: Tensor, y_target: Tensor) -> Tensor:
    """Per-sample MSE for regression; averages over (N_area, T, state) dims → [B]."""
    return ((y_pred - y_target) ** 2).mean(dim=[0, 2, 3])


class _DynamaskRegressionExplainer(DynamaskExplainer):
    """DynamaskExplainer with MSE loss (replaces log-loss for regression models)."""

    def attribute(self, x: Tensor) -> np.ndarray:
        """Compute the extremal mask for x using MSE regression loss."""
        self.base_model.eval()
        self.base_model.zero_grad()
        return self._attribute_regression(x)

    def _attribute_regression(self, x: Tensor) -> np.ndarray:
        B, D, T = x.shape
        orig_cudnn = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

        def f_reg(x_in: Tensor) -> Tensor:
            x_perm = x_in.permute(0, 2, 1)           # [B, D, T]
            out = self.base_model(x_perm, return_all=True)   # [B, 1, T]
            return out.permute(0, 2, 1)               # [B, T, 1]

        x_btd = x.permute(0, 2, 1)  # [B, T, D]

        mask_group = MaskGroup(
            self.pert, self.device, verbose=False, deletion_mode=self.deletion_mode
        )
        mask_group.fit_multiple(
            X=x_btd,
            f=f_reg,
            use_last_timestep_only=self.use_last_timestep_only,
            loss_function_multiple=_mse_loss_multiple,
            area_list=list(self.area_list),
            learning_rate=1.0,
            size_reg_factor_init=0.1,
            size_reg_factor_dilation=self.size_reg_factor_dilation,
            initial_mask_coeff=0.5,
            n_epoch=self.num_epoch,
            momentum=1.0,
            time_reg_factor=self.time_reg_factor,
        )

        y_test = f_reg(x_btd)                    # [B, T, 1]
        thresh = (0.05 * y_test.var()).expand(B)  # [B]
        mask = mask_group.get_extremal_mask_multiple(thresholds=thresh)
        mask_saliency = mask.permute(0, 2, 1)

        torch.backends.cudnn.enabled = orig_cudnn
        return (mask_saliency.detach().cpu().numpy())/(mask_saliency.detach().cpu().numpy()).sum(axis=(1, 2), keepdims=True)  # [B, D, T] normalized to sum=1


class DynamaskMethod(Method):
    """Dynamask (Crabbe & van der Schaar, ICML 2021) via DynamaskExplainer.

    Optimises a per-instance near-binary mask m∈(0,1)^(D×T) to minimise
    sparsity while preserving the model's prediction.  Uses MSE loss so
    it applies to regression models.

    Args:
        area_list:               target mask-area values (fraction of features to retain).
        num_epoch:               optimisation steps.
        blur_type:               "fadema" (fade-to-moving-average) or "gaussian".
        time_reg_factor:         temporal-smoothness regularisation weight.
        size_reg_factor_dilation: dilation ratio for the size regulariser.
    """

    def __init__(
        self,
        area_list: list[float] | None = None,
        num_epoch: int = 200,
        blur_type: str = "fadema",
        time_reg_factor: float = 1.0,
        size_reg_factor_dilation: float = 100.0,
    ):
        self.area_list = area_list if area_list is not None else list(np.arange(0.25, 0.35, 0.01))
        self.num_epoch = num_epoch
        self.blur_type = blur_type
        self.time_reg_factor = time_reg_factor
        self.size_reg_factor_dilation = size_reg_factor_dilation

    def attribute(self, f: Callable, X: Tensor) -> Tensor:
        """Optimise a sparsity-preserving mask via MSE and return [B, D, T] scores."""
        X = X.float().detach()
        _, D, _ = X.shape

        model = _FModel(f, D)
        explainer = _DynamaskRegressionExplainer(
            device="cpu",
            area_list=self.area_list,
            num_epoch=self.num_epoch,
            blur_type=self.blur_type,
            size_reg_factor_dilation=self.size_reg_factor_dilation,
            time_reg_factor=self.time_reg_factor,
        )
        explainer.set_model(model, set_eval=False)

        scores = explainer.attribute(X)  # numpy [B, D, T]
        return torch.tensor(scores, dtype=torch.float32)

    def __repr__(self) -> str:
        return f"Dynamask(epochs={self.num_epoch}, blur={self.blur_type})"



class _MarginalFITGenerator:
    """Drop-in replacement for JointFeatureGenerator using empirical marginal sampling.

    sig_inds = features to keep fixed (conditioned on).  All other features at
    the current time step are replaced with draws from the training pool.

    No training required; this is the "data-distribution" analogue for FIT.
    """

    def __init__(self, X_train: np.ndarray, device: str = "cpu"):
        self._pool = X_train   # [N, D, T]
        self.device = device

    def eval(self) -> "_MarginalFITGenerator":
        """No-op: satisfies the generator interface (no trainable parameters)."""
        return self

    def to(self, device: str) -> "_MarginalFITGenerator":
        """Store the target device; tensors are moved on creation."""
        self.device = device
        return self

    def get_z_mu_std(self, x: Tensor):
        """Return (None, None): no latent encoding for marginal sampling."""
        return None, None

    def forward_conditional_multisample_from_z_mu_std(
        self,
        past: Tensor,
        current,           # [B, D] or [B, D, 1]
        sig_inds: list[int],
        mu_z,
        std_z,
        n_samples: int,
    ):
        """Sample n_samples counterfactuals: hold sig_inds fixed, resample others from pool."""
        if isinstance(current, Tensor):
            cur = current.detach().cpu().numpy()
        else:
            cur = np.asarray(current)

        if cur.ndim == 3:
            cur = cur[:, :, 0]   # [B, D]

        B, D = cur.shape
        # x_hat_t: [n_samples, B, D, 1]
        x_hat_t = np.tile(cur[None, :, :, None], (n_samples, 1, 1, 1)).copy()

        for d in range(D):
            if d not in sig_inds:
                pool_d = self._pool[:, d, :].reshape(-1)
                x_hat_t[:, :, d, 0] = np.random.choice(pool_d, size=(n_samples, B))

        return torch.tensor(x_hat_t, dtype=torch.float32, device=self.device), None


class FITMethod(Method):
    """FIT (Tonekaboni et al., NeurIPS 2020) via FITExplainer.

    Marginal resampling (no generator training) replaces the JointFeatureGenerator.
    FIT iterates over t∈[1,T), so at T=1 all scores are zero — this is
    expected and interpretable: a single time step carries no temporal information
    gain.  For T≥2 the score reflects how much f changes when feature d at time t
    is replaced by a draw from its marginal, relative to the prediction change from
    t-1 to t.

    Args:
        num_samples: MC samples per (feature, time) pair.
    """

    def __init__(self, num_samples: int = 10):
        self.num_samples = num_samples
        self._tmpdir = tempfile.mkdtemp()

    def attribute(self, f: Callable, X: Tensor) -> Tensor:
        """Run FITExplainer with marginal resampling and return normalised [B, D, T] scores."""
        X = X.float().detach()
        B, D, T = X.shape
        X_np = X.cpu().numpy()

        model = _FModel(f, D)
        gen = _MarginalFITGenerator(X_np, device="cpu")

        explainer = FITExplainer(
            device="cpu",
            feature_size=D,
            data_name="adapt",
            path=pathlib.Path(self._tmpdir),
            num_samples=self.num_samples,
        )
        explainer.generator = gen
        explainer.set_model(model)

        scores = explainer.attribute(X)  # numpy [B, D, T]
        scores_abs = np.abs(scores)
        denom = scores_abs.sum(axis=(1, 2), keepdims=True)
        denom = np.where(denom == 0, 1.0, denom)
        return torch.tensor(scores_abs / denom, dtype=torch.float32)

    def __repr__(self) -> str:
        return f"FIT(n={self.num_samples})"
