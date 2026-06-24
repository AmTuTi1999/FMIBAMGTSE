"""Unified attribution interface: Method.attribute(f, X) -> Tensor[B, D, T]."""
from __future__ import annotations
import abc
from typing import Callable
import torch
from torch import Tensor


class Method(abc.ABC):
    """Base class for all attribution methods.

    Every method must return a score tensor shaped [B, D, T] where
      B = batch size (number of time series instances),
      D = number of input variables,
      T = number of time steps.
    Scores are non-negative and represent the importance of each (d,t) feature.
    """

    @abc.abstractmethod
    def attribute(self, f: Callable, X: Tensor) -> Tensor:
        """Compute attribution scores.

        Args:
            f: callable, takes X of shape [B, D, T] → scalar or [B] tensor.
            X: input tensor [B, D, T].

        Returns:
            scores: Tensor [B, D, T], non-negative importance scores.
        """

    def __repr__(self):
        return self.__class__.__name__
