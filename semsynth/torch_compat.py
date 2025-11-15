"""Compatibility helpers for optional PyTorch features."""

from __future__ import annotations

import numbers
from typing import Tuple


def ensure_torch_rmsnorm() -> None:
    """Ensure torch.nn exposes RMSNorm on versions that predate it."""

    try:
        import torch
        from torch import nn
    except Exception:  # pragma: no cover - torch is optional
        return

    if hasattr(nn, "RMSNorm"):
        return

    class _RMSNorm(nn.Module):
        def __init__(
            self,
            normalized_shape: int | Tuple[int, ...],
            eps: float = 1e-6,
            elementwise_affine: bool = True,
        ) -> None:
            super().__init__()
            if isinstance(normalized_shape, numbers.Integral):
                normalized_shape = (int(normalized_shape),)
            else:
                normalized_shape = tuple(normalized_shape)
            self.normalized_shape = normalized_shape
            self.eps = eps
            self.elementwise_affine = elementwise_affine
            if elementwise_affine:
                self.weight = nn.Parameter(torch.ones(*self.normalized_shape))
            else:
                self.register_parameter("weight", None)

        def forward(self, input: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
            dims = tuple(range(-len(self.normalized_shape), 0))
            rms = torch.rsqrt(input.pow(2).mean(dim=dims, keepdim=True) + self.eps)
            output = input * rms
            if self.elementwise_affine:
                output = output * self.weight
            return output

    nn.RMSNorm = _RMSNorm  # type: ignore[attr-defined]
