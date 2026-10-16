"""Numerically stable accumulation of mean and variance over several batches
(https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance); used to freeze the input
normalisation of the embedders before training."""

from __future__ import annotations

from collections import namedtuple
from typing import ClassVar

import torch

_VarianceData = namedtuple("_VarianceBase", ["sumw", "sumw2", "m1", "M2"])


class TensorVariance:
    _t = torch.float32
    _k: ClassVar[dict] = {"dim": 0, "keepdim": True, "dtype": _t}

    def __init__(self, data: torch.Tensor | None = None):
        self._raw = None if data is None else _VarianceData(*self.compute(data))

    @classmethod
    @torch.no_grad()
    def compute(cls, data: torch.Tensor):
        sumw = torch.tensor(len(data), dtype=cls._t, device=data.device)
        sumw2 = sumw.clone()
        m1 = data.mean(**cls._k)
        M2 = (data - m1).pow(2).sum(**cls._k)
        return sumw, sumw2, m1, M2

    @property
    def mean(self) -> torch.Tensor:
        return self._raw.m1

    @property
    def variance(self) -> torch.Tensor:
        return self._raw.M2 / self._raw.sumw

    @property
    def variance_unbiased(self) -> torch.Tensor:
        return self.variance / (1 - self._raw.sumw2 / (self._raw.sumw * self._raw.sumw))

    def __add__(self, other: TensorVariance) -> TensorVariance:
        if not isinstance(other, TensorVariance):
            return NotImplemented
        new = TensorVariance()
        if self._raw is None:
            new._raw = other._raw
        elif other._raw is None:
            new._raw = self._raw
        else:
            sumw = self._raw.sumw + other._raw.sumw
            sumw2 = self._raw.sumw2 + other._raw.sumw2
            delta = other._raw.m1 - self._raw.m1
            m1 = self._raw.m1 + delta * other._raw.sumw / sumw
            M2 = (
                self._raw.M2
                + other._raw.M2
                + delta**2 * self._raw.sumw * other._raw.sumw / sumw
            )
            new._raw = _VarianceData(sumw, sumw2, m1, M2)
        return new


__all__ = ["TensorVariance"]
