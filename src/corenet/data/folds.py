"""
The fold rule: a deterministic cross-validation fold from the event number.

    fold = squares(event) % n_folds        (``fold_of(event, n_folds, seed)``)

``squares`` is the Squares counter-based random number generator (Widynski, arXiv:2004.06278) keyed by
the strings ``"delphes_h5"``, ``"fold"`` and the integer ``seed``; the event number is the counter.
The fold of an event therefore depends only on its event number, the number of folds and the seed:
any subset of a sample, in any order, gets the same folds, and no random split is ever made.

This is the rule that wrote the ``EVENT/fold`` column of the example files, kept bit for bit so that
the package reproduces it: the shipped files ``examples/data/{hh4b,bbww}_{small,all}.h5`` were
written with ``n_folds=5`` and ``seed=0``, so ``--n-folds 5`` (``--fold-seed 0``) gives exactly the
folds stored in them.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Union

import numpy as np

_UINT64_32 = np.uint64(32)

# a runtime alias (not an annotation): typing.Union, so that the module imports on Python 3.9
SeedLike = Union[int, str, Iterable[Union[int, str]]]  # noqa: UP007


def _str_to_entropy(text: str) -> list[np.uint64]:
    return np.frombuffer(hashlib.md5(text.encode()).digest(), dtype=np.uint64).tolist()


def _seed(*entropy: SeedLike) -> tuple[int, ...]:
    seeds = []
    for e in entropy:
        if isinstance(e, str):
            seeds.extend(_str_to_entropy(e))
        elif isinstance(e, Iterable):
            seeds.extend(_seed(*e))
        else:
            seeds.append(e)
    return (*seeds,)


class Squares:
    """Squares: a counter-based random number generator (CBRNG), https://arxiv.org/abs/2004.06278.

    The key is drawn with ``numpy.random.PCG64`` seeded by the entropy of ``seed`` (strings enter
    through their MD5 digest, integers as they are).
    """

    def __init__(self, *seed: SeedLike):
        self._seed = _seed(seed)
        self._key_value: np.uint64 | None = None

    @property
    def _key(self) -> np.uint64:
        if self._key_value is None:
            self._key_value = self.key(np.random.Generator(np.random.PCG64(self._seed)))
        return self._key_value

    @staticmethod
    def key(gen: np.random.Generator) -> np.uint64:
        bits = np.arange(1, 16, dtype=np.uint64)
        offsets = np.arange(0, 29, 4, dtype=np.uint64)
        lower8 = gen.choice(bits, 8, replace=False)
        for i in range(16):
            if lower8[i] % 2 == 1:
                lower8 = np.roll(lower8, -i)
                break
        higher8 = np.zeros(8, dtype=np.uint64)
        higher8[0:1] = gen.choice(np.delete(bits, int(lower8[-1]) - 1), 1)
        higher8[1:] = gen.choice(np.delete(bits, int(higher8[0]) - 1), 7, replace=False)
        return np.sum(lower8 << offsets) + (np.sum(higher8 << offsets) << _UINT64_32)

    @staticmethod
    def _round(lr: np.ndarray, shift: np.ndarray, last: bool = False):
        lr *= lr
        lr += shift
        if last:
            yield lr.copy()
        low = lr >> _UINT64_32
        lr <<= _UINT64_32
        lr |= low
        yield lr

    def bit32(self, ctrs: np.ndarray) -> np.ndarray:
        x = ctrs * self._key
        y = x.copy()
        z = y + self._key
        # round 1-3
        for i in [y, z, y]:
            (_,) = self._round(x, i)
        # round 4
        x *= x
        x += z
        x >>= _UINT64_32
        return x.astype(np.uint32)

    def bit64(self, ctrs: np.ndarray) -> np.ndarray:
        x = ctrs * self._key
        y = x.copy()
        z = y + self._key
        # round 1-3
        for i in [y, z, y]:
            (_,) = self._round(x, i)
        # round 4
        (t, _) = self._round(x, z, last=True)
        # round 5
        x *= x
        x += y
        x >>= _UINT64_32
        x ^= t
        return x

    def uint64(self, counters) -> np.ndarray:
        """One uint64 per counter; a trailing dimension of several counters is reduced."""
        counters = np.asarray(counters, dtype=np.uint64)
        if counters.ndim == 1:
            return self.bit64(counters)
        while True:
            shape = counters.shape[-1]
            if shape == 1:
                return self.bit64(counters).reshape(counters.shape[:-1])
            elif shape % 2 == 0:
                counters = self.bit32(counters).view(np.uint64)
            else:
                counters = np.concatenate(
                    [
                        counters[..., -1:],
                        self.bit32(counters[..., :-1]).view(np.uint64),
                    ],
                    axis=-1,
                )


FOLD_SEED = 0  # the seed the example files were written with


def fold_rule(n_folds: int, seed: int = FOLD_SEED) -> str:
    """The rule in words, as the validator and the loader report it."""
    return f"squares(event) % {n_folds}, seed {seed}"


def fold_of(event, n_folds: int, seed: int = FOLD_SEED) -> np.ndarray:
    """fold = squares(event) % n_folds on the (non-negative, integer) event numbers; int64 (N,)."""
    if not isinstance(n_folds, int) or isinstance(n_folds, bool) or n_folds < 1:
        raise ValueError(f"n_folds must be an integer >= 1, got {n_folds!r}")
    event = np.asarray(event)
    if event.dtype.kind not in "iu":
        raise ValueError(
            f"event numbers must be integers to compute fold = squares(event) % n_folds, got dtype {event.dtype}"
        )
    if (event.astype(np.int64) < 0).any():
        raise ValueError(
            f"{int((event.astype(np.int64) < 0).sum()):,} event numbers are negative (or exceed int64): "
            "fold = squares(event) % n_folds needs event numbers >= 0"
        )
    rng = Squares("delphes_h5", "fold", int(seed))
    return (rng.uint64(event.astype(np.uint64)) % np.uint64(n_folds)).astype(np.int64)


__all__ = ["FOLD_SEED", "Squares", "fold_of", "fold_rule"]
