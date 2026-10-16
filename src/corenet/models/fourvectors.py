"""Four-vector helpers and the network non-linearity.

Tensors are laid out ``(batch, 4, objects)`` with the four components either in
``(pt, eta, phi, m)`` or in ``(px, py, pz, E)`` representation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def asinh(x):
    """Numerically stable asinh (written out so that it exports to every backend)."""
    xsign = x.sign()
    xunsigned = x * xsign
    # x + sqrt(x^2 + 1) is unstable for very negative x, so work with |x| and restore the sign
    loggand = xunsigned + (xunsigned.pow(2) + 1).sqrt()
    return torch.log(loggand) * xsign


def sinh(x):
    return (torch.exp(x) - torch.exp(-x)) / 2.0


def PxPyPzE(v):
    """(pt, eta, phi, m) -> (px, py, pz, E)."""
    pt = v[:, 0:1]
    eta = v[:, 1:2]
    phi = v[:, 2:3]
    m = v[:, 3:4]
    Px, Py, Pz = pt * phi.cos(), pt * phi.sin(), pt * sinh(eta)
    E = (pt**2 + Pz**2 + m**2).sqrt()
    return torch.cat((Px, Py, Pz, E), 1)


def PtEtaPhiM(v):
    """(px, py, pz, E) -> (pt, eta, phi, m)."""
    px = v[:, 0:1]
    py = v[:, 1:2]
    pz = v[:, 2:3]
    e = v[:, 3:4]
    Pt = (px**2 + py**2).sqrt()
    ysign = py.sign()
    ysign = (
        ysign + (ysign == 0.0).float()
    )  # py == 0: protect the sign so that phi = acos(+-1)
    Phi = (px / (Pt + 0.00001)).acos() * ysign
    Eta = asinh(pz / (Pt + 0.00001))
    M = F.relu(e**2 - px**2 - py**2 - pz**2).sqrt()
    return torch.cat((Pt, Eta, Phi, M), 1)


def addFourVectors(v1, v2, v1PxPyPzE=None, v2PxPyPzE=None):
    """Sum of two (pt, eta, phi, m) four-vectors; returns the sum in (pt, eta, phi, m) and (px, py, pz, E)."""
    if v1PxPyPzE is None:
        v1PxPyPzE = PxPyPzE(v1)
    if v2PxPyPzE is None:
        v2PxPyPzE = PxPyPzE(v2)
    v12PxPyPzE = v1PxPyPzE + v2PxPyPzE
    v12 = PtEtaPhiM(v12PxPyPzE)
    return v12, v12PxPyPzE


def diObjectMass(v1PxPyPzE, v2PxPyPzE):
    """Invariant mass of the sum of two (px, py, pz, E) four-vectors."""
    v12PxPyPzE = v1PxPyPzE + v2PxPyPzE
    # rounding can make the argument slightly negative: clip at zero before the square root
    M = F.relu(
        v12PxPyPzE[:, 3:4] ** 2
        - v12PxPyPzE[:, 0:1] ** 2
        - v12PxPyPzE[:, 1:2] ** 2
        - v12PxPyPzE[:, 2:3] ** 2
    ).sqrt()
    return M


def NonLU(x):
    """The network non-linearity: SiLU, x * sigmoid(x) (https://arxiv.org/abs/1702.03118)."""
    return x * torch.sigmoid(x)


__all__ = ["NonLU", "PtEtaPhiM", "PxPyPzE", "addFourVectors", "diObjectMass"]
