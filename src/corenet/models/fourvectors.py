"""Four-vector helpers and the network non-linearity.

Tensors are laid out ``(batch, 4, objects)`` with the four components either in
``(pt, eta, phi, m)`` or in ``(px, py, pz, E)`` representation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

W_MASS = 80.379  # GeV, used by the ``lepton + met`` recipe


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
    ysign = ysign + (ysign == 0.0).float()  # py == 0: protect the sign so that phi = acos(+-1)
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
        v12PxPyPzE[:, 3:4] ** 2 - v12PxPyPzE[:, 0:1] ** 2 - v12PxPyPzE[:, 1:2] ** 2 - v12PxPyPzE[:, 2:3] ** 2
    ).sqrt()
    return M


def NonLU(x):
    """The network non-linearity: SiLU, x * sigmoid(x) (https://arxiv.org/abs/1702.03118)."""
    return x * torch.sigmoid(x)


# ----------------------------------------------------------------------------- leptonic W
def neutrino_pz(l_pt, l_eta, l_phi, l_mass, nu_px, nu_py, mW=W_MASS):
    """Solve the W-mass constraint for the neutrino pz.

    Returns ``(pz1, pz2, is_complex, off_shell_score)``. When the discriminant is negative both
    solutions are the real part and ``off_shell_score`` is the magnitude of the imaginary part
    (0 otherwise).
    """
    l_px = l_pt * torch.cos(l_phi)
    l_py = l_pt * torch.sin(l_phi)
    l_pz = l_pt * torch.sinh(l_eta)
    l_E = torch.sqrt(l_px**2 + l_py**2 + l_pz**2 + l_mass**2)

    met_pt_sq = nu_px**2 + nu_py**2
    A = (l_px * nu_px + l_py * nu_py) + (mW**2 - l_E**2 + l_px**2 + l_py**2 + l_pz**2) / 2
    C = l_E**2 - l_pz**2
    disc = (2 * A * l_pz) ** 2 - 4 * (l_E**2 * met_pt_sq - A**2) * C

    is_complex = disc < 0
    real_part = (2 * A * l_pz) / (2 * C + 1e-8)
    delta = torch.sqrt(torch.abs(disc) + 1e-8) / (2 * C + 1e-8)
    pz1 = torch.where(is_complex, real_part, real_part + delta)
    pz2 = torch.where(is_complex, real_part, real_part - delta)
    off_shell_score = torch.where(is_complex, delta, torch.zeros_like(delta))
    return pz1, pz2, is_complex, off_shell_score


def leptonic_w(l, nu, mW=W_MASS):
    """Two leptonic-W candidates from a lepton ``l`` (N, 4, P) in (pt, eta, phi, m) and the missing
    transverse momentum ``nu`` (N, 2, P) in (pt, phi).

    Returns ``(W1, W2, off_shell_score, pz1, pz2)`` with W1/W2 in (pt, eta, phi, m).
    """
    l_pt, l_eta, l_phi, l_mass = l[:, 0:1], l[:, 1:2], l[:, 2:3], l[:, 3:4]
    met_pt, met_phi = nu[:, 0:1], nu[:, 1:2]
    nu_px = met_pt * torch.cos(met_phi)
    nu_py = met_pt * torch.sin(met_phi)

    pz1, pz2, _, off_shell_score = neutrino_pz(l_pt, l_eta, l_phi, l_mass, nu_px, nu_py, mW=mW)

    l_px = l_pt * torch.cos(l_phi)
    l_py = l_pt * torch.sin(l_phi)
    l_pz = l_pt * torch.sinh(l_eta)
    l_energy = torch.sqrt(l_px**2 + l_py**2 + l_pz**2 + l_mass**2)

    def make_w(nu_pz):
        nu_energy = torch.sqrt(met_pt**2 + nu_pz**2)
        w_px = l_px + nu_px
        w_py = l_py + nu_py
        w_pz = l_pz + nu_pz
        w_energy = l_energy + nu_energy
        w_pt = torch.sqrt(w_px**2 + w_py**2)
        w_phi = torch.atan2(w_py, w_px)
        w_eta = torch.asinh(w_pz / (w_pt + 1e-8))
        w_mass_sq = w_energy**2 - w_px**2 - w_py**2 - w_pz**2
        w_mass = torch.sqrt(torch.clamp(w_mass_sq, min=0))
        return torch.cat([w_pt, w_eta, w_phi, w_mass], dim=1)

    return make_w(pz1), make_w(pz2), off_shell_score, pz1, pz2


__all__ = ["PxPyPzE", "PtEtaPhiM", "addFourVectors", "diObjectMass", "NonLU", "neutrino_pz", "leptonic_w", "W_MASS"]
