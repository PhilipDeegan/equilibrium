"""Quasi-energy and branch split of the walk in a periodic box (exp/2026/10/06/shed/quasi_energy.py).
Two-tick map with parity U(k) = S_par(k) S(k), S_par = B S(-k) B; U is unitary with doubly degenerate eigenvalues
e^{-+i Om}, so H2 = i log U = (Om / sin Om) i (U - U^dag) / 2, cos Om = Re tr U / 4. psi (N, 4) complex on cells at grid
index ix (N, 3) in [0, P) (zero off the BCC sites)."""
import numpy as np

from equilibrium.substrate.walk import AS, HS


def symbols(kx, KY, KZ, m):
    """S(k), S_par(k) on a (ky, kz) slab, k in rad / grid unit."""
    n = np.sqrt(1 - m * m)
    K = np.stack([np.full_like(KY, kx), KY, KZ], -1)
    W = np.einsum("...h,hij->...ij", np.exp(1j * K @ HS.T), AS)
    Wm = np.einsum("...h,hij->...ij", np.exp(-1j * K @ HS.T), AS)
    sh = KY.shape
    S = np.zeros(sh + (4, 4), complex)
    Sp = np.zeros(sh + (4, 4), complex)
    S[..., :2, :2], S[..., 2:, 2:] = n * W, n * np.conj(np.swapaxes(W, -1, -2))
    Sp[..., :2, :2], Sp[..., 2:, 2:] = n * Wm, n * np.conj(np.swapaxes(Wm, -1, -2))
    I2 = np.eye(2)
    for M in (S, Sp):
        M[..., :2, 2:] = M[..., 2:, :2] = 1j * m * I2
    Sp = Sp[..., [2, 3, 0, 1], :][..., :, [2, 3, 0, 1]]
    return S, Sp


def _slabs(psi, ix, P, m):
    F = np.zeros((4, P, P, P), complex)
    for c in range(4):
        F[c][ix[:, 0], ix[:, 1], ix[:, 2]] = psi[:, c]
    F = np.fft.fftn(F, axes=(1, 2, 3))
    k = 2 * np.pi * np.fft.fftfreq(P)
    KY, KZ = np.meshgrid(k, k, indexing="ij")
    for i, kx in enumerate(k):
        S, Sp = symbols(kx, KY, KZ, m)
        U = Sp @ S
        Om = np.arccos(np.clip(np.real(np.trace(U, axis1=-2, axis2=-1)) / 4, -1, 1))
        G = 0.5j * (U - np.conj(np.swapaxes(U, -1, -2)))
        yield F, i, Om, G


def quasi_energy(psi, ix, P, m):
    """E_q = sum_k psi_k^dag H2 psi_k / (2 P^3) (per tick)."""
    E = 0.0
    for F, i, Om, G in _slabs(psi, ix, P, m):
        f = F[:, i].transpose(1, 2, 0)
        E += np.real(np.sum(f.conj() * np.einsum("...ij,...j->...i", (Om / np.sin(Om))[..., None, None] * G, f)))
    return E / (2 * P**3)


def minus_branch(psi, ix, P, m):
    """P_- psi per cell, P_- = (1 - H2 / Om) / 2 per k (H2 / Om = G / sin Om). At Om ~ pi (U = -I) the branches coincide;
    those k are left out of P_- (a convention; their norm is what deg_norm reports)."""
    for F, i, Om, G in _slabs(psi, ix, P, m):
        sO = np.sin(Om)
        deg = sO < 1e-7
        Hn = (1 / np.where(deg, 1, sO))[..., None, None] * G
        f = F[:, i].transpose(1, 2, 0)
        fm = 0.5 * (f - np.einsum("...ij,...j->...i", Hn, f))
        fm[deg] = 0
        F[:, i] = fm.transpose(2, 0, 1)
    F = np.fft.ifftn(F, axes=(1, 2, 3))
    return np.stack([F[c][ix[:, 0], ix[:, 1], ix[:, 2]] for c in range(4)], 1)


def deg_norm(psi, ix, P, m):
    """norm at the k with Om ~ pi (no branch distinction there)."""
    nd = 0.0
    for F, i, Om, G in _slabs(psi, ix, P, m):
        nd += np.sum(np.abs(F[:, i].transpose(1, 2, 0)[np.sin(Om) < 1e-7])**2)
    return nd / P**3
