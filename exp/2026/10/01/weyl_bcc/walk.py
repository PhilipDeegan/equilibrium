"""Real-form Dirac / Weyl rule on our BCC lattice (DampedLattice cells, hex distance 1): 8 real quantities per cell,
two Weyl halves L, R of 4 each (two pairs per half; a pair = what a complex amplitude was, i = quarter turn J).

    L(x, t+1) = n sum_h A_h L(x + h) + m J R(x)
    R(x, t+1) = m J L(x) + n sum_h A_h^T R(x - h)

h over the 8 hex diagonals, A_h the real 4x4 hops of real_form.py (45 deg pair turns x 0/+-1 patterns, weight
1/(2 sqrt2)), n^2 + m^2 = 1. m = 0: the two halves are independent Weyl rules of opposite handedness.
Missing neighbours (domain edge) contribute nothing; an optional quadratic sponge damps the outer shell.
"""

import numpy as np

from equilibrium.substrate import continuous as C

_ns = {}
_src = open(__file__.replace("walk.py", "symbol.py")).read()
exec(_src[: _src.index("rng = np.random")].replace("Path(__file__)", "Path('.')"), _ns)
HS, AS = _ns["transitions"](+1)
W_of = _ns["W_of"]


def real(Mc):
    n = Mc.shape[0]
    R = np.zeros((2 * n, 2 * n))
    for i in range(n):
        for j in range(n):
            z = Mc[i, j]
            R[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = [[z.real, -z.imag], [z.imag, z.real]]
    return R


J4 = real(1j * np.eye(2))
A_REAL = np.array([real(A) for A in AS])
# lattice face index k (0..7) of each walk hop h (D_HAT[k] = h / sqrt3)
K_OF_H = np.array([int(np.argmin(np.linalg.norm(C.D_HAT[:8] * np.sqrt(3) - h, axis=1))) for h in HS])
K_OF_MINUS_H = np.array([int(np.argmin(np.linalg.norm(C.D_HAT[:8] * np.sqrt(3) + h, axis=1))) for h in HS])


def to_real(psi_c):
    """(N, 2) complex -> (N, 4) real (a1, b1, a2, b2)."""
    return np.stack([psi_c[:, 0].real, psi_c[:, 0].imag, psi_c[:, 1].real, psi_c[:, 1].imag], -1)


class DiracWalk:
    def __init__(self, lat, m=0.0, sponge=0.25, damp=0.05):
        self.lat, self.m, self.n = lat, m, np.sqrt(1 - m * m)
        self.N = lat.N
        nb = lat.nb[:, :8]
        self.valid = nb >= 0
        self.nb = np.where(self.valid, nb, 0)
        self.L = np.zeros((self.N, 4))
        self.R = np.zeros((self.N, 4))
        r = np.linalg.norm(lat.pos, axis=1)
        r_in = (1 - sponge) * r.max()
        s = np.clip((r - r_in) / (r.max() - r_in), 0, 1)
        self.f = (1 - damp * s**2)[:, None]
        self.r_inner = r_in

    def _hop(self, psi, ks, mats):
        out = np.zeros_like(psi)
        for k, A in zip(ks, mats):
            v = self.valid[:, k][:, None]
            out += np.where(v, psi[self.nb[:, k]], 0.0) @ A.T
        return out

    def step(self, n_steps=1):
        for _ in range(n_steps):
            Lh = self._hop(self.L, K_OF_H, A_REAL)
            Rh = self._hop(self.R, K_OF_MINUS_H, A_REAL.transpose(0, 2, 1)) if self.m or self.R.any() else 0.0 * self.R
            L1 = self.n * Lh + self.m * self.R @ J4.T
            R1 = self.m * self.L @ J4.T + self.n * Rh
            self.L, self.R = L1 * self.f, R1 * self.f

    def density(self):
        return np.sum(self.L**2, 1) + np.sum(self.R**2, 1)


def symbol_dirac(k_ours, m):
    """4x4 complex symbol in our units (hex distance 1): k_walk = k_ours / sqrt3 ... phases e^{i k.h_ours}, h_ours = h/sqrt3."""
    n = np.sqrt(1 - m * m)
    kw = np.asarray(k_ours) / np.sqrt(3)
    Wk = W_of(kw[None], HS, AS)[0]
    I = np.eye(2)
    return np.block([[n * Wk, 1j * m * I], [1j * m * I, n * Wk.conj().T]])
