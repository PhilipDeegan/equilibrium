"""Real-form two-component BCC walk with a local density turn (exp/2026/10/01/weyl_bcc, gp.py; 10-02 coils).

Per cell 8 reals: two halves L, R of two pairs each (a pair = (a, b); i = the quarter turn J: (a, b) -> (-b, a)).
One tick:

    L1(x) = n sum_h A_h L(x + h) + m J R(x)
    R1(x) = m J L(x) + n sum_h A_h^T R(x - h)
    clock (beta != 0): sigma = 1 + beta (rho0 - rho1) / rho0 (cell.L / L0 vs depletion); (L1, R1) turned by the mass coin
    through (sigma - 1) om0 (exact rest-frequency shift om0 -> sigma om0), density turn g -> g sigma; hop not scaled
    every pair of x turns by -g rho1(x),  rho1 = |L1|^2 + |R1|^2
    sponge: X = bg(t+1) + (X1 - bg(t+1)) f(x)

h over the 8 hex-diagonal neighbours, A_h the real 4x4 of the D'Ariano-Perinotti Weyl walk (45 deg pair turn x 0/+-1
pattern x 1/(2 sqrt2)), n^2 + m^2 = 1. Missing neighbours read the background bg(t) = a0 v turned by -mu t,
v = the k = 0 positive-frequency state, mu = arcsin m + g rho0 (uniform background stationary in density).

parity=True: odd ticks (t odd before the tick) use the walk's parity image, L and R exchange matrices (L1 = n sum_h
A_h^T L(x + h) + ..., R1 = ... + n sum_h A_h R(x - h)); symbol B S(-k) B, B = L / R swap. A 90 deg turn maps the
plain walk onto its mirror (sign -1), so it prefers axes (relative anisotropy ~ k); the alternation removes that
(full cubic symmetry, residual ~ k^2), same kinetic term, same rest state (exp/2026/10/03/aniso).

`Walk` is the numpy reference on any (pos, nb) lattice; `CppWalk` runs the same tick in C++ on a level-0
lattice.continuous.AdaptiveLattice cluster (AdaptiveLattice::step_walk), threaded, double-buffered.
"""

import numpy as np

_H = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)


def transitions(sign=+1):
    """walk hops h (8, 3), hex distance sqrt3, and complex 2x2 transition matrices (8, 2, 2)."""
    zeta = (1 + sign * 1j) / 4
    zc = np.conj(zeta)
    M = {
        1: (zc * np.array([[1, 0], [1, 0]]), zeta * np.array([[0, -1], [0, 1]])),
        2: (zc * np.array([[0, 1], [0, 1]]), zeta * np.array([[1, 0], [-1, 0]])),
        3: (zc * np.array([[0, -1], [0, 1]]), zeta * np.array([[1, 0], [1, 0]])),
        4: (zc * np.array([[1, 0], [-1, 0]]), zeta * np.array([[0, 1], [0, 1]])),
    }
    hs, As = [], []
    for l in range(1, 5):
        hs += [_H[l - 1], -_H[l - 1]]
        As += [M[l][0], M[l][1]]
    return np.array(hs), np.array(As, dtype=complex)


def real(Mc):
    """complex n x n -> real 2n x 2n acting on (a_1, b_1, a_2, b_2, ...)."""
    n = Mc.shape[0]
    R = np.zeros((2 * n, 2 * n))
    for i in range(n):
        for j in range(n):
            z = Mc[i, j]
            R[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = [[z.real, -z.imag], [z.imag, z.real]]
    return R


HS, AS = transitions(+1)
A_REAL = np.array([real(A) for A in AS])
J4 = real(1j * np.eye(2))


def to_real(psi_c):
    """(N, 2) complex -> (N, 4) real (a1, b1, a2, b2)."""
    return np.stack([psi_c[:, 0].real, psi_c[:, 0].imag, psi_c[:, 1].real, psi_c[:, 1].imag], -1)


def to_complex(X):
    """(N, 4) real -> (N, 2) complex."""
    return X[:, 0::2] + 1j * X[:, 1::2]


def symbol_dirac(k, m):
    """4x4 complex symbol, k in units of 1 / hex distance."""
    n = np.sqrt(1 - m * m)
    ph = np.exp(1j * (np.asarray(k) / np.sqrt(3)) @ HS.T)
    Wk = np.einsum("h,hij->ij", ph, AS)
    I = np.eye(2)
    return np.block([[n * Wk, 1j * m * I], [1j * m * I, n * Wk.conj().T]])


def rest_state(m):
    """k = 0 positive-frequency state v (4,) and its frequency om0 = arcsin m."""
    lam, V = np.linalg.eig(symbol_dirac(np.zeros(3), m))
    om = -np.angle(lam)
    i = int(np.argmax(om))
    return V[:, i], float(om[i])


def healing_g(m, xi, rho0=1.0, c2=1.0 / 3.0, walk=False):
    """g giving envelope core width xi (cells): xi = sqrt(c^2 / (2 m g rho0)) (continuum c^2 = 1/3, the label used by
    all runs up to 2026-10-02). walk=True uses the walk's own small-k coefficient, omega = arcsin m + n k^2 / (6 m)
    (exact expansion of cos omega = n d(k)), so the true core width is xi: g = n c^2 / (2 m rho0 xi^2); the old label
    is sqrt(n) xi_true."""
    n = np.sqrt(1 - m * m) if walk else 1.0
    return n * c2 / (2 * m * rho0 * xi**2)


def face_of_hops(pos, nb):
    """face index k of each walk hop h (8,) and of -h (8,), from the lattice's own geometry (an interior cell)."""
    full = np.where((nb[:, :8] >= 0).all(1))[0]
    if not len(full):
        raise ValueError("face_of_hops: no cell with all 8 hex neighbours")
    j = full[np.argmin(np.linalg.norm(pos[full], axis=1))]  # central: periodic boxes wrap at the edges
    d = pos[nb[j, :8]] - pos[j]
    d /= np.linalg.norm(d, axis=1)[:, None]
    hu = HS / np.sqrt(3)
    kh = np.array([int(np.argmax(d @ h)) for h in hu])
    kmh = np.array([int(np.argmax(-d @ h)) for h in hu])
    if not (np.allclose(d[kh], hu, atol=1e-9) and np.allclose(d[kmh], -hu, atol=1e-9)):
        raise ValueError("face_of_hops: lattice hex directions do not match the walk hops")
    return kh, kmh


def sponge_factor(pos, sponge=0.25, damp=0.05):
    """per-cell relaxation factor f = 1 - damp s^2, s = (r - r_in) / (r_max - r_in) in the outer shell (gp.py)."""
    r = np.linalg.norm(pos, axis=1)
    r_in = (1 - sponge) * r.max()
    s = np.clip((r - r_in) / (r.max() - r_in), 0, 1)
    return 1 - damp * s**2, r_in


# beta (local clock, set_walk_clock) unused outside TestWalkClock since 10-08 (walk = reified approximation); removal pending
class _Base:
    def __init__(self, pos, nb, m=0.5, g=0.0, rho0=1.0, f=None, parity=False, beta=0.0):
        self.pos, self.N = np.asarray(pos), len(pos)
        self.parity, self.beta = parity, beta
        self.m, self.n, self.g, self.rho0 = m, np.sqrt(1 - m * m), g, rho0
        self.v, self.om0 = rest_state(m)
        self.mu = self.om0 + g * rho0
        self.a0 = np.sqrt(rho0)
        self.kh, self.kmh = face_of_hops(self.pos, np.asarray(nb))
        self.f = None if f is None else np.asarray(f, dtype=float)

    def bg(self, t):
        z = self.a0 * self.v * np.exp(-1j * self.mu * t)
        return to_real(z[None, :2])[0], to_real(z[None, 2:])[0]

    def seed(self, chi):
        """L, R = chi(x) v (chi complex per cell, e.g. a0 x phase x core envelope), t = 0."""
        psi = np.asarray(chi)[:, None] * self.v[None, :]
        self.set_state(to_real(psi[:, :2]), to_real(psi[:, 2:]), t=0)

    def density(self):
        L, R = self.state()
        return np.sum(L**2, 1) + np.sum(R**2, 1)

    def amp(self):
        """complex projection on v (pair angle of the k = 0 branch)."""
        L, R = self.state()
        return np.concatenate([to_complex(L), to_complex(R)], 1) @ self.v.conj()


class Walk(_Base):
    """numpy reference. pos (N, 3), nb (N, >= 8) neighbour index per face, -1 = none."""

    def __init__(self, pos, nb, **kw):
        super().__init__(pos, nb, **kw)
        nb = np.asarray(nb)
        self.valid = nb[:, :8] >= 0
        self.nb = np.where(self.valid, nb[:, :8], 0)
        self.L = np.zeros((self.N, 4))
        self.R = np.zeros((self.N, 4))
        self.t = 0

    def state(self):
        return self.L, self.R

    def set_state(self, L, R, t=None):
        self.L, self.R = np.array(L, float), np.array(R, float)
        if t is not None:
            self.t = t

    def _hop(self, X, ks, mats, ghost):
        out = np.zeros_like(X)
        for k, A in zip(ks, mats):
            out += np.where(self.valid[:, k][:, None], X[self.nb[:, k]], ghost[None, :]) @ A.T
        return out

    def step(self, n_steps=1):
        for _ in range(n_steps):
            gL, gR = self.bg(self.t)
            AL, AR = A_REAL, A_REAL.transpose(0, 2, 1)
            if self.parity and self.t % 2:
                AL, AR = AR, AL
            Lh = self._hop(self.L, self.kh, AL, gL)
            Rh = self._hop(self.R, self.kmh, AR, gR)
            L1 = self.n * Lh + self.m * self.R @ J4.T
            R1 = self.m * self.L @ J4.T + self.n * Rh
            rho1 = np.sum(L1**2, 1) + np.sum(R1**2, 1)
            sig = 1.0
            if self.beta:
                sig = 1 + self.beta * (self.rho0 - rho1) / self.rho0
                ph = ((sig - 1) * self.om0)[:, None]
                L1, R1 = np.cos(ph) * L1 + np.sin(ph) * R1 @ J4.T, np.sin(ph) * L1 @ J4.T + np.cos(ph) * R1
            th = self.g * sig * rho1
            c, s = np.cos(th)[:, None], np.sin(th)[:, None]
            for P in (L1, R1):
                a, b = P[:, 0::2].copy(), P[:, 1::2].copy()
                P[:, 0::2] = a * c + b * s
                P[:, 1::2] = b * c - a * s
            self.t += 1
            if self.f is not None:
                gL, gR = self.bg(self.t)
                L1 = gL + (L1 - gL) * self.f[:, None]
                R1 = gR + (R1 - gR) * self.f[:, None]
            self.L, self.R = L1, R1


class CppWalk(_Base):
    """C++ tick (AdaptiveLattice::step_walk) on cluster i of a level-0 lattice.continuous.AdaptiveLattice."""

    def __init__(self, alat, i=0, **kw):
        self.alat, self.i = alat, i
        cl = alat[i]
        pos, nb = np.array(cl.leaf_pos), np.array(alat.leaf_nb_j(i))
        super().__init__(pos, nb, **kw)
        gL, gR = self.bg(0)
        alat.set_walk(i, np.ascontiguousarray(A_REAL), np.ascontiguousarray(A_REAL.transpose(0, 2, 1)),
                      self.kh.astype(np.int32), self.kmh.astype(np.int32), self.n, self.m, self.g,
                      np.ascontiguousarray(gL), np.ascontiguousarray(gR), self.mu)
        if self.f is not None:
            alat.set_walk_sponge(i, np.ascontiguousarray(self.f))
        alat.set_walk_parity(i, self.parity)
        if self.beta:
            alat.set_walk_clock(i, self.beta, self.om0, self.rho0)

    t = property(lambda s: s.alat.walk_t(s.i))

    def state(self):
        return np.array(self.alat.walk_L(self.i)), np.array(self.alat.walk_R(self.i))

    def set_state(self, L, R, t=None):
        self.alat.set_walk_state(self.i, np.ascontiguousarray(L, dtype=np.float64),
                                 np.ascontiguousarray(R, dtype=np.float64), -1 if t is None else int(t))

    def step(self, n_steps=1):
        self.alat.step_walk(self.i, n_steps)

    def density(self):
        return np.array(self.alat.walk_density(self.i))


def cpp_box(P, n_threads=8):
    """level-0 continuous AdaptiveLattice periodic BCC box, period P (even, grid units h = 1 / sqrt3 hex distances)
    per axis, side P h; rebuilt; returns alat."""
    from equilibrium import lattice as _lat

    alat = _lat.continuous.AdaptiveLattice()
    _lat.continuous.build_bcc_box_periodic(alat[0], P, P, P, d1=1.0)
    alat.rebuild(0)
    alat.set_n_threads(n_threads)
    return alat


def cpp_sphere(R, n_threads=8):
    """level-0 continuous AdaptiveLattice sphere of radius R cells (hex distance 1), rebuilt; returns alat."""
    from equilibrium import lattice as _lat

    alat = _lat.continuous.AdaptiveLattice()
    _lat.continuous.build_bcc_sphere(alat[0], R, d1=1.0, u0=0.0, lam0=0.0)
    alat.rebuild(0)
    alat.set_n_threads(n_threads)
    return alat


SEED_APEX = np.array([0.3183, -0.2718, 0.5772])


def solid_angle(P, X, apex=SEED_APEX, shifts=None, n_threads=8):
    """solid angle of the closed polygon P (M, 3) seen from each X (N, 3), summed over P + each shift (S, 3), cone to
    a generic apex (no lattice cell on the cone), mod 4 pi; C++ threaded (lattice.solid_angle)."""
    from equilibrium import lattice as _lat

    f = lambda a: np.ascontiguousarray(a, dtype=float)
    return _lat.continuous.solid_angle(f(P), f(X), f(apex), None if shifts is None else f(np.atleast_2d(shifts)), n_threads)
