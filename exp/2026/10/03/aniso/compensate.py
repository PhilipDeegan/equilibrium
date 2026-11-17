"""isotropy of walk variants on the sphere of k directions: spread (max - min) of the upper frequency per tick at fixed
|k|, and its 2-fold part about each axis (om(k) vs om(R_axis^90 k)). Variants (one 'cycle' = product, freq / ticks):
  X        single walk sign +1
  YX       naive alternation sign -1 after sign +1
  inv      X(-k) X(k): second tick with all hops reversed (lattice inversion)
  rotz     X(Rz k) X(k): second tick with hops turned 90 deg about z
  rot4z    X(Rz^3 k) X(Rz^2 k) X(Rz k) X(k)
"""
import numpy as np
from equilibrium.substrate import walk as W

m = 0.8
n = np.sqrt(1 - m * m)
hs, As = W.transitions(+1)
hs2, As2 = W.transitions(-1)


def S(k, sign=+1):
    h, A = (hs, As) if sign > 0 else (hs2, As2)
    Wk = np.einsum("h,hij->ij", np.exp(1j * (np.asarray(k) / np.sqrt(3)) @ h.T), A)
    I = np.eye(2)
    return np.block([[n * Wk, 1j * m * I], [1j * m * I, n * Wk.conj().T]])


def R(axis):
    c = {"x": [[1, 0, 0], [0, 0, -1], [0, 1, 0]], "y": [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], "z": [[0, -1, 0], [1, 0, 0], [0, 0, 1]]}
    return np.array(c[axis], float)


Rz = R("z")
V = {
    "X": (lambda k: S(k), 1),
    "YX": (lambda k: S(k, -1) @ S(k), 2),
    "inv": (lambda k: S(-k) @ S(k), 2),
    "rotz": (lambda k: S(Rz @ k) @ S(k), 2),
    "rot4z": (lambda k: S(Rz @ Rz @ Rz @ k) @ S(Rz @ Rz @ k) @ S(Rz @ k) @ S(k), 4),
}


def om(U, ticks):
    return np.sort(np.abs(np.angle(np.linalg.eigvals(U))))[[0, -1]] / ticks  # lowest / highest |freq|


rng = np.random.default_rng(1)
dirs = rng.normal(size=(400, 3))
dirs /= np.linalg.norm(dirs, axis=1)[:, None]
for kk in [0.2, 0.5]:
    print(f"|k| = {kk}  (om0 = arcsin m = {np.arcsin(m):.6f})")
    for name, (f, t) in V.items():
        w = np.array([om(f(kk * d), t) for d in dirs])
        two = {a: max(np.abs(om(f(kk * (R(a) @ d)), t) - om(f(kk * d), t)).max() for d in dirs[:150]) for a in "xyz"}
        print(f"  {name:6s} upper spread {np.ptp(w[:, 1]):.2e} lower spread {np.ptp(w[:, 0]):.2e}; mean upper "
              f"{w[:, 1].mean():.6f}; 90-deg asym x {two['x']:.1e} y {two['y']:.1e} z {two['z']:.1e}")
