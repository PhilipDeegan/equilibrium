"""10-11 tick: on the tick-phase ring IC (phi = W wrap(dtheta)), is the gradient (divergence) part of the exchange-matrix
commutator (11/lie) the same as tick.py's clock shift q (squared offsets, a Bernoulli |v|^2 / 2)? Compares per-cell
divergences sum_k W_k X_ik of X = lie(phi) and X = W_k (q_j - q_i), away from and near the defect core."""
import sys

import numpy as np

sys.argv = [sys.argv[0], "--P", "120", "--kappa", "0", "--T", "0", "--threads", "4"]
from equilibrium.substrate import continuous as C
from equilibrium.substrate.walk import cpp_box, solid_angle

C.spacing_of = lambda lat: 1.0
P, R0 = 120, 15.0
alat = cpp_box(P, n_threads=4)
ac = C.AnchorCluster(alat)
pos, nb, N = ac.pos, ac.nb, ac.N
side = P / np.sqrt(3)
W, D = C.W, C.D_HAT
TWO_PI = 2 * np.pi
wrap = lambda a: (a + np.pi) % TWO_PI - np.pi
wrapx = lambda x: (x + side / 2) % side - side / 2
s_ = np.linspace(0, TWO_PI, 400, endpoint=False)
ring = np.stack([R0 * np.cos(s_), R0 * np.sin(s_), 0 * s_], 1)
shifts = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)], float) * side
phase = lambda X: 0.5 * solid_angle(ring, X, shifts=shifts, n_threads=4)
rng = np.random.default_rng(0)
jumps = []
for ax in range(3):
    smp = (rng.random((400, 3)) - 0.5) * side
    smp[:, ax] = -side / 2
    hi = smp.copy()
    hi[:, ax] += side
    jumps.append(np.angle(np.mean(np.exp(1j * (phase(hi) - phase(smp))))))
th = (phase(pos) - sum(jumps[a] * pos[:, a] / side for a in range(3))) % TWO_PI
off = wrap(th[nb] - th[:, None])
phi = W[None, :] * off

key = lambda v: tuple(np.round(v, 6))
hop1 = {key(D[a]) for a in range(14)}
paths = {}
for p_ in range(14):
    for q in range(14):
        s = D[p_] + D[q]
        if np.linalg.norm(s) > 1e-9 and key(s) not in hop1:
            paths.setdefault(key(s), []).append((p_, q))
KEYS = list(paths)
KIDX = {k: n for n, k in enumerate(KEYS)}
MA = [[(b, KIDX[key(D[k] + D[b])]) for b in range(14) if key(D[k] + D[b]) in KIDX] for k in range(14)]
AM = [[(a, KIDX[key(D[k] - D[a])]) for a in range(14) if key(D[k] - D[a]) in KIDX] for k in range(14)]


def lie(ph):
    a, v = ph / W, ph * W
    M2 = np.empty((N, len(KEYS)))
    for n, k_ in enumerate(KEYS):
        M2[:, n] = sum(a[:, p_] + a[nb[:, p_], q] for p_, q in paths[k_]) / len(paths[k_])
    T = np.zeros_like(ph)
    for k in range(14):
        j = nb[:, k]
        for b, n in MA[k]:
            T[:, k] -= M2[:, n] * v[j, b]
        for a_, n in AM[k]:
            T[:, k] -= v[:, a_] * M2[nb[:, a_], n]
    return W * T


L = lie(phi)
q = (off**2 * W[None, :]**2).sum(1) / (2 * np.sum(W**2))
G = W[None, :] * (q[nb] - q[:, None])
dL, dG = (W * L).sum(1), (W * G).sum(1)
x = wrapx(pos)
dist = np.hypot(np.hypot(x[:, 0], x[:, 1]) - R0, x[:, 2])
print(f"N {N}; |L| {np.abs(L).max():.3e}; |div-free part| / |L| {np.linalg.norm(C.project_div_free(ac, L)) / np.linalg.norm(L):.3f}")
for lo, hi in [(0, 1.5), (1.5, 3), (3, 6), (6, 12), (12, 99)]:
    m = (dist >= lo) & (dist < hi)
    c = np.sum(dL[m] * dG[m]) / np.sqrt(np.sum(dL[m]**2) * np.sum(dG[m]**2))
    print(f"  dist from core [{lo:4.1f},{hi:4.1f}) cells {m.sum():7d}: corr(div lie, div grad q) {c:+.3f} "
          f"scale {np.sum(dL[m] * dG[m]) / np.sum(dG[m]**2):+.3f}  rms div lie {np.sqrt(np.mean(dL[m]**2)):.2e}")
