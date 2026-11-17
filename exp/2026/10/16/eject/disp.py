"""10-16 disp: Bloch dispersion of the eject junction (14 face ports + internal stub y) on the BCC Kelvin cell.
One tick: b = S a (ports: b = V - a, stub: s' = V - s, V = 2 (sum w a + y s) / (1 + y)); port k leaves along d_k and
arrives at the neighbour on port mirror(k): a'_m = e^{-i q.d_k} b_k (complex only as the Fourier tool). Eigenvalues
e^{-i omega}. Long-wave branch slope = wave speed; branches with group velocity ~0 at small q = held branches."""
import numpy as np

from equilibrium.substrate import continuous as C

D, MIR = C.D_HAT, C.K_MIRROR
w = np.asarray(C.LAP_W, float)
w = w / w.sum()


def T(q, y):
    n = 15
    S = np.zeros((n, n))
    wy = np.r_[w, y]
    V = 2 * wy / (1 + y)
    S[:, :] = V[None, :]
    S -= np.eye(n)
    P = np.zeros((n, n), complex)
    for k in range(14):
        P[MIR[k], k] = np.exp(-1j * q @ D[k])
    P[14, 14] = 1
    return P @ S


a = 2 / np.sqrt(3)
for y in (0.0, 1.0, 3.0):
    print(f"y {y}: expected continuum speed ratio 1/sqrt(1+y) = {1 / np.sqrt(1 + y):.3f}")
    for name, dvec in (("[100]", np.array([1, 0, 0.])), ("[111]", np.ones(3) / np.sqrt(3)), ("[110]", np.array([1, 1, 0.]) / np.sqrt(2))):
        rows = []
        for qm in (0.02, 0.05, 0.1, 0.3, 0.6, 1.0):
            ev = np.linalg.eigvals(T(qm * dvec, y))
            om = np.sort(np.abs(np.angle(ev)))
            rows.append((qm, om))
        om_small = rows[0][1]
        nz = om_small[om_small > 1e-9]
        print(f"  {name}: omega at q 0.02 (all 15): {' '.join(f'{v:.3f}' for v in om_small)}")
        ac_ = [r[1][np.argmin(np.abs(r[1] - 0.52 * r[0] / np.sqrt(1 + y)))] for r in rows]
        print(f"         acoustic omega/q: {' '.join(f'q{r[0]}:{o / r[0]:.3f}' for r, o in zip(rows, ac_))}")
