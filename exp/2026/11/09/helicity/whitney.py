"""Helicity on the complex itself: H = sum over BCC Delaunay tets of int W(a) . W(da) (Whitney 1-form of the edge
integrals a = phi / W, Whitney 2-form of the triangle loop sums da). Sees all 7 edge DOF per cell.
Gauge-exact (W(df) = d W(f), periodic -> sum int dWf . dWa = 0).
Tets: square edge i -> i + d_s (s = +x / +y / +z) and the two other-sublattice vertices p, q (common hex neighbours of
both, square neighbours of each other); each tet has two square edges, owned by the smaller axis -> 6 per cell."""
import numpy as np

from equilibrium.substrate import continuous as C

D = C.D_HAT
HEX = range(8)
SQ = {0: 9, 1: 11, 2: 13}
EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
FACES = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
QA, QB = 0.5854101966249685, 0.1381966011250105
QP = np.array([[QA if j == i else QB for j in range(4)] for i in range(4)])


def kof(v):
    m = np.nonzero(np.all(np.abs(D - v) < 1e-9, axis=1))[0]
    return int(m[0]) if len(m) else -1


def templates():
    out = []
    for ax, s in SQ.items():
        ks = [k for k in HEX if kof(D[s] - D[k]) >= 0]
        for p in ks:
            for q in ks:
                if q <= p:
                    continue
                sq = kof(D[q] - D[p])
                if sq < 0:
                    continue
                ax2 = int(np.argmax(np.abs(D[sq])))
                if ax2 < ax:
                    continue
                out.append((s, p, q))
    return out


def whitney_matrix(X):
    """M[e, f] = int_tet W_e . W_f for vertex positions X (4, 3)."""
    T = np.array([X[1] - X[0], X[2] - X[0], X[3] - X[0]])
    vol = abs(np.linalg.det(T)) / 6
    G = np.linalg.inv(T)
    g = np.vstack([-G.sum(1), G.T])
    M = np.zeros((6, 4))
    for lam in QP:
        We = [lam[i] * g[j] - lam[j] * g[i] for i, j in EDGES]
        Wf = [2 * (lam[i] * np.cross(g[j], g[k]) + lam[j] * np.cross(g[k], g[i]) + lam[k] * np.cross(g[i], g[j]))
              for i, j, k in FACES]
        M += np.array([[e @ f for f in Wf] for e in We]) * vol / 4
    return M, vol, g


class Whitney:
    def __init__(self, ac):
        self.ac = ac
        nb = ac.nb
        self.parts = []
        for s, p, q in templates():
            X = np.array([np.zeros(3), D[s], D[p], D[q]])
            M, vol, _ = whitney_matrix(X)
            v = [np.arange(ac.N), nb[:, s], nb[:, p], nb[:, q]]
            eidx = [(v[i], kof(X[j] - X[i])) for i, j in EDGES]
            assert all(k >= 0 for _, k in eidx)
            self.parts.append((M, vol, eidx))
        self.vol = sum(vol for _, vol, _ in self.parts) * ac.N

    def edges(self, phi, part):
        a = np.where(self.ac.nb >= 0, phi, 0.0) / C.W[None, :]
        return np.stack([a[c, k] for c, k in part[2]], 1)

    def density(self, phi, psi=None):
        """Per-cell (tets owned by the cell) int W(a_phi) . W(da_psi); psi defaults to phi."""
        psi = phi if psi is None else psi
        h = np.zeros(self.ac.N)
        for part in self.parts:
            ae = self.edges(phi, part)
            b = self.edges(psi, part)
            da = np.stack([b[:, EDGES.index((i, j))] + b[:, EDGES.index((j, k))] - b[:, EDGES.index((i, k))]
                           for i, j, k in FACES], 1)
            h += np.einsum("ne,ef,nf->n", ae, part[0], da)
        return h

    def H(self, phi, psi=None):
        return self.density(phi, psi).sum()
