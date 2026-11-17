"""10-16 relay: DEC Maxwell on the BCC Delaunay complex (E = edge integrals on the 7 canonical hops per cell, B on the
12 Delaunay triangles per cell, stars as continuous.dec_star 'dec') written as a relay: one exact rotation per
(edge, triangle) incidence -- the edge hands sin(theta) of its content to the triangle and back. theta = dt |K|.
Strang-symmetric product over the 36 incidence types per cell. Bloch dispersion vs theta."""
import numpy as np

from equilibrium.substrate import continuous as C

D, MIR, W = C.D_HAT, C.K_MIRROR, C.W
CAN = [4, 5, 6, 7, 9, 11, 13]
r, s2 = C.dec_star("dec")
ac = 2 / np.sqrt(3)
star1 = np.array([r * ac / 8 if k < 8 else ac / 8 for k in CAN])
star2 = s2 / ac


def edge(p, k):
    if k in CAN:
        return CAN.index(k), p, 1.0
    return CAN.index(MIR[k]), p + D[k], -1.0


INC = []
for t, (s, k, m) in enumerate(C._DEC_TRIS):
    for (p, kk, sg) in ((np.zeros(3), k, 1.0), (D[k], m, 1.0), (np.zeros(3), s, -1.0)):
        c, pc, se = edge(p, kk)
        INC.append((t, c, pc, sg * se))
NT, NE = len(C._DEC_TRIS), len(CAN)


def Cq(q):
    M = np.zeros((NT, NE), complex)
    for t, c, pc, sg in INC:
        M[t, c] += sg * np.exp(1j * q @ pc)
    return M


def Kq(q):
    return np.sqrt(star2) * Cq(q) / np.sqrt(star1)[None, :]


def floquet(q, dt):
    n = NE + NT
    U = np.eye(n, dtype=complex)
    seq = []
    for t, c, pc, sg in INC:
        k = np.sqrt(star2) * sg * np.exp(1j * q @ pc) / np.sqrt(star1[c])
        seq.append((c, NE + t, k))
    for order in (seq, seq[::-1]):
        for c, b, k in order:
            th = 0.5 * dt * abs(k)
            u = k / abs(k)
            G = np.eye(n, dtype=complex)
            G[c, c] = G[b, b] = np.cos(th)
            G[c, b] = np.conj(u) * np.sin(th)
            G[b, c] = -u * np.sin(th)
            U = G @ U
    return U


if __name__ == "__main__":
    dirs = {"100": np.array([1, 0, 0.]), "111": np.ones(3) / np.sqrt(3), "110": np.array([1, 1, 0.]) / np.sqrt(2), "123": np.array([1, 2, 3.]) / np.sqrt(14)}
    kmax = np.sqrt(star2 / star1.min())
    print(f"stars r {r} s2 {s2}: *1 {np.round(star1, 4)}, *2 {star2:.4f}; |K| max {kmax:.4f}")
    qm = 0.05
    for nm, dv in dirs.items():
        w2 = np.sort(np.linalg.eigvalsh(Kq(qm * dv).conj().T @ Kq(qm * dv)))
        print(f"  DEC field form {nm}: omega/q {' '.join(f'{np.sqrt(max(v, 0)) / qm:.4f}' for v in w2[:3])}  gapped omega {' '.join(f'{np.sqrt(v):.3f}' for v in w2[3:])}")
    for thmax in (0.05, 0.3, 0.8, 1.2, np.pi / 2):
        dt = thmax / kmax
        print(f"theta_max {thmax:.3f} (dt {dt:.4f})")
        for nm, dv in dirs.items():
            om = np.angle(np.linalg.eigvals(floquet(qm * dv, dt)))
            om = np.sort(np.abs(om))
            low = om[om < 0.2 * dt * kmax + 1e-12] if False else om
            nzero = np.sum(om < 1e-9)
            small = np.sort(om[(om > 1e-9)])[:6]
            print(f"  {nm}: zero {nzero}  lowest omega/(q dt) {' '.join(f'{v / (qm * dt):.4f}' for v in small)}")
