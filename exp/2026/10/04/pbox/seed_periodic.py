"""can a vortex loop be seeded seam-free in a periodic box? phase = q/2 sum over images of the solid angle (cube of
(2n+1)^3 images), then remove the mean jump across each face by a linear term. Reports the residual phase mismatch
across the faces (mod 2 pi, rms and max) vs n, for a ring of radius a in a box of side P (integer grid units)."""
import numpy as np

P, a = 48, 8.0
APEX = np.array([0.3183, -0.2718, 0.5772])
s = np.linspace(0, 2 * np.pi, 96, endpoint=False)
ring = np.stack([a * np.cos(s), a * np.sin(s), 0 * s], 1)


def omega(Pts, X):
    A, B = Pts, np.roll(Pts, -1, 0)
    out = np.zeros(len(X))
    for i in range(0, len(X), 2000):
        x_ = X[i:i + 2000]
        aa = APEX[None, None] - x_[:, None]
        b = A[None] - x_[:, None]
        c = B[None] - x_[:, None]
        la, lb, lc = (np.linalg.norm(v, axis=2) for v in (aa, b, c))
        num = np.einsum("nmk,nmk->nm", aa, np.cross(b, c))
        den = la * lb * lc + np.einsum("nmk,nmk->nm", aa, b) * lc + np.einsum("nmk,nmk->nm", aa, c) * lb + np.einsum("nmk,nmk->nm", b, c) * la
        out[i:i + 2000] = 2 * np.arctan2(num, den).sum(1)
    return out


g = np.arange(-P // 2, P // 2)
# face sample points: x on the low face and its image across the box
yy, zz = np.meshgrid(g[::3], g[::3], indexing="ij")
lo = np.stack([np.full(yy.size, -P / 2), yy.ravel(), zz.ravel()], 1).astype(float)
for n in [0, 1, 2]:
    sh = np.array([(i, j, k) for i in range(-n, n + 1) for j in range(-n, n + 1) for k in range(-n, n + 1)]) * P
    res = {}
    for ax in range(3):
        L = np.roll(lo, ax, 1)
        H = L.copy(); H[:, ax] += P
        ph = lambda X: 0.5 * sum(omega(ring + d, X) for d in sh)
        jump = np.angle(np.exp(1j * (ph(H) - ph(L))))
        res[ax] = jump
    msg = []
    for ax in range(3):
        j = res[ax]
        msg.append(f"ax{ax}: mean {j.mean():+.3f} rms-after-mean {np.std(j):.3f} max {np.abs(j - j.mean()).max():.3f}")
    print(f"images n={n} ({len(sh)}):  " + "; ".join(msg), flush=True)
