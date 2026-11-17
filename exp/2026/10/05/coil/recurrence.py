"""Recurrence to the seed: rms(rho(t) - rho(t0)) over the structure region of t0 (cells within a + 4 xi of the t0
centre circle), after shifting t to the same zc and the best rotation about z (phi scanned over one pattern period
2 pi / N). Also the same at phi = 0 and the structure's own rms(rho - 1) for scale."""
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

name, N, a, xi, t0 = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
ts = [int(x) for x in sys.argv[6:]]
d = Path("/bk/dst/2026/10/05/coil") / name
pos = np.load(d / "pos.npy")
tree = cKDTree(pos)
x, y, z = pos.T
rcyl = np.hypot(x, y)
inner = np.linalg.norm(pos, axis=1) < 0.75 * np.linalg.norm(pos, axis=1).max()


def load(tt):
    s = np.load(d / f"state_t{tt:05d}.npz")
    rho = np.sum(s["L"] ** 2, 1) + np.sum(s["R"] ** 2, 1)
    c = (rho < 0.25) & inner
    w = 1 - rho[c]
    return rho, np.sum(w * rcyl[c]) / w.sum(), np.sum(w * z[c]) / w.sum()


r0, rc0, zc0 = load(t0)
reg = (np.hypot(rcyl - rc0, z - zc0) < a + 4 * xi) & inner
P = pos[reg]
scale = np.sqrt(np.mean((r0[reg] - 1) ** 2))
phis = np.linspace(0, 2 * np.pi / N, 91, endpoint=False)
print(f"reference t {t0}: structure rms(rho - 1) {scale:.3f}")
for tt in ts:
    r1, rc1, zc1 = load(tt)
    best = (9, 0)
    for ph in phis:
        c, s_ = np.cos(ph), np.sin(ph)
        Q = np.stack([c * P[:, 0] - s_ * P[:, 1], s_ * P[:, 0] + c * P[:, 1], P[:, 2] - zc0 + zc1], 1)
        e = np.sqrt(np.mean((r1[tree.query(Q)[1]] - r0[reg]) ** 2))
        if ph == 0:
            e0 = e
        best = min(best, (e, ph))
    print(f"  t {tt}: rms vs t {t0} at best rotation {best[0]:.3f} (phi {np.degrees(best[1]):5.1f} deg), at phi 0 {e0:.3f}; "
          f"torus radius {rc1:.1f} (ref {rc0:.1f})", flush=True)
