import sys

import numpy as np

f = np.load(sys.argv[1])
pos, b, zs = f["pos"], f["b"], float(f["zs"])
side = float(sys.argv[2])
wrapx = lambda x: (x + side / 2) % side - side / 2
from equilibrium.substrate import continuous as C

N = len(pos)
tri_area = np.array([0.5 * np.cross(C.D_HAT[k], C.D_HAT[s]) for s, k, m in C._DEC_TRIS])
A = tri_area / np.linalg.norm(tri_area, axis=1)[:, None]
bt = b.reshape(N, 12)
Bv = np.linalg.lstsq(np.linalg.norm(tri_area, axis=1)[:, None] * A, bt.T, rcond=None)[0].T
X = wrapx(pos - np.array([0.13, 0.07, zs]))
rho = np.hypot(X[:, 0], X[:, 1])
e = np.sum(Bv**2, 1)
ax = rho < 2.0
print("axis: dz, angle of B_perp (deg), |B|^2")
zz = []
for z0 in np.arange(-side / 2 + 2, side / 2 - 2, 1.0):
    m = ax & (np.abs(X[:, 2] - z0) < 0.5)
    if m.sum() == 0:
        continue
    Bm = Bv[m].mean(0)
    zz.append((z0, np.degrees(np.arctan2(Bm[1], Bm[0])), np.sum(Bm**2)))
zz = np.array(zz)
for row in zz[::3]:
    print(f"  {row[0]:+6.1f} {row[1]:+7.1f} {row[2]:.3e}")
behind = zz[(zz[:, 0] < -3) & (zz[:, 0] > -side / 2 + 6)]
ang = np.unwrap(np.radians(behind[:, 1]))
sl = np.polyfit(behind[:, 0], ang, 1)[0]
print(f"behind-axis twist d angle / dz {sl:+.4f} rad / cell -> pitch {2 * np.pi / abs(sl):.2f} cells, handedness {'+' if sl > 0 else '-'}")
print("transverse extent behind (energy-weighted rho) vs distance behind:")
for d in (5, 10, 15, 20):
    m = np.abs(X[:, 2] + d) < 1
    print(f"  behind {d:3d}: rho_rms {np.sqrt(np.sum(e[m] * rho[m]**2) / e[m].sum()):.2f}   ahead (dz +{d}) energy / behind {e[np.abs(X[:, 2] - d) < 1].sum() / e[m].sum():.2e}")
