"""azimuth-averaged toroidal flow of a ring in the periodic box: phase gradient of the dominant rest component
a = psi . v1* (grad = sum_h h dphi_h / 8, per hex distance) at every cell; toroidal momentum sum rho v_phi in shells
of distance d from the ring centre circle (rho_c, z_c from the core cells), in units of xi; whole-box angular momentum
L_z = sum rho (x v_y - y v_x) and total sum rho v (box translation). usage: toroidal_tube.py state.npz P xi"""
import sys
from pathlib import Path

import numpy as np

from equilibrium.substrate.walk import HS, rest_state, to_complex

f, P, xi = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
v1, _ = rest_state(0.8)
pos = np.load(f.parent / "pos.npy", mmap_mode="r")
N = len(pos)
ix = np.rint(np.asarray(pos) * np.sqrt(3)).astype(np.int32)
H = P // 2
lut = np.full((P,) * 3, -1, np.int32)
lut[ix[:, 0] + H, ix[:, 1] + H, ix[:, 2] + H] = np.arange(N, dtype=np.int32)
ck = np.load(f)
t = int(ck["t"])
L = ck["L"]
a = to_complex(L) @ v1[:2].conj()
rho = (L**2).sum(1)
del L
R = ck["R"]
a += to_complex(R) @ v1[2:].conj()
rho += (R**2).sum(1)
del R, ck
g = np.zeros((N, 3))
for h in HS:
    j = lut[(ix[:, 0] + int(h[0]) + H) % P, (ix[:, 1] + int(h[1]) + H) % P, (ix[:, 2] + int(h[2]) + H) % P]
    g += np.angle(a[j] * a.conj())[:, None] * h[None]
    del j
del lut
g *= np.sqrt(3) / 8
p = np.asarray(pos)
core = rho < 0.25
rc = np.hypot(p[core, 0], p[core, 1]).mean()
zc = p[core, 2].mean()
rr = np.hypot(p[:, 0], p[:, 1])
phx, phy = -p[:, 1] / np.maximum(rr, 1e-9), p[:, 0] / np.maximum(rr, 1e-9)
vphi = g[:, 0] * phx + g[:, 1] * phy
d = np.hypot(rr - rc, p[:, 2] - zc) / xi
Lz = np.sum(rho * (p[:, 0] * g[:, 1] - p[:, 1] * g[:, 0]))
print(f"{f.parent.name[:44]} t {t}: ring (rho, z) ({rc:.1f}, {zc:+.1f}); L_z (box) {Lz:+.3e}; sum rho v "
      f"({np.sum(rho * g[:, 0]):+.2e}, {np.sum(rho * g[:, 1]):+.2e}, {np.sum(rho * g[:, 2]):+.2e})", flush=True)
for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 8)):
    s = (d >= lo) & (d < hi)
    print(f"   shell {lo}-{hi} xi: cells {s.sum():8d}  sum rho v_phi {np.sum(rho[s] * vphi[s]):+.3e}  "
          f"mean v_phi {np.sum(rho[s] * vphi[s]) / np.sum(rho[s]):+.3e}  rms v_phi {np.sqrt(np.mean(vphi[s]**2)):.2e}", flush=True)
