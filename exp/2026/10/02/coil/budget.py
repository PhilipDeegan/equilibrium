"""Density budget of a saved spring run: per snapshot, with the torus centre (rc, zc) from the core cells (rho < 0.25):
deficit D_near = sum (rho0 - rho) over cells within d_tube < a + 4 xi of the centre circle (the structure),
D_core = same over d_tube < a + 1.5 xi, min rho, far-field rms(rho - rho0) beyond d_tube > a + 8 xi (inside the
sponge radius), and sum(rho - rho0) over the whole inner domain (what has left the structure but not the domain)."""
import sys
from pathlib import Path
import numpy as np

d = Path(sys.argv[1])
a, xi, R = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
pos = np.load(d / "pos.npy")
x, y, z = pos.T
rho_c = np.hypot(x, y)
r = np.linalg.norm(pos, axis=1)
r_in = 0.75 * r.max()
inner = r < r_in
print(f"{d.name}: N {len(pos)}")
print(" t      rc     zc    D_near   D_core  min_rho  far_rms   sum(rho-1)_inner  sum(rho-1)_all")
for f in sorted(d.glob("state_t*.npz")):
    s = np.load(f)
    rho = np.sum(s["L"] ** 2, 1) + np.sum(s["R"] ** 2, 1)
    core = (rho < 0.25) & inner
    if not core.any():
        print(f"{int(s['t']):6d} no core"); continue
    wt = (1 - rho)[core]
    rc, zc = np.sum(wt * rho_c[core]) / wt.sum(), np.sum(wt * z[core]) / wt.sum()
    dt = np.hypot(rho_c - rc, z - zc)
    near, cr, far = (dt < a + 4 * xi) & inner, (dt < a + 1.5 * xi) & inner, (dt > a + 8 * xi) & inner
    print(f"{int(s['t']):6d} {rc:6.1f} {zc:+6.1f} {np.sum(1 - rho[near]):8.0f} {np.sum(1 - rho[cr]):8.0f} {rho[inner].min():7.4f} "
          f"{np.sqrt(np.mean((rho[far] - 1) ** 2)):8.4f} {np.sum(rho[inner] - 1):12.0f} {np.sum(rho - 1):12.0f}", flush=True)
