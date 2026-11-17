"""C++ walk vs the 10-01 numpy ring (gp.py: r0 12, xi 2.6, m 0.5, R40, z0 +14, 400 ticks: travels -z at 0.076/tick,
radius 12 +- 0.6, winding 1). Also ticks/s vs cells."""

import argparse
import time

import numpy as np

from equilibrium.substrate.walk import CppWalk, cpp_sphere, healing_g, sponge_factor

p = argparse.ArgumentParser()
p.add_argument("--R", type=float, default=40.0)
p.add_argument("--s", type=float, default=1.0, help="scale r0, xi, z0 and ticks (GP time ~ s^2)")
p.add_argument("--threads", type=int, default=4)
args = p.parse_args()
s = args.s
r0, xi, z0, T = 12 * s, 2.6 * s, 14 * s, int(400 * s * s)
t0 = time.time()
alat = cpp_sphere(args.R * s, n_threads=args.threads)
pos = np.array(alat[0].leaf_pos)
f, r_in = sponge_factor(pos, sponge=0.25, damp=0.05)
w = CppWalk(alat, 0, m=0.5, g=healing_g(0.5, xi), f=f)
x, y, z = pos.T
rho = np.hypot(x, y)
d1, d2 = np.hypot(rho - r0, z - z0), np.hypot(rho + r0, z - z0)
ph = np.arctan2(z - z0, rho - r0) - np.arctan2(z - z0, rho + r0)
w.seed(d1 / np.sqrt(d1**2 + xi**2) * d2 / np.sqrt(d2**2 + xi**2) * np.exp(1j * ph))
print(f"N {len(pos)} cells, build+seed {time.time() - t0:.1f}s, g {w.g:.4f}", flush=True)
inner = np.linalg.norm(pos, axis=1) < r_in


def track():
    d = w.density()
    core = (d < 0.25) & inner & (rho > 0.5 * r0)
    wt = (1 - d)[core]
    return np.sum(wt * z[core]) / wt.sum(), np.sum(wt * rho[core]) / wt.sum(), core.sum()


rec = [(0, *track())]
t1 = time.time()
for k in range(1, 9):
    w.step(T // 8)
    rec.append((w.t, *track()))
dt = time.time() - t1
rec = np.array(rec)
for r in rec:
    print(f"  t {int(r[0])}: z_c {r[1]:+.2f} radius {r[2]:.2f} core {int(r[3])}")
v = np.polyfit(rec[2:, 0], rec[2:, 1], 1)[0]
print(f"speed {v:+.4f} cells/tick ({v / s * s * s / s:+.4f} in s = 1 units x s); {T / dt:.1f} ticks/s, "
      f"{len(pos) * T / dt / 1e6:.1f} M cell-ticks/s at {args.threads} threads")
