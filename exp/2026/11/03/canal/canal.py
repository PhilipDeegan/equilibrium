import argparse
import time
from pathlib import Path

import numpy as np

from equilibrium.substrate import continuous as C

p = argparse.ArgumentParser()
p.add_argument("--R", type=float, default=40.0)
p.add_argument("--D", type=float, default=24.0, help="A-B separation")
p.add_argument("--lam_bg", type=float, default=1.0)
p.add_argument("--lam_s", type=float, default=0.0)
p.add_argument("--w", type=float, default=-1.0, help="strip half-width (<0: no strip)")
p.add_argument("--s", type=float, default=1.5, help="pulse width")
p.add_argument("--T", type=float, default=80.0)
p.add_argument("--threads", type=int, default=8)
p.add_argument("--out", default="")
args = p.parse_args()

t0 = time.time()
ac = C.anchor_cluster(args.R, 1.0, n_threads=args.threads)
pos = ac.pos
A = np.array([-args.D / 2, 0.0, 0.0])
B = np.array([args.D / 2, 0.0, 0.0])
Boff = A + np.array([0.0, args.D, 0.0])
lam = np.full(ac.N, args.lam_bg)
if args.w >= 0:
    on = (np.abs(pos[:, 1]) <= args.w) & (np.abs(pos[:, 2]) <= args.w) & (pos[:, 0] >= A[0] - 2) & (pos[:, 0] <= B[0] + 2)
    lam[on] = args.lam_s
ac.cl.set_leaf_lam(lam)
r2 = np.sum((pos - A) ** 2, 1)
ac.u = np.exp(-r2 / (2 * args.s**2))
ac.phi = np.zeros((ac.N, 14))
iB = int(np.argmin(np.sum((pos - B) ** 2, 1)))
iO = int(np.argmin(np.sum((pos - Boff) ** 2, 1)))
dt = C.stable_dt(ac)
n = int(round(args.T / dt))
print(f"N={ac.N} dt={dt:.4f} steps={n} strip_cells={int((lam == args.lam_s).sum()) if args.w >= 0 else 0} "
      f"B={pos[iB]} Boff={pos[iO]} setup {time.time() - t0:.1f}s", flush=True)
t = np.arange(1, n + 1) * dt
uB, uO = np.empty(n), np.empty(n)
t1 = time.time()
for i in range(n):
    ac.alat.step(ac.i, dt)
    u = ac.cl.leaf_u
    uB[i], uO[i] = u[iB], u[iO]
    if i == 50:
        print(f"  eta {(time.time() - t1) / 51 * n:.0f}s", flush=True)


def arrivals(x):
    m = np.abs(x).max()
    return [float(t[np.argmax(np.abs(x) >= f * m)]) for f in (1e-3, 1e-2, 1e-1, 0.5)] + [float(t[np.argmax(np.abs(x))]), m]


aB, aO = arrivals(uB), arrivals(uO)
print("        t(1e-3)  t(1e-2)  t(0.1)  t(0.5)  t(peak)  peak")
print("B     " + "  ".join(f"{v:7.2f}" for v in aB[:-1]) + f"  {aB[-1]:.3e}")
print("Boff  " + "  ".join(f"{v:7.2f}" for v in aO[:-1]) + f"  {aO[-1]:.3e}")
if args.out:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, t=t, uB=uB, uO=uO, args=str(vars(args)))
print(f"done {time.time() - t0:.1f}s")
