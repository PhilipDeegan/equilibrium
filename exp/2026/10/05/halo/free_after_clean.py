"""One cleanup, then free evolution: seed psi = a v from a rotating-frame window mean (avg_c*.npy, radiation averaged
out once), evolve --T ticks with no further reseeding, every --every ticks: far-region gradient energy (as in
scf_coil.py), ring radius, minor radius, core cells, windings. Re-growth of E_far toward the uncleaned level (~350
at P 300) = the coil re-sheds (not on its attractor); staying low = near equilibrium. States every --state_every."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from equilibrium.substrate.walk import CppWalk, cpp_box, healing_g, to_complex, to_real

p = argparse.ArgumentParser()
p.add_argument("--start", required=True)
p.add_argument("--t0", type=int, required=True)
p.add_argument("--P", type=int, default=300)
p.add_argument("--N", type=int, default=4)
p.add_argument("--a", type=float, default=10.0)
p.add_argument("--m", type=float, default=0.8)
p.add_argument("--xi", type=float, default=3.9)
p.add_argument("--T", type=int, default=8000)
p.add_argument("--every", type=int, default=500)
p.add_argument("--state_every", type=int, default=2000)
p.add_argument("--threads", type=int, default=8)
p.add_argument("--out", default="/bk/dst/2026/10/05/scf_free")
args = p.parse_args()

g = healing_g(args.m, args.xi, 1.0, walk=True)
alat = cpp_box(args.P, n_threads=args.threads)
w = CppWalk(alat, 0, m=args.m, g=g, rho0=1.0, parity=True)
v = w.v
side = args.P / np.sqrt(3)
pos = np.array(alat[0].leaf_pos)
x, y, z = pos.T
rho_c, th_c = np.hypot(x, y), np.arctan2(y, x)
lo = pos.min(0)
tree = cKDTree(pos - lo, boxsize=side)
pairs = tree.query_pairs(1.0 + 1e-6, output_type="ndarray")
ps = np.linspace(0, 2 * np.pi, 256, endpoint=False)
out = Path(args.out) / f"{Path(args.start).parent.name}_{Path(args.start).stem}"
out.mkdir(parents=True, exist_ok=True)
np.save(out / "pos.npy", pos)
psi0 = np.load(args.start)[:, None] * v[None, :]
w.set_state(to_real(psi0[:, :2]), to_real(psi0[:, 2:]), t=args.t0)
del psi0


def winding(a, P_):
    idx = tree.query(np.mod(P_ - lo, side))[1]
    return np.sum(np.angle(a[idx][np.r_[1:len(idx), 0]] / a[idx])) / (2 * np.pi)


for k in range(args.T // args.every + 1):
    L, R = w.state()
    psi = np.concatenate([to_complex(L), to_complex(R)], 1)
    a = psi @ v.conj()
    rho = np.sum(np.abs(psi) ** 2, 1)
    core = rho < 0.25
    wt = (1 - rho)[core]
    rc = np.sum(wt * rho_c[core]) / wt.sum()
    zc = side / (2 * np.pi) * np.angle(np.sum(wt * np.exp(2j * np.pi * z[core] / side)))
    dz = np.abs(np.mod(z - zc + side / 2, side) - side / 2)
    dt = np.hypot(rho_c - rc, dz)
    far = dt > args.a + 8 * args.xi
    e = np.abs(a[pairs[:, 1]] - a[pairs[:, 0]]) ** 2 / 2
    Ef = np.sum(e[far[pairs[:, 0]] & far[pairs[:, 1]]])
    mr = np.sum(wt * dt[core]) / wt.sum()
    rm = args.a + 3 * args.xi
    wc = winding(a, np.stack([rc * np.cos(ps), rc * np.sin(ps), zc + 0 * ps], 1))
    wm = winding(a, np.stack([rc + rm * np.cos(ps), 0 * ps, zc + rm * np.sin(ps)], 1))
    print(f"t {w.t}: E_far {Ef:.1f}  rc {rc:.1f} zc {zc:+.1f} minor {mr:.1f} core {core.sum()} windings {wc:+.2f} / "
          f"{wm:+.2f}", flush=True)
    if (w.t - args.t0) % args.state_every == 0:
        np.savez(out / f"state_t{w.t:05d}.npz", t=w.t, L=L, R=R, z_shift=0.0)
    if k < args.T // args.every:
        w.step(args.every)
