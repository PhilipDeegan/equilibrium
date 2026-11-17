"""Density vs distance d to the vortex core line (cells with rho < rho_core; periodic box via tiling the core set)
for a springbox state; saves (d, <rho>) for --profile seeding in spring_box.py."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

p = argparse.ArgumentParser()
p.add_argument("--dir", required=True)
p.add_argument("--P", type=int, required=True)
p.add_argument("--t", type=int, action="append", required=True)
p.add_argument("--rho_core", type=float, default=0.02)
p.add_argument("--dr", type=float, default=0.5)
p.add_argument("--d_max", type=float, default=40.0)
p.add_argument("--save", default="")
args = p.parse_args()
d_ = Path(args.dir)
pos = np.load(d_ / "pos.npy")
side = args.P / np.sqrt(3)
for t in args.t:
    ck = np.load(d_ / f"state_t{t:05d}.npz")
    rho = np.sum(ck["L"]**2, 1) + np.sum(ck["R"]**2, 1)
    core = pos[rho < args.rho_core]
    tiles = np.concatenate([core + np.array([a, b, c]) * side for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)])
    d = cKDTree(tiles).query(pos)[0]
    bins = np.arange(0, args.d_max + args.dr, args.dr)
    k = np.digitize(d, bins) - 1
    ok = k < len(bins) - 1
    n = np.bincount(k[ok], minlength=len(bins) - 1)
    s = np.bincount(k[ok], rho[ok], minlength=len(bins) - 1)
    prof = s / np.maximum(n, 1)
    dc = bins[:-1] + args.dr / 2
    print(f"t {t}: core cells {len(core)}, min rho {rho.min():.4f}")
    for j in range(0, len(dc), 2):
        if n[j]:
            print(f"  d {dc[j]:5.2f}  <rho> {prof[j]:.4f}  seed-form d^2/(d^2+xi^2) {dc[j]**2 / (dc[j]**2 + 3.9**2):.4f}  n {n[j]}")
    if args.save:
        np.savez(args.save, d=dc[n > 0], rho=prof[n > 0])
