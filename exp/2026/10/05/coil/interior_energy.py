"""Inductance check (Ampere: toroid interior field ~ N, stored energy ~ N^2 at fixed current): gradient energy of the
k = 0 amplitude, sum over hex-neighbour pairs |a_j - a_i|^2 / 2, background-subtracted per cell, split by distance d to
the tube centre circle (rc, zc): interior d < a - 2 xi (inside the winding surface, off the strand cores), band
|d - a| < 2 xi (strand cores), outside a + 2 xi .. a + 6 xi. Also the mean toroidal phase gradient in the interior
times rho (= N if the interior flow is the Ampere one), and E_tor = sum over interior pairs |a_i a_j| (grad ds)^2 / 2
(the energy of that toroidal flow alone)."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from equilibrium.substrate.walk import rest_state, to_complex

p = argparse.ArgumentParser()
p.add_argument("--dir", action="append", required=True)
p.add_argument("--t", type=int, action="append", required=True)
p.add_argument("--m", type=float, default=0.8)
p.add_argument("--a", type=float, required=True)
p.add_argument("--xi", type=float, default=3.9)
p.add_argument("--r_inner", type=float, required=True)
args = p.parse_args()
v, _ = rest_state(args.m)
a_, xi = args.a, args.xi
pos = None
for d in map(Path, args.dir):
    if pos is None:
        pos = np.load(d / "pos.npy")
        pairs = cKDTree(pos).query_pairs(1.0 + 1e-6, output_type="ndarray")
        rho_c = np.hypot(pos[:, 0], pos[:, 1])
        th_c = np.arctan2(pos[:, 1], pos[:, 0])
        rr = np.linalg.norm(pos, axis=1)
        dvec = pos[pairs[:, 1]] - pos[pairs[:, 0]]
    for t in args.t:
        f = d / f"state_t{t:05d}.npz"
        if not f.exists():
            continue
        ck = np.load(f)
        L, R = ck["L"], ck["R"]
        a = np.concatenate([to_complex(L), to_complex(R)], 1) @ v.conj()
        rho = np.sum(L**2, 1) + np.sum(R**2, 1)
        del L, R, ck
        core = (rho < 0.25) & (rr < args.r_inner)
        wt = (1 - rho)[core]
        rc, zc = np.sum(wt * rho_c[core]) / wt.sum(), np.sum(wt * pos[core, 2]) / wt.sum()
        dt = np.hypot(rho_c - rc, pos[:, 2] - zc)
        e = np.abs(a[pairs[:, 1]] - a[pairs[:, 0]]) ** 2 / 2
        ec = np.zeros(len(pos))
        np.add.at(ec, pairs[:, 0], e / 2)
        np.add.at(ec, pairs[:, 1], e / 2)
        bgm = (dt > a_ + 10 * xi) & (rr < args.r_inner - 5)
        eb = ec[bgm].mean()
        regs = {"interior": dt < a_ - 2 * xi, "band": np.abs(dt - a_) < 2 * xi,
                "outside": (dt >= a_ + 2 * xi) & (dt < a_ + 6 * xi)}
        E = {k: np.sum(ec[m] - eb) for k, m in regs.items()}
        # toroidal phase gradient in the interior: pairs with both ends interior, projected on the toroidal direction
        ins = regs["interior"]
        pm = ins[pairs[:, 0]] & ins[pairs[:, 1]]
        dph = np.angle(a[pairs[pm, 1]] / a[pairs[pm, 0]])
        th = th_c[pairs[pm, 0]]
        tdir = np.stack([-np.sin(th), np.cos(th), 0 * th], 1)
        ds = np.einsum("ij,ij->i", dvec[pm], tdir)
        sel = np.abs(ds) > 0.3
        grad = np.sum(dph[sel] * ds[sel]) / np.sum(ds[sel] ** 2)
        n_eff = grad * np.mean(rho_c[pairs[pm, 0]])
        E_tor = np.sum(np.abs(a[pairs[pm, 0]] * a[pairs[pm, 1]]) * (grad * ds) ** 2 / 2)
        print(f"{d.name.split('_')[1]} t {t}: rc {rc:.1f} zc {zc:+.1f}  E interior {E['interior']:.1f} "
              f"({ins.sum()} cells)  band {E['band']:.1f}  outside {E['outside']:.1f}  bg/cell {eb:.2e}  "
              f"interior toroidal gradient x rho = {n_eff:+.2f}, E_tor {E_tor:.1f}", flush=True)
