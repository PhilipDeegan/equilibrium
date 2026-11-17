"""Background response around a wound structure (Philip 10-05: is the interaction with the background the Coulomb
mechanism, not an imposed u gradient?). Per shell about the structure centre (0, 0, zc): l = 0 parts of the density
deficit 1 - rho and of the phase relative to the background clock (phi = arg(a e^{i mu t}), mu = om0 + g rho0), and
the l = 1 (cos theta about z) part of phi. A Coulomb-like signal = an l = 0 profile that is a power law in r, fixed in
absolute r (same for R 110 and R 160, i.e. not set by the sponge radius)."""
import argparse
from pathlib import Path

import numpy as np

from equilibrium.substrate.walk import healing_g, rest_state, to_complex

p = argparse.ArgumentParser()
p.add_argument("--dir", action="append", required=True)
p.add_argument("--t", type=int, action="append", required=True)
p.add_argument("--m", type=float, default=0.8)
p.add_argument("--xi", type=float, default=3.9)
p.add_argument("--r_min", type=float, default=45.0)
p.add_argument("--dr", type=float, default=5.0)
p.add_argument("--r_max", type=float, default=150.0)
p.add_argument("--R", type=float, action="append", required=True, help="domain radius per --dir (sponge from 0.75 R)")
args = p.parse_args()
v, om0 = rest_state(args.m)
g = healing_g(args.m, args.xi, 1.0, walk=True)
mu = om0 + g
for d, Rdom in zip(map(Path, args.dir), args.R):
    pos = np.load(d / "pos.npy")
    rr0 = np.linalg.norm(pos, axis=1)
    for t in args.t:
        f = d / f"state_t{t:05d}.npz"
        if not f.exists():
            continue
        ck = np.load(f)
        L, R = ck["L"], ck["R"]
        a = np.concatenate([to_complex(L), to_complex(R)], 1) @ v.conj() * np.exp(1j * mu * t)
        rho = np.sum(L**2, 1) + np.sum(R**2, 1)
        del L, R, ck
        core = (rho < 0.25) & (rr0 < 0.75 * Rdom)
        wt = (1 - rho)[core]
        zc = np.sum(wt * pos[core, 2]) / wt.sum()
        rel = pos - [0, 0, zc]
        r = np.linalg.norm(rel, axis=1)
        ct = rel[:, 2] / np.maximum(r, 1e-9)
        print(f"{d.name.split('_')[1]} a{d.name.split('_a')[1].split('_')[0]} R{Rdom:g} t {t}: zc {zc:+.1f}; sponge from "
              f"{0.75 * Rdom:.0f}", flush=True)
        print("    r     <1-rho>     <phi>   phi_l1(cos)   N_cells", flush=True)
        for r0 in np.arange(args.r_min, min(args.r_max, Rdom), args.dr):
            sh = np.abs(r - r0) < args.dr / 2
            if sh.sum() < 100:
                continue
            ph0 = np.angle(np.mean(a[sh]))
            ph = np.angle(a[sh] * np.exp(-1j * ph0)) + ph0
            c1 = 3 * np.mean(ph * ct[sh])
            print(f"  {r0:5.0f}  {np.mean(1 - rho[sh]):+.3e}  {ph0:+.4f}  {c1:+.4f}  {sh.sum()}", flush=True)
