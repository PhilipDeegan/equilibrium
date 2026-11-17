"""Mutual induction on the EXPERIMENTAL A-E model (continuous.AE).

Loop 1 (radius a1, z=-d/2) carries J = phi_hat I1(t) gaussian tube (width s,
int over cross-section = I1); I1 ramps 0 -> 1 smoothly over T_r and holds.
Loop 2 (radius a2, z=+d/2): holonomy Phi2 = loop integral of A, measured as
2 pi a2 <A . phi_hat> over a gaussian tube around the circle.
Quasi-static prediction (A = int J / (4 pi c^2 r), i.e. mu0 = 1/c^2):
Phi2 = M I1, M = mu0 sqrt(a1 a2) [(2/k - k) K(k^2) - (2/k) E(k^2)],
k^2 = 4 a1 a2 / ((a1 + a2)^2 + d^2). Filament formula; tube width s adds
O(s^2/d^2). Window closes before the wall reflection returns.
plain step() has no J and conserves every holonomy: Phi2 stays 0.
"""
import argparse
import time

import numpy as np
from scipy.special import ellipe, ellipk

from equilibrium.substrate import continuous as C

C.set_weights("legacy")  # written for W = 3:1 on both legs (pre-2026-09-28; see continuous.WEIGHTS)
from equilibrium.substrate.lattice import SubstrateLattice

p = argparse.ArgumentParser()
p.add_argument("--h", type=float, default=0.5)
p.add_argument("--R", type=float, default=12.0)
p.add_argument("--a1", type=float, default=2.0)
p.add_argument("--a2", type=float, default=2.0)
p.add_argument("--d", type=float, default=2.0)
p.add_argument("--s", type=float, default=0.4)
p.add_argument("--Tr", type=float, default=2.5)
p.add_argument("--every", type=int, default=20)
p.add_argument("--project", action="store_true")
args = p.parse_args()

c2 = C.C0SQ
c = C.C0
lat = SubstrateLattice(radius_cells=args.R, spacing=args.h)
x = lat.pos
rho = np.hypot(x[:, 0], x[:, 1])
phat = np.stack([-x[:, 1], x[:, 0], 0 * rho], 1) / np.maximum(rho, 1e-12)[:, None]


def tube(p_, a, z0, s):
    rh = np.hypot(p_[:, 0], p_[:, 1])
    return np.exp(-0.5 * ((rh - a) ** 2 + (p_[:, 2] - z0) ** 2) / s**2)


def J_fn(p_):
    rh = np.maximum(np.hypot(p_[:, 0], p_[:, 1]), 1e-12)
    g = tube(p_, args.a1, -args.d / 2, args.s) / (2 * np.pi * args.s**2)
    return np.stack([-p_[:, 1] / rh * g, p_[:, 0] / rh * g, 0 * g], 1)


Jf = C.sample_A(lat, J_fn)
if args.project:
    Jf = C.project_div_free(lat, Jf)
w2 = tube(x, args.a2, +args.d / 2, args.s)
w1 = tube(x, args.a1, -args.d / 2, args.s)


def holonomy(A, wt, a):
    return 2 * np.pi * a * float(((A * phat).sum(1) * wt).sum() / wt.sum())


k2 = 4 * args.a1 * args.a2 / ((args.a1 + args.a2) ** 2 + args.d**2)
k = np.sqrt(k2)
M = (1 / c2) * np.sqrt(args.a1 * args.a2) * ((2 / k - k) * ellipk(k2) - (2 / k) * ellipe(k2))

t_ref = (2 * args.R - args.R * 0.0 - (args.a2 + args.d)) / c
T = min(t_ref, args.Tr + 3.0)
ae = C.AE(lat)
dt = C.stable_dt(lat)
n = int(np.ceil(T / dt))
dt = T / n
print(f"h={args.h} N={lat.N} M_theory={M:.5e} T={T:.2f} (reflection ~{t_ref:.2f}) n={n}", flush=True)


def I1(t):
    if t >= args.Tr:
        return 1.0
    return 0.5 * (1 - np.cos(np.pi * t / args.Tr))


rec = []
t0 = time.time()
for i in range(1, n + 1):
    t = (i - 1) * dt
    ae.step(dt, J_faces=I1(t) * Jf)
    if i % args.every == 0 or i == n:
        A = C.reconstruct_A(lat)
        ph2 = holonomy(A, w2, args.a2)
        ph1 = holonomy(A, w1, args.a1)
        rec.append((i * dt, I1(i * dt), ph2, ph1))
        if i % (args.every * 10) == 0 or i == n:
            print(f"  t={i*dt:.2f} I1={I1(i*dt):.3f} Phi2/(M I1)={ph2/(M*max(I1(i*dt),1e-9)):.4f}"
                  f" Phi1={ph1:.4e} |u|max={np.abs(lat.u).max():.2e} ({time.time()-t0:.0f}s)", flush=True)
rec = np.array(rec)
hold = rec[:, 0] > args.Tr + 0.5
print(f"\nhold window t>{args.Tr+0.5:.1f}: <Phi2>/M = {rec[hold, 2].mean()/M:.4f}"
      f"  (rms osc {rec[hold, 2].std()/M:.4f});  <Phi1> = {rec[hold, 3].mean():.4e}")
tag = f"induction_h{args.h}_R{args.R:g}{'_proj' if args.project else ''}"
np.save(tag + ".npy", rec)
np.savez(tag + "_final.npz", pos=lat.pos, u=lat.u, phi=lat.phi, e=ae.e)
