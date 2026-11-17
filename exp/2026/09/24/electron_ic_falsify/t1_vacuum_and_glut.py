"""Vacuum sanity tests + bare-glut T1, R=35 BCC sphere, outflow BC, plain step().

Vacuum (no electron) -- does the bare field behave as expected?
  V0 const   : u = 0.1 everywhere, phi = 0. Expect: nothing changes (the BC must
               not inject anything into a uniform state).
  V1 pulse   : free Gaussian u bump (sigma=3) at origin. Expect: radiates out at
               c and leaves through the outflow BC; residual = reflection.
  V2 held    : small core (r<3) held at u0 (fixed_u), rest starts at 0. Expect
               (doc/dst.md): exterior settles toward u ~ 1/r with shell charge
               Q(R_s) independent of R_s -- Coulomb from a sourced Poisson state.
  V3 coulomb : same held core, but exterior preset to u0*a/r. Harmonic outside,
               so u should stay put; checks whether the outflow BC erodes a
               static 1/r tail (first-order Mur treats d/dr u as outgoing).
Electron:
  E1 glut    : uniformly-charged ball, u = u0*(1.5 - r^2/(2a^2)) for r<a,
               u0*a/r outside, a=25, NOT held. T1: does Q(t) on shells stay put
               or drain (core has lap(u) != 0, nothing holds it)?

Q(R_s) = -sum_{faces crossing R_s} W_k (u_out - u_in)  (Gauss flux of grad u).
"""

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/deegan/git/eq")

import numpy as np

from equilibrium import lattice as _lat
from run.substrate.adaptive.sweep_spheromak_decay_cross_scale import _compute_dt_l0
from tst.substrate.adaptive.adaptive_flux_tube import leaf_nb_j

R_DOMAIN = 35.0
W_FACE = np.array([3.0] * 8 + [1.0] * 6)
R_SHELLS = [5.0, 10.0, 15.0, 20.0, 27.0, 32.0]
U0 = 1.0
CORE_A = 3.0
GLUT_A = 25.0

p = argparse.ArgumentParser()
p.add_argument("--arm", choices=["V0", "V1", "V2", "V3", "E1"], required=True)
p.add_argument("--steps", type=int, default=6000)
p.add_argument("--report", type=int, default=250)
p.add_argument("--threads", type=int, default=8)
p.add_argument("--bc", choices=["outflow", "neumann", "none"], default="outflow")
args = p.parse_args()


def main():
    t0 = time.time()
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(args.threads)
    cl = alat[0]
    _lat.build_bcc_sphere(cl, R_DOMAIN, d1=1.0, u0=0.0, lam0=0.0)
    alat.rebuild(0)
    pos = np.array(cl.leaf_pos)
    r = np.linalg.norm(pos, axis=1)
    N = cl.n_leaves
    nbj = leaf_nb_j(cl)

    u = np.zeros(N)
    held = np.zeros(N, dtype=bool)
    if args.arm == "V0":
        u[:] = 0.1
    elif args.arm == "V1":
        u = U0 * np.exp(-(r**2) / (2 * 3.0**2))
    elif args.arm in ("V2", "V3"):
        held = r < CORE_A
        u[held] = U0
        if args.arm == "V3":
            u[~held] = U0 * CORE_A / r[~held]
    elif args.arm == "E1":
        inside = r < GLUT_A
        u[inside] = U0 * (1.5 - r[inside] ** 2 / (2 * GLUT_A**2))
        u[~inside] = U0 * GLUT_A / r[~inside]
    cl.set_leaf_u(u)
    if held.any():
        cl.set_leaf_fixed_u(held.astype(np.uint8))
    if args.bc == "outflow":
        cl.set_bc(_lat.BoundaryCondition.outflow(np.array(cl.leaf_nb)))
    elif args.bc == "neumann":
        cl.set_bc(_lat.BoundaryCondition.neumann())

    dt = _compute_dt_l0(cl, D1=1.0)
    ts = _lat.TimeStepper.by_level(dt)
    evolver = _lat.AdaptiveLatticeEvolver(alat)
    print(f"arm={args.arm} bc={args.bc} N={N} dt={dt:.5f} steps={args.steps} t_end={args.steps*dt:.1f}"
          f" held={held.sum()} build={time.time()-t0:.1f}s", flush=True)

    # precompute shell-crossing face lists
    shells = {}
    for R_s in R_SHELLS:
        ii, jj, ww = [], [], []
        for k in range(14):
            j = nbj[:, k]
            m = (r < R_s) & (j >= 0)
            m[m] &= r[j[m]] >= R_s
            i_idx = np.nonzero(m)[0]
            ii.append(i_idx); jj.append(j[i_idx]); ww.append(np.full(len(i_idx), W_FACE[k]))
        shells[R_s] = (np.concatenate(ii), np.concatenate(jj), np.concatenate(ww))
    r_bins = [0, 5, 10, 15, 20, 25, 30, 35.1]
    bin_idx = np.digitize(r, r_bins) - 1

    rec = []

    def log(step_i):
        u = np.asarray(cl.leaf_u)
        phi = np.asarray(cl.leaf_phi)
        eu = u**2
        ep = 0.5 * np.sum(phi**2, axis=0)
        E = float(eu.sum() + ep.sum())
        E_in = float((eu + ep)[r < 30].sum())
        Q = [float(-(w * (u[j] - u[i])).sum()) for i, j, w in shells.values()]
        u_prof = [float(u[bin_idx == b].mean()) for b in range(len(r_bins) - 1)]
        ep15 = float(ep[r < 15].sum())
        rec.append([step_i, step_i * dt, E, E_in, float(ep.sum()), ep15, *Q, *u_prof])
        print(f"  step {step_i:5d} t={step_i*dt:6.2f} E={E:.5e} E_in(r<30)={E_in:.5e}"
              f" E_phi={ep.sum():.4e} E_phi(r<15)={ep15:.4e}  Q[" + " ".join(f"{q:+.3e}" for q in Q) + "]"
              f"  u(r)[" + " ".join(f"{x:+.3e}" for x in u_prof) + "]"
              f"  wall={time.time()-t0:.0f}s", flush=True)

    log(0)
    for step_i in range(1, args.steps + 1):
        evolver.step(ts)
        if step_i % args.report == 0:
            log(step_i)
            np.savez(os.path.join(_HERE, f"t1_{args.arm}_{args.bc}.npz"), rec=np.array(rec),
                     r_shells=np.array(R_SHELLS), r_bins=np.array(r_bins), dt=dt)
    print(f"done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
