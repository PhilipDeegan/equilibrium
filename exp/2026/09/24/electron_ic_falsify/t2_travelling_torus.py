"""T2: volume-filling torus carrying a TRAVELLING phi wave -- linear vs "drag".

Philip 09-24: phi (flux) is primary and moves, periodically, in a bounded
system; u is a secondary effect of that travel (drag); the electric field
around an electron is imposed by the electron. So seed the flux, not a u
profile, and measure whether a steady charge (DC u, 1/r tail, shell flux Q_u)
develops by itself.

IC (torus about z, centreline radius R0, tube radius A, domain R=70, neumann BC
-- outflow drains a uniform background, see status.md):
  g(rho')   = cos^2(pi rho'/(2A)) for rho'<A (rho' = distance to centreline)
  phi[k,i]  = amp * g * cos(m*theta) * (t_hat . hop_k)   at the face midpoint
              (toroidal flux, A.dl-style sampling -> antisymmetric by construction)
  u[i]      = amp * g * cos(m*theta)            in phase with phi -> travels in theta
  (face midpoint = pos + 0.5*_FACE_HOP: _FACE_HOP is the full neighbour offset,
  despite adaptive_flux_tube.py's comment; the first run used pos+_FACE_HOP,
  broke phi antisymmetry and made E non-conserved)
m=1 by default. Nothing else seeded: u background 0, no imposed charge.

Arms:
  linear : lam = 0 fixed (plain step())
  drag   : EXPERIMENTAL COUPLING, NOT DERIVED -- lam relaxes toward
           kappa * sum_k phi^2 / amp^2 with time constant tau (lam as "locked-
           out phi", project_dst_lam_definition), so sigma=1/(1+lam) follows
           the flux traffic. Makes step() nonlinear; the question is whether
           that lets travelling flux rectify into a steady u (a charge).

Fail criteria recorded per report: E retained in the torus region, period-mean
u inside the tube, u in an exterior shell, Q_u on shells outside the torus.
"""

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/p/git/eq")

import numpy as np

from equilibrium import lattice as _lat
from run.substrate.adaptive.sweep_spheromak_decay_cross_scale import _compute_dt_l0
from tst.substrate.adaptive.adaptive_flux_tube import _FACE_HOP, leaf_nb_j

W_FACE = np.array([3.0] * 8 + [1.0] * 6)

p = argparse.ArgumentParser()
p.add_argument("--arm", choices=["linear", "drag"], required=True)
p.add_argument("--R", type=float, default=70.0)
p.add_argument("--R0", type=float, default=35.0)
p.add_argument("--A", type=float, default=20.0)
p.add_argument("--m", type=int, default=1)
p.add_argument("--amp", type=float, default=1.0)
p.add_argument("--kappa", type=float, default=1.0)
p.add_argument("--tau", type=float, default=1.0)
p.add_argument("--lam-sat", type=float, default=0.0)
p.add_argument("--bc", choices=["neumann", "outflow"], default="neumann")
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--report", type=int, default=100)
p.add_argument("--threads", type=int, default=8)
p.add_argument("--tag", default="")
args = p.parse_args()


def torus_fields(x, y, z):
    rho = np.hypot(x, y)
    rp = np.hypot(rho - args.R0, z)
    g = np.where(rp < args.A, np.cos(0.5 * np.pi * rp / args.A) ** 2, 0.0)
    th = np.arctan2(y, x)
    safe = np.where(rho > 1e-9, rho, 1.0)
    t_hat = np.stack([-y / safe, x / safe, np.zeros_like(x)], axis=-1)
    return g, th, t_hat


def main():
    t0 = time.time()
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(args.threads)
    cl = alat[0]
    _lat.build_bcc_sphere(cl, args.R, d1=1.0, u0=0.0, lam0=0.0)
    alat.rebuild(0)
    pos = np.array(cl.leaf_pos)
    r = np.linalg.norm(pos, axis=1)
    N = cl.n_leaves
    nbj = leaf_nb_j(cl)

    g_c, th_c, _ = torus_fields(pos[:, 0], pos[:, 1], pos[:, 2])
    # u and phi IN PHASE -> travelling wave (dphi/dt ~ du/dtheta); 2026-09-24 first
    # version used quadrature (sin/cos) and produced a standing wave.
    u = args.amp * g_c * np.cos(args.m * th_c)
    phi = np.zeros((14, N))
    for k in range(14):
        mid = pos + 0.5 * _FACE_HOP[k]  # _FACE_HOP is the FULL neighbour offset
        g, th, t_hat = torus_fields(mid[:, 0], mid[:, 1], mid[:, 2])
        phi[k] = args.amp * g * np.cos(args.m * th) * (t_hat @ _FACE_HOP[k])
    cl.set_leaf_u(u)
    cl.set_leaf_phi(phi)
    if args.bc == "outflow":
        cl.set_bc(_lat.BoundaryCondition.outflow(np.array(cl.leaf_nb)))
    else:
        cl.set_bc(_lat.BoundaryCondition.neumann())

    tube = g_c > 0
    rho_c = np.hypot(pos[:, 0], pos[:, 1])
    rp_c = np.hypot(rho_c - args.R0, pos[:, 2])
    near = rp_c < args.A + 3.0
    r_out = args.R0 + args.A
    ext_shell = (r > r_out + 4) & (r < args.R - 4)
    q_shells = [r_out + 3.0, 0.5 * (r_out + args.R), args.R - 3.0]

    shells = []
    for R_s in q_shells:
        ii, jj, ww = [], [], []
        for k in range(14):
            j = nbj[:, k]
            msk = (r < R_s) & (j >= 0)
            msk[msk] &= r[j[msk]] >= R_s
            i_idx = np.nonzero(msk)[0]
            ii.append(i_idx); jj.append(j[i_idx]); ww.append(np.full(len(i_idx), W_FACE[k]))
        shells.append((np.concatenate(ii), np.concatenate(jj), np.concatenate(ww)))

    dt = _compute_dt_l0(cl, D1=1.0)
    ts = _lat.TimeStepper.by_level(dt)
    evolver = _lat.AdaptiveLatticeEvolver(alat)
    cos_t, sin_t = np.cos(args.m * th_c[tube]), np.sin(args.m * th_c[tube])
    print(f"arm={args.arm} R={args.R} R0={args.R0} A={args.A} m={args.m} N={N} tube={tube.sum()}"
          f" dt={dt:.5f} steps={args.steps} t_end={args.steps*dt:.1f}"
          f" kappa={args.kappa} tau={args.tau} lam_sat={args.lam_sat} bc={args.bc} build={time.time()-t0:.0f}s", flush=True)

    rec = []
    E0 = None

    def log(step_i):
        nonlocal E0
        u = np.asarray(cl.leaf_u)
        ph = np.asarray(cl.leaf_phi)
        e = u**2 + 0.5 * np.sum(ph**2, axis=0)
        E = float(e.sum())
        E0 = E0 or E
        E_near = float(e[near].sum())
        u_tube_mean = float(u[tube].mean())
        u_ext_mean = float(u[ext_shell].mean())
        a_c = 2 * float(np.mean(u[tube] * cos_t))
        a_s = 2 * float(np.mean(u[tube] * sin_t))
        Q = [float(-(w * (u[j] - u[i])).sum()) for i, j, w in shells]
        lam = np.asarray(cl.leaf_lam)
        rec.append([step_i, step_i * dt, E / E0, E_near / E0, u_tube_mean, u_ext_mean,
                    a_c, a_s, *Q, float(lam.max()), float(lam[tube].mean())])
        print(f"  step {step_i:5d} t={step_i*dt:6.2f} E/E0={E/E0:.5f} Enear/E0={E_near/E0:.5f}"
              f" <u>tube={u_tube_mean:+.3e} <u>ext={u_ext_mean:+.3e}"
              f" u_mode(c,s)=({a_c:+.3f},{a_s:+.3f})"
              f" Q[" + " ".join(f"{q:+.3e}" for q in Q) + "]"
              f" lam_max={lam.max():.3f} <lam>tube={lam[tube].mean():.3f}"
              f" wall={time.time()-t0:.0f}s", flush=True)

    log(0)
    lam = np.zeros(N)
    for step_i in range(1, args.steps + 1):
        evolver.step(ts)
        if args.arm == "drag":
            ph = np.asarray(cl.leaf_phi)
            target = args.kappa * np.sum(ph**2, axis=0) / args.amp**2
            if args.lam_sat > 0:
                target = target / (1.0 + target / args.lam_sat)
            lam += (dt / args.tau) * (target - lam)
            cl.set_leaf_lam(lam)
        if step_i % args.report == 0:
            log(step_i)
            np.savez(os.path.join(_HERE, f"t2_{args.arm}{args.tag}.npz"), rec=np.array(rec),
                     q_shells=np.array(q_shells), dt=dt)
    np.savez(os.path.join(_HERE, f"t2_{args.arm}{args.tag}_final.npz"), pos=pos,
             u=np.asarray(cl.leaf_u), phi=np.asarray(cl.leaf_phi), lam=np.asarray(cl.leaf_lam))
    print(f"done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
