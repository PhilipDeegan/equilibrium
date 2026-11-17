"""
Free electron survival in a Saha-style (non-dead) ambient background.

Every prior electron-dipole probe (electron_dipole.py, probe_cross_v2_gates.py,
the translation-gate trio) embedded the double-ring dipole IC in a cold,
u_bg=0 vacuum with only an absorbing sponge at the boundary -- never a
genuine ambient background with the boundary standing in for "the rest of
the universe" rather than a sink.  This probe seeds a nonzero ambient u_bg
everywhere (Saha-vacuum style, matching tst/substrate/adaptive/test_saha.py's
own convention), keeps the electron construction itself at lam_target=0 (per
project_dst_electron_dipole -- the winding carries no lam; only the
*surrounding* is meant to be Saha-like), and asks two things under PURE
step() dynamics only (no step_cross -- that mechanism is independently
unvalidated and this probe deliberately does not depend on it):

  1. Does the double ring's own poloidal circulation survive (not erode
     into the ambient background), given a genuinely non-dead surrounding?
  2. Does a free electron at rest actually STAY at rest -- no systematic
     centroid drift -- under stochastic boundary reinjection/thermalization,
     as opposed to just jittering with zero mean?

Boundary condition, per tst/substrate/adaptive/test_saha.py's own explicit
finding (TestReciprocalWithRingDecay): reciprocal reinjection ALONE (no
absorbing layer) is WORSE than a plain sponge, because a bare lattice edge
reflects outgoing waves (missing-neighbour terms just drop from the flux
sum) and reciprocal reinjection then piles MORE energy at that reflecting
edge. The correct combination is sponge (absorbs, prevents reflection) +
reciprocal reinjection of what the sponge removes (stands in for "more
universe out there", not a vacuum sink) -- both layered together, applied
in that order each step.

Sizing: object cutoff radius = R * sph_frac; sponge starts at
R * (1 - sponge_frac). The gap between them is deliberate -- it's the room
for the ambient background to actually reach/engage the electron before
any absorption starts, not just a numerical safety margin.

Run:
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/probe_electron_saha_survival.py
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/probe_electron_saha_survival.py --R 15 --n-steps 20 --report 5
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import (
    face_flux_geometry,
    init_electron_dipole_ic,
    phi_poloidal_scan,
)
from equilibrium.substrate.reciprocal_bc import ReciprocalBCCluster as ReciprocalBC

parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=60.0, help="domain sphere radius")
parser.add_argument(
    "--sph-frac", type=float, default=0.6, help="dipole radius as fraction of R"
)
parser.add_argument(
    "--u-bg",
    type=float,
    default=0.05,
    help="ambient Saha-style u background (0 = old dead-vacuum convention)",
)
parser.add_argument(
    "--sponge-frac",
    type=float,
    default=0.2,
    help="sponge starts at R*(1-sponge_frac)",
)
parser.add_argument("--damp-max", type=float, default=0.01)
parser.add_argument(
    "--thermalize-amp", type=float, default=0.02, help="boundary noise amplitude"
)
parser.add_argument("--n-steps", type=int, default=5000)
parser.add_argument("--report", type=int, default=250)
parser.add_argument("--n-threads", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--data-out", type=str, default=None)
args = parser.parse_args()


def main():
    scale = args.R * args.sph_frac

    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(args.n_threads)
    cl = alat[0]

    _lat.build_bcc_sphere(cl, args.R, d1=1.0, u0=args.u_bg, ell0=1.0, lam0=0)
    alat.rebuild(0)

    centre = np.zeros(3)
    b_max, r_axis = init_electron_dipole_ic(
        cl,
        R_sph=1.0,
        scale=scale,
        u_bg=args.u_bg,
        lam_target=0.0,
        div_clean=True,
    )

    pos0 = np.array(cl.leaf_pos)
    r_all = np.linalg.norm(pos0, axis=1)
    r_max = float(r_all.max())
    r_inner = args.R * (1.0 - args.sponge_frac)
    cl.set_bc(
        _lat.BoundaryCondition.quadratic_sponge(
            target=args.u_bg, damp_max=args.damp_max, r_all=r_all, r_inner=r_inner
        )
    )
    bc = ReciprocalBC(cl, rng=np.random.default_rng(args.seed))

    dt = compute_dt(cl)
    print(
        f"N={cl.n_leaves}  R={args.R}  scale={scale:.1f}  r_axis={r_axis:.2f}  "
        f"r_inner(sponge)={r_inner:.1f}  gap={args.R - scale:.1f}  "
        f"u_bg={args.u_bg}  dt={dt:.4e}"
    )

    # region used for the centroid diagnostic -- excludes the thermalized
    # boundary shell so ambient noise doesn't swamp the position measurement
    core = r_all < 1.5 * scale

    def diagnostics():
        phi = np.array(cl.leaf_phi)
        u = np.array(cl.leaf_u)
        mid, n_hat, phi_f = face_flux_geometry(cl, centre)
        pol = phi_poloidal_scan(mid, n_hat, phi_f, r_axis, (scale,))
        weight = (u - args.u_bg) ** 2 + np.sum(phi**2, axis=0)
        w_core = np.where(core, weight, 0.0)
        wsum = float(w_core.sum())
        com = (w_core[:, None] * pos0).sum(axis=0) / wsum if wsum > 0 else np.zeros(3)
        H = float(np.sum(u**2) + 0.5 * np.sum(phi**2))
        return dict(
            circ_pol=float(pol[0]),
            phi_rms=float(np.sqrt(np.mean(phi**2))),
            H=H,
            com=com,
        )

    d0 = diagnostics()
    print(
        f"\n{'step':>6}  {'t':>9}  {'circ_pol':>10}  {'circ/circ0':>10}  "
        f"{'phi_rms':>10}  {'H/H0':>8}  {'|COM-COM0|':>11}"
    )
    times, circ_h, com_drift_h = [], [], []
    for step in range(1, args.n_steps + 1):
        alat.step(0, dt)
        alat.step_sponge(0)
        bc.compensate(cl)
        bc.thermalize(cl, args.thermalize_amp)

        if step % args.report == 0 or step == args.n_steps:
            d = diagnostics()
            com_drift = float(np.linalg.norm(d["com"] - d0["com"]))
            times.append(step * dt)
            circ_h.append(d["circ_pol"])
            com_drift_h.append(com_drift)
            print(
                f"{step:6d}  {step * dt:9.2f}  {d['circ_pol']:10.4e}  "
                f"{d['circ_pol'] / d0['circ_pol']:10.4f}  {d['phi_rms']:10.4e}  "
                f"{d['H'] / d0['H']:8.4f}  {com_drift:11.4e}",
                flush=True,
            )

    if args.data_out:
        np.savez(
            args.data_out,
            times=np.array(times),
            circ_h=np.array(circ_h),
            com_drift_h=np.array(com_drift_h),
            scale=scale,
            r_axis=r_axis,
        )
        print(f"\nsaved -> {args.data_out}")


if __name__ == "__main__":
    main()
