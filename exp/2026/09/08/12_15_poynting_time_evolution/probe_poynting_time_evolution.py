"""
Time-evolution check for the validated Poynting-ring dipole
(run/substrate/adaptive/probe_poynting_step_only.py, reproduced 2026-09-08:
equator slope=-3.081, polar slope=-1.575, ratio=3.52 at R=30/hops=4/8000
steps -- matches the historical `now` record exactly, so this mechanism is
NOT stale). That script only computes S once, at the very end -- it has
never checked whether |S| at a given radius actually shows a genuine
OUTWARD-TRAVELING wavefront (arrival time increasing with radius, roughly
causal at the lattice's own propagation speed) as opposed to some
already-present, non-causal, quasi-static response. This is one of the two
open loose ends flagged in `now` (2026-09-01 Poynting entry).

Method: same ring seeding/lam_ring/plain-step()-only setup as the reference
script. At logged intervals throughout the WHOLE run (not just the end),
snapshot phi (before the step), take one step, snapshot u (after) -- same
symplectic-ordering pairing local_energy_flux_vector requires -- and record
the equatorial-sector-averaged |S| in a few fixed radial shells. A genuine
traveling wavefront should show each shell's |S| rising from ~0 at a time
delay that increases with shell radius (roughly linear, arrival_time ~
r/c_eff); a non-causal/instant response would show all shells rising
together from step 0.
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from equilibrium.substrate.energy_flux import local_energy_flux_vector
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import (
    find_toroidal_ring,
    init_toroidal_winding,
    leaf_nb_j,
    winding_number,
)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--R", type=float, default=30.0)
parser.add_argument("--hops", type=int, default=4)
parser.add_argument("--amplitude", type=float, default=1.0)
parser.add_argument("--lam-ring", type=float, default=1.0)
parser.add_argument("--axis", default="z")
parser.add_argument("--n-steps", type=int, default=8000)
parser.add_argument("--log-every", type=int, default=50)
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--sector-half-angle", type=float, default=20.0)
parser.add_argument("--shell-width-frac", type=float, default=0.15, help="shell half-width as a fraction of shell radius")
parser.add_argument("--shells", type=str, default="8,12,16,20,24")


def main():
    args = parser.parse_args()
    ax_idx = {"x": 0, "y": 1, "z": 2}[args.axis]
    perp = [i for i in range(3) if i != ax_idx]
    shells = [float(s) for s in args.shells.split(",")]

    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    cl = alat[0]
    _lat.build_bcc_sphere(cl, args.R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    alat.set_n_threads(args.threads)

    r_all = np.linalg.norm(np.array(cl.leaf_pos), axis=1)
    r_max = float(r_all.max())
    r_inner = r_max * (1.0 - args.sponge)
    cl.set_bc(
        _lat.BoundaryCondition.quadratic_sponge(target=0.0, damp_max=args.damp_max, r_all=r_all, r_inner=r_inner)
    )

    ring_js, ring_faces = find_toroidal_ring(cl, axis=args.axis, mode="square", hops=args.hops)
    if ring_js is None:
        print("ERROR: no toroidal ring found")
        sys.exit(1)
    omega = init_toroidal_winding(cl, ring_js, ring_faces, amplitude=args.amplitude, phi_only=True)
    pos = np.array(cl.leaf_pos)
    ring_pos = pos[ring_js]
    ring_centre = ring_pos.mean(axis=0)
    ring_r = float(np.linalg.norm(ring_pos - ring_centre, axis=1).mean())

    lam = np.array(cl.leaf_lam)
    for j in ring_js:
        lam[j] = args.lam_ring
    cl.set_leaf_lam(lam)

    dt = compute_dt(cl)
    print(
        f"N={cl.n_leaves} R={args.R} hops={args.hops} ring_r={ring_r:.3f} omega={omega:.4f} dt={dt:.4e} "
        f"shells={shells}",
        flush=True,
    )

    nb_j = leaf_nb_j(cl)
    rel = pos - ring_centre
    ax_comp = rel[:, ax_idx]
    perp_comp = np.linalg.norm(rel[:, perp], axis=1)
    theta = np.degrees(np.arctan2(perp_comp, np.abs(ax_comp)))
    half = args.sector_half_angle
    equatorial = theta > (90.0 - half)
    r_from_centre = np.linalg.norm(rel, axis=1)

    shell_masks = []
    for r0 in shells:
        w = args.shell_width_frac * r0
        m = equatorial & (r_from_centre >= r0 - w) & (r_from_centre <= r0 + w)
        shell_masks.append(m)
        print(f"  shell r={r0:.1f}+-{w:.2f}: {int(m.sum())} equatorial cells", flush=True)

    trace = []
    for step in range(args.n_steps):
        do_log = (step % args.log_every == 0) or (step == args.n_steps - 1)
        phi_before = np.array(cl.leaf_phi).T.copy() if do_log else None
        alat.step(0, dt)
        alat.step_sponge(0)
        if do_log:
            u_after = np.array(cl.leaf_u)
            S3 = local_energy_flux_vector(pos, nb_j, phi_before, u_after)
            Smag = np.linalg.norm(S3, axis=1)
            t_phys = (step + 1) * dt
            row = [t_phys]
            for m in shell_masks:
                row.append(float(np.mean(Smag[m])) if m.any() else float("nan"))
            trace.append(row)
            print(
                f"step={step:6d} t={t_phys:8.3f} " + " ".join(f"S[r={r0:.0f}]={v:.4e}" for r0, v in zip(shells, row[1:])),
                flush=True,
            )

    trace = np.array(trace)
    np.save("exp/2026/09/08/12_15_poynting_time_evolution/trace.npy", trace)

    print("\n=== arrival-time estimate per shell (first log point where S exceeds 5% of its own eventual max) ===")
    for i, r0 in enumerate(shells):
        s_col = trace[:, i + 1]
        finite = np.isfinite(s_col)
        if not finite.any():
            print(f"r={r0:.1f}: no data")
            continue
        s_max = np.nanmax(s_col)
        if s_max <= 0:
            print(f"r={r0:.1f}: S never rose above 0")
            continue
        thresh = 0.05 * s_max
        idx = np.argmax(s_col >= thresh)
        t_arrival = trace[idx, 0] if s_col[idx] >= thresh else float("nan")
        print(f"r={r0:.1f}: S_max={s_max:.4e} arrival_t~{t_arrival:.3f}")
    print("done", flush=True)


if __name__ == "__main__":
    main()
