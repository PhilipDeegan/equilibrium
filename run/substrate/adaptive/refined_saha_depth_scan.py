"""
Progressive single-point-perturbation depth scan: a real graded telescope
(same `_build_telescope` recipe as refined_neutron_decay.py) built fresh at
every depth 0..--max-level, with a single-cell perturbation (u and/or phi) at
the domain centre -- NOT a spheromak (see module history: an earlier version
of this script mistakenly seeded a real spheromak profile at every depth,
answering a different question than the "does refinement alone matter"
null test this script is for). Runs --n-steps steps at each depth (default
10, not just one), saving the FULL u/phi trajectory (every step, not just the
final one) per depth.

Two prior corrections baked into this version, both caught by direct
inspection/user feedback rather than assumption:
  1. refine_region_to_convergence needs cl.info.split_hi/split_lo set first
     (matching _build_telescope's own convention) or nothing splits at all.
  2. The perturbation must be applied AFTER the telescope is built, not
     before -- seeding then refining forces _prolong_u_gradient_wide to
     interpolate into new children, contaminating the "should be exactly
     zero away from the perturbation" baseline.

Run:
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/refined_saha_depth_scan.py \\
      --R 30 --max-level 8 --region-r0 5.0 --perturb-amplitude 1.0 --n-steps 10 \\
      --save-snapshot-dir tmp/saha_depth_scan_perturb
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from equilibrium import lattice as _lat
from run.substrate.adaptive.refined_neutron_decay import _build_telescope
from run.substrate.adaptive.sweep_spheromak_decay_cross_scale import _compute_dt_l0

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    "--R", type=float, default=30.0, help="coarse (level-0) domain radius"
)
parser.add_argument(
    "--max-level", type=int, default=8, help="deepest depth to scan (0..this)"
)
parser.add_argument(
    "--region-r0",
    type=float,
    default=5.0,
    help="level-1 refinement radius, in level-0 sim-units; shrinks by SPLIT_RATIO each level "
    "after (same convention as refined_neutron_decay.py's _build_telescope)",
)
parser.add_argument(
    "--u0", type=float, default=0.0, help="uniform background field value"
)
parser.add_argument(
    "--perturb-amplitude",
    type=float,
    default=1.0,
    help="u spike at the domain centre, applied AFTER the telescope is built (0=disabled). "
    "Added to every leaf within --perturb-radius of the centre (default: just the single "
    "nearest leaf, i.e. --perturb-radius 0)",
)
parser.add_argument(
    "--perturb-phi-amplitude",
    type=float,
    default=0.0,
    help="phi (component 0) spike at the domain centre, same footprint rule as "
    "--perturb-amplitude",
)
parser.add_argument(
    "--perturb-radius",
    type=float,
    default=0.0,
    help="physical radius (domain units) of the perturbation, held FIXED across depths -- "
    "0 (default) perturbs only the single nearest leaf to the centre (footprint shrinks with "
    "depth, since that leaf gets smaller). A nonzero radius perturbs EVERY leaf within it, so "
    "the SAME physical footprint is resolved by more/smaller cells as depth increases, isolating "
    "the AMR-junction effect from the confound of also sharpening the input at the same time.",
)
parser.add_argument(
    "--eta-h-per-level",
    type=float,
    nargs="+",
    default=None,
    help="per-level hyper-resistivity coefficients (IMEX), e.g. the established calibrated "
    "ladder '0 0 1e-5 1e-7 3e-9 1e-10 1e-12 1e-13 1e-13' -- applied every step via "
    "step_boundary_hyper_resistivity_per_level_all, same convention as refined_neutron_decay.py. "
    "Unset (default) applies NO damping at all, matching every other test in this investigation's "
    "'saha vacuum' isolation intent.",
)
parser.add_argument("--n-steps", type=int, default=10)
parser.add_argument("--n-threads", type=int, default=8)
parser.add_argument("--save-snapshot-dir", type=str, default=None)
args = parser.parse_args()


def build_and_perturb(depth):
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(args.n_threads)
    cl = alat[0]
    _lat.build_bcc_sphere(cl, args.R, d1=1.0, u0=args.u0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    if depth > 0:
        _build_telescope(alat, cl, args.region_r0, depth)

    pos = np.array(cl.leaf_pos)
    r = np.linalg.norm(pos, axis=1)
    centre_j = int(np.argmin(r))
    if args.perturb_radius > 0.0:
        mask = r <= args.perturb_radius
        mask[centre_j] = (
            True  # guarantee at least one cell even if radius < nearest-cell offset
        )
    else:
        mask = np.zeros(pos.shape[0], dtype=bool)
        mask[centre_j] = True
    n_perturbed = int(mask.sum())

    if args.perturb_amplitude != 0.0:
        u = np.array(cl.leaf_u)
        u[mask] += args.perturb_amplitude
        cl.set_leaf_u(u)
    if args.perturb_phi_amplitude != 0.0:
        phi = np.array(cl.leaf_phi)
        phi[0, mask] += args.perturb_phi_amplitude
        cl.set_leaf_phi(phi)

    cl.set_bc(_lat.BoundaryCondition.neumann())
    return alat, cl, centre_j, n_perturbed


def main():
    if args.save_snapshot_dir:
        Path(args.save_snapshot_dir).mkdir(parents=True, exist_ok=True)

    print(
        f"{'depth':>5}  {'N_leaves':>10}  {'n_pert':>7}  {'dt':>10}  "
        + "  ".join(f"H/H0@{s}" for s in range(1, args.n_steps + 1))
    )
    for depth in range(args.max_level + 1):
        alat, cl, centre_j, n_perturbed = build_and_perturb(depth)
        N = cl.n_leaves
        u_hist = np.zeros((args.n_steps + 1, N))
        phi_hist = np.zeros((args.n_steps + 1, 14, N))
        u_hist[0] = np.array(cl.leaf_u)
        phi_hist[0] = np.array(cl.leaf_phi)
        H0 = float(np.sum(u_hist[0] ** 2) + 0.5 * np.sum(phi_hist[0] ** 2))

        dt = _compute_dt_l0(cl, D1=1.0)
        has_fine = depth > 0
        ts = _lat.TimeStepper.by_level(dt) if has_fine else _lat.TimeStepper.fixed(dt)
        evolver = _lat.AdaptiveLatticeEvolver(alat)

        ratios = []
        for step in range(1, args.n_steps + 1):
            evolver.step(ts)
            if args.eta_h_per_level is not None:
                alat.step_boundary_hyper_resistivity_per_level_all(
                    dt, args.eta_h_per_level, True
                )
            u_hist[step] = np.array(cl.leaf_u)
            phi_hist[step] = np.array(cl.leaf_phi)
            H = float(np.sum(u_hist[step] ** 2) + 0.5 * np.sum(phi_hist[step] ** 2))
            ratios.append(H / H0 if H0 > 0 else float("nan"))

        print(
            f"{depth:>5}  {N:>10}  {n_perturbed:>7}  {dt:>10.4e}  "
            + "  ".join(f"{r:8.4f}" for r in ratios),
            flush=True,
        )

        if args.save_snapshot_dir:
            np.savez(
                f"{args.save_snapshot_dir}/depth{depth}.npz",
                u_hist=u_hist,
                phi_hist=phi_hist,
                pos=np.array(cl.leaf_pos),
                level=np.array(cl.leaf_level),
                centre_j=np.asarray(centre_j),
                R=np.asarray(args.R, dtype=float),
            )


if __name__ == "__main__":
    main()
