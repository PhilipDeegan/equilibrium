"""
Progressive depth-scan probe: build the real spheromak telescope (same
`_build_telescope` recipe as refined_neutron_decay.py) independently at every
depth 0..--max-level -- a fresh mesh each time, not one mesh growing deeper --
take ONE coarse step, save the full per-cell output. Pure base-mechanics only
(no cross_scale/step_lam/eta_h), matching refined_saha_vacuum.py's isolation
intent -- this is "saha vacuum" extended to a real graded telescope instead of
a flat field or single-point perturbation.

Correct methodology (established 2026-08-06): refine the mesh FIRST, then seed
the IC directly via init_spheromak_ic_refinable on the already-refined
leaves -- no prolongation/interpolation into new children at all. (An earlier
same-day probe that seeded-then-refined showed nonzero values in nominally
inert cells purely from prolongation; this script avoids that by construction,
matching how refined_neutron_decay.py's own _build_and_run already orders
things -- _build_telescope before _seed_ic.)

Run:
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/refined_saha_depth_scan.py \\
      --R 30 --max-level 8 --region-r0 5.0 \\
      --scale-override 1.3606e-3 --scale-ref-level 8 \\
      --save-snapshot-dir tmp/saha_depth_scan
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from equilibrium import lattice as _lat
from run.substrate.adaptive.refined_neutron_decay import (
    SPLIT_RATIO,
    _build_telescope,
    init_spheromak_ic_refinable,
)
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
    "--scale-override",
    type=float,
    default=1.3606e-3,
    help="spheromak scale AT --scale-ref-level, auto-rescaled by SPLIT_RATIO**(scale_ref_level - "
    "depth) at every depth so the object stays the SAME real physical size regardless of how "
    "deep the telescope goes (same convention as refined_neutron_decay.py)",
)
parser.add_argument("--scale-ref-level", type=int, default=8)
parser.add_argument("--n-threads", type=int, default=8)
parser.add_argument("--save-snapshot-dir", type=str, default=None)
args = parser.parse_args()


def build_and_seed(depth):
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(args.n_threads)
    cl = alat[0]
    _lat.build_bcc_sphere(cl, args.R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    if depth > 0:
        _build_telescope(alat, cl, args.region_r0, depth)
    scale = args.scale_override * SPLIT_RATIO ** (args.scale_ref_level - depth)
    init_spheromak_ic_refinable(
        cl,
        R_sph=1.0,
        scale=scale,
        B0=1.0,
        lam_target=1.0,
        u_max=1.0,
        amplitude=1.0,
        sigma=None,
    )
    cl.set_bc(_lat.BoundaryCondition.dirichlet(0.0))
    return alat, cl, scale


def main():
    if args.save_snapshot_dir:
        Path(args.save_snapshot_dir).mkdir(parents=True, exist_ok=True)

    print(
        f"{'depth':>5}  {'N_leaves':>10}  {'scale':>10}  {'dt':>10}  "
        f"{'H0':>12}  {'H1':>12}  {'H/H0':>10}"
    )
    for depth in range(args.max_level + 1):
        alat, cl, scale = build_and_seed(depth)
        u0 = np.array(cl.leaf_u).copy()
        phi0 = np.array(cl.leaf_phi).copy()
        H0 = float(np.sum(u0**2) + 0.5 * np.sum(phi0**2))
        dt = _compute_dt_l0(cl, D1=1.0)
        has_fine = depth > 0
        ts = _lat.TimeStepper.by_level(dt) if has_fine else _lat.TimeStepper.fixed(dt)
        evolver = _lat.AdaptiveLatticeEvolver(alat)
        evolver.step(ts)
        u1 = np.array(cl.leaf_u)
        phi1 = np.array(cl.leaf_phi)
        H1 = float(np.sum(u1**2) + 0.5 * np.sum(phi1**2))
        ratio = H1 / H0 if H0 > 0 else float("nan")
        print(
            f"{depth:>5}  {cl.n_leaves:>10}  {scale:>10.4e}  {dt:>10.4e}  "
            f"{H0:>12.6e}  {H1:>12.6e}  {ratio:>10.6f}",
            flush=True,
        )
        if args.save_snapshot_dir:
            np.savez(
                f"{args.save_snapshot_dir}/depth{depth}.npz",
                u0=u0,
                phi0=phi0,
                u1=u1,
                phi1=phi1,
                pos=np.array(cl.leaf_pos),
                level=np.array(cl.leaf_level),
                scale=np.asarray(scale, dtype=float),
                R=np.asarray(args.R, dtype=float),
            )


if __name__ == "__main__":
    main()
