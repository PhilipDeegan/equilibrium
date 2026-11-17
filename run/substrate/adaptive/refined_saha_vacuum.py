"""
Refined Saha vacuum -- IC-telescoped AMR depth on a bare (or Saha-thermalized)
uniform vacuum field, no ring/spheromak/topology structure at all.

Cross-check for the open base-update L8 blowup
(doc/mems/project_odd27_amr_junction_leak.md Parts 23-25): every existing L8
repro so far uses a spheromak/ring source IC with steep local gradients. This
script asks the same H/H0-at-step-1 question on a maximally boring field
instead -- a spatially uniform u0 background
(tst/substrate/adaptive/test_saha.py's bare-vacuum Assumption-1 setup),
refined with the exact same solid-ball telescope recipe as
refined_neutron_decay.py (radius shrinks by SPLIT_RATIO per level, cell-
count-conserving) down to --max-level, built once at IC time -- no organic/
criterion-driven regrowth during stepping. No BC by default, matching
test_saha.py's own "bare lattice conserves u exactly" test -- this isolates
the AMR step/reflux mechanics from everything else, including boundary
handling.

Two variants, controlled by --perturb-amplitude:
  0 (default): perfectly flat field everywhere, including across every new
    coarse-fine junction created by the telescope. If this still blows up,
    the bug is purely in the AMR step/reflux mechanics -- no gradient, no
    source, no IC needed to trigger it.
  >0: a single-cell u spike at the domain centre (inside the deepest part of
    the telescope), so restriction/prolongation/reflux actually have a
    nonzero difference to move across the junctions -- a flat field makes
    every one of those differences exactly zero by construction, which could
    hide a real bug rather than prove its absence.

--noise (ReciprocalBC.thermalize) defaults to 0 and should normally stay
that way: per feedback_saha_noise_scope, the Saha thermal bath is only valid
as a standalone ambient-equilibration check, not layered under another
physics question. It's exposed here only for a later, deliberately separate
"does AMR depth change the bath's own equilibrium statistics" check -- not
for use alongside --perturb-amplitude.

Run:
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/refined_saha_vacuum.py \\
      --R 20 --max-level 8 --n-steps 10 --perturb-amplitude 0
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")

from equilibrium import lattice as _lat
from tst.substrate.adaptive import refine_region_to_convergence
from equilibrium.substrate.reciprocal_bc import ReciprocalBCCluster as ReciprocalBC

SPLIT_RATIO = 3.0  # Odd27 refinement ratio: cell size shrinks by this factor per level


def _compute_dt_l0(cl, w2_sum=8 * 9.0 + 6 * 1.0, D1=1.0):
    """L0 CFL dt -- TimeStepper.by_level subcycles finer cells internally,
    so the caller only ever needs the coarse-cell dt (same formula as
    refined_poynting_ring.py/sweep_spheromak_decay_cross_scale.py). NOT
    tst.substrate.adaptive.compute_dt, which assumes ratio-2 level scaling
    and would give a much-too-large dt for an actual multi-level Odd27
    (ratio-3) telescope."""
    fu_arr = np.array(cl.leaf_fixed_u).astype(bool)
    lam_arr = np.array(cl.leaf_lam, dtype=np.float64)
    free = ~fu_arr
    sigma_max = float(np.max(1.0 / (1.0 + lam_arr[free])))
    return 0.4 * D1 / (sigma_max * w2_sum)


def _build_telescope(alat, cl, region_r0, max_level):
    """Deterministic ball telescope at the origin -- same cell-count-
    conserving recipe as refined_neutron_decay._build_telescope (radius
    shrinks by SPLIT_RATIO per level). No protected cells -- no ring here."""
    cl.info.split_hi = 0.5
    cl.info.split_lo = 0.0
    for lvl in range(1, max_level + 1):
        radius = region_r0 / (SPLIT_RATIO ** (lvl - 1))

        def _region(cl, radius=radius):
            r = np.linalg.norm(np.array(cl.leaf_pos), axis=1)
            return r < radius

        refine_region_to_convergence(alat, cl, _region, stage_max_level=lvl)
    cl.set_criterion_max(np.full(cl.n_leaves, np.nan))
    cl.clear_split_criterion()
    cl.info = _lat.DSTInfo()


parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    "--R", type=float, default=20.0, help="coarse (level-0) domain radius"
)
parser.add_argument("--max-level", type=int, default=8)
parser.add_argument(
    "--region-r0",
    type=float,
    default=5.0,
    help="level-1 refinement radius, in level-0 sim-units; shrinks by SPLIT_RATIO each level after",
)
parser.add_argument(
    "--u0", type=float, default=1.0, help="uniform background field value"
)
parser.add_argument(
    "--perturb-amplitude",
    type=float,
    default=0.0,
    help="0=perfectly flat field (pure AMR-mechanics test); >0 adds a single-cell u spike at "
    "the domain centre so junctions have a real difference to act on",
)
parser.add_argument(
    "--perturb-phi-amplitude",
    type=float,
    default=0.0,
    help="0=no phi perturbation; >0 adds a single-face phi spike (component 0) at the domain "
    "centre leaf, u stays flat/uniform (see --u0/--perturb-amplitude) -- isolates whether a "
    "real, domain-wide phi field (unlike u, which is ~0 background in the real spheromak IC) "
    "drives boundary H-growth on its own, independent of any u-curvature bias",
)
parser.add_argument(
    "--noise",
    type=float,
    default=0.0,
    help="ReciprocalBC thermalize amplitude per step (0=off; see module docstring -- do not "
    "combine with --perturb-amplitude, keep standalone per feedback_saha_noise_scope)",
)
parser.add_argument(
    "--eta-h-ratio",
    type=float,
    default=None,
    help="flat dimensionless hyper-resistivity ratio s=dt*eta_h_level*D_i, applied at every "
    "level via step_boundary_hyper_resistivity_ratio_all (0/unset = no damping, matching every "
    "prior run of this script)",
)
parser.add_argument(
    "--enable-ancestor-debias",
    action="store_true",
    help="enable AdaptiveLattice.enable_ancestor_debias_correction (Jensen-bias fix for "
    "cross-level ancestor substitution, see now/project_odd27_amr_junction_leak)",
)
parser.add_argument("--n-steps", type=int, default=10)
parser.add_argument("--h-report", type=int, default=1)
parser.add_argument("--blowup-factor", type=float, default=100.0)
parser.add_argument("--threads", type=int, default=10)
parser.add_argument("--max-cells", type=int, default=2_000_000)


def main():
    args = parser.parse_args()

    _lat.AdaptiveLattice.set_enable_ancestor_debias_correction(
        args.enable_ancestor_debias
    )

    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    cl = alat[0]
    _lat.build_bcc_sphere(cl, args.R, d1=1.0, u0=args.u0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    alat.set_n_threads(args.threads)

    print(
        f"N0={cl.n_leaves}  R={args.R}  max_level={args.max_level}  u0={args.u0}  "
        f"perturb_amplitude={args.perturb_amplitude}  noise={args.noise}",
        flush=True,
    )

    if args.max_level > 0:
        _build_telescope(alat, cl, args.region_r0, args.max_level)
        lvl_arr = np.array(cl.leaf_level)
        print(
            f"telescope built: region_r0={args.region_r0}  N_total={cl.n_leaves}  "
            f"level_max={int(lvl_arr.max())}  n_at_max_level={int((lvl_arr == args.max_level).sum())}",
            flush=True,
        )
        if cl.n_leaves > args.max_cells:
            print(
                f"ABORT: N_total={cl.n_leaves} exceeds --max-cells={args.max_cells}",
                flush=True,
            )
            sys.exit(1)

    if args.perturb_amplitude > 0.0:
        pos = np.array(cl.leaf_pos)
        centre_j = int(np.argmin(np.linalg.norm(pos, axis=1)))
        u = np.array(cl.leaf_u)
        u[centre_j] += args.perturb_amplitude
        cl.set_leaf_u(u)
        print(
            f"perturbed leaf j={centre_j}  level={int(np.array(cl.leaf_level)[centre_j])}  "
            f"pos={pos[centre_j]}",
            flush=True,
        )

    if args.perturb_phi_amplitude > 0.0:
        pos = np.array(cl.leaf_pos)
        centre_j = int(np.argmin(np.linalg.norm(pos, axis=1)))
        phi = np.array(cl.leaf_phi)
        phi[0, centre_j] += args.perturb_phi_amplitude
        cl.set_leaf_phi(phi)
        print(
            f"phi-perturbed leaf j={centre_j}  level={int(np.array(cl.leaf_level)[centre_j])}  "
            f"pos={pos[centre_j]}",
            flush=True,
        )

    bc = ReciprocalBC(cl, rng=np.random.default_rng(0)) if args.noise > 0.0 else None

    dt = _compute_dt_l0(cl)
    has_fine = int(np.max(np.array(cl.leaf_level))) > 0
    ts = _lat.TimeStepper.by_level(dt) if has_fine else _lat.TimeStepper.fixed(dt)
    evolver = _lat.AdaptiveLatticeEvolver(alat)

    def _H():
        u = np.array(cl.leaf_u)
        phi = np.array(cl.leaf_phi)
        return float(np.sum(u**2) + 0.5 * np.sum(phi**2))

    def _u_total():
        return float(np.sum(np.array(cl.leaf_u)))

    u0_total = _u_total()
    H0 = _H()
    print(f"u_total_0={u0_total:.6e}  H0={H0:.6e}  dt={dt:.4e}", flush=True)

    for step in range(1, args.n_steps + 1):
        evolver.step(ts)
        if args.eta_h_ratio is not None:
            alat.step_boundary_hyper_resistivity_ratio_all([args.eta_h_ratio] * 16)
        if bc is not None:
            bc.thermalize(cl, amplitude=args.noise)
        if step % args.h_report == 0 or step == args.n_steps:
            h_now = _H()
            u_now = _u_total()
            ratio = h_now / H0 if H0 > 0 else float("nan")
            u_drift = u_now - u0_total
            print(
                f"  step={step:6d}  H={h_now:.6e}  H/H0={ratio:.4f}  "
                f"u_total={u_now:.6e}  u_drift={u_drift:.3e}",
                flush=True,
            )
            if ratio > args.blowup_factor:
                print(
                    f"ABORT: H/H0={ratio:.4e} exceeds {args.blowup_factor}x -- "
                    f"stopping at step={step}",
                    flush=True,
                )
                sys.exit(0)

    print("\ndone.", flush=True)


if __name__ == "__main__":
    main()
