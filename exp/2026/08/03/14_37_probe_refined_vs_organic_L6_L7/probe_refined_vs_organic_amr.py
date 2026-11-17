"""Compare refined_neutron_decay.py (fixed-telescope AMR, Bohr-scale
calibrated) against sweep_neutron_decay_cross_scale.py (organic/growing AMR,
or plain uniform at --max-level 0) for matched physical parameters.

The two scripts share every physics step (evolver.step, step_cross, step_lam,
step_sponge) and differ only in how the mesh is built and the IC seeded
(init_spheromak_ic_refinable's per-leaf-level phi-hop scaling vs
init_spheromak_ic's fixed level-0 scaling -- see refined_neutron_decay.py's
module docstring). At --max-level 0 (AMR fully disabled both sides) they
must agree exactly, since init_spheromak_ic_refinable reduces to
init_spheromak_ic when every leaf is at level 0. This is the required
sanity check before trusting any AMR-vs-AMR comparison -- if it fails, the
bug is in shared setup/parameters, not AMR machinery.

Run (flat baseline check):
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/probe_refined_vs_organic_amr.py

Run (matched AMR level -- fixed telescope vs organic growing AMR):
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/probe_refined_vs_organic_amr.py \\
      --max-level 2 --split-hi 0.02
"""

import argparse
import sys

sys.path.insert(0, ".")

from run.substrate.adaptive.refined_neutron_decay import (
    SPLIT_RATIO,
)
from run.substrate.adaptive.refined_neutron_decay import (
    _build_and_run as refined_run,
)
from run.substrate.adaptive.sweep_neutron_decay_cross_scale import (
    _build_and_run as organic_run,
)

# The fixed telescope's finest shell radius is region_r0 / SPLIT_RATIO**(max_level-1)
# -- it SHRINKS as max_level grows (refined_neutron_decay.py's own design, see its
# module docstring). --sph-frac's scale = R*sph_frac is only a sane match for the
# telescope's reach AT this reference level (where region_r0's default was chosen/
# validated); at any other max_level tested, scale must be rescaled by the SAME
# SPLIT_RATIO**(SCALE_REF_LEVEL - max_level) factor -- on BOTH sides (organic and
# refined) -- or the object silently outgrows (or is swallowed by) the telescope's
# own reach as max_level changes, corrupting the comparison independent of any
# eta_h effect. Found 2026-07-23: max_level=0/1 probes looked sane but max_level=2
# showed rel_diff~0.90 from step 0 -- traced to exactly this, scale was being held
# constant across levels (scale_ref_level=max_level cancelled the rescale) while
# finest_region_r kept shrinking underneath it.
SCALE_REF_LEVEL = 1

parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=80.0)
parser.add_argument(
    "--sph-frac",
    type=float,
    default=0.06,
    help=f"scale AT max_level={SCALE_REF_LEVEL} = R*sph_frac; rescaled by "
    f"SPLIT_RATIO**({SCALE_REF_LEVEL}-max_level) for whatever --max-level is "
    "actually tested, applied identically to both organic and refined sides.",
)
parser.add_argument("--cross-scale", type=float, default=1e-3)
parser.add_argument("--lam-mode", choices=["dynamic", "frozen"], default="dynamic")
parser.add_argument("--lam-scale", type=float, default=0.05)
parser.add_argument("--gate", choices=("v1", "barrier"), default="barrier")
parser.add_argument("--m0", type=float, default=1e-2)
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument(
    "--n-steps",
    type=int,
    default=100,
    help=f"n_steps AT max_level={SCALE_REF_LEVEL}; rescaled by the same "
    f"SPLIT_RATIO**({SCALE_REF_LEVEL}-max_level) factor as scale for whatever "
    "--max-level is actually tested (characteristic timescale tracks object "
    "scale ~linearly -- see time_factor in main()), so comparisons across "
    "max_level hold characteristic time fixed instead of raw step count. "
    "--report is rescaled identically.",
)
parser.add_argument("--report", type=int, default=10)
parser.add_argument("--n-threads", type=int, default=10)
parser.add_argument(
    "--max-level",
    type=int,
    default=0,
    help="0 = both sides flat/uniform (must match exactly); >0 compares "
    "refined_neutron_decay's fixed telescope against sweep's organic "
    "growing AMR at the same max_level (physics may legitimately differ "
    "somewhat -- different refinement geometry -- but early trajectories "
    "should still track closely).",
)
parser.add_argument(
    "--split-hi",
    type=float,
    default=0.02,
    help="organic AMR split_hi (sweep_neutron_decay_cross_scale.py's --split-hi); "
    "ignored if --max-level 0.",
)
parser.add_argument("--split-lo", type=float, default=0.0)
parser.add_argument(
    "--region-r0", type=float, default=5.0, help="refined script's telescope radius"
)
parser.add_argument(
    "--eta-h-per-level",
    type=float,
    nargs="+",
    default=None,
    help="passed straight through to refined_neutron_decay's _build_and_run -- "
    "see that script's own --eta-h-per-level help. Only applied to the refined "
    "(fixed-telescope) side; the organic side has no boundary hyper-resistivity term.",
)
parser.add_argument(
    "--eta-h-ratio",
    type=float,
    default=None,
    help="passed straight through to refined_neutron_decay's _build_and_run -- "
    "see that script's own --eta-h-ratio help. Takes priority over --eta-h-per-level "
    "if both given. Only applied to the refined side.",
)


def main():
    args = parser.parse_args()
    scale_at_ref = args.R * args.sph_frac
    time_factor = SPLIT_RATIO ** (SCALE_REF_LEVEL - args.max_level)
    scale_this_level = scale_at_ref * time_factor
    sph_frac_this_level = scale_this_level / args.R
    print(
        f"scale @ L{SCALE_REF_LEVEL} = {scale_at_ref:.6g}  ->  "
        f"scale @ L{args.max_level} = {scale_this_level:.6g}  "
        f"(sph_frac_this_level={sph_frac_this_level:.6g})"
    )
    # Characteristic timescale tracks object scale ~linearly (confirmed empirically
    # 2026-07-23: H(t)/H0 trajectories at scale=4.8/3.2/1.6 collapse onto the same
    # curve when step count is rescaled by scale -- NOT an AMR/organic instability,
    # just a smaller object sampling more of its own natural evolution per fixed
    # step count). Comparing at a fixed --n-steps across max_level therefore compares
    # different phases of each object's trajectory; rescale n_steps/report by the
    # same factor as scale so the comparison holds characteristic time fixed.
    n_steps_this_level = max(1, round(args.n_steps * time_factor))
    report_this_level = max(1, round(args.report * time_factor))
    print(
        f"n_steps @ L{SCALE_REF_LEVEL} = {args.n_steps}  ->  "
        f"n_steps @ L{args.max_level} = {n_steps_this_level}  "
        f"(report {args.report} -> {report_this_level}, time_factor={time_factor:.6g})"
    )
    shared = dict(
        B0=1.0,
        lam_target=1.0,
        u_max=1.0,
        amplitude=1.0,
        sigma=None,
        lam_scale=args.lam_scale,
        lam_mode=args.lam_mode,
        lam_relax_rate=0.0,
        lam_alpha=1.0,
        sponge=args.sponge,
        damp_max=args.damp_max,
        n_steps=n_steps_this_level,
        report=report_this_level,
        gate=args.gate,
        m0=args.m0,
        n_threads=args.n_threads,
    )

    print(f"=== sweep_neutron_decay_cross_scale (max_level={args.max_level}) ===")
    r_organic = organic_run(
        args.cross_scale,
        R=args.R,
        sph_frac=sph_frac_this_level,
        max_level=args.max_level,
        split_hi=args.split_hi if args.max_level > 0 else 0.0,
        split_lo=args.split_lo,
        **shared,
    )

    print(
        f"\n=== refined_neutron_decay (max_level={args.max_level}, scale_override matched) ==="
    )
    r_refined = refined_run(
        args.cross_scale,
        R=args.R,
        region_r0=args.region_r0,
        max_level=args.max_level,
        scale_override=scale_this_level,
        scale_ref_level=args.max_level,
        eta_h_per_level=args.eta_h_per_level,
        eta_h_ratio=args.eta_h_ratio,
        **shared,
    )

    print("\n=== side-by-side ===")
    print(
        f"{'step':>6}  {'H_organic':>12}  {'H_refined':>12}  {'rel_diff':>10}  "
        f"{'phi_organic':>12}  {'phi_refined':>12}"
    )
    n = min(len(r_organic["H"]), len(r_refined["H"]))
    for i in range(n):
        ho, hr = r_organic["H"][i], r_refined["H"][i]
        rel = abs(ho - hr) / max(abs(ho), 1e-300)
        print(
            f"{i * report_this_level:6d}  {ho:12.6e}  {hr:12.6e}  {rel:10.3e}  "
            f"{r_organic['phi_rms'][i]:12.6e}  {r_refined['phi_rms'][i]:12.6e}"
        )

    print(
        f"\nn_leaves: organic={r_organic['n_leaves']} (lvl_max={r_organic['level_max']})  "
        f"refined={r_refined['n_leaves']} (lvl_max={r_refined['level_max']})"
    )


if __name__ == "__main__":
    main()
