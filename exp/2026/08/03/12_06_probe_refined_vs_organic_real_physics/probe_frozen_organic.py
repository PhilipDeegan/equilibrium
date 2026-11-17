"""
Control experiment: does freezing organic's mesh growth after the initial
pre-adapt (amr_growing=False) make its H trajectory track refined's fixed
telescope much more closely -- i.e. is the GROWING (L1) / SHRINKING (L2)
rel_diff trend during evolution driven by organic continuing to split further
(exercising prolongation into new children + accumulate_to_parent restriction
repeatedly) while refined's telescope stays completely static after IC?

Same parameters as probe_refined_vs_organic_amr.py's own max_level=1/2 runs
(R=80, sph_frac defaults, cross_scale=1e-3, lam_mode=dynamic, damp_max=0.003,
split_hi=0.02), EXCEPT organic is called with amr_growing=False so its mesh
is frozen at the pre-adapt-converged structure for the whole recorded run,
matching refined's static-telescope regime.
"""

import sys

sys.path.insert(0, ".")

from run.substrate.adaptive.refined_neutron_decay import SPLIT_RATIO
from run.substrate.adaptive.refined_neutron_decay import _build_and_run as refined_run
from run.substrate.adaptive.sweep_neutron_decay_cross_scale import (
    _build_and_run as organic_run,
)

SCALE_REF_LEVEL = 1


def run_level(max_level, n_steps=100, report=10, split_hi=0.02):
    R = 80.0
    sph_frac = 0.06
    scale_at_ref = R * sph_frac
    time_factor = SPLIT_RATIO ** (SCALE_REF_LEVEL - max_level)
    scale_this_level = scale_at_ref * time_factor
    sph_frac_this_level = scale_this_level / R
    n_steps_this_level = max(1, round(n_steps * time_factor))
    report_this_level = max(1, round(report * time_factor))

    shared = dict(
        B0=1.0,
        lam_target=1.0,
        u_max=1.0,
        amplitude=1.0,
        sigma=None,
        lam_scale=0.05,
        lam_mode="dynamic",
        lam_relax_rate=0.0,
        lam_alpha=1.0,
        sponge=0.15,
        damp_max=0.003,
        n_steps=n_steps_this_level,
        report=report_this_level,
        gate="barrier",
        m0=1e-2,
        n_threads=8,
    )

    print(
        f"\n===== max_level={max_level}  scale={scale_this_level:.6g}  "
        f"n_steps={n_steps_this_level} ====="
    )

    print("--- organic, amr_growing=FALSE (frozen after pre-adapt) ---")
    r_organic_frozen = organic_run(
        1e-3,
        R=R,
        sph_frac=sph_frac_this_level,
        max_level=max_level,
        split_hi=split_hi if max_level > 0 else 0.0,
        split_lo=0.0,
        amr_growing=False,
        **shared,
    )

    print("--- refined (fixed telescope, unchanged) ---")
    r_refined = refined_run(
        1e-3,
        R=R,
        region_r0=5.0,
        max_level=max_level,
        scale_override=scale_this_level,
        scale_ref_level=max_level,
        eta_h_per_level=None,
        eta_h_ratio=None,
        **shared,
    )

    print(
        f"\n{'step':>6}  {'H_organic_frozen':>16}  {'H_refined':>12}  {'rel_diff':>10}"
    )
    n = min(len(r_organic_frozen["H"]), len(r_refined["H"]))
    for i in range(n):
        ho, hr = r_organic_frozen["H"][i], r_refined["H"][i]
        rel = abs(ho - hr) / max(abs(ho), 1e-300)
        print(f"{i * report_this_level:6d}  {ho:16.6e}  {hr:12.6e}  {rel:10.3e}")

    print(
        f"\nn_leaves: organic_frozen={r_organic_frozen['n_leaves']} "
        f"(lvl_max={r_organic_frozen['level_max']})  "
        f"refined={r_refined['n_leaves']} (lvl_max={r_refined['level_max']})"
    )


if __name__ == "__main__":
    run_level(1, n_steps=100, report=10)
    run_level(2, n_steps=100, report=10)
