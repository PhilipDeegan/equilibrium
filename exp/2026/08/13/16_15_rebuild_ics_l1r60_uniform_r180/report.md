# Rebuild L1R60 + uniform-R180 ICs, 300-step run

## Why

Follow-on from the stale inter-parent proxy fix
(`exp/2026/08/12/17_15_ghost_cell_junction_error/report.md`): the existing
`/bk/l1_wider_margin_active_ic220.h5` / `/bk/uniform_r180_active_ic220.h5`
checkpoints were built under the OLD (buggy) code and rely on
`AdaptiveLattice.load()`'s self-heal to correct stale wiring. Rebuilding from
scratch under the current code bakes the fix in at construction time instead,
giving a clean canonical IC. Also folds in today's `TestMultipleSplits`
2:1-balance fix and the new `dst_state()` introspection helper (see `now`,
2026-08-13) — no direct dependency, just same-session provenance.

## Setup

Same parameters as the original builds
(`exp/2026/08/11/21_27_everything_active/`): `cells_across=220`,
`lam_scale=1.0`, L1R60 = `r-l0=60.0`/`r-l1=46.667`, uniform R180 =
`padding=70.0` (R_domain=180). Both configs built AND stepped **in
parallel**, 4 threads each (8-core/16-thread machine — sum stays at 8, see
`feedback_cpu_oversubscription` memory), 300 steps, plotting every 100
(`plot_telescope_xz.plot_cluster`). `cross_scale=1e-3`, `eta_h` off, matching
the original "everything active" runs exactly.

New IC/output paths (kept `_v2`-suffixed until confirmed good, so the
existing checkpoints stay available for comparison until cleanup):
- `/bk/l1_wider_margin_active_ic220_v2.h5`, stepped → `/bk/l1_wider_margin_active_v2_stepped/`
- `/bk/uniform_r180_active_ic220_v2.h5`, stepped → `/bk/uniform_r180_active_v2_stepped/`
- Plots: `plots/L1_R60/`, `plots/uniform_r180/` (this dir)

Driver: `run.sh` (also copied `build_l1_wider_margin_active.py`,
`build_uniform_r180_active.py`, `step_l1_wider_margin_active.py`,
`step_uniform_r180_active.py`, `dipole_mode_ic.py` from
`exp/2026/08/11/21_27_everything_active/`, unmodified).

## Results

TODO — fill in after `run.sh` completes: final H/H0 for both configs,
comparison against the original report's table (`uniform_r180_active`:
0.999539 @ step 130; `l1_wider_margin_active` after fix: 0.912609 @ step
130 — note that value matched `pure_l1size_control`, not
`uniform_r180_active`, since the two references have different cell sizes),
plot spot-checks, any anomalies.

## Cleanup (after results confirmed good)

Per user sign-off (2026-08-13): delete
`/bk/l1_wider_margin_active_ic220.h5`, `/bk/l1_wider_margin_active_fixed_stepped/`,
`/bk/uniform_r180_active_ic220.h5`, `/bk/uniform_r180_active_stepped/`, then
rename the `_v2` paths to the canonical (non-`_v2`) names so existing
scripts' `--out`/`--ic` defaults keep working. Leaves
`/bk/uniform_l1size_pure_*` (separate control, untouched) and `/bk/nvidia/`
(unrelated) alone.
