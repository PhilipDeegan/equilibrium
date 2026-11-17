# L6 organic-vs-refined probe — methodology breaks down, not informative

Follow-on from `exp/2026/08/03/12_06_probe_refined_vs_organic_real_physics` (Parts 20-21
of `project_odd27_amr_junction_leak`), which fully closed the organic-vs-refined
mismatch question at max_level=1/2. Attempted to extend the same check to max_level=6
(L7 was queued to follow, contingent on L6's cost/result).

## Setup

Used `--R 20` (not the probe script's own default R=80) to keep the base mesh
affordable — R=80 starts at ~2.78M L0-only cells before any refinement, much bigger
than the R=20 convention used throughout the eta_h-ladder work. `--n-steps 3645`
(scaled up via the script's own `time_factor = 3^(SCALE_REF_LEVEL - max_level)`
convention, targeting ~15 actual recorded steps at L6). Otherwise script defaults
("everything active": `lam_mode=dynamic`, `damp_max=0.003`, `cross_scale=1e-3`).

## Result: NOT a meaningful comparison

Organic's mesh **never refined at all** — `lvl_max=0` at the end of the run, still
completely flat. Its `H` stayed at exactly **0.0** for all 15 recorded steps
(`phi_organic=0.000000e+00` throughout). IC line: `init_spheromak_ic: scale=0.0
... inner_cells=1/43599` — the synthetic object at L6 (`scale≈0.00494`, produced by
the script's SCALE_REF_LEVEL=1 time_factor convention shrinking the object by
`3^(1-6)` relative to its L1 reference size) is smaller than a single level-0 cell in
an R=20 domain. Organic's mesh starts flat and only refines where its own
criterion-driven `SplitCriterion` detects meaningful field structure at the CURRENT
resolution — if the object is invisible at the coarse starting mesh, the criterion
never fires, no matter how many pre-adapt rounds run (up to 50 were available).
Refined's fixed telescope has no such dependency (built deterministically regardless
of the field), so it resolved the object down to level 6 fine (144,427 leaves,
`phi_rms` growing to ~0.11-0.14 through the run).

The logged `rel_diff` values (e.g. `2.338e+302`) are a pure numerical artifact of
dividing by organic's near-zero `H` (`abs(ho-hr)/max(abs(ho), 1e-300)` floors the
denominator at 1e-300) — not a real percentage, not evidence of anything.

## Reading

This is a limitation of `probe_refined_vs_organic_amr.py`'s own scale-shrinking
convention (designed to keep max_level=0/1/2 tests cheap by holding *relative*
resolution constant, at the cost of shrinking the object's *absolute* size every
level) — it is NOT a new AMR-correctness bug, and it is unrelated to everything
confirmed in Parts 20-21 (IC hop-scaling, mesh-shape difference, `accumulate_to_parent`,
prolongation — all independently verified correct via the C++ unit tests, which don't
depend on this script's scale convention at all). Running L7 the same way would just
reproduce the same degenerate failure (object shrinks further still).

## Decision (user, 2026-08-03)

Do not chase a fix for the probe script's scale convention, and do not pursue an
organic-vs-refined cross-check at L6/L7/L8 by any other means either (forcing organic
to pre-refine to a matching shape, using a much larger R, etc. — all considered,
none pursued). **"We might be on our own now in terms of calibration"** at these
depths — proceed relying on the independently-verified structural correctness
(`accumulate_to_parent`/prolongation confirmed exact via direct C++ unit tests, Part 21)
and the eta_h-ladder stability sweeps (Parts 16-19), not a cross-check against organic,
for L6+ work going forward.
