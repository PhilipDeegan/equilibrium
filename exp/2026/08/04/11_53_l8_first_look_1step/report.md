# L8 first look — production-target probe, session summary

First-ever attempt at `max_level=8` (the real production target per `refined_neutron_decay.py`'s
own docstring — L7 only gives ~6.0 cells across the neutron diameter, below the BCC dispersion
anisotropy threshold; L8 gives ~17.9, clearing it). Follow-on to
[`15_15_l7_dynamic_lam_ladder_test`](../../03/15_15_l7_dynamic_lam_ladder_test/report.md) and
[`11_02_l7_etah_isolation_retest`](../11_02_l7_etah_isolation_retest/report.md), both of which
cleared L7. This report covers the whole day's L8 investigation (7 sub-experiments across
sibling directories, listed inline below) as one continuous thread.

## Result: L8 blows up immediately, and it's real (not a mistake in the harness)

| # | Directory | Change from previous | Result |
|---|---|---|---|
| 1 | `11_53_l8_first_look_1step` | R=20, L8 eta_h unset (defaults 0), **no `--scale-override`** | DIVERGED step 1, H/H0=599.1 |
| 2 | `12_33_l8_r30_first_look_1step` | R=30 instead of 20 | DIVERGED step 1, H/H0=606.2 (barely moved) |
| 3 | `12_46_l8_etah_continued_r20_r30` | L8 eta_h=1e-14 (R=20 & R=30) | H/H0=599.0/606.2 — **identical to eta_h=0** |
| 4 | `13_25_l8_etah_1e-13_r20_r30` | L8 eta_h=1e-13 (R=20 & R=30) | H/H0=599.1/606.2 — **still identical** |
| 5 | `13_57_l8_gate_v1_r20_r30` | `--gate v1` instead of default `barrier` | H/H0=599.1/606.2 — **bit-identical to barrier** |
| 6 | `14_29_l8_etah1e-13_scaleoverride_r20` | **added back `--scale-override 1.3606e-3 --scale-ref-level 8`** (corrects a mistake — see below) | DIVERGED step 1, H/H0=**134.1** |
| 7 | `15_40_l8_r100_no_override` | R=100, no override (real self-calibration lands on the same object size natively) | DIVERGED step 1, H/H0=**134.1** — matches #6 to 4 sig figs |

**The scale-override correction (#6)**: runs #1-5 had dropped `--scale-override`, reasoning
(wrongly) that it was a no-op at `max_level=8` since `scale_ref_level` defaults to 8. The
multiplier *is* a no-op there, but the override's absolute *value* (`1.3606e-3`) is what makes
R=20 represent the same real-calibrated object every other rung of the L0-L7 ladder was tested
against — without it, R=20 self-calibrates to a ~5x smaller, non-comparable object
(`inner_cells=27` vs the correct `3839`). This is why #1-5's H/H0 (~599-606) don't match anything
in the historical record, while #6's H/H0=134.1 matches
[`project_odd27_amr_junction_leak`](../../../../.claude/) memory's Part 2 (2026-07-19) finding of
**H/H0=130.8 at step 1** almost exactly — confirming #6 is the properly-calibrated repro of the
known, still-open L8 blowup, and #1-5's larger numbers were an artefact of the under-resolved
object, not a worse version of the real bug.

**#7 (R=100, no override) is the clean confirmation**: real self-calibration at R=100 lands on
`scale=1.3606e-3` natively (same object as #6's override forces at R=20) — only the surrounding
domain/mesh size differs (5,578,281 leaves vs 180,671). Result matches #6 to 4 significant
figures (134.06 both). **This rules out the R=20 cost-saving domain as a contributing
factor — the blowup is a genuine, domain-size-independent property of the L8 telescope
itself.**

## What's been ruled out this session (on top of `doc/dst_amr.md`'s existing list)

- **`eta_h` (hyper-resistivity boundary correction)**: tested `0`, `1e-14`, `1e-13` at L8 —
  bit-identical results across all three (runs #1, #3, #4). Not the driver at all, in any tested
  range.
- **`--gate v1` vs `barrier`** (step_cross's threshold mechanism): bit-identical H/H0 (run #5).
  Not step_cross either.
- **Domain size (R=20 vs R=30 vs R=100)**: R=20→R=30 moved H/H0 by only ~1.2% despite 4x better
  core resolution (`inner_cells` 27→113, run #2); R=20→R=100 (properly scaled) moved it by
  <0.01% (runs #6/#7). Not a resolution or domain-size effect.
- **`face_multiplicity`/`ancestor_group_multiplicity` correctness on same-level faces**: added
  temporary diagnostic bindings (`leaf_face_multiplicity`, `leaf_ancestor_group_multiplicity`,
  `debug_check_boundary_same_level_multiplicity` in `src/eq/lattice.cpp`) and checked every
  same-size face (leaf-leaf AND leaf-to-coarse-proxy) in the exact L8 mesh: **1,577,290
  connections checked, zero anomalies** at every level 0-8. The intra-leaf duplicate-face
  correction only ever fires for genuinely cross-level faces; it never touches same-level ones.
- **Boundary cells' total same-level coupling magnitude**: added `leaf_is_boundary`/
  `leaf_n_same_level_faces` bindings and compared `leaf_effective_w2` (already-exposed diagnostic)
  between boundary and interior L8 cells. Boundary cells have **~half** the interior value
  (1.68e9 vs 3.36e9 = exactly `W2_SUM/L8²`), not more — the wrong direction for an
  over-coupling/CFL-type amplification story. `leaf_n_same_level_faces` at L8 boundary cells is
  trimodal (exactly 4, 9, or 10 of 14 — fixed geometric categories from the telescope's graded
  shells, not a continuum).

## A separate finding: the fp64 precision wall at L8 — real, but far milder than first estimated

Independent of the above: `L8^4 ≈ 5.4e-16`, only ~2.4x machine epsilon. The `eta_h` term is a
biharmonic (`1/L^4`) operator, so this was suspected as the explanation for eta_h's apparent
zero effect. **A first back-of-envelope estimate (noise floor ~0.41, comparable to the field
itself) was wrong by ~3 orders of magnitude** — caught by directly measuring it instead of
trusting the estimate (see `tst/eq/core/data/lattice/adaptive_lattice.cpp`'s
`test_biharmonic_precision_floor_by_level`, added specifically for this). Measured on a clean,
junction-free uniform mesh at each level's own cell size: the first Laplacian application stays
machine-precision-exact (`32.0`, the confirmed BCC-shell normalisation constant) at every level
0-8; the second application (what `eta_h` actually multiplies) grows from ~1e-11 at L0 to only
**~2.4e-4 at L8** — real and growing, but nowhere near the field's own O(1) scale, and not
following a clean power law (consistent with genuine bit-level rounding, not the smooth `1/L^4`
formula the original estimate assumed). Full corrected writeup in `doc/dst_amr_scaling.md`.

**Follow-on (`16_56_*`, discarded, and `17_08_l8_isolated_harness_realladder_etah_sweep`,
kept)**: re-tested eta_h's effect on the *isolated harness* (`--cross-scales 0 --lam-mode
frozen --damp-max 0` — the original Parts 6-11 calibration convention, never re-checked at L8
against today's code) to see whether eta_h matters there even though it appeared inert with
cross_scale/dynamic-lam on. First attempt wrongly zeroed L0-L7's eta_h instead of holding the
earned ladder fixed (confounding two changed variables at once) and was discarded. Corrected
version (earned L0-L7 ladder held fixed, only cross_scale/lam_mode/damp_max changed):

| eta_h[8] | H/H0 @ step 1 (isolated harness) |
|---|---|
| 0 | 134.0728 |
| 1e-16 | 134.0720 |
| 1e-15 | 134.0680 |
| 1e-14 | 134.0635 |
| 1e-13 | 134.0626 |

**eta_h does have a real, tiny, monotonic, correctly-signed effect** (more damping → lower H) —
the earlier "bit-identical, zero effect" reading was an artefact of comparing at too few
significant figures / too coarse a value set, not a true null result. But the total range across
13 orders of magnitude is only ~0.0076% — nowhere near enough to touch a 134x blowup. **And
`eta_h[8]=1e-13` here (134.0626) matches the earlier cross_scale=1e-3/dynamic-lam run with the
same eta_h (134.0621) to 5 significant figures** — cross_scale and lam dynamics being on or off
changes essentially nothing about the blowup's magnitude. All five candidates still diverged at
step 1 (`--blowup-factor 100` caught every one).

## Current state / reading

L8's base-update blowup (H/H0≈134 at step 1, immediate, no build-up) is **real, reproducible,
and independent of everything tested**: domain size, eta_h (beyond a negligible correction),
step_cross's gate, per-face multiplicity correctness, boundary coupling magnitude, AND
cross_scale/lam-dynamics being on or off. This matches and extends `doc/dst_amr.md`'s existing
"Deep-telescope H-growth" investigation (Parts 2-22 of `project_odd27_amr_junction_leak`) — L8
was already flagged there as unsolved as of 2026-07-22 (Part 11), and nothing this session found
closes it, though the multiplicity/coupling-magnitude/cross_scale-independence checks are new
negative results worth keeping (they rule out several previously-open candidates cleanly).

## Next steps

- Do not extend the eta_h ladder to L8 with any guessed value on the assumption it will meaningfully
  help — confirmed to have only a ~0.008%-scale effect across 13 orders of magnitude.
- The multiplicity/coupling-magnitude bindings added this session
  (`leaf_face_multiplicity`, `leaf_ancestor_group_multiplicity`, `leaf_is_boundary`,
  `leaf_n_same_level_faces`, `debug_check_boundary_same_level_multiplicity` in
  `src/eq/lattice.cpp`) are temporary diagnostics, flagged as such in their own doc comments —
  candidates for removal once this investigation concludes, not permanent API.
  `test_biharmonic_precision_floor_by_level` (`tst/eq/core/data/lattice/adaptive_lattice.cpp`) is
  a permanent addition to the `test_lattice` suite, not temporary.
- The delta-relative-to-parent (multiresolution) representation idea (`doc/dst_amr_scaling.md`)
  is **on hold** — its motivation (eta_h precision-starved at L8) did not survive direct
  measurement. Do not start that work without a new, measured reason.
- The base-update blowup itself remains fully open, now with a longer list of ruled-out causes
  than at the start of the day. No live lead as of this writing.
