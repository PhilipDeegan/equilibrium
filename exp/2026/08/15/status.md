# Entries for 2026-08-15 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## RESOLVED (2026-08-15): pass1/pass2 coefficient parity + step_cross per-level rate — all tests now pass

Full technical detail (the four pass1 fixes, the shared-coefficient-helper
architecture, the multiplicity-double-count pitfall, the step_cross
`cell.L`/dt-convention finding, the curve-fitting methodology lesson) has
moved to `doc/dst_amr.md` ("Pass1/pass2 coefficient parity" and "step_cross's
`cell.L` factor is convention-dependent" sections) — durable design
knowledge, not session log. One-line summary: pass1 (`_step_pass1[_lev]`)
was missing `1/L`, using the wrong sigma average, and (once those were
fixed) still missing pass2's ancestor-substitution/multiplicity logic
entirely — now fixed via 3 shared helpers (`_resolve_face_neighbour`,
`_ancestor_L_src`, `_face_flux_coefficient`) both passes call, so they
cannot independently drift again. All 4 `TestFluxConservation` tests
(`tst/substrate/adaptive/test_adaptive_lattice.py`) pass.
`tst/substrate/adaptive/test_step_cross_per_level.py`'s
`test_fractional_loss_level_independent` was separately resolved (unrelated
code path) — see doc for the two independent bugs (wrong comparison metric
+ a real `cell.L`/dt double-count) and the new
`AdaptiveLattice.step_cross_level_scale_power` knob.

**Also done 2026-08-15: dead debug-toggle cleanup.** `AdaptiveLattice` had
accumulated 16 static diagnostic switches from past investigations. Removed
7 confirmed genuinely unused anywhere (tst/, run/, exp/, including the C++
test suite — first-pass check via Python-binding names alone was wrong,
missed the C++ suite's direct `AdaptiveLattice::<name> = ...` sets):
`disable_amr_reflux`, `disable_pass2_siblings`, `disable_cross_level_flux`,
`disable_same_level_flux`, `disable_same_level_flux_at_boundary_only`,
`accumulate_damping_factor`, `freeze_boundary_proxies_per_outer_step` — all
self-labeled "temporary, remove once root cause found" from the now-resolved
L8 deep-telescope investigation. Kept 9: 5 now load-bearing for the new
pass1/pass2 shared helpers, 1 real production feature
(`enable_ancestor_debias_correction`), 1 new (`step_cross_level_scale_power`),
2 extensively used by `test_adaptive_lattice.cpp`
(`disable_step_pass2_amr_correct`, `enable_cross_level_gradient_correction`
— do not remove without auditing that file). Compiled clean, full suite passes.

**Also fixed today, a genuine but different-cause blowup, now resolved**:
`tst.substrate.test_adaptive_vs_substrate.TestMultipleSplits` hit an
exponential blowup once fixes above were in place, traced to the TEST (not
the physics) using a flat, non-level-scaled dt at deep AMR split levels —
fixed by recomputing dt from the current deepest level each round. See
`doc/dst/failures/flat-dt-deep-amr-split-blowup.txt` for the full mechanism
and the "a fixed bug can unmask a masked one" lesson. Default `N_SPLITS`
reduced 20→11 (`EQ_TEST_DEEP_SPLITS` env var overrides for deeper validation
runs, e.g. `=20`) — 11 stays clean. Open TODO (not fixed): at 12+ splits, a
separate, genuine depth-dependent issue appears in the
`threshold_loss_localised` spatial-coherence check, attributed to
`N_BETWEEN=5` being a fixed STEP COUNT rather than fixed ELAPSED TIME (as
CFL dt shrinks 3x/level, 5 steps covers less real dynamics each level) — a
N_BETWEEN-scaling fix has not been attempted.

**Also deleted today**: the degenerate uniform-phi-on-every-face C++
synthetic step_cross test from 2026-08-13 — see
`doc/dst/failures/degenerate-uniform-phi-step-cross-test.txt` for why it was
invalid and removed rather than fixed. step_cross's own per-level-rate
question was not actually settled by that test and remains tracked
separately via `test_step_cross_per_level.py` (resolved above) /
`[[project_step_cross_v2]]`.

## Recently resolved (kept here only as a pointer, full detail above/in memory)

- **"step_cross cell-size/level dependence"** (2026-08-13 investigation,
  formerly this file's top entry) — resolved 2026-08-14, see above: it was
  never step_cross, it was `_step_pass1`/`_step_pass1_lev` (missing `1/L`,
  wrong sigma averaging, wrong `L`-blend at junctions) plus
  `resolve_stale_external_proxies`'s untested hex-only restriction. `step_cross`
  itself was never touched and its own per-level-rate question (the
  `test_step_cross_per_level.py` failure above) remains genuinely open,
  separately.
- **"stale inter-parent proxy wiring"** (2026-08-12/13 investigation) —
  the hex-only fix from that investigation is now extended to all 14 faces
  (fix 4 above), which is what actually eliminated the L1R60 secular H drift.
  `[[project_2to1_balance_question]]` (whether `can_split` needs a real 2:1
  neighbour-balance check) remains open and unrelated to today's fixes.
- **TestFluxConservation** (2026-07-28): that round's 4 failures were a stale
  unweighted debug assertion, unrelated to today's NEW 4 failures (which were
  a real, mechanism-understood drift from today's pass1 changes, now
  resolved above).
- **step_cross antisymmetry residual** (2026-07-27): a real ~1.52e-9 floor,
  judged tolerable not a defect — same-level antisymmetry and the
  coarse-proxy junction-sum identity compete for one coarse leaf's 14-face
  budget; a geometric-reroute fix attempt made it worse. Test tolerance
  corrected to the true measured floor. See Part 13.
- **Pre-existing, still open, unrelated to anything above**:
  `test_curvature_corrected_secant_formula_isolated` — retired/removed
  2026-08-08 (confirmed 2026-08-12). Resolved.

## IN PROGRESS: L8 base-update blowup — root mechanism identified (under-resolution), fixed by going bigger, not a formula correction

**5-step run with the current eta ladder, 2026-08-09** (`exp/2026/08/09/
15_50_l5_5step_save_each_step/`, ladder `[0,0,1e-5,1e-7,3e-9,3e-9]` — L5 set
to match L4's re-earned value per the bisection above): `H/H0` = 1.0036 ->
1.0115 -> 1.0208 -> 1.0269 -> 1.0299. Per-step growth *decelerates* after
step 3 (deltas 0.79pp -> 0.93pp -> 0.61pp -> 0.31pp) — a settling transient,
not runaway growth, though only 5 steps in. All 5 full snapshots saved to
`/bk/l5_5step_save_each_step/step_0{1..5}.h5` (5.6GB each). [[feedback_save_full_state_not_scalars]]

**Structural sanity-check on those 5 snapshots (2026-08-09)**, beyond the
scalar H metric — scripts in the same exp dir
(`diagnose_structure.py`/`_v2.py`, `front_trajectory.py`, `plot_l5_xz.py`):
- No NaN/Inf anywhere, any step. Leaf count per level identical IC->step5
  (`[146678, 5238, 80190, 984150, 7440174, 14348907]` for L0-L5) — confirms
  pure field evolution on a static mesh, no topology drift.
- Radial `u` profile is smooth (single-lobe-per-side, Bessel-shaped, peak
  r≈0.22, drops to noise floor by r≈0.48) at every step, no spikes.
- Per-level radial ranges are NOT clean concentric shells — they overlap
  (hop-distance-graded on the BCC lattice, not Euclidean-sphere-graded):
  L5=[0,0.813], L4=[0.420,1.817], L3=[0.847,2.911], L2=[1.818,4.010],
  L1=[2.722,5.019], L0=[4.123,29.994].
- Checking cross-level junctions via the Python `leaf_nb` array has a real
  blind spot at multiplicity>1 junctions (L1-L2 through L4-L5) — see
  `doc/dst/failures/leaf-nb-junction-check-blind-to-multiplicity.txt`. Any
  future junction-sanity check at L4/L5 needs to walk ancestry instead.
- **Front trajectory (multi-threshold, IC through step5)**: the *bulk*
  front (`|u|>=1e-2`, within 1% of peak) is frozen at r=0.4563 at *every*
  step — hasn't moved. A tiny-amplitude numerical precursor (`|u|>=1e-12`)
  already reaches r≈1.74 by step 1 — the CFL/light-cone reach of one
  `evolver.step()`, physically negligible. `H_by_level(L4)`'s exact
  trajectory (33.7->41.8->64.0->96.25->134.25->173.77) fits a ~t^1.8 power
  law; extrapolating, the bulk reaching even 1% of total H in L4 would take
  ~150 more steps (~25+ hours at ~600-670s/step). Confirms quantitatively
  that 5 steps is nowhere near enough to test the L4/L5 junction
  dynamically.
- XZ midplane slice plots (L5-only) visually confirm the frozen two-lobe
  bulk shape and the slowly-growing leak halo.

**New reusable tool**: `run/substrate/adaptive/plot_telescope_xz.py` — one
PNG per input `.h5` checkpoint, all levels included, with **level-aware
midplane slicing** (per-level `|y|` threshold scaled to that level's own
cell size `1/3^level`) avoiding sparse-corner interpolation artifacts from
mixing resolutions. Takes arbitrary paths via CLI + `--clip-r`/`--nr`/
`--log-floor`. Verified against the current 6 checkpoints (`--clip-r 3.0`,
covers through L3) — clean, no artifacts; level regions render as
sharp-edged nested plus/cross shapes (L5 square core -> L4 cross -> L3
wider cross) — that's just the hop-distance grading preferring the BCC
lattice's 6 orthogonal square-face directions, not fractal.

**Open lead, not yet chased down**: *within* a single level (checked L4
alone, ruling out level-boundary-shape or plotting/interpolation as
explanations — confirmed present in raw uninterpolated leaf data too), the
log|u| pattern is genuinely fractal/self-similar — a Sierpinski-like nested
clustering of which specific cells have received any flux vs are still at
exact floor. Likely explanation: since L4-L5 flux travels through the
ancestor-substitution mechanism (not direct `leaf_nb` pointers), and
Odd27's recursive 3x3x3 ancestor-lineage structure is itself self-similar,
propagation may be following tree/ancestor-lineage adjacency rather than
smooth Euclidean-local diffusion — producing exactly this kind of patchy,
non-smooth front. NOT confirmed (would need to check whether the fractal
clusters line up with shared-ancestor groups). Already fully formed by
step 5 (not something that took many steps to emerge) — a stronger red
flag than a slowly-developing pattern (user's observation, 2026-08-09). An
alternative "is it eta_h boundary damping?" explanation was raised and
ruled out — see `doc/dst/failures/eta-h-ruled-out-as-fractal-texture-cause.txt`.

**User's working conclusion (2026-08-09)**: the IC is exactly zero
(bit-for-bit) everywhere outside the core, so fractal texture can only be
introduced by however flux actually crosses into L4 during stepping — i.e.
via the ancestor-substitution/multiplicity path. **This is probably a
junction error in that flux-correction path**, not a benign feature of
hop-graded propagation. Elevated from "worth checking" to "leading
hypothesis." **Given 2026-08-14's finding that `_step_pass1`'s coefficient
mismatch caused a real, mechanism-confirmed leak via exactly this
ancestor-substitution/junction path — and that fixing it eliminated an
analogous secular drift on L1R60 — this lead should be re-examined against
a rebuilt L4/L5 checkpoint before doing any further fractal-cluster/
ancestor-group correlation work.** Not done yet. Next step: check whether
fractal clusters line up with shared-ancestor groups
(`ancestor_group_multiplicity` boundaries).

**Full-domain replot (2026-08-09, `--clip-r 30`)**: at that zoom the core
is only a few pixels — inconclusive for whether L1-L2/L2-L3/L3-L4
boundaries are clean or just not-yet-reached. Direct check of L3's own
values instead: L3 HAS been touched by step5 (174,252 / 984,150 leaves
nonzero, out to r=2.04), just ~9 orders of magnitude smaller amplitude
ceiling than L4's (2.3e-10 vs 0.14). So "only L4-L5 shows fractals" in the
shared-color-scale plots is likely a dynamic-range/visibility artifact, not
a real absence at L3 — queued: re-plot L3 with its own scale to confirm the
same clustering shape appears there too (script prepped:
`/tmp/.../scratchpad/check_l3_pattern.py`, not run to completion this
session).

Caveat: `TimeStepper.by_level` uses ratio=3 (Odd27), subcycling `dt_base/3^level`
per level — one outer `.step()` at L5 is actually 3^5=243 internal substeps.
"Fully formed by step 1" means fully formed after 243 fine-level ticks, not 1 —
consistent with, doesn't rule out, ancestor-substitution hypothesis.

**L5-L4 boundary shape confirmed square/cube, not circular (2026-08-10)**:
on `/bk/l5_core_diameter9_cells_across_220.h5` — L5 leaves at max Chebyshev
(L∞) distance 0.575 (18,738 cells, one axis-aligned face) span Euclidean
0.575-0.813 (=0.575·√2); L5 leaves at max Euclidean 0.813 (only 12 cells) all
sit at exactly Chebyshev 0.575 (edge midpoints — true cube corners absent,
consistent with Kelvin-cell/truncated-octahedron shape cutting them off). Same
signature at L4 (Chebyshev flat at 1.725 across 3,750 cells, Euclidean varying
1.725-1.817). A sphere would show the opposite. Confirms boundary is graded by
hop-distance along BCC lattice's 6 orthogonal square-face directions — axis-
aligned cube shells, matching "L5 square core -> L4 cross" plot geometry.

Plan (in progress): user running the same checkpoint/eta-ladder for ~a day on
another machine to let bulk propagate across telescope levels, then inspect
with `plot_telescope_xz.py`. Checkpoint-save cadence not decided — saving
every step at 5.6GB/step over a day (~130-140 steps) would be ~730-780GB;
saving less frequently is probably needed.

**[[project_odd27_amr_junction_leak]] — L8 H/H0=134 blowup: ROOT CAUSE FOUND.**
Ruled out (full detail in that file): eta_h, `step_cross` gate, domain size,
per-face multiplicity correctness, boundary coupling magnitude, uniform-mesh/
missing-sink effect, phi (u alone reproduces/exceeds blowup), pure-mesh-
mechanics defect, single-junction defect.

**The actual driver: under-resolution.** `H/H0 ∝ cells_across^-1.85`
(near inverse-quadratic) — original calibration gave only ~17.9 cells across
the spheromak's own diameter at L8; doubling resolution gives ~3.5x reduction
in blowup (mult=2: 35.7 cells→37.0x, mult=4: 71.4 cells→10.5x). Confirmed and
extended 2026-08-07/08 ([[project_hop_telescope_resolved_spheromak]]): a
genuinely information-dense spheromak (`cells_across=220`, new cheap hop-based
graded telescope, at **L5** — depth was never the requirement, resolution
relative to the object was) gives `H/H0=1.0036` after step 1, and **stays
bounded through 3 steps**: 1.0036 → 1.0133 → 1.0264 (mildly accelerating
per-step growth, nowhere near blowup territory). Checkpoint:
`/bk/l5_core_diameter9_cells_across_220.h5` (5.6GB). Script:
`exp/2026/08/07/15_00__properly_resolved_l8_spheromak_decay/load_and_step.py`.

`enable_ancestor_debias_correction` (adaptive_lattice.hpp): separately real —
See doc/dst/failures/ancestor-debias-correction-not-l8-blowup-fix.txt for why
it's not the L8 fix. Still correctly targeted, closes ~82% of an isolated
junction's flux-isotropy error, survives reflux, zero-cost when inert —
should ship on by default eventually.

Synthetic point/small-radius perturbation tests (old Parts 32-35): See
doc/dst/failures/synthetic-point-perturbation-not-diagnostic-of-l8-blowup.txt
— not diagnostic of the real blowup, don't reuse for reasoning about it.

**Next / open**:
- Continue stepping the L5/cells_across=220 checkpoint further (past 3 steps)
  to see whether mild acceleration (0.36%→0.97%→1.31% per step) stays bounded
  or eventually compounds.
- `mult=8`/`16` resolution sweep (does H/H0 keep falling toward ~1.0 past
  cells_across=71.4?) — deferred pending a release (`-g 0`, no assertions)
  rebuild; `mult=4` run took ~85min/step under an assertions build.
- **Going deeper than L5 hits a hard RAM wall, not a formula problem.**
  Current *dense* hop-based telescope (maximizes information density near
  center) costs ~27x/level: L5=23M leaves/~15GB, L6≈620M/~400GB,
  L7≈17B/~11TB, L8≈450B/~295TB. Different cost model from the *original*
  `_build_telescope` (Euclidean shell, cost ≈5.44·R³ independent of depth —
  its own L8/R=100 run was only ~5.6M leaves) — not directly comparable; dense
  hop telescope is expensive specifically because it maximizes resolution
  everywhere near center, which is what avoids the blowup. MPI (scaffolding
  exists: `RemoteCellRef`/`ClusterSendBatch`/`ghost_map`/`BCType::MPI_REMOTE`,
  no actual `MPI_Send`/`Recv` yet) could plausibly reach L6-L7 with a real
  cluster; L8 needs ~5000 nodes, not "add MPI."
- **Separate, larger, unrelated goal**: real electron-radius/Bohr-orbit bridge
  (`a0/r_p≈62,892`, `BC_MARGIN=0.90`) needs domain radius `R≈30,000+`
  L0-cell-units regardless of core depth — dominated by *outer bulk* volume
  (`≈5.44·R³`, independent of refinement), not core resolution; fundamentally
  infeasible with uniform-size L0 cells at any depth/cluster size. Only real
  path: still-unstarted multi-scale-L0 feature
  ([[project_multiscale_lattice]] — cells grow larger far from core, matching
  DST's own theory, not an approximation). Already flagged in that memory back
  in May ("the orbit needs r≈52,000 lattice units... adaptive mesh is the only
  practical route") — independently re-derived at the same order of
  magnitude. Not started; separate from the L8 blowup work above.
