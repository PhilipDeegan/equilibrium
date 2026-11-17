# Fixed-physical-footprint perturbation depth scan (user-designed) — oscillation persists with a broad (non-delta) input; per-input-unit severity is much smaller than the point-source case

## Motivation (user's own request, refining Part 33)

Part 33's single-point-perturbation depth scan showed increasingly wild, oscillatory `H/H0`
trajectories past depth ~3 — but that comparison confounds two things as depth increases: more
AMR junctions being stacked, AND the perturbed cell getting progressively smaller (since the same
absolute `Δu` was always added to just the single nearest-to-origin leaf, whose physical size
shrinks with depth). User asked for the controlled version: keep the perturbation's PHYSICAL
FOOTPRINT fixed across depths (via a radius, not a single cell), so refinement changes how many
cells resolve the same physical stimulus, not the stimulus itself.

Added `--perturb-radius` to `refined_saha_depth_scan.py`: every leaf within the given physical
radius of the origin gets the perturbation (not just the nearest one). A fallback always includes
the single geometrically-nearest leaf even when the radius is smaller than that leaf's own offset
from the origin (relevant at coarse depths, where a cell center is typically offset from the
origin by roughly half a cell width or more) — this is NOT "a perturbation smaller than a cell"
(no such thing exists in a discrete model); it's a selection-rule fallback so the test degrades
gracefully to Part 33's exact behavior at depths where the radius wouldn't otherwise select
anything.

Used `--perturb-radius 0.005` (smaller than the L8 telescope's own finest shell radius, ~0.002286
x ~2, so mildly crosses into the next-coarser shell at the very deepest level, negligible effect).

## Results

`n_pert` = number of leaves actually perturbed at each depth:

| depth | n_pert | H/H0 @1 | @2 | @3 | @4 | @5 | @6 | @7 | @8 | @9 | @10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 1.0021 | 1.0041 | 1.0061 | 1.0080 | 1.0099 | 1.0117 | 1.0134 | 1.0150 | 1.0165 | 1.0178 |
| 1 | 1 | 1.0143 | 1.0528 | 1.1145 | 1.1976 | 1.2999 | 1.4187 | 1.5506 | 1.6923 | 1.8399 | 1.9895 |
| 2 | 1 | 1.1487 | 1.5788 | 2.2554 | 3.1238 | 4.1142 | 5.1477 | 6.1426 | 7.0216 | 7.7182 | 8.1815 |
| 3 | 1 | 2.4128 | 6.3086 | 11.757 | 17.472 | 22.130 | 24.694 | 24.664 | 22.171 | 17.927 | 13.016 |
| 4 | 1 | 13.500 | 42.404 | 68.138 | 74.088 | 57.941 | 32.294 | 14.713 | 15.188 | 30.299 | 47.599 |
| 5 | 15 | 23.559 | 69.588 | 100.43 | 98.722 | 70.727 | 32.457 | 7.3928 | 16.338 | 53.419 | 86.625 |
| 6 | 259 | 92.063 | 181.61 | 204.69 | 201.56 | 119.99 | 79.733 | 90.153 | 100.18 | 156.94 | 155.75 |
| 7 | 7119 | 180.17 | 232.08 | 124.44 | 77.747 | 145.02 | 139.09 | 123.48 | 130.95 | 228.89 | 308.82 |
| 8 | 25345 | 51.325 | 65.906 | 35.673 | 22.557 | 41.453 | 39.789 | 35.405 | 37.506 | 65.021 | 87.488 |

**Depths 0-4 are bit-identical to Part 33** — confirms the fallback logic works correctly (the
radius doesn't select anything extra there, so it reduces to the single-nearest-leaf case). This
isn't new information about the physics; it's a consistency check on the implementation.

## Key check before interpreting: H0 scales exactly with n_pert

Verified directly from the saved snapshots (`u0=0` background, so each perturbed leaf contributes
exactly `1.0` to `H0`): `H0_point=1` vs `H0_radius=15/259/7119/25345` at depths 5/6/7/8 — exactly
matching `n_pert`. **This means `H/H0` is NOT directly comparable between the point-source and
radius-based runs without care** — a bigger `n_pert` means a bigger denominator, so a smaller
`H/H0` doesn't by itself mean "less defect," it could just mean "same absolute defect, divided by
a bigger number."

Checked the ABSOLUTE excess (`H_final - H0`) at depth 8: point-source `4198` (from `n_pert=1`) vs.
radius-based `2,192,026` (from `n_pert=25345`) — the radius-based run produces **~522x more total
spurious energy in absolute terms**, simply because there was far more input to amplify. Per
perturbed cell, though, the radius-based run produces far LESS excess (`86.5`/cell) than the
point-source run (`4197.7`/cell) — a genuine ~48x reduction in excess generated per unit of input.

## Interpretation

**Oscillation (non-monotonic `H/H0` over the 10 steps) persists with a spatially-broad,
non-delta-function perturbation** — this rules out "it's purely a delta-function/aliasing
artifact" as the full explanation; the underlying oscillatory behavior is real and shows up
regardless of whether the input is a single sharp point or a genuine multi-cell (15 to 25,345
leaves) region.

**But the input couples to the pathology far less efficiently, per unit of injected energy, when
it's spread out**: the point-source case generates ~48x more excess energy per perturbed cell than
the radius-based case does. This is consistent with (not surprising, given) how finite-difference/
finite-volume discretizations generally respond to sharp vs. smooth inputs — a delta function has
broad high-frequency content that couples more strongly to whatever numerical-scale defect exists
at these junctions, while a smoother, broader input excites it less per unit of energy.

**An initial attempt at this comparison tried to relate these numbers to Part 31's real-spheromak
depth scan (which showed `H/H0=134.07` at L8) — this was a mistake, caught immediately (the user:
"there was no spheromak in this case").** Part 31 is a fundamentally different experiment (a real
physical field, not a flat background plus an artificial perturbation) — there's no valid basis
for lining up their `H/H0` values directly, and a closer check showed the numbers don't even
consistently trend toward each other across depths (e.g. depth 7's radius-based value is actually
FARTHER from the real-spheromak value than depth 8's is). No claim of "closer to the real
mechanism" is being made here; the only valid comparison in this experiment is point-source vs.
fixed-footprint, both on the same flat vacuum background.

**Bottom line**: this strengthens (does not weaken) the case that the deep-level oscillation is a
real property of the coarse-fine junction mechanism, not an artifact specific to point/delta
inputs — but the relative SEVERITY (per unit of input) is substantially larger for sharp inputs,
which is itself useful information about the defect's character (it is more sensitive to
high-frequency/localized input content than to smooth, broad input content).

**Not yet done**: comparing the SHAPE (not just the range) of the oscillation between the two
perturbation types more rigorously (e.g. cross-correlation of the two trajectories at each depth)
to see whether they're the same oscillatory mode at different amplitudes, or genuinely different
modes.
