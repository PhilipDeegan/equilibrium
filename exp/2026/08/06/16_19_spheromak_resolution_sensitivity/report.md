# Spheromak resolution-sensitivity test (user hypothesis) — the calibrated real-world spheromak was significantly under-resolved for this defect; blowup scales ~inverse-quadratically with cells-across-diameter

## Motivation

Two related observations from the user prompted this test. First, while discussing the perturbation
depth scans (Parts 33-35), the user noted their original session-opening proposal (a large R=100
domain with the spheromak at 45% of R, written into `CLAWD` before this investigation's AMR-junction
work began) was motivated by wanting "somewhat certain there was enough information density to
actually support the physics" — i.e. concern the calibrated spheromak might be under-resolved.
Second, directly: "it's possible our spheromak was actually too small too."

This is directly testable against the REAL blowup mechanism (not the artificial perturbation tests):
the calibrated spheromak's real-world scale (`1.3606e-3` at L8) gives only ~17.9 cells across its
diameter — clearing the documented `≥8-cell` BCC dispersion anisotropy threshold
(`project_bcc_dispersion_anisotropy`), but that threshold was calibrated for a DIFFERENT purpose
(avoiding anisotropic wave-dispersion artifacts), not for this AMR-junction blowup specifically, and
with very little margin (17.9 vs. the 8-cell minimum).

## Method

`R=30`, `max_level=8` (fixed), one bare `evolver.step()` (no cross_scale/lam/eta_h — isolated
harness, matching Part 25's own methodology), real spheromak IC via `init_spheromak_ic_refinable`,
`scale = 1.3606e-3 * mult` for `mult ∈ {1, 2, 4}` — i.e. the SAME real object made 2x/4x bigger
(more cells across it) while holding the grid resolution (L8, same finest cell size) fixed.
`region_r0` scaled with a margin factor (not fully proportional to `mult`, to keep cost tractable —
see Methodology note below) so the enlarged object stays comfortably inside the deepest, uniformly-
refined telescope shell.

**Methodology note, cost control**: an initial attempt scaled `region_r0` fully proportionally with
`mult` (matching `mult=1`'s generous 1.68x margin exactly) — this scales the volume, and thus cell
count, of EVERY shell in the telescope, not just the finest one, and was killed after `mult=2`
already reached 1.3M leaves and `mult=4`'s build was ballooning past 6GB RSS with more still to come
(would likely have needed 50GB+ for `mult=4`, more for `mult=8`). Recovered by using a fixed, more
modest margin (1.3x) computed directly from `mult`, not scaled by the earlier run's own `region_r0` —
this keeps cost from compounding across every shell simultaneously. `mult=4`'s single step also took
unusually long (~85 minutes) even at a comparatively modest 4.3M leaves — confirmed via `py-spy`
to be genuinely computing inside the native `evolver.step()` call, not stalled. Likely explanation
(user's own diagnosis, plausible and consistent with all the `[reflux-probe]` diagnostic output
filtered out of every log this session): the `.so` was built with assertions active
(`-gW 9`, confirmed via `__assert_fail` symbols present in the binary), not a release
(`-g 0`/`-DNDEBUG`) build — debug-only per-step diagnostic checks likely scale with cell count and
added substantial overhead at this size. A release rebuild is recommended before any future
multi-million-cell run.

## Results

| mult | cells across diameter | N_leaves | H/H0 | reduction vs. prior |
|---|---|---|---|---|
| 1 | 17.9 | 284,119 | **134.0728** | — |
| 2 | 35.7 | 1,298,535 | **36.9803** | 3.63x |
| 4 | 71.4 | 4,329,303 | **10.5247** | 3.51x |

`mult=1` matches every prior L8 measurement this session exactly (134.0626-134.0728, 4+ sig figs)
— harness consistent with the established repro.

## Interpretation

**Each doubling of resolution (cells across the object) produces almost exactly the same ~3.5-3.6x
reduction in `H/H0`** — a strikingly clean, near-power-law relationship:
`H/H0 ∝ cells_across^-1.85` (solving `3.6 = 2^a` gives `a≈1.85`, close to inverse-quadratic).
**This is a major, actionable finding**: the calibrated real-world spheromak was very likely
genuinely under-resolved for this specific defect — the `≥8-cell` anisotropy threshold it clears
was never designed to guarantee immunity from THIS mechanism, and the margin it provides (17.9
cells, barely over 2x the threshold) is nowhere near enough.

**This does not contradict, but refines, everything else established today about this blowup.**
Part 29 already showed a junction-FREE uniform mesh at the identical fine resolution is completely
stable (`H/H0=1.000034`) regardless of the field's own resolution — so the defect still
categorically requires a coarse-fine junction to manifest at all. What this test adds: GIVEN a
junction, the defect's MAGNITUDE scales strongly with how sharply the crossing field varies
relative to the local cell size. This is consistent with (though larger in magnitude than) the
already-characterized ancestor-substitution Jensen bias (Part 26/27), whose closed-form correction
is explicitly proportional to the field's local CURVATURE (`_ancestor_curvature_estimate`) — a
bigger, smoother object presents proportionally less curvature at any given point, including at
the telescope's outer junction, directly reducing whatever curvature-sensitive term(s) drive this.

**This also resolves an apparent tension with Part 25/35's eta_h findings**: `eta_h` had negligible
effect (~0.008%) on the real spheromak blowup (Part 25) but substantial effect (up to ~9x) on the
artificial perturbation-based depth scans (Part 35). `eta_h`'s biharmonic term specifically targets
HIGH-FREQUENCY, grid-scale noise — a genuinely different spectral component from the smooth,
low-order CURVATURE this resolution test is sensitive to. The real spheromak's blowup is dominated
by smooth curvature (which `eta_h` doesn't reach), not high-frequency noise (which it does) — while
the artificial point/small-radius perturbations are dominated by exactly the high-frequency content
`eta_h` targets. Two different quantities, two different sensitivities, both real.

**Practical implication going forward**: rather than continuing to search for a deep formula fix
to the ancestor-substitution/flux mechanism (which Part 26/27 already showed only closes a modest
~1e-5-relative fraction of the real defect), the most direct, immediately actionable path may
simply be increasing the effective resolution margin around the neutron/proton core substantially
beyond the current calibration — e.g. targeting closer to 70-150+ cells across the object (based on
today's data) rather than the ~18 the current real-world calibration provides, by either going
deeper than L8, or reconsidering whether the full physical Bohr-radius/neutron-radius ratio truly
needs to be represented at native scale in a single simulation.

**Not yet done**: extending this same test to `mult=8`/`mult=16` (to see whether the ~3.5x-per-
doubling trend continues, plateaus, or whether `H/H0` eventually reaches the ~1.0-1.14 baseline
established for a well-resolved/uniform field) — deferred pending a release (`-g 0`) rebuild, given
how expensive `mult=4` alone was under the current assertions-active build.
