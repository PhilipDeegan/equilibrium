# L8 revalidation of `enable_ancestor_debias_correction` — ruled out as the base-update blowup driver

## Context

`project_odd27_amr_junction_leak` Part 26 (2026-08-05, `now`) characterized and fixed a real,
substantial cross-level flux defect (the "ancestor Jensen-bias") in isolated synthetic-field
tests: `enable_ancestor_debias_correction` closed ~82% of a `vs_unsplit` aggregate-flux
discrepancy (19.3% -> 3.6% relative error), confirmed at 1-level and 2-level gaps, and confirmed
to survive AMR reflux. Not yet connected with certainty to the still-open base-update L8 blowup
(`H/H0≈134` at step 1, Part 23/25) — that revalidation is what this experiment does.

## Setup

`enable_ancestor_debias_correction` and `disable_step_pass2_amr_correct` existed in
`inc/eq/core/data/lattice/adaptive_lattice.hpp` as static class members but were never exposed to
Python (`src/eq/lattice.cpp` had bindings for several other AMR toggles but not these two). Added:
- `set/get_enable_ancestor_debias_correction`
- `set/get_disable_step_pass2_amr_correct`

Rebuilt via `KLOG=2 mkn build -KOqp mod,mod_lattice,test_lattice -x mpi -gW 9 -d`. Verified zero
regressions via `KLOG=2 mkn build -Oqp test_lattice -x mpi -gW 9 run test` — only the
pre-existing, already-documented `test_curvature_corrected_secant_formula_isolated` failure,
unchanged.

Added `--enable-ancestor-debias` CLI flag to `refined_neutron_decay.py` and
`refined_saha_vacuum.py`.

## Method

1. Reran the exact Part 23/25 L8 repro command
   (`exp/2026/08/04/14_29_l8_etah1e-13_scaleoverride_r20/run.sh`) to confirm the harness/build
   hasn't drifted: baseline.
2. Same command + `--enable-ancestor-debias`: debias.
3. Since (1) and (2) came back visually bit-identical, ruled out a wiring gap (toggle silently
   ignored) by bypassing the script entirely — built the identical mesh/IC directly via
   `AdaptiveLattice`/`AdaptiveLatticeEvolver`, confirmed IC arrays bit-identical on/off, ran one
   bare `evolver.step()` call on/off, and diffed the raw per-leaf `u` arrays (not just the
   aggregate `H`). Ran this diff check at both `max_level=4` (cheap) and `max_level=8` (the real
   blowup mesh).

## Results

| run | H | H/H0 |
|---|---|---|
| baseline | 4.393687e+05 | 134.0626 |
| `--enable-ancestor-debias` | 4.393687e+05 | 134.0626 |

Bit-identical to displayed precision, in both the full script run (with eta_h ladder, cross_scale,
dynamic lam all active) and a bare-mechanics-only diff check.

Raw per-cell diff (bare `AdaptiveLattice`, one `evolver.step()`, no eta_h/cross_scale/lam):

| max_level | leaves changed | max\|Δu\| | max\|u\| | relative magnitude |
|---|---|---|---|---|
| 4 | 46,521 / 110,887 (42%) | 1.4e-9 | ~1.0 | **~1.4e-9** |
| 8 | 122,724 / 180,671 (68%) | 2.4e-5 | ~0.98 | **~2.45e-5** |

`ΔH/H` between on/off at L8 (bare mechanics): ~4.8e-8.

## Conclusion

`enable_ancestor_debias_correction` is confirmed **firing correctly** on the real spheromak
telescope mesh — not a dead/no-op toggle, and the correction's magnitude even grows with depth
(1.4e-9 at L4 -> 2.45e-5 at L8), consistent with everything else in this investigation trending
worse with depth. But even at L8, on the real mesh, it is **4+ orders of magnitude too small** to
move a 134x blowup by any visible amount.

**Why the isolated tests showed ~1 (order-unity, 19%->3.6%) and the real mesh shows ~1e-5**: the
correction's magnitude is proportional to the actual local curvature of the `u`-field at each
cross-level junction (`_ancestor_curvature_estimate`). The isolated tests used a synthetic
quadratic field with O(1) curvature *everywhere*, deliberately chosen to make the effect
measurable. The real spheromak IC has `u≈0` (flat background) everywhere except a tiny core
(27-3839 cells out of 110K-180K total) — there is almost no real curvature at the telescope's
outer cross-level junctions for this correction to act on. Both measurements are correct; they're
evaluating the same formula on fields with wildly different curvature magnitudes at the junctions
that matter.

**This closes the open question from Part 26**: the ancestor-substitution Jensen-bias mechanism
is real and now fixed in general, but it is not (or not a significant contributor to) the
base-update L8 blowup. That blowup's driver remains open. Ruled out so far, cumulatively across
sessions: eta_h (any value), `step_cross`'s gate, domain size, per-face multiplicity correctness,
boundary coupling magnitude, and now the ancestor Jensen bias.

**Next lead, not yet pursued**: the blowup appears in ONE coarse step with `u≈0` background nearly
everywhere, in a regime where the u-curvature bias is ~0 — but `phi` (unlike `u`) has real
structure domain-wide at IC (`phi_rms≈0.04`, from the spheromak's own field). Worth checking
whether an analogous restriction-bias mechanism exists on the `phi` side of the update, or in how
`step_cross`/the gate interacts with a boundary carrying a real cross-level `phi` discontinuity.

**Recommendation**: promote `enable_ancestor_debias_correction` to on-by-default regardless — it's
a real, verified, zero-cost-when-inert isotropy improvement — but don't expect it to change the L8
blowup number when doing so.
