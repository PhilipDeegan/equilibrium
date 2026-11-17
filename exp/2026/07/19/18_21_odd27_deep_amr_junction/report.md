# Odd27 deep-telescope AMR junction leak — two bugs fixed, one structural cause found (unfixed)

## Context

Follow-on to the same-day Odd27 (ratio-3) AMR junction-leak investigation (single adjacent-level
boundary, smooth Gaussian IC — see memory `project_odd27_amr_junction_leak.md` part 1). User asked
for a real test: `tst/substrate/adaptive/refined_neutron_decay.py` (genuine spheromak IC, real
`step_cross`/`step_lam`/sponge physics) at `--max-level` 0/2/4/6, 100 steps, using
`--scale-override`/`--scale-ref-level` to keep the object proportionally resolved at every depth.

## Result: severe, depth-accelerating energy growth in the real script

H = sum(u^2) + 0.5*sum(phi^2), R=20, region_r0=5.0, scale_override=0.01 @ scale_ref_level=6:

| max_level | n_leaves | H_0 | H_f (100 steps) | growth |
|---|---|---|---|---|
| 0 | 43599 | 1822.5 | 1886.2 | x1.03 |
| 2 | 77139 | 1822.5 | 3084.4 | x1.69 |
| 4 | 110887 | 1822.5 | 19647 | x10.8 |
| 6 | 144427 | 1822.5 | 311000 | x170.7 |

Not present in an earlier synthetic C++ stand-in test (smooth off-center Gaussian, graded
telescope) — that test showed L2/L4/L6 tracking each other closely (stable). The real spheromak's
much sharper, hard-cutoff IC (R_sph=1, no Gaussian taper) is what exposes this.

## Isolation

1. Disabled `step_cross`/`step_lam`/sponge one at a time (`--cross-scales 0`, `--lam-mode frozen`,
   `--damp-max 0`) — runaway persisted identically with all three off. Isolated to `evolver.step()`
   (the wave/AMR-subcycle path) alone.
2. Graded telescope (L0->L1->L2->L3->L4 nested shells) vs a single direct L0->L4 jump (one
   `refine_region_to_convergence` call, no intermediate shells) at the same max_level: similar
   severity (x10.98 vs x9.44) — ruled out "more nested junctions" as the driver.
3. Genuinely uniform, zero-junction mesh at the level-4-equivalent cell size (`build_bcc_sphere`
   with `d1 = 1/3**max_level` directly, no refinement) — stable (x1.14). Ruled out base
   CFL/resolution instability. (Hit and fixed a seeding bug along the way:phi face-hop scaling
   assumes `leaf_level` reflects true resolution, false for a mesh built directly at fine `d1`.)
4. Pushing the boundary FARTHER from the steep IC (region radius 0.185->0.4) made it WORSE
   (x9.44->x14.54), not better — ruled out "boundary too close to gradient"; severity tracks total
   junction/fine-cell footprint.
5. Disabled `_apply_amr_reflux` entirely (temp static toggle) — growth got WORSE (x10.98->x12.75),
   confirming reflux is a partial legitimate mitigant, not the source.

## Root cause #2 (FIXED): cross-cell ancestor-sharing over-representation

`_step_pass2_lev`/`_step_pass2`: when a fine leaf's neighbour is coarser, the code substitutes an
ancestor's `u` (walked up via `cell.parent` until `L >= nb.L`) for the leaf's own `u_i` as the flux
source — a deliberate, dimensionally-reasonable scale-matching choice. But every fine leaf sharing
that same ancestor independently applies the identical ancestor-derived flux to its own `u` update,
over-representing one physical (ancestor, neighbour) connection by however many leaves share it
(potentially thousands at a 4+ level gap, vs ~27 at a 1-level gap — why the earlier single-boundary
fix didn't cover this).

Fix: new `AdaptiveCellCluster::ancestor_group_multiplicity` (built by
`rebuild_ancestor_group_multiplicity`, counts per-level how many leaves share each (ancestor,
neighbour) pairing), applied as a second divisor alongside `face_multiplicity`.

Verified: L4 telescope vs direct-jump — before x10.98 vs x9.44 (topology-dependent); after
**x2.5149 vs x2.5149, identical to 4 sig figs**. Confirmed end-to-end via the real script (x2.55).
Confirmed bounded across 100/400 steps @ R=20 and 1000 steps @ R=50 — all x2.5-2.7, oscillating not
monotonic (healthy resolution-dispersion signature, not runaway).

## Root cause candidate that was NOT it (fixed anyway, real bug)

`accumulate_to_parent` restricted `.u`/`.phi` bottom-up every substep but never `.lam` — non-leaf
coarse-proxy cells had permanently stale `.lam` (frozen at split time), so `sigma()=1/(1+lam)` was
wrong wherever a coarse leaf read a split neighbour's aggregate state. Real bug, fixed (`p.lam =
mean(children.lam)`), but A/B tested and changed NOTHING (L4 x2.52 vs x2.52, L6 x17.78 vs x18.86,
L8 x96.2 vs x98.1, all within noise). Kept for correctness; not the explanation for what follows.

## Remaining residual (L6/L8) — NOT FIXED, structural

Even with both fixes: L4 ~x2.5-2.7 (bounded, production-safe), **L6 ~x17.78-18.86** (100-150
steps), **L8 ~x96-98** (20-step sanity probe only). Implied per-*substep* growth rate is nearly
identical at all three depths (~0.0027-0.0035%) — `step_subcycle`'s substep count is `ratio^level`
(81/729/6561 at L4/L6/L8), so a tiny constant per-substep bias compounds multiplicatively into wildly
different totals purely from substep count.

Cause: `step_subcycle`'s loop (`for lev=0..max_lev { for sub=0..ratio^lev {...} }`,
adaptive_lattice.hpp ~line 671) is not properly interleaved/recursive (Berger-Oliger-style) — level
0 completes its entire `dt_base` step in one shot before level 1 even starts its 3 substeps, so
every finer level's substeps see their coarser neighbour's already-fully-advanced (end-of-step)
state throughout, rather than a time-interpolated value. User's framing: "double stepping" — the
same already-arrived coarse state gets effectively over-applied once per fine substep. Proper fix
needs real restructuring (time-interpolated coarse ghost values across the subcycle sequence) —
deliberately not attempted this session (user chose to consolidate rather than open that scope).

## Practical guidance

L2/L4 telescopes: confirmed production-safe. **Do not trust L6+ telescopes** (or any 5+ level gap
junction) until the coarse-first subcycle ordering is fixed.

## Artefacts

- Code changes: `inc/eq/core/data/lattice/{adaptive_lattice.hpp,cluster.hpp,bcc_geometry.hpp}`,
  `src/eq/lattice.cpp` (two temporary diagnostic toggles: `disable_amr_reflux`,
  `disable_ancestor_u_src`, both default false).
- Test fix in passing: `tst/eq/core/data/lattice/uniform_vs_adaptive.cpp` (stale `build_bcc_sphere`
  call, same bug already fixed in `adaptive_lattice.cpp` earlier the same day).
- Full narrative + numbers: memory `project_odd27_amr_junction_leak.md` (part 2 section).

## Next steps

- Fix `step_subcycle`'s coarse-first ordering for proper time-interpolated coarse-fine coupling
  before trusting L6+ telescopes.
- Consider removing the two temporary diagnostic toggles once that work lands (or keep if useful
  for it).
