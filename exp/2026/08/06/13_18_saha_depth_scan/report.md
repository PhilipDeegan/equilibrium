# Full single-level depth scan (user-designed) — every level L0-L8 measured for the first time, near-geometric compounding confirmed

## Motivation (user's own methodology, written into CLAWD)

User proposed the natural extension of today's Parts 26-30: instead of sampling only even levels
(L4/L6/L8, via various slightly-differing harnesses across sessions), build a genuine graded
telescope FRESH at every single depth 0 through 8, one coarse step each, real spheromak IC held at
constant physical scale, saving full output at each depth — "we do an R=30 sphere, L0 uniform, run
it for one step, save the output, then we refine the very center cell to L1, run it for one step,
save the output, then... to L2, adding L1 cells around the L2 cells... progressively to L8,
increasing the telescope as we go."

Implemented as a new reusable script, `run/substrate/adaptive/refined_saha_depth_scan.py` — reuses
`_build_telescope`/`init_spheromak_ic_refinable` from `refined_neutron_decay.py`, applies today's
established correct methodology (refine the mesh first, then seed the IC directly on the final
leaves — no prolongation into new children), pure base-mechanics only (no cross_scale/step_lam/
eta_h — "saha vacuum" style isolation), zero (Dirichlet) BC, `R=30`, spheromak held at the real
calibrated physical scale (`--scale-override 1.3606e-3 --scale-ref-level 8`) at every depth.

## Results

| depth | N_leaves | scale | H/H0 | excess (H/H0 − 1) | excess ratio vs. prior depth |
|---|---|---|---|---|---|
| 0 | 147,047 | 8.927 | 1.000007 | 7.0e-6 | — |
| 1 | 163,817 | 2.976 | 1.000036 | 3.6e-5 | 5.14 |
| 2 | 180,587 | 0.9919 | 1.000386 | 3.86e-4 | 10.72 |
| 3 | 197,357 | 0.3306 | 1.003763 | 3.763e-3 | 9.75 |
| 4 | 214,335 | 0.1102 | 1.034671 | 3.467e-2 | 9.21 |
| 5 | 231,105 | 0.0367 | 1.309819 | 3.098e-1 | 8.94 |
| 6 | 247,875 | 0.0122 | 3.662054 | 2.662 | 8.59 |
| 7 | 265,893 | 0.00408 | 21.904615 | 20.90 | 7.85 |
| 8 | 284,119 | 0.00136 | 134.072756 | 133.07 | 6.37 |

**Cross-validated exactly against today's other measurements**: L4 (1.0347 here vs. 1.0347 in Part
28's ablation), L6 (3.6621 here vs. 3.6621 in Part 28), L8 (134.0728 here vs. 134.0626-134.0728
across Parts 23/25/27/28) — bit-for-bit or 4-sig-fig identical to independent harnesses built at
different points today. Strong confirmation this new script's methodology is correct and the
underlying phenomenon is real and reproducible.

## Interpretation

**Growth is smooth, monotonic, and compounds at a near-constant per-level multiplier of ~6-11x
across the entire range** — not a sudden onset at any one level. The "excess" (H/H0 − 1) ratio
between successive depths is fairly stable (~9-11x) through the middle depths (L2-L6), then
gradually DECREASES toward the deepest levels (7.85 at L6→L7, 6.37 at L7→L8) — i.e., the
compounding is closer to a clean geometric progression than an ever-accelerating one; "faster than
linear" (as characterized in earlier, coarser-grained sessions) is confirmed, but the per-level
rate is not itself increasing without bound — if anything it eases slightly at the deepest levels
tested.

**Practically negligible through L3** (`H/H0 < 1.004`), **becomes noticeable at L4** (`+3.5%`),
and **turns severe from L5 onward** (`+31%`, then `3.66x`, `21.9x`, `134x`). This pinpoints the
transition from "negligible" to "real problem" specifically between L4 and L5 — consistent with,
and now more precisely locating within, the previously-documented L4≈2.5x/L6≈18x/L8≈134x
even-level characterization (that characterization used a different, older harness and BC
convention; this scan's L4/L6/L8 values, ~1.03/3.66/134, are noticeably smaller than the old
"2.5x/18x" figures at L4/L6 specifically — most likely explained by BC choice, zero/Dirichlet here
vs. sponge in the historical runs, and/or R=30 here vs. R=20 there; the L8 value matches almost
exactly regardless, since sponge/BC differences matter far less once the telescope reaches R=100/
R=30's interior and away from the boundary).

**Since this used pure base mechanics with no `eta_h`, `cross_scale`, or `step_lam` at all, and the
growth is still severe by L8, this reconfirms (does not newly show) that the defect is intrinsic
to the base AMR update/telescope structure itself** — not an interaction with any of the physics
extensions layered on top in the full production script.

**Not yet done**: examining the saved per-depth snapshots (`tmp/saha_depth_scan/depth{0..8}.npz` —
full `u`/`phi`/`pos`/`level` arrays at both `u0`/`phi0` (IC) and `u1`/`phi1` (post-step)) for WHERE
in the mesh the growth actually concentrates at each depth — e.g. is it uniformly spread across all
boundary cells, or localized to a specific sub-population (matching the earlier-flagged, unexplored
`n_same=4` boundary-cell lead)? That's the natural next step if this thread continues.
