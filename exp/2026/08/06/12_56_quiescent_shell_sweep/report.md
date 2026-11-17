> **SUPERSEDED by `exp/2026/08/06/13_04_quiescent_shell_sweep_corrected`.** This run seeded the IC
> on the coarse mesh BEFORE refining the shell, so `_prolong_u_gradient_wide` interpolated real
> nearby field structure into the new children at split time — the "pre-step nonzero even in pure
> vacuum" values below are a real but confounding artifact of THAT ORDERING, not evidence about the
> shell location itself. User caught this ("does the split cells need prolongation on IC? can we
> not just set them analytically?") — the corrected run refines first, then seeds the IC directly
> on the final mesh (no prolongation at all), giving an unambiguous zero baseline in the vacuum
> shells. The qualitative conclusion (no runaway growth anywhere) still holds and is reconfirmed,
> more cleanly, in the corrected report — kept here for the record, not deleted.

# Quiescent/growing-annulus shell sweep (user-designed) — single shallow (L0→L1) junctions stay bounded even overlapping the spheromak; no runaway growth anywhere

## Motivation (user's own experiment design)

User proposed a cleaner isolation than anything run so far: build a large, uniform R=100 domain
with the spheromak occupying 45% of the radius (90% of the diameter), zero (Dirichlet) BC, and
compare one step of (a) a fully uniform mesh against (b) the identical setup with a thin shell of
cells refined to L1 sitting in the vacuum zone between the spheromak and the boundary. "At the
refined area basically nothing should happen — if something happens, that points to a problem."
Then, per the user's follow-up, progressively grow the shell's inner edge inward until it overlaps
the spheromak itself, to see where (if anywhere) a transition appears.

**One real bug caught before trusting the first attempt**: the test script called
`refine_region_to_convergence` without first setting `cl.info.split_hi`/`split_lo` (the real
`_build_telescope` always sets these, `0.5`/`0.0`, before calling it) — without them the split
criterion never crossed the default threshold and NOTHING split (`N_after == N_before` exactly,
identically zero difference). Caught by checking the leaf-count delta directly, not by trusting
the aggregate H/H0 (which was trivially `1.000000` in both — because the two runs were, in fact,
identical). Fixed and reran as a sweep.

## Method

`R=100`, spheromak `scale=45` (R_sph boundary at r=45), `amplitude=1.0` (real IC, both `u` and
`phi`), Dirichlet(0) BC. Baseline: pure uniform, no refinement anywhere. Five shell positions, each
`[lo, 47.6]`, refined to L1, `lo` moved progressively inward: `47.4` (thin, well outside the
spheromak) → `46.0` → `45.0` (touching the spheromak's own edge exactly) → `43.0` → `40.0` (both
now overlapping into the spheromak's real field). One step (`TimeStepper.by_level` once any
refinement is present), compared pre-step vs. post-step `max|u|`/`max|phi|` within each shell
region.

## Results

| shell [lo, 47.6] | N refined | pre max\|u\| | post max\|u\| | pre max\|phi\| | post max\|phi\| | H/H0 |
|---|---|---|---|---|---|---|
| baseline (none) | — | — | — | — | — | 1.000000 |
| [47.4, 47.6] | +185,328 | 1.182e-3 | 1.182e-3 | 0 | 7.280e-5 | 1.000000 |
| [46.0, 47.6] | +1,486,160 | 8.545e-3 | 8.543e-3 | 0 | 2.153e-4 | 1.000000 |
| [45.0, 47.6] | +2,376,660 | 1.374e-2 | 1.374e-2 | 9.893e-4 | 1.010e-3 | 1.000000 |
| [43.0, 47.6] | +4,011,176 | 1.035e-1 | 1.037e-1 | 8.230e-3 | 8.276e-3 | 1.000001 |
| [40.0, 47.6] | +6,216,236 | 2.696e-1 | 2.701e-1 | 2.075e-2 | 2.080e-2 | 1.000002 |

## Interpretation

**The shell's `u` is already nonzero BEFORE the step, even in the outermost, purely-quiescent
case** ([47.4,47.6], 2.4-2.6 domain-units clear of the spheromak's own r=45 edge). This is NOT a
stepping artifact — it's `_prolong_u_gradient_wide`'s wide-stencil prolongation, applied once at
split time, correctly (or at least deliberately) reaching far enough to pick up real nonzero field
structure from the nearby spheromak edge and interpolating a small nonzero value into the new
children. (An earlier philosophical framing in this session — that a refined cell is like
injecting a tiny gravity-well, per DST's cell-size↔mass correspondence — was corrected by the user:
the simulation's local AMR resolution is a computational accounting choice, not a stand-in for the
physical substrate's real cell size, so there's no legitimate physics reason for refinement alone
to disturb a vacuum region. Any disturbance here is either real-data prolongation or a genuine
artifact, not a predicted physical effect.)

**The step-induced CHANGE (pre vs. post) is small and bounded at every shell position tested**,
including the two that now genuinely overlap the spheromak's real dynamics (`[43,47.6]`:
`u` +0.2%, `phi` +0.6%; `[40,47.6]`: `u` +0.2%, `phi` +0.25%). `H/H0` stays at `1.000000-1.000002`
across the ENTIRE sweep — no runaway growth anywhere, even at the widest, most-overlapping shell.

**Conclusion: a single, shallow (L0→L1) coarse-fine junction — whether sitting in pure vacuum or
directly overlapping the spheromak's real gradient — does NOT reproduce anything resembling the
severe base-update blowup** (134x-455x at L8). This is consistent with, and reinforces, the
established pattern from `project_odd27_amr_junction_leak` that the severe defect requires DEPTH
(many stacked coarse-fine junctions across several levels), not a single split at any one location
— matching the already-documented "faster than linear with depth" acceleration (L4≈2.5x, L6≈18x,
L8≈134x) and today's Part 29 finding (a fully uniform mesh at L8-equivalent resolution is
perfectly stable). A single junction's own defect, even directly atop a real gradient, appears to
be genuinely small — the severe blowup is a property of the DEEP, MULTI-LEVEL telescope
specifically, not of "having a junction" per se.

**Not yet done**: this sweep used only ONE level of refinement (L0→L1). It doesn't test whether
STACKING several such shells (L0→L1→L2→...) at the same physical location — closer to how the
real telescope is actually built — reproduces the accelerating-with-depth signature. That's the
natural next step if this line of investigation continues: repeat this same quiescent/overlapping
methodology but with a genuine multi-level nested shell (matching `_build_telescope`'s own
structure) instead of a single L0→L1 split.
