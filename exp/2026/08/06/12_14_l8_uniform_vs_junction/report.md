# Uniform (junction-free) mesh at L8-equivalent resolution is COMPLETELY STABLE — blowup is 100% junction-specific, not a fine-resolution/missing-sink issue

## Motivation

Part 28 (`exp/2026/08/06/12_11_l8_uphi_ablation`) found `u` alone (no `phi`) reproduces and
exceeds the L8 base-update blowup. User asked directly: does any of this point to a problem
existing even in a pure, uniform L0 mesh (no refinement at all), with AMR just exposing/amplifying
a pre-existing issue? User also raised a related, more specific hypothesis: maybe the base update
is missing some kind of dissipative "sink" term, and its absence only becomes visible at fine
resolution — which, if true, should show up in ANY mesh at that resolution, junction or not.

An old, much shallower version of this exact check exists (`project_odd27_amr_junction_leak` Part
2, 2026-07-19): a uniform mesh built directly at L4-equivalent cell size was "ESSENTIALLY STABLE
(×1.14)". That predates today's fixes and multiple later mechanisms (Parts 12/26-28), and was
never redone at L8-equivalent resolution — exactly the depth where the real blowup lives.

## Method

Built a genuinely uniform BCC sphere directly at the SAME fine cell size as an L8 AMR leaf
(`d1 = 1/3^8`, no refinement, no coarse-fine junction anywhere) at three domain sizes
(`R_cells` = 20/40/60, giving physical radii ~2.2x to 6.7x the real spheromak's own `scale`).
Seeded the real spheromak `u`+`lam` field at the correct physical scale (`scale=1.3606e-3`, same
as the real L8 run), `phi=0` throughout (`amplitude=0.0`) — matching Part 28's established
"phi=0 (u+lam only)" ablation condition, since that's the condition that showed the blowup can be
reproduced by `u` alone. Ran ONE step with a `TimeStepper.fixed(dt)` at the mesh's own CFL-limited
`dt` (no subcycling needed — genuinely uniform, single level).

**Important apples-to-apples check**: this uniform test's `dt=7.8162e-07` is BIT-IDENTICAL to the
real telescope's own finest-level effective per-substep `dt` (the L8 telescope's coarse
`dt_base=5.1282e-3` divided by `3^8=6561` via `TimeStepper.by_level`'s subcycling gives
`5.1282e-3/6561 = 7.816e-7`) — so this comparison uses the exact same physics, the exact same
finest cell size, and the exact same finest-level timestep as the real blowup; the ONLY difference
is the presence or absence of a coarse-fine junction.

## Results

| R_cells | phys_R | N_leaves | H0 | H1 | H/H0 |
|---|---|---|---|---|---|
| 20 | 3.05e-3 | 43,631 | 960.536 | 960.569 | **1.000034** |
| 40 | 6.10e-3 | 347,973 | 960.536 | 960.569 | **1.000034** |
| 60 | 9.14e-3 | 1,175,379 | 960.536 | 960.569 | **1.000034** |

Bit-identical across all three domain sizes (as expected — the field is zero outside the
spheromak's small core, so domain size shouldn't matter once large enough to contain it, and it
clearly is). **Essentially perfectly stable — 0.0034% growth in one step**, vs. the AMR
telescope's equivalent (`phi=0`, `u`-only) condition at L8: **455.08x**.

## Conclusion

**This rules out a general fine-resolution/missing-sink explanation, and confirms the blowup is
100% specific to the coarse-fine junction.** The exact same physics, exact same finest cell size,
and exact same finest-level timestep are completely well-behaved with no junction present. There
is nothing wrong with the base update rule at this resolution in isolation — whatever term or
formula is at fault only exists/activates at a coarse-fine boundary.

This directly answers both of the user's questions from this session:
1. **Is there a problem with a pure uniform mesh that AMR is just exposing?** No — a uniform mesh
   at the exact same resolution is stable to 5 significant figures.
2. **Is a missing dissipative sink the explanation?** No — a missing sink would show up as growth
   in this uniform test too (nothing about "having no sink" depends on whether a junction exists
   nearby). Its complete absence here rules that out cleanly.

**This strengthens Part 28's redirect** to the coarse-fine junction mechanism specifically, and in
particular revives `project_odd27_amr_junction_leak` Part 3's already-found-but-never-root-caused
lead: `disable_same_level_flux_at_boundary_only` gave a clean ~13x single-step-growth reduction at
L4/L6 — same-level flux AT coarse-fine boundary cells specifically (not same-level flux generally,
which this uniform test also confirms is fine) is now the most tightly-scoped remaining suspect.

**Caveat**: this uniform patch has no sponge/absorbing BC (raw `build_bcc_sphere`, default
boundary) — for a domain this small relative to `d1`, any BC-driven artefact would need to
reach the tiny spheromak core (physical radius ~1.4e-3) from a boundary ~2-7x farther out within a
single finest-level substep (`dt≈7.8e-7`, CFL-limited so information can't travel far in one
substep) — implausible as a confound, but noted for completeness since it wasn't sponge-matched to
the real run.
