# Level-border error: uniform L0 baseline vs single L0/L1 junction

## Motivation

Clarification from prior confused attempts: there is no such thing as a
"uniform L1 lattice" — a level only exists relative to L0 (it's a
refinement-depth label, not an absolute resolution). The two runs below fix
that: one is genuinely uniform (no refinement anywhere, cell size = 1
everywhere), the other has exactly one real L0/L1 junction.

## Configs

Both use the same spheromak IC (Taylor-state B-field, `init`-style B0=1,
R_SPH=1, normalized u/phi peak=1) at `cells_across=220` (the resolved-spheromak
threshold from `project_hop_telescope_resolved_spheromak` — avoids the
under-resolution-driven blowup that plagued the old ~18-cells-across
calibration).

- **Uniform L0** (`build_uniform_l0.py`): `d1=1` everywhere, no refinement.
  Spheromak radius = 110 absolute units (220/2). Domain radius R=140
  (padding=30 cells, within the requested 20-40 range). 14,930,859 leaves.
- **R60 L0 / L1 at R40** (`build_l0_l1_boundary.py`): L0 domain radius=60,
  cell size 1; region r<=40 refined once to L1 (cell size 1/3) — exactly one
  coarse-fine junction, at r=40. Spheromak radius = 36.667 absolute units
  (220 cells across *at L1's own resolution*), margin to the junction = 3.33
  absolute units = 10 L1-cells. 10,222,517 leaves (827,246 @ L0, 9,395,271 @ L1).

Both give **H0=6.114140e+06** — an exact cross-check that the "220 cells
across" normalization is being enforced consistently regardless of absolute
scale or mesh construction (u/phi are built from a scale-invariant normalized
profile, so H0 depends only on cells-across, not on the absolute SCALE or R).

## Stepped results (`_compute_dt_l0` CFL dt, `TimeStepper.by_level`, 8 threads)

| step | uniform L0, H/H0 | R60/L1@40, H/H0 |
|---|---|---|
| 0 | 1.000000 | 1.000000 |
| 1 | 1.000000 | 1.000000 |
| 2 | 1.000000 | 1.000001 |
| 3 | 1.000000 | 1.000003 |
| ... | (flat through step 10) | (not run past step 3) |
| 10 | 1.000001 | — |

At step 3 the junction case's growth (H/H0-1 ≈ 3.1e-6) is already ~15x the
uniform baseline's (≈2.1e-7) — a real, measurable junction-driven signal,
even though both are still utterly tiny in absolute terms. Consistent with
`now`'s finding that the *bulk* wavefront is still frozen near the core after
only a few steps (only a ~1e-12-amplitude numerical precursor has reached
this far out this early) — this comparison is only sampling the earliest
precursor reaching the r=40 boundary, not the mature bulk/junction
interaction. Would need many more steps (the earlier L5 case needed ~150
steps for the bulk to reach even 1% of H in the next level out) for this
junction to be stress-tested by the real wave, not just its precursor.

## Files

- `build_uniform_l0.py` / `build_uniform_l0.log` — IC build, uniform case
- `build_l0_l1_boundary.py` / `build_l0_l1_boundary.log` — IC build, junction case
- `step_uniform_l0.py` / `step_uniform_l0.log` — 10 steps, saved to `/bk/uniform_l0_stepped/step_{01..10}.h5`
- `step_l0_l1_boundary.py` / `step_l0_l1_boundary.log` — 3 steps, saved to `/bk/l0_l1_boundary_stepped/step_{01..03}.h5`

## Next

- Step the junction case further (matching the uniform case's step count, or
  well past it) to let the real bulk front actually reach r=40 and see
  whether the junction-driven excess growth stays proportionally small or
  compounds the way the deeper L4-L8 junctions did.
- The margin between the spheromak edge and the junction here is only ~10
  L1-cells — thin. Worth a second junction-case variant with more margin
  (e.g. push R_L1 out, or shrink the spheromak) to separate "junction defect"
  from "spheromak too close to the boundary" as confounds, per the existing
  boundary-buffer-check convention from `project_hop_telescope_resolved_spheromak`.
