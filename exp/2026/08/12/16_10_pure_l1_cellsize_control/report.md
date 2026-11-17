# Pure L1-cell-size control test — junction confirmed as the driver, not scale-covariance

## Context

`exp/2026/08/11/21_27_everything_active/` ran two "everything active" (dipole-mode
IC, `step_cross_all(cross_scale=1e-3)`, `lam` seeded from t=0 energy density then
frozen) configs to step 130:

- **uniform_r180_active**: single-level (L0, cell_size=1), no junction. H/H0=0.999539.
- **l1_wider_margin_active**: L0 shell (R=46.667→60, vacuum) + L1 core (cell_size=1/3,
  r<46.667, contains the whole spheromak + 30 L1-cell margin). H/H0=1.022056 — crosses
  above 1 and is still accelerating at step 130.

`H_by_level` on the L1 run showed the L0 shell's own H contribution at ~1e-50 at step
130 — the field's bulk has **not** reached the L0/L1 junction (r=46.667) in any
physically meaningful sense. So the observed growth could not yet be "real physics of
crossing into a coarser region" — it was happening entirely inside the L1-only core,
which raised two candidate explanations:

  (a) the junction/proxy bookkeeping (ancestor-substitution, multiplicity correction)
      is injecting energy into boundary-adjacent L1 cells even before bulk field
      arrives there, or
  (b) something scale-dependent (`cross_scale`, frozen `lam`) isn't actually
      scale-covariant at cell_size=1/3, independent of any junction/AMR machinery.

## Control mesh

`build_uniform_l1size_pure.py` / `step_uniform_l1size_pure.py`: a **pure single-level**
mesh built directly at `d1=1/3` (`build_bcc_sphere(..., d1=1/3, ...)`) — no refinement,
no proxies, no AMR machinery of any kind — same physical domain radius (60), same
spheromak `SCALE` (36.667, `cells_across=220` at this cell size, matching
`build_l1_wider_margin_active.py` exactly), same dipole IC, same `lam` seeding formula,
same `cross_scale=1e-3`. n_leaves=31,734,403 (vs uniform_r180's 31,734,323 and
l1_wider's 15,542,429), H0=4.462552e+06 (vs 4.462518e+06 / 4.462388e+06 — all three
ICs agree to 5 sig figs, as expected for the same normalized dipole shape).

If (b) were the explanation, this mesh — same absolute cell size as L1, zero junction
— should reproduce the growth. If (a), it should stay flat/decay like the coarse
reference.

## A real methodological bug, caught and fixed mid-run

The first pass used `dt = _compute_dt_l0(cl, D1=CELL_SIZE)` (`CELL_SIZE=1/3`) as the
**outer** step increment directly. That's the correct dt for this mesh's own CFL, but
it made each outer "step" 3x smaller in real elapsed time than the other two runs'
outer steps (`dt=1.709402e-03` here vs `dt=5.128205e-03` in both 08/11 runs — exactly
3x, confirmed from the printed `dt=` line in each log). `TimeStepper.by_level` only
subcycles levels *below* the top of a genuine multi-level mesh; here everything is
nominally "level 0" despite the small absolute cell size, so no subcycling happened —
the loop just advanced 3x slower per iteration than intended. First-pass step 130
(H/H0=0.969523) was NOT time-matched to the other runs' step 130 and is not
comparable. Fixed by resuming 260 more steps (to outer-step 390 = 390×1.709402e-03 =
130×5.128205e-03 exactly) to reach the true time-matched point.

## Result (time-matched: real elapsed time = 130 × dt_L0, same as both 08/11 runs)

| config | cell size | junction? | H/H0 | direction |
|---|---|---|---|---|
| uniform_r180_active | 1 | no | 0.999539 | flat, mild recovery toward 1 |
| **pure_l1size_control** | 1/3 | **no** | **0.912617** | monotonic decay, *stronger* than the coarse case |
| l1_wider_margin_active | 1/3 (core) | **yes** | **1.022056** | crosses 1, still accelerating |

## Conclusion

Neither junction-free reference grows. Both decay under `cross_scale=1e-3` (dissipative,
as expected) — the *finer* one decays substantially *more* (0.913 vs 0.9995 at cell
size 1 vs 1/3), the opposite of what (b) would predict if a fixed `cross_scale`/`lam`
formula were somehow under-throttling small cells. This rules out (b): plain
scale-dependence of `cross_scale`/frozen-`lam` is not the cause, and if anything cuts
against the L1 run's growth, not toward it.

The **only** configuration that gains energy is the one with an actual L0/L1 junction
present — even though bulk energy (per `H_by_level`) has not reached the coarse shell.
This confirms (a): the junction/proxy bookkeeping itself is injecting energy into the
boundary-adjacent region of the finer level, independent of whether real field has
crossed there yet. This is a genuine, previously-uncharacterized defect distinct from
the historical L4+ "deep-telescope blowup" investigation (`project_odd27_amr_junction_
leak`) — same family of symptom (junction-adjacent H growth), but caught here at the
shallowest possible depth (a single L0/L1 gap) under real, dynamic, all-physics-active
conditions (frozen lam + step_cross), not the isolated/static harness the deep-telescope
work used.

## Checkpoints / artefacts

- `/bk/uniform_l1size_pure_ic220.h5` (t=0 IC)
- `/bk/uniform_l1size_pure_stepped/step_130_final.h5` (NOT time-matched — see above, do
  not compare directly against the other runs' step 130)
- `/bk/uniform_l1size_pure_stepped/step_390_final.h5` (time-matched final state)
- `plots/xz_step_{010,020,...,390}.png` — XZ midplane renders

## Next

Investigate `enable_cross_level_gradient_correction` / design a real ghost-cell
same-level-shadow measurement to pin down exactly which part of the junction
bookkeeping is responsible — see follow-up discussion, not yet started as of this
report.
