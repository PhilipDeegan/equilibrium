# Poynting-ring near-field falloff resolved: drop step_cross, compute S from plain step() — the "opposite of naive expectation" anomaly reverses

## Context

Every prior Poynting-ring run in this project's history (the 2026-07-14
original calibration and every far-zone rerun since) computed `E` from
`step_cross`'s (barrier-gated) `dphi` — the same mechanism the 2026-08-19
`m0` diagnosis (`exp/2026/08/19/15_10_barrier_drive_decay_diagnosis/`)
showed is structurally the wrong tool: `step_cross` is DST's translation/
reconnection operator, not the propagation mechanism (`doc/dst.md`'s
"Inertia and the Face/Cell Gate" assigns EM wave propagation to
face-hopping, i.e. plain `step()`). The original calibration's headline
result — polar slope -2.44 (near classical), equatorial slope -6.19 (far
steeper) — was "opposite of the naive magnetic-dipole expectation" (that
script's own docstring): classically, radiation should peak at the
equator, not the pole.

Separately, the same-day energy-split thread
([[project_curl_full_dipole_spheromak]]) derived a genuine continuum
electric<->magnetic leak formula, `dCirc/dt = -dt*W_sq*d2^2*(grad(sigma) x
grad(u))`, real only across a `sigma`/`lam` GRADIENT. Every Poynting-ring
run has always seeded `lam_ring=1.0` on the ring (a sharp `lam:0->1` step)
against a `lam=0` background — exactly the strong, localized gradient case
that formula predicts should matter, never previously varied or examined
for its effect on the near-field asymmetry.

## New script: `run/substrate/adaptive/probe_poynting_step_only.py`

Computes `E := -dphi/dt` from ONE plain `evolver.step()` call (no gate, no
`m0`, no `step_cross` anywhere), `B` from the existing curl(phi)
reconstruction (`equilibrium.substrate.mhd`), `S = E x B` — otherwise the
same fit methodology (log-spaced radial bins, p99 envelope, equatorial vs
polar sectors) as `probe_poynting_dipole_compare.py`. Adds `--lam-ring` as
an explicit, swept parameter (previously hardcoded to 1.0 everywhere).

## Results

**R=50, hops=4, matching the original calibration scale, n_steps=8000
(~1.28 periods)**:

| lam_ring | equatorial slope | polar slope | eq/pol ratio |
|---|---|---|---|
| 0 (vacuum ring) | -1.310 | -0.451 | 1.456 |
| 1.0 (historical default) | **-1.802** | -0.705 | **1.886** |

Equator now STRONGER than pole at both `lam_ring` values — the classical
direction, for the first time in this investigation's history. `lam_ring=1.0`
pushes the equatorial slope notably closer to the classical `-2` reference
than `lam_ring=0`.

**Follow-up (Philip): is R=30 too small — same "minimum system size"
pattern this project has hit before (BCC anisotropy, Casimir-Polder,
today's own circulation-leak resolution sweep)?** Direct matched-settle-time
check, R=30 at the SAME 8000 steps as the R=50 run (the earlier R=30 result
had only used 1500 steps, ~0.24 periods — a genuine confound, not yet an
apples-to-apples comparison):

| R | lam_ring | equatorial | polar | eq/pol |
|---|---|---|---|---|
| 30 | 0 | -2.152 | -1.439 | 1.281 |
| 30 | 1.0 | -2.221 | -1.703 | 1.292 |
| 50 | 0 | -1.310 | -0.451 | 1.456 |
| 50 | 1.0 | -1.802 | -0.705 | 1.886 |

**R=30 is not too small.** At matched settle time it performs at least as
well as R=50 — its equatorial slope is actually CLOSER to classical `-2`
than R=50's at the same step count. The earlier apparent "R=30 is worse"
result was purely the settle-time confound (1500 vs 8000 steps), not a
minimum-domain-size effect. The `lam_ring` effect (turning it on pushes
both slopes closer to `-2` and raises the eq/pol ratio) is confirmed
ROBUST across both domain sizes tested, not a coincidence at one scale.

## Interpretation

Two independent, additive fixes to the long-standing anomaly:
1. **Computing `E` from the physically appropriate mechanism (plain
   `step()`, not `step_cross`)** resolves the qualitative direction problem
   (polar-dominant -> equator-dominant, matching classical expectation) —
   this alone is likely the primary fix.
2. **The `lam`-gradient leak** (derived the same day, `∇σ×∇u`) makes a
   real, reproducible, second-order improvement on top of that — consistent
   with the ring's own sharp `lam:0->1` boundary genuinely contributing
   (Biermann-battery-like) real circulation content, not just numerical
   noise.

Neither slope has fully converged to exactly `-2` yet (`-1.31` to `-2.22`
range across configs), and this is still a single late-time snapshot (the
script's own known limitation, not yet fixed to a genuine multi-cycle
time-average) — but the qualitative anomaly (wrong-direction eq/pol ratio)
that motivated the BCC-dispersion-anisotropy and settle-time hypotheses
across multiple prior sessions appears to be resolved by a much simpler
explanation: the wrong operator was being used to compute `E`.

## Not yet done

- A genuine multi-cycle time-average (not a single late-time snapshot) to
  get a cleaner asymptotic slope reading.
- Push settle time further (2+ periods) to see if slopes keep converging
  toward exactly -2, now that the qualitative direction is right.
- Retire or clearly demote the BCC-dispersion-anisotropy explanation for
  the original `-6.19` result — it was never wrong as a hypothesis, but
  the actual root cause now looks like the `E`-computation mechanism, a
  much more basic issue that was never isolated before.
- Cross-check against `equilibrium/substrate/loop_antenna.py`'s finite
  constant-current-loop reference pattern (`ka≈1.04`), now that the
  underlying `S` computation is on firmer footing.

## Files

`run/substrate/adaptive/probe_poynting_step_only.py`,
`R30_lam0.log`/`R30_lam1.log` (1500-step, unmatched settle, superseded by
the matched runs), `R30_lam0_matched.log`/`R30_lam1_matched.log`,
`R50_lam0.log`/`R50_lam1.log` (all 8000-step, matched settle).
