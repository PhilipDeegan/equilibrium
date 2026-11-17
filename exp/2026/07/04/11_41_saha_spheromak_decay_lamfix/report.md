# Saha-vacuum spheromak decay, re-run with sigma(lam)^2 step_cross fix — conclusion unchanged

**Date:** 2026-07-04 11:41 CEST
**Script:** `probe_saha_spheromak_decay.py` (unchanged from `22_08_saha_spheromak_decay_baseline`)
**Trigger:** [[project_step_cross]]'s sigma(lam)^2 resistive fix to `step_cross` was applied
after the `22_08` baseline was measured (that run's cross_scale numbers predate the fix,
and at lam=1 the fixed decay rate is ~4x slower). Rebuilt `mod`+`mod_lattice` C++
extensions (`KLOG=1 mkn build -Oqp mod,mod_lattice -x mpi -g 0 -W 9`) and re-ran the
identical config to check whether the fix changes the "signal completely swamped"
conclusion from `22_08`.

## Setup

Identical to `22_08_saha_spheromak_decay_baseline`: R=20 (N=43599), cross_scale=0.005,
lam_scale=0.05, lam_mode=dynamic, u_bg=1.0, noise=0.05, 50000 steps, report every 50.
Only difference: `step_cross` now applies `sigma(lam_i)^2` resistance per-cell.

## Result — same as before, prediction confirmed

| | env\|A\| | env\|B\| | env\|ctrl\| | A/ctrl | B/ctrl |
|---|---|---|---|---|---|
| early (7k-20k), pre-fix | 0.152 | 0.281 | 0.156 | 1.07 | 1.94 |
| late (37k-50k), pre-fix | 0.155 | 0.307 | 0.158 | 1.04 | 2.09 |
| early (7k-20k), post-fix | 0.179 | 0.284 | 0.163 | 1.10 | 1.74 |
| late (37k-50k), post-fix | 0.200 | 0.318 | 0.163 | 1.23 | 1.95 |

u_std plateau: 0.338 (was 0.335-0.340). phi_rms plateau: 0.128 (was flat, matches).
lam_A/lam_B stay pinned near 1.0 throughout (0.99-1.01 range with noise-driven jitter) —
the resistive term is working (lam isn't draining toward 0), but the signal separation
from the control ring is essentially unchanged, if anything marginally noisier (A/ctrl
late-window rose to 1.23 vs 1.04, B/ctrl roughly flat).

**Conclusion unchanged, as predicted**: the `22_08` "completely swamped by thermal
noise" finding was noise-floor-dominated, not decay-rate-limited, so the step_cross
lam-resistance fix (which only affects the topological decay rate, not the thermal
bath) has no effect on this specific question. The earlier conclusion stands: a DST
spheromak's topological signature under Saha-vacuum thermal driving (u_std~0.34) is
not observable via this circulation diagnostic in a single realization, regardless of
cross_scale lam-dependence.

## Next steps (still open, from [[project_saha_spheromak_decay]])

1. ~~Re-run baseline with fixed step_cross~~ — done here, confirmed no change.
2. Ensemble-average circ_A/circ_B/circ_ctrl over multiple independent noise seeds —
   still the natural next test.
3. Resolve DampedLattice+ReciprocalBC stacking vs test_saha.py's validated reference.
4. Port sigma(lam)^2 fix to AMR/adaptive C++ step_cross — still not done.
