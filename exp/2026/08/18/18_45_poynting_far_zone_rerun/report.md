# Poynting-ring far-zone rerun attempt — settle-time confound found, not yet fixed

## Goal

Follow-up to `project_poynting_dipole_calibration` (equatorial slope -6.19 vs
classical -2, polar -2.44 — anomaly attributed to `kr` not being large
enough: `r=8-40` at `lambda=32.05` gives `kr=1.57-7.84`, nowhere near the
far zone). Plan: rerun the same ring at a bigger domain, fit only a radial
window comfortably deeper than the original (`kr` well above ~8), and see
whether the equatorial slope relaxes toward -2.

## What was run

`run/substrate/adaptive/probe_poynting_dipole_compare.py`, `--R 100 --hops 4
--m0 1e-4 --cross-scale 0.01 --n-steps 8000 --threads 8 --r-min-fit 40
--r-max-fit 80` (gate="barrier", already correct — this script never used
the diffusion-shaped `v1`). N=5,441,209 leaves, ring_r=5.318, period=32.05,
dt=5.1282e-3. Throughput measured mid-run: ~2.82 steps/s (5000 steps in
1771s wall time). Full log + plot in this directory
(`run_R100_farzone.log`, `poynting_dipole_compare_R100_farzone.png`).

Fit window `r=[40,80]` gives `kr=[7.85,15.7]` — deliberately above the
original's outer edge (`kr_max=7.84`), to test genuinely new, deeper
territory without touching `hops`/ring geometry (avoids reopening the BCC
anisotropy question).

## Result — not what was expected, and not simple radiation physics

```
equatorial (theta>70, n=1119): slope=+0.291   (GROWING with r)
polar      (theta<20, n=3914): slope=-0.065   (essentially flat)
|S|(equator)/|S|(pole) at r=40: 0.9496         (classical dipole wants >>1)
```

Neither number resembles a decaying radiation field, let alone anything
near -2. This is not "still steep" like the original -6.19 — it's
qualitatively different (not decaying at all in this window).

## Diagnosis: insufficient settle time at the new outer radius — a
confound I introduced, not a rerun of the original question

**Oscillation period in STEPS, not time units, is what matters here**:
`period=32.05` (time units) / `dt=5.1282e-3` = **~6252 steps per cycle** —
a very slow oscillation relative to the CFL-limited timestep (the timestep
is set by lattice spacing, not by the drive frequency).

Wavefront kinematics (natural wave speed `c=sqrt(10)*a≈3.1623` cells per
time unit, `a=1`):
- Time to reach `r=80`: `80/3.1623 ≈ 25.30` time units → **~4934 steps**.
- Steps remaining after arrival, at `n_steps=8000`: `8000-4934=3066` steps
  = `15.72` time units = **~0.49 periods** — LESS THAN HALF ONE FULL CYCLE
  settled at the fit window's outer edge.

The script's own docstring assumes "a single late-time snapshot, well
after transients" — that assumption was violated by roughly a factor of 2
at `r=80` with only 8000 steps. What was actually measured is turn-on
transient behaviour at the outer edge of the window, not a settled
radiation pattern — fully sufficient to explain both the nonsensical
(positive/near-zero) slopes and the near-1 equator/pole ratio, with no need
to invoke anisotropy or finite-source-size effects at all for THIS result.

**Retroactively applying the same calculation to the ORIGINAL calibration
run** (`r_max_fit=40`, `R=50`, `n_steps=8000`, same ring/period): arrival
at `r=40` takes `40/3.1623/dt ≈ 2467` steps; settle remaining =
`8000-2467=5533` steps = `28.37` time units = **~0.885 periods** — ALSO
less than one full cycle settled at its own outer edge. This means the
original `-6.19` equatorial anomaly may itself be more of a "not enough
elapsed cycles yet" artifact than a "too close to a non-point source" one —
sharper and probably more load-bearing than the pure-`kr`/near-zone framing
used until now.

## Corrected plan for next attempt (NOT YET RUN)

Same `R=100`/`hops=4`/fit window (`r=[40,80]`), but with `n_steps` sized to
give real settle time beyond arrival at `r=80` (step ~4934):

| target settle time beyond arrival | n_steps needed | est. wall time (~2.82 steps/s) |
|---|---|---|
| 1 full period (6252 steps) | ~11,186 → round to 11,200 | ~66 min |
| 2 full periods (12,504 steps) | ~17,438 → round to 17,500 | ~103 min (1.7h) |

Recommendation: run the 1-period version (~11,200 steps, ~66 min) first —
cheap enough to actually do, and already a qualitatively different regime
(currently sitting at ~0.49 periods, not ~1.0+). If slopes are still not
sane at 1 period, escalate to 2 periods before concluding anything
structural.

**Exact command to run next session**:
```
PYTHONPATH=$PWD python3 -u run/substrate/adaptive/probe_poynting_dipole_compare.py \
  --R 100 --hops 4 --m0 1e-4 --cross-scale 0.01 --n-steps 11200 --threads 8 \
  --r-min-fit 40 --r-max-fit 80 \
  --plot-out exp/<today>/<time>_poynting_far_zone_rerun_v2/poynting_dipole_compare_R100_1period.png \
  > exp/<today>/<time>_poynting_far_zone_rerun_v2/run_R100_1period.log 2>&1
```
Launch in background (`run_in_background`), do not poll — wait for the
completion notification. Expect ~66 min wall time at ~2.82 steps/s on this
machine (8 threads); re-measure throughput early (first ~500 steps) if
re-run on different hardware, don't assume the rate transfers.

## Also worth doing alongside/before scaling further

- The classical loop-antenna far-field reference built this session
  (`equilibrium/substrate/loop_antenna.py` / `tst/substrate/test_loop_antenna.py`,
  7 tests) is a PURE ANGULAR/AMPLITUDE correction — it cannot, by
  construction, change a fitted RADIAL slope (a constant multiplicative
  factor doesn't change a log-log slope). It remains useful as a secondary
  angular-shape check once a properly-settled far-zone measurement exists,
  but it was never going to explain the slope anomaly on its own — that
  requires exactly the settle-time fix above.
- Consider whether the script's single-instantaneous-snapshot methodology
  (explicitly flagged in its own docstring as a simplification, not a true
  time-average) is adequate even once settle time is fixed — a genuine
  time-average over several LATE, settled cycles would be more robust than
  one instantaneous phase, and is not yet implemented anywhere in this
  toolchain.
