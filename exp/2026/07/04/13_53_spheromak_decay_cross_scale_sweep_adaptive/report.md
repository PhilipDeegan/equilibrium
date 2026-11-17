# Adaptive spheromak decay cross_scale sweep — post-fix recalibration

**Date:** 2026-07-04 13:53–15:24 CEST (long run — ~91 min wall-clock, 8-way parallel)
**Script:** `tst/substrate/adaptive/sweep_spheromak_decay_cross_scale.py` (new —
multiprocessing pattern from `validate_step_cross.py`, one subprocess per
cross_scale value, each builds its own deterministic Hopf-ring IC)
**Trigger:** re-run of `exp/2026/06/18/14_00_spheromak_decay_sweep` (cs=0.005
viable, cs=0.010 too aggressive), now stale for two reasons: (1) predates the
sigma(lam)^2 resistive fix to `step_cross`, and (2) the adaptive `step_cross`
was double-processing every face pair before today's sign-convention fix (see
[[project_step_cross]]). Requested range: 8 log-spaced points in [1e-4, 1e-2].

## Setup

R=40 (N=347941 leaves), Hopf ring pair (Ring A 8 tiles, Ring B 4 tiles,
phi_only=True resonant winding), lam_rings=1.0, lam_scale=0.05 (dynamic),
noise=0 (no Saha bath — per standing rule, never combine with decay probes),
50000 steps, report every 1000, n_threads=1 per job (parallelism at the
process level, 8 concurrent jobs).

## Result

| cross_scale | phi_rms_0 | phi_rms_f | H_0 | H_f | H_f/H_0 |
|---|---|---|---|---|---|
| 1.00e-4 | 1.570e-3 | 1.525e-3 | 6.0 | 5.666 | 94.4% |
| 1.93e-4 | 1.570e-3 | 1.490e-3 | 6.0 | 5.405 | 90.1% |
| 3.73e-4 | 1.570e-3 | 1.424e-3 | 6.0 | 4.937 | 82.3% |
| 7.20e-4 | 1.570e-3 | 1.306e-3 | 6.0 | 4.154 | 69.2% |
| 1.39e-3 | 1.570e-3 | 1.108e-3 | 6.0 | 2.992 | 49.9% |
| 2.68e-3 | 1.570e-3 | 8.112e-4 | 6.0 | 1.603 | 26.7% |
| 5.18e-3 | 1.570e-3 | 4.488e-4 | 6.0 | 0.4905 | 8.18% |
| 1.00e-2 | 1.570e-3 | 1.468e-4 | 6.0 | 0.05245 | 0.87% |

lam_A/lam_B stayed pinned at exactly 1.0000 across the entire sweep and every
cross_scale value — the dynamic lam step doesn't engage at this field
strength, confirming the old sweep's "lam_scale has no measurable effect"
finding still holds post-fix.

**circ_B is not a reliable diagnostic here**: starts at exactly 0 for every
run (the resonant/phi_only winding has zero net circulation by construction —
established earlier this session, see [[project_spheromak_decay]]'s
correction note) and its final values are non-monotonic across the sweep
(peaks around cs=7e-4–1.4e-3, falls off at both extremes) — this is carrier-
oscillation aliasing (the resonant loop's own ω_A/ω_B periods, ~800-1600
steps) sampled at a single point (step 50000), not a real decay signal. H and
phi_rms (scalar, non-directional) are the trustworthy numbers.

## Key finding — decay is much slower per unit cross_scale than pre-fix

Comparing to the pre-fix `14_00_spheromak_decay_sweep` (lam-blind, double-
processing adaptive step_cross): at cs≈5e-3 (matching the old "viable"
cs=0.005), pre-fix H reached the noise floor (H_50k=1.39e-3 out of H_0=6.0,
~99.98% decayed) by step 50000. Post-fix, the same cross_scale only reaches
8.18% remaining (91.8% decayed) — nowhere near the noise floor. **cs=1e-2 now
plays roughly the role cs=0.005 used to play** (99.1% decayed by 50k steps).

This is the expected direction: sigma(1)^2=0.25 alone predicts a 4x
rate-constant slowdown, and removing the pre-fix double-processing predicts
another ~2x — a combined ~8x slowdown in the rate constant compounds to a much
larger difference in cumulative decay depth over 50k steps, consistent with
what's observed.

## Conclusion / recalibration

The full cross_scale range [1e-4, 1e-2] is no longer "cs=0.005 viable, cs=0.01
too aggressive" as concluded pre-fix. Candidates for a new "viable working
point" (clean decay, not yet fully collapsed to noise floor, leaves room to
observe intermediate structure):
- cs=1.39e-3 to 2.68e-3: 50-73% decay by 50k steps — likely the new "cs=0.005
  equivalent" range for observing intermediate dynamics.
- cs=1e-2: reaches near-total decay (99.1%) by 50k steps — plays the role the
  old cs=0.01 conclusion warned against ("too aggressive... pulls flux from
  dead zones before it can reorganise"), but that conclusion was drawn
  pre-fix and needs re-checking against actual field/structure snapshots
  (not just scalar H/phi_rms) before assuming it still applies.

## Next steps

1. Inspect actual phi/u field snapshots (not just scalar diagnostics) at a
   mid-range cross_scale (e.g. 1.39e-3–2.68e-3) to check whether the "dead
   zone" over-reach behavior described in the old cs=0.01 conclusion still
   applies at the new, slower decay rates — the old warning was about
   structural corruption, not just numerical speed, and needs re-verification
   under the fixed physics.
2. Once a new viable cross_scale is chosen, resume the queued neutron-decay
   calibration work (`sweep_neutron_lam_ic_decay.py`) with it.
3. Consider a longer run (200k+ steps) at the chosen cross_scale to confirm
   the decay eventually reaches a stable dipole/noise-floor endpoint, matching
   the pattern from [[project_spheromak_decay]]'s original decay1→decay5
   snapshots.
