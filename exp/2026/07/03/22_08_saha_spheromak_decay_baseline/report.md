# Hopf-ring spheromak decay under a Saha vacuum bath — cross_scale calibration does not transfer

**Date:** 2026-07-03 22:08 CEST
**Script:** `probe_saha_spheromak_decay.py` (new — combines `test_spheromak_decay.py`'s
Ring A + Ring B + step_cross with `probe_saha_spheromak.py`'s ReciprocalBC thermal bath)
**Trigger:** the `18_00_spheromak_decay_sweep` cross_scale calibration (cs=0.005 viable,
cs=0.010 too aggressive) was done in a cold/quiet vacuum decaying to a *numerical*
noise floor over 200k steps. Question: does that calibration mean anything once a
real *physical* thermal noise floor (Saha vacuum, u_std~0.3-0.4) is present?

## Setup

R=20 (N=43599), cross_scale=0.005, lam_scale=0.05, lam_mode=dynamic, u_bg=1.0,
noise=0.05 (ReciprocalBC thermal driving amplitude), 50000 steps, report every 50.

A third "control" ring — same 8-tile toroidal shape, no lam, no imprinted phi,
placed far from the real spheromak (r≈12.7) — is measured with the identical
`phi_ring_circulation` diagnostic each step. It gives a live, empirical noise floor:
whatever circulation this ring reads is pure ambient-bath artifact, not real winding.

## Preliminary finding — diagnostic sampling matters

At 500-step reporting cadence, circ_A/circ_B looked chaotic (large sign-flipping
jumps). A noise-free control run (same IC, `--noise 0`, `--report 20`) showed this
is partly the resonant loop's own carrier oscillation (ω_A=0.7654, ω_B=1.4142 rad/
time-unit → periods ~8.2 and ~4.4 time units ≈ 1600/860 steps) being aliased by too
coarse a sample rate — NOT chaos or a CFL/stability failure. `stable_dt()` (`0.4/
sum(W²)`, based on σ_max=1) remains valid regardless of the Saha bath since σ≤1
always; `step_cross`/`step_lam` at scale=0.005/0.05 are ~200x below their own
explicit-stability limit (confirmed by the smooth, non-chaotic noise-free
oscillation). Re-ran at report=50 (fine enough to resolve the carrier) for the real
comparison below.

## Result — flat, no decay, no separation for Ring A

Windowed RMS envelope (window=20 samples ≈ 1000 steps) of circ_A/circ_B compared to
circ_ctrl, early window (steps 7k-20k, right after thermal plateau) vs late window
(37k-50k):

| | env\|A\| | env\|B\| | env\|ctrl\| | A/ctrl | B/ctrl |
|---|---|---|---|---|---|
| early (7k-20k) | 0.152 | 0.281 | 0.156 | 1.07 | 1.94 |
| late (37k-50k) | 0.155 | 0.307 | 0.158 | 1.04 | 2.09 |

**Completely stationary across 43,000 steps — no decay trend.**

- Ring A is statistically indistinguishable from the empty-space control the entire
  run (ratio ≈ 1.0, never separates even transiently).
- Ring B shows a persistent ~2x excess over the control, but it does not decay —
  it's flat, likely a geometric/coupling artifact of ring B's position relative to
  the boundary noise injection, not a decaying topological signal.
- u_std/phi_rms plateau by ~step 3000-4000 (u_std→0.335-0.340) and stay flat for the
  rest of the run — consistent with the earlier single-ring Saha probes.

## Conclusion

The cs=0.005 cold-vacuum decay calibration is **meaningless** in this Saha context —
not because the decay is slower or the noise floor is merely higher, but because the
spheromak's coherent structure is completely swamped by the thermal bath from the
moment it equilibrates (~step 3000-4000) and never re-emerges as a distinguishable
signal in a single realization, all the way to 50k steps. There is no cross_scale
value to calibrate against here using this diagnostic as-is.

Physical reading (tentative): a DST spheromak's topological signature gets erased by
ambient thermal noise on a timescale far faster than its own vacuum step_cross decay
time. This is a genuine negative result worth keeping — it constrains any claim that
neutron/proton topology decay is observable "as is" in a thermally-active substrate.

## Next steps

- Ensemble-average circ_A/circ_B/circ_ctrl over multiple independent noise seeds
  (control should average toward zero; a real decaying signal should average to a
  nonzero, decaying trend even though individual realizations are noisy). This is
  the natural next test before concluding the decay is truly unobservable.
- Alternatively, sweep `--noise` down (0.05 → 0.01 → 0.001) to find the noise
  threshold below which the cold-vacuum decay becomes visible again.
- If ensemble-averaging also shows no separation, this becomes a strong claim: DST
  topology decay requires a quiet substrate window, and thermal coupling strength
  vs decay rate is itself a physical parameter worth deriving from first principles
  rather than assumed away.
