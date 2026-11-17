# Noise=0 spheromak decay, re-run with sigma(lam)^2 step_cross fix — decay dramatically slower

**Date:** 2026-07-04 12:30 CEST
**Script:** `probe_saha_spheromak_decay.py --noise 0` (cold vacuum, no Saha bath)
**Trigger:** the earlier noise=0 comparison run (documented inline in
[[project_saha_spheromak_decay]]'s "Combined Saha-vacuum + step_cross decay probe"
section: phi_rms decayed ~44x over 50k steps, circ_B peaked then decayed ~7x
post-peak) predates the `step_cross` sigma(lam)^2 resistive fix
([[project_step_cross]]). Re-ran the identical config with the fixed `step_cross`
to see how much the resistive term (lam pinned near 1.0 throughout, `lam_mode
dynamic`) changes the decay. No C++ rebuild needed — extension already had the fix
from last night.

## Setup

R=20 (N=43599), cross_scale=0.005, lam_scale=0.05, lam_mode=dynamic, u_bg=1.0,
**noise=0** (no thermal bath — this is the physically meaningful decay
configuration; see [[feedback_saha_noise_scope]] — noise is for ambient
equilibration tests only, never combined with decay probes), 50000 steps,
report every 50.

## Result — decay much slower, as the resistive fix predicts

| quantity | pre-fix (documented) | post-fix (this run) |
|---|---|---|
| phi_rms total decrease (t0→50k) | ~44x | **3.34x** |
| circ_B peak→final decrease | ~7x | 9.52x |
| circ_A behaviour | not separately documented | rises to ≈-0.108 by step 30k, then **flat/frozen** (1.06x, essentially no further decay) |

lam_A and lam_B stay pinned at 1.0000 ± 0.001 for the entire run (dynamic lam
transport, but no net drain — consistent with the resistive term being strong
enough to prevent lam itself draining away). circ_ctrl (empty-space control) stays
5-6 orders of magnitude below the ring signal throughout (1e-6 to 1e-7 range vs
circ_A ~0.1) — full separation from noise floor, as expected in a cold vacuum.

**phi_rms is the clean summary number**: pre-fix the total field energy proxy
dropped 44x over 50k steps; post-fix only 3.34x. This is the expected direction and
roughly the expected magnitude — the isolated single-ring probe
(`23_17_step_cross_lam_resistance`) found lam=1 leaves ~12% of circulation
remaining after 20k steps (vs ~0.05% for lam=0), i.e. roughly an 8x slowdown in
that decoupled measurement; here, in the coupled two-ring resonant-loop
configuration with dynamic lam, the compounded effect over 50k steps is even
larger (44x→3.34x is a ~13x reduction in total decrease).

**circ_A freezing is a new, unanticipated observation**: after an initial
transient (rising to ≈-0.108 by step ~30k), Ring A's circulation stops decaying
almost entirely — essentially locked at lam≈1 resistance. Ring B (4 tiles, higher
carrier frequency ω_B=1.4142) keeps decaying at a comparable rate to before. This
asymmetry between the two rings wasn't visible in the isolated single-ring probe
and is worth understanding — possibly a resonance/coupling effect between the two
rings under a shared lam field, or a difference in how quickly each ring's local
lam saturates against further decay.

## Conclusion

The sigma(lam)^2 resistive fix has a large, physically-sensible effect on the
noise-free (correct/physical) decay configuration: mass-carrying topology (lam≈1)
decays much more slowly than the pre-fix lam-blind mechanism predicted. This
strengthens the "lam gives mass a derived stability-against-decay" reading from
[[project_step_cross]]. Any cross_scale calibration done before this fix
(`18_00_spheromak_decay_sweep`) is now confirmed stale for lam=1 rings, consistent
with the prediction already logged there.

## Next steps

- Investigate the circ_A freezing vs circ_B continued decay asymmetry — is this
  physical (rings differ in tile count/frequency) or a diagnostic artifact?
- Port sigma(lam)^2 fix to AMR/adaptive `step_cross` (still open, see
  [[project_step_cross]]).
- Re-run the `18_00_spheromak_decay_sweep` cross_scale calibration (cs=0.005 vs
  0.010) with the fixed step_cross now that decay is confirmed much slower —
  cs=0.010 (previously "too aggressive") may now be back in a viable range.
