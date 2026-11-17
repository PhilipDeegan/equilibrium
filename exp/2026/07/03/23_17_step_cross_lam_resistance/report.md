# step_cross sigma(lam)^2 resistive fix — confirmed working

**Date:** 2026-07-03 23:17 CEST
**Script:** `probe_step_cross_lam_resistance.py`

## Background

Session started from a question about whether the 6-spheromak-cross deadzone
misalignment was an IC asymmetry (it wasn't — see
`exp/2026/07/03/21_07_circulation_diagnostic_fix/`), then moved to testing whether
the cold-vacuum `cross_scale` calibration (`18_00_spheromak_decay_sweep`) transfers
to a Saha-vacuum context (`22_08_saha_spheromak_decay_baseline/` — it doesn't, under
thermal noise the decay signal is completely swamped; without noise it decays
cleanly). That led to the question: does a spheromak's own time dilation
(σ(λ)=1/(1+λ), already used by `step_lam`) act as a resistive force against
`step_cross`'s topology-opening? It didn't — `step_cross` had no λ dependence at
all. Fixed and confirmed here.

## Fix

`step_cross` (topology-opening / decay mechanism) previously ignored `lam` entirely
— every cell's redistribution rate was `-dt*scale*phi[i,k]`, independent of local
mass/density. Added `sigma(lam_i)^2 = (1/(1+lam_i))^2` as a per-cell multiplicative
throttle, matching (and squaring, per explicit request) the convention already used
by `step_lam`'s self-buffering term.

Files changed:
- `inc/eq/substrate.hpp` — `substrate_step_cross` takes a `lam` array, computes
  `si2 = 1/(1+lam[i])^2` per cell, multiplies into `delta`.
- `src/mod.cpp` — nanobind binding updated to accept/pass `lam`.
- `equilibrium/substrate/lattice.py` — call site passes `self.lam`. Pure-Python
  fallback intentionally left unchanged (dead code while the compiled extension
  exists; user's call).
- User rebuilt the C++ extension (`./mkn clean build -dtOqp mod -a "-fPIC
  -std=c++20"`) — not run by the assistant, per standing project convention.

## Test design (important gotcha found along the way)

First attempt used the free resonant-loop IC (`init_resonant_loop`, `phi_only=True`)
— this is a standing wave with **zero net circulation by construction** (same
finding as `21_07_circulation_diagnostic_fix/`), so there was nothing real to decay;
the measured "growth from ~1e-15" was some secondary induced effect, not decay.
Corrected to use a genuine unfrozen DC circulation (`set_dc_circulation` — same
per-edge assignment as `test_spheromak_decay.py --rings dc`, but WITHOUT
`set_flux_loop`'s face-freezing, so the loop can actually evolve/decay under
`step()`/`step_cross`).

## Setup

Single 8-tile toroidal ring, R=20, cross_scale=0.02, cold vacuum (no Saha noise —
noise swamps the signal per the prior finding), 20000 steps, report every 50.
lam frozen (no `step_lam` call) at values {0, 1, 5, 20, 100} across 5 independent
runs — isolates the `step_cross` lam-dependence cleanly, no confounds from lam
transport dynamics or Ring A/B coupling.

## Result

| lam | sigma^2 | remaining circulation fraction after 20k steps |
|---|---|---|
| 0   | 1.0      | 0.05%  (fully decayed) |
| 1   | 0.25     | 12%    |
| 5   | 0.028    | 79%    |
| 20  | 0.0023   | 98%    |
| 100 | 0.0001   | 99.9%  (essentially frozen) |

Clean, monotonic resistance-vs-lam curve confirmed. A vacuum-level knot (lam=0)
fully decays in 20k steps at cross_scale=0.02; a lam=100 knot is essentially frozen.

## Conclusion

`step_cross` now gives DST a **derived** mechanism for mass-stability-against-decay
(higher lam → higher inertia against topology-opening) instead of relying on an
arbitrary small `cross_scale` constant to fake slow decay. This is a meaningful
addition to the model: real proton-scale lam (>>100) would make topology-opening
essentially unobservable on any practical timescale, which is qualitatively the
correct physical statement (protons are extremely stable).

## Next steps (pick up here tomorrow)

1. Re-run the `18_00_spheromak_decay_sweep`-style cross_scale calibration now that
   `step_cross` respects lam — the old cs=0.005/0.010 conclusions were derived
   without this resistive term and may need revisiting (a fixed lam=1.0 ring, as
   used throughout that sweep, would now decay ~4x slower at a given cross_scale
   than previously measured — see lam=1 row above: 12% remaining vs the old
   lam-independent baseline).
2. Redo the Saha-vacuum decay baseline (`22_08_saha_spheromak_decay_baseline/`)
   with the fixed `step_cross` — check whether the resistive term changes the
   "completely swamped by noise" conclusion at all (unlikely to fully rescue it
   since that was dominated by the thermal floor, not decay rate, but worth
   checking since decay is now slower too).
3. Test at real proton-scale lam values (much higher than 100) to see where the
   resistive effect saturates / whether LAM_MAX clipping matters.
4. Consider whether `step_lam` itself (winding transport) should also interact
   with this — i.e. does a stabilized-against-cross knot still transport lam
   normally, or should the two mechanisms be coupled further.
5. Still-open from earlier in this session: ensemble-averaging the Saha-vacuum
   circ_A/circ_B/circ_ctrl comparison over multiple noise seeds was proposed but
   not run — control should average to zero, a real signal should average to a
   nonzero (now slower, thanks to this fix) decaying trend.
6. Still-open: whether `DampedLattice`'s sponge stacked on top of `ReciprocalBC`
   (used by all `probe_saha_*.py` scripts) diverges from `test_saha.py`'s validated
   single-mechanism reference implementation — flagged, not yet checked.
