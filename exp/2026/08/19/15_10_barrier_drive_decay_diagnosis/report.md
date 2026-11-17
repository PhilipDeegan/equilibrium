# Poynting-ring far-zone investigation — root cause found: m0 was never derived for this script, and no m0 value makes the diagnostic coherent

## Context

The 66-minute "1 full settled period" rerun
(`exp/2026/08/19/poynting_far_zone_rerun_v2/`, `--R 100 --hops 4 --m0 1e-4
--cross-scale 0.01 --n-steps 11200 --r-min-fit 40 --r-max-fit 80`) completed
and produced a NEW failure mode, not "closer to sane": both equatorial and
polar fits report `insufficient nonzero cells in fit range (n=0)` — the
one-shot `step_cross(gate="barrier")` diagnostic call at the end found
literally zero cells firing anywhere in `r=[40,80]`.

Philip flagged that `m0=1e-4` was picked, not derived
([[project_derive_dont_pick_parameters]]). Investigated directly rather than
just logging the n=0 result.

## Finding 1 — the picked m0 continuously drains the fit band over the run, not just at the final diagnostic call

`probe_poynting_dipole_compare.py`'s main loop calls
`alat.step_cross(0, dt, cross_scale, gate="barrier", m0=args.m0)` **every
step**, not only at the final diagnostic snapshot. Built a cheap R=30
reproduction (`probe_barrier_drive_decay.py`, same hops=4/m0=1e-4/
cross_scale=0.01, ~147k leaves, seconds/1000 steps) that logs the pooled
per-cell drive (`Σ_k W_k·max(-phi[k],0)`, the exact barrier-gate quantity)
in a fixed outer band every 500 steps, for 20,000 steps (~3.2 periods).

**With the barrier active** (`barrier_on.log`): count of in-band cells with
drive > m0 collapses monotonically — from ~59,000/65,712 (90%) around step
1000 down to 228/65,712 (0.3%) by step 20,000. `max_drive` in-band falls
from `7.5e-3` (step 500) to `1.6e-4` (step 20,000), approaching `m0=1e-4`
from above and clearly headed below it.

**Control, `cross_scale=0` (no draining at all)**, otherwise identical
(`barrier_off_partial.log`): the same band settles into a **stable bounded
oscillation** — `n_above_m0` holds around 45,000-55,000/65,712 through the
full 3.2 periods, `max|phi|` plateaus, no collapse. This rules out "natural
far-zone decay" as the explanation — the collapse is caused specifically by
the continuous barrier drainage, not by the field itself dying out.

**Conclusion**: at `m0=1e-4`, letting the system run long enough to properly
settle (the entire point of extending 8000→11200 steps) is self-defeating —
the same barrier gate used to measure `S` is also continuously eroding the
very region being measured, and depletes it to near-total silence within
about 2-3 oscillation periods regardless of `R` or `hops`. Longer settle
time does not fix this diagnostic; it guarantees its collapse.

## Finding 2 — m0=1e-4 was never derived for this script; it contradicts the sibling script's own already-reasoned default

`run/substrate/adaptive/probe_poynting_ring.py` (`probe_poynting_dipole_
compare.py`'s own sibling/predecessor) already defaults `--m0` to **0.3**,
with an explicit docstring rationale:

> "Order-unity, comparable to the ring's own driven phi amplitude, so
> crossing fires near the source but not in the decayed far field."

`probe_poynting_dipole_compare.py` silently uses `--m0` default **1e-4**
instead — three thousand times smaller, no stated reason, and never
reconciled against the sibling script's own reasoning anywhere in this
investigation's history (`project_poynting_dipole_calibration` memory, `now`
file, or either report.md from 08-16/08-17/08-18).

Measured the actual natural (undrained) pooled-drive scale directly
(`probe_near_far_drive_scale.py`, `cross_scale=0`, R=30, hops=4, same ring):

| region | max drive | p99 drive |
|---|---|---|
| on the ring itself | ~1.00-1.01 | ~1.00-1.01 |
| near zone (1-3×ring_r) | ~0.88 | ~0.013-0.023 |
| far zone (0.4R-0.8R) | ~0.010-0.014 | ~0.004-0.007 |

- `m0=1e-4` (picked) sits **below even the far zone's own natural floor** —
  the "barrier" gate fires almost everywhere, behaving like near-continuous
  diffusion (`gate="v1"`) rather than a genuine threshold, which defeats the
  entire physical point of choosing `gate="barrier"` for this experiment.
- `m0=0.3` (the sibling script's already-reasoned default) would fire on/near
  the ring only and **essentially never** in the far zone (0.3 vs a natural
  ceiling of ~0.014 there — over 20x too small to ever cross).
- The doc-derived, per-knot values from `doc/dst_next.md` §8
  (`m0_proton≈0.16`, `m0_neutron≈2.49`) sit in the same
  "fires-near-source-only-or-never" regime — `2.49` would not even fire on
  the ring itself in this construction.

## Finding 3 — no m0 value makes "far-zone S via step_cross" a coherent measurement here

This isn't just a slider needing retuning:
- Too small (`1e-4`, what was actually used): fires everywhere initially,
  behaves like unconditional diffusion, and — per Finding 1 — drains the
  measured region to exhaustion within a few periods purely as an artifact
  of measuring it this way.
- Correctly order-unity (`0.3`, or the derived proton/neutron values):
  structurally never fires in the far zone at all, by the gate's own design
  intent ("fires near the source but not in the decayed far field" — that's
  a feature, not a bug, of a real threshold mechanism).

Either way, `step_cross`'s barrier-gated `dphi` cannot produce a genuine
far-zone radiation signal. This is also consistent with `doc/dst.md`'s own
"Inertia and the Face/Cell Gate" section, which assigns **face-hopping**
(`step()`, the ordinary reversible wave dynamics) as the mechanism for EM
wave propagation ("EM waves ... use this mode. Speed: c"), and
**through-cell crossing** (`step_cross`) as a structurally different,
higher-cost mechanism for translating windings / reconnection
([[project_dst_phi_diffusion]]: "step_cross IS the DST-native resistive/
reconnection mechanism"). Measuring the propagating far-field `E`/`S` from
`step_cross`'s crossing events, rather than from the ordinary `step()`-driven
`phi` dynamics, may have been the wrong mechanism to probe from the start —
independent of any `m0` value.

## Bottom line

The entire "far-zone Poynting-flux vs classical -2 law" thread across this
investigation's history (07-14 original calibration, 08-16/17/18 far-zone
reruns, tonight's 66-minute rerun) used an `m0` that was picked, not
derived, and that value being wrong (too small) is sufficient to explain
both the earlier "wrong slope" result (probably contaminated by
near-continuous drainage acting almost like `gate="v1"`) and tonight's
"zero cells fire at all" result (that drainage finally exhausting the
region after enough steps). Fixing `m0` to the intended order-unity value
does not rescue the measurement — it just makes the gate correctly never
fire out there, which is equally uninformative for a far-zone test.

## Recommendation (not yet acted on — flagging for next step)

The near-field / equatorial-vs-polar anisotropy question can still be asked
with a properly order-unity `m0` (fires near the source, which is exactly
where the earlier `-6.19` vs `-2.44` asymmetry was measured) — that part of
the investigation may still be salvageable. But the *far-zone radiation*
question needs `E`/`S` computed from the ordinary `step()`-driven `phi`
dynamics (matching DST's own stated face-hopping propagation mechanism),
not from `step_cross`'s barrier-gated crossings, if it's to be pursued
further at all.

## Files

- `probe_barrier_drive_decay.py`, `barrier_on.log`,
  `barrier_off_partial.log` (control cut short at step 8000, already
  conclusive — the full 20,000-step control log confirms no collapse
  throughout)
- `probe_near_far_drive_scale.py`, `near_far_drive_scale.log`
