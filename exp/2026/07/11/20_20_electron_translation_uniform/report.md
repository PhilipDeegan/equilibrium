# Electron (lam-free winding) translation on the uniform lattice — direct comparison to the proton knot

Closes a gap in `exp/2026/07/11/19_45_proton_knot_translation/`: that probe and the
original electron-translation result (`exp/2026/07/10/21_15_electron_translation_gate/`,
on the AMR adaptive cluster) used two different lattice implementations and IC
constructions, so they weren't a strictly controlled comparison. This probe puts the
electron analogue (an 8-tile toroidal circulation, lam=0) through the *identical*
protocol, diagnostics, and lattice class (`equilibrium.substrate.lattice.DampedLattice`)
as the proton knot, at the same R=20, same drive levels.

## IC

`SubstrateLattice`/`DampedLattice`, R=20. An 8-tile toroidal ring
(`equilibrium.substrate.flux_tube.find_toroidal_ring`, axis z) carrying a CONSTANT
circulation flux (amplitude=1.0 around the ring), u=0 everywhere, lam=0 everywhere,
nothing frozen. This is neither of `flux_tube.py`'s two built-in modes: `frozen=True`
permanently pins the ring's phi via `fixed_phi` (the same "inexhaustible battery"
pathology found for the proton's failed pinned-IC attempt), and `frozen=False` is
deliberately a travelling-wave (oscillating) mode, not a static structure. The constant
circulation used here is instead an exact fixed point of `step()` alone by the same
reasoning as the proton IC: each ring tile has equal inward/outward flux (divergence-free)
and u is uniform (so phi is never driven) — verified numerically, max|u| and max|phi|
unchanged over 5000 steps with no kick.

Unlike the proton, `step_cross(cross_scale>0)` does NOT leave this state alone even
without a kick: half the ring's edges carry inward flux, so step_cross fires on the ring's
own circulation at rest. This reproduces the already-known R1 pathology (from the
step_cross v2 work) and is the expected baseline, not a new finding — the question is
whether the same external tilt used on the proton probe produces coherent transport on
top of that, or only erosion.

## Protocol

Identical to `probe_proton_translation.py`: n_base=1000, external u tilt pinned at a
battery ring for n_kick=6000, released for n_coast=4000. R=20 (N=43599), dt=5.128e-3.
Same four cross_scale levels as the proton scan: 0.01, 0.03, 0.1, 0.3 (`force_scale`/
`step_lam_force` doesn't apply here — lam=0 throughout, per DST convention: electrons
carry no lam). Diagnostics: winding-weighted COM (weight = per-cell RMS phi), that same
weight's total ("mass": total winding energy) and spread around its own COM, cumulative
`bc_flux()` (radiated proxy) — same three-channel accounting as the proton probe.

## Results

| cross_scale | v_kick | v_coast | sign match? | mass_frac | spread_frac | d_kick |
|---|---|---|---|---|---|---|
| 0.01 | +1.545e-3 | +2.993e-3 | yes | 33.00 | 10.11 | +0.044 |
| 0.03 | +4.640e-3 | -4.888e-3 | **no** | 20.76 | 10.15 | +0.125 |
| 0.1 | +1.482e-2 | -3.035e-3 | **no** | 7.01 | 10.47 | +0.465 |
| 0.3 | +3.161e-2 | +5.274e-2 | yes | 2.37 | 10.92 | +1.150 |

Proton knot at the identical cross_scale levels, for direct comparison (from
`exp/2026/07/11/19_45_proton_knot_translation/`): mass_frac=1.0000 and spread_frac≤1.0001
at every single level, with sign-consistent kick/coast velocity at every level from 0.03
up.

## Interpretation

The electron ring's own structure is not preserved at ANY tested drive level — the
"mass" (total winding energy) is off by 2.4×-33× and the spread has inflated ~10× at
every level, meaning the ring stops being a compact 8-tile object almost immediately once
kicked; the large `d_kick` values (0.04-1.15) are the COM of an actively spreading blob,
not evidence of coherent translation of a stable structure. The kick/coast sign match is
also inconsistent across levels (yes/no/no/yes) rather than holding steadily the way the
proton's did — that erratic pattern is itself the signature of noise superposed on
erosion, not a directed drift (a real, gradually-strengthening translational response
would be expected to show a *consistent* trend across a monotonic drive sweep, as the
proton did).

This independently reproduces — on a completely different lattice implementation, IC
construction, and diagnostic set than the original adaptive-cluster probe — the same
qualitative "erosion without transport" finding for the lam-free circulation-type
structure, now cleanly contrasted against the lam-carrying, non-circulating knot's
"conserved, rigid, sign-consistent translation" on the identical apparatus. The
electron/proton asymmetry is not an artifact of either lattice implementation
individually; it tracks the topological distinction (circulation/curl≠0-globally vs.
non-circulating monopole) across two independent codebases.

## Caveats

- The winding-weighted "mass"/"spread" diagnostic is inherently less well-behaved here
  than for the proton, precisely because the object being measured is not stable — these
  numbers describe an eroding, expanding pattern, not a moving rigid one. That's the
  finding, not a measurement flaw, but it does mean `d_kick`/mobility for the electron
  ring should not be read as a translation speed the way the proton's is.
- Only ring_axis=z tested (no isotropy check here, unlike the proton probe, since the
  headline result — total non-conservation of mass/shape — is already unambiguous
  without one).
- Only R=20 tested; resolution independence unverified (shared caveat with the proton
  probe).

## Artefacts

- `probe_electron_translation_uniform.py` — script copy at run time
- `run_output.txt` — full stdout, all four cross_scale levels
- `electron_uniform_s{0.01,0.03,0.1,0.3}.npz` — per-run (t, phase, com, mass, spread,
  phi_rms, radiated_cum) histories
