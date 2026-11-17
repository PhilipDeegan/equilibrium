# Bell/canal probe — stage 1: separable (classical) baseline

**Date:** 2026-07-16 18:34–18:50
**Script:** `probe_bell_canal.py` (copy in this directory; lives at
`tst/substrate/probe_bell_canal.py`)
**Context:** first on-lattice step of the Bell/canal programme
(`doc/dst_next.md` §3–5): extend `test_qgate.py`'s real (u,φ) dynamics to
genuinely separated cells, classical/separable baseline first. Acceptance
benchmark is Feynman 1982's two-photon numbers: QM P_agree(30°) = 3/4;
deterministic-LHV ceiling 2/3; the stochastic-Malus LHV model used here
should give 5/8 and CHSH S = √2.

## Setup

- Domain: `DampedLattice.cylinder(radius=6, half_length=24)` (wired today),
  7345 cells, sponge on lateral wall + end caps, dt = stable_dt() = 0.0051.
- A and B: 19-cell patches (radius 1.6) on the axis at z = ∓7 (D = 14).
- Hidden variable: polarization angle α ~ U(0,π); A imprinted with a uniform
  transverse-φ vector at angle α, B at α+π/2 (singlet convention, matching
  archived `bell.py::classical_lhv`). Face→axis mapping detected empirically
  from the lattice (NB: `flux_tube._AXIS_DIR_CYCLE`'s comment convention is
  mirrored — face 8 is −x here).
- Readout: Malus's law on the patch-averaged transverse-φ direction
  (angle-only; intensity enters only via SNR against noise).
- Method: the (u,φ) step is exactly linear, so the α-ensemble is composed
  from 4 basis evolutions (A-x, A-y, B-x, B-y) + K noise-only evolutions;
  Monte Carlo (200k samples/correlation) runs offline on recorded responses.

## Results

### Run 1 — noise-free, long horizon (`bell_long.log`, `bell_long.npz`)

| ticks | t | \|v_A\| | cross-talk | S | P_agree(30°) |
|---|---|---|---|---|---|
| 0 | 0.00 | 1.000 | 0 | 1.418 | 0.6248 |
| 500 | 2.56 | 0.972 | 9.4e-8 | 1.407 | 0.6240 |
| 1000 | 5.13 | 0.975 | 4.7e-4 | 1.414 | 0.6248 |
| 2000 | 10.26 | 0.975 | 1.6e-4 | 1.413 | 0.6245 |
| 4000 | 20.51 | 0.974 | 4.6e-5 | 1.411 | 0.6260 |
| 8000 | 41.03 | 0.974 | 6.0e-5 | 1.409 | 0.6247 |

### Run 2 — bulk ambient u-noise η=0.01/step, K=16 (`bell_noise.log`, `.npz`)

S = 1.408–1.421, P_agree(30°) = 0.624–0.627 at all checkpoints to t = 20.5 —
statistically indistinguishable from noise-free, despite axis u_rms
reaching 0.33 (noise field is LARGE relative to the signal's u-wake).

## Findings

1. **Pipeline validated against theory exactly.** S = √2 (±0.01 sampling)
   and P_agree(30°) = 5/8 at every checkpoint — the known analytic values
   for a stochastic-Malus LHV, comfortably below Feynman's 2/3 ceiling.
   The lattice-side machinery (imprint → real dynamics → readout →
   CHSH/Feynman statistics) measures what it should. Stage 2 (joint canal)
   now has a trustworthy meter: beating 2/3 and reaching 3/4 is the target.
2. **Polarization identity is quasi-stable without topological protection.**
   A *raw* uniform transverse-φ patch (not a protected ring) radiates ~2.5%
   of its amplitude from the rim in t ≲ 2.5, then plateaus at |v| = 0.974
   with no further decay to t = 41. Divergence-free-per-cell is enough for
   orientation persistence at this scale. (Angle preserved exactly by the
   90°-rotation lattice symmetry; intensity plateau was not predicted.)
3. **Crossing transparency.** A's radiated wave reaches B at t ≈ 5.1
   (c_eff ≈ 2.7 cell-units/time), peaks at 4.7e-4 of signal amplitude, and
   is sponge-absorbed after passing. The separable baseline stays clean
   through and past wave crossing — no correlation contamination.
4. **No gravitational signature (u-structure) from separable imprints.**
   The u-field is a propagating transient wake (peak u_rms 2.3e-2 near A
   early on, decaying to ~5e-4 residual everywhere); nothing static forms
   along the A–B line. Consistent with the tension-free-canal argument:
   a φ-correlation structure with uniform u does not gravitate. The stage-2
   joint canal gets the same diagnostic — if a non-factorizable
   construction *requires* a u-gradient, that is a discovery, not a bug.
5. **Ambient u-noise is a very inefficient decoherence channel.** Bulk
   zero-mean u-noise at η=0.01/step (u_rms 0.33 on axis by t=20) produced
   NO measurable S degradation. u-noise must convert to *coherent
   transverse-φ at the patch* to rotate the readout angle, and the
   patch-average filters it. Boundary-injected noise (ReciprocalBC) is
   filtered further by the sponge. The decoherence threshold sits at much
   higher amplitude — or requires direct φ-noise — still to be mapped.

## Artefacts

- `probe_bell_canal.py` — probe script as run
- `bell_long.log` / `bell_long.npz` — run 1 output + recorded responses
- `bell_noise.log` / `bell_noise.npz` — run 2 output + recorded responses
- `run.sh` — exact commands

## Next steps

- **Stage 2 (the real physics):** non-factorizable joint-canal construction
  — a single connected substrate structure through the cylinder linking A
  and B, such that both readouts sample one shared state. Target:
  P_agree(30°) > 2/3 (anything above is already outside factorizable-LHV);
  full success = 3/4. This is the "still needs to be derived" item — no
  design yet survives contact with spacelike ordering.
- Noise-amplitude sweep (η = 0.03/0.1/0.3 bulk, and direct transverse-φ
  noise) to actually locate the decoherence threshold and get S(η, t).
- Separation sweep (D = 7/14/28 on a longer cylinder) once stage 2 exists —
  S(D) is the coherence-length measurement.
- Intensity-aware readout (detection threshold on |v|) as a second-stage
  refinement; angle-only readout hides amplitude decay by construction.
