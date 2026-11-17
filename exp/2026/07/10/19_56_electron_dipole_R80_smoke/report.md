# Electron double-ring dipole — R=80 smoke run (5k steps)

First run of the new electron model: the poloidal component of the Taylor state
(double-ring dipole), seeded with the zero-net-force IC (uniform u=0 + phi
Hodge-projected against the step's W-weighted discrete divergence).

## Parameters

- `test_electron_dipole.py --R 80 --n-steps 5000 --report 500 --n-threads 10`
- N = 2,784,735 leaves, scale = 4.0 (sph_frac 0.05), r_axis = 2.44
- lam_target = 0 (pure phi winding — electron-has-no-lam null hypothesis)
- cross_scale = 0.01, lam_scale = 0.05, u_bg = 0, dt = 5.1282e-3
- divergence_clean: div_rms 1.90e-2 → 1.86e-10 (×9.8e-9), CG converged

## Findings

1. **Zero-net-force IC works at scale.** No shed transient; earlier cross=0
   validation (R=15) held all diagnostics frozen to machine precision. All
   evolution here is the irreversible step_cross channel.
2. **Pure-phi dipole decays with τ_φ ≈ 16 time units (≈3130 steps) at
   cross_scale 0.01.** phi_rms 2.257e-3 → 4.565e-4 over 5000 steps, clean
   exponential; H decays with τ_H = τ_φ/2 as expected.
3. **τ is domain-independent**: R=15 gave τ_φ = 15.8 t.u., R=80 gives
   16.0 t.u. (1.5% apart). The decay is intrinsic to the knot, not a
   boundary/sponge artifact.
4. **Shape-preserving decay**: shell circulation ratios nearly constant
   (s3/s2: 2.20 → 2.09 over 5 e-foldings of H); no toroidal winding ever
   appears (|circ_tor| ≤ 2e-15 throughout). The cleaned poloidal state
   behaves like a decaying eigenmode of the operator set.
5. **Interpretation (corrected same day)**: this decay is an operator
   artifact, not electron instability. Electron internal circulation is
   face-borne; step_cross physically represents the interior crossing
   channel, required only for whole-structure translation (inertia = cost
   of translation) and forced at rest only by linked windings (the neutron
   Hopf link threads one winding through the other's cells — that is why
   the neutron decays at rest and the unlinked dipole should not).
   Electrons have no lam (lam = substrate depletion proxy, nucleons only).
   The current step_cross fires on any inward face-phi and cannot yet
   distinguish coherent face transit from genuine interior crossing — a
   gated step_cross v2 is needed. The cross=0 validation run shows the
   true rest behaviour: exact stationarity.

## Artefacts

- `run.sh`, `test_electron_dipole.py` — exact script copies at run time
- `run_output.txt` — full stdout
- `tmp/electron_dipole_R80/edip_r80.npz` — time series (times, H, phi_rms,
  circ_pol, circ_tor, lam stats, S_SCAN/R_SCAN, r_axis)
- `tmp/electron_dipole_R80/edip_r80_final.npz` — final snapshot (restartable
  via --load-snapshot)
- `tmp/electron_dipole_R80/electron_dipole_r80.png` + `_cross.png` — plots

## Next steps

- Do NOT extend this configuration to 50k steps — phi is gone by ~3τ ≈ 10k.
- Translation-gate test: u-gradient kick on the dipole, cross=0 vs cross>0.
  Prediction: no coherent translation at cross=0; mobility ∝ scale when on.
  Doubles as the electron inertia (translation-cost) measurement.
- step_cross v2: gate the crossing on translation drive / transverse
  co-located flux (linkage) so coherent face transit passes free — spec
  before implementing.
- e/p inertia ratio as ratio of translation costs (dipole vs lam knot).
