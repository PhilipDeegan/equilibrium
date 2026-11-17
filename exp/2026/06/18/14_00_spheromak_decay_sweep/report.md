# Spheromak decay parameter sweep — 18 Jun 2026 14:00

## Setup

- Lattice: **adaptive only** (uniform excluded — wrong face orientation in IC)
- R=40, dt=5.1282e-03, N=347941 leaves
- Hopf ring IC (both rings), shared per lattice type
- 4 × 50k = 200k steps total per permutation
- Cross scales swept: 0.001, 0.005, 0.010
- Lam scales swept: 0.020, 0.050, 0.100
- Log: `0.txt` (captured stdout from remote run)
- NPZ snapshots on remote machine: `/home/deegan/git/eq/exp/substrate/spheromak_decay_sweep/18_JUN_14_00/`

## Key numbers (adaptive, 200k steps)

| run | phi_rms_0 | phi_rms_200k | H_200k | circ_B_0 | circ_B_200k |
|-----|-----------|--------------|--------|----------|-------------|
| cs0.001 ls0.100 | 5.65e-04 | 5.18e-05 | 6.55e-03 | -2.07e-01 | -2.60e-02 |
| cs0.005 ls0.020 | 1.57e-03 | 1.20e-08 | 3.51e-10 | -1.447 | ~noise |
| cs0.005 ls0.050 | 1.57e-03 | 1.20e-08 | 3.51e-10 | -1.447 | ~noise |
| cs0.005 ls0.100 | 1.57e-03 | 1.20e-08 | 3.50e-10 | -1.447 | ~noise |
| cs0.010 ls0.020 | 1.57e-03 | 1.23e-08 | 3.66e-10 | -1.447 | ~noise |
| cs0.010 ls0.050 | 1.57e-03 | 1.23e-08 | 3.66e-10 | -1.447 | ~noise |
| cs0.010 ls0.100 | 1.57e-03 | 1.22e-08 | 3.65e-10 | -1.447 | ~noise |

Note: cs0.001 starting phi_rms is lower because the run continued from a different chunk offset.

### Decay trajectory comparison (ls=0.050)

| step (cs0.005) | H (cs0.005) | H (cs0.010) |
|----------------|-------------|-------------|
| 0 | 6.00e+00 | 6.00e+00 |
| 5000 | 2.30e+00 | **9.32e-01** |
| 10000 | 9.28e-01 | **1.67e-01** |
| 25000 | 7.29e-02 | **1.39e-03** |
| 50000 | 1.39e-03 | **9.05e-07** |

cs0.010 decays ~2× faster in wall-clock steps (each 5k steps of cs0.010 ≈ 10k steps of cs0.005).

## Conclusions

**lam_scale has no measurable effect.** lam_ring, lam_A, lam_B remain at 1.0000 throughout all runs — the dynamic lam step is not engaged at this field strength. The 3-value lam_scale sweep was wasted; one value suffices.

**cs0.001 is too slow.** After 200k steps circ_B is still -2.6e-02 (from -2.1e-01 at step 50k), indicating the toroidal structure is decaying on a 10M+ step timescale. Not useful for structure studies.

**cs0.005 is the viable working point.** Clean exponential decay, reaches noise floor by ~150k steps, lam stays locked at 1.0 throughout.

**cs0.010 is too aggressive.** Confirmed from plots: the cross term at this strength reaches into the dead zones flanking the rings and pulls flux out of the inner structure before it can reorganise. The faster numerical decay does not represent cleaner physics — it represents the cross step overwhelming the topological constraint. Same noise floor as cs0.005 but without the intermediate structure.

## Next steps

- Use cs=0.005 for neutron decay calibration runs
- Calibrate cross_scale against physical decay rate (τ_n = 881.5 s): measure τ_decay/τ_cross ratio at cs=0.005 and compare to target
- Test cs=0.002 or 0.003 if inner structure needs more preservation time
