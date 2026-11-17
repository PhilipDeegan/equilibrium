> **CORRECTION (2026-07-03 21:07):** the axis-symmetry-breaking finding below is
> retracted — it was a bug in `phi_ring_circulation` (wrong projection plane +
> wrong centre), not IC/lattice asymmetry. See
> `exp/2026/07/03/21_07_circulation_diagnostic_fix/report.md` for the fix and
> corrected re-measurement (all 6 rings read ~0 circulation, no axis bias).

# 6-spheromak cross configuration in Saha vacuum

**Date:** 2026-07-03 19:12 CEST  
**Script:** `probe_saha_cross.py`  
**Artefact:** `saha_cross.png`

## Parameters

| Parameter | Value |
|-----------|-------|
| R | 20 |
| N cells | 43599 |
| ring count | 6 |
| ring tiles each | 8 (total 48) |
| offset | 6.0 |
| actual r per ring | ≈5.89 |
| dt | 5.1282e-03 |
| lam | 5.0 |
| noise | 0.05 |
| damp_max | 0.003 |
| n_steps | 8000 |
| report | 800 |

## Ring placement

```
  ring @ [6. 0. 0.]  axis=x  actual_centre=[ 5.77  1.15  0.  ]  r≈5.89
  ring @ [-6.  0.  0.]  axis=x  actual_centre=[-5.77  1.15  0.  ]  r≈5.89
  ring @ [0. 6. 0.]  axis=y  actual_centre=[ 1.15  5.77  0.  ]  r≈5.89
  ring @ [ 0. -6.  0.]  axis=y  actual_centre=[ 1.15 -5.77  0.  ]  r≈5.89
  ring @ [0. 0. 6.]  axis=z  actual_centre=[ 1.15  0.    5.77]  r≈5.89
  ring @ [ 0.  0. -6.]  axis=z  actual_centre=[ 1.15  0.   -5.77]  r≈5.89
6/6 rings placed  total ring tiles=48
```

Note: all actual centres are offset by ≈[1.15, 1.15, ...] from the requested positions — BCC sublattice quantisation. The x=y=1.15 offset is a lattice artefact; rings land on the nearest valid BCC sublattice point.

## Runtime output

```
  step    u_mean    u_std  circ1  circ2  circ3  circ4  circ5  circ6
------------------------------------------------------------------------------------------
     0    0.9989   0.0406     0.000    -0.000    -0.000     0.000    -3.077    -3.077
   800    0.9989   0.2273     0.692     0.334    -0.445    -0.529    -3.180    -3.213
  1600    0.9989   0.2642    -0.412    -0.366     0.091     0.820    -2.900    -3.047
  2400    0.9989   0.2917    -0.399     0.015     0.630    -0.285    -3.259    -3.257
  3200    0.9989   0.3090     2.161    -0.730    -1.120     0.788    -2.575    -3.535
  4000    0.9989   0.3212    -1.118     0.891     0.067    -0.223    -2.603    -2.747
  4800    0.9989   0.3299    -0.391    -0.908     0.913     0.784    -3.203    -2.986
  5600    0.9989   0.3331     0.530     0.112    -0.822    -0.291    -3.354    -3.509
  6400    0.9989   0.3360    -0.690     1.641     1.245     0.025    -2.716    -3.114
  7200    0.9989   0.3369     0.240    -0.022     0.728     0.010    -3.225    -2.974
  8000    0.9989   0.3441     0.047     1.177    -0.070    -0.278    -2.328    -3.330
```

## Key findings

- **Axis symmetry breaking**: ±z rings (circ5, circ6, axis='z') maintain stable circulation ≈–3, whereas ±x and ±y rings (circ1–4) oscillate between ≈–2 and +2 with no preferred sign. This is a BCC lattice geometry effect — the z-axis is the natural axis for the ring face sequences used; x and y rings use different patterns that are less stable in this lattice orientation. No physical asymmetry intended.
- **Collective confinement**: the radial u profile develops elevated u inside r<6 relative to the vacuum level. The 6 rings collectively confine u in their interior — analogous to a potential cage. This is not present in the single-ring case.
- **Saha fraction elevated inside ring cage**: fraction of cells above threshold at r<6 exceeds the far-field value at late times, confirming the collective confinement effect.
- **u_std still converging at 8000 steps**: 0.34 and still slowly growing — needs ~20000 steps for the collective structure to fully settle (compare single-ring which converged by step 7000 at the same noise/damp settings).
- **u_mean conserved at 0.9989**: the slight deficit (0.0011 below 1.0) is because ring tiles were initialized at u≈0 (cosine winding = zero mean) and the ReciprocalBC target was set from the post-init state.
- **No tile overlaps**: all 6 rings placed with distinct tile sets.

## Limitations / known issues

- The ±x and ±y ring instability is a lattice artefact, not physics. A symmetric cross would require all rings to use the same face-sequence pattern, which means rotating the lattice rather than the ring axis.
- 8000 steps insufficient for full equilibration of the collective u structure.
- BCC sublattice offset ([1.15, 1.15, ...]) breaks exact 6-fold symmetry; rings are not equidistant from the origin in the lattice sense.

## Next steps

- Run to 20000 steps to see if collective confinement deepens and whether u-depletion at ring radius sharpens.
- Redesign cross with all 6 rings using axis='z' but placed in 3D (accept non-equatorial placement) to eliminate the face-sequence asymmetry.
- Measure the u-deficit at the ring cage interior vs exterior as a function of ring count and separation (multi-body substrate depletion / gravity analogue).
