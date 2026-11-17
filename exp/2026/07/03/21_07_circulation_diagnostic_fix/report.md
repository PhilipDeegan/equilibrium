# phi_ring_circulation diagnostic bug — fix and re-measurement

**Date:** 2026-07-03 21:07 CEST
**Trigger:** revisiting `18_51_saha_spheromak` and `19_12_saha_cross` — asked whether the
±x/±y ring instability seen in the 6-spheromak cross was an IC asymmetry.

## Bug

`phi_ring_circulation(lat, tile_idx, centre)` in `equilibrium/substrate/flux_tube.py`
had two problems:

1. **Hardcoded x,y projection plane.** The azimuthal unit vector
   (`phi_x = -y/r, phi_y = x/r`) is only correct for rings wound around z. For
   x-axis or y-axis rings it silently projected onto the wrong plane.
2. **Centre passed was the global origin, not the ring's own position.** Every
   caller passed `centre = np.zeros(3)`. For a ring far from the origin (e.g. the
   6-cross rings offset by 6), the azimuthal tangent field evaluated at the global
   origin does not approximate the local tangent around the ring — it reads a
   near-constant direction across the whole (small) ring and produces a spurious
   nonzero number that has nothing to do with the ring's actual winding.

Combined with `init_toroidal_winding(..., frozen=False)` — used by both probe
scripts — imprinting a **standing-wave mode** (`phi_j = A cos(K(j+½))`, whose sum
over one full period is exactly zero by construction), the "circulation" reported
in both prior experiments was pure measurement artifact, not signal.

## Fix

`phi_ring_circulation` now takes an `axis` parameter (`'x'/'y'/'z'`, or `None` to
auto-infer from the ring's own positional variance, matching the convention
already used in `linking_number`). Callers now pass the ring's own centroid
(`lat.pos[tiles].mean(axis=0)`) instead of the global origin.

## Verification

Direct comparison, same IC, x-axis ring at offset [6,0,0] and z-axis ring at [0,0,6]:

| centre | axis | circulation (t=0) |
|---|---|---|
| local centroid | declared | ~1e-16 (zero) |
| local centroid | inferred | ~1e-16 (zero) |
| global origin | declared | -3.077 |
| global origin (old hardcoded x,y) | z-ring only | -3.077 (matches old bug output) |

This confirms the mathematical expectation: the free resonant loop mode has zero
net azimuthal flux by construction. The old readings were an artifact of centre
placement, not a real winding.

## Re-run with fix

**Single ring** (`probe_saha_spheromak.py --n-steps 4000 --report 1000`):

```
  step    u_mean    u_std    u_ring    phi_circ
--------------------------------------------------
     0    0.9991   0.0358   -0.0000     -0.0000
  1000    0.9991   0.3310    0.9785      0.1248
  2000    0.9991   0.3993    0.9048     -0.0130
  3000    0.9991   0.4338    0.7775     -0.0145
  4000    0.9991   0.4542    1.0612      0.0852
```

vs prior report's claim of "robustly persistent" circulation in [-5.4, -6.9] — that
claim is **retracted**. The correct reading: circulation starts at zero (as it must
for a standing-wave IC) and only wanders by ~0.01-0.12 under thermal noise. No
persistent winding.

**6-ring cross** (`probe_saha_cross.py`, default params):

```
  step    u_mean    u_std  circ1  circ2  circ3  circ4  circ5  circ6
------------------------------------------------------------------------------------------
     0    0.9989   0.0406    -0.000    -0.000    -0.000    -0.000    -0.000    -0.000
   800    0.9989   0.2273     0.000    -0.000    -0.000     0.001     0.003     0.002
  1600    0.9989   0.2642     0.021    -0.103     0.000    -0.011    -0.045    -0.025
  2400    0.9989   0.2917    -0.163     0.095    -0.009     0.050    -0.074    -0.074
  3200    0.9989   0.3090    -0.073     0.159     0.056    -0.025     0.005    -0.097
  4000    0.9989   0.3212     0.019     0.148    -0.081     0.124    -0.145     0.065
  4800    0.9989   0.3299    -0.000     0.133    -0.059     0.086     0.115     0.110
  5600    0.9989   0.3331     0.024     0.087     0.111    -0.014     0.033    -0.171
  6400    0.9989   0.3360     0.070    -0.011     0.206     0.158     0.047     0.044
  7200    0.9989   0.3369     0.089    -0.021    -0.086    -0.013     0.146    -0.092
  8000    0.9989   0.3441    -0.037    -0.178     0.130     0.014     0.131    -0.107
```

All 6 rings — including the ±z pair that previously looked "stable at ≈-3" — now
read ~0 at t=0 and wander symmetrically in [-0.2, +0.2] with no axis-dependent
bias. **The ±x/±y vs ±z asymmetry reported in `19_12_saha_cross/report.md` is
retracted — it was entirely the measurement bug, not a lattice or IC effect.**

## Conclusion

- No evidence of IC or lattice asymmetry between x/y/z toroidal ring construction.
  `_TOROIDAL_PATTERNS` for the three axes are structurally equivalent (verified by
  inspection — cyclic relabelings of the same face-pair square pattern).
- The free resonant loop mode (`frozen=False`) has zero net circulation by
  construction; it is not a useful IC for testing persistent circulation/moat
  formation. Use `frozen=True` (`set_flux_loop`, uniform flux) if a genuine
  nonzero winding number is needed for a circulation-persistence test.
- All prior spheromak decay work (`test_spheromak_decay*.py`) uses `set_flux_loop`
  (frozen, uniform flux, real nonzero circulation) with rings centred at/near the
  global origin, so those results are not affected by this bug — origin ≈ ring
  centroid there, and the flux pattern has genuine nonzero circulation regardless
  of axis (only single-axis 'z' rings were ever used in that codepath anyway).

## Next steps

- Re-run `probe_saha_cross.py --n-steps 20000` with the fix to see whether any of
  the 6 rings develop a genuine persistent circulation under longer thermal
  driving, now that the measurement is trustworthy.
- If a persistent-winding cross test is wanted, switch to `frozen=True` ICs.
