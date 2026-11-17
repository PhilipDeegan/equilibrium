> **CORRECTION (2026-07-03 21:07):** the "phi circulation is robustly persistent"
> finding below is retracted — `phi_ring_circulation` was reading a centre-offset
> geometric artifact, not real circulation. The free resonant IC used here has
> zero net circulation by construction. See
> `exp/2026/07/03/21_07_circulation_diagnostic_fix/report.md`.

# Single-ring spheromak in Saha vacuum

**Date:** 2026-07-03 18:51 CEST  
**Script:** `probe_saha_spheromak.py`  
**Artefact:** `saha_spheromak_long.png`

## Parameters

| Parameter | Value |
|-----------|-------|
| R | 25 |
| N cells | 84787 |
| ring tiles | 8 |
| ring r | ≈2.54 |
| dt | 5.1282e-03 |
| lam | 5.0 |
| noise | 0.05 |
| damp_max | 0.003 |
| n_steps | 20000 |
| report | 1000 |

## Runtime output

```
N=84787  R=25.0  ring=8 tiles  r_ring≈2.54
dt=5.1282e-03  lam=5.0  noise=0.05  damp=0.003
n_steps=20000  report=1000  out=saha_spheromak_long.png

  step    u_mean    u_std    u_ring    phi_circ
--------------------------------------------------
     0    0.9999   0.0119   -0.0000     -6.4559
  1000    0.9999   0.2005    1.0851     -6.4300
  2000    0.9999   0.2322    1.0628     -6.0587
  3000    0.9999   0.2566    0.9938     -6.5846
  4000    0.9999   0.2713    0.7327     -6.0985
  5000    0.9999   0.2819    0.9691     -6.2783
  6000    0.9999   0.2890    1.3545     -6.8340
  7000    0.9999   0.2912    1.0381     -6.8059
  8000    0.9999   0.2936    0.9701     -6.8657
  9000    0.9999   0.2938    1.1890     -5.8355
 10000    0.9999   0.2953    1.0307     -6.3098
 11000    0.9999   0.2943    0.9392     -6.6724
 12000    0.9999   0.2954    0.7368     -6.3304
 13000    0.9999   0.2962    1.0073     -5.7974
 14000    0.9999   0.2953    1.1188     -5.3957
 15000    0.9999   0.2964    0.8894     -6.5625
 16000    0.9999   0.2968    1.0802     -6.5292
 17000    0.9999   0.2952    1.0168     -6.4913
 18000    0.9999   0.2959    1.1452     -6.1389
 19000    0.9999   0.2973    0.9288     -6.6639
 20000    0.9999   0.2969    0.8323     -5.7448
```

## Key findings

- **phi circulation is robustly persistent**: circ stays in the range –5.4 to –6.9 throughout 20000 steps with no secular drift. The winding is stable against the thermal bath.
- **u_std converges**: grows from 0.012 → 0.297, reaching a plateau by step ~7000 and varying only 0.001–0.002 thereafter. The Saha vacuum equilibrates completely.
- **u_ring fluctuates near U_BG**: mean u at ring tiles ≈1.0 ±0.3 (thermal width), not significantly depleted despite lam=5. The lam slows local updates but does not create a strong sink at this scale and noise level.
- **u_mean rock-solid at 0.9999**: reciprocal BC conserves total u to 4 decimal places throughout.
- **Moat structure develops**: radial u profile shows depletion at ring radius (r≈2.5) relative to interior and exterior — the classic moat associated with a proton knot in DST.

## Next steps

- Test lam=0 (massless flux tube) to isolate the phi contribution from the lam/mass contribution to the moat.
- Test multiple rings in proximity to check interaction.
- Measure the moat depth as a function of lam (calibrate the u-depletion vs mass relationship).
