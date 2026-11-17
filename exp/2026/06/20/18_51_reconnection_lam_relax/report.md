# Counter-rotating spheromak reconnection — local lam relaxation

**Date:** 2026-06-20 18:51 CEST  
**Script:** `tst/substrate/adaptive/test_counter_rotating_spheromak_reconnection.py`

## Parameters

| Parameter | Value |
|---|---|
| R | 60.0 |
| scale | 12.0 |
| z0 | ±8.0 |
| omega | 0.1 |
| sigma | 0.7 |
| lam_target | 1.0 |
| u_max | 1.0 |
| lam_scale | 0.05 |
| lam_relax_rate | 0.1 |
| lam_alpha | 1.0 |
| cross_scale | 0.005 |
| n_steps | 10000 |
| dt | 5.1282e-03 |
| N | 1,175,219 cells |

## Context

Previous attempt used `step_lam_phi_mag` (conservative upwind transport between neighbours).
At mag_scale=50 it was explosive (lam_mid jumped from 0.0058 → 3.7 in 500 steps).
At mag_scale=0.05 it was a noop (~1e-9 per step).

Root cause: lam is a **proxy for substrate displacement by the local phi field** — not a
separately conserved quantity that flows between cells. Correct update is local relaxation:

```
lam_target_i = clip(alpha * |phi_i|, 0, LAM_MAX)
dlam_i/dt    = rate * (lam_target_i - lam_i)
```

No neighbour lookup. No transport. No conservation issue.

## Runtime output

```
   step       J_mid      phi_up     phi_mid      phi_dn    lam_up   lam_mid    lam_dn         czc     phi_rms
  -------------------------------------------------------------------------------------------------------------------
      0  1.1935e-20  2.2505e-03  1.2662e-03  2.2505e-03    0.0048    0.0058    0.0048    -0.00000  2.2445e-03
    500  4.7281e-03  1.7703e-02  2.6217e-02  1.7703e-02    0.0054    0.0077    0.0054    +0.00000  1.8725e-02
   1000  1.0990e-02  1.5867e-02  2.9039e-02  1.5867e-02    0.0081    0.0156    0.0081    +0.00000  1.7551e-02
   1500  1.1292e-02  1.5008e-02  2.3684e-02  1.5008e-02    0.0122    0.0229    0.0122    -0.00000  1.6008e-02
   2000  1.3968e-02  1.5207e-02  2.2038e-02  1.5207e-02    0.0176    0.0296    0.0176    -0.00000  1.5957e-02
   2500  1.4585e-02  1.4535e-02  1.9618e-02  1.4535e-02    0.0240    0.0355    0.0240    +0.00000  1.5073e-02
   3000  1.1153e-02  1.0479e-02  1.3896e-02  1.0479e-02    0.0273    0.0383    0.0273    +0.00000  1.0840e-02
   3500  9.7228e-03  9.7116e-03  1.1899e-02  9.7116e-03    0.0273    0.0369    0.0273    -0.00000  9.9364e-03
   4000  8.5130e-03  7.9093e-03  1.0022e-02  7.9093e-03    0.0277    0.0356    0.0277    +0.00000  8.1315e-03
   4500  5.7324e-03  4.1905e-03  4.6860e-03  4.1905e-03    0.0251    0.0312    0.0251    +0.00000  4.2431e-03
   5000  4.5268e-03  2.5410e-03  3.5318e-03  2.5410e-03    0.0213    0.0268    0.0213    -0.00000  2.6508e-03
   5500  3.7935e-03  2.1408e-03  3.4478e-03  2.1408e-03    0.0179    0.0232    0.0179    +0.00000  2.3066e-03
   6000  3.0734e-03  2.4246e-03  3.1041e-03  2.4246e-03    0.0152    0.0202    0.0152    +0.00000  2.5065e-03
   6500  2.5705e-03  2.3139e-03  3.5051e-03  2.3139e-03    0.0132    0.0178    0.0132    -0.00000  2.4507e-03
   7000  1.9280e-03  1.8151e-03  2.5076e-03  1.8151e-03    0.0116    0.0158    0.0116    -0.00000  1.9148e-03
   7500  1.5608e-03  2.1620e-03  2.1620e-03  2.1620e-03    0.0100    0.0134    0.0100    -0.00000  2.1840e-03
   8000  1.6356e-03  1.7054e-03  1.7712e-03  1.7054e-03    0.0085    0.0115    0.0085    +0.00000  1.7330e-03
   8500  1.8301e-03  1.7728e-03  2.4924e-03  1.7728e-03    0.0075    0.0100    0.0075    -0.00000  1.8598e-03
   9000  1.9663e-03  1.7309e-03  2.1756e-03  1.7309e-03    0.0069    0.0090    0.0069    +0.00000  1.7799e-03
   9500  1.8504e-03  1.6357e-03  1.8664e-03  1.6357e-03    0.0064    0.0081    0.0064    +0.00000  1.6620e-03
  10000  1.7255e-03  1.5403e-03  1.6239e-03  1.5403e-03    0.0060    0.0073    0.0060    -0.00000  1.5513e-03
```

## Key findings

**1. Local lam relaxation works — no blowup, no noop.**  
lam_mid rises smoothly from 0.0058 → peak 0.0383 at step 3000, then decays with phi.
Growth is ~6.5× over 3000 steps with rate=0.1, alpha=1.0 — controlled and physical.

**2. Reconnection event confirmed.**  
J_mid rises monotonically to peak 1.46e-2 at step 2500 then declines.
At step 7500: phi_mid = phi_up = 2.162e-3 — midplane concentration equalized.
This is topological reconnection completing; no secondary event at omega=0.1.

**3. lam correctly tracks phi with a lag.**  
lam_mid peaks at step 3000, ~500 steps after J_mid peak (step 2500) — consistent with
relaxation timescale 1/rate = 10 time units = ~2000 steps at dt=0.005.
lam_mid/lam_up ratio peaked at ~1.93 (step 1000) when phi_mid/phi_up = 1.83.
The slight overshoot is correct: lam integrated the earlier higher phi_mid.

**4. Late-time residual.**  
phi_rms stabilises at ~1.5–2.5e-3 ≈ initial value (2.24e-3). Sponge has absorbed
the reconnection energy; what remains is low-amplitude field bouncing near the boundary.
lam_mid/lam_up at step 10000 = 1.22 while phi ratio = 1.054 — lam still draining toward
phi equilibrium. Would reach ratio ~1.05 given ~10 more time units.

## Artefacts

- `tmp/reconnection_lam_relax.png` — 4-panel plot (J_mid, phi zones, lam zones, czc)

## Next steps

- Try omega=0.3 to check for secondary reconnection (previous test showed J_mid
  rise-peak-fall but no second peak at omega=0.1)
- Tune lam_alpha: with alpha=1 the lam values stay very small (max 0.038).
  Try alpha=10–50 to get lam into the range where it measurably affects phi dynamics
  via step_lam (div-phi coupling). Currently lam is a passenger, not a driver.
- Harris sweep: re-run with --lam-relax-rate replacing --lam-mag-scale to see
  whether lam dynamics affect reconnection rate in the sheet geometry.
