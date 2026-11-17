# Reconnection lam-alpha sweep — calibrating phi→lam coupling

**Date:** 2026-06-20 22:36 CEST  
**Script:** `tst/substrate/adaptive/test_reconnection_alpha_sweep.py`

## Parameters

| Parameter | Value |
|---|---|
| R | 60.0 |
| scale | 12.0 |
| z_sep | 16.0 |
| omega | 0.1 |
| sigma | 0.7 |
| lam_target | 1.0 |
| u_max | 1.0 |
| lam_scale | 0.05 |
| lam_relax_rate | 0.1 |
| alpha_values | 1, 10, 30, 50 |
| cross_scale | 0.005 |
| n_steps | 10000 |
| dt | 5.1282e-03 |
| N | 1,175,219 cells |

## Context

Previous run (18:51) confirmed local lam relaxation is stable and physically
correct. At alpha=1, lam_mid peaked at 0.038 → sigma_mid_min = 0.964 (4%
reduction) — lam was a passenger. This sweep locates the causal threshold.

Key formula: `sigma(lam) = 1/(1+lam)`, so lam must reach O(1) to matter.
At phi_rms0 = 2.24e-3 and relax_rate=0.1, lam_target = alpha * |phi| must
accumulate over many steps. The expected equilibrium lam ~ alpha * phi_mid_peak.

## Runtime output (summary)

```
     alpha      J_peak    J_step    lam_mid_pk   sig_mid_min
  -------------------------------------------------------
       1.0  1.4585e-02      2500        0.0383        0.9638
      10.0  1.2287e-02      4000        0.6098        0.6512
      30.0  9.9876e-03      9000        2.4587        0.3861
      50.0  6.1270e-03      6000        3.7153        0.3382
```

## Key findings

**1. Four distinct dynamical regimes.**

| α | sig_mid min | J_mid peak | Regime |
|---|---|---|---|
| 1  | 0.964 | 1.46e-2 @ step 2500 | Passenger — no back-reaction |
| 10 | 0.651 | 1.23e-2 @ step 4000 | Causal — 35% suppression, peak delayed 1500 steps |
| 30 | 0.386 | 9.99e-3 @ step 9000 | Near-stall — sheet barely reconnects in 10k steps |
| 50 | 0.338 | 6.13e-3 @ step 6000 | Stalled — phi_rms still *rising* at step 10000 |

**2. Causal threshold is between α=1 and α=10.**  
At α=10: lam_mid reaches 0.61, sigma_mid drops to 0.65 during the J_mid peak.
Reconnection still completes but is measurably retarded — peak shifts from
step 2500 to step 4000 and J_mid peak drops by 16%.

**3. α=30–50: current-sheet bottleneck.**  
At α=30, sigma_mid falls to 0.39 — throughput cut by 61%. J_mid peak
doesn't occur until step 9000. At α=50, phi_rms is still growing at step
10000 (1.37e-2 vs IC value 2.24e-3), meaning the reconnection event is
completely stalled: phi piles up in the sheet but cannot diffuse through the
high-lam barrier. This is the DST analog of resistive suppression.

**4. α=10 is the target for harris_ic_sweep.**  
35% throughput suppression during the peak is physically meaningful, reconnection
still completes within the run window, and the lam dynamics are clearly causal
rather than decorative.

## Artefacts

- `tmp/reconnection_alpha_sweep.png` — 4-panel overlay (J_mid, phi_rms, lam_mid, sigma_mid)

## Next steps

- Run `harris_ic_sweep.py` with `--lam-alpha 10 --lam-relax-rate 0.1`
- Compare reconnection rates across harris sheet thicknesses with lam driving active
- α=30 "near-stall" regime is worth a dedicated run later: may produce plasmoid instability
  if the sheet is long enough to go tearing-mode unstable despite lam suppression
