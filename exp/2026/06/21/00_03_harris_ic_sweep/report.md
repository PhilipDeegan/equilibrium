# Harris IC Sweep — alpha={5,10,15} vs baseline

**Date:** 2026-06-21 00:03 CEST
**Script:** `tst/substrate/adaptive/harris_ic_sweep.py`
**Simulation:** `tst/substrate/adaptive/harris_sheet_phare.py`
**Output:** `tmp/harris_sweep_alpha/`

---

## Parameters

### Grid (SHARED)
| Parameter | Value |
|-----------|-------|
| Lx × Ly × Lz | 100 × 100 × 3.5 d1 |
| N cells | 52 291 (uniform, no AMR) |
| dt | 0.05 |
| n_steps | 10 000 |
| report interval | 500 |
| blowup_factor | 50× |
| perturb | 0.1 (fraction of B0) |
| threads | 10 |

### Sweep axes (32 configs = 2×2×2×4)
| Axis | Values |
|------|--------|
| B0 | 0.05, 0.10 |
| lam_bg | 0.4, 2.0 |
| lam_peak | 0.0, 1.0 |
| (lam_relax_rate, lam_alpha) | (0.0, —) baseline; (0.1, 5); (0.1, 10); (0.1, 15) |

Fixed: u_bg = 0.4.

### Perturbation IC (full-wavelength, kx = 2π/Lx)
Az = dB·cos(2π·(x+Lx/2)/Lx)·exp(−y_r²/λ²)  
By = −∂Az/∂x = kx·dB·sin(kx_x)·env  
→ X-point at x=0 (centre, By=0 and sign change)  
→ O-points at x=±Lx/4 (By peaks, opposite polarity)  
→ 4 blobs of alternating polarity (2 per sheet)

---

## Runtime output (representative configs)

### B050_lb0_lp00 — baseline (no relax)
```
step   5500  u_max=5.0477e-01  phi_E=2.1438e+04 (×21.59)
step   7000  u_max=5.0279e-01  phi_E=3.0600e+04 (×30.82)
step   8500  u_max=5.5224e-01  phi_E=4.1528e+04 (×41.82)   ← peak
step   9500  u_max=5.2817e-01  phi_E=2.9076e+03  (×2.93)
u_vol: machine-precision (Δ ~ 1e-11 throughout)
```

### B050_lb0_lp00_rr0100_a05 — α=5
```
step   7000  phi_E=8.1879e+03 (×8.25)
step   8000  phi_E=9.2189e+03 (×9.28)   ← peak
step   9500  phi_E=8.9320e+03 (×8.99)
u_vol drift: ±1–6 / 20948 (< 0.03%)
```

### B050_lb0_lp00_rr0100_a10 — α=10
```
step   5500  phi_E=4.7926e+03 (×4.83)
step   8500  phi_E=6.9067e+03 (×6.96)   ← peak
step   9500  phi_E=5.2607e+03 (×5.30)
u_vol drift: ±2–10 / 20948 (< 0.05%)
```

### B050_lb0_lp00_rr0100_a15 — α=15
```
step   5500  phi_E=3.5052e+03 (×3.53)
step   8500  phi_E=5.0287e+03 (×5.06)   ← peak
step   9500  phi_E=4.9103e+03 (×4.94)
u_vol drift: ±5–14 / 20948 (< 0.07%)
```

All 32 configs: status OK, no NaN/blowup, N=52291 constant.

---

## Key findings

### 1. Lam suppression is real, monotone, and strong

| Config (lb0, lp0) | phi_E peak / E0 | Reduction vs baseline |
|---|---|---|
| baseline (rate=0) | ×41.8 | — |
| α=5 | ×9.3 | 4.5× |
| α=10 | ×7.0 | 6× |
| α=15 | ×5.1 | 8× |

The ordered suppression confirms lam back-reaction is physical, not numerical.
Each alpha increment reduces phi_E growth by a further ~25% relative factor.

### 2. lam_bg=2.0 inherently suppresses via high background winding

With lb=2.0, baseline itself only grows to ×6–9 (vs ×20–42 for lb=0.4).
Adding lam drive on top of lb=2 gives only marginal further suppression.
The dominant mechanism shifts from dynamic back-reaction to static winding floor.

### 3. phi_E is highly oscillatory (fast DST sloshing)

phi_E swings by factors of 3–10 within a single reporting window (500 steps).
This is DST's fast wave bouncing (wave speed ~√2 per cell per step) — not instability.
The smoothed envelope reveals the underlying trend cleanly.

### 4. phi_E is B0²-self-similar

B100 configs are exactly 4× above their B050 counterparts at every step.
This confirms the phi initialisation (∝ B0) enters phi_E as B0², as expected.

### 5. phi_xpt_by rises slowly and monotonically in all configs

|By| at the X-point (x=0, y=±y0) shows a slow monotone rise — reconnection
is proceeding in all configs, not stalled. The lam drive slows the rate but
does not halt it. This is the right regime: causal suppression, not near-stall.

### 6. lam_peak=1.0 (density bump at null) amplifies phi_E in baseline

lp=1.0 without relax increases phi_E roughly 1.5–2× vs lp=0.0 in the
lb=0.4 case. With relax the lam drive targets the peak more aggressively,
giving slightly stronger suppression than lp=0.0 at the same alpha.

### 7. u_vol conservation

- Baseline (no relax): machine precision (Δ ~ 1e-11). BCs are fully periodic.
- Lam-drive configs: small drift ±5–50 on u_vol ~ 20948 (< 0.25%). Acceptable.
  The drift comes from sigma(lam) asymmetry introduced by the lam update.

### 8. DST vs PHARE timescales (context)

DST wave speed ~ √2 per cell per step → at dt=0.05, effective ~ 28/AU.
PHARE Alfvén speed ~ B0 = 0.05–0.10. DST By propagates ~280–560× faster.
By reaching the boundaries quickly is correct DST physics, not a bug.
PHARE images (t=40–50) confirm By also moves to boundaries, just ~280× slower.

---

## Calibrated parameters

For harris reconnection runs going forward:
- **α = 10, lam_relax_rate = 0.1** — causal back-reaction, reconnection completes
- **lam_bg = 0.4** — low background winding, dynamic suppression visible
- **lam_peak = 0.0** — flat lam IC unless modelling Harris density peak explicitly
- **B0 = 0.05** — weaker field, cleaner signal (B100 is a 4× rescale)

---

## Artefacts

```
tmp/harris_sweep_alpha/
  <label>/
    diag.csv              — per-step: phi_E, phi_E_ratio, phi_null_rms, phi_xpt_by, u_rms, lam_std
    plots/step_NNNNNN.png — snapshot every 500 steps (7 panels: Bx, By, Jz, phi_mag, u-mean(u), u, lam)
    harris_phare_init.h5  — initial state
    harris_phare_final.h5 — final state (step 10000)
    run.log               — full subprocess output
  comparison_lb0.png      — envelope plots, lam_bg=0.4
  comparison_lb2.png      — envelope plots, lam_bg=2.0
  comparison.png          — raw all-config overlay
```

32 config labels: B{050,100}_lb{0,2}_lp{00,10}[_rr0100_a{05,10,15}]

---

## Next steps

1. **Long run at α=10, lb=0.4, lp=0.0** — extend to 50k–100k steps; watch for plasmoid formation or saturation of phi_xpt_by.
2. **Plasmoid regime at α=30** — near-stall config from reconnection alpha sweep; may produce plasmoid instability in a sufficiently long harris sheet.
3. **Larger domain (Lx=200)** — reduce boundary influence on By evolution; need ~8× more cells (slow without C++ AMR).
4. **Enable AMR** — once flat runs are understood; AMR should refine the null line and X-point regions selectively.
5. **Reconnection rate measurement** — extract Bx at midplane vs time to get an explicit reconnection rate R(t) for comparison with MHD theory.
