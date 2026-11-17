# Candidate in an ambient bath, lam from the potential (1/r) — 2026-09-30 (CONDITIONAL tests)

**INVALID as structure evidence (Philip 10-01): lump width 1.5 cells. Still valid: AmbientBC construction / tests, the linear-superposition argument (ambient cannot act on a structure under a linear step()), and the depletion >= 0 / lam_amb guard reasoning. An electron below R20 (radius 20 cells) is totally invalid, R20 maybe too small (mem feedback_structure_resolution). Survival / retention / response numbers below are records of the setup only.**

`run.py --source potential`: 1 + lam = (1 + lam_far)(1 + lam_amb + alpha psi), psi = lattice Poisson of the local
energy density minus the bath mean (Jeans swindle, assumption), psi = 0 on the boundary cells. Philip 09-30: lam
follows the potential (1/r, like Newtonian gravity), not the field (1/r^2). lam_far = 5.5e-7 (near-Earth, uniform:
assumption). Implicit midpoint (H exact), R = 12, width 1.5, 20 crossings, AmbientBC (sponge -> bath), bath relaxed
5 crossings under the rule first. T = ambient PRESSURE knob (lattice units, bath energy ~4T per cell), not a physical
temperature. Consistency (depletion >= 0) needs a held lam_amb >~ alpha T (0.2 alpha T gauss/dipole a100 + gauss a1000,
1.0 alpha T dipole a1000: candidate-bath cross terms dip further); alpha_eff = alpha / (1 + lam_amb).
Logs `cold_*.log`, `plasma_*.log`, `raw_*.log`; records `rec_*.npz`; end states (domain + bath) in
/bk/dst/2026/09/30/ambient_flux_lam/.

Metric: domain - bath departure is INVALID for T > 0 (the bath inside the domain decorrelates from the reference bath
under the nonlinear rule: departure > 1). Used instead: core (r < 4.5) energy excess over the bath's core energy,
over the candidate's initial core energy E0, at t = 20 crossings. Noise ~ +-0.04 (T = 0.01), ~ +-0.4 (T = 0.1).

| IC | alpha | T | alpha_eff | excess / E0 | core lam dom / bath | core/ring energy |
|---|---|---|---|---|---|---|
| gauss | 10 | 0 | 10 | 0 | 7.5e-7 | - |
| gauss | 100 | 0 | 100 | 0 | 9e-5 | - |
| gauss | 1000 | 0 | 1000 | 0.88 | 28 | 401 |
| gauss | 1000 | 2e-9 (raw) | 1000 | 0.88 | 28 | 401 |
| gauss | 100 | 0.01 | 83 | 0.05 | 1.27 / 1.07 | 1.02 |
| gauss | 1000 | 0.01 | 333 | 0.39 | 19.7 / 10.8 | 1.25 |
| gauss | 100 / 1000 | 0.1 | 8.3 / 48 | 0.38 / 0.45 (noise) | 11.1/10.8, 130/115 | 1.02 |
| dipole | 10 / 100 | 0 | | 0 | ~1e-5 | - |
| dipole | 1000 | 0 | 1000 | 0.34 (0.91 -> 0.76 -> 0.34: leaking) | 5.5 | 11.5 |
| dipole | 100 | 0.01 | 83 | 0.10 | 1.28 / 1.07 | 1.02 |
| dipole | 1000 | 0.01 | 91 | 0.15 | 19.9 / 18.6 | 1.03 |
| dipole | 100 / 1000 | 0.1 | 5.0 / 9.9 | 1.3 / 0.95 (noise) | 12.2/10.8, 205/192 | 1.03 |

Reading:
- Cold: a sharp transition between alpha 100 (gone) and 1000. gauss alpha 1000 keeps 0.88 flat, but core lam 28
  (sigma 0.036): the core has lived < 1 crossing of its own proper time -> frozen by its own well (black-hole-like
  limit of a Newtonian-style lam), not survival. dipole alpha 1000 leaks steadily.
- Raw near-Earth (lam_far only, bath T <= 3e-9 to keep depletion >= 0) = cold, to 5 digits.
- Pressure (T = 0.01) does not rescue anything clearly: the strong cases lose retention (0.88 -> 0.39, 0.34 -> 0.15),
  partly through the alpha_eff dilution a consistent lam_amb forces. alpha 100 keeps 0.05-0.10 vs 0 cold, i.e. 1-2
  noise units: needs several bath realisations before reading anything into it.
- T = 0.1: bath energy ~20x the candidate in the core; excess is inside the noise. Uninformative.

## Verdict (conditional)
With a Newtonian-style (1/r) depletion lam, no candidate survives in a proper-time sense, cold or pressurised. The
only "retention" is the frozen-clock limit. The ambient-pressure effect at weak coupling is at noise level.

## Next
- Several bath realisations (seeds) at T = 0.001-0.01, alpha 100-300, to settle the weak-coupling pressure hint.
- Proper-time bookkeeping: retention vs integral of sigma_core dt, so frozen clocks cannot pass as survival.
- Real-expectation track: cold + held uniform fields (sunlight ~6e-18 of the field at r_e, proton microfield by
  density: shadow/tail lobes vs sunlit/solar wind); uniform B needs the AE model.

## Bath seeds (2026-09-30 evening, `--seed_tag`, `analyze_seeds.py`, plot `seeds_T0.01.png`)
rec now also stores E_core (domain), E_core (bath), E_tot both, mean core sigma. Excess = (E_core - E_core,bath) / E0,
late = last 30% of 20 crossings, mean +- sem over seeds. Null = gauss at amp 0.1 (alpha amp^2 = 1: negligible well),
excess normalised by the amp = 1 E0 (absolute energy comparison).

| case | seeds | late excess | fifths (seed mean) | core lam dom - bath | tau/t |
|---|---|---|---|---|---|
| gauss a100 T0.01 | 4 | +0.030 +- 0.002 | .32 .028 .033 .031 .031 | +0.145 | 0.44 |
| dipole a100 T0.01 | 4 | +0.058 +- 0.005 | .21 .062 .062 .065 .057 | +0.144 | 0.45 |
| NULL gauss amp 0.1 a100 T0.01 | 4 | +0.026 +- 0.002 | .003 .020 .023 .031 .027 | +0.137 | 0.46 |
| gauss a300 T0.01 (amb 0.6) | 2 | -0.019 +- 0.005 | .83 .31 .031 -.014 -.020 | -0.097 | 0.20 |
| dipole a300 T0.01 (amb 0.6) | 2 | +0.009 +- 0.009 | .54 .035 -.007 .005 -.001 | +0.017 | 0.23 |

Reading:
- The seed spread is small (sem 0.002-0.005): the old "+-0.04 noise" was per-sample, the late mean is well determined.
- The gauss a100 excess is the METRIC, not the candidate: the null (1% energy, no well) grows to the same absolute
  excess (0.026 vs 0.030) and the same core lam offset (0.137 vs 0.145). The decorrelated domain drifts away from the
  closed bath copy (sponge-coupled vs closed walls): the bath is not an unbiased reference once they decorrelate.
- dipole a100 sits ~0.03 above the null (~5 sem; null run with the gauss IC only).
- alpha 300 removes it (<= 0.01, gauss negative; its own null not run, so its baseline is unknown).
  A retention that fades with stronger coupling is not a trapping mechanism.

Verdict (conditional): the weak-coupling pressure hint is mostly a reference bias; the dipole residual (~0.03 E0,
alpha 100 only, non-monotonic in alpha) is not evidence of retention. Any further bath scoring must subtract a
same-alpha null ensemble (or use a reference that sees the same sponge/walls as the domain).
