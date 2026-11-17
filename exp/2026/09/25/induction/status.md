# Induction on AE with a magnetisation source (2026-09-25)

`next` 09-25 item 1. 09-24 diagnosis: the anchor-sampled loop current has a
component in null(K) (not a gradient, curl-free on the anchors) that no static
A can balance, so it grows without bound and loop 1's self-flux drifts.

Fix: `continuous.current_from_magnetisation(lat, m)` = C0SQ antisym(K^T m),
m = z-hat magnetisation of the smoothed disc spanned by loop 1
(m_z = g(z) * erfc((rho - a1)/(sqrt2 s))/2; continuum curl m = the 09-24
gaussian tube). Lies in range(K^T) by construction.

Source scale check (J_mag vs sampled J, projection fit): 0.81 / 0.95 at
h=0.5 / 0.25 for s=0.4; 0.94 / 0.98 for s=0.8 -> converges ~(h/s)^2,
resolution smoothing, not a normalisation error.

## Static target (`static_solve.py`, CG on C0SQ curl_curl_sym A = J)
| src | R | its | rel residual | Phi2/M | Phi1 |
|---|---|---|---|---|---|
| mag | 12 | 1500 | 3.8e-4 | 1.0565 | 0.2224 |
| mag | 18 | 2500 | 7.6e-5 | 1.0486 | 0.2219 |
| sampled | 12 | 1500 | 7.6e+2 (diverging) | 2.5e7 | -3.5e7 |
Sampled J has no static solution: confirms the null(K) diagnosis directly.

## Dynamics (`induction.py --src mag --R 18`, h=0.5, a1=a2=d=2, s=0.4, Tr=2.5)
| t | Phi2/(M I1) | Phi1 | max u |
|---|---|---|---|
| 2.56 | 1.268 | 0.2419 | 2.1e-5 |
| 3.59 | 1.050 | 0.2216 | 2.6e-5 |
| 4.62 | 1.042 | 0.2215 | 5.7e-5 |
| 5.50 | 1.043 | 0.2216 | 3.0e-4 (at wall r=18; interior 1.9e-5) |
09-24 (sampled+proj, same run): Phi2/M 1.13-1.31, Phi1 0.22 -> -0.19,
max u 4.4e-2 growing. Now: Phi1 steady, Phi2 settles to ~1.04-1.05 after
the ramp transient, matching the R18 static target (1.0486 / 0.2219) to <1%. Late max u growth is at the
edge when the transient reaches it (no E boundary treatment).

## Refinement (C++ `continuous::step_ae`, `--cpp`; numpy-identical at h=0.5)
Continuum reference for the actual gaussian tubes (`tube_neumann.py`:
filament kernel integrated over the current, same holonomy weight,
unbounded, mu0 = 1/c^2): Phi2/M = 1.0508, Phi1 = 0.2035. The filament M is
not the right reference -- the ~5% "gap" was the tube width.
| h | N | Phi2/M (end of hold) | Phi1 | wall time |
|---|---|---|---|---|
| 0.5 | 254k | 1.043-1.048 | 0.2216 (+8.9%) | 119 s (C++, 2 thr) / 956 s numpy |
| 0.25 | 2.03M | 1.0492 (-0.15%) | 0.2066 (+1.5%) | 2124 s (C++, 5 thr) |
| 0.125 (R144) | 16.2M | 1.0505 (-0.03%) | 0.20351 (+0.005%) | 6199 s (C++, 8 thr) |
| continuum | | 1.0508 | 0.2035 | |
Phi1 error ratio 5.8 per halving (>= 2nd order). Mutual induction with
mu0 = 1/c^2 reproduced to 0.15% at h=0.25. Static target h=0.25 (CG 6000 its, res 2.5e-5): Phi2/M 1.0522, Phi1 0.2068 -- dynamics within 0.3%.

Test: `tst/substrate/test_continuous.py::TestStaticCurrentBalance`.

R144 (09-25 evening): the last step is too good (Phi1 error /300, 2nd order predicts
/4-6), so R144 has reached the continuum reference's own precision (tube_neumann
quadrature). It agrees within the reference; no convergence order can be read from it.
Final state: /bk/dst/2026/09/25/induction_mag_R144_h0.125_cpp_final.npz. Runs are now
named by R/h. None of the saved states has been plotted yet (queued in `next`).
