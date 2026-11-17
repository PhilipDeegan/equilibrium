# 10-13 cartan: phi self-transport as -i_v(da) (Cartan form) -- replaces the 10-11 exchange-matrix commutator

fit.py: per Delaunay triangle t (hex, square, hex edges) an antisymmetric coupling of the edge fluxes v = W phi,
da_e = -sum_{t ni e} omega_t (K_t v)_e. Fitting K (hex-square alpha, hex-hex beta) so the sum over a face's triangles is
the continuum i_u omega = (Omega x u) . d for uniform curl and flow: alpha -3/8, beta -3/16 (beta / alpha 1/2), residual
2e-16 (exact). In a = phi / W that is the vector-free rule
    da_e/dt = -kappa i(e),  i(e) = -(3/4) / W_e^2 sum_{t ni e} omega_t (a_next - a_prev)   (loop-oriented)
-> equilibrium/substrate/continuous.py cartan_transport (spacing h), tst TestCartanTransport (antisymmetry, sum phi . T
= 0 to 1e-13, zero on gradients and away from a localized vortex, exact -W (Omega x u) . d at h 0.5 / 0.25).
Energy: antisymmetric per triangle in v -> sum v . da = 0 exactly. Locality: a face changes only if a triangle touching
it has omega != 0 -> no circulation created in irrotational flow. Gradient part d(i_v a) left out (step()'s u sector).

diag.py (one evaluation on the vill.py whole-turn seed, --smooth (R / 9)^2, off 0.13 0.07 0.11):
              far / core dGamma     energy (sum phi.T / sum|phi T|)
  R 9  lie    0.337                 2.5e-18
       cartan 7.8e-4                4.5e-18
  R 20 lie    0.552                 -3.5e-18
       cartan 4.4e-4                -6.6e-19
cartan's far residual tracks the seed's own small far circulation (smoothing leftover, ring |Gamma| far max 8e-5 /
1e-5); falls with refinement, lie's rises. Core dGamma corr(cartan, lie) 0.77 (R 9) / 0.51 (R 20) -- lie's own errors.

kappa: net-zero smooth ring (11/lie IC, P 120 R0 12 wt 3), t 0 |Gamma|-weighted speed: lie kappa 0.132 -> 0.03541;
cartan kappa 0.9158 -> 0.03226 -> matched speed at kappa 1.005. kappa = 1 = Euler (d_t a = -i_u da with a the velocity
1-form): the 7.58 picked on 10-11 is the commutator's own normalisation, not a free constant.

Whole-turn leak test (vill.py here = 10-12 vill.py + --flow cartan; kappa 1, offset seed, linked turn on the axis):
            t:    0       1.02    2.04    3.05    3.56     sum|Gamma|        E               speed
  R 9  rms        2.011   2.025   2.035   2.038   2.044    414 -> 414        1e-6            0.17
       lie 10-12  2.01    2.87    3.46    3.89    4.19     414 -> 587
       kappa 0    2.011   frozen                           exact             exact           0
  R 20 rms        4.447   4.449   4.453   4.456   4.457    937.5 -> 937.1    1e-7            0.077
       lie 10-12  4.45    ~5.34   ~6.11   ~6.75   7.03     +15 %
Turn linked on the axis 1.000000 throughout (both). Relative widening (rms / rms0 over t 3.56): R 9 +1.6 %, R 20 +0.22 %
-> ~7x smaller at f 2.22 (f^2 4.9; f^2 log-ish) -> converging, a discretisation residual. The 10-12 widening was the
commutator, now gone: whole-turn smooth rings travel, keep shape, turn, energy and circulation.
Logs: diag_P72_R9_10-13.log, diag_P140_R20_10-13.log, vill_P72_R9_smooth1_off_cartan_k1_10-13.log,
vill_P72_R9_smooth1_off_k0_10-13.log, vill_P140_R20_smooth4.94_off_cartan_k1_10-13.log.

Consequence for quantization: this transport conserves every loop's circulation (Kelvin) up to the small core
discretisation residual. An off-quantum ring keeps its off-quantum circulation -- shedding toward a preferred value
cannot come from step() + Cartan transport; it needs another relation (pressure equalisation, Philip 10-12: what
carries it?).

## One-triangle (tick) core under Cartan (vill_P72_R9_tick_off_cartan_k1_10-13.log, offset seed, kappa 1, t 2.55)
Still smears like under lie: off-integer 0.49 at defects by t 0.5, labelled defects 420 -> 56, sum|Gamma| 420 -> 2657
(+- pairs), core rms 0.28 -> 1.53. Unlike lie: no energy runaway (E_lf 8032.0 -> 8030.75, settles), linked turn exact.
A grid-scale core is not held by the transport -- the sharp core leaks (nature not sharp); smooth cores hold.
Philip 10-13: off-quantum shedding shouldn't happen (bad test). Quantization is reciprocal: offloading needs an
activation energy (shell jumps); the discrete substrate may be an implicit valve. Record total energy.
Philip 10-13 on the one-triangle core: "like a race car, it cannot turn fast enough" -> bounded speed = minimum turning
radius; the triangle core is not a candidate, its smearing is expected. Turning limit = narrowest core that holds.

## Turning limit: core-width scan (whole-turn ring, P 72 R0 9, kappa 1, offset; vill_P72_R9_smooth*_off_cartan_k1)
  tau    core rms0   sum|Gamma| growth by t 2.55   rms at t 2.55   E drift
  0.05   0.49        +281 %                        1.43            4e-5
  0.1    0.69        +133 %                        1.38            1e-5
  0.2    0.94        +42 %                         1.31            1e-6
  0.4    1.29        +6.8 %                        1.43            1e-6
  0.7    1.69        +0.4 %                        1.73            1e-6
  1      2.01        -0.25 %                       2.04            1e-6
Turn exact in all. Holding sets in at core rms ~1.5-2 cells; narrower cores split into +- pairs and widen toward ~1.4.
Circulation at tau 0.4 (--circ, same width): circ 0.5 / 1 / 2 -> sum|Gamma| +1.0 / +6.8 / +19 % by t 2.55. At matched
transport time (t x circ) the three agree within ~2x (circ 2 at t 1.02 +2.1 % vs circ 1 at t 2.04 +3.9 %; circ 0.5 at
t 2.55 +1.0 % vs circ 1 at t ~1.3 ~+0.8 %): the leak is per distance travelled, so the minimum turning radius is ~fixed
in cells (geometric, ~1.5-2), not growing with speed (a car's v^2 / a_max would). Logs *_circ{0.5,2}_*.
Philip 10-13: smoothness = pressure equalising in all directions (balloon). Here a narrow core does NOT equilibrate: it
shatters (sum|Gamma| x3-4, +- pairs) rather than widening at fixed sum|Gamma| -> the equalising relation is missing.
Philip 10-13: don't need perfect equalising; imbalance = what bonds (energy gap = unequalised relation); pinch -> two balloons, each half supported.

## CORRECTION: long run (T 20) -- the whole-turn ring holds only to t ~4, then leaks (vill_*_T20_mid{2,3,6}_10-13.log)
P 72 R0 9 tau 1, kappa 1: sum|Gamma| 414 -> 415 (t 4) -> 435 (8) -> 465 (12) -> 508 (16) -> 555 (20, +34 %); rms 2.01 ->
2.94; speed 0.17 -> ~0.10 (decelerating); E exact 1e-6, turn exact. Identical to 4 digits for midpoint 2 / 3 / 6 (not the
integrator); same +34 % at P 120 (pair/ single run: not box size). Increment per 2 time units rises 8 -> 24 then ~flat
(steady shedding rate, not exponential) -- reads like a wake from a non-equilibrium core profile (classical ring
relaxation), or residual discretisation. The R 20 "+0.22 %" was also only t 3.6. Open: R 20 at matched travel (t ~70,
~1 h) to separate converging residual from physical wake.

## C++ port (13:00-13:20)
AdaptiveLattice::cartan_transport (adaptive_lattice.hpp, two-pass: per-triangle contributions, then a static per-face
gather table) + step_cartan(cl, dt, kappa, n_iter) = step() + implicit-midpoint transport (vill.py's carry) in C++.
Bindings alat.cartan_transport(i, phi(14, N)), alat.step_cartan(i, dt, kappa=1, n_iter=3); AnchorCluster.cartan_transport.
tst/substrate/test_continuous_cpp.py TestCartanTransport (5 tests, 0.1 s): == numpy to 1e-13, step_cartan == step() +
numpy carry over 10 steps to 1e-12, antisymmetry, energy orthogonality, zero on gradients / away from a localized vortex,
exact uniform-limit at h 0.5 / 0.25. Full test_continuous{,_cpp} 41 tests OK.
P 140 (686k): step_cartan 0.55 / 0.21 / 0.17 s per tick at 1 / 4 / 8 threads (numpy carry ~0.6-0.8 s single).
vill.py --cpp: identical to the numpy path (P 72 R 9 t 2, all printed digits).

## Late leak vs refinement: R 20 to t 70 (vill_P140_R20_smooth4.94_off_cartan_k1_T70_cpp_10-13.log, C++, 13 min)
Flat to t ~15, then steady growth: t 70 sum|Gamma| +5.4 %, rms 4.447 -> 5.083 (+14 %), z travel 5.0 (0.25 R), speed
0.077 -> 0.064; E 1e-8, turn exact.
Matched travel 0.25 R: R 9 (t ~16.5) sum|Gamma| +23 %, rms +33 %; R 20 +5.4 %, +14 %. Ratio ~4x (sum) / ~2.4x (rms)
at f 2.22 (f^2 4.9). Onset at matched travel similar (~0.08-0.09 R). -> the late leak shrinks under refinement: mostly
discretisation residual, not a resolution-independent wake of the seed core. Order between 1 and 2 (rms) / ~2 (sum).
A physical wake from the scaled seed would be the same at matched travel.
R 40 (P 278, 5.37M, T 145 = 0.15 R travel): seed ~25 min, ~1.2 s / tick (2.7 h); killed at t 10 for shutdown (t 10: z speed 0.0387 = t 0 estimate; rms 8.878 -> 8.884, sum|Gamma| -0.02 %, E 1e-9; travel 0.01 R, before the ~0.08 R leak onset; 1.3 s / tick). t 0: rms
8.878, sum|Gamma| 1867.1, E 12883.437629. Queued in next.
