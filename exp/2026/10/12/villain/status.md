# 10-12 villain: static Peierls-Nabarro landscape of a Villain ring (deciding (a) soft core vs (b) Villain)

barrier.py -> barrier_P72_10-12.log. phi = W (dtheta - 2 pi n), n = +-1 on faces crossing a disc (ring = disc edge).
For fixed n, theta relaxes linearly: E_min(n) = energy of the div-free part of -2 pi W n (project_div_free).
Scan over one lattice period (2h): z-ring moved z (commensurate), moved x, tilted (1, 2, 3.3) ring along normal /
lateral, and a kinked hop (line moved by h over azimuth 0..alpha, swept cylinder strip; full-hop closure exact to 1e-12
after replacing linear-in-r rim crossing with the exact quadratic).

  R    E      dE/dR h   spread (5 paths)   spread / (dE/dR h)   spread / E
  6    2410   299       35-71              0.12-0.24            1.4-3.0e-2
  10   4543   323       28-44              0.09-0.14            0.6-1.0e-2
  14   6907   339       32-58              0.10-0.17            0.5-0.8e-2

Spread does NOT grow with R (no coherent commensurate pinning, even z-ring along z: the curved rim samples every lattice
phase). Per unit length it falls ~1/R; relative to E ~1/(R ln R) -> vanishes under refinement (R in cells grows) even
though the core stays one plaquette. Kinked hop costs no more than a rigid shift (no kink-nucleation barrier).
Hamiltonian reading: H = E(R) + U(z) with |U| ~ 0.1-0.2 dE/dR h -> the ring would breathe by dR ~ 0.1-0.2 h, not pin.

Reading: the 10-08 freeze is NOT a lattice barrier. tick.py's wrap(dtheta) already IS Villain with n = round; a pure
phase (with or without free integers) has no place for the commutator's transverse part except n, which changes only by
whole flips -> the transverse drive is lost (dropped = the kappa 5 energy pumping). So (b) alone == current tick model.
(b) becomes distinct only with a store for the transverse part: real face remainder r, phi = W (dtheta - 2 pi n + r);
conversion n -> n +- 1, r -> r +- 2 pi leaves phi unchanged = energy-exact, no picked gate. Open: rule that keeps r's
loop circulation whole-number (else quantization is only bookkeeping). (a) supplies the store as an amplitude (maybe lam).

## Big rings, box side 4R (Philip: "big enough?" -> R 6-14 was below the R 20 floor, box ~3R): barrier_R*_box4R_10-12.log
  R    P    N      E       dE/dR h   spread range (5 paths)   mean   spread / E (mean)   per length (mean)
  10   70   86k    4545    324       28-46                    36     0.79e-2             0.57
  20   140  686k   10533   372       25-64                    44     0.42e-2             0.35
  30   208  2.25M  17020   399       27-51                    41     0.24e-2             0.22
  40   278  5.37M  23874   410       16-64 (8 shifts)         44     0.18e-2             0.18
Spread is ~flat over R x4 (coherent pinning would be x4, random per-segment sqrt -> x2; seen ~x1.2, R 40 sampled
coarser so maybe low). Spread / (dE/dR h) 0.04-0.17 at every R. Conclusion unchanged and now inside the resolution floor:
no lattice pinning of a curved Villain ring; relative landscape falls ~1/R. Sampling note: R 20 z-shift 17.9 (2 shifts)
vs 63.9 (12) -- spreads are sampled max-min, read them as order of magnitude.
Sampling check (barrier_R20_box4R_n48_10-12.log, R 20, 48 shifts): z 63.9 -> 67.9, x 24.7 -> 35.5, kinked 32.7 = 32.7
(12 -> 48). Coarse sampling reads up to ~30 % low; R 40 (8 shifts) mean likely ~55-60 -> growth R 10 -> 40 ~x1.5
(between flat and sqrt R, far from coherent x4); spread / E still falls 0.8e-2 -> ~0.25e-2.

## Dynamic (b) + remainder: vill.py (P 72, R0 9, kappa 0.132; logs vill_P72_R9_*_10-12.log)
n <-> r transfer by 2 pi is phi-neutral, so (b) + remainder evolves exactly as step() + kappa commutator on a real phi;
n = per-triangle label round(Gamma / 2 pi). IC = 10-08 tick ring (each triangle exactly 0 or +-2 pi, 408 defects).
  kappa 0: frozen, every triangle whole to 1e-15, 408 defects, E_lf bounded (+-3 leapfrog oscillation) -- as 10-08.
  kappa 0.132: max off-integer at defects 0.47 by t 0.5; leaked |Gamma| on non-defect triangles 1.6 / 6.5 / 17 / 116
  quanta per defect at t 0.5 / 1 / 1.5 / 2; labelled count 408 -> 168 -> 85 -> 15 (circulation spread below 1/2 per
  triangle); |Gamma| centroid moves z +0.13 by t 2 (~0.06 / time); E_lf 8007 -> 8151 (t 2) -> 16.7k (t 3) -> NaN.
  dt x 0.5: identical to 3 digits (smearing, collapse, E rise) -> not a time-step artefact.
Reading: the commutator advects circulation (Kelvin) but does not keep it concentrated; on a one-triangle core it
smears whole quanta into fractions within t ~ 1 and then runs away. (b) + remainder therefore gives no quantization by
itself -- the integers become labels of a smeared field. Quantization needs a restoring / concentrating relation: an
amplitude core (a: phase single-valued wherever amplitude != 0, circulation quantized by the zero) or an explicit
concentrating rule in (b) (a gate, picked). Energy rise with the commutator on a singular field (dt-independent) is
open: lie is energy-orthogonal per evaluation; check the step() / carry splitting and midpoint convergence there.

## Geometry vs leak (Philip: if the geometry is wrong, a vortex leaks instead of cycling locally) -- vill_P72_R9_wt*_10-12.log
Same rules (step() + kappa 0.132 commutator), P 72 R0 9 a0 1, t 0 -> 1.53: rms core distance / sum|Gamma| / E drift / speed
  tick (1 triangle)  0.30 -> 2.05     408 -> 1516   runaway        --
  smooth wt 1        1.82 -> 1.89     620 -> 597    1e-5           0.027
  smooth wt 1.5      2.71 -> 2.75     627 -> 624    5e-6           0.036
  smooth wt 3        5.29 -> 5.29     639 -> 643    1e-6           0.052
Leak is geometric: zero at wt 3, creeping at wt 1, fast at one triangle (|Gamma| total x3.7 = splitting into +- pairs).
Energy failure appears only on the geometry the rules cannot host. Open: thinner smooth rings SLOWER at fixed a0
(continuum thin ring is faster). Smooth rings cycle but their circulation is a0's, not 2 pi: wanted = a geometry that
cycles AND carries whole quanta ((a) amplitude core = candidate; tokamak core + windings another).

## Matched-circulation attempt -- INVALID (vill_P72_R9_wt*_c*_10-12.log)
Linked circulation on z-lines (axis, R0 / 2, outside) shows the 11/lie smooth ring carries ~zero NET circulation:
phi = curl(f e_phi) with f a Gaussian on the tube -> flow loops inside the tube, nothing threads the hole (axis 0.0x of
the R0 / 2 line). Rescaling to circ x 2 pi by the axis value blew wt 1.5 to E ~1e14. So the morning geometry comparison
was unequal: tick ring links a whole turn, smooth rings link none. Smooth cycling says nothing yet about net-turn
structures.
Philip: continuous model approximating a discrete one; no pi in a discrete model. -> theta mod 2 pi / wrap / +-pi
are U(1) idealisations; discrete counterpart = Z_N clock (integer state count per cell, offset = count difference
mod N, whole turn = loop counts close). Next: propose a Z_N update rule (Philip's call), then leak-test vs N.

## Whole-turn smooth rings (vill.py --smooth tau: phi -= eps curl_curl_dec phi on the tick IC, then div-free project)
First try blew up after t 2: 4 triangles kept a whole quantum = one square face with offset exactly pi (ring centred on a
lattice symmetry -> wrap(+-pi) ambiguous). Ring offset (0.13, 0.07, 0.11) removes it (max |a| 0.61). NOTE: the earlier
tick-IC runs today (and 10-08 tick.py) had the same exact-pi faces.
Offset runs, P 72 R0 9 kappa 0.132, linked turn on the axis kept 1.00 +- 0.014, outside 0:
  tau 1: E 2836.0689 -> 2836.0685 (1e-7); z speed 0.165; rms 2.01 -> 2.87 / 3.46 / 3.89 / 4.19 (t 1-4, slowing);
         sum|Gamma| 414 -> 587 (+42 %)
  tau 4: E exact (1e-7); speed 0.129; rms 3.95 -> 4.20 / 4.48 / 4.75 / 4.99; sum|Gamma| 398 -> 442 (+11 %)
vs the net-zero smooth ring wt 3: rms +0.1 %. Whole-turn rings move and conserve energy and the turn exactly but the
core keeps widening (decelerating for tau 1) and |Gamma| grows (+- filaments). Caveat: R0 9 in side 41.6 with a net
flow through the hole -> strong image / return-flow coupling; below the R 20 floor. Next: R 20, P 140.
R 20 (P 140, tau 4.94 = R 9 tau 1 scaled by f 2.22; vill_P140_R20_smooth4.94_off_10-12.log): E exact (1e-7), turn
1.000 -> 1.0006, speed 0.085 (R 9: 0.165, ratio 1.94 ~ f with log). rms 4.45 -> 4.90 / 5.34 / 5.74 / 6.11 / 6.44 / 6.75 /
7.03 (t 0.5 steps), still ~0.6 / time; sum|Gamma| +15 % by t 3.56.
Refinement reading (same shape, lengths x f, circulation fixed -> time x f^2): rms / R at matched scaled time 0.72:
R 9 0.223 -> ~0.29, R 20 0.222 -> 0.351. Relative widening GROWS under refinement; absolute widening ~same cells / time
at both -> a grid-scale process, not converging. R 40 not run (would only widen faster in R units).

## The widening is the commutator creating circulation in irrotational flow (fast-fail test, farfield.py)
The |Gamma|-weighted rms is linear in |eps| (both signs) -> it picks up small circulation the term creates everywhere,
not core spreading per se. Binned by distance from the core: ring |Gamma| ~1e-16 beyond ~13 cells, but kappa * term
creates 2.5e-5 / 3.7e-6 (R 9, 13-20 / 20-40 cells) and 1.2e-5 / 7e-7 (R 20, 20-40 / 40-99) per unit time, wherever
the through-hole flow is. Continuum: zero (advection of irrotational flow is a pure gradient). Net-zero rings have no
far flow -> held. kappa 0: frozen exactly.
Fast fail (one evaluation, shape scaled with R, tau = (R / 9)^2): far / core term ratio (far d > 3 rms, core d < rms):
  R 9 0.337   R 20 0.552   R 40 0.720 (P 278, 5.4M cells; seed 40 min, farfield_R40_10-12.log)
Resolution problem would fall ~1 / f^2; it rises (~ +0.25 per ln f) -> STRUCTURAL: the commutator does not map
closed (curl-free) face fields to exact changes. Continuum Cartan: L_v a = i_v da + d(i_v a), so where da = 0 the
change is a pure gradient. Fix candidate: build transport in Cartan form (vorticity-flux term i_v da + gradient), so
irrotational regions get exactly zero circulation change by construction (Elcott / DEC wedge options kept 10-11).
