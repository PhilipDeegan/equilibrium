# Option (a): two-component BCC rule, exact Bloch symbols (10-01 evening)

`symbol.py` (numpy, no lattice runs), plot `weyl_dispersion.png`. Rule: D'Ariano-Perinotti Weyl walk (PRA 90 062106,
2014; transition matrices from D'Ariano, Mosco, Perinotti, Tosini, Phil. Trans. A 375 20160394 / arXiv:1705.08552):
two complex quantities per cell, psi(x, t+1) = sum_h A_h psi(x + h, t) over the 8 hex-diagonal neighbours only,
A_h = zeta* or zeta times 2x2 0/+-1 patterns, zeta = (1 +- i)/4 (two inequivalent walks = mirror images).
Dirac rule: [[n W, i m I], [i m I, n W^dag]], n^2 + m^2 = 1 (mass = local coupling of the two Weyl halves).

| check | result |
|---|---|
| unitarity, all k, both zeta | exact (9e-16); W = d I - i sigma.a, d, a real, d^2 + \|a\|^2 = 1 |
| speed | c = 1/sqrt3 hex hop per tick, isotropic at leading order |
| dispersion error dc/c | [100] 0; [110] -0.014 (kh)^2; [111] **+-0.11 kh (first order, odd in k)**; [123] +-0.066 kh |
| parity | omega(k) != omega(-k) along body diagonals (s_x s_y s_z term in d): the two zeta walks are mirror images |
| Dirac gap | omega(0) = arcsin m exactly; omega vs sqrt(m^2 + c^2 k^2) within 0.04-0.4% at kh 0.3 |
| Dirac block choice | only (W, W^dag) is unitary; (W+, W-) and (W+, W-^dag) are not (7e-3 at kh 0.1); (W, W^dag) keeps the [111] k / -k asymmetry (0.6% at kh 0.1) |
| lattice symmetries | only 4 of 24 cube rotations (identity + three 180 deg axis turns, D2, transitive on h1..h4); no 90 / 120 deg symmetry |
| spin-1/2 sign | lifts of the x, y, z half-turns ANTIcommute (\|{U,U'}\| 3e-20): genuinely projective = quaternion group, the phase-independent form of the 2 pi sign; U_x^2 = -I |

Reading:
- The two-component rule has, at the level of the rule: motion at c by construction (each tick a hop along a hex
  diagonal), mass as the coupling of two components (gap = arcsin m, relativistic dispersion to <0.5%), and the
  spinor sign as a lattice fact (anticommuting lifts), not an imported vector space. Complex amplitudes give a compact
  phase, so phase windings can be integer (step()'s phi is non-compact).
- Costs vs step(): first-order, parity-odd phase-speed error along body diagonals (+-0.11 kh; step(): -0.018 /
  -0.031 (kh)^2, second order, even), and only D2 isotropy. A chiral rule built from one handedness carries a lattice
  handedness at first order; the mirror walk has the opposite one. Whether a pair of opposite-handed walks cancels it
  (with a coupling that keeps unitarity) is open -- the obvious (W+, W-) Dirac block is not unitary.
- Philip 10-01: unitarity / reversibility is not required (lossy allowed) -- relevant if the cancelling pair needs it.

Next (proposal): real-space runner on our BCC lattice (8 hex neighbours, 2x2 complex per hop; numpy first), then
(1) packet test vs the symbols, (2) a phase-winding (vortex) seed: does an integer winding of the compact phase travel
and persist, (3) couple m (or the hop weights) to lam -- the nonlinearity enters as mass, not as speed only.

## Real form + lattice runs (10-01 evening)
Philip: no complex numbers -- the rule is real: `real_form.py` (each amplitude = a pair of reals, i = quarter turn of
the pair, zeta = 45 deg pair turn x 1/(2 sqrt2); 8 real 4x4 hops; mass = m x quarter turn coupling the two halves).
Matches the complex rule exactly (20 steps, 0.0), conserves sum psi^2. `walk.py` = DiracWalk on DampedLattice cells
(hex neighbours = the walk's h, edge contributes nothing, quadratic sponge).

Packets (`packet.py`, `packet.log`, R40, width 5, 40 ticks): centroid follows the symbol's group velocity to ~10%
(rest is leakage into the other branch), norm to 1e-3, massive k = 0 packet stays put. Body-diagonal handedness is
large at kh 0.6: v_g 0.64 (+[111]) vs 0.48 (-[111]) (c = 0.577); m 0.2: 0.56 vs 0.41. First order -> kh <~ 0.1 needed.
Alternating the two mirror walks each tick removes the +-k asymmetry but breaks leading-order isotropy ([110] -29%,
[111] -18% at k -> 0): rejected.

Vortex ring (`vortex.py`, R40, torus r0 12, w 8/3; pair angle winds q times around the tube centre line, amplitude
(d/w)^|q| Gaussian, seeded on the k = 0 positive-frequency branch; 600 ticks ~ 4 crossings, 51 s each):

| run | tube t 150 | struct t 150 | struct t 600 | winding on d = w loops |
|---|---|---|---|---|
| m 0.5 q +1 | 0.030 | 0.146 | 0.009 | 1 -> 0 by t 150 |
| m 0.5 q -1 | 0.030 | 0.146 | 0.009 | identical (mirror) |
| m 0.5 q 2 | 0.010 | 0.051 | 0.001 | 2 -> 0 |
| m 0.2 q 1 | 0.006 | 0.026 | 0.002 | 1 -> 0 |
| m 0 q 1 | 0.012 | 0.017 | 0 | 1 -> noise |

Linear rule: every ring disperses (heavier = slower, m 0.5 keeps 15% in the structure at t 150); the winding is lost
because the amplitude leaves the loops (a winding is only defined where the pair has a size), not by a phase slip
inside a persisting core. Nothing holds, as for any linear rule.

Candidate nonlinearity in the same language: a local pair turn proportional to the local density (each tick every
pair rotates by g rho; norm-preserving, local). With a uniform nonzero background (pairs everywhere, the defocusing
sign) this is the Gross-Pitaevskii setting, where vortex rings are known persistent, self-propelled structures: a
ring travels at its own induced speed and persists BECAUSE it travels -- and the ring is a hole (deficit) in the
background, cf. moat / glut. Untested here.

## Vortex ring with a density turn in a uniform background (`gp.py`, 10-01 evening) -- FIRST PERSISTENT TRAVELLING STRUCTURE
Rule: real-form Dirac step + every pair turns by -g rho each tick (local, norm-preserving). Background a0 v (k = 0
positive branch), rho0 = 1, m = 0.5; ghost neighbours = background, sponge relaxes to the background (uniform state
stationary to 3e-15). Seed: ring r0 12 at z0 = +-14, phase atan2(z - z0, rho - r0) - atan2(z - z0, rho + r0), core
amplitude d / sqrt(d^2 + xi^2). GP limit: xi = sqrt(c^2 / 2 m g rho0), v_ring = (c^2/m)/(2 r0) (ln(8 r0/xi) - 0.615).
R40 (348k cells), 400 ticks, 50 s per run. Plots `gp_*_t*.png` (xz density + pair angle), records `gp_*_rec.npz`.

| run | xi | result |
|---|---|---|
| g 0.05, q +1 | 2.6 | ring persists 400 ticks: radius 12 +- 0.6, winding 1 on all 8 loops, deficit settles ~8500 (seed 8900), travels -z from 14 to -16.8: 0.076 /tick (GP 0.083, -9%) |
| g 0.05, q -1 | 2.6 | exact mirror: travels +z, 14.0 -> 17.0 from -14, winding -1 |
| g 0 (linear, same seed) | -- | core spreads (radius 12 -> 19, deficit x9), background density in the domain drops to 0.53, ring moves 0.016 |
| g 0.1, q +1 | 1.8 | ring breaks by t ~ 200 (core 1.8 cells: under-resolved; handedness error at kh ~ 0.5) |

(The tracker latched on a transient central dip at t ~ 100 -- the seed emits sound while relaxing; the plots show the
ring at z ~ +7, radius 12 at t 100 with intact cores.)

Reading: with the local density turn, a winding of the pair angle around a ring is a persistent, self-propelled
structure in this rule: it keeps its radius, winding and deficit, and translates at its own induced speed (GP value
within 9%, the sense set by the winding sign). Without the turn it spreads. It is a DEFICIT in a uniform background
(cf. moat / glut, low-lam-not-void), and it persists while travelling. CONDITIONAL / screening: R40, 400 ticks
(~1.5 crossings), core 2.6 cells (kh ~ 0.4: first-order handedness error a few %), m and g picked, not derived.
Next: longer runs (bigger or periodic domain), xi convergence (2.6 -> 4 -> 6 cells at fixed xi/r0), ring-ring
interaction (leapfrogging / annihilation of q +-1), what it radiates when perturbed.

## Core-width convergence (`gp.py --s`, fixed r0/xi = 4.65, GP time scaling; logs gp_conv_s*.log)

| s | xi (cells) | cells | ticks | speed / GP formula | speed in GP units (xi / tau) | winding | radius |
|---|---|---|---|---|---|---|---|
| 1 | 2.58 | 348k | 400 | 0.944 | 0.3047 | 1 throughout | 12 +- 0.9 |
| 1.5 | 3.87 | 1.18M | 900 | 0.935 | 0.3017 | 1 throughout | 18 +- 1.3 |
| 2 | 5.16 | 2.78M | 1600 | 0.941 | 0.3037 | 1 throughout | 24 +- 1.5 |

Converged: the ring speed in its own units is resolution independent to 1% from a 2.6-cell core up; relaxed deficit
scales as s^3 (8.5k, 28.7k, 66k), radius breathes +-6% (seed relaxation). The 6% below the thin-core formula is a
property of r0/xi = 4.65 (formula is asymptotic), not resolution. Caveat: the ring travels along z, an axis, where
the walk's first-order handedness error vanishes; a ring travelling along [111] would expose it (untested).
s = 2 took 35 min in numpy (1.3 s/tick at 2.8M cells).
