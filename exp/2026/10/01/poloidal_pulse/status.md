# Single-signed poloidal pulse (10-01 evening) -- does net circulation travel in step()?

`run.py` (discrete step(), R56 955k cells, torus r0 12, w 8/3). Seed: smooth single-signed Gaussian ring blob
(width s 1.5) at (rho - r0, z) = (d0, 0), u > 0, A = (r0/rho) u / c psi_hat (local forward wave along psi): net
poloidal circulation C0 != 0, concentrated at psi = 0 (|C1|). Fixed tube depletion n = 1 + lam0 exp(-d^2/2w^2) has a
light ring (max of n d) at d 3.31 (lam0 4; band to the min of n d at 6.5) -> rays orbiting there are guided.
3 crossings (t 145), ~4 min per run. `pulse_rec.png` (plot_rec.py), `pulse_*_{ic,stepN,final}.png`.

| run | E_struct end | C0 end | \|C1\| end | C1 orbit angle | u b0 / \|b1\| end |
|---|---|---|---|---|---|
| linear (lam 0) | 0.24 | 0.857 | 0.415 | 0 (never moves) | 0 / 0 |
| tube lam0 4, d0 3.3 (on ring) | 0.30 | 0.914 | 0.633 | oscillates +-0.7 rad, decays to 0 | +-0.07 sloshing / 0.02 |
| tube lam0 4, d0 4.5 (off ring) | 0.27 | 0.900 | 0.643 | same | same |

Reading (rule-level, not resolution-limited):
- The circulation does NOT travel in step(). The phi concentration stays where it was seeded (final xz slice: blob
  at psi = 0, x ~ 15) in every run; its orbit angle never advances (rate ~0.003 vs c sigma / d0 ~ 0.25-0.7).
- What moves is u and the gradient part of phi: it leaves (linear) or sloshes back and forth in the waveguide
  (b0 oscillates in sign, ~11 time-unit period, decaying) -- standing, not orbiting.
- The retained energy (0.24-0.30) is the divergence-free part of the seed, frozen in shape and place: the whole
  divergence-free projection is frozen, not only its loop-uniform mean (my earlier statement was wrong).
  Refraction (lam0 4) retains slightly more and makes the sloshing last, but cannot set the circulation moving:
  in step() phi changes only through grad u, so circulation is never transported by anything linear.
- Consequence: under step() with any FIXED sigma, "travelling circulation" does not exist. A rule where phi carries
  itself needs a nonlinear coupling (state-dependent sigma: sigma grad u is no longer a pure gradient, so it can
  feed / move the divergence-free part) or a different rule. Self-sourced lam alpha 1 run queued below.

## Self-sourced lam, alpha 1 (C++ midpoint, 22 min; .so rebuilt first -- the 15:59 build lacked the selflam binding)
lam_max 0.39 at t = 0 (g = alpha amp^2 small for this seed's energy density), then 0.05 held by the frozen phi.
Same as linear: E_struct 0.25, C0 0.879, |C1| 0.433, orbit angle never advances (rate 0.0005). sigma in the tube
0.98 -> nearly uniform, so the sigma-weighted divergence that could drive u from a frozen loop is tiny. Weak coupling
changes nothing; strong coupling (alpha >= 4 on 10-01's travelling seeds) collapses to grid scale. A frozen
divergence-free phi with u = 0 is an exact steady state whenever sigma is uniform across its faces.
