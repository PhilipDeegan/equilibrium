# Static charge exterior in the discrete step() — 2026-09-29

`charge.py`, numpy step() (current weights), R = 12, log `charge.log`. phi = -W grad psi with
G^T G psi = s: div phi = +1 at the core cell, -1 spread over the outer shell (a closed lattice has
zero net divergence: boundary faces carry no phi, so the shell holds the image charge -> discrete
spherical capacitor). u = 0.

- Exterior is the discrete Coulomb field: psi = a(1/r - 1/b), rel rms residual 1.3e-3.
  Direction: psi r / fit = 1.024 [100], 1.004 [110], 1.005 [111] at r ~ 7.
- eps_eff = Q/(4 pi a) = 6.90 = c^2 / V_cell (5.333 / 0.770 = 6.93): derived, no fit constant.
- Core and shell u held at the background (fixed_u, u = 0): exactly static for 3 crossings
  (max |u|, |phi - phi0| ~ 5e-17). No pump: u never moves, nothing is injected.
- Not held: the core divergence leaves within ~1.5 time units (core div 1 -> -0.48 -> ... +-0.05),
  sloshing in r < 3 as u waves. A bare divergent core is not self-holding.

So: the discrete model carries a static Coulomb exterior in phi (radial, div-free outside, uniform
u); what it cannot do is a static u gradient. The old point-source tests pinned u = 1 (a pump) and
asked for Coulomb in u. What holds a charge's divergence is the open charge question; here the
held core stands in for it (pinning: scope = exterior only).
