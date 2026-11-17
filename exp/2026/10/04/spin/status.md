# 10-04 item B: why a lone component ring sheds its winding

spin_symbol.py (exact linearization of the mixed rest vacuum on the parity two-tick symbol, m 0.8 xi 3.9, g 0.00822):
stable (|lam| = 1 to 1e-15, z / x / [111]). Density branch = Bogoliubov sqrt(e (e + 2 g)); SPIN branch = the free walk
n k^2 / 6m exactly (g absent: the density turn sees only total rho, invariant under any local v1/v2 rotation) -> zero spin
sound speed. Mix 20 identical.

counterflow.py (P 96 box, exact walk, background = plane waves on two-tick doublet states, w1 orthogonal w2 -> exactly
uniform density; growth = |noisy - clean|, eps 1e-7, 4000 ticks; counterflow_mix45.log / _mix20.log, png):
  rest 0,0 and coflow +2,+2: no growth (controls).
  relative velocity 0.057 (n +1,-1 and one-sided +2,0): no growth in 4000 ticks.
  relative velocity 0.113 (+2,-2 and one-sided +4,0): grows, 3.3e-3 / 3.1e-3 per tick; 0.170: 3.4e-3; 0.283: 3.6e-3.
  -> instability depends on the RELATIVE flow only (one-sided = counterflow at equal relative velocity); coflow stable.
  mix 20: +2,-2 slower (2.1e-3, weak fit), +4,0 barely grows -> weaker away from 45 deg.
  Box limit: P 96 smallest mode k = 0.113; whether 0.057 is truly stable (threshold) or its unstable band lies below
  the box's smallest k is NOT resolved -> rerun in a bigger box (P 192) before claiming a threshold.
Reading (candidate mechanism, not proven): a component-only ring carries v1 flow with v2 at rest = relative flow
~ (n / 3m) / d = 0.25 / d around the core; relative flow is dynamically unstable (zero spin stiffness) -> nucleates
other-component structure (cf. B / D2 nucleation on the axis). A close v1 + v2 pair (E2) has nearly common flow
outside the pair -> little relative flow -> no nucleation. Growth time ~300 ticks vs ring generations ~3-4k: plausible
order, not yet linked quantitatively.
