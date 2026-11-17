# 10-11 tick: can the exchange-matrix commutator (11/lie) act on the tick phase?

grad_part.py -> grad_part_10-11.log (P 120, 10-08 tick ring R0 15, phi = W wrap(dtheta)).
In a pure-phase model phi is always a wrapped gradient: theta can only take the commutator's GRADIENT part; its
divergence-free part (0.41 of the term here) has nowhere to go.
The gradient part IS tick.py's clock shift q (squared offsets = Bernoulli |v|^2 / 2): corr(div lie, div grad q) -0.98 to
-0.99 beyond 3 cells from the core (scale -42..-46), -0.93 / -0.97 inside 1.5 / 3 cells (scale -23 / -36).
=> kappa_c 0.132 on theta == tick.py kappa_q ~ +-5.7: the 10-08 kappa 5 run (moves + shrinks, energy x3). The
commutator is energy-exact only WITH its transverse part; dropping it (forced by the pure phase) is what pumps energy.
Reading: the pure tick phase cannot host the commutator conservatively. Needs a DOF that can hold the transverse part
near the defect: (a) soft / depleted core (10-09 item 1), (b) Villain integers per face (defect hops = integer flips,
discrete -> sharp), (c) phi = W wrap(dtheta) + a smooth real remainder carried by step() + commutator (conversion rule
between remainder circulation and defects needed). Decision for Philip.
