# 2026-10-19 pair: two bodies + the space between (Philip: the space is part of the charge distribution, "charge" =
# average photon energy in flight at a snapshot; its holding has a floor; offload bounded by total energy vs Z0)

pair.py: linear hop / triangle DEC (gl.py, beta 0, Stormer-Verlet), per-hop eps = n^2 inside a body, floor eps 1
elsewhere, absorbing outer shell (6 cells, energy absorbed is counted). Body = dielectric sphere a 5, held by
impedance mismatch (Z = Z0 / n). Seed: localized loop field in A (17.5 % spills outside A at t 0, gap 1.85e-2 at t 0).
B placed at whole x-lattice periods (1.1547) so A and B discretise identically. Total (field + absorbed) to 8e-4
(staggered energy readout).

## A alone (alone/, P 100): offload rate set by the mismatch
| n | A at t 0 / 20 / 50 / 100 / 200 | 1/e time |
| 3 | 0.82 / 0.42 / 0.05 / 0.010 / 0.002 | ~25 |
| 4 | 0.82 / 0.67 / 0.28 / 0.060 / 0.011 | ~50 |
| 6 | 0.82 / 0.84 / 0.76 / 0.54 / 0.33 | ~220-275 |

## Pair (b1/, P 140, n 6 bodies, T 400): pair.png
| case | B at t 100 / 200 / 300 / 400 | B max | A at t 400 | absorbed |
| A alone | -- | -- | 0.209 | 0.735 |
| B n 6, sep 20.8 | 0.009 / 0.017 / 0.030 / 0.035 | 0.037 | 0.179 | 0.731 |
| B n 6, sep 34.6 | 0.002 / 0.007 / 0.014 / 0.020 | 0.020 | 0.206 | 0.718 |
| B n 5 (mismatched), sep 20.8 | 0.010 / 0.008 / 0.002 / 0.002 | 0.011 (t 155) | 0.181 | 0.768 |
- A matched receiver ACCUMULATES: B captures ~5.5 % of what A emitted at sep 20.8, ~3.2 % at 34.6, vs a geometric
  share (pi a^2 / 4 pi d^2) of 1.4 % / 0.5 %: a matched receiver takes 4-6x its geometric share.
- A mismatched receiver (n 5) takes in about as much early on but does not keep it (peak t 155, then 0.002).
- A's own decay is nearly unchanged by B (0.21 -> 0.18): with an absorbing outside, open space is the main receiver.
- Gap snapshot energy (in flight between): ~2-6e-3, the same in all three pair cases -- set by A's emission, not by B.
Reading: the receiver's match decides what it keeps, not how fast A emits. Here A emits into an absorbing "universe"
(the shell) whatever B is. A Philip-style bounded two-body system needs NO outer sink: next = closed box (no shell),
where energy can only sit in A, B or the space between, so the space's capacity and floor enter.

## Philip 10-19: "without a receiver there is no gradient"
A lone body must not offload at all. Linear-Maxwell dielectric bodies leak into vacuum whatever lies ahead, so they are
the wrong kind of body (and the absorbing shell was itself a receiver). Closed-box runs stopped (they would only show
the same receiver-blind leak). Right kind of body: step() + Cartan structures (zoo), which hold exactly in a uniform
floor. Next: the receiver shapes the floor at the emitter; offload must run along that gradient.
