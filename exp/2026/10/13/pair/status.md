# 10-13 pair: two coaxial rings under step() + cartan_transport (kappa 1), P 120 (side 69), R0 9, d 9, T 20, midpoint 2

pair.py. turn = whole-turn smoothed rings (tau 1, offset); zero = 11/lie net-zero tubes (wt 3, a0 1; link 0.065 each, not
exactly zero). Per-ring |Gamma|-weighted z / rho. E exact (1e-6) in all; linked exact.
  run          z / speed (t 0 -> 20)                         rho                    gap            sum|G| per ring
  turn single  0.04 -> 2.51 (0.124, decelerating)            9.03 -> 9.27           --             +34 % (leak, see cartan)
  turn s +1    rear -4.51 -> -1.34, front 4.52 -> 7.74 (0.16)  rear 8.50, front 9.81  9.03 -> 9.09   +19 / +47 %
  turn s -1    closing, 0.073 each                           9.04 -> 10.60 (both)   8.92 -> 5.99   +65 %
  zero single  0 -> 0.94 (0.047)                             10.37 -> 10.41         --             +6 %
  zero s +1    rear 0.045, front 0.053                       rear 10.56, front 10.39 10.53 -> 10.70 +3 / +7 %
  zero s -1    closing, 0.054 each                           10.23 -> 10.72         10.71 -> 8.55  +13 %
Reading: whole-turn pairs interact strongly -- same sign co-move as a pair ~30 % faster than a single ring with the
leapfrog shape change starting (rear shrinks, front grows), opposite sign close head-on and widen; net-zero pairs interact
~10-15 % (their small residual link). Everything here is classical vortex-ring dynamics (Euler, Kelvin): the rules have
no bonding beyond it -- no bound state, no threshold. Interaction strength tracks the exterior (linked) flow, consistent
with "imbalance is what couples" (Philip 10-13), but nothing beyond what Euler already gives.
Caveat: single whole-turn ring leaks after t ~4 (+34 % by t 20) -- pair numbers past t ~5 include that.
