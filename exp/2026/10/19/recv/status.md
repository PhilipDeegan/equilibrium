# 2026-10-19 recv (Philip: any emitter is also a receiver; raw vacuum is not a receiver -- photons would decay into the
# background). Receiver = another body of the same kind. A periodic box makes a lone body's images its receivers.

r9.sh, zoo.py (step() + Cartan, kappa 1), R 9, T 30, 1 thread each. New: --struct side / side_a / side_b (ring A R at
x -SEP/2, ring B R2 at +SEP/2, SEP = side / 2 -> A is equidistant from B and B's image) + per-x-half E / Eu / sum|G|.

## Lone ring vs box (images at distance = box side 41.6 / 55.4 / 69.3 / 83.1 for P 72 / 96 / 120 / 144): 100 Eu / E
| t | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 | 24 | 26 | 28 | 30 |
| P 72 | 0 | .897 | 1.136 | 1.190 | .964 | 1.000 | .971 | .993 | .758 | .742 | .719 | .949 | .864 | .762 | .805 | .966 |
| P 96 | 0 | .896 | 1.125 | 1.177 | .899 | .899 | .926 | .925 | .889 | .935 | .897 | .819 | .750 | .688 | .868 | .964 |
| P 120 | 0 | .896 | 1.120 | 1.170 | .892 | .896 | .880 | .877 | .898 | .863 | .860 | .891 | .899 | .848 | .865 | .778 |
| P 144 | 0 | .895 | 1.118 | 1.166 | .889 | .894 | .877 | .869 | .861 | .871 | .884 | .824 | .851 | .862 | .877 | .843 |
- The first rise and fall (0 -> 1.17 % at t 6 -> 0.89 % at t 8) is the same in every box: local. It is the ring
  dressing itself with a u cloud once, not a cycle; after it Eu sits at ~0.87 % in the largest box.
- Each smaller box leaves the P 144 curve at a time that grows with the box (P 72 ~t 8, P 96 ~t 12-18, P 120 ~t 20):
  the later "oscillation" (and likely the 10-19 R 20 one: P 140 at R 20 is box / R 4.0, P 72 here is 4.6) is the images
  answering. -> In a periodic box the lone ring is never alone; the late Eu swings are exchange with its copies.
- The R 9 rings degrade fast in shape (rms2r +72 %), so this is a screen.

## Two bodies, A (R 9) + B (R 5.4) side by side, P 96, vs A alone and B alone in the same box
Per half: d = pair - A alone - B alone. Sum dE_A + dE_B = -60.34 at every t (the A-B cross energy, conserved since
all three runs are); what moves is its split between the halves:
| t | 0 | 6 | 12 | 18 | 24 | 30 |
| dE_A | -30.17 | -30.53 | -29.74 | -28.45 | -26.35 | -24.66 |
| dE_B | -30.17 | -29.81 | -30.60 | -31.88 | -33.99 | -35.68 |
| dEu_A / dEu_B | 0 / 0 | -0.6 / +0.6 | -2.9 / -3.1 | +0.6 / +0.6 | +0.4 / 0.0 | -2.3 / -3.0 |
- From t ~ 12 the shared energy moves from B's half to A's, accelerating: 5.5 by t 30 (0.2 % of A, 0.5 % of B).
  E_A(pair) 2737 -> 2743, E_B 1152 -> 1146.
- Eu differences fluctuate (+-3) with no trend: no extra wave-sector energy from having a partner.
- Ambiguous: B is faster (v 0.156 vs 0.111) and is 1.35 cells ahead in z by t 30, so the overlap geometry changes;
  the half split cannot tell energy carried from B to A from the cross term changing shape. Needs R 20 (held shapes)
  and a cleaner observable (equal-speed bodies, or the cross term followed against the z offset).

## R 20 (r20/, r20.sh): equal rings R 20 side by side, P 192 (SEP = side / 2 = 55.4 = 48 lattice periods, rim gap
## 15.4), T 30, 3 + 3 + 2 threads; B alone = exact lattice translate of A alone (solo_a); ends /bk/dst/2026/10/19/recv
(zoo.py diagnostics now one periodic kd query + threaded queries: byte-identical output, ~2.1 h -> ~40 min per run.)
Per half: pair - solo A (own half) - solo A (other half, = B's halo by translation). E exact to 1e-8 in all.
| t | 0 | 3 | 6 | 9 | 12 | 15 | 18 | 21 | 24 | 27 | 30 |
| same: Eu pair | 0 | 16.6 | 21.5 | 27.0 | 28.2 | 23.9 | 17.1 | 17.7 | 19.0 | 21.2 | 22.1 |
| 2 x solo Eu | 0 | 17.4 | 24.5 | 29.5 | 31.1 | 29.4 | 23.1 | 23.5 | 25.0 | 25.3 | 23.5 |
| opp: Eu pair | 0 | 18.4 | 29.1 | 39.4 | 41.3 | 34.8 | 24.2 | 22.5 | 25.2 | 28.8 | 30.3 |
- Interaction (cross) energy: same sense -349.57 per half (-5.5 % of E), opposite +346.0; same: constant to 1e-6 in
  each half (exactly symmetric, no redistribution); opp: A +0.33, B -0.33 by t 30 (opp is a mirror pair, not a
  translate: tiny).
- Wave-sector (in-flight) energy depends on the partner: same sense holds LESS than two lone rings (up to -6, -20 %,
  t 15-24), opposite sense MORE (up to +10, +33 % at t 9-12; +7 at t 30). Bounded, no trend to offload.
- Same-sense pair moves slower (v_z 0.064 vs 0.075 alone: each ring sits in the other's return flow); shape held
  (rms2r +2.2 % vs +1.65 %). sum|G| per half: same -2.4 .. -1.9, opp -0.5 -> +3.8 vs solo.
Reading: no net transfer between equal bodies (by symmetry and in fact); what the partner changes is how much energy
the pair keeps in flight in the space around / between them -- aligned less, opposed more, like the field energy of
like / unlike dipoles. Fits Philip's "two bodies + the space between = one charge distribution" (in-flight energy at a
snapshot set by the configuration). The R 9 asymmetric-pair "B -> A" shift has no counterpart at equal speed ->
consistent with geometry (overtaking), not transfer. Caveat: periodic images are receivers here too (box / R 5.5).
