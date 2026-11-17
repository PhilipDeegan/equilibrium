# 2026-10-18 zoo -- many candidate structures, one rule, one scorecard (Philip: "we are likely going to have to test
# many different potential structures"; "a lump isn't a structure, it has no reason to remain bounded")

zoo.py: structure = closed vortex curves with integer circulation per curve (whole turns, solid-angle seed as 10-16
geom), smoothed (tau) + div-free. Rule: step() + Cartan transport, kappa 1 (C++ alat.step_cartan, midpoint 2).
Scorecard: |Gamma|-weighted centroid displacement (travel), rms2 = |Gamma|^2-weighted distance to the initial curves
shifted rigidly by that displacement (shape kept), leak = |Gamma| weight beyond rcut (3 R / 9), sum|Gamma|, E.
Catalogue: ring, pair_co / pair_counter (coaxial, +-0.6 R), concentric (R + counter-rotating R / 2), hopf_link (two
linked rings), trefoil (2,3 torus knot), coil_1_3 / coil_1_5 (single loop winding 3 / 5 times round a torus tube
0.35 R), coil_3_2. Add a curve set to STRUCTS to add a structure; the rule is one line (alat.step_cartan) to swap.
Screen: R 9, P 72, T 20, 1 thread each, 4 at a time (~4 min / run). Promising -> R 20 (P 140).

## R 9 screen (r9/, P 72, T 20): resolution-dominated, ranks by smallest feature, not physics
| struct | rms2 growth | leak t 0 -> 20 | sum|G| | |v| |
| ring | +49 % | 0.11 -> 0.33 | +34 % | 0.125 |
| pair_co | +61 % | 0.11 -> 0.38 | +33 % | 0.187 |
| trefoil | +72 % | 0.11 -> 0.41 | +42 % | 0.150 |
| hopf_link | +82 % | 0.11 -> 0.43 | +45 % | 0.128 |
| pair_counter | +117 % | 0.11 -> 0.53 | +50 % | 0.003 |
| concentric | +126 % | 0.08 -> 0.46 | +68 % | 0.099 |
| coil_1_3 / 1_5 / 3_2 | +131 / +163 / +137 % | -> 0.55 / 0.59 / 0.51 | +39 / +53 / +41 % | 0.09 / 0.06 / 0.29 |
E exact (|dE/E| <= 1.1e-6). Coil tubes 0.35 R = 3 cells: worst resolved. 10-16 geom: R 9 leak = discretisation residual
(drops ~30x by R 20). rms2 uses a rigid shift only: precessing / rotating structures (knots, links) are overstated.
R 20 run (r20/, P 140, T 30, 8 structures, 4 x 1 thread).

## R 20 round 1 (r20/, P 140, T 30; R 9 was T 20)
| struct | rms2 growth R 9 -> R 20 | sum|G| growth R 9 -> R 20 | leak t 0 -> end (R 20) | |v| R 20 | dE/E |
| ring | +49 % -> +2.4 % | +34 % -> +0.35 % | 0.107 -> 0.112 | 0.077 | -1.7e-8 |
| pair_co | +61 % -> +4.3 % | +33 % -> +0.83 % | 0.106 -> 0.130 | 0.118 | -2.2e-8 |
| trefoil | +72 % -> +5.4 % | +42 % -> +1.09 % | 0.108 -> 0.128 | 0.096 | -2.0e-8 |
| hopf_link | +82 % -> +7.6 % | +45 % -> +0.98 % | 0.105 -> 0.123 | 0.089 | -2.2e-8 |
Every residual shrinks 10-100x with x2.2 resolution (longer T at R 20): discretisation, as 10-16 geom. Knots / links
travel and hold as well as rings once resolved (trefoil, Hopf link ~1 % sum|G| over T 30); nothing selects. Ranking
still follows feature sharpness. Round 2 (coil_1_3, coil_3_2, pair_counter, concentric) killed at t 11 / 9 / 5 / 0
(Philip leaving); checkpoints in /bk/dst/2026/10/18/zoo/*_ckpt.npz, partial logs r20/*_part1.txt.

## 10-19: R 20 round 2 + rms2r (best rigid motion: weighted ICP / Kabsch, rot = fitted angle; logs exp/2026/10/19/zoo/r20/)
| struct | rms2 (shift only) | rms2r (shift + rotation) | rot at t 30 | sum|G| | |v| |
| coil_1_3 | +21.6 % | +2.8 % | 7.3 deg (~0.24 deg / t, steady) | +0.80 % | 0.064 |
| coil_3_2 | +36.8 % | +11.8 % | 32.8 deg (~1.0 deg / t, 1.2 before t 9) | +1.15 % | 0.168 |
| concentric | +10.6 % | +10.2 % | 0.1 | +1.32 % | 0.075 |
| pair_counter | +17.6 % | +17.8 % | 0.0 | +3.36 % | 0.000 |
Coils precess rigidly as they travel (classical vortex-knot behaviour); with rotation removed coil_1_3 holds about as
well as ring / pair_co. Counter-rotating combinations (pair_counter, concentric) degrade most: they deform without
rotating (interaction, not discretisation? rms2 shrinks only 6-12x R 9 -> 20 vs ~20x for ring -- check at R 40 if it matters).
End states in /bk/dst/2026/10/19/zoo/end_*_R20.npz. Gotcha: *_ckpt.npz is left after a finished run -> a rerun with the
same --seed resumes from the last checkpoint, not t 0 (delete the ckpt for a fresh run).
Round 1 (ring, pair_co, trefoil, hopf_link) rerunning for rms2r.

## 10-19 round 1 rerun with rms2r (R 20, T 30)
| struct | rms2 | rms2r | rot t 30 | sum|G| | |v| |
| ring | +2.42 % | +1.72 % | 0.0 | +0.35 % | 0.077 |
| pair_co | +4.26 % | +4.20 % | 0.0 | +0.83 % | 0.118 |
| trefoil | +5.36 % | +5.10 % | 0.7 | +1.09 % | 0.096 |
| hopf_link | +7.63 % | +5.48 % | 2.6 | +0.98 % | 0.089 |
Full R 20 ranking by rms2r: ring 1.7 < coil_1_3 2.8 < pair_co 4.2 < trefoil 5.1 < hopf_link 5.5 < concentric 10.2 <
coil_3_2 11.8 < pair_counter 17.8 %. trefoil (2 toroidal / 3 poloidal, tube 0.45 R) and coil_3_2 (3 / 2, tube 0.35 R) are
the same knot: 0.7 vs 33 deg rotation -> precession set by embedding, not knot type. Philip 10-19: sim cells are coarse
cells (neighbour contrast 15-26 % vs physical ~1e-8); no spinor reading of the coils (no vectors -> no spinors).
