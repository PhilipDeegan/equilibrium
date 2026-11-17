# 10-08 couple: what carries the same-sense pair coupling (next 10-08 item 1)

Saved finals: /bk/dst/2026/10/07/slosh/{pairq+1_*_P240/300/360_*_gp8, prof_pp_*_P220_*_rp}/state_t08000.npz.

## Candidates in the field (tails.py -> tails_10-08.log)
- - branch |psi_-|^2: 1e-10 .. 3e-9 everywhere (seeds projected +, stays there) -> not a carrier.
- deficit floor: rho_mid (equidistance surface, slosh logs) fluctuates 0.96-1.00 in lone AND pair boxes (lone P 180
  0.959-0.987, no speed-up) -> floor depth does not order the drift.
- phase gradient / flow: the carrier, below.

## Flow (farflow.py -> farflow_10-08.log)
Signed u = grad arg a (a = psi . v*). Far from every tube (dt > 40) u is uniform and FORWARD (+z = the coils' direction):
lone P 220 +0.0073, pair P 240 / 300 / 360 +0.0107 / +0.0046 / +0.0022; box means 0.0085 / 0.0129 / 0.0067 / 0.0038
~ 2 pi q pi rc^2 / V per coil (87-90 %). = zero phase winding on lines outside the rings (the seed's zero mean jump),
2 pi on lines through each hole -> mean smooth gradient U0 = 2 pi pi r^2 / V per coil, NOT zero-mean.

## Prediction: superflow advection, no fit (biot_exact.py -> biot_exact_10-08.log, biot_exact_pair_10-08.log)
Vortex lines = the seed helix (N 2, r0 31, a 10), Gaussian core 3.9, periodic flow as an exact Fourier sum on the strand
+ U0; coil advected at v = n / (3 m) <u_z> (= the walk's small-k group velocity, omega = arcsin m + n k^2 / (6 m)).
Partner sublattice (body centre) at the primary strand: k != 0 part NEGATIVE (dipole return flow), U0 positive, sum:
  P     d     predicted dv    pair v measured   measured - predicted total (own + partner)
  240   120   +0.00097        0.01430           +0.00222
  300   150   +0.00041        0.01359           +0.00218
  360   180   +0.00021        0.01330           +0.00216
Differences between spacings agree to ~6 %: P 240 - P 360 predicted 0.00094 / measured 0.00100, P 300 - P 360 0.00027 /
0.00029. The constant ~0.0022 (17 % of the self speed, 0.0110 predicted vs 0.0131) = the self term's core model.
Sensitivity: r0 32 a 8 -> partner terms +13 %. The 10-07 "exponential, length 31" is not a screening length: it is the
near-cancellation of two ~1/s^3 terms; local exponent 3.1-3.7 over d 150-240, tending to d^-3. P 420: +0.00013
(exp(-d/31) would give +0.00006) -- not decisive at se ~0.00004.
q -1 at P 300 vs z-separation (biot_exact_qm1_pass_10-08.log; partner sense flips the sign):
  z-sep   0 / 9 / 17 / 26 / 35 / 52 / 69 / 87  -> +0.00062 / +58 / +47 / +33 / +16 / -15 / -35 / -41 (x 1e-5)
  measured bins 0-25 +0.00059, 25-50 -0.00017, 50-75 -0.00032, 75-90 -0.00021 (pass_q-1_10-07.log): matches the
  near speed-up (opposite sense beside = its outer return flow is forward) and the far slow-down, symmetric in approach.
Lone box (own images + U0) predicted relative to P 340: P 180 +0.00004, P 220 +0.00020; measured +0.00006..12 /
+0.00004 (se ~0.00004, breathing window systematics ~1e-4). P 220 off by ~0.00015: OPEN.

## Reading
The same-sense speed-up and the q -1 pass are both plain superflow: each coil sits in the others' (and its images')
phase-gradient field and is carried at the walk's group velocity. Geometric: the body-centre partner is near the dipole
field's magic angle (54.7 deg), so the k != 0 and uniform parts nearly cancel. No new field, no screening length;
the box (zero outer winding -> U0 ~ 1/V) is part of the answer. Envelope-level (GP advection) reading -- the walk
symbol enters only through n / (3m); not checked at walk level beyond that.
Decisive test: same-sense partner BESIDE (partner 0.5,0.5,0,1 at P 300): superflow predicts a SLOW-DOWN -0.00062
(-4.7 %); a distance-only coupling predicts a speed-up.

## Decisive test: same-sense partner BESIDE (scan_side.sh, pair_box.py --partner 0.5,0.5,0,1, P 300 8k, tag _side)
Fitted drift (drift_fit.py -> drift_side_10-08.log): 0.01249 +- 0.00005 (1k-end), 0.01250 (1k-5k); dz_rel stays 0.0
(co-moving), both coils intact (windings -2 / -1 at 8k). vs lone 0.01313: -0.00064 (-4.9 %); predicted -0.00062.
Same sense, same P, body centre -> +0.00046; beside -> -0.00064: the sign follows the flow geometry, not the distance
(beside is the NEARER placement, lateral 122 vs 3D 150). Superflow advection CONFIRMED (envelope level, no fit).
