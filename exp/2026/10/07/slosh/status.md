# 10-07 slosh: period / origin of the deficit slosh (next 10-07 (A) follow-up), GP clean seed

Machine: fedora box (no /bk, no 10-06 states) -> snapshots ~/dst/2026/10/07/slosh; clean seed = core/gp_vortex.py
(exact GP vortex of the walk's small-k equation, tail blended to 1 between d 20-40, --project_plus); no saved state needed.
spring_box.py (10-07 copy): + --tail, in-run slosh probe (deficit per dt shell every 20 ticks, own Voronoi part; walk
energy within dt 38 every 100 ticks, H2 symbols cached; checked == shed/quasi_energy.py local split, diff 0.0).

## The energy bookkeeping (refmid_to_raw.py, refmid_to_raw_10-06.log)
e = e_q - om0 rho + (g/2) rho (rho - 1) (quasi_energy.py reference) gives a UNIFORM floor of deficit delta an energy
-(g/2) delta per cell (first order): deficit moving between the structure and the floor moves E_in by -(g/2) dD_in.
Grand-potential form et = e + (g/2)(1 - rho) is second order for any uniform floor (sum conserved: D_tot is).
10-06 shell logs (all --ref_mid) recomputed raw, within dt 38:
  P (gap)    D raw mean +- sd   E raw mean +- sd   corr(D,E)   Et = E + g/2 D (t >= 1k)
  180 (41)   62.4k +- 3.4k      288 +- 16          -0.99       544.7 +- 2.7 (542-549)
  200 (53)   63.6k +- 2.5k      279 +- 12          -0.98       540.8 +- 2.9
  220 (63)   62.7k +- 2.4k      282 +- 11          -0.97       539.5 +- 2.6 (537-541)
  340 (133)  55.5k +- 5.0k      301 +- 20          -0.98       529.1 +- 4.5 (526-529 after 1k; 541 at t 0)
CORRECTION of 10-06 branch/status.md: (i) the raw D slosh is +-4-9 %, not +-20-40 % (that table mixed references);
(ii) corr(D, E) = -1 is the (g/2) floor bookkeeping, not a reference artefact; (iii) the bound quantity is Et, steady
to 0.2-0.5 % in time; E_in's 'steady 4-7 %' was the slosh leaking in through -(g/2) dD. (iv) Et falls slightly with the
spacing (545 / 541 / 540 / 529, gap 41 -> 133): a candidate neighbour energy of ~+3 % at gap 41, but at P 180 the own
Voronoi cut clips the dt < 38 region -> preliminary.

## GP seed, P 220 (box_gp_N2_a10_r30_P220.log, slosh_P220.png)
Et_in 535 (t 0) -> 529.4 +- 1.5 over 1k-8k (GP seed ~10 below the 10-06 relaxed-profile seed, same spacing); E_tot
conserved 227.80-227.83. The GP core is slightly too wide for the coil: ~21k of deficit leaves dt < 30 by 1k and spreads
as a uniform floor (E_in 207 -> 287, Et unchanged).
D_in oscillates +-7k (sd 5.4k of 69k), period 1971 (zero-padded spectrum, t >= 1k) vs the box sound mode
side / (sqrt2 c_s) = 1981 (n = (1,1,0)), c_s = sqrt(g n / (3 m)) = 0.0453 (walk Bogoliubov, unfitted): 0.5 %.
Second peak ~5100 (beat / window). -> the slosh is a standing sound wave of the periodic lattice of copies, excited by
the seed's deficit mismatch -- if P 180 / 340 follow side / c_s (prediction 110: 1622 / 3062; breathing would be fixed).
Tube geometry (slosh rows, rc = co-moving centre-circle radius): rc breathes 30.7-32.5, period ~5350 (sd 0.42) = the
D_in second peak (~5100) and the 10-06 '~5k' breathing (N 8 r0 80 box) -> two components in D_in: ~1970 (box-mode
candidate, dominant) + ~5.3k (coil breathing, corr(rc, D_in) +0.53). The P scan separates them.

## GP seed, P 180 (box_gp_N2_a10_r30_P180.log, slosh_P180_P220.png)
Et_in 533.9 +- 2.7 (wobbles with D: own-Voronoi cut clips dt < 38 at gap 41). D_in sd 4.4k of 74k; peaks 1755 (110 mode
1621, 100 2293) and 3744; rc breathing peak 4320 (P 220: 5350). Main peak moved 1971 -> 1755 (-11 %) vs side -18 %:
within the 8k-window resolution (+-0.07 / kT) of both the box mode and a weak P-dependence -> not decisive; the slow
'breathing' peak moves with P too (box-coupled?). P 340 discriminates (110: 3062 vs fixed ~1970).

## GP seed, P 340 + the three together (box_gp_N2_a10_r30_P340.log, slosh_P180_P220_P340.png)
  P    side  first D_in min  t_min / side  late D_in peak   rc breathing   Et_in (t >= 1k)
  180  104   ~900            8.7           1755             4320           533.9 +- 2.7
  220  127   ~1100           8.7           1971             5350           529.4 +- 1.5
  340  196   ~1650           8.4           ~2080 (weak)     5105           518.8 +- 1.9 (530 -> 518 over 1-3k, then flat)
Reading (revises the P 220 '1971 = (1,1,0) mode' match, a coincidence of that box): (i) the large first swing scales
with the spacing (t_min ~ 8.6 side, half the (1,1,0) period = 7.8 side): the seed's excess deficit leaves as a wave and
comes back when it meets the neighbours' -> the big slosh is the seed mismatch ringing between the copies (initial-
condition set, neighbour-mediated). (ii) The residual ~1800-2100 oscillation and the rc breathing (~5.1-5.3k; 4.3k at
gap 41) do not follow the box -> coil-local. (iii) Et_in drops at P 340 while the emitted wave content is outside dt 38
and has not returned -> the Et spacing trend (both seeds: P 180 > 220 > 340 by ~15) is at least partly where the
radiated seed mismatch sits, NOT yet a neighbour interaction energy. To separate: longer runs (16k+) or a seed with no
deficit mismatch (relax, then measure); amplitude of the ringing scales with the seed mismatch (GP core too wide: ~21k).

## GP-seed r0 / a scan (next 10-07 item 3; scan_quant.sh, quant_table.py -> quant_table_10-07.log; N 2 P 220 4k,
## Et_in = E_in + g/2 D_in within dt < a + 28, t >= 1k)
  r0  a   Et mean +- sd   Et / L(helix)
  20  10  329.9 +- 8.4    1.827   (still falling -21 over 1-4k: inner gap r0 - a ~2.6 xi, squeezed -- as 10-06)
  25  10  431.2 +- 1.9    2.123
  30   6  473.4 +- 1.2    2.329
  30   8  498.6 +- 0.6    2.327
  30  10  529.4 +- 1.5    2.323   (8k)
  30  12  565.5 +- 1.9    2.320
  30  14  607.7 +- 2.9    2.324
  35  10  627.1 +- 1.6    2.466
  40  10  720.9 +- 2.4    2.559   (lateral gap to image ~25: region clipped)
All keep 2 / 1. Reading: no dip / step in r0 or a with the clean seed either -> N 2 is a continuous family (10-06's
reading stands with the seed artefact removed). At fixed r0 the bound energy is strand length x a constant (Et / L
2.320-2.329, a 6-14, 0.4 %); with r0 Et / L grows ~logarithmically (d(Et / L) / d ln r0 ~1.1 / 0.9 / 0.7 for 25-30 /
30-35 / 35-40, last clipped) -- the shape of a loop's flow energy, the outer scale set by the loop (r0), not the tube
(a). GP benchmark only: pi rho0 / M = pi n / (3 m) = 0.785 per ln -- same order. Quantization, if any, is not in r0 / a
at fixed N: next place to look is N at fixed geometry (and the N floor ~8 xi pitch).

## CsCl-type pair (next 10-07 0c(ii); pair_box.py, scan_pair.sh, pair_compare.py; GP seed N 2 r0 30 a 10, P 300 8k,
## partner at the body centre (lateral axis distance s / sqrt2 = 122, z offset s / 2 = 87))
q -1 (opposite sense; pair_q-1_N2_a10_r30_P300.log): relative z speed -0.0263 / tick, steady per 1k block (-0.0247 ..
-0.0279), straight-line residual sd 0.65 cells = free counter-propagation (2 x the lone coil's ~0.013); one pass (closest
approach in z near t ~3.3k at lateral 122): at most a ~7 % speed-up in the 2-3k block, inside the block scatter -> no
measurable interaction at this separation. Both coils intact (rc = rcp = 31.3 +- 0.4, windings 2 / 1); tube aspect cycle
(1.0 -> 2.2 -> 1.15 over ~4.7k) the same as the single P 220 coil. Et_in 521.7 +- 3.6 (own region cut by partner tubes
too -> compare to the q +1 control only).
q +1 control (pair_q1_N2_a10_r30_P300.log): STOPPED at t ~5500 (background task's 2 h limit killed the queue wrapper and
its child); slosh series to 5000 dumped from the saved npz. Common window 1k-5k (pair_compare.py --t_max 5000):
  run          own drift (zc + shift, 1k-5k)   relative drift        Et_in           D_in
  lone P 220   0.0131                           --                    529.4 +- 1.5    69.4k
  lone P 340   0.0130                           --                    518.8 +- 1.9    54.5k
  q +1 pair    0.0136 (both, by symmetry)       0.0000 (sd 0.02)      529.9 +- 3.0    62.2k
  q -1 pair    0.0133 / partner -0.0136         -0.0269 (sd 0.38)     523.7 +- 2.7    67.4k
Reading (preliminary): same sense co-moves exactly (BCC symmetry); opposite sense counter-propagates and passes at
lateral 122 with no measurable deflection. Both pairs run ~3-4 % faster than a lone coil, independent of the partner's
sense and not ordered by coil density (P 220 is denser: 0.0131) -> a sign-blind neighbour effect at body-centre
spacing? Two-point drift estimates -> needs fitted drift + a lone coil at P 300 / same density. Et_in q -1 lower by
1.2 % (~2 sd) but confounded: the partner tubes sweep through q -1's own region, q +1's region is static.

## Quant in N at fixed geometry (next 10-07 item 3, second half; scan_N.sh, GP seed r0 30 a 10 P 220 4k, Et within 38)
  N   Et mean +- sd (1-4k)   drift 1-4k   windings at 4k   inner-side turn spacing 2 pi (r0 - a) / N
  1   489.2 +- 1.0           -2           -1/0 (offset-circle probes straddle the single turn)   126
  2   529.4 +- 1.5 (8k)      +2           -2               63
  3   569.5 +- 5.2           -13 falling  -3               42 (10.7 xi)
  4   611.6 +- 11.9          -33 falling  -4               31 (8.0 xi = the 10-04 pitch floor)
Reading so far: N 1 / 2 steady; N 3 / 4 keep losing Et with their turns still intact at 4k (~+40 per turn at the
means, but 3 / 4 are not settled). Where they end = the quantization question at this geometry -> resumed 4k -> 10k
(scan_N_resume.sh; quant_gp_N*_from4000.log).
Resumed to 10k (quant_gp_N{3,4}_a10_r30_P220_from4000.log; Et per 1k block 0 -> 9k):
  N 3: 581 576 568 565 568 564 562 568 566 557  -> nearly settled after ~3k (~-1 / 1k), 3 turns kept
  N 4: 638 625 612 598 600 587 591 573 576 560  -> steady leak ~-8 / 1k to 10k, 4 turns kept (one offset probe circle
       reads 0 from ~5k, minrho 0.00-0.01: density zeros reach the probe circles); at 10k Et ~ N 3's level
Reading: at r0 30 a 10, N 1-2 steady, N 3 nearly, N 4 (inner turn spacing 8 xi = the 10-04 pitch floor) radiates
continuously without dropping a turn (yet). The loss is a gradual leak setting in at the spacing floor, not (so far)
a step in N. Open: does N 4 drop to 3 (or 2) given time (20k+), and does the floor move with r0 (N 4 at r0 40)?

### 10-07 evening (/bk machine): N 4 to 20k at r0 30 and r0 40 (scan_N4_long.sh, tag _gpL, fresh from t 0; n4_long.py -> n4_long_10-07.log)
  r0 30 (inner spacing 8 xi): Et 632 -> 547 over 20k; leak -6.7 / 1k (2-10k) slowing to -1.4 / 1k (10-20k). No turn
       dropped: centre-circle winding -4 throughout. Instead the helix contracts onto its centre circle: core-set rad
       10.4 -> 8.0 (2k) -> 5.8 (8k) -> 4.7 (18k, ~ one core width, rho 0.5 at 5.1); rc 32 -> 29; the offset probe
       circles (2, 4 off centre) read 0 / -2 from ~8k = they now lie outside / across the thinned tube. Endpoint
       heading toward a 4-times-twisted ring (tight-winding picture) rather than N 3. The Et loss is the strand
       shortening as a shrinks, still slowly going at 20k.
  r0 40 (inner spacing 11 xi): Et 816 -> ~797-801, -1.8 / 1k (2-10k), -0.8 / 1k (10-20k) ~ inside the 2k-bin sd
       (3-13); rad 10.4 -> ~8.5, rc 41 steady, winding -4 throughout. Nearly settled.
Reading: the N 4 leak is set by turn spacing, not by N (r0 40 holds the same 4 turns). At the floor the coil does
not quantize down in N; it reels its helix in toward the axis circle, keeping the twist. Open: does r0 30 stop at a
finite rad (twisted ring) or unwind once rad ~ core width; Et / L is not constant along the contraction
(L from the seed formula with a = rad gives 2.0 -> 2.5), so "Et = strand length x const" holds only across seeds.
  r0 30 resumed 20k -> 40k (scan_N4_resume40k.sh, quant_gp_N4_a10_r30_P220_from20000.log; n4_long.py now reads both
       segments): the contraction STOPS. rad 4.5 (20k) -> 3.8-4.0 (24-30k) -> 4.0-4.5 (30-40k), Et 2k-bin means 543 ->
       536-541, slope -0.7 / 1k (20-30k), +1.0 / 1k (30-40k), bin sd falling 9 -> 4-6. Winding -4 on the centre circle
       throughout; the k4 (4-fold) modulation stays at 2.2-3.2 (it was 9.8 at seed) -> still a 4-turn helix, minor
       amplitude ~ one core width, i.e. a tightly wound 4-turn twisted ring, not a turn lost. Settled Et ~539, below
       N 3 at 10k (~557-566) and close to N 2's 529 at this geometry: at the spacing floor the extra turns cost little
       once wound tight. Open: is the tight state also reachable / preferred at r0 40 (seed a small there), i.e. one N 4
       family with a wide and a tight member, or does r0 40's wide coil only hold because its turns are far enough apart?

### 10-07 evening: relaxed-profile seed vs GP seed (next item 4b; scan_relaxed_seed.sh, 2 threads; seed_compare.py -> seed_compare_10-07.log)
N 2 r0 30 a 10 P 220 8k, same flags as scan_gp.sh, seed = 10-06 branch/ profile_pp_N2_t8000_tail1.npz (caveat: d is
to the core SET, spring_box.py --profile reads it as d to the strand centre -> core seeded ~0.6-0.95 too thin).
  seed  D_in(0)  D_in mean >1k  0-2k swing  sd >2k  Et_in >1k      first D_in minimum
  GP    79.7k    69.4k          21.2k       4.9k    529.4 +- 1.5   t 1160
  rp    61.2k    61.9k           7.3k       2.7k    535.0 +- 1.5   t 1080
-> the relaxed seed starts with ~no excess deficit (61.2k vs settled 61.9k; GP ~10k excess) and its ringing is ~1/3
(0-2k swing) and ~1/2 (later sd) of GP's: ringing scales with the seed mismatch. A residual ~7k swing remains (shape
mismatch / the core-set offset). First minimum at the same time (~1.1k), as a box-transit should. Breathing: rp minor
radius 9.7 (1k) / 10.2 (2.5k) / 9.4 (4.5k) / 10.3 (6.7k) -> ~5.3-5.5k, same as GP's ~5.1-5.3k: coil-local, confirmed.
The ~2k residual: 8k windows give FFT bins only at 7020 / k, cannot separate 2.0k from 2.3k -- not tested here.
Settled D_in differs (61.9k vs 69.4k) and Et_in +5.6 (1 %): the two seeds did not reach the same state by 8k (or
are two members of the family); the relaxed seed is NOT mismatch-free in Et either (Et(0) 543 -> 535).

### 10-07 evening: pair speed-up, fitted drifts (next item 5; drift_fit.py -> drift_fit_10-07.log)
Least squares on true z (centre z + frame shift, 500-tick log lines), slope +- standard error:
  run              1k-5k            1k-end           T
  lone GP P 180    0.01314 (7)      0.01320 (5)      8k
  lone GP P 220    0.01312 (7)      0.01312 (3)      8k
  lone GP P 340    0.01307 (7)      0.01308 (4)      8k
  lone rp P 220    0.01311 (5)      0.01309 (2)      8k
  pair q +1        0.01361 (7)      0.01354 (7)      5.5k (stopped)
  pair q -1        0.01340 (6)      0.01315 (4)      8k
-> a lone coil runs 0.0131 regardless of box (P 180-340) and seed (GP / relaxed): a lone P 300 run is redundant.
q -1 over 1k-8k = lone (0.01315); its 1k-5k excess is a window effect. 2k sliding windows: lone P 220 swings
0.0128-0.0134 (the ~5k breathing modulates the speed, +-2.5 %), q -1 0.0129-0.0136 (the 2-4k peak brackets the pass at
~3.3k), q +1 0.0140 -> 0.0133 (1-3k .. 3-5k). Only q +1's excess survives so far, and only over a window shorter than
two breathing periods -> rerun to 8k (scan_pair_q1_8k.sh, tag _gp8, pair_q1_N2_a10_r30_P300_T8k.log).
q +1 rerun to 8k (pair_q1_N2_a10_r30_P300_T8k.log): identical to the fedora run through 5.5k (deterministic), fitted
1k-8k 0.01359 +- 0.00004 vs lone 0.01312 +- 0.00003 -> +3.6 %, ~10 se. 2k windows, q +1 vs lone P 220:
  q +1  .01396 .01358 .01328 .01342 .01368 .01394
  lone  .01344 .01314 .01278 .01308 .01318 .01328   (diff +.0005 +.0004 +.0005 +.0003 +.0005 +.0007)
  q -1  .01348 .01360 .01312 .01294 .01292 .01290   (diff +.0000 +.0005 +.0003 -.0001 -.0003 -.0004)
-> the same breathing modulation, shifted up by a steady ~+0.0005 for the co-moving same-sense partner: a real,
sustained speed-up. Opposite sense: speeds up while approaching / passing (~3.3k), slows after, net ~0 over 8k.
CORRECTS the 10-07 reading ("sense-blind"): the effect is sense-dependent. Not a plain distance effect: the lone
P 180 coil has coaxial images at 104 (< the pair's body-centre distance 150) and lateral images at 104, and runs at
lone speed. Open: is it the body-centre geometry (partner offset s/2 in z, lateral 122) with matched sense, does it
scale with spacing (P 240 / 360 pair q +1), and the q -1 approach / recede asymmetry = an interaction on passing?

### 10-07 evening: q -1 through the pass (pass_q-1.py -> pass_q-1_10-07.log)
True z every 20 ticks (slosh zc unwrapped over the recentre shifts; partner = zc + dz_rel), 600-tick sliding fits, minus
the lone GP P 220 coil at the same t (same seed -> same breathing phase; baseline good to ~+-0.0003: the P 220 vs P 300
seed ringing differs, worst before 1k). Primary and partner mirror to 1e-5 (symmetry check). Excess vs minimum-image z
separation of the nearest partner copy (lateral axis distance 122 fixed), t >= 1k:
  |z-sep|   n    excess / tick     sd
  0-25      9    +0.00059 (+4.5 %)  0.00019
  25-50     10   -0.00017          0.00024
  50-75     11   -0.00032          0.00032
  75-90     4    -0.00021          0.00022
-> a speed-up only within ~25 in z of the partner, SYMMETRIC in approach (2.3-3.1k: +0.0003..+0.0009) and recession
(3.3-4.1k: +0.0004..+0.0007): not attraction-like (that would be fast in, slow out), no deflection. Far from the partner
q -1 runs at or slightly below lone. Contrast q +1: partner fixed at z-sep side / 2 = 87 (the FAR bin), yet +0.00047
sustained. So at the same far geometry same sense = faster, opposite sense = lone/slower; near, opposite sense =
faster. Preliminary (one box, baseline from another box). Spacing scan running: scan_pair_q1_P.sh (P 240 / 360).

### 10-07 evening: q +1 pair spacing scan (scan_pair_q1_P.sh; drift_fit.py)
Body-centre partner distance d = P / 2 (lateral axis distance d sqrt(2/3), z offset d / sqrt3); excess over the lone
0.01313 (P 180-340 mean), fitted 1k-8k:
  P     d     v (+- se)          excess     %
  240   120   0.01430 (4)        +0.00117   +8.9
  300   150   0.01359 (4)        +0.00046   +3.5
  360   180   0.01330 (4)        +0.00017   +1.3
2k windows: all three carry the lone coil's breathing modulation, shifted up by the excess (P 240: .01472 .01432
.01384 .01410 .01454 .01462). Fall-off: exponential with length 31 fits the 3 points to 1-2 % in log (factor 2.5-2.7
per 30); a power law needs ~d^-4.7 with 4-9 % residuals and inconsistent local exponents (4.2, 5.5). 31 cells ~ 8 xi
-- not the GP density healing scale (xi / sqrt2 = 2.8); origin open. Three points only, one geometry: suggestive of a
short-range (screened / evanescent) coupling, not a 1 / r field. Same sense only (q -1 far: no excess).
Open: sense dependence at the same d (q -1 run spends little time at fixed d -> a static opposite-sense configuration
would need the partner NOT counter-propagating, impossible with matched N / chirality?); which far field carries it
(density deficit tail vs phase / flow), e.g. the P 240 deficit shells between the coils.
