# 2026-11-04 pair regime map (queue item 1, from exp/2026/11/03/pair)

pair.py = 11-03 pair.py + --senses (per-ring sense: flips that ring's azimuth -> its circulation). Same body as 11-03:
R 8 a 3.2 q 4, depth 0.8 (depth-normalised kappa), --src local, dmax 0.9, dt 0.05, T 300, snaps t 100 / 200 / 300.
run.sh (8 single-thread jobs, launched 12:3x): saves /bk/dst/2026/11/04/pair/*.npz.
- pair24 / solo_m24 / solo_p24: x = +-24, n 128 (ring edges ~19 cells apart)
- pair30 / solo_m30 / solo_p30: x = +-30, n 144 (~31 apart)
- opp18 (senses +1, -1) / solo_p18n (+18, sense -1), n 112; solo_m18 = 11-03 solo_m (same n, sense +1)
Question: is there a separation where the pair settles (superposition holds past t 100, no approach / turn)? does
opposite sense bridge the same way?
Philip (11-04): there are no fields, only the substrate; fields are an abstraction; all particles (atoms too) are bound
states on it -> the 11-03 hold ("field-only merging, dendrites may need atoms") rested on a false split. Merging is a
question of what the bound states / rule do (scale, count, context), not of whether 'material' is present.

## Results (all done 17:23; 4 cores from ~13:00 -> ~2.3-3.3 h each)
compare.py PAIR SOLO_M SOLO_P CX. Centre = body0 (at -cx) displacement: toward other / sideways (body1 mirrored).
Solo drift <= 0.2 toward, +0.26..+0.36 sideways (same sign for both senses = box / lattice baseline).
Resid in rings ~-0.12..-0.19 at t 100 everywhere = the kappa normalisation confound (pair kappa -7 %).

run         t    middle resid d (|x|<=4)   u^2 middle pair/sum   toward   sideways (body0 / body1)
same +-18   300  +0.43 (saturated)         490x                  +2.85    +2.65 / -2.65 (pair turns)   [11-03]
same +-24   100  -0.01                     9.5x                  -0.16    +0.29 / -0.29
            200  +0.01                     0.67x                 +0.15    +0.29 / -0.29
            300  +0.05                     0.78x                 +0.50    +0.64 / -0.64
same +-30   100  -0.01                     2.9x                  -0.13    +0.21 / -0.21
            200  +0.02                     1.05x                 +0.34    +0.23 / -0.23
            300  +0.02                     52x                   +1.48    +0.67 / -0.67
opp  +-18   100  -0.09                     4.1x                  -0.35    +0.47 / +0.47
            200  +0.03                     6.8x                  +0.39    +0.45 / +0.45
            300  +0.15                     66x                   +2.68    +2.08 / +2.08 (co-drift, no turn)

Reading:
- No pair at +-24 / +-30 bridges within T 300: middle stays at the solo sum (resid <= +0.05, vs +0.43 at +-18).
  But neither is settled: both approach and the approach accelerates at t 200-300; +-30 approaches MORE than +-24
  (1.48 vs 0.50) and its middle u^2 jumps 52x at t 300 -> not monotonic in separation; either onset is delayed
  rather than absent (T 300 too short), or box size (n 144 vs 128) matters. No settled pair established.
- Opposite senses at +-18: approach about as much as same sense (2.68 vs 2.85) but the middle deepens 1/3 as much
  (+0.15 vs +0.43) and the pair does not turn: both drift sideways the SAME way (+2.08) = the pair translates.
  Same sense -> pair rotates about its midpoint; opposite sense -> pair translates. Sideways motion is handed
  (follows the senses), matching the 10-25 gradient-drift / Magnus-like idea.
- E/E0 0.93-0.95 at t 300 in all runs incl. solos (rim loss); d max falls t 200 -> 300 everywhere.
Next options: extend +-24 / +-30 to T 600 from saved state (does the approach become a bridge?); same separation
in n 144 for +-24 to rule out box size; opposite-sense sideways speed vs separation.
