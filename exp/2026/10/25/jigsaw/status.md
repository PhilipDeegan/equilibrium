# Jigsaw: periods per transit (2026-10-25)

Signposts: signposts.md (r_p = 4 hbar / (m_p c) to 0.04 %: a transit at c around the proton perimeter = 4 own periods).

## ring.py v1 -- ring mode at c with throughput-driven depletion (run1.sh, r1_q*_k*.txt)

Discrete-weight BCC box n 64 (../defect/bcc.py), u = A G cos(q psi) travelling (phi from the quadrature field), R 10,
tube a 4 (wavelength 2 pi R / q ~ 1.6 R: a q 4 body is compact, not a thin guide). Depletion d per cell relaxes over one
period toward min(dmax 0.5, kappa F), F = local |face energy throughput|; medium g = (1 - d) / (1 + d) on rate and hop;
preseeded in equilibrium with the seed (event already forced it). Sponge r > n/2 - 8. T 300 ~ 10 transits.

| q | kappa 0 | kappa 0.1 | kappa 0.5 (core saturated, speed x1/3) |
|---|---|---|---|
| in-tube at t 30 / 60 | -- | 0.20 / 0.03 (q 3), 0.17 / 0.03 (q 4), 0.17 / 0.04 (q 5) | 0.33 / 0.12 (q 3), 0.41 / 0.09 (q 4), 0.59 / 0.18 (q 5) |
| end (t 300) | dispersed | dispersed, d -> 0 | dispersed, d -> 0 |

NULL: no q self-traps. Depletion delays dispersal by ~1 transit (more for larger q: whispering-gallery ordering), then
content leaks, the depression follows it down (d tracks content) and everything leaves. The ~8 % left at the end is
frozen non-propagating face content (step() keeps div-free phi; non-uniform g makes the seed partly non-gradient).
Same outcome as 10-01 (instantaneous local self-lam: collapse or dispersal, no window). Consistent with Philip: the
bound state is FORCED (pinch from outside: CME / reconnection), not self-held by local depletion feedback.
In a fixed (externally held) depression the ordering would be whispering-gallery (higher q always better held) -> no
preference for 4 either; selection must come from a fit condition, not from holding alone.

## v2 -- lopsided seeds in a held depression (run2.sh, r2_q*_k*.txt)

Philip: pressure mediation is what makes circles and produces "symmetry" -> symmetry must be an outcome. Test: held
toroidal depression (well 0.4 on the seed tube = forced-pinch stand-in) +- throughput feedback (kappa 0.5), seeds
q 3 / 4 / 4.5 / 5, power of the tube content per lobe count m over time.

| seed | well only (kappa 0): E at t 60 / 300, lobe power | + feedback: E at t 60 / 300, lobe power |
|---|---|---|
| 3 | 0.09 / 0.01, stays m 3 | 0.60 / 0.08, stays m 3 (1.00 until it leaks out, t ~150) |
| 4 | 0.05 / 0.01, stays m 4 | 0.73 / 0.14, stays m 4 (1.00 until t ~200) |
| 4.5 | 0.22 / 0.07, spreads to high m | 0.83 / 0.51, resolves to m 5 (0.33 / 0.53 at t 10 -> 0.96 at t 200) |
| 5 | 0.09 / 0.01, stays m 5 | 0.86 / 0.62, stays m 5 (1.00 throughout) |

- The held depression alone holds nothing (gone by t ~60); feedback holds longer, higher m longest (whispering-gallery
  ordering: 5 > 4 > 3).
- Integer lobe counts are conserved exactly: 3 stays 3, 5 stays 5 -- no conversion toward an even count.
- The mismatched 4.5 resolves to the better-held integer neighbour, 5 (odd), not 4.
- -> no pressure mediation in step() + |throughput| depletion: lobes do not push each other toward balance; the only
  selection is "which integer holds longer", and that favours higher m, not symmetry. The missing relation is a
  coupling between lobes by sign (the deviation from the mean), which |throughput| (sign-blind, u^2-like) cannot carry.

## v3 -- signed depletion (run3.sh, r3_q*_s*.txt)

As v2 + feedback (well 0.4, kappa 0.5) plus a signed term relaxing (tsig 0.5) to ksig * u (contract where u > 0, expand
where u < 0; total d clipped to [-0.3, 0.5]); dt 0.05.

| seed | ksig 0.1: E at t 300, lobe power | ksig 0.3: E at t 300, lobe power |
|---|---|---|
| 3 | 0.08, stays 3 until it leaks (as v2) | 0.15, stays 3 (0.92 -> 0.67 by t 150), a little 6 (<= 0.07) |
| 4 | 0.16, stays 4 | 0.33, 4 then 8 appears (2q) |
| 4.5 | 0.52, -> 5 (0.92) | 0.54, -> 5 (0.75) with 4 / 6 admixture (<= 0.28 / 0.24) |
| 5 | 0.64, stays 5 | 0.55, stays 5 (0.98 -> 0.62) |

- Still no move to a neighbouring even count: 3 never becomes 2 / 4, 5 never 4 / 6, 4.5 still resolves to 5.
- The signed term only adds the 2q harmonic (and m 0), the plain quadratic-mixing signature; it raises retention a little
  for 3 / 4 (0.15 / 0.33 vs 0.08 / 0.16).
- Reading: with a LOCAL medium response (u^2-like or signed), the lobe count is a conserved label; a lobe never feels the
  one across the centre. Balance across the centre needs a relation that reaches across the structure -- a non-local
  depletion (Philip: depletion is macroscopic, the potential), carried by something propagating, not solved.

## v4 -- deformation / translation under an external pressure gradient (run4.sh, r4_q*_g*.txt, r4_q*_g0.005_rev.txt)

Philip: travelling flux on a circle is balanced for any count (parity only matters for standing patterns); structures
must support deformation under external pressure, translation is a product. Setup: v2 hold (well 0.4) but the well
FOLLOWS the content centroid (co-moving pinch stand-in, an assumption), feedback kappa 0.5, background depletion
gradient 0.005 / cell along x (contracted toward +x) until t 150; lop = |m 1| / m 0 of tube energy about the well centre.

| q | no gradient: E t 150 / 300 | gradient: E t 60 / 150, centre (x, y) at t 100, lop max |
|---|---|---|
| 3 | 0.09 / 0.07 | 0.54 / 0.07, (+10.0, +18.7), 0.18 |
| 4 | 0.24 / 0.14 | 0.65 / 0.07, (+5.1, +20.8), 0.20 |
| 5 | 0.73 / 0.62 | 0.75 / 0.08, (+4.9, +20.3), 0.16 |
| 4, 5 reversed sense | | same x, y mirrored: (+4.9, -18.8) / (+5.4, -17.5) at t 90 |

- Translation appears as a product of deformation: every count becomes lopsided (0.15-0.20) and moves.
- Mostly SIDEWAYS: ~20 cells perpendicular to the gradient vs ~5 along it by t 100; the sideways direction flips exactly
  with the circulation sense (gradient-drift / Magnus-like). The along-gradient part is toward the contracted side
  (gravity-like sign) and independent of sense.
- Deformation is NOT supported: under the gradient every body breaks up while moving (E ~0.07 by t 150 even for q 5,
  which keeps 0.73 without the gradient). No count deforms-and-recovers better than the others.
- Caveat: the co-moving well follows the centroid (lag) -- part of the break-up may be the hold, not the body.
- Philip: deformability as a criterion presupposes that it is true and that the model already supports it -> it is a
  requirement on the rule, not a test that can select a count. v4 says the current rule + stand-in hold does not meet it.

## v5 -- propagating depletion carrier, no external well (ring2.py, run5.sh, r5_*.txt)

Rule-requirement test (no count selection). Depletion carried by a field P on cells: P_tt = cD^2 lap P + F (F = local
|face energy throughput|), cD = c (bounded), steady part ~1/r outside a steady source; contraction d = kappa P
(clip 0.9). Self-consistent preseed (4 iterations: seed travelling at the CORE speed, face content from the actual
medium, P relaxed against that throughput, kappa set so max d = depth) -- earlier ring runs seeded at vacuum speed,
right at the guiding cutoff (bad seed). Frozen-well check (n 48): depth 0.5 does not trap a q 4 ring, depth 0.8 does.

| run | E/E0 t 100 / 200 / 300 / 400 | notes |
|---|---|---|
| A q 4, depth 0.8 | 0.94 / 0.77 / 0.49 / 0.33 | steady leak, body stays put |
| D q 5, depth 0.8 | 0.92 / 0.67 / 0.34 / 0.03 | leaks, then runaway (P 1.0 -> 0.05 at t 350-400) |
| F q 3, depth 0.8 | 0.92 / 0.79 / 0.61 / 0.39 | leaks slowest |
| G q 4, depth 0.6 | gone by t 80 | below the trapping depth |
| B / C q 4 + gradient 0.005 on t 50-200, sense + / - | as A | lop <= 0.05 during the push; centre +0.42 along x (toward contracted), sideways +0.07 / -0.07 (mirrors with sense) by t 200; stays displaced after toff |
| E q 5 + gradient | as D | jitters +-0.3, no clean drift |

- First self-held structure: a deep (saturated, core speed ~0.1 c) depression carried by a propagating field holds the
  content for many hundreds of time units without any external hold. Needs depth >= ~0.7: below, it disperses at once.
- Not permanent: a steady leak (~0.2 % per time unit) and, for q 5, runaway once the depression weakens.
- Deformation / translation: the deep body is STIFF -- it deforms little (lop <= 0.05) and moves ~0.4 cells along the
  gradient (toward contracted cells) plus a small sideways drift that flips with the circulation sense; the
  displacement persists after the push (translation, not elastic return). Same signs as v4, ~50x smaller.
- Depth 0.8 is a picked strength (kappa normalised); the threshold (0.6-0.8) is the rule's, the choice is ours.

## v6 -- where the v5 leak comes from (run6.sh, r6_*.txt; q 4, depth 0.8, n 64)

| P source | E/E0 t 100 / 200 / 300 / 400 | depletion at the ring t 10 / 200 / 400 | content at t 400 |
|---|---|---|---|
| frozen (P fixed after preseed) | 0.96 / 0.93 / 0.90 / 0.89 | fixed | in the tube, slow decreasing shed |
| |throughput| (v5) | 0.94 / 0.77 / 0.49 / 0.33 | 0.84 / 0.50 / 0.21 | 74 % moved INSIDE the ring |
| local energy density | 0.96 / 0.95 / 0.95 / 0.94 (flat after t 300) | 0.84 / 0.84 / 0.93 | 97.5 % in the tube |

- The v5 leak is the throughput source starving itself: throughput = speed x content, and the contracted core is slow,
  so the deeper the hold, the weaker its own source; the ring's depression thins (0.84 -> 0.21), the 1/r overlap makes
  the centre deepest, and the content drains inward and out.
- Sourced by local energy density the same carrier holds and settles (shed ~5 % once, then flat to t 400, depression
  at the ring deepening to 0.93): a permanent self-hold within the rule. Frozen well sits between (slow shed).
- Tension with Philip 10-25 ("density is circumstantial; flux / circulation is forced"): here flux-driven depletion
  undermines itself because a contracted region carries less flux. Open for Philip.

## v7 -- flux counted per LOCAL tick, and an energy + flux mix (run7.sh, r7_*.txt)

Philip: "probably a mix of both".

| P source | E/E0 t 100 / 200 / 300 / 400 | kappa P at the ring t 100 / 400 (clip 0.9) | tube share t 400 |
|---|---|---|---|
| |throughput| / local speed factor ("local") | 0.97 / 0.96 / 0.96 / 0.95 | 1.38 / 1.14 (saturated) | 0.95 |
| 1/2 energy + 1/2 throughput ("mix") | 0.96 / 0.89 / 0.81 / 0.74 | 0.80 / 0.52 | 0.94 |

- Flux counted in the cell's own rate (per local tick) holds as well as the energy source: saturated depression at the
  ring, ~5 % shed then a slow tail. The flux picture survives once flux is measured locally -- background-tick flux
  is what starved itself.
- The plain mix still leaks (its background-tick half thins the ring depression 0.80 -> 0.52), slower than pure flux.

## v8 -- requirements + counts on the local-tick-flux hold (run8.sh, r8_*.txt; depth 0.8, --src local)

Counts (no push): E/E0 at t 400 / tube share / lobe power on the seed count
| seed | E t 400 | tube | lobes |
|---|---|---|---|
| 3 | 0.89 | 0.95 | 3: 1.00 throughout |
| 4 | 0.95 | 0.95 | 4: 1.00 |
| 5 | 0.94 | 0.99 | 5: 1.00 |
| 4.5 | 0.94 | 0.69 | never settles: power wanders over 3-6 (t 100: 0.02 / 0.22 / 0.36 / 0.15; t 300: 0.35 / 0.20 / 0.08 / 0.04) |
- Under a working hold every whole count holds (3 slightly leakier, 4 and 5 alike); the mismatched seed neither
  settles nor leaves: it spreads around the tube and keeps reshuffling between neighbouring counts. No preference for 4.

Push (q 4, gradient on t 50-250, step switch):
- grad 0.005: deformation small (lop <= 0.034); during the push the body moves slightly AWAY from the contracted side
  (x -0.25), after switch-off it moves toward it (x +1.8 at t 300) and sideways +-2.3 by t 400 (mirrors with sense,
  growing after the push ends). The step switch is itself an impulse -> needs a ramped push to separate the two.
- grad 0.02: INVALID -- numerical blow-up (expanded cells at the box edge reach speed x3, dt 0.1 over the stability
  limit); rerun at dt 0.05.

## v9 -- ramped push (run9.sh, r9_*.txt; local-flux hold, q 4, dt 0.05, cosine ramp 50, push on t 50-300)

| grad | x (toward contracted) at t 150 / 200 / 250 / 300 (off) / 350 / 400 | sideways at 400 (+ / - sense) | lop max -> after | E t 400 |
|---|---|---|---|---|
| 0.005 | 0.0 / -0.6 / +0.3 / +2.3 / +2.2 / +3.4 | +3.5 / -3.5 | 0.094 -> 0.02-0.08 | 0.93 |
| 0.01 | +0.3 / +1.5 / +3.2 / +4.8 / +5.3 / +6.3 | +2.7 / -2.7 | 0.106 -> 0.016 | 0.92 |

- All three requirements met qualitatively on one body: it HOLDS (92-93 %), DEFORMS a little under the push and
  RECOVERS its shape after (lop 0.1 -> 0.02), and TRANSLATES as a product -- accelerating toward the contracted side
  (gravity-like sign), roughly linear in the push (x at t 300: 2.3 vs 4.8 for 1x / 2x), and it KEEPS MOVING after the
  push ends (+1.6 over t 300-400 at 0.01: inertia-like), plus a sideways drift that mirrors with circulation sense.
- 0.005 is noisier (early wobble to -0.6 before moving), so linearity is rough; one size, one depth.

## v10 -- mass scaling / universality of fall (run10.sh, r10_*.txt; gradient 0.01 ramped, on t 50-300, dt 0.05)

| body (n 80, depth 0.8) | content ~ R a^2 | x toward contracted at t 200 / 250 / 300 / 400 | sideways t 400 | E t 400 |
|---|---|---|---|---|
| R 8, a 3.2, q 4 | 1 | +1.7 / +2.1 / +3.1 / +4.6 | +3.3 | 0.92 |
| R 10, a 4, q 4 | 2.0 | -0.6 / +1.0 / +2.5 / +4.4 | +2.1 | 0.93 |
| R 13, a 5.2, q 4 | 4.3 | -0.4 / +3.0 / +4.2 / +3.5 | +2.1 | 0.93 |
| R 10, q 3 | ~2 | -0.5 / +0.6 / +4.3 / +6.1 | +2.7 | 0.93 |
| R 10, q 5 | ~2 | +0.0 / +2.0 / +2.8 / +4.0 | +2.5 | 0.94 |
| (v9: R 10 q 4 at n 64) | 2.0 | +1.5 / +3.2 / +4.8 / +6.3 | +2.7 | 0.92 |

- Over a ~4x range of body content (R 8 -> 13) and three counts, displacement at the end of the push is +2.5 to +4.3
  with no trend in size or count; a force-on-a-mass response would differ ~4x. The same R 10 q 4 body in a smaller
  box (n 64) gives +4.8: the box / sponge effect (~2x) is as large as the whole spread, and every trajectory wobbles
  (early moves away from the contracted side, then toward). Reading: CONSISTENT WITH UNIVERSAL FALL, not a force on a
  mass, at the ~+-50 % level this setup resolves. Not a precision test.
- A 1 / 1.15 at fixed coupling (--kfix 1.319e-2, n 64): INVALID -- without depth normalisation the self-consistent P
  ran away (kappa P ~8-11), the clip saturated the whole region (d 0.9 even at r 20), so the gradient was clipped away
  (x = 0 exactly) and the body later collapsed. Fixed-coupling runs need a source / clip rethink.
