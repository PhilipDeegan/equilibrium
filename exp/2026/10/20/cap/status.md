# 2026-10-20 cap: finite holding capacity, form 2 (Philip 10-19: offload because a body's capacity is finite)

Design (Philip chose per cell + stiffening): store per cell S_j = 1/4 sum_k phi_jk^2, energy H = sum u^2 / 2 + sum_j G(S_j),
G' = 1 + (S / S_c)^p. dH/dphi_f = phi_f (G'_j + G'_n) / 2 drives u in step() pass 1 (pass 2 unchanged). Cartan is a
Lie-Poisson bracket: omega stays the loop sum of a, the velocity slot takes b = dH/da = a (G'_j + G'_n) / 2 -> sum
dH/dphi . out = 0 per triangle; linear limit == old code. C++ alat.set_cell_capacity(S_c, p); test
TestCartanTransport.test_cell_capacity (off == plain, orthogonality 1e-13, u update == formula, H drift O(dt^2)).
zoo.py --scap (S_c as a fraction of the seed's max S_j) --cap_p; energy includes G; dt / sqrt(1 + (2p+1) / scap^p).

## R 9 screen (r9.sh; logs r9/; ends /bk/dst/2026/10/20/cap/end_s*.npz), lone ring, P 96, T 20, p 1
Linear baseline (vcap off, recv/r9/lone_P96): Eu 0.9-1.2 % of E, v 0.111, sum|G| +34 %.

| scap | Eu % of E t 1 / 2 / 10 / 20 | outer shell % t 0 -> 20 | v | sum|G| | rms2r | end max S / S_c | dE/E |
| 1.0  | 2.3 / 3.1 / 2.6 / 2.6     | 3.3 -> 7.4   | 0.123 | +58 %  | +61 %  | 0.70 | +2e-8 |
| 0.5  | 5.0 / 5.7 / 4.8 / --      | 2.9 -> 14    | 0.127 | +78 %  | +78 %  | 1.26 | +3e-7 |
| 0.25 | 10.0 / 10.3 / 7.9 / 9.8   | 2.3 -> 17    | 0.136 | +157 % | +134 % | 2.3  | +6e-7 |
| 0.1  | 19.7 / 18.7 / 17.8 / 17.3 | 1.4 -> 31    | 0.227 | +788 % | +220 % | 7.3  | -3e-6 |

Reading:
1. A finite store DOES shed: Eu jumps within t ~1 in proportion to the excess (2 / 5 / 10 / 20 %) and the energy runs
   outward (outer shell 3 -> 7..31 %). Then Eu stays flat: one shed, no trickle, no return seen in T 20 (P 96 box).
   Caveat: the seed was relaxed under the linear store, so part of this is the IC readjusting to the new H (the
   linear "dressing" was ~1 %); a seed relaxed under G is needed to separate shed from readjustment.
2. The ring does NOT settle below the ceiling: peak S only drops (0.5: 2 -> 1.26 S_c; 0.1: 10 -> 7.3), and total S
   GROWS (x1.02 .. x1.43) while sum|G| proliferates (x9 at 0.1): the saturated core is carried faster (b > a), shears,
   and winds up -- a tangle, not a held state.
3. p 1 is not a ceiling: G' = 1.7 already at S = 0.7 S_c (scap 1.0 changed everything with no cell above S_c). A
   ceiling reading needs larger p (flat below, steep above), at a dt cost 1 / sqrt(1 + (2p+1) (S_max / S_c)^p).

## Steeper store (r9p.sh; r9/p*_s*.txt; ends /bk/dst/2026/10/20/cap/end_p*_s*.npz), same setup
| p, scap | Eu % of E t 1 / 5 / 20 | outer shell % t 0 -> 20 | v | sum|G| | rms2r | end max S / S_c (cells > S_c end / seed) | dE/E |
| 2, 0.5 | 7.7 / 6.8 / 6.7    | 3.1 -> 14 | 0.109 | +82 % | +81 % | 1.20 (345 / 1241) | +8e-7 |
| 4, 0.7 | 3.8 / 3.8 / 3.7    | 3.6 -> 9  | 0.113 | +52 % | +56 % | 0.96 (0 / 534)    | +7e-7 |
| 4, 0.5 | 13.9 / 11.6 / 11.3 | 2.9 -> 22 | 0.097 | +92 % | +88 % | 1.17 (327 / 1241) | +3e-6 |

Shed vs the store's excess at t 0 (E0 - E_linear 2811.02): p 1 sheds 17-27 % of it, p 4 41 %, p 2 29 %.
Reading:
4. With a steep store the core IS held at the ceiling: p 4 scap 0.7 ends with no cell above S_c (seed 534), speed
   back at the linear value (0.113 vs 0.111), sum|G| +52 % ~ linear baseline (+34 % lone_P96, +38 % vcap off) -- the
   ring clips once, sheds into u (Eu jump by t 1, then flat), and carries on as a ring. Deeper below (scap 0.5) it
   clips to ~1.2 S_c and distorts more (+90 %). p 1 tangle (item 2) = the soft store acting everywhere.
5. The shed is one-shot and prompt (t < 1 = ~0.1 R / c-ish), sized by the excess, not a trickle -- closest yet to
   "offload because capacity is finite". Open: (a) seed not relaxed under G -> part is readjustment; (b) where the
   shed goes in a bigger box / to a partner (P 96 images); (c) refinement: S_c is per cell, so the same ceiling
   clips a different fraction at R 20 unless S_c scales with h (substrate floor vs structure ceiling).

## Rest-relaxed under the store (relax.sh; relax/*.txt; /bk/dst/2026/10/20/cap/{relaxed,end}_{A,B,C,D}.npz)
zoo.py --relax T --relax_tau tau: before the run, step() only (kappa 0, circulation untouched), u x (1 - dt / tau)
per tick, then u = 0 -> div F' -> 0 (rms 0.10 -> 3e-4 in t 15). p 4, scap 0.7. Relaxation took the core from 1.43 S_c
(534 cells above) to 0.99 S_c (none above), total S +1.5 % (spread outward), sum|G| unchanged.

| run | Eu % of E t 1 / 2 / 5 / 10 / 20 | v | sum|G| | rms2r | end max S / S_c |
| A lone P 96          | 0.66 / 1.28 / 1.35 / 1.10 / 1.11 | 0.114 | +52 % | +55 % | 0.96 |
| B lone P 144 (T 12)  | 0.67 / 1.28 / 1.34 / 1.11 / --   | 0.128 | (A to 1e-3 at every t to 12) | | |
| C equal pair, store  | 0.67 / 1.26 / 1.33 / 0.99 / 1.10 | 0.104 | +52 % | +205 % | 0.95 |
| D equal pair, linear | 0.40 / 0.88 / 1.15 / 0.76 / 0.92 | 0.113 | +34 % | +210 % | -- |
(linear lone baseline lone_P96: Eu 0.9-1.2 %. Pair halves identical to 1e-7 by symmetry; rms2r of the pairs is
dominated by the mutual drift, not shape.)

Reading:
6. The burst of items 1 / 4 was ENTIRELY readjustment of a linear-relaxed seed to the new store: relaxed under G, the
   ring sheds nothing beyond the linear dressing (Eu ~1.1-1.3 % vs ~1 %), box-independent (A == B), no trickle, no
   cycling. Over capacity -> one-time shed to the ceiling (the relaxation's damping was the receiver) -> an ordinary
   held ring AT the ceiling (max S 0.96 S_c through T 20).
7. So a finite store gives "excess leaves once, then held", not ongoing offload. Ongoing offload needs something to
   push the body above capacity again (absorption from a partner / incoming wave) -> emitter = receiver test: send
   energy at a ring held at the ceiling and see whether it re-sheds (and what fraction, how fast, in what form).
8. Equal pair (C) = no exchange by symmetry; needs unequal bodies or one pushed above capacity.

## Push: ring held at the ceiling hit by a plane u pulse (push.sh; push/*.txt; zoo.py --Sc --pulse --ediv --smax)
relaxed_A (S_c 0.38626, p 4), u slab AMP exp(-(x - x_r - 15)^2 / 18) -> +-x waves, hit at t ~5 and ~13 (recombine t 14).
Edf = store energy of the div-free (circulation) part of phi (projection each print).

| AMP | pulse Eu | core max S / S_c (peak) | Edf t 0 / 8 / 20 / 24 | v | sum|G| T 25 | Eu t 20 (wave only ~half of pulse) |
| 0.1        | 212  | 0.99 (never above)   | 3088 / 3011 / 2997 / 3010 | 0.108 | +76 % | 150 |
| 0.3        | 1910 | 0.99 (never above)   | 3088 / 2987 / 2974 / 3004 | 0.106 | +78 % | 1022 |
| 0.5        | 5305 | 1.02 (<= 10 cells)   | 3088 / 2963 / 2950 / 2998 | 0.105 | +75 % | 2744 |
| 0.3 linear | 1910 | --                   | 2811 / 2733 / 2736 / 2770 | 0.117 | +50 % | 1038 |

Reading:
9. The pulse passes through. Even 1.7x the ring's energy in the wave lifts the core only to 1.02 S_c (the wave's phi is
   spread and mostly not aligned with the core's), sum|G| and speed are amplitude-independent, Eu after each pass is
   just the wave's half-share (store == linear to ~2 %), Edf moves by <= 1.5 % and oscillates back. No capture, no
   re-shed.
10. Why (structural): the held content is circulation, Kelvin-conserved by step() + Cartan; a wave is a gradient (zero
   loop sums), so it cannot add or remove circulation -- the only channel is the store nonlinearity, which needs the
   core already at S_c and then only reshapes. A body whose content is circulation cannot absorb from waves -> it is
   not a receiver, so it cannot be an emitter either (Philip: emitter = receiver) beyond the one-time readjustment.
11. A receiver needs a wave <-> circulation channel. Continuum analogue: Kelvin is broken only baroclinically
   (grad rho x grad p). Here that term ALREADY exists in step() with a face medium: phi_t = -W sigma_f grad u has
   nonzero loop sums where sigma varies -> u waves crossing a sigma (lam) gradient create / destroy circulation.
   Philip 10-19: lam is a side effect of the structure -> a structure-caused sigma well around the body is exactly the
   receiving / emitting channel (Biermann-battery form, cf. curl-full spheromak memory). Next test.

## Wave <-> circulation channel via a sigma well (well.sh; well/*.txt; linear store, relaxed_A, T 25)
sigma dip / bump +-0.3 (Gaussian width 6) centred on the ring, pulse AMP 0.3 as above; base = no medium no pulse.
Pulse effect = (with pulse) - (same medium, no pulse):

| t | sum|G| none / dip / bump | Edf none / dip / bump |
| 7  | -3.3 / +0.5 / +1.7 | -41 / -46 / -33 |
| 14 | -0.7 / +0.2 / -0.4 | -1.5 / -0.2 / -3.2 |
| 20 | +0.4 / +3.9 / +4.3 | -41 / -46 / -33 |
| 24 | +0.5 / +3.0 / +2.4 | -14 / -20 / -11 |
(sum|G| ~ 413 -> 610 over the run; Edf ~ 2700-2800.)

Reading:
12. Even with no medium the pulse dips the ring's div-free energy by ~1.5 % while it passes and returns it (Cartan
   advection by the wave's velocity: reversible exchange, the 10-19 "exact part feeds u"). The well changes that by
   +-15 %, odd in depth (dip deeper, bump shallower) -- still reversible (~0 at t 14).
13. The baroclinic channel is there but makes +-pairs: the pulse's sum|G| effect is even in depth (+3..4 for both
   signs after the second pass, ~1 %), i.e. paired circulation of both signs, net zero. The ring's own (net)
   circulation is untouched. A smooth wave through a smooth medium cannot deposit a net amount into a held body.
14. So with the store, the well, or both: a held circulation body sheds once (to its ceiling) and is then neither a
   receiver nor an emitter. A net transfer between bodies needs a change in what is held -- reconnection / topology
   change (the 10-18 "moved counts" item), not a smooth coupling. All smooth routes tried today (store, sigma well,
   rate / carrier bounds 10-19) only reshape or steer.
