# 10-06 coil

## In-run probes (spring_cpp.py 10-06 copy, --probe_every 40, `P t ...` lines + probes_from<t>.npz)
Coil-only centre (core rho < 0.25 with rho_cyl > r0 / 2), centre-circle windings over 15 offset circles (0, radius 2 and
4 x 7 angles) + min rho on them, tube cross-section aspect (meridional second moments sqrt(l_max / l_min), tilt) and
rms radius, k 2 / 4 / 8 / N harmonics of the z-undulation and radial deviation (amp, pattern angle deg), axis ring
(rho_cyl < r0 / 2, >= 20 cells): radius, z, winding on a radius-4 meridional circle about it. Smoke-tested N 4 R75 80 ticks.

## Deficit budget (deficit_budget.py; D = sum(1 - rho), E = sum over hex pairs |a_j - a_i|^2 / 2, uniform bg E = 0)
Regions by distance dt to the tube centre circle: interior < a - 1.5 xi, strands |dt - a| < 1.5 xi, near < a + 4 xi,
outer < a + 10 xi, axis, floor (rest inside the sponge / whole box). 'structure' = interior + strands + near.

Tight box P 180, t 16000 (deficit_tight_P180.log; N 4-6 collapsed there, minor radius 2.9 -> only N 1-3 clean):
  N      1      2      3      4c     5c     6c
  D_str  60.1k  61.3k  65.7k  50.2k  64.1k  57.8k
  E_str  1737   2105   2222   2160   2237   2361
  D_tot  55.8k  62.1k  70.2k  77.6k  86.6k  94.1k   (walk conserves norm: total = the seed's, ~ +7.7k per seeded turn)
Big loop r0 120 R 200 (deficit_r120.log):
  N 16  t 4k / 8k / 12k / 16k: D_str 231k / 239k / 238k / 241k, E_str 10.9k / 10.9k / 10.7k / 10.6k (E/D 0.047 -> 0.044)
        axis D 11k / 33k / 1k / 6k (8k = the recurrence excursion; axis core 3482 / 2459 / 0 / 0 cells)
  N 30  t 6k (30 turns) D_str 297k E_str 15.3k; 8k (collapsing) 264k / 14.9k; 16k (4-6 turn kinked ring) 229k / 8.7k
Reading:
- Total deficit is not a usable 'mass': it is the seed's (norm conserved), and inside the sponge the floor swings
  -156k .. +229k between states (per cell ~1e-2: background breathing / sponge exchange over ~20M cells). Only the
  structure-local D is stable (N 16: +-2 % over 4-16k).
- Structure D and E grow sublinearly with turns: N 16 -> N 30 (strand length x1.6) gives D x1.24, E x1.43; the N 30
  end state (4-6 turns) keeps D ~ N 16's (229k vs 240k) but E drops to 8.7k. D tracks the tube (strand cores + near
  halo, ~70 % strands), E tracks the winding count more. Small loops: D ~flat 60-66k for N 1-3, E 1.7k -> 2.2k.
- E/D: 0.029-0.034 (small, N 1-3), 0.044-0.047 (N 16), 0.052-0.057 (N 30 intact), 0.038 (N 30 end). Not a constant
  ratio -> 'mass ~ deficit' and 'mass ~ gradient energy' are distinguishable; neither tested against anything yet.
- N 16's 8k recurrence excursion coincides with the axis anti-ring's deficit peak (33k): the excursion is the
  anti-ring episode, not a coil-shape change. Probes will resolve its cycle.
10-04 pair runs reread for a sign-blind attraction: every pair was seeded q +1 / q +1 (coaxial, same sense); no
opposite-sense pair exists -> sign-blindness cannot be judged. E2 20 apart closed to 14-16 apart (co-moving pair), but
coaxial same-sense rings interact kinematically (induced flow, leapfrog), so that is not attraction evidence.
Needs: q +1 / q -1 pair (and off-axis, side-by-side) at 20-40 apart.

## Philip 10-06: structures don't lose energy (adiabatic vacuum exchange); IC off the quantized values -> the
## unsupportable surplus is shed; don't discount neutrinos / small quanta in what is shed
- Sponge = one-way sink: every sponge-based loss reading is conditional (N 30 -> 4, scf 'loop is a sink', seed shed).
- Shed fraction (E outside structure / E total, periodic box = cumulative) tight P 180 t16k, N 1-6:
  0.28 / 0.17 / 0.19 / 0.29c / 0.34c / 0.38c -> minimum at N 2-3: quantization scan = shed fraction vs continuous r0 / a.
- Census (deficit_budget.py, rho < 0.6, dt >= a + 4 xi, connected pieces >= 10 cells): N 2 / 3 none; N 1 one cored
  piece on the inner side (axis ring); collapsed N 4-6: 3-8 NEUTRAL blobs (min rho 0.41-0.58, no core, D 6-400 each,
  E/D ~0.005 vs structures 0.03-0.04) -> rarefaction-pulse-like candidates, only where the coil changed state.
  Persistence unknown -> census added to the in-run probes (`| pieces n D: [cells minrho r th z]`).

## N 16 r0 120 resumed 16k -> 24k (spring_N16_r120_R200_resume16k.log, 4203 s; states /bk/dst/2026/10/06/coil)
16 / 1 throughout, minor radius 9.3-9.5, rc 119.4-120.5, helix phase +1.49 rad / 500 ticks (pattern ~2.1k), D_near
breathes 214k-274k. Recurrence vs relaxed t 4000 (recurrence_N16_ref4000_16-24k.log, scale 0.24): 0.053 / 0.046 /
0.047 / 0.052 / 0.050 / 0.047 / 0.049 / 0.051 / 0.054 at 16k-24k -> flat, no drift: since 9k (15k ticks) within
0.04-0.06. Best-rotation angle advances 11.75 deg / 1000 ticks (mod 22.5: 7.2 / 8.2 / 9.1 / 10.1 / 11.4 at even
thousands) -> constant rigid rotation. t 23000 at phi 0: 0.055 (pattern aligned with t 4000 unrotated).
Philip 10-06: "N16 at t=23k looks perfect". Best structure so far: rotating, not expanding, recurring.

## Quantization scan (exp/2026/10/06/quant: spring_box.py N 2 P 220 8000 ticks, budget_run.sh -> deficit_budget.py,
## summary.py -> table). Philip: IC off the quantized values -> the surplus is shed
  r0 scan (a 10): r0 20 / 24 / 27 / 30 / 33 / 36 / 40: structure E shed by 2k 0.264 / 0.209 / 0.192 / 0.185 / 0.180 /
    0.176 / 0.177, then flat (2k -> 8k within +-1.3 %) except r0 20 (+3.4 % more, relaxed minor 7.9: tube squeezed by
    its 10-cell hole, 2 small census pieces as r0 24). Relaxed minor radius 7.9 / 8.8 / 9.9 / 10.1 / 10.5 / 10.6 / 10.3.
  a scan (r0 30): a 6 / 8 / 10 / 12 / 14: shed 0.188 / 0.187 / 0.185 / 0.181 / 0.175; minor radius kept (6.0 / 8.0 /
    10.1 / 11.8 / 13.2 at 8k); a 6 and 8 regain 2.2 % / 1.7 % structure E 2k -> 8k (energy returning in the box).
Reading: no dip, no selection -- N 2 coils are a continuous family in r0 (>= 24) and a (6-14) over 8k ticks. The
~18 % one-off shed is a uniform offset = the seed's core profile (d / sqrt(d^2 + xi^2) envelope + solid-angle
phase), not geometry. Where quantization could still be: N (discrete), the core profile, or longer times.
Next: reseed from a relaxed core profile (free_after_clean showed near-zero shed for N 4) -> if the shed goes to ~0
for every r0 / a, the seed profile was the 'off-quantized' part; shed vs N at fixed geometry with the same profile.

## Rates vs geometry (Philip 10-06: we may end up calibrating a rate of change as 'the correct value')
Helix phase rate of the N 2 scan runs (rec.npz, t >= 2000, unwrapped linear fit, resid <= 0.09 rad) vs relaxed R:
  R 22.4 / 26.5 / 29.2 / 31.9 / 34.7 / 37.5 / 41.3 -> 1.362 / 1.059 / 0.911 / 0.780 / 0.665 / 0.574 / 0.477 rad / 1000
  (a scan at r0 30: a 6 / 8 / 12 / 14 -> 0.843 / 0.816 / 0.727 / 0.636, mostly via R 30.2 -> 34.7).
Kelvin ring form w = C (ln(8R / c) - 1/2) / R^2 fits to 2 % with C 0.260, c 7.6 (~2 xi); walk symbol kappa = 2 pi / M,
M = 3m / n -> kappa / 2 pi = 0.250 (not fitted). Reading (tentative: 2-param fit; helix phase = N x toroidal rotation
for N 2 is an unverified bookkeeping guess): a rate is a continuous function of R through the classical ring law ->
calibrating a rate selects R, not a constant; the law's coefficient = the walk's circulation quantum h / M is the
classical-limit reproduction worth pegging (ties M to the tick). N 16 r0 120 rigid rotation 0.205 rad / 1000 is ~5x a
thin-ring estimate (0.038) -> thick many-turn tube not covered.

## N 8 r0 120 a 10 R 200 (spring_N8_r120_R200.log, 8000 ticks 4535 s, probes every 40, states every 1000)
Coil: 8 / 1 windings over all 15 offset circles throughout, rc 119.8-120.6, aspect ~1.07, minor ~10; pitch 94 per turn
-> an undulating ring (k 8 z / r amplitude ~10), not a filled tube. Does NOT go to 4 turns in 8k (above the floor).
Interior (N8_nested_meridional.png, plane profile): concentric alternating density shells inside the loop by 2000
(+-5 % of rho0, spacing ~40, k 8 ripple phase-locked to the coil, sign alternating shell to shell), growing 1k -> 2k
= the shed field focusing on the centre (not an attached structure).
Probe time series of the centre (census + axis ring):
  2600 neutral low-density ring r 42, z +36 (ahead of the coil), min rho 0.56, no core -> shrinks / descends /
  deepens (r 22, min rho 0.18 at 3200) -> min rho 0 at 3560 (r 16) -> winding +1 at 3680 (opposite the coil's
  meridional -1) -> counter-rotating ring r ~13 in the coil plane at 4000 (same sign and radius as the N 16 axis
  anti-ring at 4000: r 12.8, +1). Then it leaves along -z: r 8.4-9.2, winding +1, ~2200-2800 core cells, steady from
  5000; lab speed 0.029 / tick (z rel. coil -37 -> -137 over 5160-7960, coil drift +0.0042). Kelvin ring speed with
  the walk's kappa (kappa / 4 pi = 0.125, not fitted): 0.034 (core xi) / 0.025 (core 7.6) at R 8.5 -> consistent.
Reading: the coil's one-off shed focuses on the axis, forms a neutral rarefaction ring that deepens into a counter-
rotating vortex ring, which self-propels away along the axis (here into the sponge). Same sequence in N 16 (where it
was read as 'shrinks to a blob'): the anti-ring is how the seed surplus leaves. In a periodic box it would come back
(Philip: adiabatic exchange) -> box version is the test of whether it returns / re-merges.

Plot / recentre bias (Philip 10-06: t 8000 coil mostly red, later ~50/50): state()'s zc was the core-weighted centre of
ALL core cells, incl. the axis ring (~2800 cells ~127 below) -> zc pulled ~7.7 below the coil (coil-only probe zc +9.6
vs +1.9 at 8000): colour, meridional panel (tube at z +18) and --recentre followed the biased centre. N 16 likewise
while its axis ring existed (log values after ~3500 already flagged as biased). Fixed in the 10-06 spring_cpp.py:
state() uses the coil-only core (rho_cyl > r0 / 2). Not a coil asymmetry.

## N 8 r0 80 R 160 sphere, 5000 ticks (spring_N8_r80_R160.log; discriminator for shed/ no-ring-in-box)
Coil 8 / 1 throughout, rc 80-81, minor ~9.8 (same as the P 480 box). Axis sequence as r0 120 but smaller and short-lived:
  2320 neutral low-density ring ahead of the coil (r 11-13, z +21..26, min rho 0.58) -> deepens / shrinks (r 8-9)
  -> core at 2680 -> winding +1 2880-3360 (core r 5-6, ~1300-1475 cells; r0 120's was r 8.4-9.2, 2200-2800 cells)
  moving -z ~0.046 / tick -> winding lost at 3400, core r 5.9 -> 1.4, gone at 3960; the neutral remnant fades by 4480
  at z -72 rel. coil, still well inside the sponge (r_inner 120).
Reading: r0 sets the counter-ring's size (r ~6 at r0 80 vs ~9 at r0 120 after leaving; ~1.5 xi here = too small to
hold a core) -> at r0 80 it is a transient that dissolves in the interior, not a persistent structure. The P 480 box
(shed/) never formed even the neutral precursor (no axis core, no census piece at any probe) -> the box suppresses
the focusing (shed returns within ~500 ticks / 26 images), independently of r0.
