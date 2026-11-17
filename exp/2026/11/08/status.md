# 2026-11-08 approach law from existing pair logs; expansion-sourced body (queue Live 1)

Philip: 1b (+-30 to T 600) dropped -- a no-bridge result says nothing (bridge could come later), a bridge only gives
an onset time; same-sign depressions only attract, so a SETTLED pair needs something else. t 375 ckpt kept.

## approach/fit.py -- approach vs separation, no new runs
toward(t) = pair centroid shift toward the partner minus its solo's shift, both bodies averaged; fit A t^2.
  t <= 300:  A(18 n112) 3.8e-5   A(24 n128) 1.2e-5   A(24 n144) 1.3e-5   A(30 n144) 0.68e-5 (+- 0.2e-5)
             24 vs 30 (n 144): A ~ d^-2.9 +- 1.5 ; 18 vs 30: d^-3.4 +- 0.6 (18 includes its bridge)
  t <= 150:  NO approach at 24 or 30 (A consistent with 0, toward +0.05 / -0.07 at t 150); 18 already +0.5.
Reading: not a constant pull -- approach starts ~t 150-200 at both 24 and 30 (same onset, so not carrier travel time:
P crosses 48-60 in ~50), then accelerates. The exponent (2.5-4, +-1.5 from two separations) can't tell 1/d^2 from
1/d^3, and the bodies are still changing (E -5-7 %, d max moving), so an approach law needs a body that has stopped
evolving first. Onset candidates: the body's own slow evolution (rim loss, deepening), the tail building up.

## expand/pair.py = 06/box/pair.py + --signs (per-ring depletion sign; source sign = nearest centre) + --dmin
Solo, n 64, R 8 a 3.2 q 4 --src local dt 0.05, share = u^2 within 1.6 R, E lost at the rim sponge (r > 24):
  contraction 0.8 (control)  t 20 share 0.89 E 0.99   (holds, as 10-25)
  contraction 0.3            t 10 share 0.54, t 20 0.002 E 0.79   (too shallow, as 10-25)
  no well (0.01)             t 10 share 0.014, t 20 E 0.13       (free dispersal)
  expansion 0.1 / 0.3 / 0.5  t 10 share 0.002 / 0.001 / 0.002, E 0.98 / 0.92 / 0.81 at t 10, ~0.05-0.1 by t 20
-> an expansion-sourced body cannot hold itself: faster cells in the core defocus (content leaves faster than with
no well at all). In this rule a self-held body is necessarily a contraction. The expansion (electron candidate) can
exist only if something else holds it: e.g. the displaced units of a contraction (conserved size: core contracts,
shell expands -> the 09-24 'glut in the moat'), or the ambient. Not built; the conserved version also removes the
1/r tail (net zero) unless the displaced units spread far -> needs Philip.

## (pm) toy track parked (Philip); back to the spheromak cascade, rule only (step() + Cartan), a 9 P 96 T 250
cascade/cascade.py = 10-22 sph.py + non-vector share per print + phi dumps (/bk/dst/2026/11/08/cascade/a9, every 10).
Bit-identical to 10-22 a9.txt. (Gotcha hit twice: sph.py resumes from SEED_ckpt.npz -> fresh seed copy, delete ckpt.)
1. Non-vector share (anchor_residual^2 / face content) grows EXPONENTIALLY FROM t 0: 1.2e-4 (sampling residue of the
   seed) x1.62 per 10 time units (rate ~0.048, doubling ~14, ~2 doublings per turnover a/v 30), slowly speeding up to
   x1.84 by t 100, saturates ~0.29 after t 200 (a9_nodump.txt / a9.txt). The "onset at ~3.4 turnovers" is just when
   it becomes visible: linear instability of the spheromak under the rule, not a threshold. Philip: the substrate
   geometry is unknown (one set, overlapping sets, staggered / flush) and nothing says the field is smooth at 1e-44
   -> "non-vector" is only this lattice's label for cell-scale content; the observable is the coarse-grained body.
2. coarse.py (curl-curl smoothing tau 1 / 4, ratios to t 0; coarse_a9.txt):
     t      E_s/E_s0  overlap  H_s/H_s0   (tau 1; tau 4 the same within ~0.02)
     50     0.996     0.998    0.994
     100    0.945     0.994    0.946
     150    0.624     0.983    0.602
     200    0.137     0.898    0.103
     240    0.038     0.637    0.015
   = the COARSE body decays IN PLACE keeping its shape (overlap > 0.98 while it loses 40 % of its energy, 0.9 at 86 %)
   -> seen coarse, the cascade is a decay into fine substrate disturbance (energy kept in total, dE/E 7e-6).
   vs MHD reference (eta 2e-3 n 64: EB 0.954 at t 50, 0.315 at t 1500, H 0.365, overlap 0.94): same reading (shape
   kept, body fades), different law: MHD constant rate from t 0; here d ln E_s / dt rises 0.003 (t 100) -> 0.015 (145)
   -> ~0.033 (175-210), ~0.029 (235) = slow start (exponential growth of the fine part) then ~constant-rate decay.
   Helicity: tracks energy early, then decays FASTER (late rate ~0.05 vs 0.033) -- opposite to MHD (helicity outlasts).
3. a 20 P 192 (restart of /bk/dst/2026/10/22/sph/end_a20.npz at t 200 -> T 450, dumps every 10; cascade/a20.txt,
   coarse_a20.txt with tau 4.938 = (20/9)^2 x 1, every 20; reproduces 10-22 at t 200 and 450):
   - fine growth (non-vector share 2.3e-3 at t 200 -> 0.29): rate ~0.045-0.058 per time unit = SAME as a 9 in time
     units (cell-set), i.e. 3.3 vs 1.45 per turnover.
   - coarse body decay (d ln E_s / dt): peak 0.036 (a 9) vs 0.019 (a 20) = 1.07 vs 1.27 per turnover -> the decay of
     the coarse body scales ~1/a (x a: 0.32 / 0.38), like an ideal (turnover) time, not resistive 1/a^2 (Taylor
     2 eta lam^2, lab spheromaks). Shape kept while decaying (overlap 0.98 at E_s 0.62 both).
   - helicity decays faster than energy in both, H/E rate ratio ~1.0 early -> 1.4 (a 9) / 2.3 (a 20) late: inverted vs
     MHD / lab, worse at better resolution.
   Two scalings: the fine part grows at a cell-set rate; the coarse body drains at ~ one e-fold per turnover once going.
   Open test: rate vs amplitude (vmax 0.15 at a 9): MHD / resistive decay is amplitude-free, turnover scaling is not.
4. Amplitude test, a 9 vmax 0.15 to T 500 (seed = 0.5 x seed_a9, E 74.1 = 1/4; cascade/a9_v15.txt, coarse_a9_v15.txt):
   EVERYTHING runs at half speed. Fine growth x1.79 per 20 (vs x1.62 per 10 at 0.3); coarse peak decay 0.0188 at t 400
   (vs 0.0357 at t 191), x a/v 1.13 vs 1.07; E_s = 0.5 at t ~320 vs ~161; late H/E rate 1.8 vs 1.5; end state the same.
   -> correction of 3: the fine growth is NOT cell-set; it is ~0.16 v / h (same at a 9 / a 20 because v was the same).
   Expected in hindsight: step() freezes div-free phi, so the dynamics are Cartan's, which is quadratic -> phi -> c phi,
   t -> t / c is a near-symmetry (broken only by the small u / wave part, Eu/E ~1e-3). The rule's spheromak decay is
   purely IDEAL (Euler-like self-advection): rate ~ v / a for the body, ~ v / h for the fine growth, no amplitude-free
   (resistive-like) part at all. MHD / lab decay is amplitude-free (2 eta lam^2) -> the rule has no resistive channel;
   what it does is ideal turbulence of the body into cell-scale content.

## QUEUE 2026-11-08 (resume 11-09) — read `now` (STATE + 11-08 session) first. Older queues verbatim in exp/2026/MM/DD/status.md (after that day's now entries). 11-04 pre-tidy copies: /bk/dst/2026/11/04/{now,next}.pre_tidy. CPU: 4 cores unless Philip says otherwise.

11-08: the ring + P body (10-25 .. 11-08: jigsaw ring2.py, pair runs, approach law, expansion test) is a hand-built
toy (P is not the rule) -> PARKED, its results are not evidence (now STATE 'TOY, NOT EVIDENCE'). The old Resume /
Live / Ideas lists that hung on it are in exp/2026/11/04/status.md and /bk/dst/2026/11/08/next.

Resume here (11-09):
1. **Decide the direction with Philip** (open question at the end of 11-08): the rule's spheromak decay is purely
   ideal (rate ~ v / a, amplitude-scaled; fine growth ~0.16 v / h from t 0) with helicity draining faster than energy;
   MHD / lab decay is amplitude-free with helicity kept. Two routes:
   a. Which offload channel (radiation / resistive / Landau, 10-24) comes out of the rule as an amplitude-free exit,
      and what in the rule would stop / slow the ideal transfer to cell scale (structure-scale stop).
   b. Why the coarse field loses helicity faster than energy under Cartan (continuum Euler keeps helicity exactly;
      where does the coarse helicity go -- to cell-scale helicity of either sign? measure the fine-scale helicity
      in the a 9 / a 20 dumps, cheap: no new runs).
   Tools: exp/2026/11/08/cascade/{cascade.py, coarse.py}; dumps /bk/dst/2026/11/08/cascade/; MHD logs
   exp/2026/10/24/ref/mhd_*.txt (common units in its compare.py).
2. Spheromak on a second lattice geometry (different count of cell-scale DOF) to check the decay is lattice-agnostic
   (continuous code is BCC only: porting cost to be checked first).
3. Settled background follow-ups (exp/2026/11/03/bg/) and the photon readings W / P: rule-only, still valid.
4. A1 relay / CHSH, A2, A3 receivers: parked until something held exists in the rule itself.

Questions for Philip:
- Spheromak: which offload route first (Resume 1a vs 1b); are the 4 extra face DOF / cell-scale content heat?
- Bell: are detector settings part of the global mutual equilibrium (superdeterminism route) or free?
- Why the displaced units (electron) gather into one body instead of spreading.
- L sign / how a structure modifies lam_bg (09-30 / 10-01 / 10-22); walk beta-clock removal (commented, 10-25).

Parked / optional:
- Embedded spheromak: n / n-bar pair inside the 10-24 ambient (deprioritised; the ambient does not hold anything).
- Hybrid reference rung: spheromak in PHARE (~/git/phare/master); prior EMHD spheromak drifts along -z (09-26/27).
- Size law of a cell-scale hand-off (a 6 / 9 / 13), once an outflow route exists.
- Memory per cell (~0.6-0.8 KB) and continuous AMR (level 0 only today). (10-22)
- Finite propagation range (Philip 09-06): energy inside an expanding shell on the 14-neighbour lattice. Cheap.
- H r = const at fixed circulation (exact, 09-02) never tested dynamically.
- Precession scan of (p, q) torus coils (10-20); Rauch-Werner 4 pi as an observation (~9 h at R 20; think first).
- Downstream photon tests (polarizer, Eddington 2x vs 4x, Casimir d^-4) -- coarse only (no discrete photon).
- Derive sigma = ell and ell(depletion) from the rule (lensing GR/T 2.02, 09-25).
- Macro-particle scaling (09-27 idea); S_c refinement (10-21); minor whole-turn ring items (10-14 / 10-15 / 10-20);
  4/3 curl calibration vs the 09-28 weights.

Hygiene:
- C++ speed (11-03): remaining step_cartan cost is neighbour gathers in cartan_transport + slab copies in
  step_cartan (~8 %); 8 threads look memory-bound. Profile at 8 threads first; only if runtime blocks.
- Code hazards in doc/dst_caveats.md "Latent hazards"; psi_pair C2_CALIBRATED fitted on legacy weights.
- Casimir-Polder redo or withdraw (flagged in dst.md).
- BC leftovers after AMR (09-30).
- /bk/dst cleanup (exp/2026/10/18/bk_cleanup.sh ~73 GB, walk states ~25 GB, ckpts /bk/dst/2026/10/{14,18,19,20}/,
  R160 'dec' partials). MEMORY.md trim (~37 KB; stale: L8 "survives many causes" -> resolution, not depth).
- Lit: discrete Lie derivative (Elcott, Mullen, Pavlov), anapole / luminal nonradiating sources, Purcell, geons
  (Wheeler: wave held by its own depression -- today's body is geon-like), boson stars / Schrodinger-Newton.
