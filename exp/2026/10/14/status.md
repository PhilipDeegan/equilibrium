# Entries for 2026-10-14 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## SESSION 2026-10-14 — R 40 whole-turn run done: the late leak converges away (sum|Gamma| second order, rms first order R 20 -> 40) = discretisation residual of step() + Cartan; valve test null as expected (off-quantum circulation is held exactly, response linear in eps down to 0.003, no threshold); vill.py gained seed cache + checkpoint / resume

1. **R 40 late leak** (exp/2026/10/14/cartan/status.md): P 278, 5.37M cells, 14:30-17:28, 1.4 s / tick (8 threads).
   E drift -1.1e-9 (R 20 -1.6e-8, R 9 -4e-7). Growth vs t 0 at matched travel, R 9 / 20 / 40:
   0.10 R: rms +4.9 / +1.0 / +0.54 %, sum|Gamma| +2.0 / +0.16 / +0.06 %;
   0.12 R: rms +7.6 / +1.7 / +0.82 %, sum|Gamma| +3.8 / +0.44 / +0.11 %.
   R 20 -> 40: sum|Gamma| x 1/4.0, rms x 1/2.1. Onset travel ~unchanged, amplitude shrinks. T 145 reached 0.135 R, not
   0.15 (speed dropped slightly). Open, small: pre-onset rms creep ~+0.2 % per 0.04 R at both R 20 and R 40 (not
   shrinking) -- maybe a slow resolution-independent core widening; watch.
2. **Valve test** (valve/, R 9 P 72 smooth 1 T 20, --circ 1 +- eps, eps 0.003 .. 0.1): linked circulation stays 1 + eps
   (1e-5); d rms / eps 1.33-1.40 and d Eu / eps ~0.014 (Eu = u-energy fraction = wave part) constant over 1.5 decades ->
   linear response, no threshold, no preferred circulation. Energy beyond 4 / 6 cells (~55 / 30 % at eps 0, the ring's
   far flow) is not a shedding metric. R 20 skipped (same exact conservation).
3. **Tooling**: exp/2026/10/14/cartan/vill.py = 10-13 + --seed (cache post-smoothing phi), --ckpt (resume, bit-identical
   at P 72), --rcut / Eout, Eu. R 40 seed + final ckpt at /bk/dst/2026/10/14/cartan/ (0.6 + 1.2 GB; delete unless a
   longer R 40 run is wanted). Needs PYTHONPATH=repo root when run from exp/.


# Entries for 2026-10-14 moved from repo-root `next` on 2026-10-25 (verbatim, newest first)

## QUEUED 2026-10-14 — read `now` 10-13 + `exp/2026/10/13/cartan/status.md`

State: transport fixed (Cartan, kappa 1, energy-exact, Kelvin-exact, converging). Whole-turn smooth rings travel and keep
shape. Circulation is conserved per loop -> quantization cannot emerge from step() + transport.

00. Late leak: DONE 10-13 + R 40 DONE 10-14 (exp/2026/10/14/cartan/status.md: sum|G| 2nd order, rms 1st order R 20 -> 40). -- shrinks under refinement (R 9 -> 20 at matched travel: sum|Gamma| +23 -> +5.4 %), mostly
   discretisation residual. R 40 NOT DONE: started 10-13 14:18, killed at t 10 (machine shutdown; speed 0.0387, flat so far). Rerun: vill.py --P 278 --R0 40 --T 145
   --every 10 --smooth 19.753 --off 0.13 0.07 0.11 --flow cartan --kappa 1 --threads 8 --cpp --midpoint 2 (seed ~25 min, run
   ~2.7 h -- start early in the day; t 0: rms 8.878, sum|G| 1867.1 (/2pi), E 12883.4376). Compare at 0.15 R: R 9 sum|Gamma| ~+5 % rms ~+9.5 %; R 20 +1.0 % / +3.4 % -> convergence order.
0. **Imbalance, not perfect equalising** (Philip 10-13): a perfectly self-equalised p / e would not bond; the energy gap
   = an unequalised relation; pinch -> two balloons needs each half to be a supported structure. Don't build a perfect
   core equaliser. Open from the core-width scan: a too-narrow core shatters into +- pairs (net-zero debris), not into
   same-sign supported halves -- whatever splits a structure must leave two holding pieces.
1. **Valve, not shedding** (Philip 10-13: off-quantum shedding shouldn't happen; quantization is reciprocal -- offload
   needs an activation energy, like shell jumps; discreteness may be the valve): perturb a holding ring (core rms >= 2)
   by eps, record total energy + core vs outside energy, sweep eps ~1.5 decades at R 9 then R 20. eps^2 down to 0 = no
   valve; ~0 below eps* = valve. Expect none in step() + Cartan (linear + smooth quadratic).
2. **Geometry under fixed transport** (doable now, cheap): tokamak core + windings, Hopf-wound ring, twisted ring --
   which cycle locally vs leak (cartan vill.py at P 72 / 140, minutes). Leak metric: core-only |Gamma| weight.
3. ~~C++ port~~ DONE 10-13: alat.step_cartan / cartan_transport, tested (TestCartanTransport in test_continuous_cpp).
4. **R 40 whole-turn** only if a question needs it (seed 40 min: cache to /bk/dst).
5. Carried from 10-13 item 7: tick smooth limit == step() test; farfield fast-fail test; barrier closure check; walk
   beta-clock; /bk/dst deletions; MEMORY.md trim; lit search (structure-preserving discrete Lie derivative -- Elcott,
   Mullen, Pavlov; is this per-triangle Whitney form known?).

---
