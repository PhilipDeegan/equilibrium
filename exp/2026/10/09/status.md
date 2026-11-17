# Entries for 2026-10-09 moved from repo-root `next` on 2026-10-25 (verbatim, newest first)

## (superseded 10-11) QUEUED 2026-10-09 — read `now`'s 2026-10-08 entry (items 6-8), `exp/2026/10/08/vs/status.md`, `exp/2026/10/08/tick/status.md`

Direction (Philip 10-08): the walk reified an approximation; work on the geometric model (step(): u per cell, phi on the
14 faces) and add what is missing as relations between cell / face quantities, not imported theories. Three pieces:
quantization (periodic tick phase: WORKS, exact), transport (phi self-transport: WORKS, energy-exact), mobility of the
quantized defects (MISSING: tick-phase defects are lattice-pinned).

1. **Soft core for tick-phase defects** (tick.py): the cells at a defect are maximally out of step (offsets ~ pi).
   Candidate: junction coupling (face sigma) weakens with depletion, depletion rises where a cell's junction offsets are
   large -> a depleted, soft core. Build as a cell quantity (lam-like) with its own update, energy accounted; test: does a
   seeded ring unpin (move / shrink) with circulation still whole-number and energy conserved? Ask Philip first what drives
   depletion here (offset size? tick-rate mismatch? both cells' sizes?) -- don't pick it.
2. **Combine**: tick phase + soft core + the energy-exact self-transport term (vs/carry.py --flow wsum --midpoint 3
   --centred; note: with phi derived from theta, a non-gradient phi term must be re-expressed at the theta / core level).
   Question: a quantized ring that persists AND moves.
3. **Port to C++** (continuous AdaptiveLattice) only once 1-2 settle; numpy prototypes are P 120 boxes (~1-3 min / t 30).
4. **Walk cleanup**: remove the beta clock (walk.py beta=, C++ set_walk_clock + Walk::beta/om0/rho0, TestWalkClock) --
   needs Philip's OK (asked 10-08). Mark walk-era memories as walk-scoped where not already.
5. Carried (walk-scoped, low priority): lone-box P 220 superflow discrepancy (+0.0002 predicted, ~0 measured); self-term
   0.0022 offset; N 4 tight state at r0 40; /bk/dst deletions (Philip's OK: 10-03 coil, 10-04 pair/pbox 16 GB, 10-05 R200
   22 GB, halo/scf 5.6 GB, 10-06 N 8 R200 t4000) + /bk/dst/2026/10/08/{couple,refine} states (~10 GB, reproducible).

---
