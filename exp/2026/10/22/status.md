# Entries for 2026-10-22 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## SESSION 2026-10-22 — focus electrons (protons a while away); Philip: translating structure = spongey + prefers default, structure = substrate depression, relations before geometry, spheromak is the trusted geometry and its DECAY is the near-term relation test (electron pinch long-term); bound structure = road block (photons can't pass, breaking costs energy, e repels e), cells contract near it, stretch only where unforced (galaxy edge), contraction maybe outside energy; substrate is not a fluid (fluid descriptions sit on it); first spheromak in the anchored model

1. Paper: exp/2026/10/22/depression/{design,depletion}.md -- rate-only depression cannot prefer a state IF H is conserved;
   energy depression blocked by frozen-in; the "depletion relation" (compressible fluid description) is SUPERSEDED
   (backwards: fluid description as source, and sign: cells contract near structures, not empty).
2. Relation tests from Philip's road block: opacity (10-20 push: pulse passes through a held ring -> current rule
   fails), threshold to break, like-like repulsion (off-symmetry pair); decay D1-D3 (decays, helicity slower than
   energy, structure-scale rate).
3. Spheromak baseline, step() + Cartan (exp/2026/10/22/spheromak/sph.py, run.sh; /bk/dst/2026/10/22/sph):
   seed lam a converges ~2.6 / a^2 (a 5 / 7 / 9 / 13: 10 / 5.3 / 3.2 / 1.5 %), Cartan residual x a / v const 0.26 =
   seed wall (non-Beltrami), not resolution -> anchored model needs ~a 9-16, not 220 across (old hop-path figure).
   a 9 (P 96, T 200): held ~100 (3 turnovers), then accelerating breakdown: |dphi| 1.17, cos 0.97 -> 0.18, helicity
   (anchor measure) 9 %, energy 6e-6, ~1 % of energy outward, drift 1.5 along axis. a 20 (P 192) running, ends ~13:30:
   same onset in turnovers -> structural (tilt?) else lattice; check helicity measure on end states.
   a 20 DONE (T 200 = 3 turnovers a / v; dE/E -1e-8): turnover-matched vs a 9 (0.5 / 1 / 2 / 3 turnovers):
   |dphi| 3.2/6.1/11.3/20.1 % (a 9) vs 3.1/5.4/9.3/14.1 % (a 20); cos 0.965/0.964/0.955/0.921 vs 0.963/0.962/0.958/0.928;
   helicity loss 0.04/0.12/0.35/2.2 % vs 0.00/0.01/0/0.9 %. Axial drift 0.033 a vs 0.030 a by 3 turnovers (same ->
   structural: the smoothed seed carries net impulse, v ~ 0.01 vmax). Reading: first turnover = seed readjustment
   (converged, set by the non-Beltrami wall); helicity loss is lattice (2.5x less at a 20); the late acceleration
   (cos drop 2.5 -> 3 turnovers) starts at the same turnover count in both -> likely structural onset, a 20 must be
   extended (T 450 from ckpt, ~95 min) to see whether it breaks down like a 9.
   a 20 EXTENDED to T 450 (6.7 turnovers; a20_ext.txt; /bk/dst/2026/10/22/sph/end_a20_T450.npz): breaks down at the
   SAME turnover count (cos < 0.9 at ~3.4 vs ~3.6), and faster once going (cos 0.69 / 0.23 / 0.056 at 4 / 5 / 6 vs a 9
   0.84 / 0.68 / 0.37) -> structural instability of the spheromak under step() + Cartan, sharper at better resolution.
   End: energy kept (dE/E -1.9e-5), 94 % still inside r < a, Eu 0.3 %, measured helicity 2 %, |dphi| 1.33 -> the
   structure scrambles in place into a small-scale tangle (cascade past the grid; helicity is lost by the measure /
   lattice, a continuum invariant), it does NOT decay: no offload (D1 fails), helicity lost faster than energy (D2
   inverted). Open: which mode (tilt? axis tracking not yet measured); slices of the end state.
   Measure check (Philip: are we measuring the right thing?): scale-filtered on saved a 20 states (curl-curl smoothing
   tau 2 / 8): t 200 overlap with seed 0.990, energy surviving smoothing 0.826 / 0.504 (seed 0.834 / 0.511) -> intact;
   t 450 overlap 0.11, energy surviving smoothing 0.051 / 0.032 -> ~95 % of the energy is at the grid scale (REAL, not a
   measure artefact); smoothed remnant cos 0.36; moment axis tilt 0.5 deg -> NOT a tilt. = delayed (~3 turnovers)
   breakdown of the whole structure into a grid-scale tangle in place; the end state is lattice-scale at both a, so
   it cannot converge (nature-not-sharp failure mode). Gradual lab decay vs this abrupt onset differ; helicity not kept.
   Unwinding? (Philip): large-scale twist does unwind -- net toroidal B (sum Bt, r < 1.5 a) +292 (t 0) -> +285 (t 200)
   -> -69 (t 450, sign flipped), net toroidal circulation of A 4100 -> 1040; but the released content goes into face
   modes that are NOT any cell vector field (anchor_residual share of face content 0.000 / 0.002 / 0.503 at t 0 / 200 /
   450), the rest to grid-scale vectors. = unwinding INTO the lattice's own 4 extra face DOF per cell (7 per cell,
   3 = A). Relation question: what are those 4 DOF physically -- suppress (not physical) or the substrate's carrier /
   heat sector?
5. Philip: substrate is one connected whole, never disconnected; pattern topology contextual. Paper:
   exp/2026/10/22/depression/contextual_hold.md. BUILT: alat.set_cartan_hold(i, kappa (N,)) / clear_cartan_hold;
   per triangle om -> kh om + (1 - kh) <om> (same template averaged over the cell + 14 face neighbours, kh = mean kappa
   of its 3 cells); numpy reference continuous.cartan_transport(hold=), AnchorCluster.set_cartan_hold. Tests:
   TestCartanTransport.test_hold (kappa 1 bit-identical, numpy 1e-13, antisym, energy-orth, zero on gradients);
   tst/substrate/test_continuous_spheromak.py (Taylor Beltrami 1e-8; a 6 P 64 first turnover: cos 0.967, dE/E 1e-9,
   overlap 0.997, non-vector face share 0.0017; 19 s on 2 threads). 48 continuous tests pass. kappa source (contraction
   from local energy) not wired yet; L-sign question open with Philip.
4. Proposed, awaiting go: slaved-geometry prototype (cell size from local energy each tick via set_face_medium n = L/h,
   no energy in contraction, drift tracked), scored on opacity (zoo --pulse R 9) then decay (sph.py).


# Entries for 2026-10-22 moved from repo-root `next` on 2026-10-25 (verbatim, newest first)

## QUEUED 2026-10-22 — read `now` 10-21 + exp/2026/10/21/valve/status.md

**Resume Saturday 2026-10-24. Philip 10-22 close: spheromak decay is likely downstream of EM effects / Poynting-type
flux** -> look at the decay through the EM sector (E / B reading of phi / u, Poynting out of the spheromak; start from
mem project_dst_em_field_convention_crisis and the step_ae E field) before wiring kappa into Cartan.
**Latest (10-22 pm):** spheromak under step() + Cartan unwinds into the 4 non-vector face DOF at ~3.4 turnovers
(converged onset). Philip: substrate never disconnected (one whole), pattern topology contextual. Paper:
exp/2026/10/22/depression/contextual_hold.md -- Cartan with context-blended vorticity om_eff = kappa om + (1 - kappa)
om_avg (energy exact per triangle, kappa = 1 == today), kappa from contraction slaved to local energy. Awaiting Philip
on the L sign question, then T0-T3 (C++ cartan_transport change is small).

**Philip 10-22: protons are a while away -> focus on electrons, or even neutrinos.** Item 1's p + e / multi-ring
aggregate framing is set aside; aggregates only as far as a lone electron / neutrino candidate needs a context.
Philip 10-22: translating structure = spongey + prefers its default; structure = substrate depression (cell gradient?).
Paper design exp/2026/10/22/depression/design.md: rate depression (sigma(S)) cannot prefer a state (not in H) and is
scale-free; energy depression is blocked by frozen-in (Lie-Poisson Casimirs fix core / size from the IC). Default =
relaxation (smooth helicity-keeping frozen-in breaking, energy -> u) + size = confining depression. Rings cannot sit
at rest (speed slaved to size); Beltrami / spheromak is static -> better electron class. Awaiting Philip on the 3 Qs.
Philip: relations before geometry; fairly sure about the spheromak (acceptance target). Depletion relation on paper:
exp/2026/10/22/depression/depletion.md -- step() + Cartan = compressible Euler minus mass weighting + Bernoulli; add
them via H = sum rho K + c^2 (rho ln rho - rho + 1) (store pattern) -> co-moving depression, core ~ 0.42 n / c cells,
spheromak static with own depression; no quantization yet. Tests T0 unit, T1 straight line, T2 antiparallel, T3 spheromak.
Memory (Philip 10-22: lower cost per cell helps refinement; refinement only where needed = continuous AMR is intended):
profile P 96 ~2.4 KB / cell (state 112 B): topology 708, step_cartan scratch 624, property copies 224, DEC 208,
recon cache 449. DONE: recon cache no longer copied when all faces valid (periodic box: -336 B / cell; 45 continuous
tests pass). Queue: (2) 7-face antisymmetric storage (halves phi + scratch +
cart); (1) DROPPED (Philip 10-22: C++ diagnostics not needed; numpy stays the independent check) -> chunked numpy diagnostics instead, no full copies; (3) neighbours are already int32 everywhere (-1 = no face) -- drop the duplicate tables instead (slab nb + anchors nbj + Python nb, 56 B each), lean slab; target ~0.6-0.8 KB / cell; then continuous AMR (step is level-0 only).

State: R 20 tilted ring pair: no clean reconnection (grid-scale tangle after t ~75); topology frozen in the rule's
continuum limit. Philip 10-21: the valve is not local, update limits are operating limits, the valve is likely
downstream of particle aggregates -> the 10-21 queue's items 1-2 (local valve, what the integer counts) are superseded.

1. **Which aggregate first** (Philip's call): needs persistent particles that can sit together -- bound p + e (Bohr /
   H2 lineage), a few identical held rings (multi-moat / ideal-gas lineage), or a cluster of rings sharing a receiver.
   Check what current persistent structures (step() + Cartan rings, walk spring coils) can form a bound aggregate at all.
   10-19 pairs cannot answer it (sep = side / 2, image-symmetric); rerun with sep ~ 1.5-2.5 R off the half-box point.
   Cost: R 20 P 192 T 30 ~ 35 min at 8 threads; binding needs much longer T (R^2 ticks per orbit-scale event).
2. S_c refinement scaling (10-21 item 3) still open if the store is used in aggregates (S_c ~ h^2 = fixed field ceiling).
3. Carried from 10-21 items 4-5.

---
