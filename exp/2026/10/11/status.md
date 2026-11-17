# Entries for 2026-10-11 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## SESSION 2026-10-11 — review of now/next; "curl without vectors?": discrete grad/div/curl are counting (fine), the 10-08 A x B carry term is a vector import; replaced by an exchange-matrix commutator (Pavlov et al. 2011 form, no vectors, energy-exact by construction) that reproduces the ring transport and converges under refinement; the pure tick phase cannot host it conservatively (only its gradient part = the q clock shift fits in theta) -> a DOF near the defect is needed

1. **Review** (no runs since 10-08): items 1-2 of the 10-09 queue are one problem -- with phi = W wrap(dtheta) circulation
   lives only on defect triangles, so any transport acts only there and moving a defect needs a slip = the pinning.
   Pinning = Peierls-Nabarro (sub-cell core); a core a few cells wide lowers the barrier ~exponentially. Soft-core risk:
   a depletion-weakened coupling is the walk's density turn / GP core under another name unless derived.
2. **Vector-free forms** (Philip chose 1, keep the rest; list in `next` IDEA 10-11): exchange-matrix commutator (Pavlov
   2011); TRiSK face-face weights; discrete wedge (Hirani / FEEC); Elcott loops (uses positions); lattice gas (walk
   lineage). Tick phase = lattice XY / Villain / compact U(1); textbook defect mobility = amplitude core (GL/GP).
3. **Commutator prototype** (exp/2026/10/11/lie/status.md, lie.py): a = phi / W (circulation per hop), v = W phi (flux =
   dH/da), antisymmetric N x N on the faces; da/dt = kappa P_S [M, v], M = a extended to 2-hop pairs by the mean over
   2-hop paths (50 displacements). 1-hop parts cancel exactly -> driven only by path dependence = loop circulation. nb
   only, no positions / cell vectors / cross product. Antisymmetric 5e-16, sum phi . term 7e-15 (energy-exact). Div-free
   part correlates +0.91 with carry's A x B (scale 7.58); has a large gradient part (-> u as sound; carry dropped it).
4. **Ring runs** (P 120, R0 12, wt 3, a0 1, kappa +-0.132 = 1 / 7.58): +-0.0328 / time, exactly sign-reversed, shape kept,
   leapfrog invariant to 4e-7 (carry: 0.031). R0 18: 0.0218, ratio 0.665 (carry 0.68, thin ring 0.74).
5. **Refinement** (all lengths x f, a0 x f -- a0 is the streamfunction amplitude, flow = a0 / wt; first f sqrt2 run had a0
   fixed, flow 1 / sqrt2 smaller): t 0 instantaneous speed 0.03435 / 0.03574 / 0.03646 / 0.03693 at f 1 / sqrt2 / 2 /
   2 sqrt2 (differences ratio 1.93 then 1.53): converging, limit ~0.0375-0.039, not a stencil artefact. t 0 estimate vs
   long run: 0.955 (f 1), 0.98 (f sqrt2); f sqrt2 long run stopped at t 2.84 (0.0349 vs 0.0351 predicted).
   kappa is still picked (matched to carry).
6. **Commutator on the tick phase** (exp/2026/10/11/tick/status.md, grad_part.py): theta can only take the gradient
   part, which IS tick.py's q (Bernoulli |v|^2 / 2): corr -0.98..-0.99 beyond 3 cells, -0.93..-0.97 at the core; kappa_c
   0.132 == kappa_q ~ +-5.7 = the 10-08 kappa 5 run (moves, energy x3). The div-free part (0.41 there) has nowhere to go;
   the commutator is energy-exact only with it. Options: (a) soft / depleted core, (b) Villain integers (sharp), (c) wrapped
   phase + smooth real remainder (conversion = a gate). Proposed: (a), depletion driven by the part theta cannot hold
   (nonzero only near defects). Philip's call pending.
7. **Practice** (Philip, memory has each): no overnight runs (estimate ETA before chaining); long runs print a t 0
   estimate + running value vs --expect, invariant drift, ETA (lie.py does). This shell reports nproc 1.


# Entries for 2026-10-11 moved from repo-root `next` on 2026-10-25 (verbatim, newest first)

## IDEA 2026-10-11 (Philip: "is curl even correct without vectors?" -> keep all options) — vector-free self-transport

Discrete grad / div / curl are fine as counting (difference across a face / net over a cell's faces / sum around a loop).
The 10-08 carry term (A x B, vs/carry.py) is NOT: cell vectors rebuilt from faces + cross product. Known vector-free
forms of the same physics (from memory, not literature-checked):
1. Exchange-matrix Lie-Poisson (Pavlov, Mullen, Tong, Kanso, Marsden, Desbrun, Physica D 2011): phi = antisymmetric
   cell-to-cell matrix, transport = commutator. PROTOTYPED 10-11 (exp/2026/10/11/lie): reproduces carry's ring motion
   (0.0328 vs 0.031 / time, transverse corr 0.91), energy-exact, no vectors, converges under refinement; kappa picked.
2. TRiSK antisymmetric face-face weights (Thuburn, Ringler, Skamarock, Klemp JCP 2009/2010; Arakawa 1966 in 2D):
   face change = sum of neighbouring face fluxes x loop circulation, geometry weights only, energy exact.
3. Discrete wedge / cup product (Hirani thesis; Arnold FEEC): Lamb term as a combinatorial wedge.
4. Circulation on loops (Elcott et al. 2007): Kelvin exact but backtracks positions -> vectors; poor fit.
5. Lattice-gas counting (Frisch-Hasslacher-Pomeau 1986): walk lineage, reification risk.
Tick-phase defects: = lattice XY / Villain / compact U(1); pinning = Peierls-Nabarro; textbook mobility = amplitude core
(GL/GP = the walk again). Possibly novel: defects moved by option 1's exchange without an amplitude. Lit search pending.

---
