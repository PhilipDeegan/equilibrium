## QUEUE 2026-11-09 (resume 11-10) — read `doc/dst_state.md` (formerly root `now`; STATE + 11-09 session) first. Older queues verbatim in exp/2026/MM/DD/status.md (after that day's now entries). 11-09 pre-tidy copies: /bk/dst/2026/11/09/{now,next}.pre_tidy. CPU: 4 cores unless Philip says otherwise.

11-09: Cartan is not legal (incompressible Euler imported) -> every step() + Cartan result (10-13 .. 11-08) is about
Cartan, not DST. The rule is step() alone again, in which divergence-free content never moves. Experiments paused on
purpose: the open question is conceptual, settle it with Philip before building anything.

THE question (talk it through first, no runs):
1. What in the substrate makes circulation move? step() only moves compression (u) and the gradient part of phi.
   In fluids, transport comes from the medium carrying its own disturbances; in DST that would be content changing the
   cells it sits on (contraction / rate / ell) and so how neighbours exchange. What is that relation, without importing
   a fluid law and without picking a coefficient?
2. What holds a structure (Philip 11-09): compression up to a limit (push-back grows without bound), 'surface tension'
   (it would diffuse if it could), spin pushing toward the centre. The pinch test showed: a convex limit only caps and
   flattens; an edge needs either a range where more compression is favoured (non-convex) or an outer push (ambient).
   Which does Philip intend, and is the limit the minimum cell size or relative to the local background? Is the limit
   one-sided (contraction only) and why does expansion differ?

Tools ready once the relation is chosen (no direction implied):
- pinch/pinch.py (exp/2026/11/09): Hamiltonian step() + s(u) + e(u); explicit RK4 with adaptive dt; um < ~0.1 needs an
  implicit integrator. Seed /bk/dst/2026/11/09/pinch/seed_a6_P96.npz.
- helicity/whitney.py: all-DOF helicity (validated); fine_hel.py, whitney_series.py; 08/cascade/coarse.py.
- References: MHD spectral (equilibrium/reference/spectral.py, exp/2026/10/24/ref), PHARE Harris / plasmoids
  (~/git/phare/master; alpha 10, lb 0.4 calibrated).

Questions for Philip (carried):
- Bell: are detector settings part of the global mutual equilibrium (superdeterminism route) or free?
- Why the displaced units (electron) gather into one body instead of spreading.
- L sign / how a structure modifies lam_bg (09-30 / 10-01 / 10-22).

Parked / optional (from 11-08 queue, see exp/2026/11/08/status.md): spheromak on a second lattice geometry; settled
background follow-ups (exp/2026/11/03/bg, step()-only parts still valid); A1-A3 receivers; finite propagation range;
H r = const; downstream photon tests (coarse only); derive sigma = ell.


## ARCHIVE NOTE (2026-10-25)

The 2026-07-12/13 context dump that lived here is obsolete (step_cross, E/B/Poynting,
pre-audit electron ICs) and was moved verbatim to `exp/2026/07/12/dst_next.md`.
References to "doc/dst_next.md sec 8" (step_cross `barrier` gate m0 derivation, in
`src/eq/lattice.cpp` and 08/09 experiment scripts) resolve there; step_cross is superseded.
Its sec 6 (decay success criterion: products as identifiable structures, no unattributed loss)
is folded into `doc/dst.md` *Structures, reception and offload*.
