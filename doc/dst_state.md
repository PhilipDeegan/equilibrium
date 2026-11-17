## LAYOUT (2026-10-25) — `doc/dst_state.md` (formerly root `now`) holds the current state + the latest session; every session's full entry lives verbatim in exp/YYYY/MM/DD/status.md (that day's now entries, then its queue); settled results and principles are in doc/ (dst.md "Circulation transport", "Structures, reception and offload", "Modelling principles"; dst_caveats.md; dst_algo.md "Exact identities", "Continuous model, level 0"). Pre-tidy copies: /bk/dst/2026/10/25/{now,next}.

## STATE (2026-11-09)

Frame (Philip, still standing):
- No E / B / dipole / Poynting: only substrate disturbances; deviation from the mean is the observation; find the
  face / loop / hop relation whose effects E and B describe (holonomies, loop sums).
- Mesh spacing outranks structure (cell-set rate = physics; rate vs a / ell is a prediction). The true lattice is
  undefined (maybe not Kelvin) -> mechanisms lattice-agnostic. At the real cell size (~1e-44) everything is lattice.
- Electrons (or neutrinos) are the focus; protons a while away. The spheromak is the trusted geometry and its DECAY is
  the near-term relation test (10-22). Every neutron is embedded in larger-scale disturbances (10-24).
- Structure = substrate depression (cells contract near it); translating structure is spongey, prefers its default;
  substrate is not a fluid. Any emitter is a receiver, vacuum is not; offload because capacity is finite; valve is
  not local (aggregate property); radiation contextual; single-cell events never happen; no spinors, no periodic
  systems, no vectors, no pi, no time (rate from cell size). One model, one rule. Loss must come out of the rule.
- Decay needs real offload channels: radiation, resistive, Landau damping. References (MHD < hybrid PHARE < PIC) are
  what DST's own dynamics must reproduce.
- (10-25) Depletion is macroscopic: a structure IS a depression; gravity / EM / strong / weak potentials are
  reifications of one depression profile (output, never input). Density is downstream of quantization (jigsaw: size
  is what fits); structures are forced into the fit by external pinching (CME / reconnection). hbar is a convenience,
  h = discrete step of photonic offload; standard values are signposts. Pressure mediation makes circles / symmetry.
  Rule requirements (hold, deform under pressure, translate as a product) are not selection tests. Nothing physical
  fits on a simulable discrete lattice (photon ~5e37 cells / wavelength): the scale gap is bridged by axiom; discrete
  runs give rule-level relations only. Use counts / relations, not "dimensionless".
- (11-04) No fields, only the substrate (fields are an abstraction); every particle, atoms included, is a bound
  state on it -> no 'field-only body' vs 'material' split.
- (11-03) Theory = signposts for truth, solid for invalidation (doc/dst_versus.md; Bell decisive). No probability
  spaces / amplitudes / abstract spaces. Observation is interpreted, it does not imply cause (list the readings an
  outcome allows). Symmetry (polarisation) is a product of competing pressures: never seed the symmetric shape.
  Photons ride ON a magnetically settled background (solar system; inside a planet, Earth's field too). Emission is
  a GLOBAL event: emitter and absorber at mutual magnetic / gravitational equilibrium already carry how much each can
  offload (quantum relational); lightning = the macro case (breakdown, forced path), a photon = small offload on a
  pre-built path. Merging can be legitimate (dendrite short circuit) but dendrites may need atoms, not a pure field
  automaton. Neutrino = smaller photon? (if a packet could be smaller it would be; not provable yet).

Rule and what it does:
- (11-09, Philip) CARTAN IS NOT LEGAL: it is incompressible Euler imported (no such thing as incompressible: an
  asymptote, compress until push-back exceeds push). Everything below that used step() + Cartan is about Cartan, not DST;
  what moves circulation in the rule is open again (exp/2026/11/09/status.md).
- The rule is step() (u / phi, face sigma, energy H = 1/2 sum u^2 + 1/2 sum phi^2 exact) + Cartan transport
  (-i_v(da) per Delaunay triangle, kappa 1; continuous.cartan_transport, alat.step_cartan). Energy-exact, Kelvin-exact,
  Lie-Poisson: circulation is a continuous label, topology frozen in the continuum limit -> quantization / offload /
  reconnection need a further relation. Lie commutator, A x B carry, tick phase, Villain, walk, step_cross: all dead.
- Every smooth coupling tried only reshapes or steers, never transfers (csig, vcap, set_cell_capacity, sigma well;
  exp/2026/10/{19,20}). Ring pairs: no transfer; R 20 tilted pair: no clean reconnection, grid-scale tangle (10-21).
- (11-09) EVERYTHING IN THIS BULLET IS step() + CARTAN -> about Cartan, not DST (kept as a record of what Cartan does).
- Spheromak (exp/2026/10/22/spheromak, seed converged at a ~9-16): MHD reference (equilibrium/reference/spectral.py)
  = held + self-similar Taylor decay 2 eta lam^2 (amplitude-free), helicity outlasts energy (exp/2026/10/24/ref).
  (11-08, exp/2026/11/08/cascade) Under step() + Cartan the cell-scale part (anchor_residual share) grows
  EXPONENTIALLY FROM t 0 out of the seed's 1e-4 sampling residue at ~0.16 v / h (no threshold: '3.4 turnovers' = when
  it becomes visible), saturates ~0.29. Coarse-grained (curl-curl smoothed) the body DECAYS IN PLACE keeping its shape
  (overlap 0.98 at 60 % energy; total energy kept: it goes to cell-scale content = unobservable at 1e-44), peak rate
  ~1.1-1.3 e-folds per turnover a / v (a 9 / a 20), exactly halved at half amplitude (Cartan is quadratic: phi -> c phi,
  t -> t / c). = purely IDEAL decay (Euler-like self-advection), no amplitude-free / resistive-like part; helicity
  decays FASTER than energy (late ratio 1.4-2.3, worse at a 20), inverted vs MHD / lab. -> missing: an amplitude-free
  exit, and something that stops / slows the ideal transfer to cell scale (or the helicity ordering stays wrong).
- Hand-off of unmatched loop content (exp/2026/10/24/handoff): local store holds a 6 at p 3 but weakens as (ell/a)^6;
  a travelling Maxwell partner only carries the near field -> outflow is two-stage (fast local oscillation, then emission).
- Closed spheromaks / n-nbar pairs: zero content beyond ~1.5 a, no interaction without contact, no independent push.
  Candidate tail: the depression. Contextual hold built (alat.set_cartan_hold; exp/2026/10/22/depression/
  contextual_hold.md), kappa source not wired; L sign open with Philip.
- (11-03) step() alone: 12 of 16 dof per BCC unit cell static, 2 scalar u branches propagate; a background superposes
  (no effect). step_cartan: a closed box thermalises to equal energy per mode at the grid scale (UV catastrophe =
  the spheromak cascade: one requirement, a stop at structure scale; offload quantum the candidate); a settled
  circulation makes a passing pulse anisotropic (~B^2, sense-blind, two-fold) and converts part into circulation
  (~B, handed).
- Embedded spheromak (10-25): an ambient does not stop the cascade (any frac / orientation); in MHD a strong aligned
  ambient absorbs the body, opposed persists. Embedding is context, not a cure.
- TOY, NOT EVIDENCE (11-08, Philip): the 'self-held depression body' (exp/2026/10/25/jigsaw/ring2.py, pair.py
  10-25..11-08) is a ring of step() content plus a carrier P (P_tt = lap P + source, d = kappa P, depth picked) that
  was ADDED BY HAND -- not the rule, not anything known to exist. Its results (holds if the core is slow, deform /
  translate, counts 3-5, mass scaling, pairs bridge / approach, approach onset ~t 150, flipped-sign source does not
  hold) are facts about that construct only; do not cite them as physics or chase them further. The only takeaway:
  a slow enough core traps waves (generic wave fact). Track parked.

Exact facts worth remembering (detail in doc/dst_algo.md):
- step() alone freezes all divergence-free phi; E_full = 0; uniform lam floor = exact time relabelling; with
  state-dependent sigma use implicit midpoint; in a linear rule domain and ambient superpose exactly.
- Charge = div phi = -u_t; static exterior only with the core held; vacuum cannot make charge; |phi|^2 drag cannot.
- Every pre-09-24 electron IC is neutral; every Taylor k=4.493 IC is closed; hop-path rings never circular.
- Weights: discrete 3:1, continuous iso 2:1, legacy 9:1 (pre-09-28 numbers are legacy-weights values).

Gotchas: ring centred on a lattice symmetry puts one face at exactly pi -> offset centre by (0.13, 0.07, 0.11);
zoo.py resumes from a leftover *_ckpt.npz; vill.py needs PYTHONPATH=repo root; sponge runs fake halos (use periodic
boxes for exterior profiles); measure the rule's own conserved energy, not a gradient proxy.; time C++ with the -g 0 build (the -gW 9 build is
numerically identical but slower); check kernel changes with exp/2026/11/03/mca/snap.py compare; continuous.spacing_of
was wrong in periodic boxes before 11-03 (wrapped face -> h ~ box size); BCC spectra need one FFT per sublattice
(one grid for both aliases); ring2.py / pair.py --kfix runs away (fixed coupling, needs a source / clip rethink).

Session index (full text in exp/2026/MM/DD/status.md): 11-08 toy parked, spheromak decay = ideal (v / a), helicity inverted | 11-06 pair box check (no box effect), restart |
11-04 pair regime map (separation, opposite senses) |
11-03 perf pass + phlop scopes, dst_versus (Bell decisive),
canal timing, A3 boxes (Cartan = UV catastrophe), settled background, pair bridge, emission / photon discussion | 10-12..10-24 Cartan transport, valve / capacity / receivers,
spheromak cascade, reference ladder, hand-off, pairs, embedding | 10-01..10-11 step() frozen, walk era (all
walk-scoped, dropped 10-08), lie commutator | 09-24..09-30 electron falsification, continuous model level 0, DEC curl,
Hodge star, BCs rebuilt, lam-floor relabelling | 09-16..09-22 electron IC audits (vecpot IC = spheromak-equivalent) |
08-08..09-08 AMR L8 (resolution, not depth), step_cross gates (closed), E/B / Poynting / (phi, psi) (closed).

## SESSION 2026-11-09 — helicity: lost, not handed down, and Cartan itself destroys it; box size irrelevant; Philip: Cartan is not legal (incompressible = an asymptote, not a law); first pinch test (no Cartan) (exp/2026/11/09/status.md)

1. helicity/fine_hel.py (LS, 3 vector DOF) on the 11-08 dumps: total helicity falls with the coarse one, no scale band
   gains net helicity, fine content is helicity-NEUTRAL (+ / - equal, ~0.09 a 9, ~0.20 a 20).
2. Box check, a 9 in P 192 (6 diameters vs 3): identical to P 96 to 3-4 digits through t 160, same end state; only the
   wave share is 20-40 % higher (no re-entry). Memory: go larger if needed (feedback_go_larger_box).
3. helicity/whitney.py: all-DOF helicity (Whitney a ^ da over the 6 Delaunay tets per cell; validated: exact volume,
   uniform curl exact, gauge 1e-15, symmetric, = LS x cell volume to 3 %). Falls like LS (0.112 at t 200); the 4 hidden
   DOF never hold > ~0.04; Cartan's instantaneous dH/dt matches the decline from t 0 -> Cartan destroys helicity (no
   discrete da ^ i_v da = 0). Fine content handed both ways, +-0.65 of H_0, net 0.
4. Philip: Cartan not legal under DST (STATE 'Rule'); memory feedback_no_incompressible_cartan_illegal. Philip on holding:
   plasmoids are compressed structures with a 'surface tension' -- if they could diffuse they would; something spinning
   pushing toward the centre also prevents diffusion.
5. pinch/pinch.py: step() + face rate s(u) = exp(-beta u / um) + one-sided asymptote e(u) (Hamiltonian, energy-exact;
   phi changes by gradients only -> loop sums frozen, circulation cannot move). a 6, P 96: beta > 0 -> static pinch
   (core compresses into the circulation's own volume, smooth, no edge); pushing the core to 0.77 um flattens and widens
   it (no surface); beta < 0 expands and runs away (soft side of the chosen e). um 0.08 needs an implicit integrator.
   A convex limit cannot make an edge: needs a favoured-compression range (non-convex) or an outer push (ambient).

## SESSION 2026-11-08 — ring + P track declared a toy and parked (Philip); spheromak cascade characterised: exponential from t 0, coarse body decays in place at an ideal (v / a) rate, helicity inverted (exp/2026/11/08/status.md)

1. 1b (+-30 to T 600) dropped: no-bridge says nothing, a bridge only an onset; same-sign only attracts.
   approach/fit.py (logs): no approach before t ~150 at +-24 / +-30, exponent unresolved. expand/pair.py: a flipped-
   sign (expansion) source does not hold (defocusing). All on the toy -> not evidence.
2. Philip: results of something we don't think exists or works can't be trusted -> the ring + P track (10-25 on) is a
   hand-built toy; parked (STATE 'TOY, NOT EVIDENCE'; memory feedback_no_studying_hand_built_toys).
3. Philip: the substrate geometry is unknown (one set, overlapping, staggered / flush) and nothing says it is smooth at
   1e-44 -> the '4 non-vector face DOF per cell' are only this lattice's label for cell-scale content; the observable
   is the coarse-grained body.
4. cascade/cascade.py (= 10-22 sph.py + non-vector share + phi dumps) and coarse.py (curl-curl smoothing, E / overlap /
   helicity vs t 0). Runs: a 9 vmax 0.3 T 250, a 20 t 200 -> 450 (restart of the 10-22 state), a 9 vmax 0.15 T 500;
   dumps /bk/dst/2026/11/08/cascade/{a9,a20,a9_v15} (a20 ~2.5 GB).
     run          fine growth      coarse peak decay   per turnover   E_s 0.5 at   late H/E rate
     a 9  v 0.3   x1.62 / 10       0.036               1.07           t ~161       1.4-1.5
     a 20 v 0.3   ~0.05 / time     0.019               1.27           t ~310       2.3
     a 9  v 0.15  x1.79 / 20       0.019               1.13           t ~320       1.8
   Reading in STATE (Spheromak). The 'cell-set' growth reading made mid-session was wrong (v / h, not cell-set).
5. Gotcha hit twice: sph.py / cascade.py resume from SEED_ckpt.npz -> copy a fresh seed, delete the ckpt.

## SESSION 2026-11-06 — pair: box check +-24 at n 144 (no box effect; 11-04 non-monotonic approach = snapshot artefact); pair.py restart / checkpoint; +-30 continued to t 375 (exp/2026/11/06/status.md)

1. pair.py (exp/2026/11/06/box/) + --restart / --t0 / --eratio / --ckpt: bit-exact vs a straight run; the runs are
   deterministic (from-t-0 rerun matched the 11-04 log at t 200).
2. +-24 at n 144 (pair + 2 solos, T 300). Centroids wobble ~+-1 within 10 time units, so the 11-04 single-snapshot
   'toward' values were phase noise. Time-averaged pair - solo toward over t 250-300: +-24 n 128 +0.97, +-24 n 144
   +1.00, +-30 n 144 +0.49 -> no box effect, approach monotonic in separation (~2x from 30 to 24). Still no bridge by
   t 300 at +-24 (middle resid 0.000; u^2 middle x71 = content arriving, not depletion).
3. +-30 continued from the 11-04 t 300 state: still approaching (avg x +0.49 -> +0.88 over t 300-360), E/E0 0.9345 ->
   0.9069 by t 360, d max 1.08 -> 1.62 (deepening). Stopped at t 375 for shutdown; resume recipe in status.md.
   Lesson: compare time-averaged / fitted centroids, never one snapshot.

## SESSION 2026-11-04 — pair regime map: separation +-24 / +-30 and opposite senses at +-18 (exp/2026/11/04/pair/status.md); Philip: no fields, only substrate; 4-core cap

1. Philip: there are no fields, only the substrate; all particles (atoms too) are bound states on it -> the 11-03 hold
   on the pair scan ("field-only merging, dendrites may need atoms") rested on a false split. Merging is a question of
   what the bound states / rule do (scale, count, context). Scan run.
2. pair.py (exp/2026/11/04/pair/) = 11-03 + --senses (per-ring sense, flips that ring's azimuth). Same body (R 8 a 3.2
   q 4 depth 0.8 --src local dmax 0.9 dt 0.05 T 300), each pair with its own solos at the same n (box effect ~2x).
   run.sh (8 single-thread jobs, pinned to 4 cores from ~13:00 -> 2.3-3.3 h each); compare.py PAIR SOLO_M SOLO_P CX.
   Saves /bk/dst/2026/11/04/pair/*.npz (full state + x-axis d / u^2 at t 100 / 200 / 300).
3. Results at t 300 (middle resid = d_pair - solo sum at |x| <= 4; toward / sideways per body; solos drift <= 0.2 /
   ~+0.3 same sign for both senses = box / lattice baseline; resid ~-0.15 in the rings at t 100 = kappa -7 % confound):
     same +-18 (11-03)  +0.43 saturated, u^2 x490   toward +2.85   sideways +2.65 / -2.65 (pair turns)
     same +-24 (n 128)  +0.05, u^2 x0.8             toward +0.50   sideways +0.64 / -0.64
     same +-30 (n 144)  +0.02, u^2 x52              toward +1.48   sideways +0.67 / -0.67
     opp  +-18 (n 112)  +0.15, u^2 x66              toward +2.68   sideways +2.08 / +2.08 (co-drift, no turn)
   No bridge at +-24 / +-30 within T 300, but no settled pair either: both approach, accelerating over t 200-300,
   and +-30 approaches MORE than +-24 (not monotonic in separation) -> two readings: onset delayed (T 300 too short)
   or box size (n 144 vs 128) matters. Opposite senses: same approach, 1/3 the middle deepening, the pair translates
   instead of turning: sideways motion follows the senses (10-25 gradient-drift / Magnus-like idea). All runs lose
   5-7 % energy at the rim by t 300 (solos too); d max falls t 200 -> 300 everywhere.
4. Runtime: 4 cores (cores 0-3 + HT siblings) for the session (Philip); 8 jobs on 4 cores ran ~2.5-3x slower than
   1 core each, own ETAs were unreliable early on.

## SESSION 2026-10-25 — now / next tidied (full sessions moved to exp/YYYY/MM/DD/{now,next}.md, doc/ updated, doc/dst_next.md archived to exp/2026/07/12/); embed matrix finished; discrete scope (rule-level relations only; unit-defect scattering); depletion = the macroscopic depression; first self-held body (propagating depletion carrier, local-tick flux source) meeting hold / deform / translate

1. Embedded spheromak, full matrix (exp/2026/10/24/embed/, a 6 P 96 T 150 = 7.5 turnovers; DST saves
   /bk/dst/2026/10/24/embed; status.md there). End values (loops/loops0, inside r < a, overlap with seed, cen z):
   DST  none        33  0.70  0.14  +1.1        MHD none        inside 0.95 overlap 0.99 (Ebody 0.74, Taylor decay)
   DST  0.1 align   38  0.71  0.07  +3.4        MHD 0.1 align   0.72 / 0.73
   DST  0.1 opp     30  0.63  0.15  +1.3        MHD 0.1 opp     0.95 / 0.95
   DST  0.1 tilt    37  0.65  0.02  +2.7        MHD 0.1 tilt    0.72 / 0.43 (cen x -0.9)
   DST  0.3 align   53  0.49 -0.01  +7.8        MHD 0.3 align   0.03 / 0.06 (body dissolves into the ambient)
   DST  0.3 opp     34  0.49  0.10  +6.5        MHD 0.3 opp     0.81 / 0.73
   DST  0.3 tilt    61  0.51 -0.02  +7.0        MHD 0.3 tilt    0.03 / 0.03
   Reading: the ambient never stops the DST cascade (loop growth x30-61 at every frac / orientation; stronger ambient =
   more, aligned / tilt worst, opposed least) and the DST body is carried along the ambient (+z, ~ frac). MHD: the
   ambient does not hold either -- an aligned / tilted body merges into a frac 0.3 ambient (relaxes into the box-scale
   Beltrami state), only the opposed body persists; no bulk drift in MHD (tilt shifts ~1 in x). Same ordering in both
   (opposed most robust), but MHD loses the body by merging, DST by cascading in place. The ambient is not a holding
   agent in either; embedding is a context for interaction, not a cure for the cascade.

2. Discrete model focus (Philip): coarse ladder needs a discrete reference first (L = 1, then L = 3 = 27 cells, 9, ...).
   Discrete cell size relative to void is what separates Earth / Moon GR; size changes are discrete (units of the min
   cell), relational (junction event between two cells), a result of substrate depletion (saturation only an idea;
   contraction + expansion = two kinds of gradient). Scope: photons, wave dispersal, maybe neutrinos -- nothing
   macroscopic (spheromak ~1e28 cells across). Memories project_discrete_size_quantized, project_discrete_model_scope.
   Fixed cell count -> sizes set the embedding; with one cell per local tick the existing step() + face sigma is the
   graph rule and the GR factor 2 is algebraic (no lensing run needed). Optical-clock redshift is continuous to 1e-19
   over mm -> units cannot form terraces, they must be sparse scattered defects (dithering).
3. Unit defect scattering (exp/2026/10/25/defect/status.md): sigma_s ~ (1 / S)^2 exactly; net-depletion cell ~ k^4,
   conserved junction pair ~ k^6 (Rayleigh). At real scale (k ~ 1e-37 for visible light) the dithered substrate is
   transparent; only the mean size survives coarse-graining at long wavelength; chromatic -> not tired light.
4. Walk beta clock: unused outside TestWalkClock -> comment added (walk.py, adaptive_lattice.hpp, test_walk.py), removal pending.

5. Philip's frame (10-25): depletion is macroscopic -- a structure is a depression, potentials (gravity, EM, strong,
   weak) are reifications of one depression profile; density is downstream of quantization (jigsaw: protons are the
   size they are because they have to be), structures forced into the fit by external pinching (CME / reconnection);
   hbar a convenience, h = discrete step of photonic offload; pressure mediation makes circles / symmetry (outcome,
   never input); travelling flux on a circle is balanced for any count; deformation / translation are rule
   REQUIREMENTS, not selection tests. Signpost: r_p = 4 hbar / (m_p c) to 0.04 % (4 own periods per perimeter transit).
6. Jigsaw / hold tests (exp/2026/10/25/jigsaw/status.md v1-v9): local depletion responses (|throughput|, signed) hold
   nothing and keep lobe counts as fixed labels; a depletion CARRIED by a propagating field P (P_tt = lap P + source,
   ~1/r outside) with a self-consistent start holds content if the core is contracted enough (speed x ~0.1, cells ~1/3:
   "that is the depression"); sourced by background-tick throughput it starves itself (slow core = little flux),
   sourced by flux per LOCAL tick (or energy density) it holds permanently (~95 % to t 400). On that hold: every
   whole count 3 / 4 / 5 holds (no preference for 4), a 4.5 seed never settles; a ramped pressure gradient deforms it
   slightly, it recovers shape, accelerates toward contracted cells ~linearly in the push and keeps moving after the
   push (inertia-like), plus a sideways drift mirroring with circulation sense.
7. Mass scaling (jigsaw status.md v10): same gradient on R 8 / 10 / 13 (content x4) and counts 3 / 4 / 5 -> end-of-push
   displacement +2.5..+4.3, no trend with size or count = consistent with universal fall (gravity-like), not a
   force on a mass; resolution only ~+-50 % (box effect ~2x, trajectories wobble). Fixed-coupling A scan invalid
   (P ran away, clip saturated everything).
