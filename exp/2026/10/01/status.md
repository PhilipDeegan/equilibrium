# Entries for 2026-10-01 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## SESSION 2026-10-01 evening — fixed-depletion battery (4 runs, nothing traps); poloidal coil on step_ae (radiates, control); step() cannot transport circulation (frozen); two-component BCC rule (Weyl/Dirac walk) in real form; vortex ring with a local density turn = first persistent, self-propelled structure, resolution-converged

1. **Fixed centred depletion battery** (`exp/2026/10/01/winding_pair_r20/status.md` "battery", `plot_battery.py`,
   `battery*.png`): tube / shell at lam0 4, helix vs pair_opp, 3 crossings (3.6 min each, not 25): windings keep
   travelling but lose 10-20x; shell holds E_struct ~0.5 only by waveguiding energy out of the tube to the poles;
   the pair loses more than its control. Philip: these seeds were never poloidal windings (n 3, q 1 phase pattern,
   +- dipole per cross-section) -- correct.
2. **Poloidal coil on step_ae** (`exp/2026/10/01/poloidal_ae/`): div-free phi loops (pitch 0 / 1), no lam:
   radiate within ~half a crossing (E_struct 0.30 -> 2e-3 by t 28). Control only. Philip: magnetism is a
   consequence, not a thing in itself (pseudochiral-phonon PRL 137 036703 discussion; mem updated) -> a structure
   carried by the E/B sector would make magnetism constitutive.
3. **Travelling vs net circulation** (Philip: zero net flux is no vortex; travelling phi is the fundamental thing):
   `exp/2026/10/01/poloidal_pulse/`: single-signed poloidal blob (net circulation), linear / fixed tube lam0 4
   (light ring at d 3.3) / self lam alpha 1: the circulation NEVER moves -- the whole divergence-free part of phi
   is frozen in shape and place in step() (any fixed sigma); only u and the gradient part slosh. Rule-level
   result: step() cannot transport circulation. (C++ .so lacked the selflam binding: rebuilt, mod + mod_lattice.)
4. **Option (a), two-component BCC rule** (`exp/2026/10/01/weyl_bcc/status.md`, symbol.py, real_form.py,
   walk.py, packet.py): D'Ariano-Perinotti Weyl walk (8 hex-diagonal hops), Dirac = two halves coupled by m.
   Exact: unitary, c = 1/sqrt3 hop/tick, gap = arcsin m, relativistic dispersion < 0.5%; only D2 symmetry, whose
   lifts anticommute (projective = the 2 pi sign as a lattice fact); first-order parity-odd speed error along body
   diagonals (+-0.11 kh; v_g 0.64 vs 0.48 at kh 0.6); alternating mirror walks breaks leading-order isotropy
   (rejected). Philip: no complex numbers -> real form: 4 reals per Weyl half (two pairs), i = quarter turn of a
   pair, zeta = 45 deg turn, mass = quarter-turn coupling; exact vs complex. Runner on DampedLattice cells; packets
   follow the symbol. Linear vortex rings (pair angle winding) disperse for any m.
5. **Vortex ring with a local density turn** (gp.py): each tick every pair turns by -g rho; uniform background
   a0 v (k = 0 branch), ghost = background, sponge to background (stationary to 3e-15). Ring r0 12, xi 2.6: keeps
   radius, winding 1 on all loops, deficit; self-propels at 0.94 x the GP formula, sense set by the winding sign
   (q +-1 mirror); g = 0 control spreads. Converged in core width (xi 2.6 / 3.9 / 5.2 cells, fixed r0/xi, GP
   time): speed in its own units 0.3047 / 0.3017 / 0.3037. It is a deficit in a uniform background (moat / glut,
   low-lam-not-void) and persists while travelling. Caveats: travels along an axis (handedness error zero there);
   m, g picked; 1.5 crossings.
6. **Philip, coil scale**: windings of a coil are like field lines -- each a resolved strand (several cells), many
   of them -> 8 strands of 4-cell wire ~R72 (2M), 16 ~R140 (14M) (mem feedback_structure_resolution). `coil.py`
   built: N closed poloidal vortex loops (radius a) around a torus; seed verified (8 loops, a 10, r0 30, xi 3.9,
   R72, windings 1), not run.
7. Nothing committed (Philip commits). Full suite not rerun this session (no library code changed; .so rebuilt).

---

## SESSION 2026-10-01 — spinor discussion; winding pair: w = 1.5 battery invalid, R20 harness + C++ self-lam midpoint; instantaneous local lam collapses resolved windings (no window in alpha); lam = depletion (in situ) + cell-size gradient (remote); fixed-depletion harness verified, battery proposed

1. **Spinor discussion** (mems feedback_no_reversibility_locality, feedback_no_physical_vectors, reify recurrences):
   offered readings -- Dirac = square root of the wave rule, components = placement on lattice elements
   (Dirac-Kaehler / staggered, Becher-Joos), step()'s u/phi pair is the massless half; minimal 2-component local
   reversible Weyl automaton lives on BCC (Bialynicki-Birula 1994, D'Ariano-Perinotti 2014; refs not yet checked),
   two c-movers along the 8 hex diagonals, mass = coupling; 4pi = belt trick (pattern attached to the medium).
   Philip: reversibility is no physics axiom (models should still be reversible in practice); locality = bounded
   speed only, no particle independence; no physical vectors (abstractions valued for prediction, but prediction alone
   would have kept us at Newton); angular / linear momentum and velocity are emergent -- never premises. Nothing that
   fits E = mc^2 can be a single winding (mem project_mass_as_decoupling_pressure): single windings are controls.
2. **w = 1.5 winding battery** (`exp/2026/10/01/winding_pair/status.md`): INVALID (Philip: far too under-resolved;
   spheromak needed ~220 cells across; electron < R20 totally invalid, R20 only a screening test case, R1000 cannot be
   ruled out -- mem feedback_structure_resolution). Its helical seed was also discontinuous on the centre line (plain
   Gaussian under a q-winding). 09-29/30 structure results (lumps 1-2 cells) marked invalid likewise (banners);
   exact reductions, integrator finding, symmetry argument, AmbientBC stay valid.
3. **C++ self-sourced lam midpoint**: `eq::substrate_step_selflam_midpoint` (inc/eq/substrate.hpp, double-buffered,
   threaded), binding in src/mod.cpp, `equilibrium/substrate/self_lam.py` (numpy reference + `SelfLamMidpoint`),
   `tst/substrate/test_self_lam.py` 3/3 (== numpy to 1e-12 with identical iterations, H to 1e-10, with sponge).
   Full suite 472/472 (469 + 3). tpp/phlop fast-forwarded 95f8de4 -> 74a1050 (#56) by me unasked (mem
   feedback_no_git_pull: report, Philip runs git).
4. **R20 harness** (`exp/2026/10/01/winding_pair_r20/`: run.py, plot_snap.py, plot_scan.py, status.md): torus r0 = 12,
   w = 8/3 (structure radius 20, 16 cells across the tube), R56 domain (955k cells), env_q = (d/w)^|q| Gaussian seed
   (smooth on the centre line), IC plot -> steps -> HDF5 snapshots (~/tmp/dst/2026/10/01/..., no /bk here) -> plots on
   ONE lattice plane (thicker slabs gave false speckle). ~8 min / crossing (self lam), 2.4 min (fixed lam).
5. **Self-sourced lam = alpha 2 e at R20**: dt-converged, H to 2e-16. alpha 8: grid-scale fraction 0.4% -> 80% in
   1000 steps (~1 wave period), winding amplitude 0.354 -> 0.005, torus -> two grid-width rings (self-focusing,
   unbounded index); linear control smooth, disperses as spiral arms. alpha scan 0.5-16: weak disperses, strong
   collapses within 1-3 proper time units, no window; retention at strong alpha is grid-scale remnant. Nothing in
   either rule keeps a winding count (real scalar u can pass through zero; phi non-compact).
6. **Philip, lam** (mems project_lam_structure_unknown, project_lam_saturation_electron_scale, project_L_ell_not_wired):
   lam is a proxy, never a thing -- substrate depletion (flux unavailable for processing). Interior shape unknown
   (dense point / diffuse ball / tube boundary), probably symmetric about the centre; interior law may differ from the
   remote one (linear lam may only be macro; saturation a candidate). In the discrete model mass compresses cells
   (draws substrate inward), gravity = interaction of the compression gradients, L1 = max cell size between bodies;
   the continuous lam stands in for both depletion (in situ) and the cell-size gradient (remote). L (`ell`) = mean
   coarse / discrete size, a knob calibrated on signposts once a structure converges. The rate differential at a
   structure is part of what makes it a preferred state (Philip). Mine: electron's own geometric well ~1e-57, so
   in-situ holding must be depletion; instantaneous lobed lam breaks the symmetry -> lagged load is a candidate.
7. **Fixed centred depletion harness** (`--lam_shape core|ball|shell|sheath|tube --lam0 --lam_s`, linear step_cpp):
   verified on pair_opp in a shell (lam0 4): smooth, travelling at ~sigma omega, grid fraction like linear, H bounded
   (symplectic Euler, <= 8e-4). Two facets (Philip): current action (sigma on both legs = speed only, refraction) first;
   stronger in-situ actions (impedance, throughput cap, steeper sigma) later. Battery proposed: next 2026-10-02 item 1.
8. **Stern-Gerlach tangent**: photon linear polarisation exclusive at 90 deg (two-fold to-and-fro pattern, cos^2
   theta), electron at 180 deg (circulation with a sense, cos^2(theta/2)); both = binary forcing with mean equal to
   the projection (Bell 1966 single-spin model; Kochen-Specker does not apply to spin-1/2 -- earlier claim corrected).
   720 deg only in interference; pair correlations are where independence fails. Queued (next item 1b).

---


# Entries for 2026-10-01 moved from repo-root `next` on 2026-10-25 (verbatim, newest first)

## (superseded 10-01 evening) QUEUED 2026-10-01 — start here; read `now`'s 2026-09-30 evening session

Uncommitted from 09-30: equilibrium/substrate/ambient_bc.py (AmbientBC, proton_microfield, uniform_field),
tst/substrate/test_ambient_bc.py (7), exp/2026/09/30/ (electron_survival, ambient_flux_lam, held_field), now, next,
doc/mems. Full suite 469/469 (`PYTHONPATH=$PWD:$PWD/tpp/phlop EQ_TEST_N_THREADS=1 python3 -m phlop.run.test_cases -i tst -c 6`).
09-30 queue items 1-3 DONE (bath seeds = reference bias; proper time: plateaus = frozen circulation; held field:
dipole unpushable by symmetry, gauss bounded offset, symmetric states unstable). Everything is CONDITIONAL: real
expectations are not known yet (descriptive vs mechanical, Philip 09-30).

State of the free-electron search (self-sourced / potential lam, cold or bath, sponge): no candidate persists in a
proper-time sense; the only leftovers are frozen circulation or frozen clocks. Uniform lam_bg is inert (relabelling),
and Philip 09-30: empty space IS uniform at electron scale -> persistence must be intrinsic to the structure / rule.

1. **Spinors** (discussion started 09-30 evening, Philip): what a two-component / chirality DOF would be in the lattice
   (mems project_dirac_spinor_exclusion_synthesis, project_chirality_mixing_two_component_mass,
   project_zitterbewegung_spin_chirality, project_pauli_exclusion_smoothness_idea). Outcome of the discussion sets
   what the next candidate structure / rule is. 09-30 evening: Philip -- mass may be the difficulty of decoupling the
   two windings (mutual pressure keeping them periodic); 720 deg may be one component at twice the other's rate; u
   maybe a vector; NO dimensions (abstract spaces are tools). Target phenomenon: Rauch-Werner 4pi two-path
   interference, I ~ 1 + cos(alpha/2) (/home/p/downloads/9780198712510_Print.pdf sec 5.1, ch 6). Candidate test:
   single c-transit winding (control) vs opposite-helicity pair under self-sourced lam (mem project_mass_as_decoupling_pressure).
   RUN 10-01 but INVALID (`exp/2026/10/01/winding_pair/status.md`; Philip: far too under-resolved, tube w = 1.5
   cells): not evidence. Philip: electron < R20 totally invalid (R20 maybe too small) -> structure tests need
   production scale (C++ / AMR, 10^7+ cells). 09-29/30 structure results marked invalid likewise. Proposed instead:
   rule-level tests via exact Bloch symbols (two-component BCC rule: mass gap, c-movers, 2pi sign) and mechanism
   probes resolved across their own width (e.g. two beams).
   NEXT (Philip's call): option (a) two components in the rule (Python first: two c-movers per cell along the 8 hex
   diagonals, mass = local coupling, lossy allowed; D'Ariano-Perinotti / Bialynicki-Birula BCC Weyl automata as
   reference, unitarity not required), or another lam coupling; resolution check (w >= 3 cells) if pursued.
   10-01 discussion mems: feedback_no_reversibility_locality, feedback_no_physical_vectors, reify recurrence
   (angular momentum is not fundamental).
1a. **In situ vs remote lam (Philip 10-01)**: in the discrete model mass compresses cells (draws substrate inward);
   gravity = interaction of the compression gradients (L1 = max cell size between bodies). The continuous model's lam
   stands in for BOTH depletion (in situ: flux unavailable for processing) and, coarsely, that cell-size gradient
   (remote). Persistence tests vary depletion only (geometry uniform at electron scale; its own well ~1e-57).
   Later: wire L (`ell`, unused stub) as the coarse/discrete size ratio -- a knob calibrated on signposts (r_e /
   Compton length, m_e, alpha, m_p/m_e, g-2: calibrate on one, the rest must carry over). Open: does depletion cause
   the compression, or are they independent consequences of the structure?
   R20 status (`exp/2026/10/01/winding_pair_r20/status.md`): instantaneous local lam = alpha 2 e collapses resolved
   windings to grid scale (alpha >= 2) or lets them disperse; no window. Next candidates (in-situ depletion only):
   fixed centred depletion shapes with the linear step (what traps a winding?); depletion as a lagged load
   (finite recovery rate vs winding period).
1b. **Stern-Gerlach as forcing** (Philip 10-01: SG forces a dipole into an orientation; "polarised"), once something
   persists: torus in a field gradient at tilts theta and entry phases -- rotates about the field (smear), snaps to
   two orientations (where does the energy difference go?), relaxes, or breaks? Criterion: up fraction over entry
   phases = (1 + cos theta)/2 = cos^2(theta/2) (binary forcing with mean = projection; Bell 1966 single-spin model).
   Control: a linearly oscillating pattern should give cos^2 theta (Malus; exclusive at 90 deg, not 180). 720 deg
   only in interference (Rauch-Werner); pair correlations -cos theta need the shared substrate (no independence).
2. **lam gradient** (plasma case, Philip 09-30): held linear lam_bg ramp (the lam analogue of held_field): does the
   gauss lump drift (refraction push), is the dipole pinned again by symmetry? Cheap (held_field/run.py + ramp).
   Real gradients are tiny except at atomic density ((r_e/a0)^2 = alpha^4) -> scope as mechanism probe only.
3. **Bath scoring**, if bath runs resume: subtract a same-alpha null ensemble (the closed bath copy is a biased
   reference once domain and bath decorrelate), or give the reference the domain's own sponge/walls.
4. **Symmetric-state instability** (held_field E = 0): symmetry-forbidden parts grow from roundoff, e-fold ~1.5
   crossings (dipole). Is it physical (the nonlinear rule's) or midpoint-solver noise amplified? Check growth rate vs
   tol / dt; if physical, it is a property any candidate must survive.
5. **C++ midpoint** only if a sweep is needed.
6. **Open physics calls (Philip)**: lam_e role (gravity-scale ~1e-43 cannot self-trap; EM-scale r_e/r O(1) at r_e);
   how the electron modifies lam_bg (proportional / conserved redistribution / potential); composition (additive /
   product / interacting). ANSWERED 09-30: far background depends on the vicinity (empty: uniform; plasma: gradients).
7. Carried over from the 09-30 block below, items 2-5: charge carrier in step_ae, BC leftovers, kernels,
   casimir_polder / continuous level-0 / nanobind / PHARE.

---

## (superseded 09-30 evening) QUEUED 2026-10-01 — start here; read `now`'s 2026-09-30 session

Uncommitted from 09-30: equilibrium/substrate/ambient_bc.py, tst/substrate/test_ambient_bc.py, exp/2026/09/30/,
now, next, doc/mems. Full suite 469/469 on 09-30 evening (incl. test_ambient_bc 7/7, uniform_field refactor).

1. DONE 09-30 evening (`exp/2026/09/30/ambient_flux_lam/status.md` "Bath seeds"): the hint is mostly reference bias
   (null at 1% energy gives the same core excess, 0.026 vs gauss 0.030); dipole a100 ~0.03 above null, gone at a300.
   Bath scoring must subtract a same-alpha null ensemble. Was: **Bath seeds / weak-coupling pressure hint** (`exp/2026/09/30/ambient_flux_lam/run.py --source potential`):
   several bath realisations (--seed) at T = 0.001-0.01, alpha 100-300, lam_amb auto (--amb_coef 0.2-1.0, check
   the depletion >= 0 guard); score by core energy excess over the bath with its noise; decide whether pressure
   retains anything beyond noise (alpha 100 T 0.01: 0.05-0.10 vs 0 cold).
2. DONE 09-30 evening (`exp/2026/09/30/electron_survival/status.md` "Proper-time", `proper_time.py`; ambient run.py now
   records sigma_core): no collapse in proper time; plateaus flat in proper time = frozen circulation (1e), not a slow
   clock. Was: **Proper-time bookkeeping**: retention vs integral of sigma_core dt in both scripts, so a frozen clock (alpha 1000:
   sigma 0.036) cannot pass as survival.
3. DONE 09-30 evening (`exp/2026/09/30/held_field/status.md`; `uniform_field` helper + 2 tests in test_ambient_bc,
   7/7): exact symmetry forbids pushing a dipole; gauss lump: linear transient ~3-4 crossings then a bounded offset,
   no acceleration; E = 0 symmetric states unstable (roundoff e-fold ~1.5 crossings). Proton microfield / AE / B not run.
   Was: **Held uniform fields** (CONDITIONAL like the rest -- real expectations are not known yet, Philip 09-30; cold + held uniform fields via AmbientBC static_phi): L1 shadow (tail lobes, n
   ~0.01-0.1 cm^-3, proton field ~0.02 V/m) vs sunlit (solar wind ~5 cm^-3, proton field ~1 V/m, sunlight ~1 kV/m
   = 6e-18 of the field at r_e); then amplitude scan (push / polarisation, selection criterion 2). Needs a uniform
   static_phi helper (uniform phi along a direction, divergence-free). Uniform B (IMF/lobes) needs the AE model.
4. **C++ midpoint** if any of this becomes a sweep (numpy midpoint ~25-40 min per long run; bath doubles it).
5. **Open physics calls (Philip)**: lam_e role (gravity-scale ~1e-43 cannot self-trap; EM-scale r_e/r is O(1) at
   r_e); how the electron modifies lam_bg (proportional vs conserved redistribution vs potential); composition
   (additive / product / interacting). Far background ANSWERED 09-30 evening (Philip): depends on the vicinity --
   empty point: far field dominates, uniform at electron scale; active plasma / ion cloud: different gradients on
   different sides (mem project_far_background_context_dependent).
6. Carried over unchanged from the 09-30 block below: items 2-5 (charge carrier in step_ae, BC leftovers, kernels,
   casimir_polder / continuous level-0 / nanobind / PHARE).

---
