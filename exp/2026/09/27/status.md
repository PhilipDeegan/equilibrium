# Entries for 2026-09-27 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## SESSION 2026-09-27 — PHARE EMHD spheromak drifts along its axis (parked); two dispersion relations pinned by the exact Bloch symbol; DEC curl (step() is DEC with a 9:1 star) ported to C++ and made the default; FQHE / calibration discussion

1. **PHARE EMHD** (`exp/2026/09/27/phare_emhd/status.md`, evolution.py reads vtkhdf with h5py; pyphare
   can't plot 3D vtkhdf): unperturbed hybrid to t=10: no tilt, ions frozen, rigid drift along -z
   (0.22 = 15% of R), W_B -1.7%, force-free residual 1.1% -> 5.5%. Parked (Philip); next: flip B_phi.
   Hall MHD with V frozen = the same EMHD physics without particles (1/rho couples mass and Hall term).
2. **Dispersion** (`exp/2026/09/27/dispersion/status.md`): plane-wave packets, 3 h x 2 directions. u =
   exact leapfrog step() prediction; LS transverse -0.176 [100] / -0.120 [111] (kh)^2, 4-10x worse.
   symbol.py: exact Bloch symbol matches to 3 digits; 80-85% from the 2h-span LS gradient (-5/36),
   rest face->cell averaging (birefringent [110]); 4 of 7 face modes per k at omega=0. LS output is
   NOT discretely div-free (docstrings corrected).
3. **DEC** (`exp/2026/09/27/dec/status.md`, dec_symbol.py): circulation on the 12 Delaunay triangles
   per cell, Kelvin stars. curl(grad)=0 exactly, no fitted constant; W=3:1 is the geometric *1 but
   step() uses W^2 = 9:1 (step() = DEC d0 with a 9:1 star). "dec" = 9:1 + *2 = 5/2: exactly div-free
   in step()'s sum W, one star for both sectors; "dec_geo" = 3:1. Gapped extra modes (omega h/c 4.5-8):
   pointwise output error O(1) but evolved fields converge. Packets ~20x better; dipole source work
   err (numpy geo) 0.011 / 0.0026, (C++ dec) 0.0136 / 0.0036 at h = 0.5 / 0.25 vs LS 0.187 / 0.033.
   C++ set_curl('ls'|'dec'|'dec_geo'), curl_curl_dec; default switched to 'dec' (Philip). 32/32 tests.
   Memory lesson: SubstrateLattice(radius_cells=R) takes the physical radius (R/h -> 52 GB, killed).
4. **Discussion** (memories): FQHE -- "2D" is not real, say planar, and even planar only suppresses
   the third direction; electron indivisibility is flawed ontology, a measured fraction is what the
   probe couples to (rest shielded / elsewhere / held by the field, cf. Streda); absolute zero = frozen
   substrate (impossible); near absolute zero = the epsilon floor just above it (see `next`). Queued ideas in `next`: electron in a strong field from a
   near-absolute-zero IC, as a selection test for the persistent electron (needs lam = mass);
   macro-particle scaling (one structure = a mole) as part of calibrating the fundamental scale;
   method: approximate, calibrate, repeat / Feynman guess-compute-compare.

---


# Entries for 2026-09-27 moved from repo-root `next` on 2026-10-25 (verbatim, newest first)

## 2026-09-27 PHARE PARKED (Philip: come back later). Unperturbed hybrid run done to t=10:
persists, no tilt, rigid self-drift along -z (see `exp/2026/09/27/phare_emhd/status.md`). Next PHARE
step when resumed: flip B_phi only (drift reverses if helicity-driven). Item 3 below is superseded.

## IDEA 2026-09-27 (Philip): electron in a strong field from a near-absolute-zero IC
Spawn an IC "near absolute zero" (ANSWERED 09-27, see top block: absolute zero = frozen substrate, an
impossible state; near absolute zero = the epsilon floor state just above it, uniform lam = epsilon, no
extra disturbance), impose a
strong uniform field (A = B x r / 2 sampled on faces; DEC transverse sector makes it trustworthy), place an
electron, watch: intact / cyclotron / breaks into sub-structures (how many, what share of circulation,
vs field strength -- rational ratios recurring = signal), and the full budget (field, outward, medium).
Context: FQHE discussion; electron indivisibility is flawed ontology
(mem project_fractional_charge_is_partial_measurement). Blocked on: a persistent electron structure
(existing ICs wrong topology; free electron needs a nonlinearity); a scale for "strong" (sigma uniform).
No slab needed (Philip): a single electron is effectively planar -- the field picks the plane (circulation,
orbit, moment perpendicular to B); along B is free and uncoupled.
Reframed (Philip): use it as the SELECTION test for the electron -- run a set of candidate structures,
keep what survives and reacts appropriately. Fix criteria before running: (1) persists with/without
field, no pinning; (2) translates rigidly under force; (3) cyclotron omega ~ B, speed/radius independent
-> one e/m, consistent across tests; (4) moment aligns, precesses; precession/cyclotron = g/2 ~ 1.00116
(classical loop 1/2); (5) cyclotron radiation. Candidates: dipole_mode / double-ring (flagged
closed), spheromak/Hopf ring, ring soliton, torus-knot windings, moat/glut (may not exist free). Catch:
step()/AE is linear, nothing holds together -> needs the lam-coupled nonlinear rule; "nothing
survives" is informative about the rule.

## IDEA 2026-09-27 (Philip): macro-particle scaling -- one structure standing for a mole, how much varies
PIC-style: q -> Nq, m -> Nm. q/m-only observables (cyclotron, drifts, orbits in external fields) should
be invariant; self-field effects ~ q^2 grow ~N per mass (radiation reaction, self-energy, binding); hbar
effects don't scale (Landau spacing, l_B, g-2); discreteness ~1/sqrt(N) (or context). In DST the rule is
nonlinear via lam, so N x winding != N structures -- which observables stay fixed is the test. Particle
analogue of coarse cells (L = h/<ell>); ties to mem feedback_em_compliance_requires_coarse_limit,
project_dst_virtual_particle. Cheapest first version (no electron needed): scale a lam sink by N in a
Kepler-type orbit (gravity via grad u is confirmed) -- orbit should depend only on the ratio.
Philip: part of calibrating the fundamental scale. Scale-invariant (q/m) behaviour can't fix a scale;
where scaling breaks (self-field q^2, hbar-type, discreteness) is where an aggregate meets the substrate
scale -- the calibration handle (same target as the T -> 0 = grid-scale asymptote). One dimensional
quantity may set units (h in metres / the tick). Method (Philip): approximate, calibrate, repeat --
calibrate on one ratio, then it must carry over unchanged to the others: alpha ~ 1/137 (e^2 vs hbar c:
exactly the two non-scaling parts), m_p/m_e ~ 1836, g/2 ~ 1.00116, cyclotron e/m.

## QUEUED 2026-09-27 — start here; read `now`'s 2026-09-26 session (items 1-6)

09-26 afternoon DONE (`exp/2026/09/26/dipole/status.md`): R160 full-field vs exact (1a-1d, 1e partial),
off-axis [111] (item 2). Dipole on AE is converged 2nd order everywhere, orientation independent.

1. **Remaining R160 checks**: grid-scale energy in the source region and u's grid-scale part (curl-grad
   leak) -- 1e was only bounded at the shells (<= 1.2% amplitude).
2. ~~Two dispersion relations~~ DONE 09-27 (`exp/2026/09/27/dispersion/status.md`): plane waves at 3 h,
   2 directions. Both 2nd order; T sector (E/B) dc/c = -0.17 (kh)^2 along x, -0.12 [111]; u (step()) =
   exact leapfrog prediction, -0.018 / -0.031 (kh)^2. T is 4-10x more dispersive. Test
   TestPlaneWaveDispersion. Reconstruction decision (09-26 item 5) now has numbers; Philip's call.
   PINNED (symbol.py, exact Bloch symbol matches to 3 digits): 80-85% = LS gradient of cell-centred A
   (2h-span centred difference, -5/36 (kh)^2 along x); rest = face->cell averaging, polarization-dependent
   (birefringence along [110]); 4 of 7 face modes per k are omega=0 (invisible to K). Next candidate:
   symbol of a compact DEC curl (Delaunay triangle circulation) before any C++ change.
   DEC SYMBOL DONE (dec_symbol.py, status.md): curl(grad)=0 exactly, no fitted constant, transverse
   -0.009..-0.012 (kh)^2 (15-20x better than LS), 4 extra modes gapped at omega h/c ~5 instead of frozen.
   Side: step() weights are W^2 = 9:1, Kelvin Hodge star is 3:1 = W (W applied on both legs).
   Decision pending (Philip): build DEC curl in C++ step_ae? and step()'s 9:1 vs geometric 3:1.
   DONE 09-27 (Philip: port if it makes sense; `exp/2026/09/27/dec/status.md`): numpy DEC dipole h=0.5 /
   0.25 source work err +0.011 / +0.0026 (LS 0.187 / 0.033); packets ~20x better. step() is DEC with a
   9:1 star; "dec" = DEC with that same star (+ *2 = 5/2 for equal c): exactly div-free in step()'s
   sum W, one star for both sectors. Ported: set_curl('ls'|'dec'|'dec_geo'), 19/19 tests. Default
   switched to 'dec' (Philip 09-27) in C++ and AE(curl=); LS tests now ask for 'ls' explicitly; new
   TestStepAEEnergy.test_bounded_dec, TestTransverseWaveNeedsE.test_AE_dec (5.5e-3 vs LS 7.2e-2).
   REBUILD PENDING until the C++ dec dipole sweep (run_dec_cpp.sh) finishes -- .so in use. Old
   09-24..26 cpp scripts would now run with 'dec' if rerun.
3. **PHARE (Philip runs)**: emhd_spheromak.py IC verified against PHARE's t=0 dump (1% L2; plot
   `exp/2026/09/26/phare_emhd/ic_meridional.png`). Exterior = uniform anti-aligned + dipole (poles have
   nulls; dipole part non-periodic, ~5% jump at the box faces). Open (Philip's call): keep, or B=0 outside
   with surface current (relaxation test, looks like a free wound-up dipole). The 2 steps in the dump are
   both t=0 (two starts appended). Then: unperturbed, PERTURB=0.2, plasmajor.py.
   Philip ran unperturbed to t=0.5: no visible change (expected; that's ~0.3 whistler period, omega ~ k^2 B
   ~ 3.6 at k = X0/R). Restart dump at t=0.5; plan: restart to t~5 (~3 periods; tilt is the thing to watch,
   the moment is anti-aligned with the external field). Then PERTURB=0.2 to t~5.
   Restart edits: uncomment restart_time=0.5, final_time=5 (timestamps follow it); watch for a duplicate
   t=0.5 step appended to the existing vtkhdf (new output dir if so).
4. Remaining 09-26 items 2-6 below (induction visual checks, cascade probe, level-0 checklist,
   reconstruction decision, charge).

---

## (superseded 09-26 afternoon by the block above; items 1a-1d, 2 done)

09-26 item 1 (R160 rerun) DONE: see `exp/2026/09/26/dipole/status.md`.

1. **Assess the R160 data** (`/bk/dst/2026/09/26/dipole_R160_h0.125/`,
   state_*.npz: t, u (N), phi (14,N), e (14,N); pos.npy). Row-restricted
   diagnostics only (22M cells; load one snapshot at a time, ~5 GB). Plots to
   exp/2026/09/27/.
   a. Time-resolved shell flux from the window snapshots (8823, 10083, 11344,
      12604, and 7562 at t=4.85): instantaneous S_r(th) per shell; any
      late drift at r=9.5 = reflection or leakage.
   b. Full-field comparison with the exact retarded gaussian-source Hertzian
      field (near + far terms, A_z and u, E, B), not just the far-field
      amplitude; error norm vs r. Same-time comparison with h=0.25's final
      state (`exp/2026/09/25/dipole/dipole_h0.25_R20_cpp.npz`, same t_end
      8.08): pointwise Richardson.
   c. Lorenz condition check: u_t + c^2 div A from consecutive snapshots is
      too coarse (1260 steps apart) -- use u vs the analytic retarded scalar
      potential of the oscillating charge instead.
   d. Lattice anisotropy: phi-dependence of S_r and |E| at fixed (r, th)
      (BCC directional bias; cheap substitute for an off-axis run).
   e. Grid-scale energy fraction per snapshot (was 09-26 item 2; ties to the
      curl-grad leak).
2. **Off-axis dipole** ([111] or generic direction) at h=0.25 (~40 min):
   does the flux/pattern depend on orientation relative to the lattice?
3. **PHARE (Philip runs)**: `emhd_spheromak.py` (unperturbed, then
   PERTURB=0.2) and `plasmajor.py` in phare/mkn/tests/functional. Look at:
   W_B flat, force-free residual stays ~0 unperturbed, perturbed relaxes vs
   radiates whistlers, max|Vi| tiny. Clean version needs a frozen-moments
   path in solver_ppc.hpp (skip moveIons_) -- Philip's call.
4. Remaining 09-26 items 2-6 below still open (induction visual checks,
   cascade probe, level-0 checklist, reconstruction decision, charge).

---
