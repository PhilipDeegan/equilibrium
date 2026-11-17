# Entries for 2026-09-26 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## SESSION 2026-09-26 — dipole on AE converged 2nd order in the full field (phase error dominates) and orientation independent ([111]); R160 done: source work and shell flux both converge 2nd order to the form-factor reference (Richardson 0.4107 / 0.411 vs 0.4114); PHARE hybrid loop read for ontology, EMHD (frozen-ion) spheromak test written; plasmoses.py spheromak IC found invalid

1. **Dipole R160** (`exp/2026/09/26/dipole/status.md`; script copied there
   from 09-25 with plots + snapshots added): R=20, h=0.125, N=22.3M, 12604
   steps, ~4 h at 8 threads. Source work 0.419 P_th, shell flux 0.427 flat to
   0.2% over r=5..9.5, slope -2.000. Errors vs F^2=0.4114: source work
   0.187/0.033/0.0076 (h=0.5/0.25/0.125), flux 0.064/0.016 (ratio 4.0).
   Flux - source work gap 7% -> 2%. pole/eq 0.059 and pattern resid 0.017 are
   exactly the 20-degree bin average of sin^2 (0.0597/0.0174), not error.
   Scope: prescribed J, EXPERIMENTAL AE, dipole along z (cubic axis).
   - Output: 101 frames `exp/2026/09/26/dipole/dipole_R160_h0.125_plots/`
     (E_z, B_y, u on y=0 slice + history), 11 full-state snapshots (u, phi, e;
     5.2 GB each) at steps 0, n/10, ..., n in
     `/bk/dst/2026/09/26/dipole_R160_h0.125/` with pos.npy. Window-interior
     snapshots: 7562 (t=4.85), 8823, 10083, 11344, 12604.
   - Script now: `--plots N` (default 100 -> 101 frames), `--snaps N`
     (default 10 -> 11, whole run), snapshot dir dated 2026/09/26.
   - Philip: plots coherent, no visible BC leakage/reflection. By design the
     reflected front is at r ~ 10.5 at t_end (outer shell 9.5).
   - Q (Philip): is E Cartesian? No: E is 14 face values per cell like phi,
     updated per face. curl_curl passes through a per-cell 3-vector
     (reconstructed A, B from 14-neighbour differences) then projects back;
     only the diagnostics (plots, Poynting) take x/y/z components.
2. **PHARE ontology** (hybrid_level_initializer.hpp:163, ohm.hpp,
   electrons.hpp): J = curl B (Ampere, no displacement current -> J has no
   memory, slaved to B's shape); Ve = Vi - J/n (electrons are a massless
   closure carrying whatever J B demands); Pe = n Te; E = -Ve x B - grad Pe/n
   (+eta J - nu lap J); Faraday. Ions are the only inertia. With n fixed,
   Vi = 0, Te = 0: dB/dt = -curl(J x B / n) = EMHD. E.J = 0 -> int B^2
   exactly conserved (pure geometry rearrangement); force-free fields exact
   equilibria; whistlers (omega ~ k^2, dt ~ dx^2). Literal zero particles
   breaks (divides by n).
3. **PHARE test files** (untracked, `/home/p/git/phare/mkn/tests/functional/`;
   Philip mid-rebase, runs them later):
   - `emhd_spheromak.py`: ion mass 1e6, cold, n=1, Te=0, 64^3 dx=0.125,
     dt 2.5e-4, t=10 (40k steps, ~13M particles: slow; shorten first).
     IC: Chandrasekhar-Kendall spheromak r<R=1.5 (kR=4.4934) + uniform
     anti-aligned B_ext = B0 j0(x0)/1.5 = -0.145 B0 + dipole outside; B
     continuous, no surface current. Checked 64/128/256: div B/|J|
     0.012/0.005/0.002, JxB residual 0.010/0.004/0.001. PERTURB scales the
     toroidal part (div-free, residual 0.14). Env: PERTURB, ION_MASS, NU
     (1e-4), ETA (0), PLOT_ONLY. Plots W_B, force-free residual, max|Vi|.
     First version had B_EXT sign wrong (caught by the check).
   - `plasmajor.py`: plasmoses.py with the fixed spheromak (same setup,
     mobile protons, n=0.5, R=1 at dx=0.1 -> ~1-2% initial residual).
   - plasmoses.py's `get_taylor_spheromak` (written by Claude earlier) is
     B0 j1 sin(th)(th_hat + phi_hat): no radial part, div B/|J| 0.35
     (non-converging), J perpendicular to B everywhere. Any plasmoses result
     started from a non-field, non-equilibrium state.

4. **Afternoon: R160 vs the exact retarded field** (`exp/2026/09/26/dipole/status.md`,
   `r160_vs_exact.py`, `r160_phase_amp.py`): exterior of the gaussian source = point dipole x F exactly;
   the ramp leaves p = -I0 cos(wt)/w (no static charge). Pointwise error h=0.25/h=0.125 on shared sites
   ~4 for A, B, E, u over 2.5<r<10 (whole field, near + far). The error is phase: E, B lag 0.0057 kr
   (phase speed 0.57% slow, ~0.15 (kh)^2), amp +1.5-2% (-> flux +3-4%); residual after phase/amp fit
   0.4-1.2%. Fields lag differently (A 0.0042 kr, u 0.0012 kr): u (face difference) and E/B (LS
   reconstruction) have different dispersion -- measurable symptom of the two-reconstruction issue.
   u matches the analytic retarded scalar potential (Lorenz) to 0.4-2.5%. No late drift at r=9.5.
   Azimuthal (4-fold) modulation of the error: +-1-4.5% of the error itself.
5. **Off-axis [111] at h=0.25** (`--axis`): source work 0.444 and flux 0.474-0.476 identical to the
   z-axis run (<=0.2%); azimuthal flux spread 0.1-0.2% (1-4% at h=0.5). Orientation independent.
6. **PHARE emhd_spheromak IC** (Philip: "looks like a weird dipole"): PHARE's t=0 B matches the analytic
   IC to 1% (`exp/2026/09/26/phare_emhd/`, plot ic_meridional.png). The two tokamak-like lobes are in B_phi
   and the flux surfaces (O-points at x~+-0.8); |B| is centre-peaked (B_z = 2B0/3 on axis) -- the old
   plasmoses field had |B| two-lobed only because it lacked the B_r cos(th) term. The odd part is the
   exterior (uniform anti-aligned + dipole, polar nulls): needed because a force-free CK spheromak cannot
   match a pure-dipole vacuum exterior (B_phi(R)=0 forces B_r(R)=0; virial theorem: a finite force-free
   field needs external stress). Philip: plasmoses' spheromak (non-force-free, has curl) would decay to a
   dipole. Open: keep the external field or use B=0 outside + surface current.
   Philip ran unperturbed to t=0.5: no visible change (~0.3 whistler period at the spheromak scale, so
   not yet a stability verdict); restart dump at t=0.5, to be continued to t~5.

---


# Entries for 2026-09-26 moved from repo-root `next` on 2026-10-25 (verbatim, newest first)

## QUEUED 2026-09-26 — start here; read `now`'s 2026-09-25 evening items 8-13

Level 0 bulletproof first (Philip); AMR later. Philip: wait for running jobs
before starting new CPU work (8 physical cores).

1. **Rerun dipole R160** (killed 09-25 22:25 at step 6400/12604, per
   Philip; ~3.5 h at 8 threads, nothing else running meanwhile):
   `cd exp/2026/09/25/dipole && PYTHONPATH=/home/p/git/eq python3 -u
   dipole_radiation.py --R 20 --h 0.125 --cpp --threads 8 --out
   dipole_R160_h0.125_cpp.npz`. It writes 5 snapshots over the window to
   /bk/dst/2026/09/25/dipole_R160_h0.125/ (the script hardcodes that date:
   change it for a new day). Then add the R160 row to
   `exp/2026/09/25/dipole/status.md` (source work vs 0.411, shell flux,
   slope; is the convergence still 2nd order from R40/R80?).
2. **Visual checks before trusting any scalar verdict** (Philip: without
   saved state or plots it may be a false positive). Plots go in
   exp/2026/09/26/, not scratch.
   - induction R36/R72/R144 final states (R144 in /bk/dst/2026/09/25/,
     others in exp/2026/09/25/induction/*_final.npz): meridional |B|, |E|, u
     on one colour scale; loop-field shape, wall reflections, residual
     ringing.
   - dipole R40/R80 final states + R160 snapshots: outgoing phase fronts,
     sin^2 lobes, nothing reflected back from the wall.
   - fraction of energy at grid scale in each state (ties to the
     curl-grad leak, now item 11).
3. **Cascade-direction probe** (level 0, no C++ change): smooth broadband
   circulating state, no source, sponge exit; track the energy spectrum over
   time -- up into coherent structure (2D-like) or down to grid scale (3D).
   First handle on "what blocks dissipation"
   (mem project_structure_is_blocked_dissipation).
4. **Level-0 checklist, continued** (C++ edits only when no run is using
   the .so): E boundary (sponge, PML later); sigma in step_ae as
   div(sigma A) at the face; face BCs: implement or document excluded.
5. **Open, Philip's call**: one reconstruction for both u<->phi and E/B (makes
   the curl-grad leak consistent by construction) vs keeping step()'s face
   difference at level 0.
6. **Charge**: topological soliton preferred; needs a field valued on a
   sphere or a compact phase (direction of A? phi-holonomy phase?) --
   Philip's physics call before building.

---
