# Entries for 2026-08-17 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## IN PROGRESS (2026-08-17): curl-full spheromak investigation — volume-filling, two-lobed deliverable BUILT and tested; not yet stepped

**Goal**: build a DST-native, coordinate-free (no Cartesian cross-product formulas)
volume-filling, two-lobed object, genuinely curl-full (unlike `dipole_mode_ic`,
which never merges — see doc/dst/failures/dipole-mode-ic-no-merging-curl-free.txt),
comparable to the dipole plots and PHARE's `~/tmp/EM_B.vtkhdf` reference, then step
it and see whether real merging happens. **That object does not exist yet** as of
this entry — what's built so far is confirmed groundwork (see below), and a bare
wire-loop diagnostic that must NOT be called "a spheromak" (see
doc/dst/failures/curl-full-spheromak-mislabeled-wire-filament.txt).

**Step 1, done and legitimate on its own**: rebuilt R180L0/R60L1 (existing
curl-free dipole-mode ICs) from scratch under the current (post-2026-08-15
pass1/pass2-fix) code, replacing the per-experiment `dipole_mode_ic.py` copy with
the now-promoted production module. Both clean, no blowup, NDEBUG/release build.
H/H0 @ step 300: L1R60=0.998604, uniform_r180=0.995181 — both decay slowly and
smoothly; L1R60 no longer shows the old anomalous fast decay (0.912609 pre-fix) —
confirms the pass1 fix genuinely changed L1R60's physical decay rate, not just
cosmetics. Full detail: `exp/2026/08/16/14_50_rebuild_r180l0_r60l1_post_pass1_fix/report.md`.

**Step 2: coordinate-free discrete curl derived and validated** on the Kelvin/BCC
lattice: every cell has 12 elementary triangle loops (2 hex hops == 1 direct sq hop
to the SAME cell, verified 100% on real leaf data, no coordinates needed) —
`equilibrium/substrate/discrete_curl.py`, `tst/substrate/adaptive/test_discrete_curl.py`.
Key confirmed distinction: a discrete-derivative field (`phi[k,i]=u[nb[i,k]]-u[i]`)
has EXACTLY zero circulation (pure telescoping, float64 noise, not an
approximation) at any resolution — stronger than continuum Stokes. Built + validated
a genuine curl-full counter-example: a CONSTANT phi value around a ring (see
doc/dst/failures/cos-pattern-toroidal-winding-curl-free-standing-wave.txt for the
cos-pattern trap that was avoided). Confirmed a Kelvin-circulation-conservation
theorem empirically: pure `evolver.step()` (no `step_cross`) conserves a ring's
circulation exactly; `step_cross` erodes it monotonically (~2.6-3.4% over 10 steps
at both test and production scale) — `step_cross` IS the DST-native
resistive/reconnection mechanism. Full writeup: `[[project_dst_discrete_curl]]` memory.

**Wire-loop diagnostic validated the mechanism (not the deliverable)**:
`equilibrium/substrate/curl_full_spheromak_ic.py` (`seed_curl_full_pair`), built on
`adaptive_flux_tube.find_toroidal_ring` + a locally-fixed `find_linked_loop_anchored`
(see doc/dst/failures/find-linked-loop-centroid-anchor-breaks-at-scale.txt for the centroid
bug that was fixed). Produces a Hopf-linked ring PAIR with real circulation,
validated at hops=41 (production-relevant radius). Matching ICs built for both
R180L0/R60L1 (`/bk/curlfull_uniform_r180_ic220.h5`,
`/bk/curlfull_l1_wider_margin_ic220.h5`, both H0=332.0, ring A=328/B=4 tiles),
geometry plotted at `exp/2026/08/16/18_56_curlfull_hopf_spheromak_ic/`. Confirmed
the circulation-conservation/step_cross-erosion mechanism. Not yet stepped (queued:
"we will step later").

**Step 3, done (2026-08-17): the actual volume-filling, two-lobed, genuinely
curl-full deliverable, built and tested** — `equilibrium/substrate/curl_full_dipole_ic.py`
(`seed_curl_full_dipole_ic`), `tst/substrate/adaptive/test_curl_full_dipole_ic.py`
(6 tests, all pass, run together with `test_discrete_curl.py`/`test_ring_circulation_
conservation.py` — 14/14). Design: keep poloidal and toroidal halves of the classic
axisymmetric l=1 Chandrasekhar-Kendall/Taylor force-free state STRICTLY SEPARATE and
additive, instead of PHARE's `get_taylor_spheromak`'s single non-separable Cartesian
cross-product formula (which mixes both terms and can't be independently
curl-checked):
- **u, phi_pol**: unchanged call into `dipole_mode_ic.seed_dipole_ic` (same K=4.493
  poloidal mode) — already gives the two-lobed, volume-filling shape.
- **phi_tor (new)**: `B_phi(r,theta) = j1(K*r)*sin(theta)`, projected onto face
  normals via `phihat=(-y,x,0)/rho` — reuses the existing primitive
  (`adaptive_flux_tube.apply_toroidal_rotation`), not a new formula. Constant around
  the azimuthal loop at fixed (rho,z) — the same condition already proven curl-full
  on the wire-loop diagnostic.
- `phi = phi_pol + phi_tor`, each independently peak-normalized, same dual-zero
  boundary convention as `init_spheromak_ic`/`seed_dipole_ic`.

**Correct diagnostic for genuine curl-full structure** (real finding, not assumed
going in — see doc/dst/failures/per-triangle-curl-metric-doesnt-distinguish-curl-full.txt
for the metric that does NOT work): the pre-existing MACRO ring circulation
(`adaptive_flux_tube.phi_ring_circulation`, same tool the wire-loop tests use) is the
correct test — phi_pol (axisymmetric, no phi-dependence) gives exactly 0.0 net
circulation around any full ring encircling the axis; phi_tor gives a large, real,
O(1)+ net circulation scaling linearly with its own amplitude. Now the primary check
in `test_curl_full_dipole_ic.py`.

**Step 4, done (2026-08-17): R180L0/R60L1 production checkpoints rebuilt with this
IC.** `exp/2026/08/17/12_25_curl_full_dipole_ic_rebuild/`
(`build_curlfull_dipole_uniform_r180.py`, `build_curlfull_dipole_l1_wider_margin.py`)
— same mesh/envelope convention as the 08-16 14:50 dipole rebuild and 18:56
wire-loop rebuild (cells_across=220, R_SPH=1, padding=70→R_domain=180 for uniform;
r_l0=60/r_l1=46.667 for L1), `lam = u^2+0.5*sum(phi^2)` seeded the same way.
Checkpoints: `/bk/curlfull_dipole_uniform_r180_ic220.h5` (31,734,323 leaves,
H0=8.655935e+06) and `/bk/curlfull_dipole_l1_wider_margin_ic220.h5` (15,542,429
leaves, level_hist=[622634,14919795], H0=8.655806e+06) — the two H0 values agree to
within 0.0015%, same cross-check the dipole-only rebuild's H0 pair (4.462518e+06 vs
4.462388e+06) already passed. (An OOM bug hit during this rebuild — see
doc/dst/failures/curl-full-dipole-ic-oom-boundary-masking.txt; fix continues in the
next log section.)
Concurrent-build race in `seed_dipole_ic`'s unbatched face-loop found and fixed (batched to match rest of pipeline in `equilibrium/substrate/curl_full_dipole_ic.py`); retried alone, succeeded cleanly (peak_rss_GB=30.84, total_t=140.6s); both `now`-relevant test suites re-verified passing. See doc/dst/failures/seed-dipole-ic-concurrent-batching-race.txt.

**Step 5 (2026-08-17): both production checkpoints stepped 200 steps at `cross_scale=1e-3`** (`step_curlfull_dipole_uniform_r180.py`/`_l1_wider_margin.py`, `run_step.sh`+`run_step2.sh`). H/H0 stable and smooth throughout (uniform: 0.999→0.9979 at step 200; L1R60: 0.9996→0.9993) — no blowup. Two-lobed u shape and toroidal band topologically unchanged through step 200 in plots, just a slowly-growing thin leaked-field ring past the original r_sph boundary. CAVEAT (see PHARE comparison below): this run was in a much weaker regime than PHARE's reconnection-dominated phase, so "no merging" should not be over-read. Full detail + "erosion dose" comparison vs `test_ring_circulation_conservation.py`: [[project_curl_full_dipole_spheromak]] memory.

**Step 6: cross_scale=1000 stress test (Philip: "blowup is evidence too")** — confirmed unbounded exponential blowup (not NaN) on both checkpoints within 10 steps (H/H0 reaching 1e17-1e28), traced to `_step_cross_local`'s `delta` term having zero clamping on `scale` (confirmed by direct C++ read). Combined with Philip's "there should be *A* derived cross_scale value, not a pick" position: now governs how `cross_scale`/`eta_h`/similar knobs get treated going forward (Standard-Model/classical numbers OK as initial checkpoints, never final justification). [[project_derive_dont_pick_parameters]]

**Theory thread opened, not yet built on**: proton dipole = toroidal + poloidal flux BOTH at c, cross product ~ c² (geometric origin of E=mc²); proton mass = cost to translate that structure, imposed by `cross_scale` (extends the electron's already-settled "inertia = cost of translation", [[project_dst_electron_dipole]]). Also standing framing correction: [[feedback_no_particles_no_kinetics]] — DST has no particles/kinetics, only subsystem interaction between persistent flux patterns; important for any future PHARE (PIC-code) comparison.

**Step 7: poloidal/toroidal "lockstep" probe** — real (if inconclusive) negative result: circulation exactly frozen everywhere tested, but `u_rms` swing not locked to circulation in any simple way. Matching the IC's shape did not by itself produce co-propagation. See doc/dst/failures/poloidal-toroidal-lockstep-inconclusive.txt. Full saga: [[project_curl_full_dipole_spheromak]].

Resolution sweep of `swing` (cells_across=22→250) apparent dip at 220 ruled out as a lam-seeding artifact — see doc/dst/failures/cells-across-220-swing-dip-artifact.txt. Corrected (all-lam=0) table smoothly decreasing 0.72(22)→0.086(250), not yet plateaued.

**Open question flagged before trusting the above trend**: fixed 60-step budget vs. period scaling with `cells_across` (`T_steps≈136·cells_across`) may make the "convergence" an artifact of sampling a shrinking phase-fraction of each object's own period as resolution grows. See doc/dst/resolution_requirements.txt, REQUIREMENT 4. **Not yet fixed**: rerun the sweep scaling `N_STEPS` with `cells_across` (or compare matched phase-fractions) before trusting these numbers.

**PHARE `~/tmp/EM_B.vtkhdf` quantitative read (2026-08-18)**: clean two-timescale relaxation confirmed — fast phase (t=0→1.8): `Bx_rms` collapses 0.136→0.011 (~92%, real reconnection, previously only eyeballed); slow phase (t≈1.8→3.4): `Bx_rms` plateaus (~5% further loss) while in-plane rms decays smoothly throughout. Whole-slice field energy 26.93→10.30 (−62%) end to end, almost all in the fast phase. **Actionable finding**: H/H0 is likely the wrong diagnostic for "has merging happened" — track the toroidal/poloidal component split (or macro circulation) instead, since real topology change can be small as a fraction of total H while huge as a fraction of one component. See doc/dst/failures/cross-scale-1e-3-wrong-regime-vs-phare-reconnection.txt for the full mismatch this exposed. No unit-conversion between PHARE `t` and DST `cross_scale` attempted yet — queued. Full detail: [[project_curl_full_dipole_spheromak]].

**Correction (2026-08-18): don't just push `cross_scale` harder under the current default gate.** All cross_scale-dependent findings above used `step_cross`'s default `gate="v1"` (continuous diffusion), but `doc/dst.md` specifies through-cell translation as a discrete all-or-nothing gate condition. The code already has this as `gate="barrier"` (stateless, pooled per-cell drive vs threshold `m0`), and `doc/dst_next.md` §8 already derived a non-picked `m0` (`m0 = scaling_factor × void_limit`, `void_limit = m_p/m_Planck≈7.7e-20`; example values `m0_neutron≈2.49`, `m0_proton≈0.16`) — ready to use but never carried into this work. See doc/dst/failures/gate-v1-diffusion-vs-barrier-mechanism-mismatch.txt. **Next step, superseding the cross_scale-sweep plan**: re-run circulation-erosion and curl-full-dipole-merging checks under `gate="barrier"` with a properly derived `m0` before spending more compute tuning `scale` under `gate="v1"`.

**Resume here next**: fix the step-count-vs-period confound in the lockstep probe before drawing conclusions from the resolution sweep; separately still open: push cross_scale=1e-3 stepping further, work out a derived cross_scale, pursue the PHARE quantitative comparison (unit conversion between PHARE `t` and DST `cross_scale`), and test the calibrated barrier gate against the real neutron probe. All scripts/logs/plots live in `exp/2026/08/17/12_25_curl_full_dipole_ic_rebuild/`.
