## SESSION 2026-11-03 — performance pass on the continuous C++ kernels (llvm-mca + perf, exp/2026/11/03/mca/status.md); doc/dst_versus.md (what DST must survive; Bell is decisive)

1. Build: the -gW 9 debug build (-g3 -pg, no NDEBUG) gives bit-identical numbers to -g 0 (snap.py), so earlier runs
   stand; it only costs time (EQUILIBRIUM_DEBUG_DO dphi check every discrete step, asserts, mcount). Release:
   mkn clean build -KOqdp mod,mod_lattice -x mpi -g 0 -W 9 (-DNDEBUG -O3 -march=native, clang 22 via ccache mpicxx).
2. Fixes, all in inc/eq/core/data/lattice/continuous/adaptive_lattice.hpp (discrete untouched): per-leaf sigma once
   per step(); 1/h and 0.75 / (h w_k) hoisted; tris() / face table hoisted, face table fixed-capacity; phi / w_k ->
   phi * (1 / w_k) in cartan_transport (the only non-bit-identical change: <= 2.6e-16 of max after 20 steps).
   step_cartan R 16 (N 178k): 1 thread 94.9 -> 76.4 ms (-20 %), 8 threads 23.2 -> 20.0 ms (-14 %).
   test_continuous_cpp / test_continuous / test_continuous_spheromak OK.
3. perf beats mca here: the cartan face loop (_cart_sum) was 52 % of step_cartan, the triangle loop 15 %; mca ranked
   them the other way (it counts cold option branches, assumes L1). A two-pass face sum (each sum once) is faster on
   1 thread but slower on 8 (extra 20 MB scratch + barrier) -> reverted; 8 threads now likely memory-bound.
4. Tools: exp/2026/11/03/mca/snap.py (fixed-seed before / after snapshot: discrete step uniform + AMR, continuous
   step lam / medium, step_cartan plain / sigma / hold / vcap / cap; `compare` exits 1 on any bit change;
   snap_baseline.npz = HEAD, snap_current.npz = today's final), bench.py (--only step_cartan), EQ_MCA_BEGIN/END
   markers (inc/eq/core/mca.hpp, only with -DEQ_LLVM_MCA). mkn clang.llvm-mca mod now extracts loop regions:
   mkn build -KOqp mod_lattice -x mpi -g 0 -W 9 -a -DEQ_LLVM_MCA -m 'clang.llvm-mca{link{regions: only}}'
   (rebuild without -a after). mca_regions.py superseded (it dropped nested-loop headers).

5. doc/dst_versus.md (new): what DST must survive from established physics (theory = signposts for truth, solid for
   invalidation). A = theorem + experiment: Bell / CHSH, single-photon anticorrelation g2(0) < 1, Planck law vs
   equipartition; B = precision bars: Lorentz / isotropy / preferred frame (PPN alpha1, alpha2), free fall 1e-15,
   gamma, c_gw, QED g-2; C = observed, no no-go: spin 1/2 / 4 pi, stable atoms. Bell is decisive: its "locality" is
   local causality (nothing outside the light cone), not particle independence; lambda may be a shared global field.
   It excludes a class of RULES (local causality + measurement independence), not hidden variables in general (Bohm
   survives). The current rule (nearest-neighbour, bounded speed) is in the excluded class -> doc/dst.md "DST is
   outside Bell's theorem" does not hold as written (not edited). Philip: no probability spaces, no probability
   amplitudes, no abstract spaces (Bohm's configuration-space wave is aphysical) -> only Bohm's lesson carries
   (a preferred frame makes superluminal influence consistent); the non-local influence must be a physical substrate
   structure. Candidate: the canal as a sub-wavelength thread of less-depleted cells, faster than ambient light
   (vacuum is low but non-zero lam), unresolved by photons (10-25 defect result). Bars: no signalling, Tsirelson
   2 sqrt 2 and no further, cos^2 from physical content. u / phi stay provisional DOF.

6. Canal timing (exp/2026/11/03/canal/status.md): a lam-0 strip in a lam-1 ambient delivers ahead of the ambient
   front down to ~3 cells across (~1.5x ambient at 1e-2, 1/4 the peak); 7 cells across = full sigma ratio 2x.
   A pulse ~5x wider still notices the 3-cell strip (-12 % at the 0.5 level): not transparent at that ratio.
   lam enters step() only via sigma (ell not wired), so faster-globally is built in; the width results are new.
   Scan D 48 (s 1.5-12): the 3-cell strip's own early signal falls ~s^-2..-2.7 once the pulse is wider than
   it (heading for transparency); the remaining ~7 % lead is mean-field. A canal's early signal is invisible to
   light at real ratios; what rides it must be matched to its scale. Nothing yet about Bell itself.

7. Test designs written into doc/dst_versus.md (A1, A3 sections; A1-A3 table rows). A1: timing bar from the
   theorem itself, S > 2 needs D < v_c dt, so experiments' influence-speed bounds (~1e4 c) need canal / ambient
   >~ 1e4; today's sigma law caps it at 1 + lam_bg -> vacuum lam_bg >~ 1e4 (sigma law provisional). Shape = relay
   (first measurement in the substrate frame re-orients the canal to a / a-perp; no signalling and Tsirelson fall
   out if faithful, share f otherwise); steps 1 detector (firing frequency linear in passed share), 2 relay
   (imprinted then rule-formed canal), 3 joint CHSH / P30. A3: 0 linear box never thermalises (expected), 1 finite-
   capacity receivers, continuous vs quantum offload, against the lattice's own mode count. Nothing run.

8. Ran A3 step 0 + an A1 step 1 precondition (exp/2026/11/03/planck/status.md). Linear periodic BCC box (n 32,
   T 4000): uniform keeps its k-spectrum to round-off and its frequency spectrum exactly; fixed rough sizes (9 / 10 /
   11) scatter 34 % of content to higher k by t ~800, then plateau, at unchanged frequencies (elastic); nowhere near
   equipartition. step() one-step map: 12 of 16 dof per unit cell static, 4 oscillating = 2 scalar u branches ->
   no polarised propagating content, so a polariser / Malus detector cannot be posed in step(). Both next steps
   (A1 detector, A3 receivers) need things the model lacks: a transverse propagating carrier and a threshold /
   quantised-offload body.

   Correction (Philip): 'no polarised content' was an E-field reading. Polarisation as observed = cos² transmission
   through oriented filters; a scalar packet with a two-lobe cross-section at angle a gives exactly that by overlap.
   A1 step 1 is posable in step(): two-lobe packet through a slot at b; failure modes = lattice birefringence
   (lobe orientations travel at different speeds), loss of coherence over distance.
   Philip: wrong question anyway: polarisation implies symmetry, symmetry is a product of competing pressures ->
   don't seed the symmetric packet; ask what balance in emitter / carrier / filter forms the orientation.

9. Settled background (exp/2026/11/03/bg/status.md, Philip: photons ride on a magnetically settled background). Uniform
   circulation about z (phi ~ A = B/2 z x r), settled by balancing its own Cartan push with u pressure; small u pulse,
   response = (bg + pulse) - bg. step(): no effect (exact, linear). step_cartan: pulse stronger across the axis, weaker
   along it (+2.0 / -1.2 % at B 0.05, +8.1 / -4.8 % at 0.1: ~B^2, same for -B = pressure-like, two-fold); part of the
   pulse converted into circulation (share 0.9 / 3.2 %, first order in B, flips with it = handed). Arrival unchanged.
10. phlop scope counters wired in: EQ_SCOPE (= PHLOP_SCOPE, inc/eq/core/phlop.hpp) on continuous step / step_ae /
   cartan_transport / step_cartan (time) and per-worker _pass1 / _pass2 / cartan_tri / cartan_face (time, perf_light;
   each worker thread counts itself). start_phlop now creates .phlop/ (mkn::kul::Dir). Bit-identical (snap compare),
   test_continuous_cpp 29 OK. Use: eval "$(cd tpp/phlop && PYTHONPATH=$PWD python3 -m phlop.app.pfm.stat --env
   perf_light=light)"; lattice.start_phlop() ... stop_phlop(); PYTHONPATH=tpp/phlop python3 -m phlop.timing.scope_timer
   perf .phlop/timings.txt -v. libpfm not built here -> generic events (cycles, instructions, branches); example
   exp/2026/11/03/phlop/counted.py (R 16, 4 threads: cartan_face ipc 5.4, cartan_tri 5.2, pass2 2.0).

11. A3 Cartan box (exp/2026/11/03/planck/status.md): closed periodic box, smooth seed |k| <= 2, step_cartan -> reaches
   equal energy per mode (white-noise reference) shell by shell by t ~120 at amp 0.3 (above-seed share 0.992 vs 0.994),
   ~3x slower at amp 0.1; kappa 0 keeps its spectrum. The rule thermalises classically = UV catastrophe, the same
   failure as the 10-24 cascade -> one requirement (stop at structure scale; offload quantum the candidate). Fixed
   continuous.spacing_of for periodic boxes (wrapped face gave h ~ box size; now min over cells).

12. Pair step 1 (exp/2026/11/03/pair/status.md; Philip: emitter / absorber at mutual equilibrium carry how much each can
   offload). Two 10-25 self-held bodies (R 8, x = +-18, n 112) vs each alone: up to t 100 the depletion between = sum
   of the solos (superposition); then pair-dependent: the middle deepens past the sum (+0.09 t 200, +0.43 t 300,
   saturated), content drains into the gap (u^2 x20 / x490), bodies approach (+2.9 each) and drift sideways in
   opposite senses (pair turns): a bridge forms, they start to merge, no settled pair at this separation. Fixed-
   coupling pair: the known kfix runaway, invalid. Next: larger separation, opposite senses.

Detail: exp/2026/11/03/mca/status.md, exp/2026/11/03/canal/status.md, doc/dst_versus.md.

## QUEUE 2026-11-03 — read `now` (STATE + 11-03 session) first. Older queues verbatim in exp/2026/MM/DD/status.md (after that day's now entries). 11-03 pre-tidy copies: /bk/dst/2026/11/03/{now,next}.pre_tidy.

Resume here (11-03 -> 11-04), all tied to doc/dst_versus.md A1-A3 and Philip's emission picture:
1. **Pair regime map** (exp/2026/11/03/pair/, pair.py; HELD for Philip): at x = +-18 two self-held bodies bridge and
   start to merge (superposition to t ~100, then the middle deepens past the sum, content drains in, they approach and
   turn). Merging may be legitimate (dendrites) but dendrites may need atoms -> field-only merging is the rule's
   behaviour, not validated. Options: separation scan +-24 / +-30 (n 128+) and opposite senses at +-18 (~37 min each,
   1 core each, 3 in parallel), or first the MHD reference (equilibrium/reference/spectral.py: two Taylor bodies at the
   same separation, do they merge?). Need a pair that SETTLES for step 2 (small kick on the emitter: does what leaves
   depend on the receiver = inhibited emission in the rule). Solo --kfix control before trusting any fixed-coupling run.
2. **The collecting body** (main blocker for A1 detector, A2, A3 receivers, neutrino absorption): a body that releases
   content in steps set by itself. Constraints: sits near threshold, preloaded by the settled background (prompt
   photoemission); quantised units collective (10-19); capacity exists only as smooth set_cell_capacity. Under
   Philip's global picture the step is relational (emitter + absorber pair), so the pair (1) comes first.
3. **Settled background follow-ups** (exp/2026/11/03/bg/): does the converted (handed) circulation propagate and with
   what orientation; background axis vs lattice (z only); kappa; refinement. Emission-pattern test (needs a held,
   shedding body): does shedding align with the background axis, null along it (Zeeman)?
4. **Two readings of the photon** (W: spreads, unit set at absorption; P: held quantum on the settled background,
   guided by the background's disturbance). Test for P: held packet + its disturbance through two slits, do landings
   form fringes? Needs a transversely held packet (what holds it = background pressure balance?).
5. A1 relay / joint CHSH, A2 one-click, A3 receivers, neutrino absorption coupling, passive-target k-exponent scan
   (expected Rayleigh): parked until 1-2 exist. A1 bars: canal / ambient speed >~1e4 (sigma law caps at 1 + lam_bg);
   superdeterminism flag if settings belong to the global equilibrium (Philip's view open).

Done 11-03 (detail in now): perf pass + phlop scopes; dst_versus.md + designs; canal timing; A3 step 0 (linear box, no
thermalisation); A3 Cartan box (equipartition = UV catastrophe); settled background coupling; pair step 1 (bridge).

