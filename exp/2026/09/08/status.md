# Entries for 2026-09-08 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## Compressed 2026-09-08 -> 2026-09-16 BC/outflow/PML/ring-reflection-quality thread (see doc/dst/bc.md for full technical detail; full blow-by-blow narrative available via `git log`/`git show` on this file if ever needed)

That investigation (outflow one-way Mur BC vs. split/PML volumetric sponge,
tested against both a toroidal ring and a minimal two-point dipole source)
reached its own closure on 2026-09-16: **outflow BC is solid and
near-perfect for isotropic sources but has a real, bounded, non-growing
angular reflection floor for any multipole-structured source** — confirmed
independently on the ring AND a from-scratch dipole with zero shared
machinery (rules out the ring's own construction as the cause). Ideas
tried and RULED OUT for fixing that angular floor: taper band (idea 1,
purely radial, can't touch an angular deficit), PML+outflow combo (idea 4,
same radial-only limitation, though the implementation fix -- gamma=0 on
true boundary leaves -- is a genuine permanent improvement, kept), and a
passivity-ledger cap on outflow's ghost (idea 5, the constraint is real but
never binds, ~200x margin). **Dipole floor: RESOLVED** — direct
measurement confirmed it's a static, near-exactly-divergence-free IC
remnant (curl-only content that was never going to reach the boundary),
not a BC defect. **Ring floor: still OPEN** — its outer shell is still
measurably evolving (not frozen like the dipole), likely a slow ongoing
leak from the ring's own known-decaying core
([[project_spheromak_decay]]/[[project_saha_spheromak_decay]]) rather than
live boundary reflection; queued check (not run): compare outflow's vs.
absorbing()'s outer-shell DECAY RATE over a much longer horizon to
separate core-leak-limited from boundary-quality-limited. Practical
conclusion unchanged throughout: split/PML remains the production tool for
real Poynting-ring work; outflow is not (yet) a drop-in replacement.

**The bigger, still-load-bearing outcome of that whole thread** was a
conceptual pivot on what ring/dipole-based investigation is even for,
captured permanently in four cross-linked memories (read before restarting
ANY ring/dipole/magnetism work): [[feedback_ring_test_source_not_physical]]
(the ring itself doesn't persist — its BC behavior is legitimate only as
simulation-hygiene validation, or as a hypothesis SOURCE that must be
independently re-tested against something real, never as a physics finding
about the ring's own mechanism); [[feedback_em_compliance_requires_coarse_limit]]
(classical Maxwell/Poynting compliance may only be a coarse-grained/
statistical property, not something any single hand-built microscopic
construction should be expected to show directly); [[project_dst_magnetism_requires_structure_first]]
(every existing DST magnetism hypothesis already presupposes a persistent
structure exists first — testing magnetism via the non-persistent ring was
the wrong order of operations); [[project_dst_diffuse_default_pressure_structure]]
(a newer, more radical hypothesis: diffuseness may be the lattice's DEFAULT
state, with real winding structure only emerging under sufficient local
pressure/confinement from neighbouring structures — tensions with
[[project_dst_bohm_stepping_stone]], unresolved).

## SESSION 2026-09-08 REAL FIX (supersedes the "plotting bug" correction below) — find_toroidal_ring's search target only coincidentally centered the SQUARE at the origin; the octagon was landing its centroid at 2x that offset, a genuine physical bug (asymmetric sponge distance), now fixed generally for any mode

Philip correctly pushed back on the previous "just recenter the plot"
conclusion: the ring's real position in the domain matters physically (an
off-center ring is asymmetrically close to the sponge on one side), not
just visually. Root-caused properly instead of patching the plot:

`find_toroidal_ring`'s search `target = _REF_DIR[axis]*a*scale` is meant to
land the resulting loop's CENTROID at the origin, but actually just picks
a start tile near `target` and walks the face_seq from there -- whether
the centroid ends up at the origin depends entirely on that specific
face_seq's own geometry, not the target formula. Computed this exactly
(new `_face_seq_centroid_offset`, pure 2D vector math on the face
sequence, no lattice needed): mode="square"'s particular sequence happens
to start at its own centroid-symmetric point (offset = -hops, matching the
existing target formula by coincidence), but mode="octagon"'s chamfered
sequence does not (offset = -2*hops exactly, confirmed across hops=1/2/4/
8/12) -- landing every octagon ring off-origin, including the R=100 run
(centroid at x=-13.86, not 0).

**Fixed generally**: `target` is now computed as `-a * _face_seq_centroid_
offset(axis, face_seq)` for any non-"minimal" mode, instead of the fixed
formula -- works for square (regression-verified unchanged) AND octagon
(now correctly centers) without per-mode special-casing, so it should also
be correct for any future mode built the same way. First implementation
attempt had a component-swap bug (caught immediately by testing square
regressed to off-origin too -- fixed before considering this done). New
permanent regression test: `test_ring_centers_at_origin`
(tst/substrate/adaptive/test_octagon_ring.py) checks BOTH modes center to
<1e-9 at 4 hops values each -- guards against this exact class of bug
recurring silently.

**Reran R=100 octagon to t=20** (shortened per Philip's time constraint,
`fine_R100_octagon_centered_*`): genuinely, honestly symmetric now -- the
domain boundary itself is centered at x=0 (matching the ring), and the
dumbbell/peanut envelope is clean and left-right symmetric at both t=10
and t=20. This is the trustworthy version of the earlier finding (coarse
envelope shape improved by the octagon vs square) -- previous entry's
conclusion stands, now on solid ground rather than a plotting-recentered
approximation.

**Not yet done**: full run back out to t=40 (time-limited today, only
reached t=20); the fine-scale streak/polar-stripe persistence question
(does it survive with a properly-centered domain too) not re-checked at
the longer time horizon.

## SESSION 2026-09-08 CORRECTION — the R=100 octagon's "leaning left" was a plotting bug (ring not centered at the lattice origin), not a physics asymmetry; separately, a genuine real seeding fix was made (idx-based cos -> real-angle-based cos) that's worthwhile but was NOT the cause of the lean

Philip noticed the R=100 octagon cross-section looked asymmetric ("leaning
to the left"). Investigated two candidate causes in parallel:

1. **Real fix made and validated, but NOT the cause of the lean**:
   `init_toroidal_winding` seeded `u/phi` using each tile's raw PATH-INDEX
   (`cos(K*idx)`), not its real geometric angle -- confirmed the octagon's
   tile spacing in real angle is genuinely non-uniform (dtheta/tile ranged
   1.03-2.39 degrees on a 192-tile ring, tied to straight-vs-chamfer legs).
   Fixed to use each tile's actual angle (ring's own plane found from
   position variance, no hardcoded axis -- [[project reasoning]]: "no
   preferred axis"). All existing regression tests still pass. BUT: direct
   before/after comparison on the SAME setup (hops=8, R=45) gave nearly
   IDENTICAL results (-8.763 old vs -8.769 new weighted-mean-x) -- this fix
   changes nothing for the specific asymmetry Philip flagged. Still a real,
   worthwhile correctness fix (matters whenever idx-to-angle non-uniformity
   is large enough to matter, just wasn't the culprit here). Kept.

2. **Actual cause, found and fixed**: `find_toroidal_ring`'s octagon search
   does not land the ring at the coordinate origin the way the square
   search happens to (confirmed: square ring centroids are ~1e-16, exactly
   origin, at every R/hops tested; the R=100/hops=12 octagon's centroid is
   at x=-13.86, NOT the origin). All of today's cross-section plots were
   drawn in absolute lattice coordinates, never recentered on the ring's
   own position -- fine for the square (coincidentally centered), silently
   wrong for the octagon. Redid the R=100 octagon cross-section properly
   ring-centered (`fine_R100_octagon_cross_sections_centered.png`, no
   rerun needed, recomputed from already-saved data): the pattern IS
   genuinely symmetric once correctly centered -- polar stripe sits exactly
   on x=0, the t=10/20 dumbbell envelope is clean and left-right symmetric.
   **The "leaning left" was entirely a plotting bug, not a real
   asymmetry.**

**Standing conclusion, corrected**: the earlier "coarse envelope shape
improved with the octagon, fine streak/polar-stripe pattern didn't" finding
(previous entry) still holds and is now on firmer footing (properly
centered, not just eyeballed) -- both halves of that conclusion are
confirmed, not undermined, by this correction. Not yet fixed: whatever in
`find_toroidal_ring`'s octagon candidate-selection logic causes it to land
off-origin (likely the `target = _REF_DIR[axis]*a*scale` heuristic assumes
a symmetric-around-target loop, true for the square's face_seq but not
verified for the octagon's) -- low priority now that the PLOTTING fix
(recenter on the ring's own centroid) is known and sufficient for
visualization purposes; anyone building on `find_toroidal_ring(mode=
"octagon")` should recenter using the ring's own centroid, not assume it
sits at the lattice/domain origin.

**Also flagged, separate and unrelated pre-existing issue, NOT caused by
today's work**: running the broader adaptive test suite surfaced
`test_injecting_bc.py::TestCollapsedElectron::test_lam_on_ring_kills_
circulation` failing, comparing two circulation values already at the
floating-point noise floor (~1e-15) at t=0, before any stepping. Proved
directly (bit-identical reproduction with the OLD pre-fix formula) this is
NOT caused by today's seeding-angle fix. Almost certainly the same,
already-documented ([[project_dst_discrete_curl]]) mathematical fact that
a symmetric cos(theta) standing-wave circulation integrates to exactly
zero around any closed loop -- this test (and possibly its siblings in the
same class) may have never been measuring real physics, just noise. Not
fixed here -- flagged for whoever owns that test file next; likely needs a
uniform-amplitude (`seed_constant_current`-style) IC instead of the cos
pattern if a genuinely nonzero circulation is required for the comparison
to mean anything.

## SESSION 2026-09-08 STILL YET FURTHER CONTINUED — octagon ring at R=100 (N=5.44M) decomposes the anisotropy into TWO separate effects with different causes: coarse envelope shape WAS source-shape-driven (now improved), fine streak/polar-stripe pattern is NOT (still present, likely genuine BCC dispersion anisotropy)

Ran the full pipeline with the new octagon ring at R=100/hops=12
(ring_r=29.58, N=5,441,209, sponge_fraction=0.3/damp_max=0.01, fine-logging
deliberately disabled to control cost -- `exp/2026/09/08/
12_15_poynting_time_evolution/fine_R100_octagon_*`).

**Result, genuinely mixed and informative**:
- **Coarse envelope shape improved**: at t=10/20 the overall |S| envelope
  is a smooth two-lobe "peanut"/dumbbell shape (fatter equator, thinner
  poles) -- much more classically dipole-like than the sharp diamond the
  SQUARE ring produced at comparable times. This part of the anisotropy
  really was source-shape-driven, and really is improved by a rounder
  source -- confirms Philip's hypothesis for this aspect.
- **Fine structure NOT improved**: by t=30/40 the same persistent bright
  vertical polar-axis stripe and fine radiating streak pattern reappear,
  essentially unchanged in character from the square-ring runs. Since the
  octagon has no sharp 4-fold corners left, this part is very unlikely to
  be corner/shape-driven -- it looks like a genuine, separate BCC lattice
  dispersion anisotropy effect (property of the propagation medium itself,
  [[project_bcc_dispersion_anisotropy]]), independent of source shape.

**Conclusion**: the "diamond pattern" investigated across this whole
session thread was actually TWO conflated effects with different causes --
one now resolved (coarse envelope shape, fixed by using a rounder ring)
and one still genuinely open (fine streak/polar-stripe structure, real and
likely a fundamental lattice property at this resolution, not an artifact
of domain size, ring size, or ring shape -- all three have now been ruled
out as its cause across this session's tests).

**New reusable code**: `_octagon_face_seq`/`find_toroidal_ring(mode=
"octagon")` (adaptive_flux_tube.py) + 3 passing regression tests
(`tst/substrate/adaptive/test_octagon_ring.py`) -- a genuinely rounder
(measured: 5.4% vs square's 11.6% radial spread at matched hops) ring
construction, reusable for any future work needing a less-artificially-
cornered current-loop-like source. `probe_poynting_fine_and_snapshots.py`
gained `--mode square|octagon`.

**Not yet done**: quantify the polar-stripe/streak pattern directly (angle-
resolved |S| at fixed radius comparing axis-aligned vs diagonal directions)
to confirm it really is resolution-independent BCC dispersion rather than
some other still-unconsidered cause; no resolution/domain-size test has
been run specifically on the OCTAGON construction yet (only done for the
square) -- in principle should be re-checked, though given the effect
persists identically in character between square and octagon at the same
domain/ring proportions, a domain-size origin looks unlikely for it too.

## SESSION 2026-09-08 STILL EVEN FURTHER CONTINUED — the "diamond" pattern's real cause found: find_toroidal_ring(mode="square") is a literal permanently-sharp-cornered square, not a resolution artifact; built a genuinely rounder octagon generator, testing at R=100 now

Philip's chain of corrections got to the real root cause faster than pure
resolution-scaling would have: "should the current loop be a square? this
sounds a bit odd" -- correct, it's a pragmatic historical simplification
(only the 4 cardinal sq-face directions are trivially guaranteed to close
exactly), not a physical modeling choice. Confirmed directly in code
(`_square_face_seq`/`_AXIS_DIR_CYCLE`, tst/substrate/adaptive/
adaptive_flux_tube.py): 4 straight legs, hard 90-degree corners, ANY hops
value -- scaling hops just makes a bigger square, never rounder. This
means the earlier "increase hops to test resolution convergence" idea
(hops=4 vs hops=8, still square both times) could never have shown
convergence to isotropy regardless of outcome -- corrected before wasting
more compute on it. The R=30-vs-R=50 matched-window test from the
previous entry also never actually varied source shape (same hops=4 both
times) -- it correctly ruled out "domain size" but was never capable of
ruling out "source shape" either, which is the real culprit.

**Built `_octagon_face_seq`/mode="octagon"`** (adaptive_flux_tube.py): a
genuine 8-DIRECTION loop -- same 4 cardinal straight legs as the square,
but each corner replaced by a diagonal "staircase" chamfer (alternating
hops between the two adjacent cardinal directions) instead of a sharp
turn. Net displacement exactly zero by construction (verified both
algebraically and against the real lattice graph). Measured roundness
directly (radial std/mean over the ring's own tiles) at matched hops=4:
square=11.6% relative spread, octagon=5.4% -- genuinely, measurably
rounder, not just visually. 3 new regression tests, all passing:
`tst/substrate/adaptive/test_octagon_ring.py` (closure algebra, real-
lattice closure + winding=1 at several sizes, octagon-rounder-than-square
comparison).

Note on naming confusion avoided: `_TOROIDAL_PATTERNS`'s existing 8-TILE
minimal loop is sometimes called "the minimal octagon" elsewhere in this
codebase/docs -- that is a square (4 sides), not an 8-direction shape;
documented the distinction directly in both docstrings to prevent future
confusion between "8 tiles" and "8 directions".

**Now running**: full Poynting fine+snapshot pipeline with the octagon
ring, scaled to R=100 (N=5,441,209 -- Philip: "fine as long as it fits in
RAM and doesn't take forever"), hops=12 (ring_r=29.58, matching the
earlier-discussed ~60-diameter target), sponge_fraction=0.3/damp_max=0.01
(today's optimized-sponge value, fraction toward the low end of Philip's
suggested 30-40% range to maximize clearance between the ring and the
sponge at this larger scale). Fine-logging (the multi-modal-period
diagnostic) deliberately disabled for this run to control cost -- this run
is purely about the cross-section SHAPE (does the octagon's near field
look meaningfully less diamond-like/more isotropic than the square's at
matched conditions), not the oscillation-period question. Probe script
(`exp/2026/09/08/12_15_poynting_time_evolution/probe_poynting_fine_and_
snapshots.py`) gained a `--mode square|octagon` flag for this comparison.

## SESSION 2026-09-08 EVEN YET FURTHER CONTINUED — R=50 reproduces the envelope-decay fix and reveals a clear BCC-anisotropy radiation pattern (diamond/X-shaped, plus a persistent polar-axis stripe) -- a real, separate confound from the reflection issue already fixed

Per Philip's push to scale up once the optimized-sponge signal looked
solid: reran the full fine+snapshot pipeline at R=50 (N=680,211, same
optimized sponge_fraction=0.35/damp_max=0.01, `exp/2026/09/08/
12_15_poynting_time_evolution/fine_R50_*`).

**Envelope-decay fix reproduces at bigger scale**: every shell (r=8/16/24/
30) shows genuine monotonic-ish decay over the full run (r=30: peak ~6e-8
near arrival down to ~1-1.5e-10 by t=70, a ~400-600x drop) -- confirms the
sponge fix from the R=30 run wasn't a small-domain fluke.

**Fine-window period estimate tightens, doesn't fully clean up**: 35 peaks
recorded over t=10.2-24.6, most peak-to-peak intervals now cluster in
0.3-0.55 (mean 0.412) with only 2 outliers (1.05, 1.83) -- a real
improvement over R=30's much wider scatter (0.03-2.05), and reassuringly
CONSISTENT with R=30's own mean estimate (~0.39-0.47) despite being a fully
independent run at a different domain size -- same ring geometry (ring_r,
hops unchanged by R) driving a real, reproducible near-field oscillation
mode around period~0.4-0.5, with genuine secondary-mode jitter still
riding on top (not yet explained/isolated).

**New, more important finding from the cross-section snapshots**: at R=50
resolution, the late-time (t>=15) spatial pattern is clearly NOT a smooth
isotropic falloff -- it's a pronounced diamond/X-shaped radiating
structure plus the same persistent bright polar-(z)-axis stripe already
noticed (fainter) in the R=30 plots, both stable in shape across t=15
through t=70 while the overall amplitude keeps decaying (color-scale max
log10|S| falls -7.1 -> -8.4 over that span, consistent with the real-decay
finding above). This is almost certainly genuine BCC lattice dispersion
anisotropy (`[[project_bcc_dispersion_anisotropy]]`) imprinted directly on
the radiation pattern, not noise -- reproduced independently at two domain
sizes.

**Why this matters**: the classical "equator > pole, slope~-2" dipole
comparison this whole thread has been chasing implicitly assumes smooth,
isotropic angular falloff. A strong, reproducible lattice-imposed angular
structure sitting ON TOP of whatever the "true" radiation pattern is means
the naive equatorial/polar sector-average slope fit was likely ALWAYS
confounded by this anisotropy, independent of (and probably compounding)
the reflection-contamination issue already fixed today. This is a
plausible SECOND real reason (not yet quantified) the historical -3.08/
-1.58 numbers never cleanly matched -2, on top of "field hadn't settled."

**Domain-too-small hypothesis directly tested and rejected**: Philip
proposed the diamond pattern might just be a finite-domain artifact.
Cropped both the R=30 and R=50 snapshots to the SAME absolute +-20 window
near the source (`matched_window_comparison.png`) -- the two independent
runs show essentially the SAME pattern at the same absolute scale despite
R=50 having much more separation from its boundary. If this were a
domain-size artifact, the R=50 version should look different/cleaner in
that same window; it doesn't. Fairly strong direct evidence this is a
genuine, ring-source-intrinsic effect, not a finite-domain artifact.

**Not yet done**: (1) quantify the anisotropy directly (e.g. compare |S| at
matched radius across several distinct angular directions -- axis-aligned
vs face-diagonal vs body-diagonal -- rather than the current crude
equatorial/polar sector split, which doesn't resolve this structure at
all); (2) isolate whether the polar-axis stripe specifically is a genuine
physical feature of THIS toroidal-ring source's on-axis field or a
lattice-alignment artifact independent of the source (e.g. check if a
non-ring, spherically-symmetric point-source control shows the same
stripe); (3) R=100 not yet attempted -- would further separate genuine
radiation-pattern signal from any residual near-boundary effects, at
significant additional compute cost.

## SESSION 2026-09-08 YET FURTHER CONTINUED — sponge tuning fixes the envelope (real decay instead of eternal reverberation) but NOT the fine-scale multi-modal messiness; genuine optimum found (not just "more damping is better"); pushing to R=50+ now

Philip's question ("since the dipole radiates outward, can't we just let modes
leave/be absorbed at the boundary properly?") led directly to the fix.

**Sponge quality sweep** (`exp/2026/09/08/12_15_poynting_time_evolution/
probe_sponge_quality_v2.py`/`_v3.py`): first attempt used total domain
energy H(t) as the diagnostic -- WRONG choice, came back flat (99.96-99.97%
retained) across every config regardless of tuning, because H is totally
dominated by the ring's own near-field/reactive content that never reaches
the boundary at all (this source has ka<1, a sub-wavelength/inefficient
radiator -- most energy just sloshes reactively near the source). Also
4000 steps is under 1 period of the ring's own natural oscillation
(period~32) -- nowhere near enough cycles for radiation loss to register
in a global sum regardless of sponge quality. Switched to the correct
diagnostic: shell-averaged |S| near/at the boundary (already-built from the
time-evolution work above).

**Result -- a genuine, real optimum, not a monotonic trend**: sponge_frac/
damp_max swept from (0.15,0.003)=default up through (0.35,0.01). Weak/gentle
settings are WORSE than default (deeper_gentler (0.35,0.001) increases the
outer-shell signal vs default). More aggressive damping helps up to a point
-- (0.35,0.01) cuts the r=24 signal ~30x and even the INNER r=8 signal
~3.6x vs default. But continuing past that (damp_max=0.03/0.05/0.1/0.2,
same 0.35 fraction) makes things monotonically WORSE again at every shell
tested -- classic too-strong-sponge impedance-mismatch reflection, exactly
as PML/sponge theory predicts. **Best found: sponge_fraction=0.35,
damp_max=0.01** (~3x deeper, ~3x max-damping vs the historical default).

**Re-ran the full fine-resolution + cross-section pipeline at this
optimized sponge**: genuine fix confirmed both numerically and visually.
Every shell's |S| now shows real monotonic decay over time (r=8: 1.9e-6 ->
4.4e-8, ~40x drop by t=40; r=24: ~1e-7 -> 7.4e-11, ~1000x drop) instead of
the flat/sustained-oscillation pattern under the old default sponge.
Cross-section plots (`fine_optsponge_cross_sections.png`) confirm visually
-- color-scale max steadily drops across t=15->40 (log10 max -6.8 -> -7.3
-> -7.5 -> -7.6), a real decaying pattern, not eternal reverberation at
constant amplitude.

**But**: the fine-grained peak-to-peak spacing (same fine-window analysis
as before) is STILL irregular even with the much-better sponge (0.03 to
2.05 range, no dominant period) -- so the multi-modal messiness is NOT
primarily a boundary-reflection artifact after all. Two SEPARATE issues,
only one fixed: (1) overall envelope/amplitude trend -- WAS
reflection-contaminated, NOW fixed by sponge tuning; (2) fine time-
structure/local oscillation content -- genuinely multi-modal, likely
intrinsic to how this specific ring IC's near-field couples to the BCC
lattice (multiple real superposed harmonics), not yet explained or fixed.

**Aside, not yet investigated**: cross-section snapshots (both old and new
sponge) show a persistent bright stripe along the polar (z) axis at
several snapshot times -- possibly a genuine BCC-lattice axis-alignment
discretization artifact (cf. [[project_bcc_dispersion_anisotropy]]), not
yet checked separately.

**Next (in progress)**: per Philip, pushing the same optimized-sponge
pipeline to a bigger domain (R=50, ~680k cells, running now; R=100 if that
goes well) -- more separation from the boundary gives a longer genuine
causal window and should make any residual reflection effect even smaller,
letting the R=30 result (still real but modest at 30x/1000x improvement)
be checked for whether it continues improving with domain size or has
already converged.

## SESSION 2026-09-08 STILL FURTHER CONTINUED — Poynting-dipole reproduced exactly (not stale), but its single-snapshot radial slope is an arbitrary phase-sample of a field that's STILL OSCILLATING at the measurement time, not a converged steady state -- real correction to the standing -3.08/-1.58 record

Prompted by Philip asking whether prior magnetism work helps the Poynting
dipole (a plausible prerequisite for "particles"). Checked the existing
`run/substrate/adaptive/probe_poynting_step_only.py` (AdaptiveLattice/AMR-
based, NOT the flat SubstrateLattice used in today's earlier psi_pair
work -- these are two separate representations in this codebase, see
below) fresh rather than trusting the record:

**Reproduced exactly**: R=30, hops=4, 8000 steps -> equatorial slope=-3.081,
polar slope=-1.575, ratio=3.518 -- bit-identical to the historical `now`
record. This mechanism (curl(phi)=B geometry + the real local energy-flux
law S_i,k=W_k*phi[i,k]*ubar_i,k, [[project_dst_em_field_convention_crisis]])
is NOT stale.

**New check, never done before**: does |S| at fixed radius show a genuine
CAUSAL, outward-traveling wavefront (arrival time increasing with radius),
or an already-present non-causal response? `exp/2026/09/08/12_15_poynting_
time_evolution/probe_poynting_time_evolution.py` -- logs S at 5 fixed
equatorial shells (r=8/12/16/20/24) throughout the WHOLE run, not just the
end. **Answer: yes, clean and unambiguous.** Each shell's |S| stays at
literal floating-point noise (1e-35 to 1e-78 at larger r) until a sharp,
delayed rise -- arrival times 0.262/1.544/2.313/3.338/4.364 (t in lattice
units) for r=8/12/16/20/24, implying a roughly constant propagation speed
~3-5 (same ballpark as this morning's independently-measured traveling-wave
v_phase~3.6 -- good cross-check consistency between two unrelated probes
built on different lattice representations).

**But**: examining the ALREADY-COLLECTED trace at late time (t>10, well
past every shell's arrival) shows |S| at each shell still genuinely
OSCILLATING, not settling -- refitting the equatorial log-log slope at
EVERY late-time snapshot (122 points, t=10.01 to 41.03) gives mean=-3.627,
std=1.184, range -6.19 to -1.40 -- a huge, persistent swing. Checked
whether this damps over time (three windows, t in (10,20]/(20,30]/(30,41]):
std stays 0.5-1.6 in all three, mean drifts -3.81/-2.89/-4.13 -- **no sign
of convergence toward a stable value anywhere in this 8000-step run.**

**Conclusion**: the standing -3.08/-1.58 record is real and reproducible
as that ONE arbitrary snapshot's value, but it was never actually a
converged steady-state radiation-pattern slope -- nobody previously checked
whether the field had settled before taking it. The real "does DST's
dipole radiation match classical -2" question is still genuinely open, not
answered (in either direction) by the existing record. Two live
explanations, not yet distinguished: (a) genuine slow-decaying transient --
a longer run would show the oscillation amplitude finally damp toward a
static profile; (b) persistent standing-wave/reverberation within the
finite sponge-bounded domain (the sponge may not be a good enough absorber,
[[project_static_boundary_consistency_test]]-adjacent concern) -- would
need either a much better absorbing BC or a period-averaged (not
single-snapshot) slope measurement to get a trustworthy number either way.
Not yet done: check if oscillation period matches an obvious round-trip
reflection time (domain_size/v_phase-ish), which would directly implicate
(b); or extend the run 2-4x longer to test (a) directly.

**Aside, worth remembering**: this whole Poynting-dipole thread runs on
`equilibrium.lattice.AdaptiveLattice` (the AMR/C++-backed representation),
completely separate from the flat `equilibrium.substrate.lattice.
SubstrateLattice` used by today's earlier psi_pair/c2-calibration work.
Per [[project_amr_update_order]] (validate on the simpler flat layer
first, port to AMR only once settled), nothing from today's psi_pair work
needs porting to the AMR lattice now -- `discrete_curl.py`'s
`triangle_triples`/`leaf_nb_rows` already bridges the two representations
if/when that becomes appropriate.

## SESSION 2026-09-08 FURTHER CONTINUED — resolution scan: the ~9-12% velocity deficit does NOT shrink with resolution (not a discretization artifact); promoted the whole (phi,psi) construction into a real module + 4 permanent regression tests; two steers from Philip folded in (judge by phenomena not narrative; Pauli-as-smoothness idea parked)

**Resolution scan** (`exp/2026/09/08/11_11_resolution_convergence/`): held
the calibrated weighted c2=0.7 fixed and re-ran the short-horizon
co-propagation diagnostic at n_cells_per_wavelength = 8/12/16 (coarser
discretization of the SAME wave, same domain). If the residual were a
discretization artifact it should shrink toward the continuum limit
(larger n_cells_per_wavelength). It does not: frac_deficit = 8.6% (n=8) ->
9.2% (n=12) -> 11.7% (n=16) -- flat to slightly GROWING with better
resolution, no blowup at any point. **This is a genuine, resolution-
independent property of this specific (phi,psi) construction at c2=0.7,
not a finite-lattice-spacing artifact that would vanish in the continuum
limit.** Whether that ~10% is itself a real predicted signature of this
candidate mechanism or a sign that some other structural piece is still
missing remains open -- not resolved by this scan, which only rules out
"it's just under-resolved."

Philip's reframing, folded in going forward
([[feedback_narrative_vs_phenomena]]): the acceptance criterion used
throughout this whole thread ("does B co-propagate the way classical E/B
mutual-regeneration says it should") is itself a textbook NARRATIVE, not a
directly observed phenomenon -- classical EM's own "self-propagating"
leapfrog story may not be the literal mechanism even conventionally. The
actual falsifiable targets are real observations: photovoltaic conversion
(photons couple to the same charge/current DOF that drives a circuit --
whatever DST's photon is, it must interface with charge the way real light
does), Bell-test polarization correlations, energy/momentum transport. A
candidate failing the co-propagation-narrative test is informative but not
by itself dispositive; conversely passing it isn't sufficient either. Also
flagged, unrelated to this thread and not investigated: Pauli exclusion as
a possible downstream local-smoothness effect on neighbouring windings'
orientation (a wave's local vector can't point wildly differently from its
neighbours if the gradient stays ~smooth) -- see
[[project_pauli_exclusion_smoothness_idea]], parked.

**Promoted to permanent code + tests** (per [[feedback_save_validations_as_tests]]):
`equilibrium/substrate/psi_pair.py` (circ_D/scatter_DT as reusable,
documented functions, C2_CALIBRATED=0.7) + `tst/substrate/test_psi_pair.py`
(4 tests, ~28s, all pass): the scatter_DT/circ_D adjoint identity holds
exactly; c2=0 stays inert (no spurious curl, guards the frozen-B identity);
no blowup across the full previously-swept c2 range 0-5 (guards against
reintroducing the earlier unconditional-instability bug); the calibrated
c2 produces a genuinely nonzero, bounded, non-frozen curl disturbance on a
small fast test lattice (the qualitative "recipe" captured as a hard
regression, independent of whether the mechanism is ontologically final).
This is now real, durable, re-runnable machinery -- not just exp/ scratch
scripts -- capturing the whole thread's empirical "recipe of effects" even
without a complete ontological explanation for the residual ~10%.

## SESSION 2026-09-08 CONTINUED — weighting circ_D/scatter_DT with the base system's own W_HEX/W_SQ face weights (the "next" queued below) cuts the residual roughly in HALF again and turns it into a clean constant-rate slip instead of noisy oscillation-plus-drift -- still not full co-propagation, but the best result of the whole thread

Rebuilt `circ_D`/`scatter_DT` (`exp/2026/09/08/10_32_weighted_circ_scatter/`)
with the SAME W_HEX=3/W_SQ=1 weights the base (u,phi) system already uses
on both its own legs -- confirmed geometrically exact for all 12 TRIANGLES
families (h1,h2 always hex, mirror(sqk) always sq, so the weighting is
literally `3*phi[a,h1] + 3*phi[b,h2] + 1*phi[c,m]` on every family, no
per-family branching needed), with `scatter_DT` built as the EXACT
transpose of that weighted `circ_D` (same per-leg weight applied when
depositing back) -- the correct adjoint pairing, not a new free parameter.

Re-swept c2 (two rounds, since the weighted operators are intrinsically
~3-7x stronger so the matching scale moves): clean optimum at **c2~0.7-1.0**
(short-run full_slope ~-0.31, vs the unweighted best -0.71 at c2=2.0 --
already half the residual). Long-horizon (5000-step) verification at
c2=0.7 confirms this holds up: slope_first_half=-0.3845,
slope_second_half=-0.3813, slope_last_quarter=-0.3618 -- remarkably
CONSTANT across the whole run (unlike the unweighted c2=2.0 case, whose
per-window slopes varied wildly, -0.62/-0.71/-0.18, an artifact of fitting
through a large, noisy oscillation). This weighted version's gap trace is
still noisy step-to-step but its LONG-TIME trend is now a clean, small,
well-determined constant slip, not chaotic beating -- meaning B has settled
into steady quasi-phase-locked propagation at v_B~3.25, about 90% of the
wave's own v_phase=3.62, rather than genuinely getting left behind.

**Net progress across the whole 2026-09-08 thread**: v_B went from ~0
(c2=1e-3, unweighted -- B frozen where deposited, the original failure) to
~2.9 (c2=2.0, unweighted, calibrated -- 80% of v_phase but noisy) to ~3.25
(c2=0.7, weighted+calibrated -- 90% of v_phase, clean constant rate). Each
step (calibration, then structural weighting) produced a real, measured,
reproducible improvement in the SAME direction predicted by the underlying
physical diagnosis (speed/shape matching) -- not tuning to make a number
look better, an actual falsifiable mechanism doing what it was derived to
do. Per Philip's standing note that discrete-lattice deviations from a
continuum expectation are not automatically a fail: a clean, constant ~10%
velocity deficit is a qualitatively different (and much more promising)
result than the original unbounded wake, though it is not yet the clean
zero-gap co-propagation the phenomenology criterion asks for.

**Not yet done / open**: is the remaining ~10% a genuine, irreducible
lattice/dispersion effect (a real "index of refraction"-like correction
this candidate mechanism would predict, potentially itself a falsifiable
signature rather than a defect) or still a fixable mismatch (e.g. the
overall dt/leapfrog ordering between the (u,phi) and (phi,psi) legs, or a
finer c2 tune between 0.7-1.0, was not swept below the coarse 0.1 grid
spacing used here)? Also not checked: whether this same weighted
(phi,psi)+c2=0.7 construction still passes the ORIGINAL wave-plus-B
sanity/stability checks from 2026-09-07 (baseline c2=0 gives exactly zero
B, no blowup over the full run) -- assumed inherited from the algebra but
not re-verified independently on this exact weighted code path.

## SESSION 2026-09-08 — displacement-current-style derivation for the (phi,psi) trailing-wake failure: speed-matching found and calibrated, cuts the lag ~5x, but does NOT fully fix it -- real residual secular drift + oscillation remains

Followed up on the queued next step below (derive an actual u-dependent/
derived curl-sourcing term rather than another bolted-on analogy).
Diagnosis: the (phi,psi) pair from 2026-09-07 night (`dpsi/dt=-circ_D(phi)`,
`dphi/dt += c2*scatter_DT(psi)`) IS already the theoretically-correct fix
for the earlier "bare curl(curl(phi)) is unconditionally unstable" failure
(`naive-reverse-curl-term-unconditionally-unstable.txt`) -- a genuine
symplectic conjugate pair, confirmed stable. The missing piece was never
the STRUCTURE, it was that `c2` (the Ampere-leg coefficient) was an
undesigned "small perturbation" (1e-3), never checked against any matching
condition. This is DST's own analogue of Maxwell's `eps0*mu0=1/c^2`: real
E/B co-propagate only because that product is pinned to the wave's actual
speed, not because curl-curl structure alone guarantees it.

Per Philip's steer not to force this into a continuum-derived formula
(discrete lattice deviations from continuum are expected, not an automatic
fail), calibrated `c2` empirically on the real lattice instead of
analytically: reused the exact wave-plus-(phi,psi) probe geometry from
`00_10_wave_plus_phi_psi_b_field`, swept `c2` from 1e-3 to 20 (four rounds,
`exp/2026/09/08/09_34_c2_speed_matching_sweep/`), measuring the growth rate
of `B_centroid_z - u_centroid_z` (the "gap"). No blowups anywhere in the
swept range (0.001-20) -- confirms the (phi,psi) pair really is
stably-behaved wherever the earlier bare self-loop was not.

**Result**: clean, reproducible, well-resolved minimum at **c2~2.0**
(gap-growth slope -0.71, vs -3.2 to -3.7 at the original c2=1e-3 -- B
genuinely moves at ~80% of the wave's own speed instead of ~0%, a real ~5x
improvement, confirming the speed-matching diagnosis was directionally
correct).

**But**: re-ran c2=1e-3 vs the calibrated c2=2.0 out to the full 5000-step
horizon (`probe_c2_long_verify.py`). c2=1e-3 reproduces the original
linear, unbounded wake exactly (slope ~-3.2 to -3.7 throughout,
gap_final=-88 at t=25.6). c2=2.0 is genuinely better (slope ~-0.6 to -0.7,
gap_final=-19.25) but the full trace shows an **oscillating gap (period
~9-10 time units, amplitude ~7-10) riding on top of a real, still-nonzero
secular drift** -- successive cycle peaks drift from +0.5 (t=3.1) to -12.0
(t=23.1). Not a bounded, phase-locked co-propagating solution; a slower,
oscillating wake.

**Diagnosis of the residual**: a single scalar `c2` can only match the
(phi,psi) subsystem's phase velocity at ONE wavenumber (the wave's
dominant K). `circ_D`/`scatter_DT` as built are geometrically unweighted
(raw per-face phi sums, no W_HEX/W_SQ distinction) -- almost certainly
giving the (phi,psi) subsystem a dispersion relation with a different SHAPE
(not just different overall scale) than the base system's own `omega^2(K)`
(doc/dst.md, built from the genuine W_HEX=3/W_SQ=1 + hex/sq distance
weighting). A wavepacket has finite spectral width, so a shape-mismatched
dispersion relation gives real, unavoidable group-velocity spread/beating
on top of any single-K amplitude calibration -- plausibly exactly the
oscillation-plus-drift seen here.

**Next (not yet done)**: rebuild `circ_D`/`scatter_DT` with the SAME
W_HEX/W_SQ face weighting the base (u,phi) system already uses (not the
bare unweighted version built for the 2026-09-07 diagnostic), so the
(phi,psi) dispersion relation is structurally derived from the same
geometry rather than merely amplitude-matched at one point -- then re-run
the same co-propagation test. If that closes the residual, this would be
the actual derived, from-DST's-own-structure Ampere/Faraday-like sector; if
it doesn't, the group-velocity-mismatch diagnosis above is wrong and
something else is going on.

Files: `exp/2026/09/08/09_34_c2_speed_matching_sweep/` (probe_c2_sweep.py +
4 rounds' logs/traces, probe_c2_long_verify.py + long_verify_log.txt).
