# Entries for 2026-09-21 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## SESSION 2026-09-21 — the double-ring dipole electron IC is shown, by two independent arguments, to have the wrong topology (closed/neutron-lineage, not open/proton-like); a real, resolution-independent directional bias found and characterized (then correctly de-scoped) in the shared discrete-curl B-reconstruction; dipole_mode_ic.py rediscovered as an existing, better-shaped, already-validated l=1 dipole construction

Long, wide-ranging session starting from yesterday's Coulomb/point-source
work, moving through the electron dipole IC's perturb-and-restoring-force
behavior, into a full reassessment of the IC's own topology. Full detail:
`exp/2026/09/21/electron_dipole_perturb_restoring/status.md` and this
session's memory writes (listed below). Read those before continuing any
of this thread.

**Perturb-restoring-force test on the double-ring dipole (v1/v2/v3,
[[project_dipole_perturb_restoring_force]])**: v1 (dynamic
`step_lam_relax` nonlinearity) found an apparent alpha-monotonic
confinement signal that did NOT survive a spatial-localization check --
it was broadband nonlinear divergence between paired trajectories, not
real localized trapping (three real diagnostic bugs found/fixed along the
way: a degenerate per-cell-multiplicative perturbation that can never
break divergence-freeness, single-world background contamination from
the dipole's own large static field, and boundary-reflection
contamination at late times -- all three are reusable lessons for any
future perturbation test on this IC family). v2 (Philip's correction:
lam should be STATIC/invariant for an electron "at rest," not dynamically
relaxing) gave a real, spatially-confirmed but PARTIAL/LEAKY trapping
signal, monotonic in lam_target, that survived the same spatial check v1
failed. v3 (free, unpinned u-deficit, testing whether charge persists)
found NO persistence in ANY tested config (vacuum, dipole+lam=0/1/2) --
the deficit always fully disperses in the clean pre-reflection window,
though static lam does measurably slow the dispersal rate. Re-run at
proper resolution (R=80, electron_scale=50, vs the initial R=30/scale=4.5
toy size -- Philip caught this, "we've done R180 before" for a related
poloidal+toroidal construction) confirmed the same qualitative picture at
cells_across=100: vacuum and lam=0-dipole exactly identical (confirms
linear superposition holds), static lam genuinely and monotonically slows
dispersal (frac_rate 100%->62%->42.5% at lam=0/1/2), but nothing is
truly invariant, not even the strongest lam tested.

**g-factor / winding-number thread**: a from-scratch physics argument
(setting a self-circulating mass's angular momentum L=m_e*c*r to the
electron's real spin ħ/2 gives the Zitterbewegung radius; the electron's
actual g~2, not the classical single-loop g=1, implies effectively TWO
windings) converges with two things already on record (the double-ring
dipole itself, and the Dirac chirality-mixing two-component mass
argument) -- a real, independently-derived reason "2" keeps showing up.
Built a dimensionless (Gamma/H) ratio test comparing a single-ring
current loop to the double-ring dipole on the same geometry, avoiding
absolute-unit calibration (this project has none yet). Result: ratio
came back 0.0056, nowhere near the target ~2, confounded by a real
energy-normalization mismatch (the dipole is volume-filling, the ring is
concentrated -- not apples-to-apples). Layering a toroidal admixture on
top of the poloidal dipole (tokamak/stellarator-motivated, since
confinement devices need poloidal+toroidal together) moved the ratio
FURTHER from 2, not closer, while leaving the structure an exact static
fixed point at every amplitude tested -- a real negative result, though
still under the same normalization confound. The neutron/Hopf-link
construction was considered and correctly rejected as a stand-in for
"poloidal+toroidal" (Philip: neutron is inherently metastable/decaying,
a bad reference for an at-rest structure) -- real confinement requires a
TUNED ratio, not just "both components present."

**IC topology reassessment -- the session's main structural finding
([[project_electron_ic_topology_wrong]])**: Philip asked why a plotted
slice of the dipole's own energy density looked like the neutron/
spheromak, not the proton. Two independent lines of evidence confirm
this suspicion: (1) analytic -- `init_electron_dipole_ic`/`_vecpot` uses
k=4.493/R_sph, the self-confined force-free Taylor-state eigenvalue,
which makes the field EXACTLY zero outside R_sph*scale by construction
(already flagged 2026-09-16, under-weighted until now); (2) visual --
directly plotting the dipole's own phi-energy-density in a meridional
slice gives two separate, same-sign, hollow-gapped lobes, matching real
PHARE reference images Philip supplied (`phare_plasmoses_t0.000.png`,
labeled a pre-decay neutron-like plasmoid pair) rather than PHARE's own
actual-dipole reference (`phare_dipole.png`, a single compact peak). Both
arguments converge: the current electron IC is topologically a closed,
hollow, neutron-lineage torus, not an open, compact, proton-lineage
dipole -- likely the root cause of most of this session's negative
confinement/charge/g-factor results, not a failure of those specific
mechanisms.

**Two existing, better-shaped constructions rediscovered** (both from
2026-09-02, found only after Philip pointed out they must exist):
`equilibrium/substrate/dipole_mode_ic.py` (u=U, phi=grad(U), a genuine
l=1 Helmholtz dipole mode, K=4.493 but WITHOUT the extra rho-factor that
made the old IC hollow -- "the current, validated production IC" per its
own docstring) plots as a single compact energy-density peak, matching
PHARE's real dipole image, AND has genuine nonzero built-in u structure
(odd in z) -- resolving both the shape problem and the missing-charge
problem in one existing module nobody had checked this session.
`exp/2026/09/02/20_00_proton_loop_dipole/probe_curl_loop_dipole.py` --
a bare current LOOP (Ampere's-law mechanism, not hand-built poloidal
superposition) correctly reproduces the classical circular-loop dipole
field SHAPE via curl(phi)=B, on-axis and equatorial, over a wide dynamic
range.

**Discrete-curl directional bias, found/characterized/then correctly
de-scoped ([[project_discrete_curl_directional_bias]])**: `b_vec_
vectorized` (used only in the loop-dipole validation above, not in this
project's real physics) has a real, resolution-independent (checked
R=80 vs R=140, not shrinking) directional bias, ~28-29% on the dipole
lam profile, confirmed a third way by re-analyzing the 09-02 loop test
theta-resolved (the original "clean ~1.3x calibration constant" was an
artifact of azimuthal averaging hiding real 1.12-to-1.59 variation with
angle). A constant-vector/single-bad-reference-cell explanation was
tested directly and RULED OUT (the bias shrinks with radius, roughly
tracking the true field's own falloff, not staying constant). Final,
important self-correction: this whole investigation was chasing the
accuracy of B, which this project already decided (twice, independently)
NOT to model -- and even the theta-resolved cosine fit implicitly assumed
continuity that a 14-fixed-direction discrete lattice has no right to
have. Both are good reasons to stop chasing this specific mechanism, not
reasons the bias measurement itself is wrong -- kept as a permanent
record, not pursued further.

**Housekeeping**: MEMORY.md compacted (was approaching its read-size
limit) -- every entry now a single tightened line, detail kept in the
per-topic files, nothing substantively lost. New feedback memory
[[feedback_no_scratch_for_plots]] -- a prior session's (09-17/09-19)
plots were lost because they went to `/tmp` instead of the repo; this
session's own plots (correctly) all went to
`exp/2026/09/21/electron_dipole_perturb_restoring/`.

**Queued next** (see `next` for the actionable version): rebuild the
confinement/charge-persistence/g-factor tests on `dipole_mode_ic.py`'s
corrected foundation instead of the flawed double-ring dipole; do a real
quantitative comparison against the raw PHARE data at
`/home/p/EM_B.vtkhdf` (only a PNG-level visual comparison was done this
session); retest the winding-number=2 g-factor hypothesis using
phi-circulation diagnostics only (no curl/B, per this session's own
redirect); the u-wave/phase-coincidence magnetism idea from earlier this
session remains flagged and unbuilt; Saha-noise stability stress-testing
remains queued for AFTER a genuinely invariant configuration is found (still
hasn't been, on any construction tested to date).

---


# Entries for 2026-09-21 moved from repo-root `next` on 2026-10-25 (verbatim, newest first)

## QUEUED 2026-09-21 — electron IC topology correction is the priority thread; four concrete follow-ups, roughly in priority order

Context: `now`'s 2026-09-21 entry (read that first). The session's main
finding is that `init_electron_dipole_ic`/`_vecpot` (the double-ring
dipole, used in essentially every electron test this project has run
since 2026-07-10) has the wrong topology -- closed/hollow/neutron-
lineage, not open/compact/proton-lineage -- confirmed two independent
ways (the k=4.493 self-confined eigenvalue's zero-external-field math,
and a direct visual match to real PHARE plasmoid-decay reference images
rather than PHARE's actual-dipole reference). This likely explains most
of the negative confinement/charge/g-factor results logged across this
project's whole electron-dipole history, not just today's.

### 1. Rebuild today's tests on `dipole_mode_ic.py` instead of the double-ring dipole

`equilibrium/substrate/dipole_mode_ic.py` (u=U, phi=grad(U), genuine l=1
Helmholtz dipole, already on the AdaptiveLattice cluster API, no
adaptation needed) plotted as a single compact energy-density peak
matching PHARE's real dipole image, with genuine nonzero built-in u
structure -- unlike the double-ring dipole, which showed hollow two-lobe
structure and never self-sourced any charge in any tested config. Redo
the perturb-and-restoring-force / charge-persistence / g-factor tests
from this session (`exp/2026/09/21/electron_dipole_perturb_restoring/`)
on THIS construction instead. Highest-priority item -- it's the corrected
foundation everything else should build on.

### 2. Quantitative PHARE comparison, not just visual

Real PHARE AMR magnetic-field data exists at `/home/p/EM_B.vtkhdf`
(Level0: 1,110,392 points, Level1: 65,429 points, PointData is 3-component
B, AMRBox geometry per level) -- only inspected the HDF5 structure this
session, never actually reconstructed a proper sliced/positioned
comparison against it. Today's PHARE comparison was PNG-vs-PNG visual
only (`phare_dipole.png`, `phare_plasmoses_t0.000.png`). A real
quantitative check (radial profile, or at minimum a matched-orientation
slice) against the raw data would be much stronger evidence than the
visual match already found.

### 3. Retest the winding-number=2 g-factor hypothesis, phi-circulation only

The Zitterbewegung-radius argument (L=m_e*c*r=ħ/2 fixes r; the electron's
real g~2 vs the classical single-loop g=1 implies two windings) converges
with two independent things already on record (double-ring dipole omni-
balance, Dirac chirality-mixing). Today's (Gamma/H) ratio test of this on
the double-ring dipole came back 0.0056 (wrong direction, badly
confounded by a real energy-normalization mismatch between a thin ring
and a volume-filling dipole) -- worth retrying on the corrected
`dipole_mode_ic.py` foundation, and reusing `phi_poloidal_scan`/`phi_
ring_circulation` (both confirmed to use all 14 face directions
correctly) rather than anything touching `b_vec_vectorized`/curl(phi)=B,
per this session's own redirect (see `now`'s discrete-curl-bias section --
real bug, correctly de-scoped, not worth reopening).

### 4. Lower priority / exploratory, still open from this session

- u-wave/phase-coincidence magnetism idea (project_dst_parker_spiral_
  rotating_frame_idea's 4th candidate) -- still entirely unbuilt, needs a
  phase function over the full 3D volume, not just on a ring.
- Saha-noise stability stress-testing -- blocked until a genuinely
  invariant (not just partially/leaky confined) configuration is found on
  ANY construction; still hasn't happened on any test run to date.
- [[project_dst_lam_asymptotic_zero_idea]] -- flagged, not pursued,
  low priority.

### Explicitly deprioritized, closing the loop from 2026-09-19

The two background jobs queued 2026-09-19 (`c2_finish.py` weighted-c2
EM co-propagation refine, `v8b_resume.py` free-electron phase-scan) were
never run. Session 2026-09-21 examined both directly and made a
deliberate call NOT to finish them: `c2_finish.py`'s whole construction
is a hand-bolted-on, non-native scatter term with a free-fitted constant
(calibration to a target, not derivation -- the exact pattern
[[feedback_mechanism_not_description]] warns against), and `v8b_resume.py`
continues a single-ring free-electron construction already established
this session (and in [[project_chirality_mixing_two_component_mass]]) as
structurally the wrong topology for the question it was trying to answer.
Neither is queued further -- superseded by the electron-IC-topology work
above.
