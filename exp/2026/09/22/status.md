# Entries for 2026-09-22 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## SESSION 2026-09-22 — Dirac/ontology tangent -> full dipole-IC-gallery audit -> `init_electron_dipole_ic_vecpot` confirmed circulation-equivalent to the spheromak (invalidating all of 09-21's perturb-restoring/g-factor results) -> "circulation-free" itself corrected to "must not erode dynamically" -> winding investigated as DST's own fundamental (not continuum-field) mass mechanism -> tokamak/stellarator-motivated discrete helical-winding construction begun, real but unfinished

Very long, wide-ranging session. Started as a pure conceptual tangent (Dirac
equation/ontology) at Philip's request, then pivoted into the most thorough
IC-construction audit this project has done. Ended low on context with a
real, promising new construction thread (discrete winding-as-mass) started
but not finished. Read this whole entry before continuing -- it supersedes
several working assumptions carried in `next`'s old 2026-09-21 queue below,
without fully rewriting that entry (kept for history).

### Conceptual throughline (Dirac tangent, still valid, motivates everything below)

Discussed the Dirac equation's ontological content: the mass term is
literally the COUPLING between two otherwise-massless chiral (Weyl)
components -- rest mass isn't a substance, it's a relation between two
coupled pieces, structurally the same kind of claim
[[feedback_reproduce_dont_reify]]/[[project_lam_dual_role]] already make
about DST's own mass concept. This reinforces (not just repeats)
[[project_chirality_mixing_two_component_mass]]'s standing requirement:
E=mc^2-like rest energy needs (at least) two genuinely COUPLED components,
not one field doing all the work, and not two components merely
superposed. Zitterbewegung = the beat frequency between the two chiral
pieces; a "resting" particle is actually two lightlike (c-speed) circulations
beating together. Philip's later framing distilled this to a concrete bar:
"if we have an IC, with two components, both travelling at c, and it just
sits there and doesn't decay, that's a good start" -- don't chase full
physical realism (charge, exact confinement) before clearing that first.

### 1. Full dipole/electron IC gallery audit -- `exp/2026/09/22/dipole_ic_gallery/`

Built (`plot_all_dipole_ics.py`, `dipole_ic_gallery.png`) an at-a-glance
comparison of every dipole IC construction in the repo, matched domain
(R_DOMAIN=80, R_sph=1.0, scale=50.0). Philip caught a real methodological
flaw immediately: a LINEAR-scale energy-density plot cannot distinguish
open (real exterior field) from closed (exactly zero exterior) topology --
a current loop's near-source peak swamps the same color scale that would
show the difference, so "two blobs vs one peak" (what 09-21's PNG-vs-PNG
PHARE comparison also implicitly relied on) is a CONFOUNDED diagnostic, not
a real topology test.

Fixed with `plot_per_ic_diagnostics.py` (`per_ic/*.png`): LOG-scale
meridional + equatorial slices, and a radial mean-energy-density profile
(semilogy vs r) that directly shows hard-floor-at-boundary (closed) vs
continuing-tail (open). **Load-bearing new finding**: `seed_dipole_ic`
(`dipole_mode_ic.py`, treated since 09-21 as "the corrected production IC")
is ALSO topologically closed -- same k=4.493 eigenvalue, same exactly-zero
exterior past r=R_sph*scale as the flagged-wrong double-ring dipole and the
spheromak. The 09-21 shape fix (single compact peak vs. hollow two-lobe
torus) never actually fixed the open/closed question; nobody had checked
until this radial-profile diagnostic existed. Only a bare current-loop
reference (Ampere's law, no eigenvalue cutoff) showed genuine open exterior
among constructions tested.

Philip also supplied a real PHARE post-spheromak-decay dipole reference
image (saved permanently: `exp/2026/09/22/dipole_ic_gallery/reference/
phare_post_spheromak_decay_dipole.png`) showing a bright elongated CENTRAL
COLUMN + two small flanking blobs + a smooth diffuse OUTER SHELL -- a
sharper target than either "single peak" or "two lobes". Flagged that the
non-vecpot `init_electron_dipole_ic` (dismissed in the first pass as
"doubly defective" from a messy `divergence_clean` correction) actually
shows real substructure resembling this reference (central peak, ring-gap
features, a genuine non-hard-floor tail past r=50) -- deserves a real
second look, not the dismissal it got.

### 2. Macro ring circulation measured -- `init_electron_dipole_ic_vecpot` is spheromak-equivalent, not just spheromak-shaped

Philip's key catch: a static energy-density SHAPE match (why does the
"electron" look like the spheromak) isn't the right test for "is this
dynamically the same kind of object" -- the spheromak's actual, MEASURED
decay mechanism ([[project_spheromak_decay]]) is specifically its toroidal
circulation unwinding (`circ_B`: -1.447 -> ~0, while poloidal `circ_A`
survives). `measure_ring_circulation.py` used `phi_ring_circulation`
(already validated for exactly this in
[[project_curl_full_dipole_spheromak]]) on the actual seeded phi:

| Construction | circulation |
|---|---|
| `init_electron_dipole_ic_vecpot` | **+632.96** |
| `init_spheromak_ic` (control) | +491.86 |
| `init_electron_dipole_ic` (non-vecpot) | -1.17e-15 |
| `seed_dipole_ic` (`dipole_mode_ic.py`) | 0.0 exact |

Root cause confirmed real, not just suspected: `init_electron_dipole_ic_
vecpot`'s own docstring had flagged (unchecked, until now) that its vector
potential `A=(-y,x,0)*cf(r)` reuses the exact same Cartesian direction as
`apply_toroidal_rotation`'s real toroidal-FIELD formula, i.e. a
toroidal-field-shaped formula misapplied as a vector potential.

**Consequence, written into project memory this session**: `init_electron_
dipole_ic_vecpot` is the ONLY construction used anywhere in
`exp/2026/09/21/electron_dipole_perturb_restoring/` (confirmed by
repo-wide grep -- every one of 09-21's ten scripts: perturb-restoring v1/v2,
charge-persistence v3/v4/v5, both g-factor tests, `circulating_ring_
three_way.py`, both snapshot plots). **All of 09-21's results are
invalidated as electron-confinement evidence** -- they were measured on an
object dynamically equivalent to the decaying spheromak, not a candidate
electron. Full corrections written to [[project_electron_ic_topology_wrong]]
and [[project_dipole_perturb_restoring_force]] (both got dated 2026-09-22
UPDATE sections), `doc/mems/MEMORY.md` index lines updated, `next` got a
new top QUEUED 2026-09-22 section (now itself partly superseded further
below -- see "Queued next" at the end of this entry for the live version).

### 3. CORRECTION: "circulation-free" was too strong a bar -- the real discriminator is dynamical erosion, not a static t=0 reading

While scoping a torus-knot construction (below), a prototype closed ring
seeded with a bare constant-amplitude current showed real circulation
(12.9-13.6 on a small ring, 107-108 on a large one) just from being a
closed current-carrying loop -- expected, not a bug: ANY topologically
closed loop with current flowing around it has nonzero circulation by
definition. Re-checking [[project_spheromak_decay]]'s own numbers: the
SURVIVING ring (`circ_A`) was never zero either (+0.007 to +0.032) -- what
distinguished it from the DECAYING ring (`circ_B`) was EROSION RATE under
real `step_cross` dissipative dynamics over time, not a static
zero-vs-nonzero circulation value at t=0. Corrected in
[[project_electron_ic_topology_wrong]] (second dated section) and `next`.
This does NOT undo finding #2 above -- vecpot-dipole's +632.96 is still a
real, large, same-order-as-spheromak number, a meaningful red flag even if
"exactly zero" was never the right bar to clear.

### 4. Plain-`step()`-only persistence test (`step_cross` OFF) -- confirms vecpot dipole IS an exact fixed point, but shows the test itself can't discriminate anything

Philip: `step_cross` is specifically DST's translation/inertia mechanism
(confirmed directly from `doc/dst.md`'s own "Through-cell crossing"
section and `step_cross`'s docstring: "enabling the collective gate
condition for winding translation") -- NOT a general decay test, and its
own ontological correctness is unconfirmed ("we're trying to be
progressive"). `test_plain_step_persistence.py` ran 5 constructions 300
steps under plain `step()` only, `step_cross` never called:

- `init_electron_dipole_ic_vecpot`: energy/circulation/shape frozen to
  numerical precision -- confirms (for the first time quantitatively) a
  claim this project had only ever carried forward unchecked.
- `init_spheromak_ic`: circulation ALSO frozen (<0.001% drift) -- its
  known decay was ENTIRELY produced by `step_cross`/`cross_scale`, never
  shown under plain reversible dynamics. Real internal RESHAPING happens
  anyway (radial-profile shape_drift up to 11.10, >1000% off in some bins)
  even while energy/circulation stay frozen -- a real dynamic neither
  scalar diagnostic alone would catch.
- `init_electron_dipole_ic` (non-vecpot), `seed_dipole_ic`: both stable/
  bounded, consistent with earlier findings; non-vecpot's `divergence_
  clean` confirmed to have actually CONVERGED (final div_rms 6.73e-10,
  not stuck at the iteration cap) -- resolves that open question
  positively, the earlier "possible solver artifact" caveat is lifted.
- Plain 8-tile ring (`init_toroidal_winding` on the minimal square ring):
  the one construction with real non-trivial dynamics -- energy dips to
  0.9933 by step 50, recovers to ~1.0006 by step 200-300. Genuine
  oscillation, not decay or blowup.

**Epistemic upshot, important**: reversible dynamics conserves energy by
construction -- nothing can show "decay" under plain `step()` regardless of
circulation content, so this test cannot discriminate the vecpot dipole
from the circulation-free constructions. With `step_cross` off-limits as a
general test (per Philip, above), this project currently has NO validated
way to test "will this decay" for a stationary structure. `step_phi_
diffuse` (existing, described elsewhere as genuine irreversible
face-flux-diffusion/reconnection, unrelated to translation) was flagged as
a better-motivated candidate for that role than `step_cross` -- NOT YET
TRIED, still open.

### 5. Windings investigated as DST's own fundamental mechanism (not a continuum-field borrow)

Per Philip's redirect ("we're not expecting full physical realism
immediately... let's investigate windings, they're a fundamental
component"), read `doc/dst.md` itself rather than continuing to patch
Taylor-state variants. Two genuinely DIFFERENT things doc/dst.md calls
"winding":
1. **Charge**: does flux close (neutral, neutron) or escape (charged,
   proton) -- binary, topological, no charged-closed case exists.
2. **Mass/inertia**: a purely DISCRETE, topological closed loop across N
   tiles -- "the electron is light because its minimal toroidal winding
   spans 8 tiles." Field-amplitude-independent. This is `winding_number`
   (pure path topology, integer-valued) -- NOT `phi_ring_circulation`
   (how much field executes that path, continuous). **These are separable
   quantities** -- a topologically-closed loop carrying near-zero net
   circulation is not a contradiction, nothing tested before this session
   had ever actually tried building that combination (every construction
   tested was a continuum Taylor-state/Bessel field, not a discrete
   tile-loop at all).

**Also surfaced, unresolved, flagged for Philip's decision, NOT this
session's to resolve**: `doc/dst.md` itself is internally inconsistent
about what an electron IS -- passive u<0 deficit-moat companion ("no
meaning without an interior") vs. independent minimal winding vs. language
implying electron/proton are the same kind of object at opposite
chirality (contradicts proton being explicitly non-winding/open-knot).
This project has hit this exact wall at least 3 times before (07-10,
07-12, 09-18 -- see [[project_dst_electron_topology]]'s flat contradiction
"8-tile ring is photon topology, NOT electron topology" vs. doc/dst.md's
own Inertia section) and always defaulted to "use the double-ring dipole
anyway" instead of resolving it. Not resolved this session either --
genuinely open, load-bearing, needs a decision before too much more
construction work.

### 6. Tokamak/stellarator-motivated discrete helical winding -- real progress, unfinished

Philip's physical framing: a tokamak/stellarator's real confinement
mechanism is ROTATIONAL TRANSFORM -- pure toroidal field alone doesn't
confine (curvature/grad-B drift); you need field lines HELICALLY coupling
major (toroidal) and minor (poloidal) circulation into ONE path, not two
superposed fields (`curl_full_dipole_ic.py` already shown, this session, to
still carry real circulation and still be closed -- superposition doesn't
help). "Poloidal windings around a central torus" = literally winding
copper wire around a core (a coil), at the lattice's own discrete scale.

**What was tried, in order, all in `exp/2026/09/22/dipole_ic_gallery/`**:
- First prototype (`prototype_torus_knot_winding.py`): flat ring on the
  minimal 8-tile SQUARE pattern + a z-ONLY detour ("corrugated ring", zero
  rho variation, NOT genuinely helical). Closes cleanly at every hump
  height tested, `winding_number`=1.0000 always. This is what surfaced
  finding #3 above (real circulation from a bare current, not a bug).
- **DISCARDED for using the wrong ring regime**: Philip caught that the
  minimal 8-tile SQUARE ring is exactly the regime this project's own code
  already flags as anisotropic ("BCC k.L dispersion anisotropy (diamond
  artifact)" below ~3 cells, `init_electron_dipole_ic`'s own docstring) --
  the whole reason `find_toroidal_ring(mode="octagon")` was built back on
  2026-09-08. Square-ring numbers from the first prototype should not be
  trusted as representative.
- Redid the corrugated-ring prototype on `mode="octagon"` (hops=4, 64
  tiles): closes cleanly at every m tested, `winding_number`=1.0000,
  circulation 107-108 (same qualitative shape as square version, larger
  because more tiles) -- confirms the circulation finding wasn't a
  small-ring artifact.
- **Built a real, production-scale candidate** (`build_toroidal_winding_
  octagon.py`): the EXISTING `init_toroidal_winding` seeder (already
  genuinely two-component: u=amplitude*cos(theta) AND phi=amplitude*
  cos(theta_mid), phase-coupled, real resonant omega, built to oscillate)
  on a proper octagon ring (hops=12, 192 tiles, ring_radius=29.58,
  R_DOMAIN=80, 2,784,735 leaves -- first time this seeder has been used at
  real production resolution instead of the toy square ring). Results:
  circulation -2.23e-14 (circulation-free to float precision, the cos(theta)
  zero-circulation property holds at real scale); persistence (300 steps,
  plain `step()`): E/E0 dips to 0.9939 at step 50, recovers to 0.99991 by
  step 300 -- real BOUNDED oscillation, no runaway, LESS shape-reshaping
  (0.25/0.32) than `dipole_mode_ic.py` (0.61/0.81). **Assessment: on the
  mass/two-component side specifically, this clears Philip's stated bar**
  (two coupled components, real discrete winding, circulation-free, holds
  together under plain dynamics at proper resolution) -- genuinely the
  best result of the session, though it's currently a thin ~192-tile
  filament (nothing seeded elsewhere in the 2.78M-leaf domain), not yet
  volume-filling, and the charge/exterior-field question is completely
  untouched for it.
- Philip: current construction is only 1 toroidal winding, 0 poloidal --
  "should be basically a solid block like winding copper coil." Discussed
  two approaches: (A) many parallel/stacked simple rings (tokamak-external-
  coil-like, easy, but NOT genuinely helical -- doesn't provide real
  rotational-transform confinement, just a denser one-directional current)
  vs (B) one continuous multi-turn helical path (stellarator-like, twist
  baked into the geometry itself, harder). Philip: "more like a stellarator
  than a tokamak" (wants B) -- then partially walked back: "individual
  rings could together count as a single component I suppose" -- (A) could
  still satisfy the Dirac two-component BOOKKEEPING requirement (paired
  with a separate second component) even without providing genuine
  drift-cancelling confinement -- flagged as a real but SEPARATE
  requirement from the tokamak/stellarator physics motivation, not yet
  reconciled.
- Attempted the real thing (`build_helical_torus_knot.py`): first design
  (minor loop in each leg's own local "outward" rotating frame) failed an
  algebraic pre-check (two legs' local-outward directions both worked out
  to -y -- a real reasoning bug, caught before touching the lattice, not a
  build failure). Fixed by using a FIXED global (x,z) plane for the minor
  loop instead (an honest simplification, not a literal (p,q) torus-knot
  parametrization). **Result: closes cleanly at p=1,q=1** (R_DOMAIN=50,
  680,211 leaves, `winding_number`=1.0000, 64 tiles), with GENUINE rho+z
  coupling confirmed this time (rho ranges 0 to 17.166, correlated with z
  sign -- unlike every earlier prototype's zero rho variation).
  `init_toroidal_winding` seeded cleanly on it unmodified (omega=0.098135,
  circulation=-0.0000, the cos(theta) zero-circulation property held on
  the non-planar path too). **But the shape is crude**: the meridional
  (rho,z) plot (`helical_torus_knot_p1q1.png`) shows two blocky rectangular
  "staples," not a smooth coil loop -- a direct consequence of the fixed-
  plane simplification. Session ended here, mid-thread, out of context
  before smoothing this or extending it.

### What's been effectively DISCARDED/superseded this session (don't re-try these without a real reason)

- Treating `init_electron_dipole_ic_vecpot` as a usable electron IC AT ALL
  -- it's circulation-equivalent to the spheromak, not a shape variant of
  it. All of 09-21's perturb-restoring/g-factor numbers built on it are
  dead.
- "Circulation-free" as a hard pass/fail criterion for IC viability --
  replaced by "does it erode under real dissipative dynamics," which is
  still untested for everything (see open item below).
- Plain-`step()` persistence as a discriminating stability test -- it
  can't discriminate anything (reversible dynamics never decays, full
  stop); useful only for confirming boundedness/no-blowup, not for ruling
  constructions in or out against each other.
- The flat "corrugated ring" (z-only detour) as a stand-in for genuine
  helicity -- it has zero rho variation, doesn't touch the actual
  confinement mechanism motivating this whole thread. Superseded by the
  p=1,q=1 rho+z-coupled attempt, itself still crude.
- The minimal 8-tile SQUARE ring as a base for any new ring-shaped
  construction -- always use `mode="octagon"` (or better) going forward;
  the square regime is a known-anisotropic artifact-generator.

### Queued next (supersedes `next`'s stale 2026-09-22 top section -- that file should be refreshed from this list before resuming)

1. **Smooth the p=1,q=1 helical path** -- replace the fixed-(x,z)-plane
   minor loop with a genuinely rotating local frame (octagon-style
   chamfering applied to the minor direction too, not just the major one),
   so the meridional projection is a real loop, not two rectangles. Small,
   well-scoped next step.
2. **Decide (A) vs (B)** from section 6 above -- ring-bundle-as-one-
   component (easy, satisfies Dirac bookkeeping only) vs. smoothed single
   helical path (harder, satisfies both bookkeeping AND real confinement
   physics if it works) -- or pursue both, bundle as a near-term fallback
   while (B) is refined.
3. **Extend toward "solid block" density** (higher q / multiple parallel
   windings) once single-path smoothness (item 1) or the bundle approach
   (item 2) is solid -- not attempted yet, explicitly deferred until the
   single-turn case is right.
4. **The dynamical-erosion test, still never run on anything**: does ANY
   existing closed winding (even the plain octagon `init_toroidal_winding`
   candidate from section 6, already circulation-free and stable under
   plain `step()`) actually hold up under real dissipative dynamics --
   `step_phi_diffuse` was flagged as a better-motivated candidate mechanism
   than `step_cross` for this (untried). This is the actual answer to "is
   this winding stable enough to be an electron," still open.
5. **Charge/exterior-field question, untouched for the winding-family
   candidates**: the octagon `init_toroidal_winding` build is a thin
   filament (192/2.78M leaves seeded) -- does a real exterior field
   develop dynamically as it oscillates (unmeasured, would need an
   end-of-run radial profile, not just t=0), or does it need to be paired
   with something volume-filling to have a charge signature at all?
   Separately, resolve the standing doc/dst.md contradiction from section 5
   (is "electron" a bound companion or a standalone winding) before
   assuming a free-standing winding should have a field of its own at all.
6. **Quantitative PHARE comparison** (carried over from 09-21's `next`,
   still not done): the raw `/home/p/EM_B.vtkhdf` AMR data was only
   inspected structurally, never actually used for a matched-slice/radial
   comparison against any candidate construction.

---


# Entries for 2026-09-22 moved from repo-root `next` on 2026-10-25 (verbatim, newest first)

## QUEUED 2026-09-22 — see `now`'s 2026-09-22 session entry FIRST (long, full detail) -- this is just the live action list distilled from it, replacing this section's own earlier (now stale) version written mid-session

Read `now`'s top entry before doing anything below -- it has the full
narrative (the dipole-IC-gallery audit, the circulation-equivalence finding
that invalidated all of 09-21's perturb-restoring/g-factor results, the
"circulation-free isn't the right bar" correction, the plain-step
persistence findings, the windings-as-fundamental-mechanism investigation,
and the tokamak/stellarator-motivated helical-winding construction work,
including what was explicitly discarded and why). Six items queued there,
reproduced here for quick reference:

1. **Smooth the p=1,q=1 helical torus-knot path** (`exp/2026/09/22/
   dipole_ic_gallery/build_helical_torus_knot.py`) -- it closes and has
   genuine rho+z coupling, but the minor loop is built in a fixed (x,z)
   plane, giving two blocky rectangles instead of a smooth coil. Needs a
   genuinely rotating local frame (octagon-style chamfering applied to the
   minor direction too). Small, well-scoped next step.
2. **Decide ring-bundle-as-one-component vs. the smoothed single helical
   path** (or pursue both) -- the bundle is easy and satisfies the Dirac
   two-component bookkeeping requirement only; the smoothed helical path is
   harder but would satisfy real rotational-transform confinement too, if
   it works.
3. **Extend toward "solid block" winding density** (higher q / multiple
   parallel windings, Philip's "like winding copper coil") -- explicitly
   deferred until item 1 or 2 is solid, not attempted yet.
4. **The dynamical-erosion test, still never run on anything** -- does ANY
   existing closed winding (even the already-built, already-stable-under-
   plain-`step()` octagon `init_toroidal_winding` candidate from `now`'s
   section 6) actually hold up under real dissipative dynamics.
   `step_phi_diffuse` flagged as a better-motivated candidate than
   `step_cross` for this test (`step_cross` is specifically translation/
   inertia, not general decay, and is being deliberately kept out of scope
   for now per Philip). Still the load-bearing open question -- everything
   else this session either couldn't test it (plain `step()`) or was
   explicitly told not to use the one mechanism that could (`step_cross`).
5. **Charge/exterior-field question, untouched for every winding-family
   candidate** -- the production octagon `init_toroidal_winding` build is a
   thin 192-tile filament in a 2.78M-leaf domain; does a real field develop
   dynamically as it oscillates (unmeasured, needs an end-of-run radial
   profile), or does it need pairing with a volume-filling structure to
   have a charge signature at all? Also unresolved: `doc/dst.md`'s own
   internal contradiction about whether "electron" means a bound
   deficit-moat companion or a standalone winding (`now`'s section 5) --
   this needs Philip's decision, not more code, before charge work on a
   free-standing winding makes sense.
6. **Quantitative PHARE comparison** (carried over from 09-21, still not
   done): `/home/p/EM_B.vtkhdf` only inspected structurally, never used for
   an actual matched-slice/radial comparison against any candidate.

---
