## 2026-09-22 -- at-a-glance gallery of every distinct dipole/electron IC in the repo

`plot_all_dipole_ics.py` builds a fresh `AdaptiveLattice` cluster
(R_DOMAIN=80, d1=1.0 -> 2,784,735 leaves, ~3.2s/build) for each of seven
distinct IC constructions, seeds it at t=0 (no time-stepping -- this is a
structural/topology review, not a physics run), and plots the meridional
(x-z, |y|<=1) energy-density slice (u^2 + 0.5*sum(phi^2)) for all seven side
by side. Every construction uses the same R_sph=1.0, scale=50.0 (DST-unit
boundary radius 50, matching `ELECTRON_SCALE` from 09-21's status.md, the
resolution already validated there as properly resolved) so the comparison
is apples-to-apples. Output: `dipole_ic_gallery.png`.

### What's in it, and what it shows

| Panel | Source | Verdict |
|---|---|---|
| `init_electron_dipole_ic` | `adaptive_flux_tube.py` (FLAGGED WRONG 09-21) | Small, compact, off-shape blob near center with a faint diffuse halo -- NOT the clean two-lobe hollow torus 09-21's PNG-vs-PNG comparison showed. See surprise note below. |
| `init_electron_dipole_ic_vecpot` | `adaptive_flux_tube.py` | Clean two-lobe pattern, lobes at x=+/-r_axis (~30.5), z=0 -- textbook closed/neutron-lineage double-ring, exactly as 09-21 described. |
| `init_spheromak_ic` | `adaptive_flux_tube.py` | Same two-lobe shape as the vecpot dipole, same amplitude order -- direct visual confirmation these two constructions are topologically the same family (poloidal Taylor state), independent of the "electron" vs "spheromak" naming. |
| `seed_dipole_ic` | `dipole_mode_ic.py` (CORRECTED production IC) | Single compact peak filling most of the domain -- matches PHARE's real dipole reference, as found 09-21. Confirmed here on a fresh build, not just carried over from that session. |
| `seed_curl_full_dipole_ic` | `curl_full_dipole_ic.py` | Adding the toroidal admixture (amplitude_pol=amplitude_tor=1.0) turns the single compact peak into a clear 4-lobe pattern (two bright + two dim lobes) -- the toroidal term visibly breaks compactness even at t=0, before any dynamics. Structural companion to 09-21's finding that this admixture moved the (Gamma/H) g-factor ratio further from target, not closer. |
| `seed_curl_full_pair` | `curl_full_spheromak_ic.py` | Two isolated few-pixel bright dots, rest of the domain empty -- exactly matches the module's own docstring ("explicitly not a spheromak... bare 1-cell-wide filament, u=0 everywhere"). Included for completeness, not a fair shape comparison to the volume-filling constructions. |
| bare current loop (A.dl, ported from `probe_curl_loop_dipole.py`) | new local helper in the gallery script | Two tight bright points where the wire crosses the slice plane (x=+/-30), diffuse long-range field below the wire-amplitude color scale. Open/proton-lineage by construction (Ampere's-law current loop, no hard R_sph cutoff) -- shape reads correctly for a current loop, though the color scale (set by the wire's own near-singular self-field) makes the far dipole field hard to see; a log-scale or separate near/far color range would show it better if this panel needs to carry more weight later. |

### A real surprise, not anticipated going in

`init_electron_dipole_ic`'s panel does NOT show the same two-lobe hollow
shape as its own `_vecpot` sibling, even though both share the same
B-field formula, k=4.493 eigenvalue, and R_sph/scale -- the only
difference is which quantity phi is seeded from (`init_electron_dipole_ic`:
B.n_hat, a flux sample; `_vecpot`: A.dl, a circulation sample; see both
functions' docstrings). The printed `divergence_clean` diagnostic shows why:
`init_electron_dipole_ic`'s raw B.n_hat seed starts at div_rms=7.08e-2
before cleaning, ~260x worse than `_vecpot`'s A.dl seed (div_rms=2.76e-4)
-- the Hodge projection has to do far more corrective work on the flux-based
seed, and that correction visibly warps the resulting energy density into a
small, oddly-shaped central concentration rather than preserving the
intended double-ring pattern. This is a DIFFERENT, additional defect from
the k=4.493/closed-topology issue 09-21 already found and is fully
consistent with what `init_electron_dipole_ic_vecpot`'s own docstring
already documented algebraically (B.n_hat is the wrong physical quantity
for phi, and the raw B formula has an exact B.r_hat=0 degeneracy) -- this
gallery is the first time that defect's actual visual consequence (after
divergence_clean, which is what every real caller actually uses) has been
seen directly, rather than inferred from the algebra or from a PNG
comparison that happened to use the already-corrected vecpot construction.
Practical implication: nothing changes for the priority-1 next-step (build
on `dipole_mode_ic.py`), but if `init_electron_dipole_ic` (the non-vecpot
one) is ever used as a baseline/comparison case again, its post-cleaning
shape should be treated as doubly unreliable, not just topologically wrong.

### Excluded constructions, with reasons

- `init_dipole_ic` / `init_phare_ic` (`adaptive_flux_tube.py`): both require
  a PHARE relaxed-spheromak npz (`ic_b.npz`) as an argument. Searched the
  whole filesystem (`find / -iname ic_b.npz`, plus `/home/p` and
  `/home/deegan` specifically) -- it does not exist anywhere in this
  environment, so these two could not be run. Only PHARE data found on disk
  is the unrelated `/home/p/EM_B.vtkhdf` (referenced in `next`'s item #2)
  and one unrelated farfield-profile npz under `exp/2026/09/08/`.
- `init_toroidal_winding` (`adaptive_flux_tube.py`): confirmed on reading it
  to be a lower-level primitive that seeds phi/u onto an ALREADY-FOUND ring
  (`ring_js`/`ring_faces` from `find_toroidal_ring`) -- not itself a
  dipole/electron IC, and structurally a component other constructions in
  this gallery either call internally or replace with a different
  (non-ring-based) seeding method.

### Practical notes for reuse

- Build is fast (~3.2s per cluster at this resolution, ~35s total for all
  seven) -- safe to rerun without `run_in_background`.
- `seed_curl_full_pair`'s ring search (`find_square_ring_on_leaves`, `hops=35`)
  succeeded on the first try at this domain/resolution; no tuning needed.
- The bare-current-loop helper (`seed_bare_loop_dipole` in the gallery
  script) is new, factored just enough to seed consistently with the other
  panels via the same A.dl/`_FACE_HOP` pattern `init_electron_dipole_ic_vecpot`
  already uses -- not promoted to `equilibrium/substrate/` since this was a
  review script, not a proposal for a new production IC; promote it there
  first if a future probe wants to reuse it directly.

---

## 2026-09-22, LATER -- per-construction multi-panel diagnostics find the
real linear-scale confound Philip caught, AND overturn the working
assumption that `dipole_mode_ic.py` fixed the open/closed topology question
(it only fixed the lobe SHAPE)

Philip looked at the first gallery and caught a real problem: the bare
current-loop panel (open/proton-lineage by construction) looked basically
identical to the spheromak/vecpot-dipole panels (closed/neutron-lineage) --
both just "two blobs, black everywhere else." Reasoned through together: a
current loop's energy density always concentrates where the wire crosses a
slice, regardless of open/closed topology, and on a LINEAR color scale set
by that near-wire peak, an exactly-zero exterior and a small-but-real
decaying exterior tail both render as indistinguishable black. "Blob count
on a linear scale" -- the test the first gallery used, and that 09-21's
PNG-vs-PNG comparison against PHARE references implicitly used too -- cannot
actually distinguish open from closed topology.

`plot_per_ic_diagnostics.py` (new script, reuses `plot_all_dipole_ics.py`'s
cluster-building/seeding code directly rather than duplicating it, same 7
constructions/exclusions) replaces the single linear-scale panel with FOUR
panels per construction, one figure per construction under `per_ic/`:
linear meridional slice (kept for continuity), LOG-scale meridional slice
(floor-clipped, so exact-zero and small-but-real both stay visible instead
of both reading as black), LOG-scale equatorial slice (second viewing
angle), and a radial mean-energy-density profile (semilogy vs r, vertical
line at r=R_sph*scale=50) -- the real, quantitative version of the test:
does the profile hit a hard floor at the boundary (closed/neutron-lineage)
or keep decaying past it with a real tail (open/proton-lineage)?

### Verdicts (`tail_frac_above_floor` = fraction of radial bins beyond
r=50 still above a small tail-floor threshold; see script for exact def)

| Construction | tail_frac | Verdict | vs. previous read |
|---|---|---|---|
| `init_electron_dipole_ic` | 1.000 | TAIL (open-lineage signature) | Likely NOT real physical openness -- see caveat below, don't read this as "actually open." |
| `init_electron_dipole_ic_vecpot` | 0.000 | HARD FLOOR (closed) | Confirms 09-21. |
| `init_spheromak_ic` | 0.000 | HARD FLOOR (closed) | Confirms 09-21/reference expectation. |
| `seed_dipole_ic` (`dipole_mode_ic.py`) | 0.000 | HARD FLOOR (closed) | **Overturns the working assumption -- see below.** |
| `seed_curl_full_dipole_ic` | 0.000 | HARD FLOOR (closed) | New info -- the poloidal+toroidal admixture doesn't open the topology either. |
| `seed_curl_full_pair` | 0.120 | HARD FLOOR (closed), tiny residual tail (max 3.9e-4 vs e_max=1.0) | Not a meaningful open signal -- this construction is a bare 1-cell filament, not volume-filling, per its own docstring; the small nonzero count beyond r=50 reads as edge-of-domain/discretization noise on a near-empty field, not a real tail. |
| bare current loop | 1.000 | TAIL (open-lineage signature) | Confirms the intended open behavior -- visually a smooth, broad, genuinely decaying field filling most of the domain with no cliff at r=50 (see `per_ic/bare_current_loop.png`), the clean positive control for this whole diagnostic. |

### The load-bearing correction: `seed_dipole_ic` (`dipole_mode_ic.py`) is
ALSO topologically closed -- exactly zero outside r=R_sph*scale, same hard
floor as the flagged-wrong double-ring dipole and the spheromak

Visually confirmed in `per_ic/seed_dipole_ic.png`: the log-scale meridional
slice shows a sharp, near-perfect circular cliff exactly at the cyan
r=50 boundary line, and the radial profile panel shows the mean energy
density fall from ~O(1) to the numerical floor (~1e-6, ~7 orders of
magnitude) within a single radial bin at r=50 -- not a gradual decay, a
wall. `tail_max_beyond=0.0` exactly (not just small).

This should have been predictable from `dipole_mode_ic.py`'s own docstring
("same K=4.493/R_SPH convention as before... U already vanishes at r=R_SPH
by construction"), and from 09-16's own finding about the ORIGINAL double-
ring dipole using this same eigenvalue ("At that k, the field is exactly
zero outside R_sph*scale by construction -- no external field is possible
AT ALL regardless of the phi-seeding fix") -- but 09-21's session treated
the shape fix (single compact peak vs. hollow two-lobe torus, matching
PHARE's real-dipole reference image) as if it had also resolved the deeper
open/closed topology question, and that conflation was never actually
checked until this gallery's radial-profile diagnostic. **It hadn't been
fixed. The lobe-shape problem and the exterior-field problem are two
separate defects, and only the first one has actually been addressed by
any construction currently in this repo (except the bare current loop,
which has no scalar u/background structure at all -- u=0 everywhere, a
pure current-loop diagnostic, not a volume-filling electron-like IC).**

Practical implication for `next`'s item #1 (rebuild confinement/charge/
g-factor tests on `dipole_mode_ic.py`): the shape fix is still real and
still worth keeping -- it correctly avoids the hollow double-lobe artifact
and matches PHARE's actual dipole image. But "dipole_mode_ic.py is open/
proton-lineage" should NOT be assumed going forward; by this gallery's own
measurement it is not. If doc/dst.md's "proton=open knot" requirement is
taken literally (an external field must exist), none of the tested volume-
filling (u,phi) constructions in this repo currently satisfy it. This
reopens 09-16's shelved k=4.493-eigenvalue question (relax k to allow a
real exterior tail vs. add an explicit exterior matching field vs. abandon
this eigenvalue-based construction family for the electron entirely) as
still fully unresolved -- not something the 09-21 shape fix closed, contrary
to how that session's conclusion was carried forward into `next`.

### Caveat on `init_electron_dipole_ic`'s TAIL result -- probably not real

`init_electron_dipole_ic` (the non-vecpot, already-known-doubly-defective
construction from the first gallery entry above) shows tail_frac=1.000 and
a non-negligible tail_max_beyond=4.5e-2 (~1% of its own e_max=4.13). This is
NOT being read as evidence of genuine open topology: this construction's
raw B.n_hat phi seed starts ~260x further from divergence-free than the
vecpot version (div_rms 7.08e-2 vs 2.76e-4, per the first gallery's finding)
and the resulting Hodge/`divergence_clean` projection is a global operation
-- it is entirely plausible the large correction it has to make redistributes
energy density outward as a numerical side effect, not because the
construction has any genuine physical exterior field. Not independently
verified this session (would need e.g. comparing the tail's radial falloff
shape against a real 1/r^n dipole law, or checking whether the tail
magnitude scales with how far the raw seed started from divergence-free
across several seeds) -- flagged as an open question, not resolved either
way, and should not be cited as "the original construction was actually
open after all."

### Files

- `plot_per_ic_diagnostics.py` (new script, alongside `plot_all_dipole_ics.py`)
- `per_ic/init_electron_dipole_ic.png`, `per_ic/init_electron_dipole_ic_vecpot.png`,
  `per_ic/init_spheromak_ic.png`, `per_ic/seed_dipole_ic.png`,
  `per_ic/seed_curl_full_dipole_ic.png`, `per_ic/seed_curl_full_pair.png`,
  `per_ic/bare_current_loop.png`

---

## 2026-09-22, EVEN LATER -- macro ring circulation measured directly:
`init_electron_dipole_ic_vecpot`'s "poloidal-only" claim is FALSE -- it
carries real, spheromak-scale circulation; the non-vecpot original and
`dipole_mode_ic.py` are both genuinely circulation-free

Philip's framing, in conversation: the spheromak is a closed knot that
genuinely decays, and that decay is driven by real circulation in its
toroidal component (`doc/mems/project_spheromak_decay.md`: `circ_B` unwinds
from -1.447 to ~0 over an actual decay run, while `circ_A`, the poloidal
piece, survives -- the established, measured mechanism of spheromak decay,
not a guess). An electron doesn't decay, so a genuine electron IC cannot
carry that same decay-enabling circulation. Everything checked in this
gallery so far (the hard-floor/tail radial-profile test) measured static
energy-density SHAPE, never whether an IC actually carries this circulation
-- that assumption was only ever taken from each function's own docstring
claim to be "poloidal-only, toroidal term dropped."

`measure_ring_circulation.py` (new script, reuses `plot_all_dipole_ics.py`'s
R_DOMAIN=80/R_sph=1.0/scale=50.0 convention) measures macro ring circulation
directly via `phi_ring_circulation` -- the tool `test_curl_full_dipole_ic.py`
already validated as the correct discriminator (gives EXACTLY 0.0 for a
genuinely poloidal field, real O(1)+ for a genuine toroidal one; the
per-triangle circulation/RMS ratio does NOT distinguish them and is the
wrong metric -- see that test file's own docstring). Ring: `find_toroidal_ring`,
`mode="square"`, `hops=27` (lands at ring_radius=35.79, close to this
project's established centreline radius r_axis~30.5=0.63*scale), centred on
the ring's OWN centroid per `phi_ring_circulation`'s calling convention (using
the global origin instead was already flagged once before as a way to get a
spurious nonzero reading, `project_spheromak_decay.md`'s 2026-07-03
correction -- avoided here by construction).

### Result

| Construction | circulation | Verdict |
|---|---|---|
| `init_electron_dipole_ic` (non-vecpot) | -1.166e-15 | CIRCULATION-FREE (float-precision zero) |
| `init_electron_dipole_ic_vecpot` | **+6.330e+02** | **REAL CIRCULATION -- spheromak-scale** |
| `init_spheromak_ic` (known-decaying reference) | +4.919e+02 | REAL CIRCULATION (positive control) |
| `seed_dipole_ic` (`dipole_mode_ic.py`) | 0.000e+00 | CIRCULATION-FREE (exact) |

### The load-bearing finding

`init_electron_dipole_ic_vecpot`'s own docstring flagged, but never checked,
a specific worry: its vector potential `A = cf(r)*(-y,x,0)` reuses the exact
same `(-y,x,0)` azimuthal Cartesian direction that `apply_toroidal_rotation`'s
real toroidal-FIELD formula uses elsewhere in this module -- i.e. it may be
applying a toroidal-field-shaped formula in a vector-POTENTIAL role. That
concern is confirmed real: this construction's circulation (632.96) is not
just nonzero, it is the SAME ORDER OF MAGNITUDE as the actual spheromak's
own toroidal circulation (491.86) on the identical ring. This is not merely
"visually looks like the spheromak" (09-21/09-22's earlier finding) -- by the
one diagnostic this project has already established as the real discriminator
for spheromak-like decay-driving content, `init_electron_dipole_ic_vecpot`
IS dynamically the same kind of object as the spheromak, despite being
labeled "poloidal-only, toroidal term dropped" in its own docstring. The
"vecpot fix" (built 2026-09-16 specifically to correct the B.n_hat/A.dl
convention mismatch in the original) introduced this real circulation as an
unintended side effect -- the fix traded one defect (wrong phi convention,
zero external field) for a different, arguably worse one (secretly
toroidal/decay-enabling, not actually poloidal-only as claimed).

**Meanwhile the non-vecpot `init_electron_dipole_ic`** -- dismissed earlier
today as "doubly defective" (small distorted blob from a poorly-conditioned
`divergence_clean` correction, per the first gallery's surprise note) -- is
genuinely circulation-free, to float precision, on this same ring. So is
`seed_dipole_ic` (`dipole_mode_ic.py`), exactly (0.0, not just small). Both
pass the test that `init_electron_dipole_ic_vecpot` fails. This directly
supports Philip's separate observation (this session, in conversation) that
the non-vecpot construction's actual plotted structure (central peak + real
ring/shell substructure + smoothly-decaying tail, see `per_ic/init_electron_dipole_ic.png`)
looks more physically plausible than it was given credit for -- it is not
just "less wrong" topologically, it also passes the circulation test the
vecpot sibling and the spheromak both fail.

### Practical implication

Of every construction measured across this whole gallery effort today, only
two are genuinely circulation-free (no spheromak-like decay-enabling content):
`seed_dipole_ic` (`dipole_mode_ic.py`) and the non-vecpot `init_electron_dipole_ic`
-- and both are ALSO the two constructions still flagged with real, separate
open problems (the k=4.493 hard-floor/closed-exterior issue for
`seed_dipole_ic`; the poorly-conditioned `divergence_clean` correction and
the still-unverified tail for `init_electron_dipole_ic`, see the earlier
caveat above). `init_electron_dipole_ic_vecpot` -- the construction actually
used as "the electron IC" in essentially every test since 2026-07-10 per
`now`'s 2026-09-21 entry -- should not be treated as a poloidal-only
reference for ANY future comparison; it is circulation-equivalent to the
spheromak it was meant to be distinguished from. Not yet done: re-derive
`init_electron_dipole_ic_vecpot`'s `A` formula properly (a genuine poloidal
vector potential, not a reused toroidal-field direction) if this construction
is ever needed again, rather than continuing to treat its vecpot fix as
settled; and independently verify whether the non-vecpot construction's
real, structured tail (flagged as "probably an artifact" earlier today,
never actually checked) is genuine, now that it has passed this circulation
test too.

### Files

- `measure_ring_circulation.py`

---

## 2026-09-22, STILL LATER -- plain-`step()`-only persistence test (NO
`step_cross`, ever): `init_electron_dipole_ic_vecpot` and the non-vecpot
original are BOTH exact static fixed points to numerical precision, INCLUDING
the circulation-carrying vecpot one; the spheromak's circulation is also
frozen without `step_cross`, confirming its decay needs that mechanism
specifically; `init_toroidal_winding`'s own `cos(theta)` seeding is
circulation-free by construction, same as `dipole_mode_ic.py`

Philip's correction, in conversation: `step_cross` is specifically the
translation/inertia mechanism (`doc/dst.md`'s "Through-cell crossing"
section -- it exists so a winding can physically translate through the
lattice, nothing more general), and it's explicitly provisional/unconfirmed
ontology, not a settled general-purpose decay test ("we're not sure it's
ontologically correct, we're trying to be progressive"). The general
"does this structure persist" question should use the base reversible wave
equation alone -- matching the convention
`probe_pol_tor_lockstep_convergence.py` already used ("`evolver.step(ts)` --
pure wave dynamics, NO `step_cross`").

`test_plain_step_persistence.py` (new): 5 constructions, same
R_DOMAIN=80/R_sph=1.0/scale=50.0 convention, 300 steps of plain
`evolver.step(ts)` -- `step_cross` never called anywhere in the script.
Tracked every 50 steps: total field energy `E(t)=sum(u^2)+0.5*sum(phi^2)`,
macro `phi_ring_circulation` on a fixed ring, and radial energy-density
profile shape (t=0 vs. mid-run vs. end-of-run, `shape_drift` = max relative
deviation across bins where the t=0 profile is >1% of its own peak).

### Result

| Construction | E/E0 final | circ0 | circ final | shape_drift (mid / end) |
|---|---|---|---|---|
| `init_electron_dipole_ic_vecpot` | 1.00000000 | +632.96 | +632.96 (bit-identical) | 0.0000 / 0.0000 |
| `init_electron_dipole_ic` (non-vecpot) | 1.00000000 | -1.17e-15 | -1.87e-15 | 0.0000 / 0.0000 |
| `seed_dipole_ic` (`dipole_mode_ic.py`) | 0.99986 (bounded, oscillates 0.9998-1.0) | 0.0 | ~1.7e-16 | 0.61 / 0.81 |
| plain 8-tile ring (`init_toroidal_winding`) | 1.00061 (dips to 0.9933 then settles) | -1.22e-15 | -1.22e-15 | 0.28 / 0.42 |
| `init_spheromak_ic` | 1.00004 | +491.86 | +491.86 (drifts <0.001%) | 2.39 / **11.10** |

### The headline result: `init_electron_dipole_ic_vecpot` really is an
EXACT static fixed point, circulation and all -- decay is NOT automatic
just from carrying spheromak-scale circulation

Energy, circulation, AND radial shape are all frozen to numerical precision
over 300 steps -- this directly confirms (for the first time quantitatively,
not just carried forward as an unchecked claim from prior sessions) that
`init_electron_dipole_ic_vecpot` is a genuine static fixed point of the
plain reversible dynamics. Carrying real, spheromak-order circulation
(+632.96) does NOT by itself cause decay under plain `step()` -- exactly
consistent with `project_spheromak_decay.md`'s own methodology, where the
spheromak's measured decay (`circ_B`: -1.447 -> ~0) was produced BY
deliberately applying `step_cross`/`cross_scale`, never shown to happen
under plain wave dynamics alone. **Confirmed directly here**:
`init_spheromak_ic`'s own circulation (+491.86) is ALSO essentially frozen
under plain `step()` (drifts <0.001% over 300 steps) -- the spheromak does
not spontaneously decay either, without `step_cross` engaged. This is the
expected result, not a surprise, but it had never actually been checked
before now -- it was inferred from the decay experiment's own methodology,
not independently confirmed on a matched-resolution IC.

**Practical consequence**: the "must not decay" criterion cannot be settled
by any test that never engages `step_cross` -- both the circulation-heavy
vecpot dipole AND the circulation-free non-vecpot dipole look IDENTICALLY
stable under plain dynamics (both exactly frozen, shape_drift=0.0000 for
both). The two constructions are indistinguishable on this test. If
`step_cross` itself is being set aside as ontologically unconfirmed (per
Philip's framing this session), then NEITHER a static-circulation reading
NOR a plain-`step()` persistence check settles "will this decay" -- both
tests currently available in this project return the same (uninformative,
for this purpose) answer for every construction that has no explicit
translation/crossing dynamics applied to it at all.

### A genuine, unanticipated finding: `init_toroidal_winding`'s own
`cos(theta)` seeding is circulation-free by construction -- same family as
`dipole_mode_ic.py`, not a special property of the spherical eigenmode ICs

The plain 8-tile ring, seeded via the existing, doc-literal
`init_toroidal_winding` (its own built-in `phi_val = amplitude*cos(theta_mid)`
pattern around the ring, NOT a naive constant/uniform current), reads
circulation = -1.22e-15 -- float-precision zero, same class as
`dipole_mode_ic.py`'s exact 0.0. This is a real algebraic fact, not a
coincidence: a `cos(theta)` circulation pattern integrates to exactly zero
around a closed loop, the same "standing wave, zero net circulation by
construction" property `project_spheromak_decay.md`'s own pre-2026-07-03
resonant-loop IC already had (`sum_j cos(K(j+0.5))` over one period = 0,
verified there analytically). **This means the earlier torus-knot-scoping
prototype's real +12.9-13.6 circulation reading (this same file, LATER
section above) came specifically from seeding a naive CONSTANT-amplitude
current, not from the discrete winding construction itself being
inherently circulation-bearing.** `init_toroidal_winding`'s own default
convention is already circulation-free. Practical implication: a future
torus-knot/helical construction should keep this same `cos(theta)`-style
(or equivalent phase-varying, not constant-amplitude) seeding convention if
avoiding net circulation remains a design goal -- the earlier prototype's
nonzero reading was a seeding-choice artifact, not evidence that closed
discrete windings are inherently circulation-bearing.

### `seed_dipole_ic` and `init_spheromak_ic` both show real shape
reorganization under plain dynamics, for different likely reasons

`dipole_mode_ic.py`'s shape_drift (0.61 mid, 0.81 end) is expected and
benign: it's a genuine standing wave by construction
(`u=a*U*cos(wt)`, `phi=b*gradU*sin(wt)`), so its energy density
legitimately sloshes in shape over a cycle even though total energy stays
bounded (0.9998-1.0, no monotonic drift) -- this is oscillation, not decay.

`init_spheromak_ic`'s shape_drift is much larger (2.39 mid, **11.10** end --
some radial bin ended up over 1000% different from its t=0 value) DESPITE
both its total energy and its macro circulation staying essentially frozen
over the same run. This is a genuine, unanticipated finding: real internal
spatial reorganization can happen even when the two previously-tracked
scalar diagnostics (H, circulation) show no problem at all -- a construction
can look perfectly stable by both of today's established scalar tests and
still be reshaping substantially in real space. Not yet explained (possibly
a genuine internal breathing/relaxation mode of the full poloidal+toroidal
Taylor state under the plain wave equation, distinct from the vecpot/
non-vecpot dipoles which are both exactly static in every diagnostic
tracked) -- flagged as a real open question, not resolved here.

### How to apply

Don't treat "frozen circulation + frozen energy under plain step()" as
sufficient evidence of genuine stability -- check the radial shape too
(`init_spheromak_ic` passed both scalar tests while its shape drifted by
>1000% in places). Don't treat plain-`step()`-only persistence as the
decisive "will this decay" test either -- per the practical-consequence
note above, it currently can't distinguish the circulation-heavy vecpot
dipole from the circulation-free ones, since none of them are exposed to
`step_cross`/any dissipative mechanism in this test. Before trusting any
future torus-knot/helical winding construction's circulation reading as
"safely small," check what seeding convention was used
(`cos(theta)`-style vs. constant-amplitude) -- they are not equivalent, and
today's default convention (`init_toroidal_winding`) already gives exactly
zero.

### Files

- `test_plain_step_persistence.py`, `plain_step_persistence.png`,
  `run_plain_step_persistence.log`

---

## 2026-09-22, YET LATER -- first full production-resolution characterization
of `init_toroidal_winding` on a real octagon ring: circulation-free (as
expected), closed at t=0 (expected -- nothing was ever seeded outside the
ring), holds together better than `dipole_mode_ic.py` under plain dynamics

Philip: "can we use this to build something real?" -- `init_toroidal_winding`
already combines the two things assembled piecemeal today (a genuine
discrete winding, satisfying `doc/dst.md`'s own mass mechanism; and a real
two-component `u`/`phi` pair with a defined resonant frequency, not a bare
static current), and its default `cos(theta)` seeding was already shown
circulation-free on the small square ring. The one thing never done: run it
on a properly-resolved ring (this session already fixed the small-square-
ring anisotropy problem earlier today) at this gallery's actual production
resolution, with the full diagnostic treatment every other candidate got.

`build_toroidal_winding_octagon.py`: `find_toroidal_ring(mode="octagon",
hops=12)` (calibrated empirically -- gives ring_radius=29.58, centred at the
origin to float precision, matching this project's established electron
centreline-radius convention ~30-31.5) on the standard R_DOMAIN=80 cluster,
`init_toroidal_winding(amplitude=1.0)`, then the same 4-panel static
diagnostic, `phi_ring_circulation`, and 300-step plain-`step()`-only
persistence run used throughout today.

### Results

- **n_tiles=192, resonant omega=0.032723** (matches `2*sin(pi/192)` from the
  function's own documented formula).
- **circ0 = -2.23e-14** -- circulation-free to float precision, confirming
  today's earlier square-ring finding generalizes to a properly-resolved
  192-tile ring, not just the small one. Stays pinned near zero the entire
  300-step run (max |circ| ~5.4e-14) -- no drift toward nonzero.
- **Static (t=0) radial profile: HARD FLOOR at the ring radius, tail_frac=0.000.**
  Read this correctly, not as equivalent to the continuum family's problem:
  the Taylor-state constructions are forced to exactly zero everywhere
  outside R_sph*scale by an eigenvalue, at every time, by construction. This
  construction is zero outside the ring's radius at t=0 for a much more
  mundane reason -- `init_toroidal_winding` only ever writes to the ~192
  tiles on the ring path itself; nothing was ever seeded anywhere else in
  the 2.78M-leaf domain. Whether a genuine exterior field develops
  DYNAMICALLY as the seeded oscillation radiates outward via ordinary
  face-hop wave propagation is an open, different, and more interesting
  question than the continuum family's permanent structural defect -- not
  checked here (would need the radial profile at end-of-run, not just t=0),
  flagged as the natural next check on this construction specifically.
- **Persistence (plain `step()`, no `step_cross`, 300 steps): E/E0 dips to
  0.9939 by step 50, recovers to 0.99991 by step 300 -- bounded, no
  monotonic drift, broadly consistent with the smaller/anisotropic 8-tile
  version's earlier dip-and-recover (0.9933 -> 1.0006) but not identical in
  detail (settles just under 1.0 here vs. just over there) -- real dynamics,
  not a fixed point, but not runaway either. shape_drift: 0.25 (mid) -> 0.32
  (end) -- real reshaping, but LESS than `dipole_mode_ic.py`'s own
  standing-wave shape_drift (0.61/0.81) despite this construction touching
  a tiny fraction of the domain's cells, and far less than the spheromak's
  11.10.

### Assessment

Against everything established today, this is the most complete candidate
built so far on the "mass" side of the puzzle: a genuine discrete winding
(not a continuum eigenmode), two real coupled components with a defined
oscillation frequency (not a bare current or a single field doing all the
work), circulation-free at production resolution (not just the small,
now-known-anisotropic ring), and it holds together under plain dynamics
better than `dipole_mode_ic.py` does, not just as well. It does not resolve
the charge/exterior-field question -- nothing here suggests it does or
doesn't have a real long-range field once evolved, that's simply
unmeasured yet. Per Philip's own framing this session ("we're not expecting
full physical realism immediately... two components, both travelling at c,
it just sits there and doesn't decay, that's a good start"), this
construction now clears that bar on real, measured grounds, not by
assumption -- promoted from "excluded lower-level primitive" (this
gallery's first entry, 2026-09-22 EARLIER) to genuine candidate status.

### Files

- `build_toroidal_winding_octagon.py`, `run_toroidal_winding_octagon.log`,
  `per_ic/toroidal_winding_octagon.png`, `toroidal_winding_octagon_persistence.png`
