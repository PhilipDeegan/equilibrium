# Boundary conditions — status and findings

## 2026-09-29 review (read first; supersedes older sections where they conflict)

Reviews: `exp/2026/09/29/{bc_review,sponge_assess,bc_design,charge_exterior}/status.md`.
Invariants: `tst/substrate/test_bc_invariants.py` (face antisymmetry, dt consistency, target, E).

- Every sponge damped phi per cell row, breaking phi_ki = -phi_ik; the mismatch acted as an
  invisible flux sink at the sponge edge (it held the old point-source "Coulomb" profiles).
  Fixed: `quadratic_sponge(gamma_max=... | damp_max=legacy)` + `step_sponge(i, dt)` (both
  models; one factor per face for both copies; E included in the continuous model; u toward
  target). Integrated / directional / PML: face-symmetric decay and projection. numpy
  `DampedLattice`: rate, per face, default gamma_max = 10 c / width.
- Design rule (bc_design): gamma_max ~ 10 c / width; reflection 5.6e-3 / 2.8e-3 / 8.3e-4 at width
  2.4 / 4.8 / 9.6 cells (old sponge 2.4e-2 / 3.2e-2 / 7.2e-3).
- Statics: step() has no static u gradient (phi_t = -W grad u grows); a charge's static exterior
  is radial phi with uniform u (charge_exterior). A phi-damping sponge cannot hold a static
  potential.
- fn / remote boundary faces now reach u on unrefined clusters (`_bc_face_to_u`): dirichlet is a
  fixed-u neighbour (energy from the target conserved, TestDirichletBC); absorbing stays inert
  (ghost = u_i). Face ghosts stay excluded. `step_subcycle` on refined clusters throws for face
  BCs instead of skipping them silently.
- Removed: `outflow_ledger` (bit-identical to outflow), `pml_outflow` (identical to outflow).
  `outflow` default c_lattice = `bcc::wave_speed()` (2.309), was sqrt10.
- Open: BC arrays not remapped after AMR, not persisted.

Tracks what's actually validated about each boundary-condition type's real
behavior (reflective vs. genuinely absorbing, what's been measured, what
hasn't) — consolidating findings that would otherwise stay scattered across
dated `now`/`exp/` session logs. Not a failure log (see `doc/dst/failures/`)
and not a resolution-requirement log (see `doc/dst/resolution_requirements.txt`)
— this is BC-specific status, organized by BC type.

## Architecture — what actually reaches `u`

`_step_pass1` (`inc/eq/core/data/lattice/adaptive_lattice.hpp`) only sums a
face's `phi[k]` into `u`'s update where `nb[k] >= 0`. A genuine domain-edge
face always has `nb[k] < 0`, so **any** `bc.fn`-driven boundary value
(absorbing/dirichlet/sponge-via-fn) or `remote_flux` is structurally
invisible to `u` — `BCType`/`fn_type` was written to let pass1 dispatch on
it but is never read anywhere in the stepping code (dispatch documented,
never implemented). This was tried once before (naively including
ghost-wired phi in pass1's accumulator) and caused **exponential
instability** — the `nb[k]>=0` guard exists specifically to prevent that.
Full history: `doc/mems/project_bc_fn_inert_on_u.md`, `doc/mems/project_bc_enum.md`.

Only two mechanisms actually move `u`:

1. **Volumetric annulus family** (`split`/`integrated`/`directional`/`pml`) —
   a radius-based decay applied directly to real interior cells' own
   u/phi over an annular region (`_step_sponge_pass`, `_step_pass1_damped`,
   `_step_pass1_pml`). Needs `sponge_fraction*r_max` of actual domain width
   to work with (this is what `sponge_fraction`/`r_inner` tune).
2. **`outflow`** — the one BC that is actually wired into `_step_pass1`
   itself (gated per-(leaf,face) by `outflow_active`, everything else stays
   an exact no-op). Zero annulus needed by construction (delta=0).

## `absorbing()` / `dirichlet()` / `remote_flux` — reflective, not absorbing

`ghost u = u_i` (`absorbing`) is a Neumann zero-flux wall, not an absorber —
confirmed both analytically (see architecture note above) and by test
(`TestFnBasedBCInertOnU::test_dirichlet_target_never_moves_u`: `u` is
bit-identical between `dirichlet(target=99)` and no-BC runs of the same
pulse). Useful as a **known-100%-reflective control** when calibrating other
diagnostics (see decay-to-quiescence section below).

## Volumetric annulus (`split`/`integrated`/`directional`/`pml`)

r_inner-fixed width sweep (isolating layer width from the
near-field-clearance confound of naively widening `sponge_fraction` at fixed
R): **both split and PML genuinely improve with width** — split
8.67e-8→5.97e-8→4.56e-8 (sf=0.15→0.3→0.4), PML 1.29e-7→6.22e-8→4.32e-8 at
matched r_inner=55.25. PML benefits more than split at matched width,
consistent with the Nataf ABC/PML review's (`hal-00799759`,
`doc/mems/reference_nataf_abc_pml_paper.md`) reflection formula
`R = (e^{-2σδ})^{cosθ}` — reflection depends on the σ·δ product, and PML's
coordinate-stretching construction uses added width more effectively than
plain amplitude damping. PML also trades a lower peak amplitude for a
spatially WIDER disturbed background (confirmed via shared-colorscale
XZ |S| renders at t=65/70) — a real mechanism difference, not an artifact.

Full history/numbers: `doc/mems/project_poynting_perf_r150_farfield.md`,
`doc/mems/project_poynting_dipole_calibration.md`.

## `outflow` — first-order Mur/Sommerfeld, genuinely wired into u

**Status summary (2026-09-16)**: outflow is a correctly-implemented, cheap,
genuinely-draining BC. It is a near-perfect absorber for isotropic/monopole
sources (~1e-9 of IC floor, zero re-growth). For ANY source with real
angular/multipole structure — toroidal ring AND a minimal two-point dipole
alike, ruling out ring-construction as the cause — it settles at a small but
non-zero, non-growing reflection floor (dipole: ~1.9e-4 of IC; ring: 5-9x
worse than split/PML's own post-round-trip shell-|S|). This floor is set by
angular incidence, not by interface thickness or amplitude: BOTH tried
radially-symmetric fixes — the `outflow_tapered()` taper band (thin,
radial-only) and the full `pml_outflow()` volumetric annulus (idea 4,
below) — speed up or shape the transient but converge to the same floor
either way. **Practical implication, unchanged from 2026-09-14**: for the
actual Poynting-ring production work, the r_inner-fixed split/PML
volumetric sponge remains the right tool, not outflow — outflow's angular
floor isn't currently fixable by anything tried so far. **Next ideas, not
yet started**:
1. A genuinely ANGLE-dependent mechanism — e.g. `outflow_c[j]` or a taper
   weight keyed to local incidence angle relative to the source's own
   radiation pattern, rather than radius alone. No design yet.
2. Analytically check the measured dipole floor (~1.9e-4) against the known
   angle-dependent first-order-Mur reflection-coefficient formula from the
   classical FDTD ABC literature — would confirm the magnitude is exactly
   what plain Mur theory predicts, not just the right qualitative shape.
3. Old idea 4, still not built: a permanent flat-box reflection-coefficient
   regression test (`ghost=u_i`, `ghost=-u_i`, `outflow`) in `tst/`, per
   [[feedback_save_validations_as_tests]] — would also be the natural place
   to assert the monopole-vs-dipole floor distinction found this session so
   it can't silently regress.
4. ~~Combining outflow with the split/PML volumetric annulus~~ — TRIED
   2026-09-16 (`BoundaryCondition::pml_outflow()`), RULED OUT for the ring's
   angular floor: after fixing a genuine exponential-blowup bug the naive
   combination first triggered (see "PML+outflow combo" section below), the
   fixed result is statistically identical to plain outflow at every shell
   (r=20: 1.75e-7 vs 1.84e-7). A volumetric annulus is exactly as
   radially-symmetric as the already-ruled-out taper band, so it hits the
   same wall for the same reason.
5. The ~430%/800% thread-utilization anomaly in the full sweep harness
   (2026-09-13/14) is still open and unexplained, lower priority than the
   above since it doesn't affect correctness.

New BC (2026-09-13): `u_ghost^{n+1} = u_i^n + r·(outflow_ghost - u_i^{n+1})`,
`r = (L_i - c_i·dt)/(L_i + c_i·dt)` (proper Mur 1981 derivation — the first
attempt had the sign flipped and injected energy instead of draining it,
caught by `TestPulseOutflowBC::test_u_vol_genuinely_drains`).

**Drainage**: confirmed genuine on a flat box (all 14 boundary faces
outflow, no sponge) — u_vol drops ~40%, peak u drops ~90%, vs. <0.1% for
periodic/absorbing.

**Performance**: the originally-reported "~2x slower than split/PML" does
**NOT** reproduce as a step()-cost effect when profiled properly (phlop
scope timer, serial, matched N=1.49M) — split 144.4ms/step vs outflow
148.4ms/step, ~3% apart. Every dominant sub-scope matches within 1-2%; split
additionally pays a whole `_step_sponge_pass` (~14.5ms) outflow skips
entirely. The ~430%/800% thread-utilization anomaly on 8 threads DOES still
reproduce, but isn't explained by per-step compute cost — likely something
in the full-sweep harness (checkpoint I/O, frame plotting) rather than
`outflow`'s own stepping code. Open.

**Reflection quality — geometry-dependent, this is the key finding**:

| Source geometry | Result | Verdict |
|---|---|---|
| Toroidal ring, R65/R150, spherical domain | post-round-trip shell-|S| is **5-9x WORSE** than split/PML's own post-round-trip numbers | outflow loses |
| Isotropic Gaussian bump at sphere centre, R=10 | global `u`-energy decays ~2 orders of magnitude per round trip, flat floor (~1e-9 of IC) by round trip 7-8, **zero re-growth** through 15 round trips | outflow wins, decisively |

The isotropic-source case is the best possible incidence angle for a
first-order Mur condition (wavefront curvature == boundary curvature
everywhere) — and it performs almost perfectly (first-pass transmission
alone is already ~99%). This strongly suggests the ring-source result isn't
about first-order-Mur being fundamentally inadequate, nor about BCC lattice
dispersion anisotropy (`project_bcc_dispersion_anisotropy`) — it's the
ring's genuine off-normal/angular radiation pattern hitting the boundary at
non-uniform incidence. **Practical implication**: 2nd-order
(Bayliss-Turkel) Mur — a curvature correction for an assumed isotropic
radial wave — probably would NOT fix the ring case, since the gap there is
angular/directional, not radial-curvature mismatch. Not yet tried; lower
priority than previously thought given this result.

Reproducer: `exp/2026/09/15/19_16_outflow_isotropic_decay_probe/probe_outflow_decay.py`
(`--ic bump` for the isotropic case, `--ic ring --hops N` for the toroidal
case).

### Taper band tried and ruled out for the ring case (2026-09-16)

Built `BoundaryCondition.outflow_tapered(nb_all, pos_all, taper_width, c_lattice)`
(`bc.hpp`'s `outflow_taper_weight`/`outflow_taper_ghost`, wired as an
ADDITIVE relaxation sink in `_step_pass1`, real interior leaves only within
`taper_width` of the true edge, true boundary leaves excluded to avoid
double-counting the existing exact ghost there) — the former queued "idea 1"
(give outflow some absorbing thickness). Two results:

- **Isotropic bump (R=10)**: genuinely improves an already-near-perfect
  case further — 3-4x lower leftover energy at every round trip vs plain
  outflow (round_trip 5: 1.03e-7 vs 4.13e-7 fraction of IC). Confirms the
  mechanism itself works as designed.
- **Toroidal ring (R=65/octagon/hops=8, matched to the exact config that
  produced the "5-9x worse" verdict)**: statistically indistinguishable
  from plain outflow — late-time shell-|S| means match to 3 significant
  figures at r=8/20/24 (e.g. r=20: 7.7179e-8 vs 7.7194e-8). **Confirms the
  angular-incidence hypothesis, not the zero-thickness-interface one**: the
  taper is a purely radial modification (weight depends only on `r`), so it
  cannot fix a reflection deficit whose root cause is the ring's angular/
  off-normal radiation pattern hitting the boundary non-uniformly. Idea 1 is
  now RULED OUT for the ring case (not just deprioritized) alongside idea 2
  (2nd-order Mur) for the same underlying reason — both are radially-
  symmetric fixes for an angular problem.

Reproducer: `exp/2026/09/16/12_30_outflow_taper_probe/` (`probe_outflow_taper.py`
for the small-R decay methodology + `outflow_tapered` binding sanity check;
`probe_sponge_quality_v3.py --variants outflow,outflow_taper` for the R65
apples-to-apples shell-|S| comparison, log in
`sponge_v3_R65_taper.log`). **Next actionable idea for the ring case**: idea
3 from the original queued list (isotropic-source discriminator) is already
answered by this doc's isotropic-vs-ring table above; what's left is
something that actually varies with ANGLE, not radius — e.g. a
direction-dependent `outflow_c[j]` or taper weight keyed to local incidence
angle relative to the ring's own radiation pattern, not yet designed.

### PML+outflow combo tried and ruled out for the ring case (2026-09-16)

Implemented idea 4: wired `outflow_active` into `_step_pass1_damped` and
`_step_pass1_pml` (previously only plain `_step_pass1` consumed it) and
added `BoundaryCondition::pml_outflow(nb_all, pos_all, gamma_max, r_inner,
c_lattice)` — a genuine PML volumetric annulus PLUS an outflow one-way Mur
ghost at the literal domain edge, so the sponge has somewhere non-reflective
to send whatever it doesn't absorb, instead of the fully-reflective
absorbing()/dirichlet() wall every other sponge variant sits in front of.

**First attempt blew up exponentially**, starting almost exactly at the
sponge annulus's own predicted round-trip time (R65/octagon/hops=8/sf=0.15/
gamma=0.5: clean through t~28, then ~2 orders of magnitude growth per ~2.5
time units, landing 3-4 orders of magnitude worse than either component
alone by t=41). **Root cause, confirmed by reading the code**:
`_step_pass2_bc`'s outflow-ghost recovery (`adaptive_lattice.hpp`, the
outflow branch's `u_i_old = u_i - cl.temp_du[j]`) assumes `temp_du[j]` IS
the leaf's entire u-update for the step — true for plain `_step_pass1`
(`cell.u += du`), but NOT for `_step_pass1_pml`, which does `cell.u =
cell.u + du - decay_loss` (an extra PML decay subtracted after `du`). The
true boundary leaves where `outflow_active` lives are exactly the leaves
where PML's `gamma`/`decay_loss` is LARGEST (r=r_max), so the recovered
"previous u" was wrong by `decay_loss` every step — a persistent,
compounding error feeding the Mur one-way extrapolation. A different
instance of the same bug CATEGORY as [[project_bc_enum]]'s old ghost-wiring
instability (a mismatched/corrupted u-history feeding a one-way
extrapolation), not a re-trigger of that specific bug.

**Fix**: `pml_outflow()` deliberately keeps `gamma[j]=0` on every true
boundary (`outflow_active`) leaf, ramping gamma up to `gamma_max` only
across the real interior annulus cells just inside it. Since
`_step_pass1_pml` already reduces exactly to plain `_step_pass1` wherever
`gamma==0`, this restores the exact `temp_du`-is-the-whole-update invariant
outflow's ghost recovery needs — the two mechanisms are spatially
separated (sponge damps the interior annulus, outflow handles the exact
edge) rather than co-located. Confirmed clean by direct probe (small-sphere
pulse decay, no re-growth, zero NaN) and by the full C++ (`test_lattice`)
and Python (`test_boundary_conditions.py`, 17/17) suites staying green.

**Re-ran the R65/octagon/hops=8 comparison with the fix**, all extended to
t=41.03 (past every variant's own true round-trip: outflow/pml_outflow's
edge round trip is 28.59, plain PML's annulus-only round trip is 22.42):

| variant | r=8 | r=20 | r=24 |
|---|---|---|---|
| pml_g0.5 (annulus only) | 2.41e-8 | 3.48e-7 | 7.86e-8 |
| outflow_c3.16228 (edge only) | 8.18e-9 | 1.84e-7 | 4.30e-8 |
| pml_outflow_g0.5 (both, fixed) | 8.15e-9 | 1.75e-7 | 4.29e-8 |

**Verdict**: pml_outflow is statistically indistinguishable from plain
outflow at every shell (well within each other's std) — adding the PML
annulus gave no measurable improvement. Same underlying reason as the
taper band: a volumetric annulus is exactly as radially-symmetric a
mechanism as a radial taper, and the ring's residual floor is angular, not
radial-amplitude. **Idea 4 is now RULED OUT for the ring case** (though the
bug fix is a genuine, permanent correctness improvement — `pml_outflow` is
a correct, usable BC, just not a fix for this specific angular problem).

Reproducer: `exp/2026/09/16/14_25_pml_outflow_combo/` (`probe_sponge_quality_v3.py --variants pml_outflow`,
logs `sponge_v3_R65_pml.log`/`_outflow.log`/`_pml_outflow_fixed.log`).

### Ledger-capped (passive) ghost tried: no effect, but reframes the whole problem (2026-09-16)

Idea: cap the outflow ghost's occasional sign-flipped ("would-be inject")
contributions with a per-face running energy ledger (`outflow_credit`,
`bc.hpp`) — a face can never inject more than it has ever genuinely
drained. A passivity guarantee, more permissive than a hard diode (still
lets short-cycle reactive exchange through). New factory
`BoundaryCondition.outflow_ledger()`, wired into `_step_pass1`'s existing
outflow branch. Distinct from [[project_dst_energy_recycling_boundary_idea]]
(a much bigger non-local substrate-physics hypothesis) — this is a purely
local numerical constraint on one face.

**Zero measurable effect**, both cases: `outflow_ledger` is bit-identical
to plain `outflow` on the R=10 dipole probe (every round trip through 15)
and on the R65 ring probe (r=8/20/24 to 5 sig figs). Confirmed by direct
offline replication of the exact per-face flux math against the real
trajectory: sign-flip events are real and frequent (~10.7% of active-face
checks on the dipole probe, at every one of 1538 boundary leaves), but
cumulative injected magnitude stays ~200x smaller than cumulative drained
magnitude throughout the run — the ledger never comes close to exhausted,
so the cap never binds. Not an implementation bug; the constraint is real,
just not the tight one here.

**What the same instrumentation found instead, and it's the more important
result**: cumulative drain AND inject both completely flatten by round trip
~4 on the dipole probe — the boundary stops exchanging energy in EITHER
direction — while the domain's own total energy keeps falling for many
more round trips after that before settling into the ~1.9e-4 floor. The
boundary goes inert *before* the floor is reached, which is the wrong
timing for a live reflect/return cycle sustaining it. This points at a
genuinely bound, slowly-decaying residual mode in the outer region that the
outflow ghost has simply stopped coupling to, rather than ongoing
reflection actively feeding the floor. **This reframes the open question**:
not "is the Mur angle/formula wrong" (which idea 1's taper, the angle-
correction design, and this ledger all implicitly assumed) but "why does
this residual mode stop coupling to the boundary at all" — not yet
investigated (would need to visualize/characterize the floor-plateau state
directly: spatial structure, source-proximity, frequency content).

Reproducer: `exp/2026/09/16/14_25_pml_outflow_combo/` (`outflow_ledger`
variant in both `probe_outflow_taper.py` and `probe_sponge_quality_v3.py`).

### Ruled out: ring-construction artifact (2026-09-16)

Before trusting the angular-incidence conclusion above, checked whether it's
actually an artifact of the ring/winding IC itself (topological `lam`
binding, the octagon mesh construction, `init_toroidal_winding`'s cos(theta)
seeding) rather than genuine multipole-radiation physics. Built a
completely independent, minimal dipole source with none of that machinery:
two ordinary opposite-sign Gaussian `u` bumps (`--ic dipole` in
`probe_outflow_taper.py`, zero net monopole moment by construction, no
ring/winding/lam/mesh code path at all) at the same small-R decay-to-
quiescence scale as the isotropic-bump control.

**Result**: outflow decays fast through the first few round trips then
settles at a small but non-decaying FLOOR (~1.9e-4 of IC, vs the isotropic
bump's clean ~1e-9 floor) — a genuine, bounded reflection signature, not
catastrophic but clearly worse than the monopole case. Applying
`outflow_taper` speeds up the early transient (round_trip 1: 0.051 vs 0.113)
but converges to the EXACT SAME floor (1.865e-4) by round trip 4+ regardless
of taper — identical qualitative signature to the ring case (taper helps
monopole content, can't touch the part set by non-normal-incidence angular
content). **Since this floor appears with zero ring-specific machinery
involved, the ring-construction-artifact hypothesis is ruled out**: the
angular-incidence degradation is real, generic first-order-Mur physics that
responds to a source's multipole order, not a peculiarity of how this
project's ring IC happens to be built.

### Decay-to-quiescence methodology (new, 2026-09-15)

Instead of a single late-time shell-|S| snapshot, run a small sphere
(R~10, cheap), put energy in the middle, track a global energy trace over
MANY round trips (fast in a small box), and check whether it decays to a
flat floor (genuine absorption) or plateaus/re-grows (reflection). Run
`absorbing` alongside as a known-100%-reflective calibration control on the
same curve. Adapted from `tst/substrate/test_reciprocal_bc.py` (was test_saha.py)'s
`TestEquilibration.test_peaked_ic_equilibrates` peaked-IC methodology —
`ReciprocalBC`'s noise/thermalize machinery is NOT used here (that's scoped
strictly to ambient-vacuum-equilibration questions per
`doc/mems/feedback_saha_noise_scope.md`); only the "watch a disturbance
relax and check it fully quiesces" pattern is reused, with `outflow`
(absorbing, not conserved) in place of `ReciprocalBC` (reciprocal,
conserved).

**Caveat, found the same session**: this only works cleanly for an IC whose
energy is genuinely free to radiate. Tried with a toroidal-ring IC
(`lam_ring=1`) at R=10/hops=2 and got a **false negative** — outflow and
absorbing both stayed flat at ~99.9% of IC energy for the entire run,
indistinguishable. Root cause: a `lam=1` ring is a quasi-bound topological
structure (`project_spheromak_decay`/`project_saha_spheromak_decay` — these
only decay over tens of thousands of steps), so almost all its energy stays
locally bound in the winding and never reaches the boundary at all; the
tiny genuinely-radiated fraction is swamped in a whole-domain `u²+φ²` sum.
Same failure *shape* as the Saha-noise-swamps-decay mistake
(`feedback_saha_noise_scope`) — a different mechanism (bound structure
energy, not injected thermal noise) producing the same symptom (global
energy diagnostic insensitive to the actual signal of interest). **Fix, not
yet built**: restrict the energy sum to an outer shell excluding the ring's
near-field core (same idea as the existing shell-|S| diagnostic, just
tracked as a running trace instead of one snapshot) before drawing any
ring-source conclusion from this methodology.
