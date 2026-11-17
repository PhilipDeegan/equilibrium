# DST next — context dump, 2026-07-12

Session covered: the step_cross_v2 curl-gate R-sweep, a long detour into
electron stability/topology (what does a genuinely free, non-decaying
electron look like?), and a pivot into entanglement/Bell/canal physics.
Dumping full context + open questions + next ideas before picking a thread
back up.

> **STATUS (2026-08-20): the E/B conventions used below (§5, §7) are
> superseded, not settled.** `E := -grad(u)-dphi/dt` (§5) and the
> `step_cross`-sourced `E := -dphi_cross/dt` (§7) are each just one of
> THREE mutually-contradictory conventions later found in this codebase —
> none validated for a genuine 3D Poynting/energy-flux vector. See `now`
> and `doc/mems/project_dst_em_field_convention_crisis.md` before citing
> any specific `E:=...`/`B:=...` formula from this file as current.

## 1. step_cross_v2 curl-gate anisotropy confound — REOPENED, not settled

`project_step_cross_v2` memory has the full history. Short version: the
curl/holonomy gate failed to separate electron (must survive) from neutron
(must decay) at R=15/scale=3.0 — raising `ref` moved both survivals up
together instead of discriminating. Suspected this was confounded by the
BCC dispersion anisotropy (significant at 1.5-8 cell wavelengths, negligible
only above ~3.6 cells) since the curl plaquette is a single-hop measurement.

Retested at R=30 (scale=5.1) — got an almost identical failure curve to R=15,
initially logged as "confound ruled out, structural failure confirmed." **That
call was wrong and retracted same day**: `scale` sets the knot's overall
sphere radius, not the actual ring-current cross-section the curl gate
samples. The electron dipole IC's minor-width (ring cross-section) is only
≈0.562×scale cells — so scale=3.0 → 1.7 cells, scale=5.1 → 2.9 cells. **Both
are still below the 3.6-cell safe threshold.** The R=30 retest never actually
left the anisotropic band for the feature that matters.

**Open**: rerun with scale ≳6.5 (minor-width > 3.6 cells) before drawing any
conclusion — e.g. R=30-50 with `--sph-frac`≈0.22-0.25, not the 0.17 used so
far. R=80 companion run (sph_frac=0.06, N=2.78M cells) was killed by the user
for being too slow before completing — not required to settle this, a
correctly-scaled R=30-50 run should suffice.

If curl still fails at genuinely-resolved minor-width, THEN it's fair to call
it structural (not anisotropy-confounded). Remaining relational-gate options
at that point: (a) a loop threading BOTH windings' actual interpenetration
region rather than a fixed Cartesian plaquette, (b) build the linkage
detector from face-adjacent phi-diffusion patterns
([[project_dst_phi_diffusion]], already known to reconnect) — (b) is the
better-supported direction regardless.

## 2. Electron stability / topology — what does a non-decaying free electron look like?

Long chain of reasoning, corrected several times along the way:

- **lam reframed**: not a categorical "electrons have zero lam" flag, but a
  continuous mc²-type bookkeeping of phi/energy bound into a knot's winding.
  Electron likely carries a *small*, mass-ratio-scaled lam (~1/1836 of the
  proton's), not exactly zero. `probe_electron_translation_uniform.py` and
  `probe_electron_translation.py` already use lam=0 as the "electron: no
  lam" convention — that should be revisited as "lam≈0, not exactly 0" if
  this line of thinking is pursued further.
- **probe_moat_translation.py is the one file that got this wrong**: it
  carves the electron's moat as a deficit in **lam** (not u), contradicting
  both CLAUDE.md's canonical "electron = deficit of u" and the corrected
  lam-is-mass-only framing. Not yet fixed.
- **Why a lam-deficit "just works" but a u-deficit can't (without help)**:
  traced to the actual wave-step equations (`DampedLattice.step()`,
  `equilibrium/substrate/lattice.py:312-339`): `dphi_ik = -dt·σ_face·W·(u_k -
  u_i)` is a strictly **per-face** equation — for `dphi=0` everywhere you
  need u_k==u_i on *every* connected face, i.e. u exactly globally uniform.
  There's no collective/relational escape hatch on this side (unlike the
  u-update, which *sums* over 14 faces and so *does* admit divergence-free-
  phi cancellation — exactly the trick the dipole/ring electron already
  uses). lam's shape, by contrast, never enters this wave-step equilibrium
  condition at all (only sets σ, the rate) — which is *why* lam is free to
  hold any static shape, and why probe_moat_translation.py reached for it.
- **Corrected a real mistake mid-session**: first attempt at a "confining"
  term added `-m²u_i` directly to u's own update — that's first-order decay/
  friction toward zero, NOT confinement (confirmed empirically: this drained
  field energy to ~40-49% over 4000 steps regardless of mass value). The
  actual Klein-Gordon mass term needs to live in **phi's** equation (adding a
  uniform, isotropic `+σ_i·m²·u_i/30` correction across all 14 faces, since
  ΣW_k=30 exactly) so its divergence reproduces `-m²u` in u's *second*
  derivative, matching `∂²u/∂t² = ∇²u - m²u`. Implemented correctly in
  `run/substrate/probe_electron_trap.py`.
- **Earnshaw's theorem** ruled out a *static* linear "trap" well — a static
  potential in vacuum has no local minima in 3D (only saddle points), so a
  simple relaxation-to-target trap can't stably confine (confirmed: it
  didn't confine, and was additionally found to be dissipative — draining
  mass/energy rather than redirecting it, since it's linear damping/friction
  toward a fixed target, not derived from a real energy functional).
- **"Capacitance not heat" (Philip's framing) is the key distinction**:
  reversible delocalization (energy exactly conserved, structure spreads
  out — like a capacitor's energy oscillating losslessly between two
  conjugate forms, or a free quantum wavepacket spreading) is NOT the same
  as decay/dissipation (heat, photon emission — genuinely irreversible).
  Confirmed empirically: free (lam=0, mass=0) wave dynamics on the plain
  `SubstrateLattice` (no sponge, no C++) shows field energy jump once during
  an initial ~100-step leapfrog "phase-in" transient (phi starting exactly
  at 0 isn't the natural staggered-leapfrog phase relative to u) then hold
  flat while spread keeps growing — genuine reversible delocalization, not
  decay. A genuinely FREE electron delocalizing over time is not a bug —
  real free quantum wavepackets do this too (Δx grows with zero confining
  potential); it only needs to be non-spreading once BOUND (near a proton,
  with a real potential/lam well).
- **KG mass-term test results (R=15 and R=60, mass=0.05)**: inconclusive —
  mass=0 and mass=0.05 give nearly identical spread/energy curves in both
  domain sizes. The R=60 run confirmed the ~40-49% energy "loss" seen
  earlier is entirely the sponge boundary absorbing the wave once spread
  physically reaches `r_inner=0.85R` (energy drop happens at exactly the
  same step for both mass=0 and mass=0.05 runs) — not a conservation bug,
  not the mass term doing anything. mass=0.05 is simply too weak to show a
  confinement effect before the sponge dominates. **Open**: rerun with a
  much larger mass, or a domain/timescale where spread never approaches the
  sponge, to properly test whether the (corrected) KG term produces a
  genuine Yukawa-profile plateau.
- **Bulk vs boundary confinement — a live, unresolved fork** (Philip's
  question): the KG mass term is a *bulk* effect (intrinsic to the medium
  everywhere, like a plasma-frequency cutoff). An alternative, arguably more
  natural mechanism: a *boundary/interface* effect — a real, localized σ(λ)
  gradient/step somewhere inside the domain (not the sponge, which is built
  to absorb, not reflect) acting like a fiber-optic core/cladding boundary
  (total internal reflection) or a waveguide wall. Not yet built. Best
  version of this: **self-trapping** — if the electron's own (small) lam
  sources a local σ gradient, that's a self-consistent soliton-like
  mechanism (analogous to optical self-focusing solitons that create their
  own waveguide), not an externally-imposed apparatus. This is probably the
  most physically well-motivated remaining option and hasn't been started.

**Existing BC infrastructure relevant to this** (found while investigating,
not previously in memory):
  - `DampedLattice` sponge (`equilibrium/substrate/lattice.py:779-810`) —
    absorbing, pulls u/phi toward u_bg/phi_bg at the outer edge. Used in
    every probe today. NOT a confinement mechanism (opposite purpose).
  - `RadiationBCLattice` (`equilibrium/substrate/lattice.py:834+`) —
    Sommerfeld-style: refits and preserves the DC Coulomb 1/r tail at the
    boundary instead of damping it, rather than a plain sponge. Exists only
    for the **uniform** lattice (no equivalent found in the adaptive C++
    side — `inc/eq/core/data/lattice/bc.hpp`'s `BCType` enum only has
    NONE/ABSORBING/FUNCTIONAL/MPI_REMOTE, with sponge/dirichlet both mapping
    to FUNCTIONAL — no distinct radiative/Sommerfeld type there). Still an
    open/absorbing-style BC, not reflective.
  - `ReciprocalBC` (`equilibrium/substrate/reciprocal_bc.py`) — a third, distinct
    category: conservative + thermalizing. Whatever u leaves any boundary
    face gets matched by an equal random-weighted injection elsewhere on the
    boundary (`compensate()`), plus an optional zero-mean noise term
    (`thermalize()`) setting a genuine vacuum "temperature" without changing
    total u. Already validated (`TestSahaEquilibrium` etc.). Directly
    relevant to §4 below (decoherence-as-dilution-into-background) —
    not yet wired into any canal/entanglement probe.

### probe_electron_trap.py — status

File: `run/substrate/probe_electron_trap.py` (uniform lattice, not committed
— worktree only, see note at bottom). Docstring has a live checklist.
Completed/tested: baseline conservation check (confirmed conservative past
startup transient), linear relaxation well (confirmed non-confining,
dissipative — informative negative result), KG mass term (corrected
derivation, implemented, but inconclusive at the tested mass/domain sizes —
sponge dominates before any effect shows). Not started: harmonic well,
lam-shaped trap rehearsal, weak/strong mass sweeps at properly isolated
domain, oscillating/ponderomotive trap, the bound (proton+electron) case,
curl-feedback extension, and the boundary/guide-field (σ-step) mechanism
from the Earnshaw discussion above.

## 3. Bohmian mechanics / hidden-variable comparison

Raised as a "how would DST's guiding-wave analog work" question. Key points
established:

- DST is field monism (particle IS the field's own persistent feature) vs.
  Bohm's explicit dualism (separate real particle + real pilot wave) — more
  economical ontologically, if the electron-stability work above can
  actually produce a persistent structure.
- **Sharp, unresolved tension**: Bohm's nonlocality specifically comes from
  the pilot wave living in 3N-dimensional configuration space, not physical
  3-space. DST's u/phi field is strictly per-cell, 14-neighbor local — no
  configuration-space object anywhere. This means DST, as built, is
  structurally a **local hidden-variable theory**, and Bell's theorem says
  no local hidden-variable theory (deterministic or stochastic) can
  reproduce the experimentally-confirmed Bell/CHSH violations (S up to
  2√2≈2.828, loophole-free tests: Hensen 2015, Giustina 2015, Shalm 2015).
  This is a structural obstruction, not a matter of degree or propagation
  speed.
- Candidate resolutions discussed: (a) a genuine nonlocal channel (cuts
  against the whole locality ethos), (b) superdeterminism — reject that
  measurement settings are freely/independently chosen — the route taken by
  **Gerard 't Hooft's Cellular Automaton Interpretation of Quantum
  Mechanics**, the closest existing serious academic precedent for a theory
  structurally like DST (deterministic, local, discrete substrate);
  controversial for the same reason it works. (c) accept DST doesn't match
  full QM correlations, as a bold falsifiable claim.
- **(d), and the one DST actually already has a sketch of**: deny Bell's
  *factorizability* premise instead — see §4.

## 4. Bell inequality / "canal" entanglement — bell.py found, corrected

Philip pointed to `res/zip/ham/equilibrium/physical/dst/bell.py` — archived
(along with a lot of other older, non-lattice-backed conceptual sketches,
not because it was found wrong) but directly relevant and a genuinely
clever resolution I'd missed initially.

**The core idea**: Bell's proof requires the hidden variable λ to be
factorizable — P(A,B|θ_A,θ_B,λ) = P(A|θ_A,λ)·P(B|θ_B,λ) — assignable
*separately* to each particle. DST has no particles to hand separate copies
to; both measurement patterns live in the same substrate field. The "canal"
is a single non-factorizable region of substrate connecting A and B,
established causally at pair-creation time — not two independent local
variables. Bell's proof doesn't apply if λ can't be split that way. This is
structurally similar to Bohm's escape (a single non-factorizable object for
the pair) but the shared object is ordinary field structure in real 3-space,
not an abstract configuration-space wavefunction.

Three models in the file, all still just hand-rolled Malus's-law/CHSH
numerics, NOT derived from the actual lattice:
  - `classical_lhv` — genuinely separable λ, S=√2 (no violation) — a sanity
    check that the framework can represent Bell's assumed world correctly.
  - `dst_canal` — cascade approximation (A measures, re-emits, B reads the
    propagated result), S=2√2, but needs timelike ordering — not valid for
    genuinely spacelike-separated measurements.
  - `dst_joint_canal` — the intended correct model: both sides sample a
    pre-existing joint state simultaneously, no ordering/cascade/FTL, S=2√2,
    valid for spacelike separation.

**Objection raised and accepted (Philip)**: `dst_joint_canal` currently
assumes the canal is perfectly preserved regardless of distance/elapsed
time — unrealistic ("in practice it's nonsense" over large distances,
matching how real long-baseline entanglement distribution is fragile and
lossy). Real decoherence should degrade S from 2√2 toward √2 (or lower) with
distance/time, and that's a genuine DST-specific, falsifiable prediction
(standard QM attributes all such loss to apparatus/environment, not to the
entangled state itself decaying with distance).

**"Decoherence is just regression to the mean" (Philip)** — reframed the
degradation mechanism correctly: not an ad hoc exponential decay constant
bolted on, but the same dilution/delocalization mechanism already measured
in §2's conservation checks. A special, low-entropy, correlated
configuration relaxes toward whatever's statistically typical for the much
larger surrounding substrate, purely through conservative dynamics — no new
dissipation needed, same phenomenon as free-electron delocalization, just
applied to a correlated pair instead of a single knot.

**Refinement/caveat**: this reasoning applies cleanly to a bare u-deficit
disturbance (which genuinely self-disperses per Huygens, confirmed in §2).
But if the canal's carrier is a properly-built **photon** — per CLAUDE.md's
own vocabulary, "photon = face topology (phi ring)" — and built as a
topologically-protected circulation (uniform u + divergence-free phi, the
same non-dispersing construction validated for the electron dipole) rather
than a raw pulse, it should NOT self-disperse via Huygens as it propagates
(matching how real photons keep a well-defined wavelength across
cosmological distances). So if canal decoherence is real for a properly-
built photon carrier, the mechanism has to be something other than plain
self-dispersal — most likely interaction with whatever's actually sitting
in the intervening substrate (background structure, other windings, noise)
— i.e. a genuine step_cross-style dissipative/gate-mediated process, or
dilution into a `ReciprocalBC`-style thermal background (see §2's BC
inventory) — not the photon unwinding on its own.

**Resolution requirement (Philip)**: whatever carries the canal signal needs
enough cells across its own characteristic wavelength/ring-size (≥3.6,
better ≥8 for safety margin, per the R-sweep anisotropy threshold in §1) to
avoid it being a lattice-anisotropy artifact rather than real physics. Since
cost is driven by the *overall* domain radius R (∝R³ for a sphere,
independent of what's inside), this is a one-time "build the photon right"
constraint, not something that compounds with separation distance.

**Cylinder domain (Philip)**: for testing genuinely large separation
distances cheaply, a cylindrical domain (long axis along the propagation
direction, modest transverse radius) is O(radius²·length) cells instead of
O(radius_sphere³) — a large saving when only one axis needs to span a big
distance. **Implemented** (uncommitted, worktree only):
  - `KelvinLattice.generate_cylinder(radius, half_length, axis)` in
    `equilibrium/math/geo/kelvin.py` — same candidate-point generation as
    `generate_sphere`, different (cylindrical) mask.
  - `SubstrateLattice._build_cylinder()` / shared `_build_from_points()`
    helper in `equilibrium/substrate/lattice.py` (refactored out of
    `_build_sphere` without changing its behaviour — sanity-checked,
    sphere construction still gives identical N).
  - **NOT YET wired to a public constructor path** — `SubstrateLattice.
    __init__`/`DampedLattice.__init__` still only accept `radius_cells` →
    sphere. Needs a `shape`/cylinder kwarg or a separate factory method
    before it's actually usable from a probe script.

**test_qgate.py** (existing, already committed) is the natural physical
building block for an on-lattice canal/Bell probe: two coupled cells with a
"canal" flux φ_x reproduce Rabi oscillation, X-gate, S₂-gate, and Born-rule
statistics from the real `(u,phi)` step dynamics on exactly two cells — not
a toy, genuine substrate physics. The natural upgrade discussed: extend this
from two *adjacent* cells to two *genuinely separated* cells (real lattice
distance D), connected by an actual extended canal structure at t=0 (not an
assumed joint-probability formula), then measure how CHSH S degrades as a
function of real separation and elapsed time — deriving the coherence
length/time empirically instead of assuming an ad hoc exponential. Proposed
staged approach (mirroring §2's discipline): build the classical/separable
baseline first (expect S≈√2, validates the measurement pipeline), THEN
attempt the harder non-factorizable joint-canal construction.

## 5. Concrete open next steps (unordered — pick based on priority)

- [ ] Rerun step_cross_v2 curl/opposed gate sweep at scale ≳6.5 (minor-width
      >3.6 cells) — e.g. R=30-50, `--sph-frac`≈0.22-0.25 — to actually settle
      whether curl's failure is structural or still anisotropy-confounded.
- [ ] Fix `probe_moat_translation.py`'s moat construction (lam-deficit →
      u-deficit or a proper self-trapping construction), OR retire it in
      favour of the boundary/guide-field mechanism below.
- [ ] Build the boundary/guide-field confinement probe: a real σ(λ) step or
      gradient inside the domain (not the sponge), testing reflection-based
      confinement; ideally self-consistent (lam sourced by the structure's
      own presence, soliton-style) rather than externally imposed.
- [ ] Re-test the KG mass term at a domain/timescale where spread doesn't
      reach the sponge, and/or a much larger mass value, to get a real read
      on whether it produces a genuine Yukawa-profile plateau.
- [ ] Wire up a public constructor path for the cylinder lattice (shape
      kwarg or factory method on `SubstrateLattice`/`DampedLattice`).
- [ ] Build the on-lattice canal/Bell probe: extend `test_qgate.py`'s real
      2-cell physics to genuinely separated A/B cells on an (ideally
      cylindrical) domain; classical/separable baseline first, then the
      non-factorizable joint-canal construction; use `ReciprocalBC`'s
      `thermalize()` to give the intervening path a real, measurable
      background noise level rather than an assumed decay constant.
  - If/when a real canal is built: reuse `bell.py`'s CHSH machinery
    (`chsh()`, `correlation_curve()`) against the lattice-derived
    correlations instead of the hand-rolled Malus's-law numerics.
- [ ] Revive/extend `bell.py` out of `res/zip/ham/` once the on-lattice
      canal work has something real to ground it in.
- [ ] Longer-term/theoretical: DST's strict per-cell locality structurally
      cannot reproduce Bell violation via any local-hidden-variable
      mechanism (Bell's theorem) unless factorizability is denied — the
      canal/non-factorizable joint-state idea is the live candidate; still
      needs to be *derived* from real lattice dynamics, not just asserted.

## Note on file locations

This session ran as an isolated background job — file edits went to a git
worktree (`.claude/worktrees/mighty-percolating-squirrel`), not directly to
the main checkout, and nothing was committed/pushed per standing
instruction. New/modified files from today (`probe_electron_trap.py`, the
cylinder-lattice additions to `kelvin.py`/`lattice.py`, this file) currently
only exist in that worktree path — they won't show up in the main checkout
until copied over manually.

**Resolved same day**: everything above (this file's rewrite, the topology
memo, `probe_electron_trap.py`, the cylinder additions) was copied out of the
worktree into the main checkout on 2026-07-13 and verified byte-identical.
The worktree itself was left in place, untouched, in case it's still wanted.

---

# 2026-07-13 session — guide field, lam-protection sweep, "too lossy?",
neutron-decay-quasi-particle reframing, and the step_cross quantization/
barrier design (mostly resolved this session)

## 1. Housekeeping done this session

- Corrected `feedback_cpp_workflow.md`: building/running C++ (including
  Python scripts that dispatch into the compiled extension) IS allowed —
  the old blanket ban was overcorrection for a past session getting build
  commands wrong. Refinement added same day: for anything beyond a trivial
  build/run, write a proper test rather than iterating ad hoc (build → run a
  probe → eyeball → tweak → rebuild). `tst/substrate/test_cpp_compare.py`
  now has `test_energy_conservation_nonuniform_lam` as an example of this.
- New memory: `doc/mems/project_dst_lam_definition.md` — canonical lam
  definition (mc² worth of phi *geostationarily locked out* of the broader
  context, not just "mc² bookkeeping"), the corrected magnetic-coupling
  reasoning (see §4 below), and the "reinvigorate lam" fudge-labelling rule.

## 2. Guide-field probe — built, but needs a redesign before it's used again

`run/substrate/probe_electron_guide_field.py` (new) tests a *boundary*
confinement mechanism (a static `lam` shell acting as a waveguide wall via
face-averaged `sigma_face`) as the alternative to `probe_electron_trap.py`'s
*bulk* Klein-Gordon mass term. Baseline run (`--lam-wall 0`) reproduced the
known free-spreading behaviour; a real `lam_wall=5` run surfaced a bigger
problem (§3).

**Corrected same day**: the probe's spherically-symmetric radial `lam(r)`
shell has no axis, no handedness — it's a scalar/dielectric boundary, not a
magnetic one. A genuine magnetic-type guide field couples via a cross
product (`F=qv×B` for orbital motion, OR `μ×B`/`∇(μ·B)` for an intrinsic
moment — Stern-Gerlach/Zeeman couple to spin with zero translational
velocity, so "static things can't feel B fields" is FALSE, corrected after
Philip pushed back). The scalar "ball of charge" toy has neither orbital
motion nor any moment-analogue, so it's the one construction a magnetic
field genuinely cannot act on. The dipole/ring electron *does* have internal
circulation (the poloidal Taylor winding) — that circulation is DST's analog
of an intrinsic moment, so it's the right target. **Not yet rebuilt**: the
next version should be axisymmetric, sharing the dipole's own circulation
axis, with `lam` increasing toward both ends (mirror/bottle geometry,
converging field lines) rather than a spherical shell — same family as the
existing `B_lam` cylindrical shell in `probe_electron_lorentz.py`, pinched
into a mirror instead of a straight cylinder.

## 3. Energy conservation under nonuniform lam — bug found, "fix" was WRONG, reverted, still open

Running the guide-field probe surfaced `energy_frac=1.21` (field energy
*increasing*, not just failing to be conserved) whenever `lam` varies
spatially across a face. Root cause identified precisely: `step()`'s
u-update uses each cell's own local `sigma_i`, while the phi-update uses the
face-averaged `sigma_face=(sigma_i+sigma_j)/2` — confirmed identical in both
the Python fallback (`equilibrium/substrate/lattice.py:328-356`) and the C++
kernel (`inc/eq/substrate.hpp`'s `pass1`/`pass2`). For uniform `lam` these
coincide; for nonuniform `lam` they don't, breaking the adjoint pairing
needed for energy conservation.

**The "obvious" fix (swap the u-update to use `sigma_face` too) was tried
and made the naive energy check WORSE (1.39x → 2.99x drift), not better —
reverted immediately, no trace left in `lattice.py`.** Diagnosis: `σ(λ)` is
documented as each cell's own *local clock rate* (a per-cell property, not a
shared medium impedance), so using the cell's own `sigma_i` in its own
u-update may be the physically correct behaviour, not a bug — and under
genuine time dilation, naive flat-coordinate energy (`0.5u²+0.5ΣWphi²`, no
lapse-type reweighting) isn't expected to be conserved anyway (direct GR
analogy: coordinate energy isn't conserved across differing gravitational
time dilation without a lapse-weighted correction).

**Open**: what IS the correctly-conserved quantity (if any) under nonuniform
`lam`, and does it matter for the guide-field mechanism? `tst/substrate/
test_cpp_compare.py::test_energy_conservation_nonuniform_lam` (new) documents
the current behaviour (~1.4x drift on a nonuniform lam shell over 2000
steps) as a fact, not an assertion of correctness — deliberately not
resolved, needs the user's read on whether the naive quadratic form is even
the right invariant to expect here.

## 4. Electron dipole + nonzero lam sweep — DONE, mechanism understood, doesn't validate the mass-ratio hypothesis

`sh/run.sh` ran `test_electron_dipole.py --lam-target {0.04, 1.0}` at R=80
(results in `tmp/electron_dipole_lam_sweep/`), compared against the existing
`lam_target=0` baseline (τ_φ≈16 t.u., `exp/2026/07/10/19_56_...`):

| lam_target | phi_rms @ step 5000 | fraction remaining | implied τ_φ |
|---|---|---|---|
| 0 (baseline) | 4.565e-4 | 20.2% | ≈16.0 t.u. |
| 0.04 (~m_e/m_p × lam_proton) | 4.795e-4 | 21.2% | ≈16.6 t.u. (negligible) |
| 1.0 | 9.842e-4 | 43.6% | ≈30.9 t.u. (real, ~2x) |

Traced the mechanism: `step_cross`'s C++ kernel already multiplies its drain
by `sigma_i²` (`inc/eq/core/data/lattice/adaptive_lattice.hpp:1210,1221`) —
lam suppressing crossing is a **pre-existing** feature, not something new
found this session; this sweep is the first *quantitative* test of it
against the actual electron-dipole IC. `lam_target=0.04` (naive mass-ratio
guess, `~1/1836` of `lam_proton≈73`) is too weak to matter — same
"too-weak-to-observe" pattern as the KG-mass-term probe's `mass=0.05`.
`|tor|max` stays at machine-noise level even at `lam_target=1.0` — the
dipole doesn't collapse toward a linked/toroidal topology under nonzero lam.
**Implication**: if the real electron's lam really is mass-ratio-scaled,
that's nowhere near enough to protect it from the `step_cross` decay
artifact — consistent with the standing 2026-07-10 view that the decay needs
a `step_cross` v2 *gate* fix, not a lam patch.

## 5. "Are we too lossy?" — yes, and it's bigger than decay-rate tuning

Philip's question: real photon emission is causal and quantized; EM fields
(if not externally imposed) require genuine circulation ("a rotating
frame"); is `step_cross`/`phi_diffusion` too continuous/acausal compared to
that? **Confirmed, and sharper than expected**: `probe_poynting_ring.py`
(paused work, see §7) already established that `E := -grad(u)-dphi/dt` is
IDENTICALLY zero under the free wave step alone (the two terms cancel by
construction) — a real, non-tautological E only appears from `step_cross`'s
own `dphi`. **`step_cross` isn't just an inertia/decay mechanism — it's
literally what sources the E-field (and via `curl(phi)`, B, and `S=E×B`,
Poynting radiation) in this model.** So quantizing/gating `step_cross`
correctly doesn't just fix "electron decays too fast" — it makes radiation
itself causal-and-discrete, directly answering the original question. `J`
(current density) is likewise already `~div(phi)` in the reconnection test
(`current_sheet_j`) — no independent electrical unit anywhere, confirming
this codebase already implicitly works in `μ₀=ε₀=1` natural units (see §6).

Separately: `doc/dst.md`'s own inertia section already specifies the
crossing should be **all-or-nothing** ("the crossing either happens or it
doesn't: a gate condition on the instantaneous phi state"), quantized like
overcoming a real energy-level barrier — but `step_cross` v1 AND every v2
gate prototype tried before today (shear/product/opposed/curl, all in
`project_step_cross_v2`) are continuous, dt-proportional trickles, just with
smarter and smarter continuous coefficients. None of them were ever actually
quantized. This invalidates re-using v1/v2's survival numbers as evidence
about a properly-gated mechanism — they measured a structurally different
kind of operator.

## 6. Why photons/phi-waves at all for neutron decay? — real answer already in doc/dst.md

Philip's question (neutron → proton + electron + "quasi particle", why
photons/waves?) has a direct answer already in the spec: **"the antineutrino
is the topological accounting entry for the change in winding — it carries
the Δu needed to close the ledger"** (`doc/dst.md:210-212`) — a *ledger
entry*, not radiation escaping to the environment, and no photon anywhere in
the primary process (matches real beta decay). This reframes the neutron-
decay success criterion: not "does the electron survive long enough /
does the decay rate look sensible" but **"after a decay event, can we point
at three separate things in the lattice state — proton knot, electron
dipole, and a third coherent structure carrying exactly the remaining Δu —
with nothing left over as unattributed diffuse loss?"** Current `step_cross`
has no explicit realization of this third channel — shed circulation just
diffuses/gets absorbed by the sponge, which is functionally the same as
"radiating it away," the exact thing that shouldn't happen if the spec's own
ledger-entry framing is right. **Not yet built**: a probe that checks
whether the shed circulation, released *coherently* (no `step_cross`/
diffusion draining it), organizes into an identifiable propagating structure
(the quasi-particle) rather than dispersing like generic noise.

## 7. Poynting-ring calibration sweep — RUNNING, results not yet seen

`project_poynting_ring_dipole` (paused 2026-07-03) had confirmed the E/B/S
mechanism at a cheap smoke-test scale (R=25, 400 steps) but the fuller run
needed to see the near-field-vortex → far-field-radial transition (R=35-40,
several thousand steps) was killed by an external timeout before finishing.
Relaunched this session via `sh/run.sh` as a 3-point sweep (R=25/40/60,
2000-8000 steps, output in `tmp/poynting_ring_sweep/`) — chosen because it's
**derivable with zero particles** (pure oscillating phi ring, no proton/
electron mass-ratio calibration needed at all), isolating the field/
radiation behaviour so different candidate real-world scales can be
compared by which one "looks like" a real dipole radiation pattern, before
tackling the particle-mass calibration separately. **Completed and reviewed — inconclusive/negative, not yet diagnosed.** All
three R values (`tmp/poynting_ring_sweep/poynting_R{25,40,60}.png`) show
`|E|_max`/`|B|_max`/`|S|_max` decaying smoothly and slowly, matching closely
across different R at the same t (near-field physics correctly insensitive
to domain size). But visually (checked `poynting_R60.png`, 6 snapshots
t=0..41): the bright Poynting-flux region stays a compact, roughly
STATIC-looking blob near the ring the whole run — it does not visibly
expand outward into a far-field radial pattern over time. No clear
near-field-vortex → far-field-radial transition seen. Not diagnosed why —
candidates: (a) run isn't long enough for propagation to reach the domain
edge at whatever the effective local wave speed is, (b) the `E:=
-dphi_cross/dt` construction (sourced only from `step_cross` at
`cross_scale=0.01`) may just be too weak to produce a visible propagating
component against the near-field, (c) something genuinely off in the setup.
**Next**: check the actual light-crossing time (ring radius to `slice_r_max`
at the model's local propagation speed) against the run's total `t` to rule
out (a) first — cheapest check, no rerun needed if the numbers say the run
was simply too short.

## 8. step_cross quantization / vacuum-capacitance barrier — design mostly RESOLVED this session

Long back-and-forth (many wrong turns, all corrected in-thread — logged here
so they aren't re-derived from scratch):

- **No accumulator.** First instinct was a persistent per-cell accumulator
  (charge building up like a capacitor over ticks). WRONG — contradicts
  `doc/dst.md`'s own "gate condition on the *instantaneous* phi state," and
  every other operator in this codebase (`step`, `step_lam`, `step_cross`)
  is a pure stateless function of the current arrays. Corrected: purely
  instantaneous, per-tick, no memory.
- **`μ₀=4π×10⁻⁷` (raw SI) is a category error.** `H/m` decomposes to
  `kg·m·s⁻²·A⁻²` — DST has no primitive for "mass" (`feedback_conventions`)
  or "current/charge" yet, so the SI value is literally inexpressible, not
  just unit-converted wrong. Also: since `Z_0=ℓ_P/t_P` in DST's own
  normalization (`K=ℓ_P/(t_P·Z_0)≡1`), `μ₀=Z_0/c` reduces to a dimensionless
  ratio near 1 in DST's natural units, not `1.26e-6` — confirmed empirically
  by `probe_poynting_ring.py`'s own E/B/S formulas, which never reference
  `μ₀`/`ε₀` at all (already implicitly natural units).
- **`m_p/m_Planck≈7.7×10⁻²⁰`** (`project_dst_proton_model`, computed against
  the *real* physical Planck length) is the right DST-native, dimensionless
  anchor instead — no undefined kg/A conversion needed. Labelled `M0_VOID`
  (void = true-Planck reference in `doc/dst.md`'s own terminology).
- **`N_SAGA` cubing was WRONG.** `N_SAGA=2880` is a pure *throughput/tick*
  ratio (`σ_min=1/N_SAGA` at the BH core; the one existing code precedent,
  `_LAM_SINK_COST=N_SAGA*_KELVIN_VOLUME`, combines it with volume *linearly*,
  as a ticks-cost multiplier, never as a spatial exponent). Retracted the
  whole "black-hole-limit vs void-limit range for `m0`'s magnitude" idea —
  `lam`/rate must NEVER modulate the barrier's magnitude, only how fast a
  cell's own state evolves toward it (same principle re-violated and
  re-corrected twice more below before it stuck).
- **AMR-mesh `L`/`d1`/`leaf_level` was WRONG** as a stand-in for real cell
  size. It's a numerical accuracy knob (mesh resolution), carries zero real
  density information. Real local compactness is carried entirely by `lam`
  (that's the whole point of `σ=1/(1+lam)` — GPS-style time dilation IS
  real cell-size compression, same underlying fact, not two things).
- **`σ³`-via-`lam` was WRONG DIRECTION.** Since real size shrinks with `lam`
  (denser = smaller true cells), `m0·σ³` only ever *shrinks* the barrier for
  dense cells — cannot close the "too small" gap, and reintroduces lam into
  the barrier's magnitude (re-violating the same principle again).
- **Cubing in general was WRONG.** `doc/dst.md`'s own accounting only ever
  uses `L¹` (`σ=L/(1+λ)`, `d_ij→L`) and `L²` (`A_ij→L²A_0`, bandwidth) — no
  `L³`/volume term anywhere in the model's dynamics. Squaring (bandwidth-
  matching) was also tried and also wrong — not because of a shape/prefactor
  issue (same-shape ratios cancel prefactors exactly, sphere-vs-cube-vs-
  Kelvin-cell doesn't matter for a ratio) but because the based premise
  (cubing/squaring anything here) was never right to begin with.
- **RESOLVED**: `cell.L` is simply the real centre-to-centre distance
  between adjacent cells (`doc/dst.md`: "the centre-to-centre distance
  between adjacent cells is `ℓ_P`") — a plain linear quantity, full stop.
  `scaling_factor = cell.L(real) / local_discrete_cell.L(real Planck length,
  1.616e-35m)` is therefore just a linear ratio, no exponent at all. Final
  formula:

  ```
  m0 = scaling_factor × void_limit
  scaling_factor = (real knot radius in metres) / (1.616e-35 m) / (probe's own `scale` units)
  void_limit = m_p/m_Planck ≈ 7.7e-20   (or the relevant knot's own mass ratio)
  ```

  Neutron example (`r=2.51e-15m`, `scale=4.8`): `scaling_factor≈3.24e19`,
  `m0≈2.49` — genuinely order-unity, directly comparable to `u_depth=1.0`/
  `B0=1.0` used throughout these probes (unlike every earlier attempt, which
  was either ~17 orders of magnitude too small, always trivially exceeded,
  or absurdly large, never crossable).
- **Proposed default going forward (not yet settled — needs to stay
  case-by-case, not hardcoded)**: build future probes at `scale = 0.25·R`
  (proton radius = 25% of domain) as a standard convention — also comfortably
  clears the `project_step_cross_v2` BCC-anisotropy confound threshold
  (needs minor-width ≳3.6 cells; `scale=25` at `R=100` clears it with a lot
  of margin, unlike the `scale≈4-5` used in probes so far). Under this
  convention, `m0_proton≈0.16` (`r_p=0.84fm`, `scale=25`). **Philip's
  explicit caveat**: defaults are fine, but the calibration must still
  support **per-knot, case-by-case computation** (electron vs proton vs
  neutron each have their own real radius and mass ratio) — not a single
  hardcoded constant for everything.

**Implemented**: `gate="barrier"` in `run/substrate/adaptive/cross_v2.py`'s
`CrossV2` class — stateless, per-cell **pooled** (not per-face — "a cell
prefers a certain potential" is one preference per cell, not 14 independent
face budgets) drive `Σ_k W_k·max(-phi[i,k],0)` compared against `m0`,
all-or-nothing (`G_link∈{0,1}`, no partial/fractional crossing), reusing the
v1 drain+mirror kernel exactly (only the gate value changed). Self-tests
added and passing: v1-equivalence and conservation unaffected; a sub-barrier
drive produces **exactly zero** drift over 500 ticks (not just "small" —
genuine rejection, no accumulator, no leakage); an over-barrier drive fires
and matches the v1 kernel bit-for-bit on firing cells.

## 9. "Reinvigorate lam" fudge — noted, not yet needed

If dipole/flux-tube structures are found to genuinely decay under
`step_cross` (distinct from the operator-artifact decay already understood),
a forcing term that restores/"reinvigorates" `lam` to keep the structure
alive for study is legitimate — **but only if explicitly labelled as a
fudge** in code comments and any report.md, never presented as derived
physics (`project_dual_goals` discipline: keep the cheap-approximation and
rigorous-theory layers honestly distinguished). Not applied anywhere yet.

## 10. Concrete next steps (unordered)

- [ ] Look at the Poynting-ring sweep results (`tmp/poynting_ring_sweep/`,
      job `bmwk5698w`) once done — does any R show a real near-field-vortex
      → far-field-radial transition?
- [ ] Test the calibrated barrier gate (`m0≈2.49` neutron / `≈0.16` proton-
      convention) against the REAL neutron probe (not just the synthetic
      self-test) via `CrossV2` — does it change `step_cross_v2`'s R1/R2
      (electron survives / neutron decays) discrimination result at all?
- [ ] Decide the electron's own real-radius anchor (classical electron
      radius? reduced Compton wavelength? something else?) rather than
      borrowing the proton's `r_p` — needed before the electron-dipole probe
      can get its own properly-calibrated `m0`.
- [ ] Redesign `probe_electron_guide_field.py` per §2 — axisymmetric mirror/
      bottle geometry sharing the dipole's own circulation axis, not a
      spherical shell.
- [ ] Settle (or explicitly leave open) what quantity, if any, IS conserved
      under nonuniform `lam` (§3) — bears on whether the guide-field
      mechanism's numbers can ever be trusted as-is.
- [ ] Build the neutron-decay "third product" probe per §6 — release the
      toroidal winding's shed circulation coherently (no `step_cross`/
      diffusion draining it) and check whether it organizes into an
      identifiable propagating structure rather than dispersing like noise.
- [ ] Settle the `scale=0.25·R` default convention (§8) — or confirm
      per-probe custom calibration should remain the norm with no shared
      default at all.
- [ ] Re-run the `step_cross_v2` curl/opposed R-sweep at genuinely-resolved
      minor-width (≳6.5 scale, still open from 2026-07-12) — now easier
      given `scale=25`-style defaults would already clear this incidentally.
