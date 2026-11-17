# Electric/magnetic energy conversion probe — leak mechanism derived, confirmed, and scale-checked against real gradients

## Motivation

Following the Poynting-ring far-zone diagnosis (same day,
`exp/2026/08/19/15_10_barrier_drive_decay_diagnosis/`), before chasing any
more far-field radiation patterns: check the more basic question directly —
does DST's plain wave dynamics (`evolver.step()`, no `step_cross`, no gate,
no `m0`/`cross_scale` — none of that was active anywhere in this probe)
support real energy exchange between an electric-like form (`u`) and a
magnetic-like form (field circulation)?

## Attempt 1 — `equilibrium.substrate.mhd`'s SQ_PLAQUETTES (4-square-face loop)

`magnetic_energy = 0.5*sum(|B|^2)`, `B` via `mhd._b_vec`'s vectorized
equivalent. Result: magnetic energy EXACTLY frozen (bit-identical every
step) for both a poloidal-only and a full curl-full-dipole seed. Looked
clean but Philip flagged `mhd.py` may not be the valid/current module —
correct to question, see below.

## Attempt 2 — `equilibrium.substrate.discrete_curl` (the canonical 2-hex+1-sq triangle)

Switched to the doc-consistent module. Completely different result:
"magnetic energy" (`0.5*sum(circulation(phi,nb)**2)` over all valid
elementary triangles) is NOT stable — grows 40-80x within a few tens of
steps on a PURELY POLOIDAL field (no toroidal content seeded at all).
Two confounds checked and ruled out directly (not assumed away):
- **Resolution sweep** (`cells_across`=22/44/88, cell size fixed at 1.0,
  step count scaled to `cells_across` for a fair matched-phase
  comparison): peak ratio INCREASES with resolution (41x→63x→80x), ruling
  out "too coarse a test."
- **Boundary/margin check** (tight-margin+no-BC, tight-margin+sponge,
  wide-margin 4.91x/27x-volume+no-BC, all at `cells_across=22`): all three
  give IDENTICAL growth curves to 3-4 significant figures. Rules out
  reflection pile-up.

## Root cause, derived analytically from the actual update rule

```
du_i        = -dt · sigma_i    · sum_k W_k · phi[i,k]           (u from old phi)
dphi[i,k]   = -dt · sigma_face · W_k · (u_new[nb] - u_new[i])    (phi from new u)
```

For the 2-hex+1-sq triangle (legs A→B via hex h1, B→C via hex h2, C→A via
mirrored square sqk):

```
dCirc/dt = -dt·sigma·[W_hex(u_B-u_A) + W_hex(u_C-u_B) + W_sq(u_A-u_C)]
         = -dt·sigma·(W_hex - W_sq)·(u_C - u_A)     [W_hex=3, W_sq=1]
```

Nonzero whenever `u_C != u_A` — a real, derived, non-vanishing source term,
NOT a numerical instability. It only vanishes if `W_hex = W_sq`, which they
aren't. This explains every empirical symptom: boundary-independent (it's
purely local), grows with resolution (more triangles, each still driven by
this term), and doesn't discriminate toroidal from poloidal content
(depends only on `u`, fires regardless of what's seeded).

**Why the macro ring circulation (`phi_ring_circulation`, used throughout
this project's other work) IS exactly conserved**: that loop is built
entirely from same-weight square-face hops. Uniform `W_sq` around a closed
loop telescopes to exactly zero by the same algebra, with no asymmetry term
to survive.

**Philip's catch**: the 2-hex+1-sq triangle isn't "wrong" in general — it's
`doc/dst.md`'s own real minimal neutron-winding topology. It's just the
wrong tool for a conserved circulation/magnetic-energy proxy, because it
mixes two different face weights.

## Rebuilt with a weight-consistent (all-square) loop + physically-motivated background

Philip: even a same-weight loop's cancellation should require uniform
sigma across the loop — real space isn't uniform (`doc/dst.md`'s own
vacuum hierarchy: solar system vacuum is compressed relative to
intergalactic, never quite reaching true void within a finite region).
Rebuilt with `mhd.py`'s SQ_PLAQUETTES-style B (same-weight, all `W_sq`) and
a physically-shaped background `lam(r) = LAM_SURFACE·R_SPH/max(r_cell,R_SPH)`
— a `1/r` falloff, the same shape as `u=GM/(c²r)` (Coulomb/Newtonian, per
Philip) already used in the doc's own per-body throughput table, rather
than an arbitrary on/off toggle.

**Result, two configs (`cells_across=22`, 30 steps)**:
- `lam=0` everywhere (pure void): magnetic_ratio = exactly 1.000000 at
  every step. Matches the analytical prediction — uniform sigma (including
  uniformly zero) telescopes exactly.
- Background `lam(r)` ranging 0.61→1.0 across the domain (mild, `LAM_SURFACE=1.0`):
  magnetic_ratio genuinely deviates — 1.000476, 1.000840, 1.000502,
  1.000212, 1.000524, 1.000850 — small but real (the control sits at
  exactly 1.0 to the last printed digit, so any deviation here is signal,
  not floor noise).

**Conclusion**: DST's plain wave dynamics CAN convert energy between
electric (`u`) and magnetic (circulation) form, but the mechanism is
switched on specifically by a GRADIENT in `sigma`/`lam` (matter's edge),
not by matter being present per se — a uniformly nonzero `lam` telescopes
to zero just as cleanly as uniformly zero `lam` does, by the same algebra.
This is a real, derived, non-trivial structural prediction, not an
assumption: conversion happens crossing into/out of a mass's influence, not
inside a uniform vacuum OR inside a uniform medium.

## Scale check against real gradients — the tested effect is many orders of magnitude stronger than anything realistic

`LAM_SURFACE=1.0` was explicitly a qualitative "does the mechanism exist at
all" check, not a calibrated value. Checked against:
- `run/substrate/earth_moon_gradient.py` (already-existing, validated,
  real-number model: `u=GM/(rc²)` along the Earth-Moon axis, cross-checked
  against full Schwarzschild GR to sub-ns/day agreement): `u` ranges
  **6.96e-10** (Earth surface) to **1.86e-11** (L1) — **9-10 orders of
  magnitude weaker** than `LAM_SURFACE=1.0`.
- Classical `u=GM/(rc²)` estimates at galactic/cluster scale (this session,
  not from an existing script): Milky Way at solar radius ~5.6e-7, Milky
  Way halo at virial radius ~2.9e-7, galaxy cluster edge (3 Mpc) ~1.6e-5,
  void 300 Mpc from the nearest big cluster ~1.6e-7 — **2-5 orders of
  magnitude weaker** than `LAM_SURFACE=1.0`, even at the strongest
  realistic case tested (cluster edge).

**Scaling also checked against the already-derived single-proton value**
(`void_limit = m_p/m_Planck ≈ 7.7e-20`, [[project_dst_proton_model]] /
`doc/dst_next.md` §8). Reran `probe_energy_split_lam_background.py` with
`LAM_SURFACE` swept 1.0 → 1e-3 → 1e-6 → 1e-9 → 1e-12 → 7.7e-20: the
deviation shrinks consistent with the derived formula's linear dependence
on the background `u`-difference, indistinguishable from the pure-void
control by `1e-6`, extrapolating to ~6.5e-23 at the proton value — below
float64's ~1e-16 relative precision floor.

**Framing correction (Philip)**: comparing against a single isolated
proton's own surface value is not the physically meaningful comparison —
we never observe an isolated single proton; the macroscopic `u` values
(Earth, Sun, galaxy) already computed above ARE the aggregate sum over
~10^51+ individually-tiny proton contributions ("the upper limit on time
dilation is the evidence of the available protons nearby" — same
aggregate/superposition logic as [[project_proton_earth_dipole_unification]]'s
Earth-magnetic-dipole framing earlier the same day). The single-proton
value is the microscopic unit, not a separate weak-case benchmark in its
own right. **Conclusion, on the corrected (aggregate) comparison**: Earth
(~7e-10), Milky Way (~6e-7), galaxy cluster (~1.6e-5) are all still many
orders of magnitude below `LAM_SURFACE=1.0` — the tested effect remains far
stronger than any real aggregate gradient, and the real-world effect size
is likely undetectable at any of these scales. Consistent with this
project's recurring pattern (Vacuum hierarchy table, GPS/Pound-Rebka
numbers): real, derived, mechanistically confirmed, numerically
inaccessible at real aggregate scale, only demonstrable at artificially
strong test gradients.

## Continuum-limit formula for the leak — the actual "working continuous approximation"

Philip's framing for the whole thread: we're never simulating the true
discrete floor at any scale (a proton is `~10^58` Planck cells,
coarse-grained even in "particle-scale" ICs — `doc/dst.md`'s own
"Particle topology" section), so `lam`/`sigma` are already used as
continuous proxies everywhere in this codebase, not just macroscopically.
Given that, the goal isn't to keep discretely probing for an effect that's
provably unmeasurable at real scale (see above) — it's to derive a genuine
CONTINUUM formula for the leak directly from the discrete update rule,
usable analytically without simulation noise.

Redid the weight-consistent (all-square) plaquette derivation keeping
`sigma` non-uniform across the loop this time (previously assumed uniform
`sigma`, which is why the earlier `-2*dt*sigma*(u_C-u_A)` triangle result
was `sigma`-independent). Taylor-expanding both `sigma(r)` and `u(r)` to
first order around a point and summing the four leg contributions
(confirmed symbolically via sympy, exact cancellation of all `sigma_0`
and diagonal cross terms):

```
dCirc/dt = -dt · W_sq · d2² · (∇σ × ∇u)
```

**The leak is exactly the cross product of the throughput-gradient and the
potential-gradient** (`d2` = square-face hop distance). Consequences:
- **Vanishes when `∇σ ∥ ∇u`** — explains why a PURELY RADIAL background
  `lam(r)` still produced a nonzero measured deviation earlier in this
  report: the curl-full dipole's own `u` field (`u ~ z·g(r)`, poloidal
  mode) has a genuine non-radial/angular component, so `∇σ` (radial) and
  `∇u` (radial + angular) were never parallel. A purely radial `u` field
  combined with the same radial background would have given exactly zero,
  regardless of gradient strength.
- **Structurally identical to the Biermann battery mechanism** — the
  standard real-astrophysics process generating seed magnetic fields from
  scratch via misaligned density/temperature gradients (`∇n × ∇T`) in an
  initially unmagnetized plasma. DST's own discrete update rule reproduces
  the same cross-gradient structure natively in its own primitives (`σ`,
  `u`), not by construction/analogy — this fell out of the actual discrete
  algebra.

**Not yet done**: numerically validate this continuum formula's predicted
magnitude against the actual measured deviation (~0.05-0.09% at
`LAM_SURFACE=1.0`) as a direct consistency check — derived and confirmed
symbolically, but not yet cross-checked quantitatively against the discrete
simulation result it's meant to explain.

## Aside: "galactic voids have basically 0 throughput, photons can't cross" — not supported

Raised as a possible implication of the leak needing matter. Checked
against `doc/dst.md`'s own definition (`sigma(lambda=0)=L`, explicitly "no
zero" — vacuum is the UN-suppressed reference point, not a suppressed one;
matter suppresses relative to it) and against real astronomy (voids are
observed to be highly TRANSPARENT — background quasar/galaxy light and the
CMB routinely traverse voids, which is how void contents are mapped via
absorption spectroscopy; "no photons FROM a void" is fully explained by
"no emitting sources there," not by any transmission barrier). Philip
correctly flagged that "void = literal maximum throughput" is itself not
independently testable (we only ever compare clocks relatively, never
against an absolute external reference) — so the honest position is: the
RELATIVE ordering (matter suppresses relative to void) is well-tested: the
ABSOLUTE claim about where void sits is a model convention, not itself
falsifiable by anything run here. Either way, nothing found here supports
photons being blocked by voids.

## Files

`probe_energy_split.py`/`_v2.py` (first two attempts, mhd.py then
discrete_curl.py), `probe_energy_split_resolution_sweep.py` (resolution
check), `probe_energy_split_bc_check.py` (boundary/margin check),
`probe_energy_split_lam_background.py` (the decisive lam-gradient test).
