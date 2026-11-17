# Hertzian dipole attempt + E/B convention crisis

## Context

Same-day follow-up to the Poynting-ring `step_cross` fix
(`exp/2026/08/19/17_30_poynting_step_only_lam_sweep/`) and the traveling-wave
reproducer (`exp/2026/08/19/18_30_traveling_wave_reproducer/`). Tried to
build a genuine Hertzian (point oscillating electric) dipole as a simpler,
better-understood classical target than the toroidal ring, using the
Poynting-ring convention (`E:=-dphi/dt`, `B:=curl(phi)`).

## Pegged values (derived, not picked)

- `c = √10` (cell units) — dispersion-relation continuum limit, already
  validated in `test_maxwell.py`.
- `ε₀ = 1` — definitional, from `doc/dst.md`'s `Q ≡ Σ A_ij·∂u/∂r` (no
  separate scaling constant in the charge definition).
- `μ₀ = 1/(ε₀c²) = 0.1` — derived, corrects an earlier session's looser
  "μ₀≈1" note.

## Attempt 1: continuous cos(θ) source, radius sweep

Continuously-driven `u = A·cos(θ)·cos(ωt)` on a small sphere of source
cells (matching `equilibrium/substrate/double_slit.py`'s established
continuously-driven-source pattern, avoiding the harder problem of an
analytic "already outgoing" IC for a localized 3D source). Settle time
sized properly (arrival at `r_fit_max` + 1 full period, the same fix that
resolved the ring's far-zone saga).

Result: **equator/pole ratio pinned at ~0.28-0.30 (WRONG direction —
classical Hertzian dipole predicts equator≫pole) across the ENTIRE source
radius sweep, 1.5 to 6.0 cells (15 to 1211 source cells)**. Magnitude scales
up with source size; ratio does not move at all. Rules out under-resolution
of the continuous source as the cause.

## Attempt 2: lattice-native discrete two-cell dipole

Per Philip's steer ("cos and sin are fundamentally continuous, we might
need a discrete counterpart" / "a mesh so the ratios are projectable"):
built the EXACT discrete realization of the `l=1` harmonic instead of
sampling a continuous function — the Kelvin lattice's 6 square-face
directions ARE `±x,±y,±z` exactly. Drove just 2 cells: `+A·cos(ωt)` at the
`+z` square-neighbor, `-A·cos(ωt)` at the `-z` neighbor, nothing else.

Result: **ratio=0.313, same wrong direction, same suspiciously tiny
magnitude (`~1e-18`)** as the continuous version. Switching from
continuum-sampled to exactly lattice-native changed nothing. Rules out
source representation (continuous vs. discrete) as the cause.

(Minor bug found, doesn't affect the conclusion: face index 12 is actually
`-z` and 13 is `+z` in this convention, opposite of the variable names used
— a sign-flipped dipole has the identical `|S|` angular pattern, so this
doesn't change the result.)

## ε₀/μ₀ flip ruled out analytically

Any self-consistent `(ε₀,μ₀)` split satisfying `ε₀μ₀=1/c²=0.1` only
rescales `S=E×B` by an overall positive constant. This cannot change a
ratio of two percentiles of `|S|` (invariant under uniform positive
rescaling) and cannot explain a ~18-order-of-magnitude gap (any sensible
split is O(0.1)-O(10)). Confirmed this is not a units/convention question.

## The real finding: `E_full := -∇u - ∂φ/∂t` is identically zero under free `step()`

Verified directly (not just cited from `doc/dst_next.md`'s older note),
using `SubstrateLattice`, a Gaussian `u` bump, one plain step:

```
E_full (grad(u)+dphi/dt) rms: 8.6e-18   (machine precision)
E_dphi_only rms: 0.0854
grad(u) term rms: 0.0854                (exactly equal, opposite sign)
```

This is a genuine, exact algebraic identity of the update rule, general
(not specific to a plane wave) — `-∇u` and `-∂φ/∂t` are each real and
nonzero, and cancel exactly. Consequence: treating `u` as a scalar
potential and `φ` as a vector potential (the older note's convention)
makes the "complete" E trivially zero under any free dynamics — a dead
end for computing radiated flux, not evidence that nothing is happening.

## Conclusion: three incompatible E/B conventions, none validated for 3D `S`

1. `test_maxwell.py`: `u<->E` directly, called *"a scalar proxy for the
   electric field COMPONENT"* (singular) — built only to verify the 1D
   dispersion relation, never a 3D vector-field recipe.
2. Poynting-ring probes (`E:=-∂φ/∂t`, `B:=curl(φ)`): failed the local
   Poynting-theorem check earlier today AND both Hertzian-dipole attempts
   here — three independent failures of the same formula.
3. The older 4-potential note (`E:=-∇u-∂φ/∂t`): real, but identically zero
   under free dynamics — concludes `step_cross` is required to source E at
   all, the OPPOSITE of today's "drop step_cross" Poynting-ring fix.

**Open tension, not resolved**: did dropping `step_cross` for the
Poynting-ring equator/pole result actually fix something real, or was it
an artifact of convention #2 (which has since failed three separate
checks)? The traveling-wave reproducer's success doesn't settle this
either way — it validated that `(u,phi)` co-propagate correctly as a
*whole state* (a direct shift-match), never split into E/B or computed S.

## Next (see `now` file for the full plan)

Derive the actual local energy-flux identity from first principles
(starting from `H_i=0.5u_i²+0.5Σφ_ik²`, find what flux exactly satisfies
`dH_i/dt = -div(J)_i`, the same rigorous way `E_full≡0` was confirmed) —
not a 4th guessed convention. Rebuild everything on top of that once found.
