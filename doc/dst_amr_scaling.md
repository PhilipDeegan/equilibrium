# fp64 precision wall at deep AMR levels

Found 2026-08-04 while chasing the L8 first-look blowup (see `project_odd27_amr_junction_leak`
memory, session following Part 22). Distinct from the still-open Deep-telescope H-growth
question in `doc/dst_amr.md` — this is about arithmetic precision, not a formula/conservation
bug, and it explains one specific symptom (eta_h's total unresponsiveness at L8) without
explaining the base-update blowup itself.

## The numbers — CORRECTED after direct measurement (2026-08-04)

**Original back-of-envelope estimate (below, kept for the record) was wrong by ~3 orders of
magnitude — do not trust it. Measured values are in the next section.** Under Odd27
(`SPLIT_RATIO=3`), cell size at level 8 is `L8 = 1/3^8 ≈ 1.524e-4`:

```
L8^4                = 5.40e-16
machine epsilon      = 2.22e-16   ->  L8^4 is only 2.4x epsilon
```

The hyper-resistivity term (`eta_h`, `step_boundary_hyper_resistivity_ratio[_all]`,
`_leaf_laplacian`) is a biharmonic (`Laplacian(Laplacian(u))`) operator, needing two
applications of a `1/L²` divide — i.e. an effective `1/L⁴` dependence. Naive back-of-envelope
noise floor, for `u ~ O(1)`, assuming each Laplacian application injects a fresh
`~epsilon·|u|` absolute error before dividing:

```
lap1 (1st Laplacian) noise floor  ~ epsilon * |u| / L^2   ~ 9.6e-9
lap2 (biharmonic) noise floor     ~ lap1_noise / L^2      ~ 0.41     (vs u ~ O(1) signal)
```

**This estimate does not match direct measurement (below) — it overstates the noise floor by
roughly 3 orders of magnitude at L8.** The mistake: it assumed `lap1` itself carries a fresh,
generic `~epsilon·|u|` error at every cell, but `_leaf_laplacian` is confirmed (see next
section) to reproduce its target constant **exactly**, to double precision, at every level up
to L8 — there's no accumulating "noise" being injected at the first application to amplify at
the second. The actual mechanism producing `lap2`'s nonzero residual is subtler (bit-level
rounding in the underlying position→field evaluation, not a clean analytic noise-floor formula)
and was only found by measuring it directly rather than estimating it.

## Direct measurement: `test_biharmonic_precision_floor_by_level`

Added `tst/eq/core/data/lattice/adaptive_lattice.cpp`'s
`test_biharmonic_precision_floor_by_level` (part of the `test_lattice` C++ suite) to check this
properly instead of trusting the analytical estimate above — seeds the exact isotropic field
`u = x²+y²+z²` on a clean, junction-free uniform mesh at each level's own cell size (`build_bcc_
sphere` directly, no AMR/refinement — isolates pure fp64 arithmetic from the separate AMR-
junction question), on a 2-shell-deep interior target cell (itself and all 14 neighbours have
14 same-level neighbours each, so `lap1` is trustworthy everywhere `lap2`'s own sum reads from).

```
level    d1          lap1        lap2 (continuum value: exactly 0)
    0  1.000e+00     32.0        9.13e-12
    1  3.333e-01     32.0        2.15e-11
    2  1.111e-01     32.0       -5.41e-11
    3  3.704e-02     32.0       -7.07e-10
    4  1.235e-02     32.0       -1.12e-08
    5  4.115e-03     32.0       -5.31e-07
    6  1.372e-03     32.0       -1.40e-06
    7  4.572e-04     32.0       -2.23e-05
    8  1.524e-04     32.0        2.41e-04
```

Two findings, both now measured rather than estimated:

1. **`lap1` (the first Laplacian application) is exactly `32.0` — historically confirmed
   constant (project_odd27_amr_junction_leak Part 6; NOT the literal continuum Laplacian of 6.0,
   but a fixed BCC-coordination-shell normalisation constant times it) — at every level 0
   through 8, no drift. A genuine, clean correctness confirmation with no caveats.
2. **`lap2` (the biharmonic quantity `eta_h` actually multiplies) grows from ~1e-11 at L0 to
   ~2.4e-4 at L8** — real, and growing with depth, but the growth is NOT a clean power law
   (per-level ratios range ~3x to ~47x, sign flips unpredictably) — consistent with genuine
   bit-level floating-point rounding, not a smooth analytic noise-floor formula. **At L8 it
   reaches only `~2.4e-4`, not `~0.41`** — there remains ~12 orders of magnitude of headroom
   down to machine epsilon. This is nowhere near "indistinguishable from an O(1) field."

**Revised conclusion: the fp64 precision wall is real and growing with depth, but far less
severe than originally estimated, and does NOT by itself explain why `eta_h` measured as
completely inert (bit-identical at `0`, `1e-14`, `1e-13`) in the real L8 spheromak run.** A
deterministic `~2.4e-4`-scale quantity should still be something `eta_h` can act on in
principle. That inertness needs a different explanation — not yet found. Do not reuse the
original back-of-envelope estimate above for anything; it's kept only so the mistake and its
correction are visible.

The base (first-order) flux/diffusion update does **not** have this problem regardless: it
scales as `1/L` (not `1/L^4`), and the per-level subcycle timestep `dt_lev = dt_base /
ratio^level` is designed to compensate exactly (`CLAUDE.md`: "update rate ∝ 1/L"). Accumulated
rounding noise over all 6561 L8 substeps of one coarse step comes out to ~1.5e-12 — negligible.
So the L8 blowup actually observed (independent of `eta_h` and of `--gate`, localized entirely
to level-8 leaf cells within a single coarse step) remains **not explained by any fp64
precision question** — it points at a separate, still-open base-update/structural issue.

## Why naive per-level rescaling of stored values would NOT fix this

The instinct "normalize data per level, closer to zero, since we know the exact ratio between
levels" is the right direction, but a literal implementation — multiplying stored `u`/`phi` by
some per-level constant before storing — would not help. The precision loss happens at
**subtraction**: `u_i - u_j`, two independently-rounded O(1) numbers, loses absolute precision
of ~`epsilon·|u|` regardless of `L`. Rescaling both operands by the same constant is just a
change of units — it doesn't add mantissa bits back, so the *relative* precision of the
subtraction is unchanged. Dividing by `L^4` afterward doesn't destroy anything further (IEEE
division is accurate to ~1 ulp) — by the time you divide, the damage from the subtraction is
already done.

## What would actually help: delta-relative-to-parent representation

Store each leaf's field value as a **delta relative to its parent's (interpolated) value**,
reconstructing the absolute value on demand by walking/summing the ancestor chain, rather than
always storing/reading the absolute value directly. For a smoothly-varying field, the delta at
a fine level is genuinely small (that's what the level *represents* — the physical detail added
at that scale) — critically, it is then **evolved and stored directly**, not reconstructed via
subtraction of two O(1) numbers every time it's read. That gives the delta its own full 52-bit
*relative* precision budget, decoupled from the absolute value's fixed ~1e-16 *absolute* noise
floor. This is the same idea underlying wavelet/multiresolution PDE representations, used
precisely to represent fields with large dynamic range across scales in fixed precision.

**Scope, if pursued**: this is a genuine architecture change, not a patch. Every read of a
leaf's `u`/`phi` becomes "walk the ancestor chain and sum" instead of a direct field access;
every physics update (`_step_pass1_lev`, `_step_pass2_lev`, `step_cross`, `step_lam`, ...) has to
act on the delta representation; and it has to be threaded through the existing Odd27
coarse-proxy/`face_multiplicity`/`ancestor_group_multiplicity` machinery (`doc/dst_amr.md`),
which is already subtle.

**Motivation weakened by direct measurement (2026-08-04)**: the biharmonic term's actual
precision floor at L8 is `~2.4e-4` (see above), not the originally-estimated `~0.41` — there's
still ~12 orders of magnitude of headroom to machine epsilon. `eta_h`'s complete inertness in
the real L8 run is therefore NOT well-explained by fp64 precision after all, which means this
large architecture change is not clearly justified by what we currently know — its original
premise (eta_h's biharmonic term is precision-starved at L8) didn't hold up under measurement.
**Do not start this without a new, measured reason to believe it's needed.** The base-update L8
blowup (still open, `project_odd27_amr_junction_leak` memory) remains unrelated to this thread
and is the more urgent open question either way.
