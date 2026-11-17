# DST AMR — Design Notes

## Impedance mismatch at level boundaries

In the DST wave equation, wave speed is `c = σ·L` — larger cells propagate faster.
This is intentional: it models local time dilation (small cells near mass update slower).

A boundary between level-0 (L=1) and level-4 (L=1/16) cells creates a 16× wave speed
ratio. The reflection coefficient at such a boundary is ~88–99%, creating a near-resonant
cavity. Any persistent energy source (fixed_u, frozen phi) inside the refined region will
pump energy into this cavity and cause exponential growth.

**This is not a bug in the flux correction.** The correction mechanism below correctly
conserves flux at multi-level boundaries. The mismatch is a consequence of DST physics.

---

## The actual cross-level mechanism: proxy/mirror pair correction

The implemented approach (`inc/eq/core/data/lattice/adaptive_lattice.hpp`) does **not**
average fine cells up to a virtual parent before computing the boundary flux. Instead,
every leaf keeps computing its own flux at its own resolution against whatever it is
actually adjacent to, and a **post-pass correction** balances the books afterward.

### Why direct interaction needs correcting

At a coarse-fine junction, both sides read the same physical interface but at different
distances:
- fine leaf descendants use `d = 2/(L_fine + L_coarse)`
- the coarse neighbour Q uses `d = 2/(L_Q + L_P)`

Because `L_fine ≠ L_Q`, the `dphi` computed on each side of the junction does not cancel —
net `u` is silently created or destroyed at every level boundary, every step.

### The fix: `_collect_leaves_facing` + `_step_pass2_amr_correct`

`rebuild_amr_corrections()` (called from `rebuild_leaf_indices`) walks every non-leaf
(proxy) cell in the tree and builds a flat list of correction pairs, once, whenever the
tree topology changes (split/coarsen) — not every step:

- **Coarse-fine pairs** — proxy `P` faces a leaf `Q` on face `k`. `_collect_leaves_facing`
  recurses through `P`'s subtree (through any nested sub-proxies) and collects every leaf
  descendant whose `nb[k] == Q`. A **parent-skip guard** (`par.nb[k] == Q_g` check) ensures
  that if an ancestor proxy already covers this junction, the deeper proxy does not
  duplicate the correction — this is what makes arbitrary-depth level jumps (not just 2:1)
  work correctly.
- **Proxy-proxy pairs** — two adjacent non-leaf cells `P` and `P2`. Leaf descendants of `P`
  facing `P2` and leaf descendants of `P2` facing `P` are collected on both sides
  (deduplicated by `P_g < P2_g`, with parent-skip guards on both sides).

Each pass (`_step_pass2_amr_correct`, run between pass 2 and pass 3 every step):
1. Sums `temp_phi` already computed independently on each side of the junction
   (`sum_A` for the fine/first side, `sum_B` for the coarse or second-proxy side).
2. `error = sum_A + sum_B` — the amount of flux that failed to cancel.
3. Redistributes `-error` evenly across the fine (coarse-fine case) or both (proxy-proxy
   case) sides, so the junction nets to zero net flux creation.

`step_subcycle()` cannot use this pass directly — its two sides of a junction are computed
in different (level, substep) iterations, not the same pass. It instead accumulates `sum_A`
/`sum_B` into per-pair registers (`_accumulate_amr_registers`) across every substep of the
outer step, then applies one reflux correction directly to `phi` after the finest level has
finished (`_apply_amr_reflux`) — the standard AMR "reflux at the end of the coarse timestep"
pattern.

### What this buys over a virtual-parent averaging scheme

- No extra state: no virtual parent `u`/`phi` values need to be computed or stored — the
  correction only touches `temp_phi`, which is already being computed for every step.
- Fine cells' own dynamics are completely unaffected; only the boundary-facing flux is
  corrected, after the fact.
- Arbitrary level jumps are handled by the same code path, not a special case — the
  recursive collection in `_collect_leaves_facing` walks through however many sub-proxy
  levels sit between the two sides.

---

## Why not 2:1 balance enforcement?

2:1 balance (no adjacent cells more than 1 level apart) would reduce the maximum impedance
mismatch at any single boundary to 2× (~33% reflection). It is the standard approach in
wave-equation AMR codes.

However, it is an artificial constraint that wastes cells in sharp-gradient regions — a
planet surface or knot boundary should be able to jump from L0 to L4 in one step. In smooth
regions (slow lam gradients, open space between planets), the AMR criterion naturally
produces gradual level transitions of ~1 level anyway.

The proxy/mirror correction above handles arbitrary level jumps correctly without any
level-smoothing constraints. Sharp gradients can be as aggressively refined as the physics
demands.

---

## Initialisation with pre-splitting

The impedance mismatch instability only occurs when a **persistent energy source** exists
inside a refined region surrounded by a reflective coarse boundary:
- `fixed_u` at origin — continuous energy pump → resonant cavity → exponential growth
- `frozen phi` on ring cells — antisymmetric drive → growing u difference → cascade

**Stable AMR initialisation:**
- Initialise from the analytic spheromak IC with `amplitude > 0` (phi set from B·n̂)
- `∇·B = 0` means the initial phi drives negligible net u change — IC is self-consistent
- No `fixed_u`, no frozen phi → no persistent source → no resonant cavity
- Pre-split the knot region to target level before init; level jump at knot boundary creates
  reflections from residual energy but not growth

The warmup requirement with `amplitude=0` (phi=0 IC) was an artefact of the inconsistent
initial condition, not a physical necessity.

---

## Deep-telescope H-growth: mechanism, elimination history, current mitigation

Running the real production spheromak (`refined_neutron_decay.py`, fixed telescope)
at increasing `--max-level` shows H (`Σu² + 0.5·Σphi²`) growing faster than linearly
with depth: bounded/mild at L2 (~1.5x over 500 steps) and L4 (~2.5–2.7x, confirmed
production-safe), substantial at L6 (~5–19x, oscillating not runaway), severe at L8
(~13–131x). In every case the growth is a **bounded, persistently-elevated
oscillation**, not a divergence to infinity — but it is real, gets worse with depth,
and the root cause inside the flux formula is still open.

**Update 2026-08-07..09 — resolution, not depth.** The growth scales as H/H0 ∝
cells_across^-1.85 (17.9 / 35.7 / 71.4 cells across the spheromak: 134x / 37x / 10.5x).
A hop-graded telescope at L5 with cells_across=220 stays bounded (H/H0 1.0036 → 1.0299 over
5 outer steps, decelerating, no NaN; checkpoint `/bk/l5_core_diameter9_cells_across_220.h5`;
L5 eta_h knee ≈ 3e-9, as at L4). Depth was never the requirement; resolution relative to the
object was. The dense hop telescope costs ~27x per level (L5 ≈ 23M leaves / 15 GB, L6 ≈ 400 GB),
so large domains need multiscale level 0, not more depth. The fractal L4/L5 texture along
ancestor lineages was never re-checked after the pass1 fix. AMR is deferred (continuous model
level 0 only). The history below is kept for reference.

### Ruled out, with strong (exact-invariant or clean A/B) confidence

- Same-level flux's own formula at an Odd27 coarse-fine boundary — proven **exactly**
  conservative via static (zero-stepping) unit tests (pairwise antisymmetry, zero
  group sum, live-mesh-vs-compile-time-table cross-check, all to fp precision). This
  rules out same-level flux's formula itself as the cause.
- The cross-level reflux's face-index-matching incompleteness (`_collect_leaves_facing`
  only matching a single shared face index) — a real, now-fixed bug
  (`_collect_leaves_facing_multi` + `amr_js_face`, see the proxy/mirror section above),
  but shown to be a ~1e-10-relative (linear, `Σdphi`) defect — far too small to explain
  a 10–100x quadratic (`H`) blowup. Linear-conservation defects and quadratic-norm
  growth are different-magnitude invariants; fixing one does not bound on the other.
- `_step_pass2_siblings` (same-parent square-face coupling): zero effect when disabled.
- AMR reflux itself: negligible effect, slightly *protective* when disabled.
- CFL/temporal resolution: growth is identical to 6 sig figs whether the subcycle
  ratio is 3x or 9x finer — fully converged in time, not a stability-limit issue.
- Shell width/domain extent: growth unchanged to 7–8 sig figs across a >20x cell-count
  range — not about telescope buffer size.
- The ancestor-substitution `u_src` mechanism itself, sigma Jensen-bias
  (`sigma(mean(lam))` vs `mean(sigma(lam))`), proxy-refresh timing
  (per-substep vs per-outer-step), and continuous accumulate-damping/cross-level-flux
  scaling from 0 to 1: all negligible.
- Extending `ancestor_group_multiplicity` to same-level proxy sharing: a real but rare
  edge case (~4k of ~1.5M relevant connections), zero measurable effect, reverted.
- Same-level mirror-reciprocity (`A.nb[k]=Q ⟹ Q.nb[mirror(k)]=A`): verified 100%.
- The recursive/interleaved Berger–Oliger subcycle restructuring
  (`_step_subcycle_recurse`/`_step_subcycle_one`, replacing the old flat
  coarse-first-then-fine loop) is real and correctly implemented, but its own
  effect on H-growth is tiny (~1.7e-8 relative) — kept as a genuine improvement,
  not the dominant fix.

### Localized but not root-caused

Disabling same-level flux **only** at cells that also have ≥1 cross-level face
(interior cells' same-level dynamics untouched) gives a clean, reproducible ~13x
single-step-growth reduction at both L4 and L6 — this is the real, dominant,
localized mechanism. But per-face tracing shows the formula itself is internally
consistent at a traced boundary cell (weight ratios dimensionally sensible,
reciprocity exact) — nothing obviously wrong has been found in the formula for that
specific cell. The exact within-formula reason a boundary cell's same-level flux
behaves differently from an interior cell's remains **open**. Leading unconfirmed
candidates: an interaction between `face_multiplicity` and a boundary cell's reduced
total same-level neighbour count, or a link between the oscillation's amplitude and
the telescope's own geometric structure.

### Current mitigation: hyper-resistivity boundary damping (IMEX, ratio-parameterized)

A k⁴-selective damping term (`step_boundary_hyper_resistivity_ratio[_all]`,
`adaptive_lattice.hpp`) is applied only at boundary leaves: computes a discrete
Laplacian twice (`_leaf_laplacian`, correctly scaled `1/L²` per application — an
earlier `1/L` version was a real dimensional bug, since a curvature/second-derivative
estimator needs `1/length²` on any point set, not just a square grid), then damps `u`
by this biharmonic quantity. Solved diagonal-implicitly (IMEX: only the operator's own
diagonal/self term is implicit, everything else explicit) — a fully explicit version
has a sharp, narrow stability cliff typical of hyperviscosity terms; IMEX is
unconditionally stable with respect to the diagonal part.

The coefficient is parameterized as a **dimensionless ratio**
`s = dt · eta_h_level · D_i` (`D_i` = the IMEX diagonal coefficient, ~`1/L⁴`) rather
than a raw per-level `eta_h`, because raw sweet-spot values span ~4 orders of
magnitude across levels 6→8 while their corresponding `s` values span less than 1 —
`s` is what actually governs the update's own saturation behaviour
(`u_new → -offdiag/D` as `s → ∞`), so it's the right unit to reason and calibrate in.

**Confirmed**: a single flat `s=2`, applied uniformly across every level, is durably
stable at L6 (best long-horizon L6 result found) — but **diverges at L8** past
~step 24 even though it looks fine in a short (~10-step) sweep. At L8 the only
durable-to-completion fix found is an **isolated single-level array** (only level 8's
own coefficient nonzero, every shallower level's coefficient exactly 0) — not yet
established whether this is because L8 genuinely needs full per-level isolation, or
whether a ratio restricted to just the 1–2 deepest levels (rather than all 0–8) would
also work; not yet tried.

**Calibration methodology** (reusable, confirmed accurate across 4+ levels): for a
new level, run a 150-step bisection to find the instability cliff and apparent-best
coefficient, then ALWAYS re-run the 2–3 smallest comparable 150-step candidates to
500 steps — a near-cliff "150-step winner" reliably underperforms or outright diverges
by 500 steps (confirmed at L3/L4), while the smallest of a close cluster tends to win
the long-horizon check. Seed the search range for the next level by dividing the
current level's cliff and best-known coefficient by `ratio⁴ = 81` (from `D_i ~ 1/L⁴`)
— this extrapolation has landed within a factor of ~2 of the real cliff at every level
tested so far.

**Important caveat**: a hex(weight 3)/sq(weight 1) face-weight-mixing bug in the
underlying correction formulas (`ancestor_group_multiplicity`, `face_multiplicity`,
the plain-`step()` reflux) was found and fixed after most of the calibration above —
see `project_odd27_amr_junction_leak` memory, Part 12 onward. **Every specific eta_h
value or ratio quoted above was calibrated against the pre-fix formula and needs
re-earning against the corrected code before being trusted for a production run.**
The qualitative picture (bounded oscillation, worse with depth, IMEX-ratio mitigation
shape, L6-vs-L8 divergence of the uniform-ratio approach) is expected to still hold —
only the exact numbers are suspect.

**See also `doc/dst_amr_scaling.md`**: at L8, the biharmonic (`eta_h`) term's own `1/L⁴`
divide sits right at the fp64 precision floor — likely explains why `eta_h` became totally
unresponsive at L8 (2026-08-04), but is a separate issue from the base-update L8 blowup, which
remains open.

## Visualisation note

When rendering the lam terrain from an AMR checkpoint with `--min-level N`:
- Level-0 cells span large volumes; griddata interpolation smears their lam across the domain
  and buries fine-scale mass features
- Filtering to level ≥ 2 shows only refined cells, which are concentrated around mass
  concentrations — the AMR refinement pattern is itself a map of gravitational structure
- The boundary ring artefact in the u field (elevated u at the outer sphere shell) is from
  level-0 boundary cells and disappears at `--min-level 2`

---

## Known open bug: `restore()` NaN sentinel

Carried over from the retired `dst_amr_review.md` (2026-06-10 review; its other findings were
either already fixed or superseded — see git history if the detail is ever needed). Re-verified
against current code 2026-08-13, this one is still open:

`restore()` sets `criterion_min`/`criterion_max` to `0.0`, not NaN (`cluster.hpp:697-698`).
Every other code path uses NaN as the "not yet evaluated" sentinel for these two fields
(matching what pass 3 assigns to freshly split children, and what `adapt()` already guards
against, `adaptive_amr.hpp:114,134`: `!std::isnan(cmax) && ...`). `restore()` is the one
exception. Since `split_lo > 0` in practice, a `0.0` sentinel reads as "below threshold" for
every leaf — the first `adapt()` call after loading ANY checkpoint tries to coarsen the entire
mesh before a single step has run. **High priority**: any checkpoint-based workflow (`save()` +
`restore()`/`AdaptiveLattice.load()`) hits this. Fix is a one-line change to use
`std::numeric_limits<double>::quiet_NaN()` instead of `0.0` in both `assign()` calls.

## Memory layout: CellSlab byte-shrinking (toward R=200)

`build_bcc_sphere`'s cell count is `≈5.44·R_cells³`, independent of refinement depth
(R=100 → ~5.44M cells, matching the "R=100/L8 → 5.58M leaves" figure — the refined
core itself is only ~2–3% of the total). Running the real production telescope at
R=200 instead of R=100 needs ~43.5M cells, an 8x jump that current RAM doesn't support.
Per-cell byte-shrinking alone (even the aggressive case below) buys only ~13% more R —
nowhere near the 2x needed; the actual 8x lever is a future multi-scale-L0 idea (bigger
cells far from the object of interest, see `project_multiscale_lattice` memory), not
started. The notes below are about squeezing safe, zero-risk bytes out of the
*existing* uniform-resolution `CellSlab` layout in the meantime.

**Done**: removed the per-cell `level` array from `CellSlab`
(`inc/eq/core/data/lattice/cluster.hpp`) — every slab already belongs to exactly one
level (`slab_level_[s]`), so per-cell level storage was 100% redundant.
`AdaptiveCellRef::level` is now computed at `operator[]` construction time
(`slab_level(slab_of(g))`) instead of stored. The redundant "level" HDF5 dataset was
also dropped from `cluster_io.hpp` (`slab_levels`, saved separately, is the sole
source of truth). Net: −1 byte/cell — small on its own, but de-risks the "value
derived from which slab you're in, not stored per-cell" pattern the next item needs.

**Designed, NOT implemented — read this in full before attempting**: dropping the
`parent` array for level-0 slabs (100% of L0 cells are root, `parent` always `-1` by
construction — same redundancy class as `level`). This is a bigger change than
`level`'s removal because `parent` genuinely IS written at runtime: `free_cell`/
`free_children` (`cluster.hpp`) write a `-2` "unallocated" sentinel into `parent[i]`
when a cell is freed, and that sentinel is read back by **six** separate raw
`slab.parent[i] == -2` liveness checks — several in `adaptive_lattice.hpp`, several in
`cluster.hpp`'s `restore()` (re-grep before trusting any specific line numbers; the
file will have moved). Every one of these would crash or silently misbehave if
`parent` were `nullptr` for an L0 slab, unless updated first.

Planned approach: (1) `CellSlab` takes a `has_parent` bool at construction (skip
allocating `parent` when false), set via `_new_slab(lv)` passing `lv != 0`;
(2) `AdaptiveCellRef::parent` changes from `std::int32_t&` to a small `MaybeParentRef`
wrapper (`operator std::int32_t() { return ptr ? *ptr : -1; }`, assignment a no-op when
unbacked) — safe because every write site that could ever target an L0 cell's
`.parent` already only ever writes exactly `-1` (verified by a full repo grep of
`.parent = `, direct and via migrate code's `remap_if`); (3) one centralized
`CellSlab::is_free(i)` helper (`parent ? parent[i]==-2 : first_child[i]==-2`,
repurposing `first_child`'s spare `-2` as the L0-specific marker, since `first_child`
is still stored for L0 slabs) replaces all six raw checks, rather than duplicating the
branch six times; (4) `restore()` keeps `has_parent=true` unconditionally regardless of
level — checkpoint loading is rare, not worth the L0-marker complexity there; only
fresh, live-built slabs (`_new_slab`, i.e. `build_bcc_sphere`/`split()`) get the leaner
layout; (5) `cluster_io.hpp`'s `save_cluster` needs a null-guard on
`memcpy(par.data()+off, slab.parent, ...)`, synthesizing `-1`/`-2` for L0 rows instead
of reading a null pointer.

The six-site liveness-check surface was only found by grepping *after* starting a
naive "just do `parent` like `level`" attempt — doing this without the full picture
first would very likely ship a crash or a silent AMR-rebuild correctness bug, in code
that already has a long history of exactly that kind of subtle defect (see
`project_odd27_amr_junction_leak`). Find every raw array touch first if resuming this.

`u`/`phi` as `float32` (−60 bytes, ~26%, the single biggest remaining win) is not
attempted — it's the actively-integrated dynamical field, so it carries real precision
risk to the H-growth investigation above and would need re-validating against existing
regression numbers before trusting.

**Tally so far**: `level` removal done (−1 byte/cell). `parent`-for-L0 designed but not
implemented (−4 bytes for ~100% of L0 cells). `u`/`phi`→float32 not attempted (−60
bytes, risky). None of this reaches the 8x needed for R=200 alone — that needs the
separate multi-scale-L0 idea.

---

## Pass1/pass2 coefficient parity (resolved 2026-08-15)

The wave step is a symplectic-Euler pair: pass1 (`_step_pass1`/`_step_pass1_lev`)
integrates `u` from the persistent `phi[k]` state; pass2 (`_step_pass2`/
`_step_pass2_lev`) integrates `phi[k]` from `u`. For this pairing to conserve
`H = Σu² + 0.5·Σphi²` (or Σu, at a real junction), the per-face coefficient pass1
multiplies against `phi[k]` must be EXACTLY the coefficient pass2 used to construct
that same `phi[k]` value — not just approximately equal, not "close enough at
same-level faces". Historically pass1 and pass2 were two independently hand-written
formulas that silently drifted apart (missing `1/L`, wrong sigma averaging, wrong
L-blend at junctions, missing ancestor-substitution/multiplicity entirely) — each
drift showed up as a real, measurable H or Σu leak, worst at genuine coarse-fine
junctions (`TestFluxConservation`'s `coarse_fine`/`common_denominator` cases exist
specifically to catch this).

Fixed by extracting three shared static helpers (`adaptive_lattice.hpp`) that BOTH
passes call, so they structurally cannot drift apart again:
- `_resolve_face_neighbour` — the stale-proxy 1:1 resolution (hex faces only, one-level
  gap) that the flat `_step_pass2` performs at runtime.
- `_ancestor_L_src` — the ancestor-substitution walk (when `nb.L > cell.L`, walk this
  cell's own parent chain to the ancestor at `nb`'s level) that determines `L_src` for
  the face-weight blend, gated by `disable_ancestor_u_src` only for the `_lev` variant
  (matching `_step_pass2_lev`'s own gating — the flat `_step_pass2` gates it
  unconditionally, a pre-existing asymmetry between the two variants, left as-is).
- `_face_flux_coefficient` — `sig_face * w` (face_weight folded into `w`), with the
  `face_multiplicity`/`ancestor_group_multiplicity` divisors and `cross_level_flux_scale`
  applied identically to how pass2 builds `dphi`.

**Critical asymmetry, easy to get wrong**: the multiplicity divisor belongs ONLY to
`phi[k]`'s construction (pass2) — pass2 divides `dphi` by `face_multiplicity`/
`ancestor_group_multiplicity` each step when accumulating into the persistent `phi[k]`
state, so `phi[k]` is already each face's correctly-scaled private share of a fan-in
junction's total flux. Pass1 reading `phi[k]` back and applying the SAME divisor AGAIN
double-counts the correction — confirmed empirically (2026-08-15): doing so made the
`coarse_fine`/`common_denominator` drift WORSE (grew from ~0.6% to ~1.2%, sign-flipped)
rather than better. `_face_flux_coefficient` therefore takes an `apply_multiplicity`
bool; pass1's call sites always pass `false`, pass2's pass `nbg_r == nbg` (matching its
pre-existing gate). Everything else in the coefficient (L_src, sigma blend,
`cross_level_flux_scale`) IS shared identically between the two passes.

---

## step_cross's `cell.L` factor is convention-dependent, not universal (resolved 2026-08-15)

`_step_cross_local`'s per-face `delta` includes a `cell.L` multiplier
(`step_cross_level_scale_power`, default exponent `1.0`). Its original, still-valid
purpose: convert a FLAT, shared L0-calibrated `dt` — the same numeric value reused
unchanged across every level of one mixed AMR mesh (`_compute_dt_l0(cl)`, default
`D1=1.0`, the convention every current production script uses) — into an effective
per-cell dt (`dt_L0 * cell.L ≈` the cell's own CFL-stable dt).

That derivation assumes `dt` is ALWAYS the flat L0-reference value. A caller that
instead passes a `dt` already matched to its own uniform mesh's cell size
(`_compute_dt_l0(cl, D1=cell_size)` — `pure_l1size_control`'s own convention, also used
by `tst/substrate/adaptive/test_step_cross_per_level.py`) has `dt` ALREADY equal to the
cell's own locally-correct value; multiplying by `cell.L` again double-counts the same
D1 scaling `dt` already carries. Confirmed: with the default `power=1.0` and a
locally-matched `dt`, the measured single-call fractional-H-loss ratio between two
resolutions (D1=1 vs D1=1/3) was `0.111 ≈ (1/3)²` — literally the square of the
resolution ratio, i.e. two compounding D1 factors instead of one.

**The right fix is convention-dependent, not a universal constant.** For the
flat-shared-dt/AMR convention, `power=1.0` (default) is correct and unchanged. For the
per-mesh-matched-dt convention, `power=0.0` (no extra `cell.L` factor — `dt` already
is the target quantity) is correct. There is no single hardcoded exponent that serves
both; `AdaptiveLattice.step_cross_level_scale_power` is a settable escape hatch for
callers using the second convention (production scripts using
`_compute_dt_l0(cl, D1=cell_size)` on a genuinely uniform, non-AMR mesh should set it
to `0.0` before calling `step_cross`/`step_cross_all` — not yet done for
`pure_l1size_control`'s own step script, exp/ scripts are one-off/lower priority).

**Separately, the comparison metric matters as much as the coefficient.** A resistive
process's fractional loss over ONE call legitimately scales with that call's own `dt`
— comparing one call at `dt0` against one call at `dt1=dt0/3` directly is not the
right invariant; the level-independent quantity is loss accumulated over EQUAL real
elapsed time (1 step at `dt0` == 3 steps at `dt1`, since `dt0 = 3·dt1`). Fixing only
the coefficient (`power=0.0`) with the wrong (1-vs-1) comparison still gives `0.333`,
not 1. Fixing only the comparison (time-matched) with the wrong (`power=1.0`)
coefficient gives the same `0.333` (same number, different route — both fixes remove
exactly one of the two compounding D1 factors). Both together give `0.999`.

**Methodology note**: an earlier attempt found `power=-1.0` gives `ratio≈0.999` under
the (wrong) 1-vs-1 comparison — this was pure curve-fitting a single free parameter to
a single data point (exactly one ratio to match), not a real relationship; re-tested
against the correct time-matched comparison, `power=-1.0` overshoots to `ratio=2.997`,
exposing it as noise. Don't trust a single-point coefficient fit without a second,
independent check (a third resolution, or an orthogonal metric like the time-matched
comparison here) before treating it as a real fix.
