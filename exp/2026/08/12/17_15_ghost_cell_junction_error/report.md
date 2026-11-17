# Stale inter-parent proxy wiring — found, fixed, and validated

## Origin

Follow-on from `exp/2026/08/12/16_10_pure_l1_cellsize_control/`: the `l1_wider_margin_active`
run (`exp/2026/08/11/21_27_everything_active/`) showed H/H0 crossing above 1 and accelerating
(1.022056 at step 130) while `H_by_level` showed the L0 shell's own contribution was still at
fp-noise (~1e-50) — the growth was happening entirely inside the L1 core, nowhere near the real
L0/L1 boundary at r=46.667.

## Diagnosis

`AdaptiveCellCluster::leaf_is_boundary` (any face pointing at a different-`L` neighbour) reads
**~96% true throughout the entire L1 core, uniformly at every radius including r≈0** (the
spheromak's own centre) — nothing to do with the real domain boundary.

Traced to `wire_split_children()`'s Odd27 branch (`bcc_geometry.hpp:428-448`):
```cpp
cluster[cg].nb[k] = (t < ODD27_EXTERNAL_BASE) ? (fc + t) : par.nb[t - ODD27_EXTERNAL_BASE];
```
A child whose true neighbour is a *sibling* (same parent) wires directly to it. A child whose
true neighbour would be a child of the *adjacent* parent's own cube instead falls back to
`par.nb[...]` — the parent's own neighbour pointer, i.e. the **adjacent parent itself**. This is
a fixed, compile-time-table lookup resolved once at split time and never revisited. If that
adjacent parent is later split too (exactly what happens throughout
`refine_region_to_convergence`'s uniform L1 core — every ~3 L1-cells), the child keeps pointing
at the now-non-leaf adjacent parent instead of its own genuine same-level leaf sibling one hop
further in.

Checked whether the *runtime* flux formula already resolves this correctly regardless of the
stale pointer (in which case it would be a pure memory/bookkeeping optimisation, not a
correctness bug, matching an earlier "this came up before" recollection): it does not, uniformly.
`_step_pass2_lev`'s ancestor-substitution branch fires for *any* non-leaf neighbour, replacing
`u_i` with a "shared, multi-descendant-averaged ancestor u" unconditionally — it never checks
whether a single genuine same-level partner exists first. `_step_cross_mirror_proxy`'s own
handling of the identical situation is inconsistent by face type: its **sq-face** branch already
resolves the true partner via position descent (`_descend_to`) and mirrors 1:1; its **hex-face**
branch instead falls back to distributing across the whole group. So every one of these stale
inter-parent seams — pervasive throughout any uniformly, independently-refined region — was
getting Jensen-biased averaging where a plain same-level face update belonged, regardless of hex
or square face type.

## Failing test (added, `tst/eq/core/data/lattice/adaptive_lattice.cpp`)

`test_stale_parent_proxy_after_neighbour_refine`: builds two adjacent interior leaves, splits
BOTH to level 1, and checks that a child of one — whose external face pointed at the other
pre-split — resolves to a genuine same-level leaf, not the other's now-non-leaf parent.
Confirmed failing pre-fix: `19/19 child faces ... still routed through the STALE non-leaf
parent proxy`.

## Fix (`inc/eq/core/data/lattice/adaptive_lattice.hpp`)

Chose a persistent rewrite at the source over patching each consumer (flux formula,
`leaf_is_boundary`, `step_cross`, ...) individually — "regardless of square or hex, don't do
double duty" when a genuine same-level neighbour exists.

`resolve_stale_external_proxies(cluster)`: for every leaf's non-leaf-neighbour face, uses the
same `_descend_to` position-descent primitive `_step_cross_mirror_proxy` already had for this
exact situation, to check whether the neighbour has since been split down to a genuine LEAF at
this leaf's own level. If so, rewires both directions directly to it — no proxy, no averaging.
If descent instead stops at a coarser leaf (genuine partial refinement) or another non-leaf
(neighbour refined deeper), that's a real multiplicity/coarse-fine junction and is left
untouched. Called from `rebuild_leaf_indices()` — already invoked after every split/coarsen pass
and by `AdaptiveLattice.load()` — right before the multiplicity tables rebuild, so every
downstream consumer (flux, `leaf_is_boundary`, checkpoints loaded from disk) sees corrected
topology automatically, including checkpoints saved under the old buggy code.

Also added a defensive same-resolution check directly inside `_step_pass2_lev`/`_step_pass2`
(the main flux passes) as a second line of defense, with a matching bypass of the
`face_multiplicity`/`ancestor_group_multiplicity` divisors when resolution fires (those tables
are keyed to the stored, possibly-stale topology and would otherwise wrongly attenuate an
already-corrected flux). New diagnostic toggle `disable_resolve_stale_proxy_neighbour` (default
`false`) added for symmetry with the file's existing toggle convention.

Post-fix: `test_stale_parent_proxy_after_neighbour_refine` passes (`0/19` stale). Full suite
(`adaptive_lattice.cpp` + `uniform_vs_adaptive.cpp`) passes clean, no regressions on any asserted
check (verified with assertions active, `mkn build -Oqp test_lattice -x mpi -gW 9 run test`, not
an NDEBUG build).

## Side-effect investigated: `test_cross_level_aggregate_isotropy_full_group_vs_unsplit_2level_graded`

This diagnostic-only (no `check()`) test's printed `rel_diff` moved from ~0.18 to ~5.8-6.6.
Traced and explained, not a red flag: `_build_graded_l0_l1_l2_telescope` deliberately pre-splits
the shell around `target0` to L1 *before* splitting `target0` itself, specifically so `target0`'s
external faces would route to *already-split* neighbours — i.e. it's a textbook construction of
the exact stale-proxy pattern. Confirming evidence the OLD number was itself the bug, not a
baseline: it was suspiciously close to the *plain, ungraded, direct-L0-to-L2-jump* test's own
number (0.179 vs 0.193) — the graded shell was being silently collapsed into behaving like a raw
jump, defeating the entire reason that test exists (to check whether grading through a real L1
shell, instead of jumping straight to L0, changes the isotropy error). Every other test in the
file — including the sibling `_2level_ungraded` test and the plain `_full_group_vs_unsplit` test
— is bit-identical before/after, exactly as expected since neither has independently-refined
same-level structure for the fix to touch.

## Validation against the real regression

Reused the existing `l1_wider_margin_active_ic220.h5` IC directly — no rebuild needed, since
`AdaptiveLattice.load()` already calls `rebuild_leaf_indices()`, so the fix self-heals any stale
wiring baked into an old checkpoint on load, including checkpoints saved under the old buggy
code. Re-stepped from t=0, 130 steps, identical `cross_scale=1e-3`, into fresh output paths
(`/bk/l1_wider_margin_active_fixed_stepped/`, `plots_fixed_rerun/`) so the original buggy run's
checkpoint stayed available for direct comparison.

| config | H/H0 @ step 130 (real elapsed time = 130×dt_L0) |
|---|---|
| `uniform_r180_active` (no junction, cell size 1) | 0.999539 |
| `pure_l1size_control` (no junction, cell size 1/3) | 0.912617 |
| `l1_wider_margin_active` — **before fix** | **1.022056** (crossed 1.0, still accelerating) |
| `l1_wider_margin_active` — **after fix** | **0.912609** |

The fixed run now matches `pure_l1size_control`'s time-matched value to 5 decimal places —
expected, since `H_by_level` confirms the L0 shell still carries ~zero field at step 130 (no real
boundary crossing has happened yet), so with the spurious inter-parent-seam growth gone, the
core's decay should be (and now is) indistinguishable from a boundary-free control at the same
cell size. Not just "less wrong" — the fix lands on the physically correct answer.

## Superseded artefacts (cleaned up)

- `/bk/l1_wider_margin_ic220.h5` + `/bk/l1_wider_margin_stepped/` — pre-"everything active"
  vintage from `15_05_l1_wider_margin`, superseded by the `_active` variants.
- `/bk/uniform_l0_r180_ic220.h5` + `/bk/uniform_l0_r180_stepped/` — same vintage, superseded by
  `uniform_r180_active`.
- `/bk/l1_wider_margin_active_stepped/` (the *old buggy-code* stepped checkpoints, step_030 and
  step_130 showing the 0.994→1.022 trajectory) — fully superseded by this report's table and by
  `/bk/l1_wider_margin_active_fixed_stepped/`. The t=0 IC (`l1_wider_margin_active_ic220.h5`)
  itself was kept — it isn't "buggy" data, just u/phi/lam state that self-heals on load, and
  remains the correct starting point for any future rerun.

## Open / next

- The `enable_cross_level_gradient_correction` toggle (the earlier, unrelated dead-end
  extrapolation approach) is now doubly confirmed obsolete — still present as a disabled-by-
  default diagnostic, not removed this session; candidate for deletion.
- `[reflux-probe]` debug output (`print_amr_reflux_magnitude`, `EQUILIBRIUM_DEBUG_DO`-gated,
  only visible in a non-NDEBUG build) showed `matched: n=0` / `mismatched: n=132662` every step
  during the fixed rerun. Not yet determined whether this is pre-existing behaviour of that
  specific probe (never visible before since prior runs were NDEBUG builds) or something the fix
  changed — flagged, not chased down this session.
