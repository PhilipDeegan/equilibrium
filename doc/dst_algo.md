# Algorithm/numerics risk log

Running log of numerical-correctness concerns in the core lattice/AMR
algorithms — not necessarily confirmed bugs, but candidates worth checking
before trusting results in that code path.

## AMR numerical-risk survey (2026-08-04)

Triggered by finding and fixing a real catastrophic-cancellation bug in
`accumulate_to_parent` (`inc/eq/core/data/lattice/adaptive_lattice.hpp:259-321`):
the restrict step computed `p.u = p.u + damping*(new_u - p.u)` (damping=1.0
in practice). When `p.u` and `new_u` differ by tens of orders of magnitude
(routine at deep AMR levels, where fields decay by 60+ orders of magnitude
with distance from a source), `new_u - p.u` rounds to just `-p.u` in double
precision — the entire `new_u` term is lost — and the result silently
collapses to `0` instead of the tiny-but-real restricted value. Caught via
`refined_poynting_ring.py --max-level 4`, which tripped the existing
`check_state_priot_accumulate_to_parent_end` conservation assertion
(`debug_adaptive_lattice.hpp:69-93`) deterministically (reproducible at
`--threads 1`, ruling out a race) at the same leaf every time. Confirmed via
direct instrumentation (`EQ_DEBUG_ATP_TRACE` env var trace) that `p.u` swings
across ~20 orders of magnitude between successive substeps at this cell,
then collapses to exactly `0` on the call where `new_u` is far smaller than
the prior `p.u`.

**Fix**: rewrite as the convex-combination form, which is bit-exact-safe
when damping=1:
```cpp
p.lam = (1.0 - accumulate_damping_factor) * p.lam + accumulate_damping_factor * new_lam;
p.u   = (1.0 - accumulate_damping_factor) * p.u   + accumulate_damping_factor * new_u;
```
Verified fix: the `refined_poynting_ring.py --max-level 4` crash is gone,
runs clean through step 5 at `--max-level 8` (though that config still only
reaches `level_max=4` in practice — see caveat below).

**Re-tested against `refined_neutron_decay.py`'s L8 base-update blowup
(`exp/2026/08/04/21_37_l8_accumulate_to_parent_fix_validation`): NO EFFECT.**
Exact reproduction of the historical L8 repro
(`exp/2026/08/04/14_29_l8_etah1e-13_scaleoverride_r20`, `H/H0=134.1` at step
1) against the freshly-fixed `.so` gives `H/H0=134.0626` — bit-identical to
the pre-fix baseline (matches `project_odd27_amr_junction_leak` Part 25's
isolated-harness table for `eta_h[8]=1e-13` exactly). This bug is **not** a
contributing cause of the L8 blowup — the two issues are unrelated (found
via a completely different harness, poynting ring not spheromak/neutron-
decay, and manifested as a hard crash at level 3-4, not smooth H-growth at
L8). The base-update L8 blowup remains fully open; see the neutron-decay
memory for the live `n_same=4`/ancestor-substitution lead.

**Caveat**: `refined_poynting_ring.py`'s shell-telescope construction
(`_build_shell_telescope`) doesn't actually reach the requested `--max-level`
at default `--shell-width0` — it tops out around `level_max=4` regardless of
the requested depth, a separate pre-existing script limitation, not part of
this fix.

Following the fix, surveyed the codebase for the same two risk classes: (1)
the same increment/damping pattern (`x = x + f*(new-x)`), and (2) naive
summation over terms with potentially huge dynamic range. Ranked findings,
highest to lowest priority:

1. **`inc/eq/core/data/lattice/debug/debug_adaptive_lattice.hpp:214-268`**
   `check_total_dphi_balanced` — naive global `total`/`scale` sum over every
   leaf × face; the ratio check can in principle mask real imbalances that
   get swamped by unrelated large magnitude elsewhere. **Update (same day):
   attempted a per-level-bucketed hardening of this and DISPROVED it as the
   right fix — see the dated section below.** The global check turned out to
   be correct as designed; the "masking" risk is real in principle but
   narrower than first thought (see below), and no fix is pending.
2. **`inc/eq/core/data/lattice/adaptive_lattice.hpp:765-783`** `_leaf_H_split`
   — naive running sum of `u²`/`phi²` over all leaves; this is the C++
   diagnostic instrument used to chase the H-growth bug itself (note: the
   headline H/H0 numbers used throughout the investigation come from
   Python's `np.sum`, which is pairwise/compensated, not this function — so
   this is a risk to the diagnostic tool, not (so far as checked) to the
   documented results).
3. **`inc/eq/core/data/lattice/bcc_geometry.hpp:279-356`**
   `_prolong_u_gradient_wide` — `du = neighbour.u - par_u` subtraction
   feeding a least-squares gradient fit for prolongation; same code family
   (restriction's counterpart) as the fixed bug. **Checked (same day) — see
   the dated section below: lower risk than `accumulate_to_parent`,
   confirmed via a derived stress test rather than assumed.**
4. **`inc/eq/core/data/lattice/adaptive_lattice.hpp:2258-2436`**
   `_step_cross_mirror`/`_step_cross_mirror_proxy` — atomic multi-writer
   fan-in into `temp_phi_mirror` at AMR junctions; non-deterministic
   accumulation order (already flagged as such in an existing code comment),
   no compensation.
5. **`inc/eq/core/data/lattice/adaptive_lattice.hpp:1773-1829`**
   `_step_pass2_amr_correct` / `_accumulate_amr_registers` — junction reflux
   `werror`/`sum_A`/`sum_B` naive sums.
6. Face-flux `acc += face_weight[k]*phi[k]` loops
   (`adaptive_lattice.hpp:1984-1985`, `:2474-2475`, `substrate.hpp:61-62,171`)
   — only 14 terms, lower risk, but duplicated 4x; candidate for a shared
   compensated-sum helper.
7. **`equilibrium/substrate/lattice.py:758`**,
   **`equilibrium/substrate/adaptive.py:168`** `step_lam_relax` — textbook
   `x + f*(new-x)` pattern, but `lam` is bounded to `[0, LAM_MAX≈2879]`, not
   the huge dynamic range that broke `accumulate_to_parent` — low risk.
8. `accumulate_to_parent`'s own remaining `u_sum`/`lam_sum`/`phi_sum` loops
   (`adaptive_lattice.hpp:277-280,322`) — low risk: sibling terms are
   same-level/same-neighbourhood by construction, unlike the cross-generation
   blend that was actually broken.

**Non-issue / positive control**: `inc/eq/core/data/lattice/bc.hpp:91-93`
(`BoundaryCondition::sponge`) already uses the safe convex-combination form
— confirms this shape is the established idiom elsewhere in this codebase.

**Stale doc note**: `debug_adaptive_lattice.hpp:66`'s comment ("p.u +
(new_u - p.u) is not always bit-identical to new_u") still describes the
*old* increment formula's ULP-drift behaviour, not the cancellation failure
mode that was actually found, and not the new convex-combination arithmetic
— worth updating for consistency.

## Derived regression tests (2026-08-04)

The `accumulate_to_parent` bug above didn't need a lucky organic repro to
find in the first place: `fl(a+b) == a` exactly, in IEEE-754, whenever
`|b| <= 0.5*ulp(a)` — i.e. whenever `|new_u/p.u|` drops below `~2⁻⁵³ ≈
1.1e-16`. That's a deterministic property of the arithmetic, not a
probabilistic risk, so it can be constructed directly rather than discovered
by chance. Added three permanent tests to
`tst/eq/core/data/lattice/adaptive_lattice.cpp` (all pass) that turn the
findings above into falsifiable, deterministic checks instead of
one-off organic observations:

1. **`test_accumulate_extreme_dynamic_range`** — poisons a parent's `.u`
   with a "stale" `O(1)` value and sets the children's mean many orders of
   magnitude smaller, sweeping the ratio from `1e-1` down to `1e-300`
   (spanning the derived `~1.1e-16` threshold). Confirms the fixed restrict
   step tracks the children's mean at every ratio — this is the direct
   regression guard for the fixed bug; it fails immediately if the old
   increment formula ever comes back.
2. **`test_sibling_sum_order_dependence`** — turns finding #8 (children
   `u_sum`/`lam_sum` loop rated "low risk") from an assumption into a
   measured fact: 1 dominant `O(1)` sibling + 26 near-ulp-scale (`1e-16`)
   siblings, summed in two different index orders. Confirms the naive loop
   *is* genuinely order-sensitive (`u_sum` differs by `2.665e-15` between
   orderings — a real, measurable effect) but the discrepancy stays bounded
   by the small siblings' own total magnitude (`26*1e-16`) rather than
   losing the signal entirely — categorically different from the fixed
   bug's total, guaranteed annihilation.
3. **`test_check_total_dphi_balanced_global_blind_spot_is_expected`** — crafts
   one leaf's 14 faces into a large (`O(1)`), exactly-canceling "region A"
   (6 faces) and a tiny (`O(1e-90)`), all-same-sign, 100%-locally-unbalanced
   "region B" (8 faces). Confirms `check_total_dphi_balanced` does NOT throw
   despite region B being genuinely, fully unbalanced in isolation
   (`rel=1.000`) — and, per the finding below, this is now understood to be
   **correct, intended behaviour of a global invariant**, not a limitation
   to fix. (Originally written as a "characterization test for a known,
   unfixed blind spot"; renamed and reinterpreted after the per-level
   attempt below was disproven.)

## Per-level bucketing attempted and disproven (2026-08-04, same day)

Acting on finding #1 above, `check_total_dphi_balanced` was changed to
bucket `total`/`scale` by each leaf's own level, on the theory that a
mesh-wide sum could mask a real defect confined to one level. This
immediately tripped inside an existing, previously-passing test —
`test_boundary_same_level_conservation` (single Odd27 split of an
off-center Gaussian bump, one flat, non-subcycled `lat.step()`) — with
`level=0 total=8.4517e-06 scale=2.484 rel=3.40e-06`, four orders of
magnitude past the `1e-10` tolerance.

Added a temporary `EQ_DEBUG_DPHI_LEVELS` trace (per-level total/scale/rel,
non-throwing) to see every level's numbers instead of just the first
failure, and reran the same test:

```
DPHI_LEVEL lvl=0 total=+8.451674961e-06  scale=2.484423889   rel=3.40e-06
DPHI_LEVEL lvl=1 total=-8.451674961e-06  scale=0.01377966743 rel=6.13e-04
```

Level 0's total and level 1's total are **exact negatives of each other**
(to displayed precision). This is not two independent defects — it's
legitimate cross-level flux exchange, conserved exactly in aggregate, which
is precisely what the reflux-correction machinery is supposed to do. A
level's own bucket being nonzero is the *expected*, *correct* signature of
healthy cross-level flux transfer, not evidence of a leak. Per-level
bucketing was therefore the wrong invariant: any two levels that correctly,
legitimately exchange flux will make each side's own total nonzero, so the
per-level check produces a false positive on entirely healthy behaviour.
(This also retroactively explains why `test_multilevel_telescope_dphi_conservation`
passed cleanly under the per-level version — that test's particular
geometry happens to produce a very small net cross-level exchange, not
because it's somehow more "correct" than the failing test.)

**Reverted** `check_total_dphi_balanced` to check the global aggregate
(mathematically the sum of the per-level totals/scales, so bit-identical to
the pre-2026-08-04 behaviour) — kept the per-level breakdown as an opt-in,
non-throwing diagnostic (`EQ_DEBUG_DPHI_LEVELS`), since it's a genuinely
useful view for future investigation even though it isn't the right thing
to assert on. `test_check_total_dphi_balanced_catches_deep_level_imbalance`
(the two-different-levels version written to validate the per-level fix)
was replaced by
`test_check_total_dphi_balanced_global_blind_spot_is_expected` (back to a
same-leaf, two-synthetic-region scenario), asserting the ORIGINAL
no-throw behaviour, now understood as correct rather than a gap.

**Net effect of this whole detour**: finding #1's "masking" risk is still
theoretically real (an isolated, totally uncancelled defect far smaller
than unrelated activity elsewhere in the mesh could in principle still be
invisible to the global check), but per-level bucketing is the wrong
mitigation for it, since legitimate multi-level meshes routinely have
nonzero per-level totals as a matter of course. If this is ever revisited,
the right invariant would need to isolate genuinely *unmatched* imbalances
(e.g. per matched-junction-pair, or per connected flux-exchange component)
rather than per raw level number — not attempted, no clear design for it
yet. This wasn't a wasted detour, though: it directly confirmed (via a
real, previously-untested measurement) that the AMR reflux-correction
mechanism conserves flux exactly in aggregate across a live L0/L1 junction,
which is a genuinely new, positive piece of evidence for the mechanism's
correctness — this had not been directly measured before, only inferred
from the older `test_multilevel_telescope_dphi_conservation`'s (as it turns
out, coincidentally small-exchange) result.

## `_prolong_u_gradient_wide` checked (2026-08-04, same day)

Checked the prolongation counterpart to the fixed `accumulate_to_parent`
bug. Its own doc comment claims `mean(children.u) == parent.u` "exactly
regardless of the fitted gradient" because the 27 Odd27 offsets are
symmetric under inversion (`Σoffset == 0`). Verified this premise directly
(scratch check against `bcc_defs.hpp`'s `odd27_table`, integer arithmetic,
no ambiguity): summing `offset[c]` over all 27 children gives **exactly
`(0, 0, 0)`**. The premise is true.

Added `test_prolong_u_gradient_wide_extreme_gradient`
(`tst/eq/core/data/lattice/adaptive_lattice.cpp`): seeds a field that's
`base_value` (`O(1)`) plus a huge (`K=1e18`) LOCAL linear gradient relative
to the parent's own position, so every neighbour/child ends up `O(K)` —
~18 orders of magnitude larger than the parent's own value — then checks
(a) the fit itself still reconstructs the true analytic field at each
child's position, and (b) `mean(children.u)` still recovers `parent.u`.

**First attempt used an arbitrary tolerance (`1e-3`) and "failed"**:
measured `mean(children.u) = -2.3703703703703702` vs expected `1.0`
(`conservation_err = 3.37`) — a suspiciously clean-looking number
(`-64/27`) that looked like it might be a real, systematic bug (which is
why the offset-table symmetry was verified directly above, to rule out an
asymmetric table as the cause). But `max_rel_err` was only `8.3e-15`
(near machine epsilon) — a *relative* error that small on a term of
magnitude `~1e18` is an *absolute* error of order `10³`, and summing 27
such independently-rounded terms that must cancel down to `~1` will
ordinarily leave noise around that scale. This is the same universal
floating-point cancellation phenomenon as `test_sibling_sum_order_dependence`,
just far more extreme — **not** the same failure mode as the fixed
`accumulate_to_parent` bug (which was an unbounded, guaranteed
annihilation from a specific "increment" formula; this is bounded,
expected noise proportional to `K`). Fixed the test's tolerance to be
properly *derived* (`K * DBL_EPSILON * n_children`, ×100 margin) instead
of an arbitrary constant — the measured error (`3.37`) sits comfortably
within the derived bound (`~6000` for `K=1e18`), confirming the function
isn't doing anything worse than ordinary floating-point arithmetic
permits.

**Conclusion: `_prolong_u_gradient_wide` is genuinely lower risk than
`accumulate_to_parent` was**, for a structural reason, not just luck this
time: it only faces *spatial* magnitude variation across one local (~5-hop)
BFS stencil, where nearby cells' values are expected to vary smoothly and
stay comparable in magnitude (an 18-order-of-magnitude jump between
immediately-adjacent cells is not physically realistic for this codebase —
the 60+ orders of magnitude seen elsewhere is a whole-domain, near-source-
vs-deep-vacuum phenomenon, not a single-neighbourhood one). It never faces
the *temporal* staleness mechanism that broke `accumulate_to_parent` (an
old, much-larger value from a previous substep compared against a much
smaller current one). No fix needed here — closed.

## Exact identities of step() (2026-08..10)

- Uniform σ: dφ_ik/dt = −σ W_k (u_nb − u_i) is a weighted discrete gradient, so every closed-loop
  φ sum is time-invariant (fp level): ring circulation is frozen, `E_full = −∇u − ∂φ/∂t ≡ 0`
  (valid only with W_k-weighted gradients). Where σ varies, loop sums change in proportion to
  (σ_face − 1), i.e. linearly and weakly in λ. Transport of circulation needs the separate
  Cartan relation (dst.md, *Circulation transport*).
- Under the map φ_k = W_k A(x_face)·d_k step() is u_t = −c² div A, A_t = −∇u: the Lorenz
  condition with E ≡ 0 (no Faraday). 7 face DOF per cell: 3 are A, 4 are non-vector.
- For a ring of fixed circulation Γ, H·r is exactly constant (H ∝ 1/r), hops 1–10
  (`exp/2026/09/02/18_28_circulation_quantization/`; legacy weights). Never tested dynamically.
- Charge = div φ = −u_t: there is no static u gradient (a held gradient drives φ as t²); a static
  Coulomb exterior ψ ~ 1/r − 1/b exists only with the core held; ε = c²/V_cell.
- Two DC flux sources repel as 1/d² (0.92–0.96 of continuum); the residual vs classical is
  discrete-Laplacian truncation.

## Continuous model, level 0 (2026-09-24..28)

- Weights: discrete step() 3:1 star (Kelvin geometry, W = √3:1, c = 2.309); continuous model
  iso 2:1 (W = √2:1, C0SQ = 4, c = 2; the only ratio isotropic at 2nd order in both sectors);
  legacy 9:1 (pre-09-28). `anchor::Weights`, `equilibrium/substrate/bcc_symbol.py`,
  `tst/substrate/test_hodge_star.py`.
- C++ `continuous::AdaptiveLattice` level 0: step, step_sponge, step_ae, step_cartan,
  `set_face_medium(dsig, dn)` (L = h/<ℓ>), per-instance C0SQ; Python `continuous.AnchorCluster`;
  numpy-identical, ~170x faster, ~590 B/cell. AMR prolong/restrict deferred. Runs are named by R/h.
- Curl is DEC: circulation on the 12 Delaunay triangles per cell, curl∘grad = 0 exactly
  (`set_curl('ls'|'dec'|'dec_geo')`, default 'dec'). The LS reconstruction is 4–10x more
  dispersive transversally and leaks gradients into curl. 4 extra DEC modes are gapped at
  ωh/c ~ 5; sampled ICs need the (1 − curlcurl/27) filter (`exp/2026/09/27/dec`, `10/16/relay`).
- Benchmark: gaussian-source Hertzian dipole on AE converges 2nd order to F² P_th = 0.4114
  (R160, orientation independent; `exp/2026/09/{26/dipole,28/iso_dipole}`). Use the finite-source
  continuum answer as reference, not the point dipole.
- Sponges must be face-symmetric: any per-cell damping of φ breaks φ_ki = −φ_ik and is a hidden
  flux sink (fixed 09-29, `tst/substrate/test_bc_invariants.py`; doc/dst/bc.md).
- `SubstrateLattice(radius_cells=R)` takes the physical radius; passing R/h asks for 52 GB.
