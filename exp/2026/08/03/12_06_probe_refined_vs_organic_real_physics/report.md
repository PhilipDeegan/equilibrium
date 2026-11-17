# probe_refined_vs_organic_amr under real physics (max_level 0/1/2)

Before committing further effort to L7/L8 eta_h calibration, checked whether
`refined_neutron_decay.py`'s fixed-telescope AMR actually reproduces
`sweep_neutron_decay_cross_scale.py`'s organic/growing AMR under REAL physics
(`lam_mode=dynamic`, `damp_max=0.003`, `cross_scale=1e-3` — all script defaults,
"everything active"). Previously only validated at `max_level=0` (trivial, both
flat/uniform — see [[project_bohr_orbit_amr_validation]]); `max_level>0` was deferred.

## Result

**max_level=0** (re-check against current `.so`): rel_diff=0.000e+00 at every one of
11 reported steps (0 to 300). Exact match still holds — confirms the shared
physics/IC path is unaffected by everything that changed since this was last checked
(Part 12-14 fixes etc).

**max_level=1**: rel_diff is **NOT zero even at step 0** (1.054e-03) and grows
monotonically to 9.583e-02 (~9.6%) by step 100. `n_leaves` differ too: organic=2893987
vs refined=2801505.

**max_level=2**: rel_diff at step 0 is **12.9%** (`H_organic`=1.2215e4 vs
`H_refined`=1.3794e4) — already large before any stepping — then *shrinks* over 33
steps down to 0.71%. `n_leaves`: organic=2968347 vs refined=2818275.

## Root cause traced (IC construction, not evolution drift)

Checked the IC-seeding lines directly:

| level | organic inner_cells | organic phi_rms | refined inner_cells | refined phi_rms | phi_rms rel diff |
|---|---|---|---|---|---|
| 1 | 16361 | 2.2254e-02 | 15657 | 2.2314e-02 | ~0.27% |
| 2 | 15815 | 2.0397e-02 | 15657 | 2.2247e-02 | ~9.1% |

`H` depends roughly quadratically on `phi_rms` (plus the `u` term), so a 9.1% phi_rms
gap at L2 plausibly explains the ~13-19% H gap seen at step 0 there, and the small
0.27% phi_rms gap at L1 plausibly explains its much smaller (0.1%) step-0 H gap.

**This means the two IC-seeding functions (`init_spheromak_ic` used by the organic
script vs `init_spheromak_ic_refinable` used by the refined script) do NOT produce
matching fields once real (non-flat) AMR structure is involved**, even at the same
nominal `scale`/`R_sph` — despite matching exactly at `max_level=0`. The mismatch is
present *before* any stepping, so it is an IC-construction difference, not (only) an
evolution-time AMR-mechanics difference. Likely mechanism (not yet directly confirmed):
organic's `split_hi`-driven adaptive refinement produces a different actual leaf-level/
cell-size distribution near the object than the refined script's fixed pre-built
telescope shells, and phi-hop scaling (`cell_size`-dependent, per each IC function) is
evaluated against those different local structures.

## Reading

The module's own docstring anticipated "physics may legitimately differ somewhat —
different refinement geometry" at `max_level>0`, but a 9-13% step-0 (pre-stepping) `H`
mismatch is larger than "should still track closely" suggests, and is a DIFFERENT
concern from the eta_h ladder's stability question — this is about whether the fixed-
telescope approach represents the same physics at all, independent of any boundary
H-growth/eta_h issue. **The entire L1-L7 eta_h ladder validated so far (Parts 6-19)
has only ever been checked for its OWN internal stability (bounded H/H0), never cross-
checked against the organic-AMR reference at any level above 0.** This is an open gap,
not previously flagged in `project_odd27_amr_junction_leak`.

## Follow-up: discrepancy CONFIRMED and decomposed (not a new defect)

Built organic's actual grown mesh directly (`confirm_ic_hop_discrepancy_L1.py`/`_L2.py`,
replicating `sweep_neutron_decay_cross_scale.py`'s `_build_and_run` build+pre-adapt+reseed
sequence), then recomputed phi on that SAME mesh two ways: organic's actual fixed
level-0-sized hop, and a per-leaf `cell_size`-scaled hop (refined's own documented fix,
see `init_spheromak_ic_refinable`'s docstring — a known bug found 2026-07-18, "a 560-cell
inner region at level 14 seeded phi_rms=0.0 exactly").

Sanity check: reimplementing organic's own formula on its own mesh reproduces its
measured phi_rms exactly (L1: 2.225368e-2 vs logged 2.2254e-2; L2: 2.039677e-2 vs logged
2.0397e-2) — confirms the replica is faithful.

**Result: the known hop-scaling bug explains part, not all, of the gap**:

| level | organic actual phi_rms | scaled-hop-on-organic-mesh phi_rms | refined actual phi_rms | total gap | hop-scaling share |
|---|---|---|---|---|---|
| 1 | 2.225368e-02 | 2.227957e-02 | 2.2314e-02 | 6.03e-5 | **43%** |
| 2 | 2.039677e-02 | 2.109299e-02 | 2.2247e-02 | 1.85e-3 | **38%** |

The remaining ~57-62% of the gap is NOT explained by the hop-scaling formula alone — it
tracks a genuine mesh-geometry difference: organic's threshold-driven growth produces a
structurally different actual refined region than refined's deliberately-shaped fixed
telescope. Directly visible at L2: organic's own converged mesh has only **21** level-1
leaves but **125550** level-2 leaves (a near-absent level-1 shell, most of the region
jumping straight to level 2) — a different shape than the telescope's deliberate
per-level shell sizing. The per-cell hop-scaling ratio itself also grows with level
(L1 level-1 ratio=1.0012 → L2 level-1 ratio=1.1995, level-2 ratio=1.0337), consistent
with the original 2026-07-18 finding that this bug's impact compounds at deeper levels.

**Conclusion: both contributing effects are principled and already anticipated by the
module's own docstring ("physics may legitimately differ somewhat — different
refinement geometry"), not evidence of a new, unexplained bug.** Refined's per-leaf IC
scaling is verified more correct on a mixed-level mesh (organic's own documented
limitation); the rest of the gap is an expected consequence of organic (criterion-driven,
emergent) and refined (deterministic, pre-built) choosing genuinely different actual
refined regions. This closes the open question from the first half of this report —
no further investigation needed to explain WHY they differ.

## Follow-up: which mechanism drives the EVOLUTION-TIME trend (not just step 0)?

User proposed 3 candidate mechanisms for the growing-at-L1/shrinking-at-L2 rel_diff
trend seen AFTER step 0 (distinct from the already-explained step-0 IC gap above):
(1) `accumulate_to_parent` restriction, (2) prolongation into newly-split children,
(3) per-level eta_h.

(3) ruled out immediately: this comparison ran with `eta_h_per_level=None`/
`eta_h_ratio=None` throughout — no boundary hyper-resistivity active on either side.

**Control experiment for (2)** (`probe_frozen_organic.py`): organic's mesh keeps
splitting further during the recorded run by default (`amr_growing=True`, via
`GrowingAMRSchedule`) — this is what exercises prolongation into new children
repeatedly. Refined's telescope is fully static after IC (confirmed by reading
`refined_neutron_decay.py`: `refine_region_to_convergence` + `clear_split_criterion()`,
no further splitting logic exists on that side at all). Re-ran organic with
`amr_growing=False` (frozen at the pre-adapt-converged mesh, matching refined's static
regime) and compared again.

**Result: barely changed anything.**

| level | rel_diff, organic growing (original) | rel_diff, organic frozen (control) |
|---|---|---|
| L1 @ step 100 | 9.583e-2 | 9.564e-2 |
| L2 @ step 33 | 7.126e-3 | 6.692e-3 |

Confirms continued organic splitting/prolongation-into-new-cells is NOT the driver of
the evolution-time trend.

**Current reading**: the evolution-time growing/shrinking pattern is best explained as
two individually-static-but-differently-shaped meshes (the shape difference already
established at step 0, see above) evolving forward under the same shared physics and
naturally drifting apart or converging differently — not a distinct evolution-mechanics
bug. (1) `accumulate_to_parent` was NOT isolated by this control — it runs every
substep regardless of `amr_growing`, in both the frozen and growing organic variants —
so there's no direct evidence either way on whether it's contributing beyond "operating
correctly on two different mesh shapes."

## Follow-up: (1) `accumulate_to_parent` directly isolated (not just ruled out by absence of evidence)

User wanted (1) tested directly rather than left as "no evidence either way" after the
frozen-organic control only addressed (2). Added a new C++ test,
`test_static_boundary_consistency_unbalanced_multilevel`
(`tst/eq/core/data/lattice/adaptive_lattice.cpp`, copy saved here as
`adaptive_lattice_with_new_tests.cpp`), extending the project's existing static
(no-stepping) consistency-check methodology to a topology matching what organic
actually produces at L2: split one L0 target into 27 L1 children, then split only ONE
of those further into 27 L2 grandchildren — 26 L1 siblings stay unsplit (mirrors the
21-vs-125550 L1/L2 leaf-count imbalance found earlier). Seeded an analytic affine field,
checked prolongation accuracy at the deepest leaves, `accumulate_all`'s full L2→L1→L0
restriction round-trip, and the intermediate L1 parent's own restricted value.

**Result: exact at every check** (prolong err=2.220e-16, round-trip restrict
err=0.000e+00, L1-parent restrict err=0.000e+00). `accumulate_to_parent`'s arithmetic is
confirmed correct through genuinely unbalanced multi-level trees — not just the
balanced/single-gap case the pre-existing tests covered. Combined with the earlier
frozen-organic control (which ruled out (2)), both of the user's suspected mechanisms
are now positively confirmed NOT the source of the refined-vs-organic mismatch.

**Bonus catch while running the full suite**: `test_multilevel_telescope_dphi_conservation`
(a different, pre-existing test, added 2026-07-21) failed on a fresh run. Verified via a
proper before/after check (backed up the edited file, temporarily restored the exact
HEAD version via `git show HEAD:path` + plain file copy, rebuilt, same failure) that
this is PRE-EXISTING and unrelated to the new test above. Root cause: this test computes
its own raw (unweighted) `Σphi[k]` conservation check inline, written before a later fix
(elsewhere in the project) switched the actual AMR correction to target the
face-weight-WEIGHTED sum instead — the same class of stale-invariant bug already fixed
once in a different function, but this test's own inline check was never updated to
match. Confirmed directly: raw sum rel=6.5e-7 (fails), weighted sum rel=1.3e-16 (exact).
Fixed to match the established weighted convention — both `test_lattice` C++ binaries
now report `ALL PASSED`.

## Conclusion

The refined-vs-organic mismatch investigation (this report, in full) is closed. Step-0
gap = IC hop-scaling bug (43%/38% share) + differing actual refined-region shape (rest).
Evolution-time trend = neither prolongation-during-growth nor `accumulate_to_parent`'s
restriction arithmetic (both now positively confirmed correct) — just two
differently-shaped, individually-correct meshes evolving forward under the same shared
physics. No outstanding suspicion of a hidden AMR-mechanics bug behind this comparison.

## Next steps

- This does not itself validate that refined's fixed telescope is *quantitatively*
  faithful to organic at L6/L7/L8 (only checked at L1/L2 here) — the same two-effect
  decomposition should generalize, but wasn't re-verified at deeper levels.
- Given the mismatch is fully understood and every contributing mechanism checked is
  either legitimate (not a defect) or directly confirmed correct, this meaningfully
  de-risks proceeding to L7 dynamic-lam and eventually L8.
- Not yet tested: max_level=3+ (would need `--n-steps` rescaled up substantially per the
  script's own time_factor convention, since n_steps shrinks fast with level by default).
- The new test and the `test_multilevel_telescope_dphi_conservation` fix are uncommitted
  in the working tree (`git diff --stat` shows the file changed) — left for the user to
  review/commit themselves, per standing policy.
