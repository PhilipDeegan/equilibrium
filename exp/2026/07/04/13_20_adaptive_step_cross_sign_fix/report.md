# Adaptive step_cross sign-convention fix — confirmed against uniform-lattice reference

**Date:** 2026-07-04 13:20 CEST
**Trigger:** while porting `probe_step_cross_lam_resistance.py`/`test_saha.py` to
the adaptive lattice (`tst/substrate/adaptive/`) as part of porting the
sigma(lam)^2 resistive fix, noticed the adaptive `step_cross`
(`AdaptiveLattice::_step_cross_local`/`_step_cross_mirror`,
`inc/eq/core/data/lattice/adaptive_lattice.hpp`) applies `delta` to phi of
**either sign**, while the uniform C++ path (`eq::substrate_step_cross`,
`inc/eq/substrate.hpp`) restricts to **inward-only** (`phi < 0`). See
[[project_step_cross]] for the full history.

## Diagnosis

New regression test `tst/substrate/test_adaptive_vs_substrate.py::
TestStepCrossSignConvention`: for a single antisymmetric face pair
(`phi[i,k]=-A` inward at cell i, `phi[nb_k,mirror_k]=+A` outward at its
neighbour, `lam=0` so `sigma=1`), predicted and verified to 12 decimal places:

- Uniform: drains the shared face by exactly `n_adj*dt*scale*A` (processes the
  pair once, from the receiving cell's perspective).
- Adaptive (pre-fix): drains the shared face by exactly `2*n_adj*dt*scale*A`
  (processes the pair twice — once from each adjoining cell — double-counting
  the same physical face crossing).

This confirmed the discrepancy is precisely the sign-restriction, not an
unrelated face-index/convention bug (which would not produce such a clean,
exactly-predicted 2x factor).

## Fix

`_step_cross_local`/`_step_cross_mirror` now compute
`inward = cell.phi[k] < 0.0 ? cell.phi[k] : 0.0` and skip when zero — identical
restriction to the uniform path. Rebuilt: `mkn build -Oqp mod,mod_lattice -x mpi
-g 0 -W 9`.

## Verification

1. `TestStepCrossSignConvention` updated to assert the fixed (matching,
   single-processing) behaviour — passes on both backends.
2. Full `test_adaptive_vs_substrate.py` suite: 19/19 pass (no regressions in
   splitting, Harris uniform-vs-adaptive comparison, etc. — none of those
   exercise `step_cross`).
3. Ported `tst/substrate/adaptive/test_saha.py`: 11/11 pass.
4. **Before/after comparison**, `probe_step_cross_lam_resistance.py` (adaptive,
   single ring, cross_scale=0.02, 20k steps, frozen lam per run):

   | lam | pre-fix remaining_frac (double-processed) | post-fix remaining_frac (fixed) | uniform reference (`23_17_step_cross_lam_resistance`) |
   |---|---|---|---|
   | 0   | 7.75e-7  | 4.97e-4 | 0.05%  (5.0e-4) |
   | 1   | 1.68e-2  | 0.1204  | 12%    (0.12) |
   | 5   | 0.627    | 0.7874  | 79%    (0.79) |
   | 20  | 0.963    | 0.9806  | 98%    (0.98) |
   | 100 | 0.998    | 0.9992  | 99.9%  (0.999) |

   **The post-fix adaptive numbers now match the uniform-lattice reference
   almost exactly** (all agree to within ~1% relative), despite the two
   backends being independent C++ implementations exercised through very
   different Python code paths (single flat array vs AdaptiveCellCluster leaf
   arrays). This is strong, independent confirmation both that (a) the sign-fix
   is correct and (b) the sigma(lam)^2 resistive mechanism itself is
   implementation-independent — a genuine property of the physics, not an
   artifact of one code path.

## Consequence — stale adaptive decay numbers

Any adaptive-lattice `step_cross` result measured before this fix now
understates decay resistance by roughly 2x in rate-constant terms (compounding
to much larger differences over long runs — e.g. lam=0's remaining fraction
changed by ~640x over 20k steps). Affects `test_spheromak_decay.py` (adaptive)
runs and `sweep_neutron_lam_ic_decay.py` — same staleness pattern as the
sigma(lam)^2 fix itself caused for the uniform lattice.

## Next steps

- Re-run adaptive `test_spheromak_decay.py`/neutron-decay sweeps with the fixed
  step_cross if their conclusions are still load-bearing.
- Consider whether the Python fallback in `equilibrium/substrate/lattice.py::
  step_cross`'s "symmetric in sign" docstring should be corrected (it's dead
  code while the compiled `.so` exists, but the stale docstring is what
  originally justified the adaptive lattice's now-fixed symmetric behaviour).
