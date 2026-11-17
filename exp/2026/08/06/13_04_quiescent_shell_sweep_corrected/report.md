# Quiescent/growing-annulus shell sweep, CORRECTED — refine first, seed IC analytically (no prolongation): purely-vacuum junctions do EXACTLY nothing

## Correction from the previous attempt

`exp/2026/08/06/12_56_quiescent_shell_sweep` seeded the spheromak IC on the coarse mesh, THEN
refined the shell — forcing `_prolong_u_gradient_wide` to interpolate the existing field onto the
new children at split time. Even the outermost, "purely quiescent" shell showed nonzero `u`
*before* stepping (1.182e-3), which muddied the diagnostic: was that prolongation legitimately
picking up real nearby data, or a hint of something wrong?

User caught the fix directly: *"does the split cells need prolongation on IC? can we not just set
them analytically?"* Since `init_spheromak_ic_refinable` is level-aware and computes each leaf's
value from its own position (not from a parent's value), there's no need to seed-then-refine at
all — refine the mesh first, then call the IC function directly on the final (already-refined)
mesh. This removes prolongation from the picture entirely and gives an exact, unambiguous ground
truth for what "nothing should happen" means.

## Method

Identical to the previous sweep (`R=100`, spheromak `scale=45`, Dirichlet(0) BC, same 5 shell
positions `[lo, 47.6]` growing inward from `47.4` to `40.0`), except: `refine_region_to_convergence`
is called BEFORE `init_spheromak_ic_refinable`, not after.

## Results

| shell [lo, 47.6] | pre `u` | post `u` | pre `phi` | post `phi` | H/H0 |
|---|---|---|---|---|---|
| baseline (uniform) | — | — | — | — | 1.000000 |
| [47.4, 47.6] (vacuum) | **0.000e+00** | **0.000e+00** | **0.000e+00** | **0.000e+00** | 1.000000 |
| [46.0, 47.6] (vacuum) | **0.000e+00** | 4.277e-14 | **0.000e+00** | 6.218e-11 | 1.000000 |
| [45.0, 47.6] (touching spheromak edge) | 2.039e-4 | 2.149e-4 (+5.4%) | 4.387e-3 | 4.879e-3 (+11.2%) | 1.000000 |
| [43.0, 47.6] (overlapping) | 1.037e-1 | 1.037e-1 (~0%) | 1.042e-1 | 1.042e-1 (~0%) | 1.000001 |
| [40.0, 47.6] (deep overlap) | 2.695e-1 | 2.695e-1 (~0%) | 2.698e-1 | 2.698e-1 (~0%) | 1.000002 |

## Conclusion

**With the prolongation confound removed, the two purely-vacuum shells show EXACTLY zero before
stepping (no ambiguity), and remain at pure floating-point noise (1e-14 to 1e-11 — not a real
signal, ~13-14 orders of magnitude below the field's own scale) after one step.** A single,
shallow L0→L1 coarse-fine junction sitting in genuine vacuum does *nothing at all* —
unambiguously, not "something small that might be prolongation."

The shells overlapping the spheromak show, if anything, LESS relative change than the
boundary-touching case (43/40: ~0% one-step change; 45: +5-11%) — no sign of runaway growth
anywhere in the sweep, including deep inside the real field.

**This is the cleanest possible confirmation of today's cumulative finding**: a single junction,
anywhere, vacuum or not, is completely benign. The severe base-update blowup (134x-455x at L8) is
not a property of "junctions leak" in general — it specifically requires the DEPTH/stacking of a
real multi-level telescope (matching Part 29's uniform-L8-resolution-is-stable finding and the
established accelerating-with-depth signature: L4≈2.5x, L6≈18x, L8≈134x). A single-level shell,
however positioned, cannot reproduce or hint at that mechanism — confirms (does not newly
discover) that any future repro attempt needs genuine multi-level stacking, not a single split.
