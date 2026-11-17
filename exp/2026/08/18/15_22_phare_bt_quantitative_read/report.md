# First quantitative read of PHARE's B(t) series (t=0->3.4)

## Question

Does PHARE's evolved spheromak state (specifically requested: t=3.0) give
any hint about what a correctly-behaving DST spheromak should look like?

## Method

`~/tmp/EM_B.vtkhdf` (35 unique steps, t=0->3.4, dt=0.1, YZ slice at x=51 --
see `project_dst_discrete_curl` memory for the reverse-engineered file
layout / duplicate-series gotcha). Reused
`exp/2026/08/16/14_50_.../plot_phare_b_slice.py`'s `load_step_grid` reader
unchanged. Tracked, over the r<15-cell core:

- `Bx_rms` -- out-of-plane component, toroidal-like
- `sqrt(By^2+Bz^2)_rms` -- in-plane component, poloidal-like
- whole-slice total field energy `sum(Bx^2+By^2+Bz^2)`

No PHARE-t to DST-cross_scale unit conversion attempted -- this is a
qualitative "which regime" read, not a calibrated match.

## Result: a clean two-timescale relaxation

| phase | window | Bx_rms (toroidal-like) | in-plane rms (poloidal-like) | total E |
|---|---|---|---|---|
| fast | t=0 -> ~1.8 | 0.136 -> 0.011 (-92%) | 0.138 -> 0.127 (-8%) | 1.13 (t=0.1 peak) -> 0.43 |
| slow | t~1.8 -> 3.4 | 0.0109 -> 0.0103 (-5%) | 0.127 -> 0.119 (-6%) | 0.43 -> 0.38 |

Full table and plot: `plots/phare_bt_two_timescale.png`,
`analyze_phare_bt.py` stdout.

- The toroidal-like component (`Bx`) collapses almost entirely (~92%) in
  the fast phase, then plateaus -- this is a real reconnection/annihilation
  burn-off, not slow diffusion. This is almost certainly WHERE the two
  lobes actually merge into one connected loop (previously only eyeballed
  at t=0/1.4/3.4).
- The poloidal-like component (in-plane By,Bz) decays smoothly through
  BOTH phases, no plateau -- once the fast toroidal burn-off is done, the
  slow phase is a steady bleed of the poloidal piece, not a frozen system.
- Total field energy briefly RISES (+13%) at t=0.1-0.2 (transient
  reorganization) before falling steadily; by t=3.4 total energy is down
  62% from t=0, almost all of that loss inside the fast phase.
- At t=3.0 specifically: already deep in the slow/plateau phase
  (Bx_rms=0.0103, barely distinguishable from t=3.4's 0.0103) -- t=3.0 is
  NOT a representative snapshot of the interesting (merging) dynamics,
  which is essentially over by t~1.8-2.0.

## Why this matters for DST

DST's own curl-full dipole IC test (`project_curl_full_dipole_spheromak`
memory, `exp/2026/08/17/12_25_.../step_curlfull_dipole_*.py`,
cross_scale=1e-3, 200 steps) kept H/H0 in [0.9979, 1.0] the ENTIRE run --
total field-energy loss under 0.3%. PHARE loses >50% of its total field
energy within its fast phase alone (t<1.8). That's a >100x mismatch in
erosion strength between what's been tried and what PHARE's own
reconnection actually looks like.

Conclusion: the earlier "no visible merging through step 200" result was
almost certainly read entirely inside a regime analogous to PHARE's
SLOW/plateau phase -- DST has not yet been stepped anywhere near a
`cross_scale` strong enough to reach PHARE's fast/reconnection-dominated
regime at all. This reframes the open question from "does DST ever merge"
to "has DST's step_cross erosion ever actually been run hard enough to
find out."

A second, independent point: H/H0 is likely the wrong diagnostic for
"has merging happened." PHARE's own topology change (loop connecting the
two lobes) is carried almost entirely by the toroidal-like component,
which is a small fraction of total field energy at any given time -- a
scalar like H/H0 can look nearly unchanged while the actual field topology
has completely reorganized. The better diagnostic (matching what
distinguished `phi_tor` from `phi_pol` in the curl-full IC's own test
suite) is a poloidal/toroidal (or macro-circulation) component split, not
total H.

## Next

- Re-run the curl-full dipole IC at several `cross_scale` values between
  1e-2 (last trusted calibration point) and 1000 (confirmed unstable),
  watching the toroidal-like (`phi_tor` amplitude / macro ring circulation)
  vs poloidal-like (`u` / `phi_pol`) component split for a PHARE-like
  fast-collapse-then-plateau shape, not just H/H0.
- Eventually attempt a real PHARE-t <-> DST-cross_scale unit conversion
  (both are frozen-in-flux erosion-rate knobs) -- not attempted here.
