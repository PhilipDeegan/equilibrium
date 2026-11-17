# Single-point-perturbation depth scan, 10 steps each (user-designed) — step-1 decelerating pattern reconfirmed; multi-step trajectory turns genuinely oscillatory past depth ~3

## Motivation (user's own request)

Following Part 32's clean null-test result (pure mesh mechanics on a flat field: zero growth at
every depth 0-8), user asked for the natural next probe: put a single-point perturbation at the
domain centre for each graded telescope depth 0-8, run 10 timesteps (not just one), save the full
`u`/`phi` trajectory after every step.

Extended `refined_saha_depth_scan.py` (previous null-test version) with `--perturb-amplitude`
(applied AFTER the telescope is built, per the established correct methodology),
`--n-steps` (default 10), and full per-step `u_hist`/`phi_hist` snapshot saving.

## Method

`R=30`, `u0=0.0`, single-point `u` spike (`amplitude=1.0`) at the domain-centre leaf, Neumann
(zero-flux) BC, `TimeStepper.by_level`, 10 steps, at each depth 0 through 8 independently (fresh
mesh each time).

## Results

H/H0 at each step (columns 1-10), by depth (rows):

| depth | @1 | @2 | @3 | @4 | @5 | @6 | @7 | @8 | @9 | @10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.0021 | 1.0041 | 1.0061 | 1.0080 | 1.0099 | 1.0117 | 1.0134 | 1.0150 | 1.0165 | 1.0178 |
| 1 | 1.0143 | 1.0528 | 1.1145 | 1.1976 | 1.2999 | 1.4187 | 1.5506 | 1.6923 | 1.8399 | 1.9895 |
| 2 | 1.1487 | 1.5788 | 2.2554 | 3.1238 | 4.1142 | 5.1477 | 6.1426 | 7.0216 | 7.7182 | 8.1815 |
| 3 | 2.4128 | 6.3086 | 11.757 | 17.472 | 22.130 | 24.694 | 24.664 | 22.171 | 17.927 | 13.016 |
| 4 | 13.500 | 42.404 | 68.138 | 74.088 | 57.941 | 32.294 | 14.713 | 15.188 | 30.299 | 47.599 |
| 5 | 101.09 | 221.53 | 158.59 | 44.62 | 71.23 | 156.11 | 153.05 | 108.17 | 100.15 | 97.68 |
| 6 | 611.94 | 285.81 | 267.66 | 467.67 | 297.13 | 283.38 | 481.81 | 320.30 | 189.69 | 663.69 |
| 7 | 1424.5 | 1402.2 | 896.85 | 1376.9 | 206.44 | 1293.6 | 1834.9 | 511.73 | 1096.1 | 746.47 |
| 8 | 2403.9 | 2545.7 | 1701.6 | 4931.6 | 4216.6 | 4125.8 | 6085.5 | 3722.8 | 3506.4 | 4198.7 |

**Step-1 values cross-validate exactly** against earlier same-day single-step u-perturbation
measurements (depth 4/6/8 step-1: 13.500/611.9385/2403.8919, bit-for-bit identical to the earlier
ablation-context u-perturbation scan). Excess ratios per 2-level jump at step 1
(L2→L4→L6→L8: 84.1x → 48.9x → 3.9x) reconfirm the DECELERATING pattern already documented before
today's session (58x→37.5x→3.1x) — same shape, comparable magnitude.

## Interpretation

**Depths 0-2**: smooth, monotonic growth across all 10 steps — no oscillation.

**Depth 3 onward**: the trajectory becomes genuinely non-monotonic. Depth 3 peaks at step 6
(24.69x) then decays back to 13.02x by step 10. Depth 4 peaks at step 4 (74.09x), dips to 14.71x
by step 7, then rises again. From depth 5 on, the swings become large and irregular relative to
any smooth trend: depth 5 ranges 44.6x-221.5x within the 10 steps; depth 6 ranges 189.7x-663.7x;
depth 7 ranges 206.4x-1834.9x (a ~9x swing); depth 8 ranges 1701.6x-6085.5x.

**This is a genuinely new observation, not previously characterized in this investigation at this
granularity**: a delta-function point perturbation, once the telescope goes a few levels deep,
does not simply grow (or decay) smoothly — it drives what looks like a real oscillatory/resonant
mode, with amplitude and irregularity both increasing with depth. This is DIFFERENT from anything
seen in the real-spheromak depth scan (Part 31), which showed smooth, monotonically-accelerating
single-step growth — though that scan was only ever measured at step 1 per depth, so a like-for-
like multi-step comparison against the real spheromak hasn't been done yet (natural next step, if
wanted).

**Consistent with, and extends, the standing conclusion from before today's session** ("a
single-point perturbation does NOT reproduce the real mechanism — it's tapping a different,
smaller, saturating error source"): not only does the step-1 magnitude pattern differ qualitatively
from the real spheromak (decelerating vs. accelerating with depth), the full multi-step *dynamics*
differ too — oscillatory/chaotic-looking here, vs. smooth and monotonic in the one-step-per-depth
real-spheromak measurements. Whether the real spheromak ALSO oscillates over many steps at depth
(it very plausibly might — deltas and smooth fields can excite the same underlying mode, just at
different amplitudes) is not yet tested and would be the most direct way to settle whether this
oscillation is the SAME mechanism as the real blowup, just more visibly excited by an extreme
point-source input, or a genuinely separate numerical artifact specific to sharp inputs.

**Not yet done**: (1) run the real-spheromak depth scan (Part 31's script) for 10 steps too, not
just 1, to see if the same oscillatory character appears there; (2) inspect the saved per-step
snapshots (`tmp/saha_depth_scan_perturb/depth{0..8}.npz`, full `u_hist`/`phi_hist` arrays) to see
WHERE in the mesh the oscillation is concentrated — is it localized near the perturbed centre cell,
or spread across the whole telescope boundary structure?
