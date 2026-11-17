# Dynamic-lam ladder test — L2/L4/L5/L6

First-ever test of the fixed provisional eta_h ladder with `lam_mode=dynamic` (every
prior sweep in `project_odd27_amr_junction_leak` used `lam_mode=frozen`). cross_scale
held fixed at 1e-3 (the already-validated smallest candidate from Parts 17-18) so this
isolates the lam_mode change alone. `damp_max=0` throughout, matching the isolated
harness convention. Cheap levels first (L2/L4/L5/L6); L7 deliberately deferred.

## Parameters (per level)

```
--R 20 --max-level <L> --region-r0 5.0
--scale-override 1.3606e-3 --scale-ref-level 8
--lam-target 1.0 --lam-scale 0.05 --lam-mode dynamic
--cross-scales 1e-3 --damp-max 0
--eta-h-per-level <ladder truncated to L levels>
--n-steps 150 --report 15 --n-threads 8
```

Ladder truncation: L2=`0 0 1e-5`, L4=`0 0 1e-5 1e-7 3e-9`, L5=`... 1e-10`, L6=`... 1e-12`
(from the fixed provisional L0-L7 ladder `0 0 1e-5 1e-7 3e-9 1e-10 1e-12 1e-13`).

## Result

All four completed the full 150 steps, no blowup (`--blowup-factor 100` never tripped).

| max_level | H_0 | H_f | H/H0 peak | H/H0 @150 | lam_mean_f | lam_max_f | circ_r12_f |
|---|---|---|---|---|---|---|---|
| 2 | 3.2773e+03 | 3.4093e+03 | 1.14 | 1.04 | 0.0203 | 0.9920 | -4.33e-26 |
| 4 | 3.2773e+03 | 7.7920e+03 | 2.70 | 2.38 | 0.0139 | 0.9811 | -3.44e-40 |
| 5 | 3.2773e+03 | 6.7713e+03 | 5.41 | 2.07 | 0.0120 | 0.9746 | +3.59e-42 |
| 6 | 3.2773e+03 | 1.3257e+04 | 14.14 | 4.05 | 0.0107 | 0.9696 | +2.02e-43 |

Comparable to (not worse than) the frozen-lam baseline at the same levels — L6 frozen-lam
history (Parts 3/10) settles into a persistent ~5-19x oscillating band; this dynamic-lam
run peaks at 14.14 (step 30) then relaxes down to 4.05 by step 150, at least as good.

`lam_mean_f`/`lam_max_f` now genuinely evolve with level (unlike the frozen case, which
pins them at a fixed 0.0097/1.0000) — confirms lam dynamics are actually active. `circ_r12_f`
is no longer pinned at exact 0 (as it always was under frozen lam) but stays at
floating-point-noise magnitude (1e-26 to 1e-43) — no real winding/circulation has emerged
within 150 steps yet, expected at this cross_scale/step count.

## Diagnostic noise observed during the run (not a defect)

The installed `mod_lattice.so` was built with assertions/debug prints active (not `-g 0`),
so `print_amr_reflux_magnitude`'s `[reflux-probe]` lines fired every step — large log files
as a result. Checked: `rel` (relative reflux magnitude) stayed in a stable ~2.5-4.9% band at
every level, flat-to-slowly-decaying over the 150 steps, not growing — matches the documented
"expected, not a bug" characterization (Odd27's coarse/fine junctions have genuinely unequal
face counts on each side; reflux exists to correct exactly that). Not evidence of a new
problem; just noisier logs than a `-g 0` release build would produce.

## Reading

Dynamic lam does not introduce a new instability mode at any of L2/L4/L5/L6, at least at
this small cross_scale (1e-3). This meaningfully de-risks moving to L7 (the next rung, and
the last before the actual production target L8).

## Next steps

- L7 dynamic-lam test, same recipe (`--eta-h-per-level 0 0 1e-5 1e-7 3e-9 1e-10 1e-12 1e-13`),
  cross_scale=1e-3 — the expensive one (~7h/candidate per Part 16's costing), not yet run.
- Once L7 confirmed: consider combining dynamic lam with the larger cross_scale values
  (1e-2, 1e-1, 1e0) already validated under frozen lam (Part 18), rather than testing
  every combination — a single combined stress test may be enough before L8.
- L8 (`--eta-h-ratio`) still deferred until L7 clears under dynamic lam.
