# Dynamic-lam ladder test — L7 (last rung before L8)

Follow-on to [`10_40_dynamic_lam_ladder_sweep`](../../10_40_dynamic_lam_ladder_sweep/report.md)
(Part 19, L2/L4/L5/L6). Same recipe, extended to the full provisional L0-L7 eta_h ladder at
`max_level=7`, `lam_mode=dynamic`, `cross_scale=1e-3`. This was the last untested rung before
the actual production target L8 (Part 16 costed it at ~7h/candidate under the isolated
harness). Run completed on another machine — the local `dynamic_lam_L7.log` in this directory
stalled at step 45 (still from an earlier attempt on this machine) and is stale; the numbers
below are from the completed run's console output.

## Parameters

```
--R 20 --max-level 7 --region-r0 5.0
--scale-override 1.3606e-3 --scale-ref-level 8
--lam-target 1.0 --lam-scale 0.05 --lam-mode dynamic
--cross-scales 1e-3 --damp-max 0
--eta-h-per-level 0 0 1e-5 1e-7 3e-9 1e-10 1e-12 1e-13
--n-steps 150 --report 15 --n-threads 8
```

## Result

Completed the full 150 steps, no blowup (`--blowup-factor 100` never tripped — peak H/H0
stayed under 60).

| max_level | H_0 | H_f | H/H0 peak | H/H0 @150 | lam_mean_f | lam_max_f | circ_r12_f |
|---|---|---|---|---|---|---|---|
| 2 | 3.2773e+03 | 3.4093e+03 | 1.14 | 1.04 | 0.0203 | 0.9920 | -4.33e-26 |
| 4 | 3.2773e+03 | 7.7920e+03 | 2.70 | 2.38 | 0.0139 | 0.9811 | -3.44e-40 |
| 5 | 3.2773e+03 | 6.7713e+03 | 5.41 | 2.07 | 0.0120 | 0.9746 | +3.59e-42 |
| 6 | 3.2773e+03 | 1.3257e+04 | 14.14 | 4.05 | 0.0107 | 0.9696 | +2.02e-43 |
| **7** | **3.2773e+03** | **6.0878e+04** | **58.47** | **18.58** | **0.0093** | **0.9215** | **-7.01e-46** |

(L2/L4/L5/L6 rows carried over from Part 19 for the trend; only L7 is new.)

`circ_r12_f` stays at floating-point-noise magnitude, as at every prior level — no real
winding has emerged within 150 steps at this cross_scale, expected.

## Reading

No blowup — the ladder survives L7. But this is not a clean "flat, no new instability"
result like L2/4/5/6 were: the H/H0 peak-per-level sequence (1.14, 2.70, 5.41, 14.14, **58.47**)
and the @150 settle sequence (1.04, 2.38, 2.07, 4.05, **18.58**) both show L6→L7 jumping by
roughly 4-4.5x, well above the ~2-2.6x per-level growth seen between L4/L5/L6. Whether that's
the ladder's own eta_h values (`1e-12`→`1e-13` at L6→L7) losing damping headroom faster than
earlier steps, or a genuine feature of going one level deeper, isn't distinguished by this run
alone. Framed against the frozen-lam cross_scale bound from Parts 17-18 (H/H0 bounded
~[10.3, 57.6] across cross_scale values at fixed L7) — L7 dynamic-lam's peak (58.47) and floor
(9.84, at step 45) both sit right at the edge of that band, not comfortably inside it.

Bottom line: cleared, not blown up, but the safety margin visibly narrowed at this rung
compared to L2-L6. Worth treating as a flag, not a green light, before committing to L8.

## Next steps

- Before L8: understand whether the L6→L7 jump is eta_h-driven (retest L7 with the L6 eta_h
  value substituted at the L7 slot) or intrinsic to the extra level, since L8 will add another
  rung on top of whatever margin is left. **Resolved in
  [`11_02_l7_etah_isolation_retest`](../../../08/04/11_02_l7_etah_isolation_retest/report.md):
  eta_h-driven — substituting L6's `1e-12` at the L7 slot diverges by step 15 (H/H0=376.8),
  far worse than the ladder's own `1e-13`. Each level's eta_h must be earned individually; L8
  cannot be extended with a guessed value.**
- If L8 is attempted directly, treat the first checkpoint like a canary — the trend above
  suggests L8's peak H/H0 could plausibly clear the blowup-factor=100 threshold, not just creep
  toward it as L7 did toward the frozen-lam band edge.
- Consider re-running this local machine's stalled attempt to completion (or deleting the
  stale partial log) so this directory's own log matches the report instead of only going to
  step 45.
