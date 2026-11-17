# L7 dynamic-lam eta_h isolation retest

Follow-on to [`15_15_l7_dynamic_lam_ladder_test`](../../../08/03/15_15_l7_dynamic_lam_ladder_test/report.md)
(Part 23). That run's L6->L7 jump in H/H0 (peak 14.14->58.47, @150 4.05->18.58) was much
sharper than the L4->L5->L6 progression. Question: is the jump eta_h-driven (the ladder's L7
slot dropping to `1e-13`) or intrinsic to running one level deeper regardless of eta_h? This
retest substitutes L6's eta_h (`1e-12`) into the L7 slot, keeping everything else identical to
Part 23.

## Parameters

Identical to Part 23 except the last ladder entry:

```
--eta-h-per-level 0 0 1e-5 1e-7 3e-9 1e-10 1e-12 1e-12   (L7 slot = 1e-12, not 1e-13)
```

## Result

Diverged. `--blowup-factor 100` tripped at step 15:

```
step=    15  H=1.234990e+06  H/H0=376.8271
ABORT: H=1.2350e+06 exceeds 100.0x H0=3.2773e+05 -- stopping this cross_scale early
```

| eta_h @ L7 | H_0 | outcome | H/H0 @ step15 |
|---|---|---|---|
| 1e-13 (Part 23, the ladder value) | 3.2773e+03 | survived full 150 steps | 32.30 |
| 1e-12 (L6's value, this run) | 3.2773e+03 | **diverged, aborted at step 15** | **376.83** |

## Reading

Answered: the L6->L7 jump is **eta_h-driven, not intrinsic to the level**. `1e-12` is not a
mild degradation at L7 — it's not viable at all, diverging over 10x faster and harder than
even Part 23's least-favorable checkpoint. L6 tolerates `1e-12` fine; L7 requires the tighter
`1e-13`. This means the ladder's tightening progression (`... 1e-10 1e-12 1e-13`) is load-bearing
at every step, not conservative padding — each level's eta_h earns its own value and doesn't
carry over from the level below.

This also sharpens the L8 outlook from Part 23's "narrowing margin" framing: if the pattern
holds, L8 will need an eta_h tighter than `1e-13` at its own slot, and guessing "one more
zero" without earning it the way L0-L7 were earned is now a demonstrated way to diverge in
15 steps rather than degrade gracefully.

## Next steps

- Do not extend the ladder to L8 with a guessed eta_h value. Earn L8's eta_h the same way the
  prior levels were (per `project_odd27_amr_junction_leak`), likely starting from `1e-13` or
  tighter and sweeping down until stable, rather than assuming a fixed multiplicative step.
- The other open question from Part 23 (frozen-lam cross_scale bound [10.3, 57.6] vs L7's
  edge-hugging [9.84, 58.47]) is still unresolved by this retest — that compared different
  cross_scale values at fixed L7/frozen-lam, not eta_h at L7, and remains a separate thread.
