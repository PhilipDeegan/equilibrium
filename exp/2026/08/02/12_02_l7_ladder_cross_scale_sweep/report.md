# L1-L7 eta_h ladder — cross_scale sweep (1e-2, 1e-1, 1e0)

Follow-on from `exp/2026/08/01/16_15_l7_ladder_cross_scale_sanity_check` (Part 17 of
`project_odd27_amr_junction_leak`), which tested only the smallest planned cross_scale
value (1e-3) and left the rest of the sweep open.

## Parameters

Fixed provisional L1-L7 ladder, `lam_mode` still frozen, `damp_max` still 0:

```
--R 20 --max-level 7 --region-r0 5.0
--scale-override 1.3606e-3 --scale-ref-level 8
--lam-target 1.0 --lam-scale 0.05 --lam-mode frozen
--cross-scales 1e-2 1e-1 1e0 --damp-max 0
--eta-h-per-level 0 0 1e-5 1e-7 3e-9 1e-10 1e-12 1e-13
--n-steps 150 --report 15 --n-threads 4
```

## Result

All three candidates ran the full 150 steps with no blowup (`--blowup-factor 100` never
tripped). H/H0 trajectories are essentially IDENTICAL across all three cross_scale values,
and match the earlier cross_scale=1e-3 result from Part 17 (bounded ~[10.3, 57.6]):

| cross_scale | phi_rms_0 | phi_rms_f | H_0 | H_f | H/H0 peak | H/H0 @150 | circ_r12_f |
|---|---|---|---|---|---|---|---|
| 1e-2 | 4.5138e-02 | 2.2491e-01 | 3.2773e+03 | 5.8462e+04 | 57.56 | 17.84 | +4.20e-45 |
| 1e-1 | 4.5138e-02 | 2.2491e-01 | 3.2773e+03 | 5.8459e+04 | 57.56 | 17.84 | +0.00e+00 |
| 1e0  | 4.5138e-02 | 2.2486e-01 | 3.2773e+03 | 5.8434e+04 | 57.55 | 17.83 | +1.33e-44 |

`phi_rms_f`, `lam_mean_f` (0.0097), `lam_max_f` (1.0000), and `n_leaves`/`lvl_max` are
identical to 4+ significant figures across the full 100x range of cross_scale tested.
`circ_r12` stays at floating-point-noise level (~1e-44/1e-45) at both start and end for
all three — no real winding/circulation yet, consistent with `lam_mode=frozen` still
suppressing mass-side dynamics.

## Reading

The provisional L1-L7 ladder (earned under the isolated harness, Part 16) survives the
**entire planned cross_scale sweep**, not just the smallest candidate — extends Part 17's
single-data-point result to cross_scale spanning 1e-3 to 1e0 (3 orders of magnitude) with
no sign of the ladder needing re-earning at any of them.

**Caveat**: the near-total insensitivity of H to a 100x change in cross_scale (1e-2 vs 1e0)
is itself a data point worth flagging, not just a clean pass — it suggests cross_scale's
coupling doesn't move H much while `lam_mode=frozen` suppresses the mass-side dynamics it
would otherwise drive. This is consistent with Part 17's own reasoning (circ_r12 pinned at
zero), but means this sweep is a necessary, not sufficient, test — the real stress test for
the ladder is still ahead: turning `lam_mode` dynamic (or `damp_max` nonzero).

## Next steps

- Do NOT yet treat this as full validation of the ladder under "real physics" — cross_scale
  is confirmed harmless in isolation, but lam is still frozen throughout.
- Next real test: same ladder with `lam_mode` dynamic (not frozen) at a small cross_scale,
  before combining both changes at once.
- Still deferred: L8 (`--eta-h-ratio`) — per Part 17's decision, wait until the ladder is
  validated under dynamic lam too, not just cross_scale, before spending the ~7h/candidate
  cost on L8.
