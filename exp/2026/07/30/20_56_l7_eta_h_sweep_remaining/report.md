# L7 eta_h_per_level boundedness sweep (remaining candidates)

Goal: complete the L7 leg of the per-level hyper-resistivity re-earn against
the Part-12-fixed `.so` (see `[[project_odd27_amr_junction_leak]]` Part 15/16),
holding L1-L6 fixed at their already-confirmed values (`0 0 1e-5 1e-7 3e-9
1e-10 1e-12`) and bisecting only `eta_h_per_level[7]`. This run covers the
remaining candidates after an earlier baseline/low-value pass: `3e-14`,
`1e-13`, `3e-13`, `1e-12`.

150-step sweep, R=20, max-level=7, `--scale-override 1.3606e-3
--scale-ref-level 8` (matched-footprint technique, see `doc/dst_amr.md`).

## Results (H/H0)

| eta_h[7] | H/H0@15 | H/H0@150 | verdict |
|---|---|---|---|
| `3e-14` | 37.67 | 13.01 | stable, bounded [9.4, 40.5] over the full run |
| `1e-13` | 32.36 | 17.84 | stable, bounded [10.3, 57.6] — noisier than `3e-14` but no growth trend |
| `3e-13` | 35.86 | 3.58e34 | diverges — genuine exponential blowup, ~2900x/15-step compounding once it takes off |
| `1e-12` | 380.49 | 5.08e52 | diverges faster/worse than `3e-13` — already 380x by step 15 |

Cliff sits between `1e-13` (stable) and `3e-13` (diverges) — confirmed
monotonic worsening past the cliff (`1e-12` is uniformly worse than `3e-13`
at every checkpoint), so no value in bisecting further above `3e-13`.

## Finding

L7 provisional seed for the future cross-scale-active re-earn: **`1e-13`**
(or `3e-14` for extra margin — both stable, `3e-14` slightly tighter-bounded).
This, combined with L1-L6, gives the full provisional isolated-harness ladder:
`0 0 1e-5 1e-7 3e-9 1e-10 1e-12 1e-13`.

Caveat carried forward (per Part 16): this entire ladder was earned under
`--cross-scales 0 --lam-mode frozen --damp-max 0` — deliberately isolating the
AMR-boundary H-growth from real physics. See the follow-up sanity check at
`exp/2026/08/01/16_15_l7_ladder_cross_scale_sanity_check` for the first test
of this ladder with `cross_scale` turned on.

## Artefacts

- `eta_3e-14.log`, `eta_1e-13.log`, `eta_3e-13.log`, `eta_1e-12.log` — full
  150-step logs for each candidate.
- `refined_neutron_decay.py` — snapshot of the driver script at time of this
  experiment.
