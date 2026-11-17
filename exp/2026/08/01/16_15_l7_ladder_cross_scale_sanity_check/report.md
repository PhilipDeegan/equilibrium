# L7 ladder cross_scale sanity check (first cross_scale-active test)

Goal: the full provisional isolated-harness eta_h_per_level ladder
(`0 0 1e-5 1e-7 3e-9 1e-10 1e-12 1e-13`, earned entirely under `--cross-scales
0 --lam-mode frozen --damp-max 0` — see `exp/2026/07/30/20_56_l7_eta_h_sweep_remaining`
and `[[project_odd27_amr_junction_leak]]` Part 16) had never been tested with
`cross_scale` actually turned on. Part 16 flagged this as an open risk: there
was no basis to assume the isolated-harness values would survive real
cross-scale physics driving the same boundary. This run turns on the smallest
candidate value, `cross_scale=1e-3`, keeping `lam_mode=frozen` and
`damp_max=0` so only one variable changes at a time (same isolate-one-
variable methodology as the rest of the investigation).

150 steps, R=20, max-level=7, region-r0=5.0, `--blowup-factor` at its default
(100) as an early-abort guard (never triggered this run).

## Result

| cross_scale | phi_rms_0 | phi_rms_f | H_0 | H_f | H/H0 range over run | circ_r12_f |
|---|---|---|---|---|---|---|
| `1e-3` | 4.5138e-02 | 2.2491e-01 | 3.2773e+03 | 5.8462e+04 | bounded, ~[10.3, 57.6] | +7.0065e-45 (≈0) |

Step-by-step H/H0: 32.36, 33.03, 10.26, 16.96, 26.18, 14.26, 28.31, 57.56,
29.85, 17.84 (steps 15-150, every 15). No monotonic growth trend, no blowup.

**This range is essentially identical to the pure isolated-harness L7=`1e-13`
result** (`exp/2026/07/30/20_56_l7_eta_h_sweep_remaining`: bounded [10.3,
57.6]) — enabling real `cross_scale=1e-3` physics did not visibly worsen the
boundary H-growth beyond what the frozen-boundary case already showed.
`phi_rms` did grow ~5x (real field activity from the cross-scale coupling),
but `circ_r12` stayed at machine-precision zero at both ends — no ring/winding
circulation developed at this cross_scale/step count, consistent with
`lam_mode=frozen` still suppressing the mass-side dynamics.

## Finding

More encouraging than Part 16's "expect to restart from L1" framing
anticipated: at least at the smallest test value (`cross_scale=1e-3`), the
isolated-harness ladder did **not** need re-earning. This is a single data
point, not proof the ladder survives at larger `cross_scale` — `1e-3` was
deliberately the smallest of a planned sweep, not a representative "real"
value. Not yet conclusive either way for `cross_scale >= 1e-2`.

## Next

Sweep larger `cross_scale` candidates (`1e-2`, `1e-1`, ...) against this same
fixed L1-L7 ladder to find where (if anywhere) it actually breaks, before
investing in L8 (`--eta-h-ratio`) — L8 needs a validated L7 as its base, and
each ladder level costs hours regardless of outcome, so confirming the
shallower ladder's robustness across the intended cross_scale range first
avoids building L8 on a value that might still shift.

## Artefacts

- `cross_scale_1e-3.log` — full 150-step run log.
- `run.sh` — exact commands used (copy of `sh/run.sh` at time of this run).
- `refined_neutron_decay.py` — snapshot of the driver script at time of this
  experiment.
