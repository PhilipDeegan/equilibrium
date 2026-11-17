# eta_h L5 fine bisection between 1e-9 and 1e-8

## Context

Follow-up to `exp/2026/08/08/20_09_eta_h_l5_resolved_calibration/`, which
found monotonic improvement 0 -> 1e-8 (step3 growth 2.64% -> 2.04%, ~23%
reduction), then saturation (1e-7/1e-8 within 0.02%). That sweep only
bracketed the knee to one order of magnitude (1e-9 -> 1e-8). This run
bisects with two log-spaced intermediate points: 3.3e-9 and 6.6e-9.

Same checkpoint (`/bk/l5_core_diameter9_cells_across_220.h5`, 23,005,337
leaves), same held ladder for L0-L4 (`[0.0, 0.0, 1e-5, 1e-7, 3e-9]`), same
protocol (3 steps, 4 threads, `dt=5.128205e-03`).

## Results

| eta5 | step1 | step2 | step3 |
|---|---|---|---|
| 0 | 1.003582 | 1.013292 | 1.026412 |
| 1e-9 | 1.003611 | 1.011893 | 1.021735 |
| **3.3e-9** | **1.003634** | **1.011474** | **1.020760** |
| **6.6e-9** | **1.003642** | **1.011330** | **1.020519** |
| 1e-8 | 1.003646 | 1.011273 | 1.020440 |
| 1e-7 | 1.003652 | 1.011163 | 1.020309 |
| 1e-6 | 1.003653 | 1.011151 | 1.020296 |

Step3 growth as a function of eta5, with successive deltas:

- 1e-9 -> 3.3e-9: 2.1735% -> 2.0760% (-0.0975pp, largest remaining step)
- 3.3e-9 -> 6.6e-9: 2.0760% -> 2.0519% (-0.0241pp)
- 6.6e-9 -> 1e-8: 2.0519% -> 2.0440% (-0.0079pp)
- 1e-8 -> 1e-7: 2.0440% -> 2.0309% (-0.0131pp, already in the flat saturated regime)

## Conclusion

The knee sits close to **eta5 ~ 3e-9** — most of the total 0->1e-8
improvement (0.60pp of the full 0.60pp step3 reduction from eta5=0 to
eta5=1e-7) is already captured by 3.3e-9, and each further doubling toward
1e-8 buys a rapidly shrinking increment (roughly halving per octave: -0.098,
-0.024, -0.008pp). This is consistent with (not a coincidence — same order
of magnitude as) the already-re-earned **L4 value of 3e-9** in the
production ladder. No new instability appeared at any bisection point (all
runs monotonic, no oscillation, no blowup) — same qualitative behavior as
the full order-of-magnitude sweep.

Practical takeaway: eta5 in the low-1e-9 to low-1e-8 range are all
reasonable choices with little to separate them at 3 steps; the previously
re-earned L4 value (3e-9) is a defensible default for L5 too if a single
shared value across levels is preferred, though it was not re-derived from
first principles here (this sweep only bisected between two 3-step
endpoints, at fixed L0-L4).

## Not done

- Still only tested at 3 steps / L5 in isolation (L0-L4 fixed) — not
  re-earned at L0-L4 against this checkpoint, not tried at longer step
  counts, not tried at L6+ (blocked on the RAM wall, see `now`).
- The remaining tiny deltas (<0.03pp) are close to the point where run-to-run
  noise/summation-order effects could matter; not checked for reproducibility
  at fixed eta5 across repeated runs.
