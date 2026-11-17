# Entries for 2026-08-24 moved from repo-root `now` on 2026-10-25 (verbatim, newest first)

## 2026-08-24 (prior top entry): barrier-gate R=50 resolution check — cs=0.1 confirmed resolution-robust

| m0 | cs | R=20 sep | R=50 sep | Delta |
|---|---|---|---|---|
| 2.49 | 0.1 | 2.550 | 2.558 | +0.3% |
| 3.0 | 0.1 | 2.217 | 2.293 | +3.4% |
| 2.49 | 0.03 | 1.895 | 2.211 | +16.7% |

**Real finding**: at cs=0.1, separation is resolution-robust for both m0 values (R=20->R=50 barely moves it) — the 08-21 result was not an R=20-only artifact. cs=0.03 is NOT resolution-converged even at R=50 — see `doc/dst/resolution_requirements.txt`, REQUIREMENT 2.

**Next (queued)**: push the same 3-pair check to R=100 — cs=0.1 pairs expected to hold flat; determine whether cs=0.03's separation finally plateaus, which decides whether low-cross_scale results anywhere else in the project are trustworthy at their tested resolutions.
