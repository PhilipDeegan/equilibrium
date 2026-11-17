# eta_h applied to the fixed-footprint perturbation depth scan — real, substantial damping (unlike Part 25's negligible effect on the real spheromak blowup)

## Motivation

User asked, after seeing Part 34's oscillatory perturbation depth scan: "did we have eta_h per
level? maybe it's just that this is not well calibrated sometimes." Correct observation — none of
Parts 32-34 applied any `eta_h` at all (deliberately, to isolate pure base mechanics). But the
REAL spheromak L8 blowup (the core subject of this whole investigation) was already tested with
the full calibrated ladder and found essentially unaffected (Part 25: ~0.008% effect across 13
orders of magnitude). That hadn't been checked against these new isolated perturbation tests.

**Caveat surfaced mid-investigation**: only L1-L6 of the ladder are properly re-earned/validated
(`project_odd27_amr_junction_leak`'s own tracking table). L7's re-earn sweep was left incomplete
(3 of 6 candidates ever completed) and L8 was never calibrated at all — the `1e-13, 1e-13` values
used for L7/L8 throughout this whole investigation's L8-blowup repro commands (Parts 23-28) are
carried-over placeholders, not bisected values. User's own framing: "well, the current ladder is
probably always preliminary" — apt, given the ladder has already shifted more than once across
this investigation's history (e.g. Part 12's fix moved L5 from `3e-11`→`1e-10`).

Added `--eta-h-per-level` to `refined_saha_depth_scan.py` (reuses
`step_boundary_hyper_resistivity_per_level_all`, IMEX, same convention as
`refined_neutron_decay.py`). Reran Part 34's exact fixed-footprint perturbation scan
(`--perturb-radius 0.005`, R=30, 10 steps) with the established ladder
(`0 0 1e-5 1e-7 3e-9 1e-10 1e-12 1e-13 1e-13`) applied every step.

## Results

`H/H0` at step 10, no `eta_h` (Part 34) vs. with `eta_h` (this run):

| depth | no eta_h | with eta_h | eta_h[depth] | status | reduction |
|---|---|---|---|---|---|
| 2 | 8.18 | 7.74 | 1e-5 | confirmed | ~1.06x |
| 3 | 13.02 | 13.55 | 1e-7 | confirmed | (no reduction) |
| 4 | 47.60 | 32.40 | 3e-9 | confirmed | ~1.5x |
| 5 | 86.63 | **9.41** | 1e-10 | confirmed | ~9.2x |
| 6 | 155.75 | 88.24 | 1e-12 | confirmed | ~1.8x |
| 7 | 308.82 | 122.71 | 1e-13 | **unconfirmed** | ~2.5x |
| 8 | 87.49 | 35.20 | 1e-13 | **unconfirmed** | ~2.5x |

(Depths 0-1 unaffected as expected — `eta_h[0]=eta_h[1]=0` by construction.)

## Interpretation

**`eta_h` has a real, substantial damping effect from depth 4 through 8** — including at depths 5
and 6, where the coefficient is genuinely validated, not just carried over. Depth 5 in particular
shows a ~9x reduction in the step-10 value. This directly answers the user's question: the
oscillation seen in Parts 33/34 is NOT immune to the same damping mechanism that's already
established elsewhere in this investigation — it responds to `eta_h` the way ordinary
boundary-cell high-frequency noise is expected to.

**This is a striking, important CONTRAST with Part 25's finding for the real spheromak L8
blowup**, where the full 13-orders-of-magnitude `eta_h` sweep produced only a ~0.008% change —
essentially nothing. Here, the same class of coefficients cuts growth by 1.5x-9x. `eta_h`
(hyper-resistivity) is a biharmonic term specifically targeting high-frequency, grid-scale noise —
a point/small-radius perturbation has far more high-frequency spectral content than the real
spheromak's smooth, broad (Bessel-function-shaped) field. This divergence in eta_h sensitivity is
itself informative: it suggests **the oscillation characterized in Parts 33-35 may be a genuinely
different phenomenon from the real L8 blowup**, not simply a smaller-scale version of the same
mechanism. The perturbation-based tests are likely tapping something closer to ordinary grid-scale
numerical noise (which `eta_h` is designed for and effective against), while the real blowup's
driver — already shown immune to `eta_h` at any tested magnitude — is something `eta_h`'s
biharmonic damping doesn't reach at all.

**Practical implication for the rest of this investigation**: today's perturbation-depth-scan
thread (Parts 32-35) has been genuinely informative about general AMR-junction mesh mechanics (the
null test, the oscillation's persistence with non-delta inputs, its response to `eta_h`) — but
should NOT be assumed to be directly diagnostic of the SPECIFIC mechanism behind the real
spheromak L8 blowup this whole investigation set out to find. That mechanism remains defined by
its OWN specific signature: immune to `eta_h` (Part 25), driven by `u` not `phi` (Part 28), absent
in a uniform mesh at the same resolution (Part 29), absent from a single junction even overlapping
real gradients (Part 30), and absent from pure mesh mechanics with no field content at all
(Part 32). Any future work aimed at THAT specific blowup should test directly against ITS OWN
established signature, not assume findings from the perturbation-based tests transfer.

**Not yet done**: given L7/L8's `eta_h` values are unconfirmed, it's still open whether a properly
bisected value would tame the perturbation-driven oscillation even further at those depths — but
given the mechanism-divergence finding above, that's now a lower-priority question than pursuing
the real blowup's own signature more directly (e.g. Part 3's still-unexplained
`disable_same_level_flux_at_boundary_only` lead).
