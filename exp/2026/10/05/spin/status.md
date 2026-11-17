# 10-05 item 1: v1 / v2 relative-flow instability, threshold or box limit

counterflow.py (copy of 10-04, unchanged), P 192 box (smallest k 0.0567, half of P 96), mix 45, m 0.8 xi 3.9, 8000 ticks.
Logs: counterflow_P192_mix45.log, counterflow_P192_edge.log (+ png).

| v_rel  | cases (n1,n2)        | P 96 (k_min 0.113) | P 192 (k_min 0.0567)          |
|--------|----------------------|--------------------|-------------------------------|
| 0      | 0,0                  | flat               | flat                          |
| 0.0283 | +1,-1 ; +2,0         | -                  | flat (2.8e-7 -> 2.8e-7)       |
| 0.0425 | +3,0 ; +2,-1         | -                  | marginal: x7 / x10 in 8000 (~2.5e-4 / tick) |
| 0.0567 | +2,-2 ; +4,0         | flat (4000)        | 1.82e-3 / 1.79e-3             |
| 0.085  | +3,-3                | -                  | 2.70e-3                       |
| 0.113  | +4,-4                | 3.3e-3             | 3.19e-3                       |

- 0.057 is NOT a threshold: unstable in P 192; the 10-04 "stable" was the P 96 box not holding the unstable k.
- Edge scales with the box: unstable iff k_min < ~1.33 v_rel (P 192: ratio 1.33 marginal, 1.0 unstable, 2.0 stable;
  P 96: ratio 2.0 stable, 1.0 unstable). Equivalently unstable band k < ~Dq / 3 (v_rel = (n / 3m) Dq = 0.25 Dq).
  Consistent with ZERO threshold in an infinite medium (zero spin sound speed): any relative flow is unstable at long
  enough wavelength; a finite structure sets the threshold through its longest mode.
- One-sided = counterflow at equal relative velocity confirmed again (0.0425, 0.0567).
- Growth rate rises from the edge and levels at ~3.2e-3 ~ 0.4 g.

Consequence for item 6 (phase-carried windings on a loop of radius r0, twist q = N / r0, longest along-loop mode
k ~ 1 / r0): stable iff 1 / r0 > Dq / 3 = N / (3 r0) -> N < ~3, independent of r0. Plane-wave estimate only (a ring
core is not uniform; transverse confinement may shift the edge). Note: the N 4 small torus unwound 4 -> 2 (10-03),
a different (geometric) winding -- coincidence until checked.
Not done: the actual ring-flow profile vs this band (item 2), E2 pair relative flow (item 3).

CAVEAT (Philip 10-05 "are results impacted by the size?"): yes, by construction. In a cubic periodic box v_rel and
k_min share the side: k_min / v_rel = 4 / Dn, so every case is just "Dn relative windings per period" and both boxes
agree in Dn (2 stable, 3 marginal, 4 unstable) -- which a box artefact would also give. Only fixed-v cross-box point:
0.057 stable P 96 / unstable P 192. "Zero threshold" = extrapolation from ONE doubling; a finite threshold in
(0, 0.057) not excluded. Transverse size = axial size in both boxes (ring cores are transversely small: untested).
8000 ticks blind below ~2e-5 / tick. The N < ~3 item-6 estimate inherits all of this.
Running: counterflow.py now takes --Pz (axial only): P 192 Pz 384 (v 0.028 / 0.0425 / 0.057 at Dn 4 / 6 / 8),
P 96 Pz 192 (v 0.057 at Dn 4 with transverse halved; v 0.028 Dn 2), T 12000.

Elongated / narrowed boxes (--Pz; regress.log: P 96 Pz 96 +2,-2 = 3.25e-3, reproduces 10-04 3.27e-3):
| v_rel  | box (P, Pz)  | Dn | growth / tick                    |
|--------|--------------|----|----------------------------------|
| 0.0283 | 192, 384     | 4  | x2.8 in 12000 (<~1e-4), not the ~9e-4 a linear-in-v law predicts |
| 0.0283 | 96, 192      | 2  | x2.0 in 12000                    |
| 0.0425 | 192, 384     | 6  | 1.28e-3 (marginal at Dn 3)       |
| 0.0567 | 192, 384     | 8  | 1.81e-3 (= Dn 4 cubic 1.82e-3)   |
| 0.0567 | 96, 192      | 4  | 1.91e-3 (transverse halved)      |
- At 0.057 the rate is box-converged (Dn 4 vs 8 equal; transverse halving +5%): a velocity effect, not a Dn count.
- Resolved rates ~ linear in v: 0.030-0.032 v for 0.0425-0.085 (0.028 v at 0.113, saturating).
- 0.028 does NOT follow: at k_min / v = 1 (which gave the full rate at 0.057) it barely grows -> either a finite
  threshold in (0.028, 0.0425) or a much narrower band there. "Zero threshold" now doubtful.
Running: v 0.028 at Pz 768 (Dn 8); v 0.035 at Pz 384 (Dn 5).
All of this is lattice-regime (m 0.8, Compton ~ 1 cell; mem project_scale_gap_forcing_coarse).
Result of the last two: v 0.028 at Pz 768 (Dn 8): x2.5 in 12000 = flat. v 0.0354 at Pz 384 (Dn 5): 9.3e-4 / tick
(onset slow, 21 pts). -> finite threshold v_c ~ 0.03 (in (0.028, 0.035)), box-independent at 0.028 over Dn 2 / 4 / 8.
Resolved rate law ~ 0.031 v above it (0.035: 0.026 v, dropping toward the edge). Lattice regime (m 0.8).
