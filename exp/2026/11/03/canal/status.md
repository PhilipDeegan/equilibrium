# 2026-11-03 canal timing: does a low-lam strip deliver ahead of the ambient front? (doc/dst_versus.md A1)

canal.py: continuous step() (u / phi, face sigma = mean of 1/(1+lam)), sphere R 40, h 1 (N 348k), lam_bg 1,
strip lam_s 0 (sigma 0.5 -> 1) along x from A (-12, 0, 0) to B (+12, 0, 0), square cross-section |y|, |z| <= w.
Gaussian u pulse at A (width s), phi 0. Probes: B (11.55 along x) and Boff (24.25 off-axis, outside the strip).
Arrival = first t with |u| >= f x its own peak; T 45 (the boundary reflection reaches B at ~t 75).
Caveat: in today's step() lam enters only through sigma; cell size (ell / L) is not wired, so "faster globally" is
built in. What the run can say: whether a NARROW strip still leads, and whether a pulse wider than it notices.

s 1.5            t(1e-2)  t(0.1)  t(0.5)  peak at B
  no strip        17.3     19.2    21.5    2.2e-2
  w 1 (121 cells) 11.4     13.9    17.4    5.3e-3     leads by 6 at 1e-2 (~1.5x ambient), weak (1/4 peak)
  w 3             8.9     10.1    20.4    2.8e-2     ~2x ambient = sigma ratio
  w 8             8.6      9.6    10.6    2.0e-2     full 2x; Boff also earlier (15.8 vs 18.0)
s 6 (pulse ~5x a w 1 strip; low thresholds contaminated by the initial tail at B, use 0.1 / 0.5)
  no strip                  7.0    12.1
  w 1                       5.6    10.6     -12 % at the 0.5 level: a pulse ~5x wider still notices a 3-cell strip
  w 3                       3.8     7.0
  w 8                       4.0     6.8
Boff unchanged for w <= 3 (strip does not leak sideways at that distance).

Reading: in the current rule a strip of lower lam is a globally faster channel down to ~3 cells across, with the
leading signal's amplitude falling with width. It is not transparent to a pulse only ~5x wider: the strip acts as a
guide carrying content forward, more than its filling fraction would suggest. For "photons do not resolve the
canal" the width ratio must be far larger (real: ~5e37 cells / wavelength); the onset of transparency vs s / w is
the next scan (needs larger D so the initial tail does not reach B: D 48, R 60, ~3 min / run).
Not addressed: Bell itself (no signalling, Tsirelson, settings); this only shows the channel exists in the rule.

Scan (14:09-14:25): D 48 (A -24, B +24), R 60 (N 1.18M), T 60 (reflection at B ~t 88), w in {none, 1, 3}, s in
{1.5, 3, 6, 12}. lead % = (t0.1 ctrl - t0.1 strip) / t0.1 ctrl; precursor = max |u_strip - u_ctrl| at B before
the control reaches 1e-2 of its peak, / control peak.
   s    w   t0.1 ctrl  t0.1 strip  lead %   precursor
  1.5   1     43.67       38.09     12.8     7.0e-2
  1.5   3     43.67       33.87     22.4     1.3
  3     1     39.87       35.95      9.8     6.9e-2
  3     3     39.87       21.47     46.1     1.3e-1
  6     1     31.65       29.44      7.0     1.8e-2
  6     3     31.65       18.27     42.3     2.8e-1
  12    1     15.11       14.02      7.2     2.8e-3
  12    3     15.11        9.11     39.7     1.7e-2
Reading: for the 3-cell strip (w 1) the strip-carried precursor falls steeply once the pulse is wider than the
strip: x0.26 from s 3 -> 6 (~s^-1.9), x0.15 from 6 -> 12 (~s^-2.7), steepening -> heading for transparency, as the
10-25 defect result implies. The ~7 % timing lead that remains at s 6-12 is the main front slightly sped by the
strip's share of the medium (mean-field), not a separate early signal. The 7-cell strip (w 3) is still a large
part of the cross-section for every s here (lead ~40 %, precursor non-monotone), so it is not in that regime.
=> in the current rule a thin low-lam channel carries its own early signal that wide pulses (light) increasingly do
not see; extrapolated to real width ratios (photon ~5e37 cells) the photon-visible part is nil. What would ride
the canal is something matched to the canal scale, not light. Still nothing about Bell itself (settings,
no signalling, Tsirelson).
