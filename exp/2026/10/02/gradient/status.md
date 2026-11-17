# Uniform-gradient / background stability (exact, rule-level), 10-02

`gradient.py` (+ probe_*.py, box_q.py, variants.py, resonance.py): a uniform state with pair angle advancing k per cell
(Y0 = a0 v(k), the hop + mass eigenstate) is mapped to itself by a tick; departures c e^{iq.x} evolve by the exact
linearised tick M(q) (turned frame; see docstring). Ground truth: periodic BCC box in numpy (`periodic_check.py`, no
edge) and the C++ sphere (`lattice_check*.py`).

Result: the UNIFORM BACKGROUND ITSELF (k = 0) is unstable at grid scale for the current rule (m 0.5, turn -g rho):
| xi | g | exact growth / tick | periodic box |
|---|---|---|---|
| 2.6 | 0.0493 | 2.19e-2 | 2.11e-2 (32^3), O(1) by t ~1200 from 1e-10 |
| 3.9 | 0.0219 | 9.68e-3 (band half-width ~0.08 in q) | 32^3: none (no allowed q in the band); 64^3: 9.24e-3, O(1) by t ~2800 |
| 5.2 | 0.0123 | 5.43e-3 | not run at 64 |
Growth ~0.44 g per tick at a fixed grid-scale q (|q| ~4.45, e.g. (-3.79, -1.65, -1.65)); nonzero k changes it little
(k 0.2: +10%). The R30 sphere (sponge or not) hides it at xi >= 3.9 (band not resolved / leaks out); R72 runs are
large enough to contain it.
Mechanism (walk level): at q* two modes of the tick, seen in the background's co-turning frame, have nearly equal
phase with opposite sign structure (one carried by the departure, one by its mirror via the turn's density coupling):
1.69 / 1.84 rad at xi 2.6, 1.72 / 1.79 at xi 3.9; the turn closes the gap and they lock -> growth. Bare tick phases at
q*: +-1.27, +-2.26; the locking needs 2.26 - 1.27 ~ 2 mu (mu = arcsin m + g rho0 = the background's turn per tick).
I.e. the background feeds a pair of grid-scale departures whose combined phase matches twice its own turn.
Variants (exact, seeded search): splitting the turn half before / half after the hop: identical growth (not an
ordering artefact); turn from the 8-neighbour mean density: 3.8e-3 / ~0 / 9.5e-4 (xi 2.6 / 3.9 / 5.2); from the
9-cell mean: 4.9e-3 / 4.3e-4 / ~0 -- reduced, not removed (band search not exhaustive).
Consequence: every walk run is time-limited by its own background (from round-off: xi 2.6 ~1000-1200 ticks, xi 3.9
~2500-3000; sooner where structures seed grid-scale content). The dense xi 2.6 coils' late grid-scale state fits this.
The 10-01 ring runs (<= 1600 ticks) are inside the window. Whether the cure is a different turn (which cells count as
"local density"), the mass, or accepted as a property of the substrate: Philip's call.

## Cure search (10-02 afternoon)

- Averaged density for the turn (`variants_dense.py`, 40^3 + refinement): 8-neighbour mean 3.2e-3 / 1.4e-3 / 7.9e-4
  (xi 2.6 / 3.9 / 5.2), cell + 8: 4.9e-3 / 2.2e-3 / 1.2e-3, cell + 14 faces: 3.9e-3 / ~0 / ~0. Reduced ~7x, not removed.
- Mass scan (`mass_scan.py`, plain turn, xi fixed): growth / g ~0.5 for small m, falling to 0 at m_c; m_c 0.709-0.710
  (xi 3.9), 0.708 (xi 7.8), i.e. m_c -> 1/sqrt2 as g -> 0 (56^3 confirms 0.72-0.9 stable at xi 2.6 and 3.9).
  Walk statement: the uniform background is stable iff its rest turn per tick arcsin m > pi/4 (+ O(g)), i.e. a clock
  period < 8 ticks. m 0.5 (12 ticks) is on the unstable side. A constraint on m from the rule, not a pick.
- Ground truth, 64^3 periodic box, m 0.8: xi 3.9 flat (3e-10 -> 3e-10) over 3000 ticks (m 0.5 broke up by ~2800).
- Side finding: the walk's own small-k coefficient is omega = arcsin m + n k^2 / (6 m) (exact, checked numerically to
  1e-4 along [100], 0.4% along [111]), not 1/(6m): runs labelled xi (healing_g, c^2 = 1/3) have true core width
  sqrt(n) xi (0.93 xi at m 0.5). `healing_g(..., walk=True)` gives the true width.
