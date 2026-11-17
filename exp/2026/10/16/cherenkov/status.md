# 2026-10-16 cherenkov -- compact imposed core in the hop / triangle sector (Philip: compactness first; Cherenkov,
# barber pole = sugar-solution optical rotation)

chk.py: DEC Maxwell (7 hops, 12 triangles per cell, stars dec r 2), leapfrog dt 0.25, P 96 (side 55). Source:
gaussian magnetisation (sigma 2) on triangles moving along +z -- neutral, current exactly div-free (no charge left).
Imposed (held like a partial derivative), not self-consistent.

## Cherenkov calibration, n 1.5 (eps 2.25, c_w 0.667), T 50
v 0.95: Mach-front slope 1.000 vs cot mu 1.015 | v 0.8: 0.702 vs 0.663 (6 %) | v 0.5: no cone, field ~30x weaker
(only the switch-on transient). Threshold and cone angle right, no fitting.

## Rotating core at v = c, vacuum (eps 1), M rotating in xy at omega 0.5, T 40 (helix.py)
- Field piles up at the core (axis |B|^2 0.83 within 2 cells behind vs ~3e-3 in the wake): compact co-moving
  "pancake" near field.
- Behind: a helical wake (B_perp angle winds steadily along the axis), pitch 21.1 cells (backward-emission estimate
  2 pi (1 + v) / omega = 25.1; periodic box wraps after ~30 cells, so rough). Handedness flips exactly with the
  rotation sense (+0.2975 / -0.2976 rad per cell), everything else identical -> helicity = rotation sense.
- The wake is NOT compact: rho_rms 8.5 / 14.8 / 21.0 / 23.8 at 5 / 10 / 15 / 20 behind (radiates, fills the box).
Reading: in the linear sector an imposed compact rotating core at c gives a compact near field + a handed helical
wake that spreads. Neither the core's persistence nor a compact wake comes from the linear rule: compactness needs a
further mechanism (self-coupling / nonlinearity / context), as expected.
Not done: barber pole (chiral context: two circular components at different speeds, linear polarisation rotates).
