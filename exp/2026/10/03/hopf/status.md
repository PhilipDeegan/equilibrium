# Clean Hopf seed (`hopf.py --clean`, m 0.8, true xi 3.9; next item 0)

Seed: beta = sin f (x + i y) / r e^{i (p - 1) phi}, alpha = cos f + i sin f z / r, f = pi / (1 + (r / a)^k): fill
profile identical for every p (t 0: 7816 cells n_z < -0.5 for p 1 and 8), only the windings along the loop change.
Read-out "loop turns" = winding of arg(n_x + i n_y) round the loop on the fill's outer shoulder (|n_z| < 0.5,
rho_cyl > fill radius), 256 phi bins (64 bins misread p 30 at t 0 as 26). States: /bk/dst/2026/10/03/hopf/<name>/.

## a 15, k 4, R80 (2.8M cells, 2000 ticks, states every 500)
- p 1: fill holds and travels -z (z -9 at t 300), then shrinks (rho_cyl 15 -> 7, t 300-600), rho min 0.009
  (t 500-600), fill gone by t 800; box near background by t 1750. Same as the 10-02 p 1.
- p 8: fill empties fast (n_z < -0.5 cells 8506 (t 50) -> 582 (t 150) -> 88 (t 200)) while the loop winding stays
  exactly 8 as long as a fill exists -- the windings do not slip off one by one; they leave as an 8-armed spiral
  wave travelling outward (t 250 plot: arg(n_xy) spirals out to the box, partial flip region pushed out to
  rho ~30, n_z ~0.3), leaving a plain density ring (rho min 0.0015-0.003, radius 15 -> 10, -z) to t ~1750.
- p 30: under-resolved at a 15 (30 turns on a ~94-cell loop = 2 rad/cell): fill gone by t ~100; not evidence.
Reading: same as the beta^p seed -- the fill-width confound was not the cause; twist along the loop is radiated
outward as spiral arms, not held on the loop.

Resolution note: windings along the loop cost p / a rad per cell on the core loop. Resolved (>= ~6 cells per turn,
<= 1 rad/cell) needs a >= p: p 30 -> a 30 (R ~140, 15M cells), p 60 -> a 60 (R ~240, ~75M cells, ~5 GB per state).
The 10-02 next item's "0.64 rad/cell" for p 60 at a 30 was turns/cell x 2pi confusion: it is 2 rad/cell.

## Stopped 10-03 (Philip: the Hopf picture was only an illustration that the windings are like fibres, a tangent)
k 2 / k 8 fill-width runs finished (logs `hopfc_p8_a15_k{2,8}.log`, not analysed); a 30 runs killed, not run.
