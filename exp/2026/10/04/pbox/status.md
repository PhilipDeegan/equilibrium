# 10-04 periodic box (item A, spacing scan) and C++ seed

## Seed cost (Philip: "why so expensive")
numpy solid angle = memory-bound temporaries (~5M segment evaluations / s / core). 10-04 first a fork pool (bit-identical
to serial, only ~2.3x at P 450: bandwidth bound), then C++: `inc/eq/core/math/solid_angle.hpp` (threaded, fused, one
call sums over translated copies), `lattice.solid_angle` / `continuous.solid_angle`, `walk.solid_angle`;
tst/substrate/test_walk.py TestSolidAngle (vs numpy 1e-11, circle on axis vs exact). Benchmark 200k points x 768
segments: numpy 10.2 s, C++ 1 thread 1.52 s, 8 threads 0.18 s (57x). P 450 seed ~30 min -> ~1.5 min.
spring_box.py here uses it; `--state_every` (disk: /bk was at 36 GB free).

## N 8 --parity, P 450 (side 260, 22.8M cells, 20k in 78 min; box_pieces_P450.log, ring_link_P450.log)
- Turns 8 -> 4 (4.5k) -> 0 (5k) on the centre-circle probe (unreliable on a tight helix, see ../octagon/status.md). At
  5k ALL 8 outer legs pinch off at |c| ~110, ~20 inside the faces (P 346: legs joined the copies across the faces, only
  4 detached). Pinch time ~4.5-5k as in R150 sphere and P 346 -> intrinsic to the a 20 spring, not the boundary.
- 6k-18k: 8 closed rings, |winding| 1, flat (sd ~1.5 thick), radius 12-13, 3.1-4.6k cells, in TWO inequivalent sets of 4
  (4-fold, not 8-fold: cubic box or residual walk term). Main ring 23-24k cells, flat, r ~61 (tight helix).
- Rings travel opposite to the main ring (rings ~-5 / 1k in z, main ~+5 / 1k), out to the faces (|c| 132-141 at 8k),
  back in (|c| 69 at 12k). 16k (Philip: "almost looks like a stellarator"): the 8 rings stand vertical (normals
  horizontal) on a circle at the main ring's radius (rho 61 / 69) but ~115 away in z = halfway to the periodic copy;
  NOT threaded (main ring never crosses a ring's plane inside its hole). Coil-like arrangement round an empty torus.
- 20k: the 8 rings fused pairwise -> 4 pieces of ~9.6k cells (> 2 x 3.3k), no longer flat (extents 11 x 11 x 6), at
  |c| 95, still ~110 from the main ring through the box. Main ring 23.7k.
Resumable: /bk/dst/2026/10/04/pbox/springbox_N8_S1_q1_a20_r60_xi3.9_m0.8_P450_z0_wxi_rc_par/state_t20000.npz.

## Tight-helix N scan (Philip 10-04: "far too few loops"; octagon = N 8 helix with a ~4.5)
r0 60, a 4.5, P 300 (side 173, 6.75M cells, gap between copies ~44), N 8 / 12 / 16 / 24 (pitch along the loop 47 /
31 / 24 / 16 cells), 10k ticks each, states every 5000: minimum stable pitch -> size needed for N ~60
(r0 = N pitch / 2 pi; 48M cells ~ 45 GB allows r0 ~130 in P 576).
Result (../octagon/ring_shape.py top harmonics = helix amplitude r_N / h_N, cells; centre-circle probe unreliable):
N 8 (pitch 47): 4.1 / 3.8 (5k), 4.4 / 3.6 (10k) steady; N 12 (31): 3.6 / 3.3, 3.3 / 3.1 ~steady (slight decline);
N 16 (24): 2.4 / 2.0, 1.6 / 1.8 shrinking; N 24 (16): 0.5 / 0.5 at 5k = helix gone (collapsed by ~2k).
-> minimum pitch along the loop ~25-31 cells ~ 7-8 xi: the SAME ~8 xi spacing as the loose springs' inner side.
Tightening does not buy turns: N 60 with per-turn zeros still needs ~60 x 31 cells of loop (r0 ~300, ~350M cells).
Internal carrier (../octagon/core_twist.py): the degenerate partner rest state v2 is never populated (|v2|^2 < 1e-4 at
the core centre, all N, 0-10k; core-centre rho ~0.004 = empty core) -> the lost windings did not move into the core as
internal twist; they are simply gone. A phase-winding (v2-filled core) structure would have to be seeded.
