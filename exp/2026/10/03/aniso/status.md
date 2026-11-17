# 10-03 walk anisotropy (item 1; Philip: "we might have to compensate")

`symbol_aniso.py`, `compensate.py`, `compensate2.py` (symbol only, m 0.8).
- The walk X (sign +1) is NOT symmetric under a 90 deg turn about any axis: spectrum(+1, R90 k) == spectrum(-1, k) to
  7e-16 (the turn maps the walk onto its mirror, sign -1); in the xy plane (kz = 0) it looks 4-fold, the asymmetry
  needs all three k components (odd sx sy sz-type term). Relative anisotropy (frequency spread over k directions /
  kinetic part) grows ~ linearly in k: 4.4% (k 0.1), 8.8% (0.2), 13% (0.3), 22% (0.5) -- all of it 90-deg asymmetry.
  Core scale k ~ 1 / xi: ~11% at xi 3.9, ~8% at xi 5.2 -> fits R167 failing like R125 but later (falls only as 1/xi).
- Naive alternation (sign -1 tick after +1), inversion X(-k) X(k), 90-deg-rotated ticks: worse or kill propagation.
- PARITY IMAGE alternation, tick 2 = B X(-k) B (B swaps the L / R halves): kinetic part unchanged (1.248e-3 vs 1.249e-3
  at k 0.1), 90-deg asymmetry about x, y, z = 0 to 1e-13 (full cubic symmetry), residual spread purely cubic and
  ~ k^2: 0.15% (0.1), 0.59% (0.2), 1.3% (0.3), 3.7% (0.5) -- 15x smaller at the core scale. Rest state v is an
  eigenvector of both ticks with the same frequency arcsin m (W0 = sum A_h = identity) -> background, mu, seed
  unchanged. Real form, lossless; in real space tick 2 = L and R exchange their hop rules.
Implemented (Philip: "sure we should test it"): `parity=True` on Walk / CppWalk (odd t: L and R exchange A_h / A_h^T,
hop faces unchanged -- in real space B S(-k) B is just the matrix swap); C++ step_walk + set_walk_parity binding;
tst/substrate/test_walk.py: periodic plane wave == symbol (1 plain tick, plain + parity), C++ == numpy, background
stationary. spring_cpp.py --parity (name suffix _par).
Test running: N 8 R125 10-02 setup with --parity to 20k (coil/springcpp_N8_R125_par.log); compare long-axis parking /
elongation with turns_followed_N8big.log (10-02: parked ~+-85 deg, 1.18 at 13k, 8 -> 6 at 14k).

## R125 --parity result to 20k (turns_followed_N8_R125_par.log)
Elongation 1.03 (5k), 1.04 (8k, 10k), 1.02 (12k), 1.03 (13k), 1.05 (14-15k), 1.04, 1.03, 1.05 (16-18k), long axis
wandering (-81, -68, +43, -4, +61, +60, -34, +71, +53, +30): no parking, 8 turns (plain 10-02: 1.21 at 13k, 6 turns
at 14k). Core cells / minor radius equal to the plain run within ~1% (kinetic term unchanged). BUT 19k 1.10, 20k 1.14
on y (-89.6) -- same time as the first sponge contact (core r max 97.6 vs r_inner 93.7, 1656 cells; <= 92.5 before).
Resume to 30k running (coil/springcpp_N8_R125_par_resume20k.log): falls back after the breath retracts -> sponge;
keeps growing on y -> residual cubic anisotropy (then: R150 with room, or a further rotation compensation).
23k (pieces_N8_R125_par.log): 4-fold pinch -- 4 blobs (398, 398, 395, 395 cells, radius ~3.8, hole 0.4 = ~one core,
not proper rings) on +-y and +-x exactly (centroids (-+3.9, +-83.5), (+-83.4, -+3.8)), |c| 88, r max 94 = sponge edge;
main strand 25.5k cells, centre-circle 8 -> 4. Plain walk pinched 2-fold on y; parity pinches 4-fold on the cubic axes
-> residual cubic term picks the sites; sponge still co-located. R150 decides.
Charge of pinch products (coil/ring_charge.py): R175 rings (13k, 18k) all tube winding -1 = parent strand (like-sign
fission, one turn each). R125 parity 23k blobs: winding 0 at loop radius 2 / 3 / 5 (strand -1 at 5) -> NEUTRAL
density lumps (Jones-Roberts-like rarefaction pulse?); strand 8 -> 4 turns not carried by them. Philip 10-03: an
electron should not pinch at all; very maybe neutrinos form that way.
22-30k (square_lock_N8_R125_par.log): after the 23k pinch a 4-turn square-ish loop (Philip: "basically a square at
27k") ROTATES rigidly, corner angle +44.8 -> -44.5 deg over 23-30k (~-12.8 deg / 1k, uniform, not locked to the axes),
4-fold amp 0.66-0.80, core cells 25.4-26.3k flat, rho_cyl max 77-83 (inside r_inner 93.7), 2-fold amp 0. First
rotating, non-expanding 4-turn loop (after a pinch, so not an electron candidate as is). 90-deg period ~7k ticks:
resume to 40k from state_t30000 for recurrence (after R150).

## R150 --parity (sponge 112.5), killed at 15.5k (Philip), resumable from state_t15000
Outer legs to the sponge by 4k (113.9), 4-fold pinch at 5k (4 rings ~4170 cells, radius 13.9, hole 8.1, at the
sponge edge); 8 -> 9k the turns unwind: one flat plain vortex ring (tube winding -1), radius ~61, with an 8-fold
ripple (Philip: "octagon"; Kelvin-wave-like, remnant of the 8 turns). PERSISTENT 9-15k (octagon_N8_R150_par.log):
radius 60.3-61.2, 8-fold amplitude 0.711-0.716 constant, core cells 21-24k, travels +z at ~8.8 / 1k ticks, ripple
turns ~22-24 deg / 1k (direction aliased at 1k cadence). Philip: persistent structures are still interesting.
