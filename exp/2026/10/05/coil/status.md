# 10-05 coil (Philip: continuous helical winding = electron, toroid image; Ampere: interior ~ N, energy ~ N^2, exterior one-turn)

Runs: spring_cpp.py (10-03 copy) N 2 / 4 / 8, a 10, r0 30, R 110 (sponge from 82.5), m 0.8, xi 3.9 walk_xi, parity,
4000 ticks, states /bk/dst/2026/10/05/coil. Windings: centre circle N, meridional 1, held to 4000 for N 2 and N 4.
N 8 (inner turn spacing 15.7 < 8 xi floor): tube collapses to minor radius ~2 by 2000, centre winding flickers
8 / 0 / 4 / 8 (probe off the core) -> not a coil after ~1000, excluded below.
(R125 N 8 attempt invalid: structure reaches the sponge, no exterior to measure.)

far_field.py (phase on shells about (0, 0, zc) vs q Omega / 2 of a plain loop, fitted rl, zl):
- t 0 (seed): plain loop rl 30.8 fits; residual falls with N (shell 66: N 2 0.047, N 4 0.016, N 8 0.002 rad) =
  helical modulation decaying ~ exp(-N d / r0). By construction (seed built from Omega).
- evolved (2000, 4000), N 2 and N 4 alike: residual 0.04-0.15 rad of an rms phase 0.08-0.25, fitted rl shrinks with
  shell radius (25 -> 21 -> 18) and zl drifts -> exterior NOT a single plain-loop field after evolution; no N
  dependence between N 2 and N 4. Likely seed-transient waves (core cells 4.6k -> 11k by t 1000), unproven.
interior_energy.py (gradient energy sum |a_j - a_i|^2 / 2 over hex pairs within 16 of the centre circle, bg-subtracted):
- t 0: N 2 1939, N 4 2547, N 8 4278 -> exact 3-point fit A + b N + c N^2: A 1503, b 175, c 21.5 (between strand
  line energy ~ N and interior flow ~ N^2). Seeds only.
- t 4000: N 2 1483, N 4 1736 (+17 %); ~25-30 % of the seed energy radiated.
Verdict: neither the one-loop exterior nor the N^2 law is established; the states are transient-dominated and only
two valid N. Needs relaxed coils (longer / gentler seed), a larger tube for N 8+, and a split of strand-core vs
interior-flow energy. All lattice regime (mem project_scale_gap_forcing_coarse).

## Big loop r0 120, a 10, R 200 (sponge from 150), 43.5M cells
N 16 (inner turn spacing 43 = 11 xi), 8000 ticks (4166 s): 16 / 1 windings throughout, minor radius ~9.5 to 3000
(log values after 3500 biased by an axis ring), round (no lattice squaring), rigid rotation ~12 deg / 1000 (16-fold
pattern period ~2000 ticks). Recurrence (recurrence.py, tube region): vs IC at best rotation 0.08-0.10 (scale 0.19);
vs relaxed t 4000: 0.077 / 0.095 / 0.125 / 0.166 at 5k / 6k / 7k / 8k (scale 0.24) -> slow monotonic shape drift,
not yet periodic; t 8000 mid-plane shows radial bars (tube cross-section flatter, or plane offset -- unresolved).
Axis anti-ring: absent at 3000, radius 12.8 at 4000 with winding +1 around its core vs the coil's -1 (opposite sense,
centre_ring_N16.log), shrunk to a blob by 5000, still small at 6000-8000. Philip: ignore the middle; fine if it cycles.
Same N / r0 as N 4 on r0 30 but no collapse -> the simple N / r0 (pitch) rule does not carry over to the big loop.
N 30 (inner spacing 23 = 5.9 xi) running.
N 30 (inner spacing 23 = 5.9 xi, below the 8 xi floor), 8000 ticks (3868 s): 30 / 1 windings, minor radius ~8.8 to
3000, axis ring again ~3.5-4k (log values biased after). Recount with the coil-only centre, 15 offset centre circles
(winding_N30_check.log): t 6000 all -30, interior clean (min rho 0.46); t 7000 -30 / -26 / -24 / -20 (min rho 0.17);
t 8000 -24 .. -12, cores on the centre line (min rho 0.00) -> tube collapse starts ~7k (small loops at similar
spacing: ~2k). Strong density excess (to 1.2) between strands on the outer side at 8k (absent in N 16).
Recurrence (tube region; scale 0.24-0.27): vs IC 0.130 / 0.144 / 0.123 / 0.209 at 2k / 4k / 6k / 8k; vs relaxed
t 4000 0.102 / 0.096 / 0.146 / 0.211 at 5k-8k -> held shape to ~6k, then the collapse.
Reading: the ~8 xi floor holds; a bigger loop / thicker tube only delays collapse (N 16 at 11 xi: none in 8k).
Next: N 30 past 8k (does it shed to a stable count, near the floor's ~22?), N 22 at the boundary.
N 30 resumed 8k -> 16k (3973 s; winding_N30_late.log, coil-only recount): turns 12-24 at 8k -> 4-6 at 10k, stable
4-6 to 16k (circles cross the strand: count is a range); meridional -1 throughout; coil cells 71k -> 50k (steady
from 11k). t 16000: one strand at r ~120 with ~4-6 large z-kinks, axis blob gone. -> from above the coil does not
stop near the floor's ~22: the collapse cascades to a nearly plain kinked ring (cf. small-loop N 4-6 end state).
N 30 end state orientation (square_orient_N30.log, 10-16k): 4 up/down turns (k 4 z-undulation, amp 5-6.6) and square
shape (k 4 radial, amp 5-6) both rotate steadily ~3 deg / 1000 ticks through 0 deg (lattice axes) without stopping
-> not lattice-locked. 4-turn square = recurring end state: small-loop N 4 stays 4, 10-03 R125 N 8 -> 4, N 30 -> 4
(r0 120); N 5 / 6 small loop collapse (counts unreadable); N 2 / 3 keep theirs. Open: does N 16 decay to 4 too; does
4 survive a xi change.
N 16 resumed 8k -> 16k: 16 / 1 throughout; coil-only recount at 12k and 16k: all 15 offset circles -16, interior
clean (min rho 0.21 / 0.24), axis ring gone (0 cells at r < 40). Recurrence vs relaxed t 4000 (best rotation, scale
0.24): 6k 0.095, 8k 0.166 (excursion), 9k-16k 0.052 / 0.054 / 0.047 / 0.041 / 0.064 / 0.048 / 0.044 / 0.053 ->
returns to within ~0.04-0.06 of the relaxed shape and stays: recurring / near-periodic state, the best so far.
Best-rotation angle mod 22.5 deg wanders (4-18 deg), so the rotation rate is not exactly constant.
