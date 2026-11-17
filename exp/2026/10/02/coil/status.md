# 8-strand poloidal coil (next 2026-10-02 item 1)

`coil.py` = copy of `exp/2026/10/01/weyl_bcc/coil.py` (only change: sys.path to import `walk` from 10/01). Defaults:
N 8, a 10, r0 30, xi 3.9, m 0.5, g 0.0219, R72 (2.03M cells), 1200 ticks, ~20 min each (two in parallel, 4 threads).
Single-loop GP speed 0.080 -> expected coil rotation 0.00267 rad/tick (294 ticks per loop spacing).

## Result: both unwind; no rigid coil rotation

| | windings | minor radius | core cells (rho < 0.25) |
|---|---|---|---|
| same sense | 1 to t 120; inner legs gone by t 180-240 | 10.1 -> 8.7 (t 120) -> 14 (t 300) -> 23 (t 540) | 14k -> 0.6k at t 540, none from t 600 |
| `--alt` | +-1 to t 120; lost at t 180 | 10.1 -> 12 -> 21-27 (t 300) -> 32 | 14k -> 0.7k at t 360, none after |

- Same sense (`*_t00300.png`): the inner legs (rho = r0 - a = 20, spacing 15.7 cells = 4 xi, half the outer spacing)
  annihilate / reconnect first; what remains are the outer arcs (meridional streak at rho ~ 40, |z| < 10), tilted in
  the xy plane, which drift outward into the sponge. The fitted rotation (-0.0019 rad/tick) comes from this debris
  and is not a coil rotation.
- `--alt`: adjacent opposite-sense loops travel toward each other and collide head-on; by t 300 they have merged into
  4 large loops (`*_t00300.png`), which then expand and leave. That is expected for the control.
- The winding probe (radius 2 xi around the outer core point) is meaningless once a loop breaks (t >= 180).

Reading: at this geometry the coil does not hold; the inner-side crowding (aspect r0 / a = 3) is the first suspect,
ahead of the rule. Not separated yet: self-induced loop motion on a curved track vs mutual interaction.

Cheap next controls (same R72): N 1 (single loop on the track: does one loop alone tilt / drift?), N 4 (inner
spacing 31 cells), and pitch / aspect variants once one of those holds.

# Spring coil (Philip 10-02: the loops above are stellarator-like; a coil is a spring, turns draped at an angle)

`spring.py`: one continuous strand on the torus helix theta_pol = N theta_tor (+ 2 pi j / S for S strands), phase =
q Omega / 2 (solid angle of the closed strand, cone to a generic apex -- with the apex at the origin, lattice cells on
the cone got Omega = +-2 pi and the seed windings came out wrong). Seed verified on R40: centre-circle winding N S,
meridional S. Same rule / xi / g / R72 as coil.py. N 8: 23.6 cells toroidal advance per turn (lean ~21 deg), N 4: 47
(~37 deg).

Result: neither keeps its turns; both end as a single plain ring travelling along +z.
- Whole structure self-propels along +z (the single toroidal circuit = net ring circulation): z ~ 10 at t 300, ~15 at
  t 600 (GP ring speed at r0 30: 0.039 cells/tick). The probes are FIXED at z = 0, so the centre-circle winding
  (lost at t 260-280 for N 4) and the growing "minor radius" are mostly that translation, not a measure of the turns.
- N 8 (`spring_N8_*_t00300/600.png`): by t 300 the outer parts of the turns are drawn out into hooks; by t 600 eight
  small rings have pinched off outward and the rest is a flat ring of radius ~28 at z ~ 15 (uniform z: no wrap left).
- N 4 (`spring_N4_*`): stays ONE closed strand throughout (no pinch-off), but flattens: z spread 0-14 at t 300, flat
  at t 600, then a flat ring with a 4-lobed wave that rotates (t 1200), still travelling.
- Helix phase arg sum w exp(i (theta_pol - N theta_tor)) advanced steadily at N 4 (~0.0072 rad/tick to t 440) while
  the turns were intact.

Reading: the poloidal winding does not survive; the toroidal circulation does (as an ordinary ring). Fewer turns ->
no reconnection, smooth unwinding instead. Caveats: no saved state (rec + plots only) and probes not co-moving -> the
turn count vs time is visual, not measured. Next if pursued: co-moving probes (centroid z_c, r_c), state saves.

# Dense (close-wound) runs: 24 lines, xi 2.6 (inner spacing 5.2 = 2 xi, outer 10.5 = 4 xi), R72, full state in
# /bk/dst/2026/10/02/coil/<name>/

- 24 closed loops (`coil_N24_*`): windings (sum over loops) 24 -> 4 by t 120, 0 by t 300; loops expand outward
  (minor radii 10 -> 22 by t 300) into the sponge, inner legs gone, centre compressed (density 2); by t 900 the whole
  domain is grid-scale noise (checkerboard in density and angle, interior depleted ~0.25): run invalid from ~t 600.
- 24-turn spring (`spring_N24_*`): same outer debris flung to the sponge by t 300, leaving a flat ring (radius ~25,
  z ~ 5) -- the toroidal circuit again; co-moving tracker confused by the debris after t 300.

Walk-level facts (Philip 10-02: no "supersonic" / "sound" -- those are emergent-layer words, reification): the seeds
force the pair angle to change by N turns along the inner circle, i.e. a gradient of N / rho rad per cell along the
tube (dense 24: 1.2 / 0.8 / 0.6 rad/cell inner / centre / outer; 8 loops: 0.4 / 0.27 / 0.2; spring N 4: 0.2 / 0.13 /
0.1). The one seed that stayed a single strand had the smallest gradient. The others lost their windings within
~100-250 ticks and their content spread into cell-to-cell variation over the whole domain. Correlation only, no
mechanism yet. (Envelope-limit reading, for reference only: GP gives a parameter-free ratio 2 pi sqrt2 xi / s for the
interior drift vs the background's collective speed -- not a statement about the rule.)

Level of description (Philip 10-02: the walk is flux moving at c periodically, not a fluid; GP is its envelope
approximation): the dense coils' interior pair-angle gradient was ~1.2 rad/cell (k h ~ 1.2), outside the envelope
regime (k h <~ 0.3), so "supersonic" is the envelope reading only. Walk-level statement: the poloidal winding forces a
phase gradient near the walk's band edge. Check: exact linear spectrum of the walk about the (flowing) background.

# N 4 spring rerun with co-moving probes (`spring_N4_*_track_*`, 1800 ticks, plots every 140, state in /bk/dst)

CORRECTS the earlier N 4 reading ("flattens, loses its turns" was the fixed-probe artefact): the strand keeps all 4
poloidal turns (centre-circle winding -4) and its toroidal circuit (meridional -1) from t 0 to t 1600, minor radius
10 -> ~8.5-9 (slowly shrinking), and goes up and down around the tube (z - zc spans +-12 at t 980). The whole spring
travels +z at ~0.027 cells/tick (zc 0 -> 44 by t 1600) while the helix pattern advances steadily (helix phase
~0.0065 rad/tick). It ends only by reaching the sponge (r ~ 50 at t ~1500: core cells fall, radius shrinks, winding
-4 -> -2 at t 1700). First coil that keeps its windings in this rule. Caveats: R72, xi 3.9 (where the background
itself is unstable at ~1%/tick on large domains -- exp/2026/10/02/gradient), one geometry, sponge-limited duration.

# N 4 spring in C++, R110 (7.2M cells), z0 -45 (`spring_cpp.py`, `springcpp_N4_*_R110_z-45_*`, 3000 ticks in 4 min)

`spring_cpp.py` = spring.py on the C++ walk; reproduces the numpy R72 track run exactly (all rec columns, t <= 280).
Spring intact to t ~1900: winding -4 / -1 throughout, minor radius 10 -> 8.0 (t 840) -> 8.5 (t 1800, steady),
helix phase advancing steadily, travels +z at 0.026 cells/tick (-45 -> +1 by t 1800), rc 30.6 -> 31-34. Grid-scale
diagonal stripes in the background visible by t 1680 (+-0.1); by t 2040-2160 the background itself has broken up
(density < 0.25 in 160k-560k cells, O(1) grid-scale field everywhere) and the spring dissolves with it. So the spring
is ended by the background instability (exp/2026/10/02/gradient), not by its own decay or the sponge.
Not run: the 4/3-scaled R150 check -- the background growth per unit of structure time is resolution independent
(rate ~0.44 g, g xi^2 = c^2 / 2m fixed), so it would end the same way; needs the background question settled first.

# N 4 spring at m 0.8 (stable-background side, `springcpp_N4_*_m0.8_R110_z-45_wxi_*`, true core width 3.9, 7000 ticks)

Background stays smooth (no grid-scale stripes at t 7000; density 0.97-1.03 away from the spring). Spring intact for
all 7000 ticks: winding -4 / -1 throughout, minor radius 10 -> 7.5 (t 2300) -> 8.0-8.2 (t 3500-7000, steady), core
cells ~12-15k steady, travels +z at 0.0113 cells/tick (-45 -> +34), helix phase advancing steadily. In structure time
(walk coefficient n/(6m): 0.125 vs 0.289 at m 0.5) 7000 ticks ~ 3000 m-0.5 ticks, beyond where the m 0.5 run died
(t ~1900-2000); speed ratio 0.0113 / 0.026 = 0.43 = the coefficient ratio. So the m 0.5 spring was ended by the
background, and on a stable background the 4-turn spring persists as a travelling, shape-keeping coil (to the end
of the run; still inside the domain). Not yet: longer runs, resolution (true xi 5.2), N 3 / 5 / 6, steady-state frame.

# Resolution check, R150 (18.4M cells), true xi 5.2, r0 40, a 13.33, z0 -60, 12440 ticks (35 min, 8 threads)

Converged with R110 in structure units: winding -4 / -1 throughout; minor radius 12.5 -> 10.1 (t 4400) -> 10.6-10.8
(late; R110 x 4/3 = 10.7-10.9); torus radius 42-43.6 (R110 x 4/3 = 42-43.3); speed 0.0085 cells/tick (R110 x 3/4 =
0.0085). The 4-turn spring on the m 0.8 background is a resolution-independent travelling structure.

Density "leak" (Philip): `budget.py` over the R110 m 0.8 snapshots: deficit within a + 4 xi breathes 46k-63k with no
trend over 7000 ticks (back to 63k at t 4500), core deficit 40-48k, core min rho ~0.001 throughout; the far field
carries the seed's shed disturbance (rms 0.5% -> 4% at t 2000-2500 -> 1-2%), and the xy panel was a FIXED z = 0 slice
the spring had left (zc +34 at t 7000). spring_cpp.py now: xy slice through zc, D_near / D_core every record,
--recentre (whole-lattice-vector shifts along z, exact; background enters) for unlimited co-moving runs.

# Long co-moving run (R110, m 0.8, true xi 3.9, 20000 ticks, --recentre 500; `rigid.py`, `shape_modes.py`)

- Windings exactly -4 / -1 in all 1001 records (to 1e-15); travels 0.01164 cells/tick (226 cells of frame shift);
  helix phase 0.002700 rad/tick (period 2327 ticks), steady over the whole run.
- Breathing of the deficit D_near: damped, period 4268 +- 6 ticks, decay time 18.5k +- 0.75k ticks, centre ~55.9k with
  a slow drift -0.19 / tick (-3.8k, ~7%, over the run) -- slow loss or fit coupling, not separated yet.
- Rigid rotation (Philip: the density stays constant while it rotates): the best rotation about z between snapshots
  equals -dhel / 4 to ~1 deg in every pair (pure turning about z, not sliding round the tube); mismatch after the
  rotation 0.066-0.080 (structure rms(rho-1) 0.22, unrotated 0.17-0.28) up to t ~13000, the residual set by the
  breathing (same-breathing-phase pair 9000 -> 13500: 0.075).
- Late (t > 12000): a 2-fold shape distortion of the core (A_2 of the toroidal distribution; A_1 = A_3 = 0 exactly by
  the lattice's 180-deg symmetry) rises in beats: 0.04 (t 4500), 0.043 (8500), 0.124 (13500), 0.189 (18000), down to
  0.02 (15500) and 0.067 (20000) in between; its axis precesses (~ -0.0008 rad/tick), not locked to lattice axes ->
  a travelling shape wave exchanging with the breathing rather than lattice forcing (not proven). The 4-fold
  amplitude A_4 stays 0.32-0.36 late. Rotated mismatch late 0.13-0.21 reflects this mode.
Reading: the 4-turn spring is a rigidly turning, travelling structure whose seed excess breathes out slowly; a
2-fold shape oscillation grows in beats late in the run. Steady state not yet reached; nothing has broken.

# N 8 spring, original torus (r0 30, a 10), m 0.8, R110, co-moving (`springcpp_N8_*_R110_*_rc_long`, stopped at t ~7000)

Turns lost: centre-circle winding 8 (t <= 500) -> 4 (t 760) -> 0 (t 1000-1700); toroidal circuit kept (meridional -1
throughout); the strand reconnects into a single ring core along the centre circle (minor radius ~2.3 from t 2000 =
the core width; centre-circle probe then sits on the core, its +-2/+-6 readings meaningless). Same fate as N 8 at
m 0.5 -> on this torus N 4 holds, N 8 unwinds into a plain ring (Philip: "just turns into a torus"). Stopped (Philip).
Discriminator running: N 8 on a doubled torus (r0 60, a 20, R125): inner turn spacing back to 31.4 (as N 4) at the
same xi -> holds = spacing; unwinds = the winding ratio itself.

# N 8 on the doubled torus (r0 60, a 20, R125, m 0.8, true xi 3.9; running): holds
To t ~11000: 8 turns + toroidal circuit held throughout (original torus lost them by t 1000) -> spacing, not the 8:1
ratio, was the cause there. Travels only ~0.002-0.003 cells/tick (ring-size scaling from N 4 predicts ~0.007: the
extra turns slow it, cf. Oberti & Ricca 2019); torus radius breathes 62 -> 71 -> 63, minor radius 18.8 -> 14.2 ->
15.7; helix ~0.0021 rad/tick. Philip: "this one is awesome".
Central dip (Philip: "spontaneous circle in the middle", t 2500; `centre_blob.py`): an on-axis density hole, min
rho 0.32 (t 2000) -> 0.013 (t 2500, ~10 cells wide) -> 0.16 (3000) -> 0.64 (4000), weak repeat 0.68-0.75 at t
6000-7000, gone (> 0.92) from t 8000. No winding round it (meridional loops: 0) -> not a vortex; the axis carries the
usual through-hole pair-angle change (0.6-0.8 turns). Formed in the spring's first contraction: focusing of what the
8 turns send inward; recurs weakly one breathing period later and fades.
Full run (20000 ticks): circle-probe "turn loss" from t 13360 was an artefact of the elongating torus. Counted along
the followed tube centre line (`turns_followed.py`): 8 turns to t 16000, 6 at 17000-18000, 4 at 19000-20000, while
the centre line elongates 1.05 (t 5000) -> 1.25 (15000) -> 1.42 (16000) -> 1.53 (20000): the 2-fold mode (as in
N 4's late beats) grows here and turns are lost in pairs at the two ends (180-deg lattice symmetry). Speed rises
0.0012 -> 0.0046 -> 0.0058 -> 0.0068 cells/tick over 5000-tick blocks; helix 0.0023 rad/tick; D_near mean 175k / 167k
/ 161k / 137k. Open: is the elongation lattice-driven (walk has only 180-deg symmetries: x and y inequivalent) or a
shape instability -> test: same seed rotated 45 deg about z.

# Winding scan at fixed turn spacing (`winding_scan.sh N threads`, `springcpp_N{3,5,6,7}_*_rcscan`, 13000 ticks)
r0 = 20 + 5 N, a 20 (inner turn spacing 31.4 = N 8 doubled torus), R = r0 + 65, m 0.8, true xi 3.9, co-moving
(--recentre 500), 2 threads each (4 jobs at once; 13000 not 20000 for time). N 4 skipped (held 20k on r0 30 a 10).
Turns counted along the followed centre line (`turns_followed_scan_N*.log`). N lobes in the xy core-cell panel =
the spring itself in projection (grey seed shows them), not a deformation.
- N 3 (R100, aspect 1.75): 3 turns throughout; wide excursion at t ~4000 (strand out to ~70 cells, near the
  sponge; centre-line follower loses the core, its "4 turns" / elongation there are artefacts), compact again by
  t 13000. Elongation numbers unreliable at this aspect.
- N 5 (R110): 5 turns throughout; elongation flat 1.08-1.13; torus radius breathes 38-53.
- N 6 (R115): 6 turns throughout; elongation 1.03-1.15, long axis near y (+-90 deg) t 4000-10000 (as N 8).
- N 7 (R120): 7 turns throughout; elongation 1.03 (t 2-4k) -> 1.10 (8-10k) -> 1.20 (13k), long axis on y
  (-73..-89 / +73..+84 deg) -- same axis and size as N 8 at 13k: odd N is not protected.
- N 8 (earlier, same spacing): 8 turns to 13-16k, elongation 1.05 (5k) -> 1.2 (13k) -> 1.5 (20k).
Speed along +z (cells/tick, t 1000-7000 / 7000-13000): N 3 0.0076 / 0.0082, N 5 0.0048 / 0.0059, N 6 0.0040 /
0.0051, N 7 0.0033 / 0.0045, N 8 0.0028 / 0.0039 -- smooth in N, no selection visible; all accelerate slightly.
Reading: no N in 3-8 is rejected within 13k ticks; no odd/even distinction. The y-locked 2-fold elongation at 13k
grows with N (N 5 1.13, N 6 1.14, N 7 1.20, N 8 1.2) -- the mode that strips N 8 is present in N 7 at the same
size, so 8 is not singled out; larger N (larger torus at fixed spacing) feels the lattice anisotropy more. Whether N 8's late loss is a
larger-N version of a lattice 2-fold mode that every N eventually meets needs 20k+ runs (spring_cpp.py has no resume yet;
states every 1000 in /bk/dst) and the resolution check (next item 1).
