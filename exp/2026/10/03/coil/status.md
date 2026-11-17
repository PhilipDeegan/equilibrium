# 10-03 springs: N 4 control at a 20, resume, elongation vs N / resolution (next items 0b, 1)

`spring_cpp.py` (copy of 10-02 + `--resume state.npz` (keeps t, z_shift, skips the solid-angle seed), `--snap_root`,
default /bk/dst/2026/10/03/coil). Resume check: N 7 at t 13000 reproduces the 10-02 record exactly.
`turns_followed.py`: optional 3rd arg = snapshot root (default 10-03).

## N 7 resumed 13k -> 20k (R120, a 20, r0 55; run dir `springcpp/N7/.../rcscan/`: plots, rec_from13000.npz, run_resume13k.log (reconstructed subset, original deleted by mistake), turns_followed_N7_resume.log)
Turns along the followed centre line: 7 to t 16000, 6 at 17000, 5 at 18000-20000; elongation 1.20 (14k) -> 1.23
-> 1.36 (16k) -> 1.49 -> 1.56 (18k) -> 1.47 (20k); long axis +73 -> +55 deg (drifts off y). Same loss window and
elongation as N 8 (8 -> 6 -> 4, 1.2 -> 1.5, 13-20k), but turns go one at a time (7 -> 6 -> 5), not in pairs.
Caveat: min rho on the followed line drops to 0.06-0.08 at 16k / 18k / 19k (line grazes the cores): those counts are
less certain; 17k (0.21) and 20k (0.28) are clean.
-> "8 is not a correct winding number" is not supported: N 7 on the same tube fails the same way at the same time.

## N 4 elongation is NOT lattice-locked (saved 10-02 states; `turns_followed_N4_R110.log`, `_R150.log`)
- R110 (a 10, r0 30, true xi 3.9, 20k): 4 turns throughout; elongation 1.10-1.19 to t 17k, 1.23-1.29 at 18-20k;
  long axis rotates steadily, +85 (2k) -> -87 (10k) -> +88 (20k): two full half-turns, ~-0.00035 rad/tick.
- R150 (x 4/3, true xi 5.2, 12.3k): 4 turns; elongation 1.10-1.18 bounded; long axis rotates +67 (1.8k) -> -1
  (12.3k), slower in ticks.
So on the small torus the 2-fold mode precesses with the structure and stays bounded; on the big tube (N 6-8 at a 20)
it parks near y and grows. Whether the parking/growth is resolution-dependent needs the big tube at higher
resolution (N 8 at xi 5.2: a 26.7, r0 80, R ~167, ~25M cells).

Layout (Philip 10-03): cadence plots go in one directory per run, run name split on "_" into nested dirs
(`springcpp/N7/S1/q1/.../t13000.png`, rec.npz alongside). Scripts write that way from now on; `../nest_plots.py`
moves flat `<name>_tNNNNN.png` / `<name>_rec.npz` files into it (used for the hopf and N 7 runs; the two runs
started before the change -- N 4 ctl, N 8 xi 5.2 -- get moved when they finish).

## 0b: N 4 on the big tube (a 20, r0 40, R105, true xi 3.9, 20k; run dir `springcpp/N4/.../rcctl/`)
(first run hit the 30-min background limit at 13.5k; resumed from the 13k state.) 4 turns throughout (followed
line clean, min rho >= 0.14). Elongation oscillates with the breathing, no trend: 1.16-1.23 (1-5k), 1.09-1.16
(6-8k), 1.24-1.28 (9-11k), 1.13 (13k), 1.32-1.33 (16-17k), 1.15-1.22 (18-20k). Long axis wanders (+88, -73, +41,
+80, ..., -7 (14k), -52 (17k), -82 (18k), +86 (20k)), not parked on y. Travels +z (frame shift 133 over 20k).
-> Tube size is not the cause: on the same a 20 tube, N 4 holds with a bounded wandering 2-fold mode, N 7 / N 8 park
on y, grow past ~1.4 and lose turns from ~16k. The mode needs larger N on the tube (N 5 / N 6 not run past 13k).

## Item 2: N 4 spring (a 10, r0 30, R110) resumed 20k -> 40k (run dir `springcpp/N4/.../rc/long/`)
NOT a steady state: the strand unwinds 4 -> 2 at t ~22-24k. Centre-circle probe: 4 (20k), 2 from 22k on. Followed
line: 4 to 23k, then mostly 2 (4 at 27k, 32k, 37-38k) -- unreliable here (min rho on it 0.00-0.07, the line runs
through the core). Plots decide it: t 20000 the core winds round the tube (meridional cut: core at rho 19, tube
centre 30); t 40000 the core sits ON the tube centre (rho 30) and the loop is a flattened ring with a 2-fold z
saddle (+-10), i.e. at most 2 turns left. Elongation 1.29 -> 1.42 (21-23k) just before the loss, then steady
1.28-1.38 to 40k; long axis keeps rotating (~-180 deg per 8-9k ticks). Torus radius ~30, minor radius 7.4 -> 6.1-6.6,
D_core flat ~41k from 24k; still travelling +z (~0.013/tick).
Pattern across runs: every spring loses turns once its 2-fold elongation passes ~1.35-1.4 (N 4 small torus ~23k,
N 7 / N 8 big tube ~16k); the 2-fold mode is the common failure, N and size only set when.

## Item 4: recurrence (N 4 small torus, the only run that never touched the sponge)
- 4-turn phase vs relaxed t 4500 (`.../rc/long/recurrence_ref4500.log`, 4-fold): best-rotation rms 0.045-0.088 over
  5-13.5k (slowly tightening 0.069 -> 0.045; structure rms 0.236), jumps to ~0.10 at 14k (2-fold beats) and climbs
  to 0.119 at 20k -> leaves the near-steady 4-turn state before the 4 -> 2 unwinding.
- 2-turn phase vs t 30000 (`recurrence_late_ref30000.log`, 2-fold): returns every ~5000 ticks -- minima 0.047 (25k),
  0.036 (35k), 0.038 (40k); maxima 0.167 (27k), 0.160 (32k), 0.137 (38k); torus radius 29.3-31.5. Not decaying,
  not drifting: a periodic orbit (breathing ~5k ticks) in a rigidly turning frame. Philip: "interesting even if it
  deforms, it doesn't appear to deform any further." First candidate steady (periodic) structure.

## Sponge contact (10-03): big-tube runs breathe into the sponge
Core cells (rho < 0.25) beyond r_inner = 0.75 R: N 8 R125 (10-02) 6996 at 3k, 1468 at 16k; N 7 R120 5501 at 3k,
711 / 760 at 16k / 20k; N 4 a 20 ctl 2624 at 3k, 0 later; N 4 small torus R110: never (max core r 46 vs 83).
N 8 xi 5.2 R167 (exact 4/3 scale of the 10-02 N 8, sponge included): 9004 at t 6000, outer arcs at r 125 = r_inner.
-> big-tube turn loss / y-parking / the t 6000 "tightly packed" crescents (Philip) are sponge-affected; R167 stays a
like-for-like resolution check vs R125. Rerun with room: `springcpp/N8/.../R175/.../rcres/` (R 175, --sponge 0.14,
r_inner 150.5, 29M cells, ~24 GB, 6 threads) -- sponge not in the run name: R175 = this one.

## R175 at t 12000: 4 outer legs snapped off (Philip: "failure case")
Core cells at t 12000 form 5 connected pieces: main strand 99788 + four detached pieces in equal pairs (9332, 9332,
8644, 8644 -> 2-fold symmetric loss). Centre-circle winding 8 -> 4 by t 10000. Outer legs kept expanding (core r max
125 / 149 / 148 / 152 at t 3k / 6k / 9k / 12k vs r_inner 150): the snapped pieces are outer legs at the sponge edge.
Plot window (±127, sized from the seed) hides the outer legs -- fixed in spring_cpp.py for new runs (whole domain +
sponge circle). Reading: on the big tube the outer legs are not held by the structure at any domain size tried; they
expand to the sponge and break there.
Detached pieces at t 12000 are CLOSED RINGS (standalone tori): each is flat (sd 2.0-2.1 across, 12.6-13.9 in-plane),
with a hole at its centroid (nearest core cell 12.1-12.7 away) -> rings of radius ~18, centred at r 127-130, outer
edge at r 150-152 (sponge edge). Philip: standalone tori are interesting -- possibly part of the "pinching"
mechanism that generates dipoles. Open: do they persist / travel away once free, or does the sponge take them
(their outer edge touches it)? A pinch-off well inside the domain is the clean test.

## R175 killed at t 18000 (Philip); detached pieces tracked 12k -> 18k (`pieces.py`, `pieces_R175.log`)
Full states every 1000 ticks to t 18000 in /bk/dst (resumable with `--resume state_t18000.npz`). All pieces come in
2-fold symmetric pairs. Each pinch-off goes the same way: the gap to the main strand grows, it does not close.
- pair born 12-13k (|c| 117, 10020 cells, ring radius 20, hole 13): gap to main 13.3 -> 15.5 -> 18.0 -> 27.1 -> 33.9
  -> 41.6 (13k -> 18k); centroid moves out |c| 117 -> 134 and turns with the structure (azimuth -109 -> -148 deg);
  normal tilts outward and up (normal.radial +0.16 -> +0.32, normal_z +0.29 -> +0.42): it moves along its normal,
  outward = self-propelled ring. Size steady ~10k until its outer edge reaches the sponge (r max 150 at 16k), then 9121.
- the two pairs born before 12k sit with their outer edge in the sponge (r max 151-156): they shrink 8644 -> 5019 and
  8288 -> 4788, ring radius 18 -> 10, hole 12 -> 0.4 (closed: collapsing to a blob or below resolution) = sponge-eaten.
- t 18000: a new pair pinched off (7669, |c| 123, gap 17); main strand down to 49264 cells, r max 95.6.
-> no reattachment in this data: freshly pinched rings start ~13-17 from the strand and move away. Rings are
persistent while free (radius ~20, ~2 cells thick, size steady), only the sponge removes them.

## Neck (closest self-approach of the main strand, geodesic >= 60; `neck.py`, `neck_R175_R167.log`)
Philip 10-03: the knots tighten then loosen (R167); tighter -> would pinch off rings like R175.
R175: 34.6 (0) -> 26.6 (3k) -> 15.6 (5k) -> 11.2 (6k, rho_cyl 94) -> reconnected 6-7k (measure jumps to its cutoff;
centre-circle winding 8 -> 4 by 9k; closed rings by 12k). R167: 34.6 -> 23.4 (4k) -> 19.5 (5k) -> 18.9 (6k, at max
sponge contact) -> 21.4 -> 27.5 -> 31.5 -> 32.8 (11k), 8 turns intact. Neck sits at rho_cyl 85-96 (outer legs).
-> pinch threshold between ~11 and ~19; R167 turned back at 19. Apply neck.py to R167 12k-28k (second breath).

## R167 recurs with period ~10k (Philip 10-03: "perfect periodicity is what we're aiming for")
Resume verified (seed skipped, t 11000 line identical to the original). t 16000 ~ t 6000: core 97034 / 95898, minor
radius 19.9 / 19.7, centre rho 93.1 / 94.1, D_near 303k / 310k, 8 turns; 15k ~ 5k, 17k ~ 7k likewise; only the helix
phase differs (pattern turned). Sponge-assisted: outer legs touch the sponge each breath. Full-field rms vs t 6000:
`recurrence_R167_ref6000.log`.
Recurrence vs t 6000 (best rotation; structure rms 0.208): 5k 0.095, 7k 0.096, 11k 0.211 (anti-phase), 15k 0.141,
16k 0.096, 17k 0.096 -> one period later (16k) as close as +-1k from the reference; floor set by the 1000-tick state
cadence. Next return ~26k: save states every ~100 ticks around it to bound the true return. Run at 22k: 8 turns
(past the R125 turn-loss time scaled by length, ~19k; xi^2 scaling would put it ~25k).
Elongation (turns_followed_R167.log) vs R125 (10-02): R167 1.04 (5k), 1.05 (8k), 1.04 (10k), 1.06 (13k), 1.08 (15k),
1.07 (16k), 1.08 (17k); R125 1.05, 1.09, 1.10, 1.18 (13k), 1.25 (15k), 1.42 (16k). Much smaller even on scaled time
(R125 at 8k = 1.09 vs R167 at 10.7k / 14k ~1.05 / 1.07) -> 2-fold growth rate falls with resolution (item 1: points to
discretisation). But still creeping (+0.04 over 12k) and the long axis drifts -24 -> -82 deg, towards y, where R125 parked
and grew. Watch 18-28k.

## R167 18-27k: goes bad (Philip: "starts to go bad at 27k") -- same lattice-locked 2-fold failure as R125, later
Long axis parks on y from ~20k (-83, -86, +88, +82, +81, +86, +84 deg at 18-27k); elongation then grows 1.08 (18k) ->
1.11 (20-22k) -> 1.14-1.15 (24-26k) -> 1.19 (27k) (R125: 1.18 at 13k). Neck second breath tighter: min 16.1 at 25k vs
18.9 at 6k (neck_R167_late.log) -> orbit drifting towards the pinch threshold (11-19). Centre-circle 6 at 25-26k is a
probe artefact (followed line: 8, min rho on it 0.10-0.16). Periodic while the long axis turned; once locked on y the
lattice pumps the 2-fold mode each breath. Resolution x4/3 delays it ~2x, does not remove it -> fix at the rule level
(item 1: which hops favour y; orientation scan of the 2-fold growth rate as the anisotropy measure).
t 28000 (run end, resumable from state_t28000): 8 turns along the followed line, elongation 1.26 (axis +79 deg), neck
26.0, one piece; centre-circle probe 6 (artefact). Growth accelerating: 1.19 -> 1.26 in 1k (R125: 1.25 -> 1.42 just
before losing turns).
