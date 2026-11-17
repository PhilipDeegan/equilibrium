# 2026-11-06 pair: box check (+-24 at n 144) and +-30 to T 600 (queue Resume 1a / 1b)

box/ = 11-04 pair.py + --restart / --t0 / --eratio / --ckpt (full state incl. E0 every --ckpt time to OUT_ckpt.npz).
Restart is bit-exact (n 48 test: T 20 straight vs T 10 + restart -> max diff 0 in P, V, u, phi, E0). Determinism: a
from-t-0 rerun of 11-04 pair30 matched its log exactly at t 200.
Saves /bk/dst/2026/11/06/box/. run.sh holds both commands and the resume recipe.

1a. pair24 / solo_m24 / solo_p24 at n 144 (= the +-30 box), T 300, launched 10:33, ~3 h on 1 core each.
    Started before --ckpt existed: no checkpoint, a kill loses them (rerun: run.sh, ~3 h + 30 min preseed).
1b. pair30 continued from /bk/dst/2026/11/04/pair/pair30.npz (t 300, E/E0 0.9345) to T 600, launched 13:09,
    checkpoint every 25, ETA ~15:35. (A first from-t-0 attempt was killed at t 200: pair30_T600_killed.log.)
    Resume: same command with --restart /bk/dst/2026/11/06/box/pair30_T600_ckpt.npz --t0 <t stored in ckpt>.
    Note: its solos only exist to T 300 (11-04); solo drift baseline beyond 300 needs solo_m30 / solo_p30 restarted too.

## Results (13:45)
1a done (all 3, ~3 h each). compare.py now takes names relative to /bk/dst/2026/ (11/06/box/pair24.npz ...).
Centroids wobble by ~+-1 in x within 10 time units (pair30 restart: +0.45..+1.43 over t 320-360), so the 11-04
single-snapshot t 300 'toward' values are wobble-dominated. Averaged over the log lines (every 10), pair - solo:
  run                    t 200-250            t 250-300
  +-24 n 128 (11-04)     toward +0.73 / side +0.26    +0.97 / +0.36
  +-24 n 144 (11-06)     toward +0.56 / side +0.33    +1.00 / +0.49
  +-30 n 144 (11-04)     toward +0.30 / side +0.17    +0.49 / +0.23
  +-30 n 144 pair only, t 300-360: x +0.88, y +0.86 (solo baseline at t 300 ~+0.2 / +0.3; no solo beyond 300)
Reading: NO box effect (n 128 vs 144 same to ~0.03 averaged) and the approach IS monotonic in separation (+-24 ~2x
+-30). The 11-04 "+-30 approaches more than +-24" was a snapshot-phase artefact -> both readings a and b of the queue
were answering a non-effect. What stays: both separations approach, slowly accelerating, no bridge by t 300 (middle
resid +-24 n 144: -0.000 at t 300; u^2 middle x71 at t 300, like +-30 x52 = content arriving, not depletion).
+-30 beyond 300: still approaching (avg x +0.49 -> +0.88), E/E0 falling faster (0.9345 t 300 -> 0.9069 t 360) and
d max rising 1.08 -> 1.62 -> something deepening; open whether that is the bridge onset.
Lesson: compare time-averaged centroids (or fit the trend), never one snapshot; compare.py should average windows.

pair30 stopped right after its t 375 checkpoint (Philip leaving 14:00); copy kept as pair30_T600_ckpt375.npz. Resume:
  cd exp/2026/11/06/box; taskset -c 0-3,8-11 python3 -u pair.py --R 8 --a 3.2 --q 4 --depth 0.8 --src local --dmax 0.9
  --dt 0.05 --n 144 --cx=-30,30 --T 600 --snap 400,500,600 --restart /bk/dst/2026/11/06/box/pair30_T600_ckpt.npz
  --t0 375 --ckpt 25 --out /bk/dst/2026/11/06/box/pair30_T600.npz >> pair30_T600.log   (~2 h for the remaining 225)
  (copy the ckpt first: the run overwrites it). Also needed for 1b: solo_m30 / solo_p30 restarted from
  /bk/dst/2026/11/04/pair/solo_*30.npz to T 600 (both E/E0 0.9338 at t 300):
  for s in m:-30 p:30; do n=${s%%:*}; x=${s#*:}; taskset -c 0-3,8-11 python3 -u pair.py <same C> --n 144 --cx=$x
    --T 600 --snap 400,500,600 --restart /bk/dst/2026/11/04/pair/solo_${n}30.npz --t0 300 --eratio 0.9338 --ckpt 25
    --out /bk/dst/2026/11/06/box/solo_${n}30_T600.npz > solo_${n}30_T600.log 2>&1 & done
  3 jobs on 4 cores, ~2-2.5 h; all checkpoint every 25.
