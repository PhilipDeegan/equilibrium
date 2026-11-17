# 10-05 halo / equilibrium (Philip: background interaction = Coulomb mechanism?; decay stays attached, diffuses;
# uncompensated wavefunction -> infinity; we start out of equilibrium)

Sponge runs (coil/radial_profile.py, R110 vs R160): l = 0 phase and density profiles stretch to the sponge -> imposed;
l = 1 (loop dipole) pulled to 0 by the sponge (R110 mismatch), close to the loop value at R160.
Periodic boxes (spring_box.py with C++ seed, N 4 a 10 r0 30, P 300 side 173 / P 450 side 260, no sponge, 8000):
- structure identical to the R160 sponge run (core, radii, helix phase): boundary-insensitive.
- seed is a 4-petal clover (projection of the N 4 helix); arms fling out by 2000 (pinwheel, wakes), settle to a rounded
  square rotating rigidly ~40 deg / 1000 ticks (pattern period ~2300; t 7000 'looks like IC' = 3 pattern periods),
  drifting +0.012 cells / tick along z.
- l = 1 steady (loop dipole). l = 0 phase: flat in r, box clock drift (mean density shift), faster in the smaller box.
  l = 0 deficit: flat box floor ~ 1 / volume.
- far gradient energy E_far: P 300 flat 345-387 from t 3000; P 450 407 -> 447 (3000 -> 8000). Far volume x3.4, E_far
  x1.15 -> mostly a fixed one-off seed shedding, plus a slow leak ~10 / 1000 ticks (P 450).
SCF by rotating-frame window averaging (scf_coil.py; k1 = nearest-cell remap, k8 = 8-cell IDW near the structure only,
exact lattice z shift elsewhere -- k1 remap noise doubled E_far on a 20 deg round trip and made diagonal streaks):
- shape converges (d_avg 0.019 -> 0.008), E_far after reseed 167 -> 77, streaks gone (k8).
- but shed per window RISES +11 -> +42 / 1000 ticks and the ring contracts (rc 30.9 -> 30.3, minor 6.9 -> 6.6):
  averaging strips the far field each cycle, the coil refills it -> the loop is a sink (like the sponge). Dropped.
- co-rotating 4-lobe density pattern at r ~ 50 in the ring plane survives the rotating-frame mean (attached).
free_after_clean.py (one cleanup = k8 avg_c0, then free 8000 ticks, P 300): E_far 104 -> 90-140 (mean ~110, drift
~ +3 / 1000) vs 345-387 uncleaned -> no regrowth: the cleaned coil is near equilibrium; 4 / 1 windings, rc 30.2-30.7,
minor 6.8 -> 6.4 (slow), drifts through the periodic box in z at 0.012 / tick.
Queued (Philip: compressed lattice = identical electrons): tight box P 180 (tube gap ~24), N 1-6, 16000 ticks: do
seeds converge to one N? Then N 16 / N 30 on r0 120 R 200 (8000).
