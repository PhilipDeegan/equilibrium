# 10-07 core: why the relaxed core is fatter (next 10-07 (B))

gp_vortex.py: exact straight q = 1 GP vortex of the walk's own small-k equation (omega = om0 + n k^2 / (6 m),
g = n / (6 m rho0 xi^2), i.e. --walk_xi; xi^2 (f'' + f'/r - f/r^2) + f - f^3 = 0), f'(0) = 0.5832 (lit. 0.5831).
Distance measured as core_profile.py does (to the rho < 0.02 core-cell set), xi 3.9:
  profile                    rho 0.5 at d (centre)   rho 0.02 at   d to core set at rho 0.5
  GP exact                   6.06                    0.95          5.11
  seed d/sqrt(d^2+xi^2)      3.90                    0.56          3.34     (logged 10-06: 3.3 -> metric checks)
  Pade d/sqrt(d^2+2 xi^2)    5.52                    0.79          4.73
  relaxed N 2 coil t 8000    --                      --            5.25     (branch/status.md)
Reading: the relaxed coil core IS the GP vortex of the walk's small-k equation (5.25 vs 5.11, 3 %; remainder:
curvature a = 10 ~ 2.6 xi, the helix's own flow, lattice dispersion beyond k^2). The --walk_xi label is right; the
seed envelope d / sqrt(d^2 + xi^2) is ~sqrt2 too narrow (it is not the GP profile; Pade with 2 xi^2 is). So the
'+ branch excess of the seed's narrow core' (~24 % of structure E, branch/status.md) = the seed's envelope error.
Clean seed without any saved state: gp_profile_xi3.9.npz (d to the strand centre, as spring_box.py --profile expects;
core_profile.py's npz was d to the core SET, offset ~0.6-0.95 -- a small inconsistency in the 10-06 clean seed).
Test in slosh/: GP-seeded N 2 r0 30 a 10 + projected, P 180 / 220 / 340 (slosh/status.md).

### 10-07 evening: same-metric comparison (core_compare.py -> core_compare_10-07.log)
GP rho sampled on the BCC lattice (nn 1) around a straight line, core = cells rho < 0.02, d = distance to the nearest
core CELL (core_profile.py's metric exactly; 12 orientations / offsets). The continuum shortcut above (r50 - r02 =
5.11) undercounts: the nearest core cell centre sits inside the rho 0.02 radius, so the lattice d is larger.
  rho level   0.2    0.5    0.8    0.9    0.95
  GP-lattice  2.75   5.43   9.79   13.49  18.13
  relaxed     2.65   5.25   9.34   12.44  16.39
-> sign flips: the relaxed coil core is slightly THINNER than GP (3 % at rho 0.5), not fatter. Inner core d < 3
matches to < 0.01; the difference is a ~+0.02 density excess over d 5-16 (relaxed fills in faster than GP's
1 - xi^2 / d^2 tail), ~+0.01 beyond. Not a single xi rescale (ratio 0.97 at rho 0.5, 0.90 at rho 0.95). Candidates
(not separated): neighbouring turns / the box's deficit average at large d (the relaxed profile bins the whole box
by distance to ANY core cell), curvature a / xi, the background level the run actually sits at (xi ~ 1 / sqrt(rho_bg)).
The --walk_xi GP vortex stays the right seed to ~3 % at all d.
