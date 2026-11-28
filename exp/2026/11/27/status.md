# 2026-11-27

## Discussion: spin-orbit -> g-2 -> step back to rules-based QED + Bohm

- Spin-orbit: the doc puts its origin in the 8+6 face split. Philip: orientation relative to the lattice is unobservable
  (all probes, light included, live on the lattice), so "no orientation dependence is seen" is no argument against it.
  The discrete lattice has one resolution; AMR only "resolves" regions until the model converges. The coarse model may
  be wrong and may impose behavioural rules.
- g-2: measured to 0.13 ppt (Penning trap, ratio of spin precession to cyclotron frequency). Philip: no measurement is
  unbiased; to measure is to interfere (the magnetic bottle reads spin through orbital motion).
- Plan: reproduce accepted physics with a rules-based model. A particle is the whole cell, stored as cell index + 3D
  vector from the cell centre. Fixed lattice frame. Bohmian guidance. Avoid any preferred axis (Kelvin/BCC). Wave
  amplitudes are real 2-component arrows.

## bohm_free/: free Schrodinger packet on BCC + Bohm-guided particles

`bohm_free.py --n 64 --k K --sigma0 4 --t-end 40 --ens 500 --out run_kK.npz`, `analyse.py` -> `bohm_free.png`.

- Wave: real (R, I) leapfrog (Visscher) with the 14-face Laplacian (2/3 hex, 1/3 square). Norm drift ~1e-9.
- Guidance: v = J/rho. J = (1/2m) sum_k w_k c_k Im(psi* psi_k), the rule's own face fluxes; it sums exactly to the
  centroid velocity. J and rho are fit with a local quadratic over the cell + 14 neighbours and evaluated at the
  offset. Heun step; hop when the offset leaves the Voronoi cell.
- |k| ~ 0.69 along [100], [111], [321]:
  - Centre particle v / |v_g(k0)| = 0.9917-0.9919 in all three directions (isotropic to 2e-4). The 0.8% shortfall is
    the packet's k-spread on the sine-like dispersion. vs continuum k0/m: 4.6% slow (lattice dispersion at k a ~ 0.7).
  - Spreading of particle offsets along / across k = 1.482 / 1.553 at t = 40. Lattice quadratic-dispersion reference
    1.491 / 1.563; continuum 1.601. The anisotropy between along and across comes from the dispersion at this k, not
    from the lattice axes: it is the same in all three directions to 1e-3.
  - Equivariance: the ensemble's mean offset from the rho centroid spreads exactly as the Bohm map A(t) predicts
    (residual 0.004-0.025 cells). Ensemble var / rho var is constant to 1e-3 until the [100] and [321] packets touch
    the periodic boundary (t > 35), which corrupts the rho moment, not the particles.
  - Hops use both face types: [100] centre uses square faces only; off-axis particles mix hex and square.

Next: Bohmian double slit (fringes, non-crossing trajectories, Born statistics on the screen). Then smaller k a, to
show convergence to continuum; deterministic ensemble sampling in place of a seeded draw.

Code home (Philip): when this matures, eq/physical/model/bohmian (Python) and inc/eq/core/model (C++ if needed); dedupe common BCC pieces (offsets, 14-face Laplacian, walk) once duplicates appear.

## bohm_slit/: Bohmian double slit on BCC (same rule as bohm_free, vectorised particle mover)

`bohm_slit.py --t-end 300 --out run.npz` (~5 min, 1 core); `--traj-from run.npz --out run_traj.npz` re-traces the
particles that reach the screen (deterministic: identical hits); `analyse.py` -> `bohm_slit.png`.

- 192 x 192 x 2 cell index box, z-uniform (quasi-2D). Hard wall: psi = 0 on 3 BCC layers at x = 72; slits w = 4,
  d = 24. Packet lambda = 10.1, sigma x/y = 8/20. Sponge (20 cells, psi *= exp(-gamma dt (e/W)^2)) on x and y. Screen
  at x = 152 (L = 80).
- 20000 particles, deterministic stratified sampling of |psi0|^2 (inverse normal CDF on a midpoint grid, no RNG).
- Fringes: peaks at 0 and +-34, Fraunhofer lambda L / d = 33.7. The outer peaks at +-78 sit in the y sponge.
- Born: particle arrival histogram vs time-integrated flux J_x at the screen, chi2/dof = 78/76. Transmitted fraction:
  particles 0.0581 (1162), flux in the screen window 0.0599 (~1 sigma).
- No trajectory crosses the symmetry axis (0 of 20000); the exact mirror symmetry of BCC about y = yc holds.
- Transmitted particles all start in the front of the packet; which slit is fixed by the starting side.
- Caveat: 41k particle-steps (~0.2%) hit the speed clamp |v| <= 2, during reflection at t = 50-100 (standing-wave nodes
  in front of the wall, rho -> 0). The transmitted statistics agree, but the clamp is a guard, not physics. Check per
  particle whether any transmitted particle was clamped, and whether a smaller dt removes it.
- Guidance fix: the speed clamp hid a bug. Fitting J and rho separately and taking the ratio gives spurious huge
  speeds near nodes (the zeros of the two fits don't coincide). Without the clamp: 18-22 axis crossings, 20-32
  particles at a 1024-substep cap. Now: fit psi and the lattice gradient G = (1/2m) sum_k w_k c_k psi_k, and use
  v = Im(psi* G) / |psi|^2. That is exactly J/rho at cell centres (v_g kept) and has a consistent 1/distance form near
  nodes. Per-particle substeps (<= 0.05 cells each, fields linear in time) replace the clamp.
  run_g.npz (dt 0.05, 19 min): 0 axis crossings, max 28 substeps, 1158 hits, chi2/dof = 62/76, transmitted 0.0579
  vs flux 0.0599. The dt = 0.025 run was stopped: per-trajectory dt convergence near nodes is not needed for the
  statistical claims. analyse.py defaults to run_g.npz (args: run file, optional trajectory file).
