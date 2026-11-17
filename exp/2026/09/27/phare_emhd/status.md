# PHARE emhd_spheromak_p0 (hybrid, ion mass 1e6, unperturbed, t=0..10)

`evolution.py` reads EM_B / ions_bulkVelocity vtkhdf with h5py (pyphare's Run can't plot 3D vtkhdf;
that's why the run's own plot() failed). Plots: `emhd_spheromak_p0_plots/{history,cuts}.png`.

- Rigid drift along -z (the axis; = direction of B_ext, against the spheromak moment). J^2-centroid
  z: +0.016 at t~1.8, then -0.217 at t=10; late speed ~ -0.032 (t=6-10), still slightly accelerating.
  x, y centroid stay 0 to 1e-4. Displacement 15% of R at t=10.
- No tilt: moment angle 0.1-0.2 deg (noise level) throughout.
- Ions frozen: max|Vi| 4.1e-7 -> 4.9e-7.
- W_B -1.7%, ||J|| -11% (smoothing: nu=1e-4 hyper-resistivity + particle noise); |m| drops 18% in the
  first dump (sharp J jump at r=R smoothed at once), then ~flat.
- Force-free residual 1.1% (IC) -> 5.5%, rising steadily.
- |J| cuts: faint vertical streaks along the axis above/below (whistler wake along B), J body
  flattens toward a ring.

Why the axis direction points at physics, not the box: the exterior (uniform + dipole, box jump) is
z-mirror symmetric; only the spheromak's helicity breaks z -> -z. The EMHD equation
dB/dt = -curl(J x B)/n is even in B, so -B runs the same evolution backwards in time (drift reverses).
Tests, cheap in Hall MHD (V frozen) or hybrid:
1. flip B_phi only (helicity sign) -> drift reverses if helicity-driven;
2. flip B_ext only -> does the direction follow the external field or the moment?
3. nu -> nu/2 and 128^3: speed converges or tracks the numerical dissipation?
