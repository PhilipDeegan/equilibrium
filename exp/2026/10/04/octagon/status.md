# 10-04 item B: the R150 --parity "octagon" ring (exp/2026/10/03 R150 par, states 9-15k)

`ring_shape.py` (core rho < 0.25, SVD plane, r(phi) / h(phi) in 128 azimuth bins, Fourier amplitudes in cells):
t 10000: r 60.2, r_8 4.54, h_8 4.33, 8-fold phase r -9.8 / h -20.5 deg (phi); t 15000: r 60.1, r_8 4.78, h_8 4.34,
phase r -6.4 / h +4.9. All other harmonics 0 except a 4-fold ~0.2 (residual cubic). r and h 8-fold parts are 90 deg
apart in 8 phi -> a HELIX, not a corrugation: the core winds 8 times round a circle of radius ~4.5 (~1.2 xi) about
the r 60 centre line = an N 8 spring with a ~4.5 (seed a 20). The 8 turns did not unwind 8 -> 9k, they tightened.
Next: seed it directly (spring_cpp.py --N 8 --a 4.5 --r0 60 --R 150 --parity) and check a / radius / ripple persist.
