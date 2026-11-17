# 2026-10-19 vcap: a finite capacity, form 1 = smooth carrier bound (Philip: systems prefer the lowest energy; offload
# because a body's capacity is finite; thermalization = equilibrating with the surroundings)

alat.set_cartan_vcap(v) / zoo.py --vcap: each triangle's Cartan terms x 1 / sqrt(1 + <a^2> / v^2) (scalar per
triangle -> energy exact; test TestCartanTransport.test_vcap). R 9 ring core max |a| 0.61 (p99.9 0.38), C0 2: vcap
below that = the ring above capacity. P 96, T 20, 4 x 2 threads (r9.sh); baseline = recv/r9/lone_P96.

| vcap | z t 20 (v) | Eu % of E t 5 / 10 / 20 | outer shell t 0 -> 20 | rms2r | sum|G| | dE/E |
| off | 2.49 (0.111) | 1.18 / 0.90 / 0.90 | 111 -> 130 | +38 % | +34 % | -3.4e-7 |
| 0.6 | 2.00 (0.097) | 0.99 / 0.76 / 0.78 | 111 -> 127 | +48 % | +49 % | -2.6e-7 |
| 0.3 | 1.38 (0.065) | 0.69 / 0.54 / 0.56 | 111 -> 122 | +53 % | +67 % | -1.9e-7 |
| 0.15 | 0.89 (0.042) | 0.33 / 0.26 / 0.27 | 111 -> 116 | +36 % | +57 % | -8.5e-8 |
| 0.075 | 0.60 (0.030) | 0.11 / 0.09 / 0.09 | 111 -> 112 | +14 % | +28 % | -2.6e-8 |
(rms2r / sum|G| off: lone_P96 at t 20.)
Reading: the opposite of offload. A bound on the carrier only slows the structure (v drops ~ with vcap) and it puts
LESS into the wave sector (Eu x0.1 at vcap 0.075) and less to the outside; nothing is shed. Same as csig: a rate is
not a store -- capping how fast the content is carried does not make any of it excess. Capacity has to bound what a
cell HOLDS (energy cost rising past a capacity: form 2, or lam as holding capacity: form 3), with an energy-consistent
coupling to the transport (Cartan is orthogonal in sum phi out, so a nonlinear store F(phi) needs the transport
re-weighted by F'(phi) to stay exact).
