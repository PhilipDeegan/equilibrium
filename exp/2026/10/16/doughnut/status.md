# 2026-10-16 doughnut -- source-free toroidal pulse (Philip: packet rotating about x and y while travelling along z;
# wake from that rotation)

fd.py: Hellwarth-Nouchi flying doughnut (TE: A = curl(z f), f = 1/[(q1 + i tau)(q2 - i sigma) + rho^2], real part;
analytic field verified to solve Maxwell to 1e-7) in the hop / triangle DEC sector, P 140, leapfrog dt 0.25.
IC fixes: edge window centred on the pulse; gapped cell-internal modes removed exactly by one (1 - curlcurl / 27)
(they sit at omega^2 = 27 for every q; a sampled smooth field puts a few % there and pollutes centroid / error).
--plane check: plane wave tracks exact to 1.2 % at t 20 (dispersion phase error).

| q1 q2 | ring r0 | speed (centroid) | sim z_c t 30 | exact z_c | rho_rms t 0 -> 30 | |b - exact| t 30 | dE/E |
| 3 30 | 9.5 | 0.78-0.80 | +9.06 | +9.10 | 9.4 -> 18.6 | 17 % | 7e-5 |
| 2 60 | 11.0 | 0.91-0.92 | +12.49 | +12.89 | 10.9 -> 14.8 | 33 % | 3e-5 |
B ~80 % poloidal throughout (circulates in planes containing z = rotation about the transverse axes).
Centroid speed below c is the solution's own P/E (backward content), rising with q2 / q1; matched by the exact
solution. Spreading = the solution's own diffraction (slower for larger q2 / q1). Error grows ~linearly in t (short
pulse q1 2-3 cells: lattice dispersion phase error, larger at q1 2) -- resolution, not structure; refinement not run.
Reading: the sector carries the poloidal-rotating packet correctly (speed, shape, energy), but in the linear sector it
diffracts: the ring widens. Compactness needs an additional mechanism; this is the baseline to compare it against.

## Option 1: thresholded, saturating self-throughput (nl.py), q1 3 q2 30, th 0.3 x initial peak energy density
eps_hop = 1 + Delta s / (1 + s), s = max(0, u - u_th) / u_th (slower where the packet's own density is high).
| Delta | rho_rms t 0 / 10 / 20 / 30 | speed (t 20-30) | len_z t 30 | dE/E t 30 |
| 0 | 9.4 / 10.6 / 13.9 / 18.6 | 0.79 | 7.0 | 7e-5 |
| 1 | 9.4 / 9.9 / 12.1 / 16.2 | 0.77 | 6.7 | +0.18 |
| 4 | 9.4 / 8.9 / 10.0 / 13.5 | 0.40 | 10.0 | +0.40 |
Spreading is slowed, never stopped: as the ring widens its peak density drops below threshold (nonlinear hops
1.7e-3 -> 0.8e-3 of all), so the trap switches itself off. The trap that does act costs a large slowdown (0.40 at
Delta 4) and a core lagging behind / smearing (len_z 4 -> 10). Matches the estimate delta n >~ (lambda / w)^2 / 2 ~ O(1)
for w ~ lambda: refractive self-trapping of a wavelength-sized core requires an O(1) slowdown of the core, against
photon speed = wave speed and no energy dependence (GRB timing). CAVEAT: this explicit update does not conserve the
nonlinear energy (+18 / +40 %); a conservative scheme is needed before quoting numbers -- qualitative only.
Verdict: option 1 (threshold) does not give a compact core compatible with c; points to option 2 (core = a separate
kind of pattern, carried by the linear wake).
