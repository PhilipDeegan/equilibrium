"""10-08 vs: linear modes of step() and of the walk about its uniform background, in each one's own units, scaled so
both long-wave sound speeds are 1 (k in 1 / xi for the walk; step() has no length, so the same k axis is just k).
step(): longitudinal omega = c k (+ leapfrog lattice correction), transverse (div-free A) omega = 0 exactly.
walk (GP limit of omega = om0 + n k^2 / (6 m), M = 3 m / n): Bogoliubov omega = sqrt(c_s^2 k^2 + (k^2 / 2M)^2) for the
density / phase pair (no transverse mode: smooth vorticity is impossible, circulation only on phase-singular lines) plus
the gapped branch at ~2 om0 (the - quasi-energy branch)."""
import numpy as np

from equilibrium.substrate.walk import healing_g, rest_state

m, xi = 0.8, 3.9
n = np.sqrt(1 - m * m)
g = healing_g(m, xi, walk=True)
M = 3 * m / n
cs = np.sqrt(g / M)
_, om0 = rest_state(m)
print(f"walk m {m} xi {xi}: c_s {cs:.4f} cells/tick, M {M:.3f}, om0 {om0:.4f} rad/tick, gap 2 om0 = {2 * om0:.3f} "
      f"= {2 * om0 / (cs / xi):.0f} x (c_s / xi)")
print(" k xi   step omega/(c k)   walk Bogoliubov omega/(c_s k)   walk group v / c_s")
for kx in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0):
    k = kx / xi
    om = np.sqrt(cs**2 * k**2 + (k * k / (2 * M))**2)
    vg = (cs**2 * k + k**3 / (2 * M * M)) / om
    print(f"{kx:5.2f}   1 (transverse 0)    {om / (cs * k):6.3f}                         {vg / cs:6.3f}")
