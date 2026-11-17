"""10-12 villain: fast fail for the whole-turn ring widening. One commutator evaluation on the smoothed whole-turn seed
(vill.py --smooth, shape scaled with R: tau = (R / 9)^2). Spurious circulation creation in the vortex-free far field
(d > 3 core rms, d = distance from the core circle) relative to the core's own transport (d < 1 core rms):
ratio = sqrt(sum_far term^2) / sqrt(sum_core term^2), triangle loop sums of kappa * term. Resolution problem -> ratio
falls with R (~1 / f^2); structural -> flat."""
import sys
import runpy
import time

import numpy as np

P, R0 = sys.argv[1], float(sys.argv[2])
tau = (R0 / 9) ** 2
t0 = time.time()
sys.argv = ["vill.py", "--P", P, "--R0", str(R0), "--T", "0", "--every", "1", "--smooth", f"{tau:.4f}", "--off", "0.13", "0.07", "0.11"]
g = runpy.run_path("vill.py")
print(f"seed built ({time.time() - t0:.0f} s)", flush=True)
ac, lie, W, nb, tri, x, wrapx, D = g["ac"], g["lie"], g["W"], g["nb"], g["tri"], g["x"], g["wrapx"], g["D"]
off = np.array([0.13, 0.07, 0.11])
phi = ac.phi.copy()


def G(ph):
    a = ph / W
    return np.stack([a[:, ka] + a[nb[:, ka], ks] - a[:, kb] for ka, kb, ks in tri], 1).ravel()


cen = np.stack([wrapx(x + (D[ka] + D[kb]) / 3 - off) for ka, kb, _ in tri], 1).reshape(-1, 3)
d = np.hypot(np.hypot(cen[:, 0], cen[:, 1]) - R0, cen[:, 2])
g0 = np.abs(G(phi))
rms = np.sqrt(np.sum(g0 * d**2) / g0.sum())
gt = 0.132 * G(lie(phi))
core, far = d < rms, d > 3 * rms
r = np.sqrt(np.sum(gt[far]**2)) / np.sqrt(np.sum(gt[core]**2))
print(f"R0 {R0} P {P} tau {tau:.3f}: core rms {rms:.3f} (rms / R {rms / R0:.4f}); far / core term ratio {r:.4e}; "
      f"far ring |Gamma| max {g0[far].max():.2e}; far term max {np.abs(gt[far]).max():.2e}  ({time.time() - t0:.0f} s)", flush=True)
