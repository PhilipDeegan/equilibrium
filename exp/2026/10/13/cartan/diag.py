"""10-13 cartan: one evaluation on the vill.py whole-turn seed (--smooth (R / 9)^2, off 0.13 0.07 0.11). cartan vs lie:
antisymmetry, energy orthogonality, far / core circulation creation (farfield.py), core d(term) correlation + scale
(kappa that matches lie's vorticity transport)."""
import runpy
import sys
import time

import numpy as np

P, R0 = sys.argv[1], float(sys.argv[2])
tau = (R0 / 9) ** 2
t0 = time.time()
sys.argv = ["vill.py", "--P", P, "--R0", str(R0), "--T", "0", "--every", "1", "--smooth", f"{tau:.4f}", "--off", "0.13", "0.07", "0.11"]
g = runpy.run_path("vill.py")
print(f"seed built ({time.time() - t0:.0f} s)", flush=True)
ac, lie, cartan, W, nb, tri, x, wrapx, D, MIR = (g[k] for k in ("ac", "lie", "cartan", "W", "nb", "tri", "x", "wrapx", "D", "MIR"))
off = np.array([0.13, 0.07, 0.11])
phi = ac.phi.copy()


def G(ph):
    a = ph / W
    return np.stack([a[:, ka] + a[nb[:, ka], ks] - a[:, kb] for ka, kb, ks in tri], 1).ravel()


cen = np.stack([wrapx(x + (D[ka] + D[kb]) / 3 - off) for ka, kb, _ in tri], 1).reshape(-1, 3)
d = np.hypot(np.hypot(cen[:, 0], cen[:, 1]) - R0, cen[:, 2])
g0 = np.abs(G(phi))
rms = np.sqrt(np.sum(g0 * d**2) / g0.sum())
core, far = d < rms, d > 3 * rms
gt = {}
for name, fn in (("lie", lie), ("cartan", cartan)):
    t_ = time.time()
    T = fn(phi)
    anti = np.abs(T + T[nb, MIR[None, :]]).max()
    en = np.sum(phi * T) / np.sum(np.abs(phi * T))
    gt[name] = G(T)
    r = np.sqrt(np.sum(gt[name][far]**2)) / np.sqrt(np.sum(gt[name][core]**2))
    div = np.abs(np.sum(W * T, 1)).max() / np.abs(T).max()
    print(f"{name:7s} max|T| {np.abs(T).max():.3e} antisym {anti:.1e} sum phi.T / sum|phi T| {en:+.1e} max|div| / max|T| "
          f"{div:.2e} far / core dGamma {r:.3e} far max {np.abs(gt[name][far]).max():.2e} ({time.time() - t_:.1f} s)", flush=True)
m = d < 2 * rms
a_, b_ = gt["lie"][m], gt["cartan"][m]
print(f"core (d < 2 rms) dGamma: corr(cartan, lie) {np.dot(a_, b_) / np.sqrt(np.dot(a_, a_) * np.dot(b_, b_)):+.4f}; "
      f"lie = s * cartan, s {np.dot(a_, b_) / np.dot(b_, b_):+.4f} -> kappa_cartan = s * 0.132 = {0.132 * np.dot(a_, b_) / np.dot(b_, b_):+.4f}")
print(f"core rms {rms:.3f}; far ring |Gamma| max {g0[far].max():.2e}", flush=True)
