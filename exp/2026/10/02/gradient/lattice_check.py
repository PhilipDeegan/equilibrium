"""C++ lattice check of the k = 0 background growth: uniform background + 1e-10 random departure, R30, sponge;
interior departure norm vs t; compare the fitted rate with the exact linearised tick (probe_g.py)."""
import sys
import numpy as np
from equilibrium.substrate.walk import CppWalk, cpp_sphere, healing_g, sponge_factor

rng = np.random.default_rng(1)
for xi, pred in ((2.6, 2.19e-2), (3.9, 9.68e-3)):
    alat = cpp_sphere(30.0, n_threads=4)
    pos = np.array(alat[0].leaf_pos)
    f, r_in = sponge_factor(pos)
    w = CppWalk(alat, 0, m=0.5, g=healing_g(0.5, xi), f=f)
    w.seed(np.ones(len(pos), complex))
    L, R = w.state()
    w.set_state(L + 1e-10 * rng.normal(size=L.shape), R + 1e-10 * rng.normal(size=R.shape), t=0)
    inner = np.linalg.norm(pos, axis=1) < 0.6 * r_in
    ts, ds = [], []
    for k in range(12):
        L, R = w.state()
        gL, gR = w.bg(w.t)
        ts.append(w.t)
        ds.append(np.sqrt(np.mean(np.sum((L - gL) ** 2 + (R - gR) ** 2, 1)[inner])))
        w.step(100)
    ts, ds = np.array(ts), np.array(ds)
    sel = (ds > 1e-9) & (ds < 1e-2)
    rate = np.polyfit(ts[sel], np.log(ds[sel]), 1)[0] if sel.sum() > 2 else np.nan
    print(f"xi {xi}: departure rms " + " ".join(f"{d:.1e}" for d in ds) + f"\n   fitted growth {np.expm1(rate):.2e}/tick vs exact {pred:.2e}", flush=True)
