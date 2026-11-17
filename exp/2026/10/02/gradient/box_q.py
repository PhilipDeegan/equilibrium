import numpy as np, sys
src = open("probe_q.py").read(); src = src[: src.index("ax_ = np.linspace(-np.pi, np.pi, 20")]
exec(src)
from scipy.optimize import minimize
for nn in (32, 64):
    qg = 2 * np.pi * np.arange(nn) / nn
    for xi in (2.6, 3.9):
        gg = healing_g(m, xi)
        best = 0
        # q_ours = sqrt3 * q_grid; scan a coarse subset of the box's allowed q plus the neighbourhood of the optimum
        res = minimize(lambda q: -gr(np.zeros(3), q, gg)[0], np.array([-3.7, -1.6, -1.6]), method="Nelder-Mead",
                       options=dict(xatol=1e-8, fatol=1e-15, maxiter=4000))
        qs = res.x / np.sqrt(3)  # in grid units
        j = np.round(qs / (2 * np.pi / nn))
        cand = []
        for d in np.stack(np.meshgrid(*(np.arange(-2, 3),) * 3, indexing="ij"), -1).reshape(-1, 3):
            q = np.sqrt(3) * 2 * np.pi * (j + d) / nn
            cand.append(gr(np.zeros(3), q, gg)[0])
        # band width: growth along a line through the optimum
        line = [gr(np.zeros(3), res.x + s * np.array([1, 0, 0]), gg)[0] for s in np.linspace(-0.3, 0.3, 61)]
        w = (np.array(line) > 0.5 * -res.fun).sum() * 0.01
        print(f"box {nn}: xi {xi}: optimum {-res.fun:.2e} at q {np.round(res.x, 4)}; best allowed box q {max(cand):.2e}; "
              f"half-max width along x ~{w:.3f} (box q spacing {np.sqrt(3) * 2 * np.pi / nn:.3f})")
