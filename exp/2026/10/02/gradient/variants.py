"""Exact linearised growth of departures from the k = 0 background for turn variants:
  plain   : hop + mass, then turn by -g rho (current rule)
  split   : turn by -g rho / 2, hop + mass, turn by -g rho / 2
  avg     : turn by -g rho_bar, rho_bar = mean of rho over the 8 hex neighbours (after the hop)
  avg_self: turn by -g (rho + sum_h rho(x+h)) / 9"""
import numpy as np, sys
src = open("gradient.py").read(); src = src[: src.index("ax_ = np.linspace(-np.pi, np.pi, args.nq")]
sys.argv = ["x"]; exec(src)
from scipy.optimize import minimize

Y0, om = plane_state(np.zeros(3))
I8 = np.eye(8)


def M(q, gg, var):
    H = H_of(np.zeros(3), q)
    if var == "split":
        dA = rot8(-gg / 2) @ (I8 - gg * np.outer(J8 @ Y0, Y0))
        Z = rot8(-om - gg / 2) @ Y0
        dB = rot8(-gg / 2) @ (I8 - gg * np.outer(J8 @ Z, Z))
        return rot8(om + gg) @ dB @ H @ dA
    Z0 = rot8(-om) @ Y0
    s = {"plain": 1.0, "avg": np.mean([np.cos(q @ h) for h in hu]),
         "avg_self": (1 + sum(np.cos(q @ h) for h in hu)) / 9}[var]
    return rot8(om + gg) @ rot8(-gg) @ (I8 - 2 * gg * s * np.outer(J8 @ Z0, Z0)) @ H


Q0 = [np.array(p) * sg for p in ([-3.79, -1.65, -1.65], [-1.65, -3.79, -1.65], [-1.65, -1.65, -3.79],
                                   [-1.93, -3.96, 1.49]) for sg in (1, -1)]


def gmax(gg, var):
    ax_ = np.linspace(-np.pi, np.pi, 24, endpoint=False) + np.pi / 24
    Q = np.stack(np.meshgrid(ax_, ax_, ax_, indexing="ij"), -1).reshape(-1, 3)
    f = lambda q: np.abs(np.linalg.eigvals(M(q, gg, var))).max() - 1
    vals = np.array([f(q) for q in Q])
    best = vals.max()
    Q = np.concatenate([Q[np.argsort(vals)[-6:]], Q0])
    for i in range(len(Q)):
        r = minimize(lambda q: -f(q), Q[i], method="Nelder-Mead", options=dict(xatol=1e-8, fatol=1e-15, maxiter=3000))
        best = max(best, -r.fun)
    return best


for var in ("plain", "split", "avg", "avg_self"):
    print(var, " ".join(f"xi {xi}: {gmax(healing_g(m, xi), var):.2e}" for xi in (2.6, 3.9, 5.2)), flush=True)
