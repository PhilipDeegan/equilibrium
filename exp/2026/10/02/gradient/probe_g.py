import numpy as np, sys
src = open("probe_q.py").read(); src = src[: src.index("ax_ = np.linspace(-np.pi, np.pi, 20")]
exec(src)
ax_ = np.linspace(-np.pi, np.pi, 20, endpoint=False) + np.pi / 20
Q = np.stack(np.meshgrid(ax_, ax_, ax_, indexing="ij"), -1).reshape(-1, 3)


def gmax(k, gg):
    vals = np.array([gr(k, q, gg)[0] for q in Q])
    i = np.argmax(vals)
    res = minimize(lambda q: -gr(k, q, gg)[0], Q[i], method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-14, maxiter=4000))
    return max(vals.max(), -res.fun), res.x


for xi in (2.0, 2.6, 3.0, 3.3, 3.6, 3.9, 5.2):
    gg = healing_g(m, xi)
    a0, q0 = gmax(np.zeros(3), gg)
    a1, _ = gmax(np.array([0.2, 0, 0]), gg)
    print(f"xi {xi}: g {gg:.4f} (g rho0 per tick): k=0 growth {a0:.2e} at |q| {np.linalg.norm(q0):.2f}; k=0.2 [100] {a1:.2e}", flush=True)
