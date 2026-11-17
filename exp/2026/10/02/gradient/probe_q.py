import numpy as np, sys
src = open("gradient.py").read(); src = src[: src.index("ax_ = np.linspace")]
sys.argv = ["x"]; exec(src)
from scipy.optimize import minimize


def M_of(k, q, gg):
    Y0, om = plane_state(k)
    Z0 = (H_of(k, np.zeros(3)) @ Y0).real
    rho0 = Z0 @ Z0
    dT = rot8(-gg * rho0) @ (np.eye(8) - 2 * gg * np.outer(J8 @ Z0, Z0))
    return rot8(om + gg * rho0) @ dT @ H_of(k, q)


def gr(k, q, gg):
    lam = np.linalg.eigvals(M_of(k, q, gg))
    i = np.argmax(np.abs(lam))
    return np.abs(lam[i]) - 1, np.angle(lam[i])


ax_ = np.linspace(-np.pi, np.pi, 20, endpoint=False) + np.pi / 20
Q = np.stack(np.meshgrid(ax_, ax_, ax_, indexing="ij"), -1).reshape(-1, 3)
for kk in (0.01, 0.05, 0.2):
    k = np.array([kk, 0, 0])
    vals = np.array([gr(k, q, g)[0] for q in Q])
    i = np.argmax(vals)
    res = minimize(lambda q: -gr(k, q, g)[0], Q[i], method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-12, maxiter=4000))
    qb = res.x
    a, ph = gr(k, qb, g)
    print(f"k {kk}: grid max {vals.max():.2e} -> refined {a:.3e} at q {np.round(qb, 3)} |q| {np.linalg.norm(qb):.3f}, "
          f"eigen-phase {ph:+.4f}; growth at g/2 {gr(k, qb, g / 2)[0]:.2e}, g/4 {gr(k, qb, g / 4)[0]:.2e}, g=0 {gr(k, qb, 0)[0]:.1e}")
