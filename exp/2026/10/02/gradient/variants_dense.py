"""Dense, vectorised exact scan of departure growth from the k = 0 background for turn variants (see variants.py):
plain, avg8 (8 hex-neighbour mean density), avg9 (cell + 8 neighbours), avg15 (cell + 14 face neighbours: 8 hex + 6
square). q on an nq^3 grid over [-pi, pi]^3 (our units) + Nelder-Mead from the 12 largest."""
import sys
import numpy as np
from scipy.optimize import minimize
from equilibrium.substrate.walk import A_REAL, HS, J4, healing_g, real, rest_state

nq = int(sys.argv[1]) if len(sys.argv) > 1 else 40
m = 0.5
n = np.sqrt(1 - m * m)
hu = HS / np.sqrt(3)
sq = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]) * 2 / np.sqrt(3)
J8 = np.kron(np.eye(2), J4)


def rot8(th):
    return np.kron(np.eye(2), real(np.exp(1j * th) * np.eye(2)))


v, om = rest_state(m)
Y0 = np.concatenate([np.stack([v[:2].real, v[:2].imag], -1).ravel(), np.stack([v[2:].real, v[2:].imag], -1).ravel()])
Y0 /= np.linalg.norm(Y0)
Z0 = rot8(-om) @ Y0


def Hq(Q):
    ph = np.exp(1j * Q @ hu.T)  # (N, 8)
    HL = np.einsum("nh,hij->nij", ph, A_REAL)
    HR = np.einsum("nh,hji->nij", ph.conj(), A_REAL)
    H = np.zeros((len(Q), 8, 8), complex)
    H[:, :4, :4] = n * HL
    H[:, 4:, 4:] = n * HR
    H[:, :4, 4:] = m * J4
    H[:, 4:, :4] = m * J4
    return H


def s_of(Q, var):
    c8 = np.cos(Q @ hu.T).sum(1)
    c6 = np.cos(Q @ sq.T).sum(1)
    return {"plain": np.ones(len(Q)), "avg8": c8 / 8, "avg9": (1 + c8) / 9, "avg15": (1 + c8 + c6) / 15}[var]


def growth(Q, gg, var):
    C = rot8(om + gg) @ rot8(-gg)
    P = np.outer(J8 @ Z0, Z0)
    s = s_of(Q, var)
    M = np.einsum("ij,njk->nik", C, np.eye(8)[None] - 2 * gg * s[:, None, None] * P[None]) @ Hq(Q)
    return np.abs(np.linalg.eigvals(M)).max(1) - 1


ax_ = np.linspace(-np.pi, np.pi, nq, endpoint=False) + np.pi / nq
Q = np.stack(np.meshgrid(ax_, ax_, ax_, indexing="ij"), -1).reshape(-1, 3)
for var in ("plain", "avg8", "avg9", "avg15"):
    out = []
    for xi in (2.6, 3.9, 5.2):
        gg = healing_g(m, xi)
        a = np.concatenate([growth(Q[i:i + 20000], gg, var) for i in range(0, len(Q), 20000)])
        best = a.max()
        for i in np.argsort(a)[-12:]:
            r = minimize(lambda q: -growth(q[None], gg, var)[0], Q[i], method="Nelder-Mead",
                         options=dict(xatol=1e-9, fatol=1e-16, maxiter=3000))
            best = max(best, -r.fun)
        out.append(f"xi {xi}: {best:.2e} (grid frac > 1e-12: {(a > 1e-12).mean():.1e})")
    print(var, " | ".join(out), flush=True)
