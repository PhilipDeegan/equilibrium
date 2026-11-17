"""10-13 cartan: per-triangle antisymmetric flux coupling K_t (alpha: hex-square edge pairs, beta: hex-hex pair) for
da_e = -sum_{t ni e} omega_t (K_t v)_e. Fit alpha / beta so the summed edge change matches i_u omega = (Omega x u) . D_e
for uniform Omega, u (v = W^2 u . D). Antisymmetric K -> sum v da = 0 exactly; local in omega by construction."""
import numpy as np
from equilibrium.substrate import continuous as C

D, W = C.D_HAT, C.W
key = lambda v: tuple(np.round(v, 6))
DI = {key(D[k]): k for k in range(14)}
eye = np.eye(3)


def edge(E):
    return DI.get(key(E))


def maps(e):
    """bilinear maps Omega^T M u for the alpha and beta parts of da_e (without -lambda)."""
    Ma, Mb, nt = np.zeros((3, 3)), np.zeros((3, 3)), 0
    for k in range(14):
        p = D[k]
        if edge(p - D[e]) is None:
            continue
        nt += 1
        E = [D[e], p - D[e], -p]
        ks = [e, edge(E[1]), edge(E[2])]
        S = 0.5 * np.cross(D[e], p)
        hexes = [k_ < 8 for k_ in ks]
        for j in (1, 2):
            sgn = 1.0 if j == 1 else -1.0
            pair_hh = hexes[0] and hexes[j]
            # Omega^T (S (W_j^2 E_j)^T) u = (Omega . S)(W_j^2 E_j . u)
            m = sgn * np.outer(S, W[ks[j]]**2 * E[j])
            if pair_hh:
                Mb += m
            else:
                Ma += m
    return Ma, Mb, nt


def target(e):
    return np.array([[np.dot(np.cross(eye[i], eye[j]), D[e]) for j in range(3)] for i in range(3)])


A = np.array([np.concatenate([maps(e)[0].ravel(), ]) for e in range(14)]).ravel()
B = np.array([maps(e)[1].ravel() for e in range(14)]).ravel()
T = np.array([target(e).ravel() for e in range(14)]).ravel()
print("triangles per edge:", [maps(e)[2] for e in range(14)])
X = np.stack([A, B], 1)
c, *_ = np.linalg.lstsq(X, T, rcond=None)
res = np.linalg.norm(X @ c - T) / np.linalg.norm(T)
print(f"alpha {c[0]:+.6f} beta {c[1]:+.6f} (beta / alpha {c[1] / c[0]:+.6f}) rel resid {res:.2e}")
for ab in ((1, 1), (1, 0), (0, 1)):
    s = np.dot(X @ ab, T) / np.dot(X @ ab, X @ ab)
    print(f"alpha:beta {ab}: best scale {s:+.4f} rel resid {np.linalg.norm(s * X @ ab - T) / np.linalg.norm(T):.2e}")
