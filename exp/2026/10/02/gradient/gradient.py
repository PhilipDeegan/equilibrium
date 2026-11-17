"""Uniform-gradient states of the walk (exact, rule-level; no envelope approximation).

A state whose pairs are all turned by k.x (pair angle advancing k per cell along k) with a uniform 8-vector Y0 = a0 v(k)
(v(k) = the positive-branch eigenvector of the hop + mass symbol at k, continued from the k = 0 rest state) is mapped
by one tick to itself, turned uniformly: the hop + mass gives e^{-i omega(k)}, the density turn -g rho0. Departures:
X(x) = Rot(k.x) [Y0 + c e^{i q.x} + cc]. The walk commutes with every pair turn, so in the turned frame the tick is
translation invariant with hops A_h Rot(k.h) (L, from x + h) and A_h^T Rot(-k.h) (R, from x - h). Linearised tick on c:

    H(q)  = [[n sum_h A_h Rot(k.h) e^{i q.h}, m J], [m J, n sum_h A_h^T Rot(-k.h) e^{-i q.h}]]
    dT    = Rot(-g rho0) (I - 2 g J Z0 Z0^T),  Z0 = H(0) Y0,  rho0 = |Z0|^2
    M(q)  = Rot(omega + g rho0) dT H(q)        (co-turning frame: Y0 fixed)

Stable: every eigenvalue of M(q) on the unit circle; |lambda| > 1 = a departure that grows by |lambda| per tick.
Scan k along [100] and [111] (+ and -), q over the zone; report the largest |lambda| - 1 and where.
Sanity: k = 0, g > 0 stable; g < 0 grows (the attracting sign).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from equilibrium.substrate.walk import A_REAL, HS, J4, healing_g, real, rest_state, symbol_dirac

here = Path(__file__).parent
p = argparse.ArgumentParser()
p.add_argument("--m", type=float, default=0.5)
p.add_argument("--xi", type=float, default=2.6)
p.add_argument("--g", type=float, default=None, help="override g (default from xi)")
p.add_argument("--nq", type=int, default=24, help="q grid per axis over the zone")
p.add_argument("--tag", default="")
args = p.parse_args()
m, n = args.m, np.sqrt(1 - args.m**2)
g = healing_g(m, args.xi) if args.g is None else args.g
hu = HS / np.sqrt(3)  # walk hops in our units (hex distance 1)
J8 = np.kron(np.eye(2), J4)


def rot4(th):
    return real(np.exp(1j * th) * np.eye(2))


def rot8(th):
    return np.kron(np.eye(2), rot4(th))


v0, _ = rest_state(m)


def plane_state(k):
    lam, V = np.linalg.eig(symbol_dirac(k, m))
    i = int(np.argmax(np.abs(V.conj().T @ v0)))
    om = -np.angle(lam[i])
    vk = V[:, i] * np.exp(-1j * np.angle(V[:, i] @ v0.conj()))
    Y0 = np.concatenate([np.stack([vk[:2].real, vk[:2].imag], -1).ravel(), np.stack([vk[2:].real, vk[2:].imag], -1).ravel()])
    return Y0 / np.linalg.norm(Y0), om


def H_of(k, q):
    HL = sum(A_REAL[h] @ rot4(k @ hu[h]) * np.exp(1j * q @ hu[h]) for h in range(8))
    HR = sum(A_REAL[h].T @ rot4(-k @ hu[h]) * np.exp(-1j * q @ hu[h]) for h in range(8))
    return np.block([[n * HL, m * J4], [m * J4, n * HR]])


def growth(k, Q):
    Y0, om = plane_state(k)
    Z0 = (H_of(k, np.zeros(3)) @ Y0).real
    assert np.allclose(Z0, rot8(-om) @ Y0, atol=1e-10), "plane state is not a hop + mass eigenstate"
    rho0 = Z0 @ Z0
    dT = rot8(-g * rho0) @ (np.eye(8) - 2 * g * np.outer(J8 @ Z0, Z0))
    C = rot8(om + g * rho0) @ dT
    best, qbest = 0.0, None
    for q in Q:
        lam = np.linalg.eigvals(C @ H_of(k, q))
        a = np.abs(lam).max() - 1
        if a > best:
            best, qbest = a, q
    return best, qbest, om


ax_ = np.linspace(-np.pi, np.pi, args.nq, endpoint=False) + np.pi / args.nq
Q = np.stack(np.meshgrid(ax_, ax_, ax_, indexing="ij"), -1).reshape(-1, 3)
print(f"m {m} g {g:.4f} (xi {args.xi}); q grid {len(Q)} points", flush=True)
for gg in (g, -g):
    g_save, g = g, gg
    a, q, _ = growth(np.zeros(3), Q)
    print(f"  sanity k = 0, g {gg:+.4f}: max |lambda| - 1 = {a:.2e}" + (f" at q {np.round(q, 2)}" if q is not None else ""))
    g = g_save
ks = np.linspace(0, 1.5, 31)
dirs = {"[100]": [1, 0, 0], "[111]": [1, 1, 1], "-[111]": [-1, -1, -1], "[110]": [1, 1, 0]}
res = {}
for lab, d in dirs.items():
    d = np.array(d, float) / np.linalg.norm(d)
    rows = []
    for kk in ks:
        a, q, om = growth(kk * d, Q)
        rows.append((kk, a, om))
    res[lab] = np.array(rows)
    first = next((r[0] for r in rows if r[1] > 1e-9), None)
    print(f"  {lab}: first k with growth > 1e-9: {first}; growth per tick at k 0.2 / 0.4 / 0.8 / 1.2: "
          + " / ".join(f"{np.interp(x, ks, res[lab][:, 1]):.2e}" for x in (0.2, 0.4, 0.8, 1.2)), flush=True)
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for lab, r in res.items():
    ax[0].semilogy(r[:, 0], np.maximum(r[:, 1], 1e-16), "o-", ms=3, label=lab)
    ax[1].plot(r[:, 0], r[:, 2], label=lab)
ax[0].set_xlabel("k (rad per cell, pair-angle gradient)")
ax[0].set_ylabel("max |lambda| - 1 per tick")
ax[0].set_title(f"growth of departures from a uniform gradient, m {m}, g {g:.4f}")
ax[0].legend()
ax[1].set_xlabel("k")
ax[1].set_ylabel("omega(k) of the plane state")
ax[1].legend()
fig.tight_layout()
fig.savefig(here / f"gradient_m{m:g}_xi{args.xi:g}{args.tag}.png", dpi=90)
np.savez(here / f"gradient_m{m:g}_xi{args.xi:g}{args.tag}.npz", ks=ks, **{k_.strip('-[]').replace(']', '') + ('m' if k_.startswith('-') else ''): v for k_, v in res.items()})
