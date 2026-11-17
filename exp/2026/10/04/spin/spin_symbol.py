"""Item B, step 1: exact linearization (Bogoliubov on the walk symbol, parity two-tick) of the mixed rest vacuum
psi0 = a0 (cos b v1 + sin b v2). Tick: psi -> exp(-i g |S psi|^2) S psi. In the frame turning with the background
(U~ = e^{i om0} S), delta = u e^{ikx} + w* e^{-ikx}:
    p = U~(k) u, r = U~(-k)* w, c = psi0^H p + psi0^T r
    u' = p - i g psi0 c,  w' = r + i g psi0* c
Two-tick Floquet M = M_odd M_even; multipliers |lam| > 1 = growth. Branches labelled by overlap with the density
direction (psi0) vs the orthogonal rest direction (spin).
"""
import argparse

import numpy as np

from equilibrium.substrate.walk import healing_g, symbol_dirac

p = argparse.ArgumentParser()
p.add_argument("--m", type=float, default=0.8)
p.add_argument("--xi", type=float, default=3.9)
p.add_argument("--mix", type=float, default=45.0)
args = p.parse_args()
m, g = args.m, healing_g(args.m, args.xi, 1.0, walk=True)
B = np.block([[np.zeros((2, 2)), np.eye(2)], [np.eye(2), np.zeros((2, 2))]])


def S(k, odd):
    return B @ symbol_dirac(-np.asarray(k), m) @ B if odd else symbol_dirac(k, m)


lam, V = np.linalg.eig(S(np.zeros(3), 1) @ S(np.zeros(3), 0))
om = -np.angle(lam) / 2
pf = np.argsort(-om)[:2]
om0 = om[pf[0]]
v1, v2 = V[:, pf[0]], V[:, pf[1]]
v2 = v2 - (v1.conj() @ v2) * v1
v2 /= np.linalg.norm(v2)
print(f"m {m} g {g:.5f} om0 {om0:.5f} (arcsin m {np.arcsin(m):.5f}); rest doublet om {om[pf]}")
for odd in (0, 1):
    for v in (v1, v2):
        print(f"  tick {'odd ' if odd else 'even'}: |e^(i om0) S(0) v - v| = {np.linalg.norm(np.exp(1j * om0) * S(np.zeros(3), odd) @ v - v):.2e}")
b = np.radians(args.mix)
psi0 = np.cos(b) * v1 + np.sin(b) * v2
perp = -np.sin(b) * v1 + np.cos(b) * v2


def M1(k, odd):
    Up = np.exp(1j * om0) * S(k, odd)
    Um = (np.exp(1j * om0) * S(-np.asarray(k), odd)).conj()
    P = np.outer(psi0, psi0.conj())
    Q = np.outer(psi0, psi0)
    N = np.block([[np.eye(4) - 1j * g * P, -1j * g * Q], [1j * g * Q.conj(), np.eye(4) + 1j * g * P.conj()]])
    return N @ np.block([[Up, np.zeros((4, 4))], [np.zeros((4, 4)), Um]])


print(f"\nmix {args.mix} deg; k along z (also x, [111] max |lam|)")
print(f"{'k':>6} {'k xi':>6} {'max|lam|-1':>11} {'max x':>10} {'max 111':>10}  low branches (om per tick): density-like / spin-like")
for kk in [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5]:
    res = []
    for d in ([0, 0, 1], [1, 0, 0], np.ones(3) / np.sqrt(3)):
        Mk = M1(kk * np.array(d), 1) @ M1(kk * np.array(d), 0)
        res.append((np.linalg.eigvals(Mk), Mk))
    lz, Mz = res[0]
    ev, W = np.linalg.eig(Mz)
    w_ = -np.angle(ev) / 2
    dens = np.abs(psi0.conj() @ W[:4]) ** 2 + np.abs(psi0 @ W[4:]) ** 2
    spin = np.abs(perp.conj() @ W[:4]) ** 2 + np.abs(perp @ W[4:]) ** 2
    lo = np.argsort(np.abs(w_))
    di = max(lo[:4], key=lambda i: dens[i] / np.linalg.norm(W[:, i]) ** 2)
    si = max(lo[:4], key=lambda i: spin[i] / np.linalg.norm(W[:, i]) ** 2)
    print(f"{kk:6.2f} {kk * args.xi:6.2f} {np.abs(lz).max() - 1:11.2e} {np.abs(res[1][0]).max() - 1:10.2e} {np.abs(res[2][0]).max() - 1:10.2e}"
          f"  {abs(w_[di]):.5f} / {abs(w_[si]):.5f}   free n k^2/6m = {np.sqrt(1 - m * m) * kk**2 / (6 * m):.5f}"
          f"  bogo sqrt(e(e+2 g)) = {np.sqrt((np.sqrt(1 - m * m) * kk**2 / (6 * m)) * (np.sqrt(1 - m * m) * kk**2 / (6 * m) + 2 * g)):.5f}")
