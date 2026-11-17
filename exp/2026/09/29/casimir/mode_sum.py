"""Exact lattice Casimir: BCC stacked along z (layer spacing h = a/2), per transverse K a
layer chain with hex coupling to j+-1 and square coupling to j+-2. Dirichlet plates are two
fixed layers each side (square links hop one layer). E(N) = 1/2 sum omega per area; fit
alpha N + beta + gamma/N^3 + delta/N^5 and compare gamma with -pi^2 c/(1440 h^3)."""

import argparse

import numpy as np

from equilibrium.substrate.lattice import W_HEX, W_SQ

p = argparse.ArgumentParser()
p.add_argument("--nxy", type=int, default=256)
p.add_argument("--ns", type=str, default="8,10,12,16,20,24,32,40")
p.add_argument("--powers", type=str, default="3,5")
args = p.parse_args()

a = 1.0
h = a / 2
c = np.sqrt(W_HEX**2 + W_SQ**2) * a
nxy = args.nxy
dk = 2 * np.pi / (nxy * a)
k = np.arange(nxy) * dk - np.pi / a + dk / 2
Kx, Ky = np.meshgrid(k, k, indexing="ij")
Kx, Ky = Kx.ravel(), Ky.ravel()

D0 = 8 * W_HEX**2 + 2 * W_SQ**2 + W_SQ**2 * (4 - 2 * np.cos(a * Kx) - 2 * np.cos(a * Ky))
t1 = 4 * W_HEX**2 * np.cos(h * Kx) * np.cos(h * Ky)
t2 = W_SQ**2 * np.ones_like(Kx)


def energy(N):
    m = N - 1
    M = np.zeros((len(Kx), m, m))
    i = np.arange(m)
    M[:, i, i] = D0[:, None]
    M[:, i[:-1], i[:-1] + 1] = -t1[:, None]
    M[:, i[:-1] + 1, i[:-1]] = -t1[:, None]
    M[:, i[:-2], i[:-2] + 2] = -t2[:, None]
    M[:, i[:-2] + 2, i[:-2]] = -t2[:, None]
    w = np.sqrt(np.maximum(np.linalg.eigvalsh(M), 0.0))
    return 0.5 * w.sum(axis=1).mean() / a**2


Ns = np.array([int(s) for s in args.ns.split(",")])
E = np.array([energy(N) for N in Ns])
pw = [int(s) for s in args.powers.split(",")]
A = np.stack([Ns, np.ones_like(Ns)] + [1.0 / Ns**q for q in pw], axis=1).astype(float)
coef = np.linalg.lstsq(A, E, rcond=None)[0]
alpha, beta, gamma = coef[:3]
delta = coef[3] if len(coef) > 3 else 0.0
gamma_th = -np.pi**2 * c / (1440 * h**3)
print(f"powers={pw} Ns={list(Ns)}")
print(f"nxy={nxy} c={c:.4f}  gamma={gamma:.6e}  theory={gamma_th:.6e}  ratio={gamma/gamma_th:.4f}")
print(f"alpha={alpha:.6f} beta={beta:.6f} delta={delta:.4e}")
res = E - A @ coef
print("fit residual / gamma/N^3:", np.round(res / (gamma / Ns**3), 4))
