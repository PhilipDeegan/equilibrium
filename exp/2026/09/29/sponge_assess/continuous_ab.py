"""Continuous model (C++ step_ae, DEC curl, iso weights): sponge variants on an outgoing packet.

prod   : C++ step_sponge (per cell row: u, phi; E untouched)
row_e  : per cell row on u, phi and E
face_e : u per cell; phi and E per face (both copies scaled by the mean of the two cells' factors)
Packets: T = rotational A (phi) gaussian, E = 0; S = gaussian u. Interior amplitude
sum(u^2 + phi^2 + e^2) over r < r_inner vs a reference domain big enough that nothing returns.
"""

import argparse
from pathlib import Path

import numpy as np

from equilibrium import lattice as _lat
from equilibrium.substrate import continuous as C
from equilibrium.substrate.lattice import _K_REV

out = Path(__file__).parent
p = argparse.ArgumentParser()
p.add_argument("--R", type=float, default=8.0)
p.add_argument("--h", type=float, default=0.5)
p.add_argument("--damp", type=float, default=0.02)
p.add_argument("--sponge", type=float, default=0.2)
p.add_argument("--width", type=float, default=1.5)
p.add_argument("--threads", type=int, default=4)
args = p.parse_args()

c = C.C0
t_end = 3.0 * args.R / c
r_in = args.R * (1 - args.sponge)


def build(R):
    ac = C.anchor_cluster(R, args.h, n_threads=args.threads)
    ac.set_curl("dec")
    ac.r = np.linalg.norm(ac.pos, axis=1)
    return ac


def ic(ac, kind):
    X = C.face_midpoints(ac) / args.width
    if kind == "T":
        g = np.exp(-0.5 * np.sum(X**2, -1))
        A = np.stack([X[..., 1] * g, -X[..., 0] * g, 0 * g], -1)
        ac.phi = np.where(ac.nb >= 0, np.einsum("k,nkc,kc->nk", C.W, A, C.D_HAT * args.h), 0.0)
        ac.u = np.zeros(ac.N)
    else:
        ac.u = np.exp(-(ac.r**2) / (2 * args.width**2))
        ac.phi = np.zeros((ac.N, 14))
    ac.e = np.zeros((ac.N, 14))


def amp(ac, m):
    return float(np.sum(ac.u[m] ** 2) + np.sum(ac.phi[m] ** 2) + np.sum(ac.e[m] ** 2))


def antisym(f, nb):
    i, k = np.nonzero(nb >= 0)
    return float(np.max(np.abs(f[i, k] + f[nb[i, k], _K_REV[k]])))


def run(kind, variant):
    R = args.R if variant != "ref" else (c * t_end + r_in) / 2 + 2.0
    ac = build(R)
    ic(ac, kind)
    m = ac.r < r_in
    t = np.clip((ac.r - r_in) / (ac.r.max() - r_in), 0, 1) if variant != "ref" else np.zeros(ac.N)
    d = args.damp * t**2
    nb = ac.nb
    valid = nb >= 0
    dface = 0.5 * (d[:, None] + np.where(valid, d[np.where(valid, nb, 0)], d[:, None]))
    if variant == "prod":
        BC = _lat.continuous.BoundaryCondition
        ac.cl.set_bc(BC.quadratic_sponge(target=0.0, damp_max=args.damp, r_all=ac.r, r_inner=r_in))
    dt = C.stable_dt(ac)
    n = int(round(t_end / dt))
    a0 = amp(ac, m)
    hist = []
    for s in range(n):
        ac.step_ae(dt)
        if variant == "prod":
            ac.alat.step_sponge(0)
        elif variant in ("row_e", "face_e"):
            ac.u = ac.u * (1 - d)
            f = (1 - d)[:, None] if variant == "row_e" else 1 - dface
            ac.phi = ac.phi * f
            ac.e = ac.e * f
        if s % 50 == 0 or s == n - 1:
            hist.append((s * dt, amp(ac, m) / a0))
    hist = np.array(hist)
    tail = hist[hist[:, 0] > 0.7 * t_end, 1]
    print(
        f"{kind} {variant:6s} N={ac.N} end {hist[-1, 1]:.3e}  mean(last 30%) {tail.mean():.3e}  "
        f"antisym phi {antisym(ac.phi, nb):.2e} e {antisym(ac.e, nb):.2e}",
        flush=True,
    )
    return hist


res = {}
for kind in ("T", "S"):
    for variant in ("ref", "prod", "row_e", "face_e"):
        res[f"{kind}_{variant}"] = run(kind, variant)
np.savez(out / "continuous_ab.npz", **res, args=str(vars(args)))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for a, kind in zip(ax, ("T", "S")):
    for variant in ("ref", "prod", "row_e", "face_e"):
        h = res[f"{kind}_{variant}"]
        a.semilogy(h[:, 0], h[:, 1], label=variant)
    a.set_title(f"{kind} packet, continuous step_ae (dec), R={args.R} h={args.h}")
    a.set_xlabel("t")
    a.set_ylabel("interior sum(u^2+phi^2+e^2) / initial")
    a.legend()
fig.tight_layout()
fig.savefig(out / "continuous_ab.png", dpi=110)
