"""Row-wise sponge (production DampedLattice.apply_sponge: each cell scales its own copy of
each face) vs face-wise sponge (both copies scaled by the mean of the two cells' factors;
boundary faces by the cell's own), same strength, discrete step().

A  point source: fixed u = 1 at the centre cell, run to steady state; u(r) profile, window
   slope d ln u / d ln r, and the apparent charge (divergence of the antisymmetric part of phi)
   by radial shell.
B  outgoing gaussian pulse: interior energy after the front has left, vs a reference domain
   big enough that nothing has come back yet.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from equilibrium.substrate.lattice import _K_REV, _W, DampedLattice, SubstrateLattice

out = Path(__file__).parent


class FaceSponge(DampedLattice):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        nb = self.nb
        valid = nb >= 0
        d = self._sponge
        dn = np.where(valid, d[np.where(valid, nb, 0)], d[:, None])
        self._face_damp = 0.5 * (d[:, None] + dn)

    def apply_sponge(self):
        u_new = self.u - self._sponge * (self.u - self._u_bg)
        self.u = np.where(self.fixed_u, self.u, u_new)
        phi_new = self.phi - self._face_damp * (self.phi - self.phi_bg)
        self.phi = np.where(self.fixed_phi, self.phi, phi_new)


def antisym_err(lat):
    i, k = np.nonzero(lat.nb >= 0)
    return float(np.max(np.abs(lat.phi[i, k] + lat.phi[lat.nb[i, k], _K_REV[k]])))


def true_div(lat):
    """Divergence of the antisymmetric part of phi, per cell."""
    i, k = np.nonzero(lat.nb >= 0)
    a = 0.5 * (lat.phi[i, k] - lat.phi[lat.nb[i, k], _K_REV[k]])
    div = np.zeros(lat.N)
    np.add.at(div, i, _W[k] * a)
    return div


def energy(lat, mask):
    valid = lat.nb >= 0
    return 0.5 * np.sum(lat.u[mask] ** 2) + 0.25 * np.sum((lat.phi[mask] * valid[mask]) ** 2)


def point_source(cls, R, steps, damp, sponge):
    lat = cls(radius_cells=R, sponge_fraction=sponge, damp_max=damp)
    r = np.linalg.norm(lat.pos, axis=1)
    src = int(np.argmin(r))
    lat.fixed_u[src] = True
    lat.u[src] = 1.0
    dt = lat.stable_dt()
    hist = []
    for s in range(1, steps + 1):
        lat.step(dt)
        if s % (steps // 20) == 0:
            hist.append(lat.u.copy())
    return lat, r, np.array(hist)


def window_slope(r, u, lo, hi):
    m = (r > lo) & (r < hi) & (u > 0)
    return float(np.polyfit(np.log(r[m]), np.log(u[m]), 1)[0])


def pulse(cls, R, t_end, damp, sponge, width=1.5, **kw):
    lat = cls(radius_cells=R, sponge_fraction=sponge, damp_max=damp, **kw) if cls is not SubstrateLattice else cls(radius_cells=R)
    r = np.linalg.norm(lat.pos, axis=1)
    lat.u[:] = np.exp(-(r**2) / (2 * width**2))
    dt = lat.stable_dt()
    n = int(round(t_end / dt))
    return lat, r, dt, n


p = argparse.ArgumentParser()
p.add_argument("--R", type=float, default=12.0)
p.add_argument("--damp", type=float, default=0.02)
p.add_argument("--sponge", type=float, default=0.2)
p.add_argument("--src-steps", type=int, default=20000)
args = p.parse_args()

# ---------------- A: point source
res = {}
for name, cls in (("row", DampedLattice), ("face", FaceSponge)):
    lat, r, hist = point_source(cls, args.R, args.src_steps, args.damp, args.sponge)
    rin = r.max() * (1 - args.sponge)
    late = np.abs(hist[-1] - hist[-5]).max() / hist[-1].max()
    div = true_div(lat)
    shells = np.arange(0.0, r.max() + 1.0, 1.0)
    q = np.array([div[(r >= a) & (r < a + 1)].sum() for a in shells[:-1]])
    res[name] = dict(u=lat.u.copy(), r=r, q=q, shells=shells, rin=rin)
    print(
        f"A {name}: antisym {antisym_err(lat):.3e}  late change {late:.2e}  "
        f"slope[2,{rin - 1:.1f}] {window_slope(r, lat.u, 2.0, rin - 1):.4f}  "
        f"apparent charge in sponge-edge shells {q[(shells[:-1] >= rin - 1.5) & (shells[:-1] < rin + 1.5)].sum():.4e}  "
        f"source cell div {div[int(np.argmin(r))]:.4e}",
        flush=True,
    )
du = res["face"]["u"] - res["row"]["u"]
inner = res["row"]["r"] < res["row"]["rin"]
print(f"A face-row: max |du| interior {np.abs(du[inner]).max():.3e} (rel {np.abs(du[inner]).max() / np.abs(res['row']['u'][inner]).max():.3e})")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for name in res:
    d = res[name]
    o = np.argsort(d["r"])
    ax[0].loglog(d["r"][o][1:], np.abs(d["u"][o][1:]), ".", ms=2, label=name)
    ax[1].plot(d["shells"][:-1] + 0.5, d["q"], "o-", label=name)
for a in ax:
    a.axvline(res["row"]["rin"], color="k", lw=0.5)
    a.legend()
ax[0].set_xlabel("r")
ax[0].set_ylabel("|u|")
ax[1].set_xlabel("r")
ax[1].set_ylabel("apparent charge per unit shell")
fig.tight_layout()
fig.savefig(out / "A_point_source.png", dpi=110)

# ---------------- B: pulse
c = 2.309
t_end = 3.0 * args.R / c
ref, r_ref, dt, n = pulse(SubstrateLattice, args.R + 2.5 * args.R, t_end, 0, 0)
Rin = args.R * (1 - args.sponge)
series = {}
for s in range(n):
    ref.step(dt)
e_ref = energy(ref, r_ref < Rin)
for name, cls in (("row", DampedLattice), ("face", FaceSponge)):
    lat, r, dt2, n2 = pulse(cls, args.R, t_end, args.damp, args.sponge)
    m = r < Rin
    e0 = energy(lat, m)
    ts, es = [], []
    for s in range(n2):
        lat.step(dt2)
        if s % 50 == 0:
            ts.append(s * dt2)
            es.append(energy(lat, m))
    series[name] = (np.array(ts), np.array(es) / e0)
    print(f"B {name}: interior energy at t={t_end:.1f}: {es[-1] / e0:.3e} of initial (reference {e_ref / e0:.3e}); antisym {antisym_err(lat):.3e}", flush=True)

fig, ax = plt.subplots(figsize=(6, 4.5))
for name, (t, e) in series.items():
    ax.semilogy(t, e, label=name)
ax.axhline(e_ref / e0, color="k", lw=0.5, label="no-reflection reference")
ax.set_xlabel("t")
ax.set_ylabel("interior energy / initial")
ax.legend()
fig.tight_layout()
fig.savefig(out / "B_pulse.png", dpi=110)
