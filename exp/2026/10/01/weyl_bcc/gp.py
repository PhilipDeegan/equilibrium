"""Vortex ring in a uniform background with a local density turn (real-form Dirac rule + nonlinearity).

Each tick: the linear real-form Dirac step (walk.py), then every pair in a cell turns by -g rho (rho = sum psi^2 of
the cell; norm-preserving, local). Background: uniform a0 v (v = k = 0 positive-frequency eigenvector), exactly
stationary in density, phase advancing by omega0 + g rho0 per tick. Edge: missing neighbours read the background
(ghost = background), and a sponge relaxes the outer shell toward the background (not toward zero).

Non-relativistic limit (omega ~ m + c^2 k^2 / 2m): Gross-Pitaevskii with hbar/M = c^2/m, healing length
xi = sqrt(c^2 / (2 m g rho0)), sound speed sqrt(g rho0 c^2 / m), ring speed v = (c^2/m) / (4 pi r0)... standard
v = (hbar/M) / (2 r0) (ln(8 r0 / xi) - 0.615) for the defocusing sign (g > 0 on the positive branch).

Seed: ring of radius r0 in the xy plane at z = z0, phase theta = atan2(z - z0, rho - r0) - atan2(z - z0, rho + r0)
(-> 0 far away), amplitude prod d_i / sqrt(d_i^2 + xi^2). q = +-1 sets the circulation sense.
Scores: winding of the pair angle around loops of radius 2 xi about the tracked core (8 toroidal angles), core
radius / z from the deficit (rho0 - rho), deficit integral, translation speed vs the GP formula.
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from equilibrium.substrate.lattice import DampedLattice
from walk import A_REAL, J4, K_OF_H, K_OF_MINUS_H, DiracWalk, symbol_dirac, to_real

here = Path(__file__).parent
p = argparse.ArgumentParser()
p.add_argument("--m", type=float, default=0.5)
p.add_argument("--g", type=float, default=0.05)
p.add_argument("--rho0", type=float, default=1.0)
p.add_argument("--q", type=int, default=1)
p.add_argument("--R", type=float, default=40.0)
p.add_argument("--r0", type=float, default=12.0)
p.add_argument("--z0", type=float, default=-12.0)
p.add_argument("--T", type=int, default=400)
p.add_argument("--every", type=int, default=10)
p.add_argument("--damp", type=float, default=0.05)
p.add_argument("--s", type=float, default=1.0, help="refinement: lengths x s, g / s^2, ticks x s^2 (fixed r0/xi, GP time)")
p.add_argument("--tag", default="")
args = p.parse_args()
if args.s != 1.0:
    args.r0 *= args.s
    args.z0 *= args.s
    args.R *= args.s
    args.g /= args.s**2
    args.T = int(round(args.T * args.s**2))
    args.every = int(round(args.every * args.s**2))
    args.tag = f"_s{args.s:g}" + args.tag
name = f"gp_m{args.m:g}_g{args.g:g}_q{args.q}_r{args.r0:g}_R{args.R:g}{args.tag}"
c2 = 1.0 / 3.0
xi = np.sqrt(c2 / (2 * args.m * args.g * args.rho0)) if args.g > 0 else np.sqrt(c2 / (2 * args.m * 0.05))
cs = np.sqrt(args.g * args.rho0 * c2 / args.m)
v_ring = (c2 / args.m) / (2 * args.r0) * (np.log(8 * args.r0 / xi) - 0.615)

lat = DampedLattice(radius_cells=args.R, sponge_fraction=0.25)
pos = lat.pos
x, y, z = pos.T
rho_c = np.hypot(x, y)
r_all = np.linalg.norm(pos, axis=1)

lam, V = np.linalg.eig(symbol_dirac(np.zeros(3), args.m))
om_all = -np.angle(lam)
v = V[:, int(np.argmax(om_all))]
om0 = om_all.max()
a0 = np.sqrt(args.rho0)


class GPWalk(DiracWalk):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.t = 0
        self.f = self.f  # sponge factor (1 inside)

    def bg(self, t):
        z_ = a0 * v * np.exp(-1j * (om0 + args.g * args.rho0) * t)
        return to_real(z_[None, :2])[0], to_real(z_[None, 2:])[0]

    def _hop_bg(self, psi, ks, mats, ghost):
        out = np.zeros_like(psi)
        for k, A in zip(ks, mats):
            vv = self.valid[:, k][:, None]
            out += np.where(vv, psi[self.nb[:, k]], ghost[None, :]) @ A.T
        return out

    def step(self, n_steps=1):
        for _ in range(n_steps):
            gL, gR = self.bg(self.t)
            Lh = self._hop_bg(self.L, K_OF_H, A_REAL, gL)
            Rh = self._hop_bg(self.R, K_OF_MINUS_H, A_REAL.transpose(0, 2, 1), gR)
            L1 = self.n * Lh + self.m * self.R @ J4.T
            R1 = self.m * self.L @ J4.T + self.n * Rh
            th = args.g * (np.sum(L1**2, 1) + np.sum(R1**2, 1))
            cth, sth = np.cos(th)[:, None], np.sin(th)[:, None]
            # turn every pair by -th: (a, b) -> (a cos + b sin, b cos - a sin)
            for P in (L1, R1):
                a_, b_ = P[:, 0::2].copy(), P[:, 1::2].copy()
                P[:, 0::2] = a_ * cth + b_ * sth
                P[:, 1::2] = b_ * cth - a_ * sth
            self.t += 1
            gL, gR = self.bg(self.t)
            self.L = gL + (L1 - gL) * self.f
            self.R = gR + (R1 - gR) * self.f


def ring_field(q):
    d1 = np.hypot(rho_c - args.r0, z - args.z0)
    d2 = np.hypot(rho_c + args.r0, z - args.z0)
    th = q * (np.arctan2(z - args.z0, rho_c - args.r0) - np.arctan2(z - args.z0, rho_c + args.r0))
    amp = d1 / np.sqrt(d1**2 + xi**2) * d2 / np.sqrt(d2**2 + xi**2)
    return a0 * amp * np.exp(1j * th)


w = GPWalk(lat, m=args.m, damp=args.damp)
chi = ring_field(args.q) if args.q else a0 * np.ones(lat.N, complex)
psi0 = chi[:, None] * v[None, :]
w.L, w.R = to_real(psi0[:, :2]), to_real(psi0[:, 2:])
inner = r_all < w.r_inner
print(f"{name}: N={lat.N} omega0 {om0:.4f} xi {xi:.2f} c_s {cs:.4f} GP ring speed {v_ring:.4f} (x q sign) "
      f"-> {v_ring * args.T:.1f} cells in {args.T} ticks; sponge from r {w.r_inner:.1f}", flush=True)

tree = cKDTree(pos)


def amp(wk):
    Lc = wk.L[:, 0::2] + 1j * wk.L[:, 1::2]
    Rc = wk.R[:, 0::2] + 1j * wk.R[:, 1::2]
    return np.concatenate([Lc, Rc], 1) @ v.conj()


def core():
    """deficit-weighted ring radius and height (inner region only)."""
    rho_ = w.density()
    dfc = np.where((rho_ < 0.25 * args.rho0) & inner, args.rho0 - rho_, 0.0)  # vortex cores only (transient dips stay above)
    if dfc.sum() == 0:
        return np.nan, np.nan, np.sum(np.clip(args.rho0 - rho_, 0, None) * inner)
    return np.sum(dfc * rho_c) / dfc.sum(), np.sum(dfc * z) / dfc.sum(), np.sum(np.clip(args.rho0 - rho_, 0, None) * inner)


def windings(a, rc, zc, rad):
    out = []
    for th in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        ps = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        Rr = rc + rad * np.cos(ps)
        P = np.stack([Rr * np.cos(th), Rr * np.sin(th), zc + rad * np.sin(ps)], -1)
        l = tree.query(P)[1]
        out.append(np.sum(np.angle(a[l][np.r_[1:64, 0]] / a[l])) / (2 * np.pi))
    return np.round(out, 2)


def plot(tag):
    from scipy.interpolate import griddata

    rho_ = w.density()
    a = amp(w)
    lim = w.r_inner
    g = np.linspace(-lim, lim, 301)
    G1, G2 = np.meshgrid(g, g)
    m_ = (np.abs(y) < 0.3) & (np.abs(x) < lim + 2) & (np.abs(z) < lim + 2)
    bgph = np.exp(-1j * (om0 + args.g * args.rho0) * w.t)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for i, (val, lab, cmap) in enumerate([(rho_, "density", "viridis"), (np.angle(a / bgph), "pair angle rel. background", "twilight")]):
        V_ = griddata((x[m_], z[m_]), val[m_], (G1, G2), method="nearest" if i else "linear")
        im = ax[i].imshow(V_, origin="lower", extent=(-lim, lim, -lim, lim), cmap=cmap)
        ax[i].set_title(f"{lab} (xz)")
        ax[i].set_xlabel("x")
        ax[i].set_ylabel("z")
        fig.colorbar(im, ax=ax[i], shrink=0.8)
    fig.suptitle(f"{name}  t = {w.t}")
    fig.tight_layout()
    fig.savefig(here / f"{name}_{tag}.png", dpi=80)
    plt.close(fig)


plot("t0000")
rec = []
t0 = time.time()
for t in range(args.T + 1):
    if t % args.every == 0:
        rc, zc, D = core()
        wd = windings(amp(w), rc, zc, 2 * xi) if np.isfinite(rc) else np.full(8, np.nan)
        rec.append([t, rc, zc, D, np.mean(wd), np.min(wd), np.max(wd), w.density()[inner].mean()])
        if t % (args.every * max(1, args.T // (4 * args.every))) == 0:
            print(f"  t {t}: core r {rc:.2f} z {zc:+.2f}  deficit {D:.1f}  windings {wd.tolist()}  "
                  f"mean rho inner {rec[-1][-1]:.4f}", flush=True)
            if t:
                plot(f"t{t:04d}")
    if t < args.T:
        w.step()
rec = np.array(rec)
sel = (rec[:, 0] >= 0.25 * args.T) & np.isfinite(rec[:, 2])
vz = np.polyfit(rec[sel, 0], rec[sel, 2], 1)[0]
vr = np.polyfit(rec[sel, 0], rec[sel, 1], 1)[0]
tau = xi**2 * args.m / c2
print(f"  {args.T} ticks in {time.time() - t0:.0f}s; ring speed dz/dt {vz:+.4f} (GP {v_ring:.4f}, ratio {abs(vz) / v_ring:.3f}), "
      f"dr/dt {vr:+.4f}; in GP units (xi, tau = xi^2 m / c^2 = {tau:.1f}): v {abs(vz) * tau / xi:.4f}, r0/xi {args.r0 / xi:.2f}", flush=True)
np.savez(here / f"{name}_rec.npz", rec=rec, cols="t r_core z_core deficit wind_mean wind_min wind_max rho_inner",
         args=str(vars(args)), xi=xi, v_ring=v_ring)
