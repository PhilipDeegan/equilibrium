"""Poloidal coil of vortex lines (Philip 10-01: windings like a copper coil, each winding a resolved strand): N closed
poloidal loops ("wires"), each a vortex ring of radius a in the meridional plane at toroidal angle theta_j = 2 pi j / N,
centred on the torus centre circle (radius r0). Same rule as gp.py (real-form Dirac step + local density turn,
uniform background, ghost = background, sponge to background). Each loop propagates along its own axis = the toroidal
direction, so a same-sense coil should rotate around z as a whole (travelling circulation).

Seed: psi = a0 v prod_j chi_j, chi_j = ring field of loop j (phase atan2(s, rho' - a) - atan2(s, rho' + a) in the loop's
frame, s = axial coordinate along the toroidal tangent, rho' = distance from the loop axis; core d / sqrt(d^2 + xi^2)).
--alt: alternate the loop senses (net zero propagation; control). Full state (L, R) at plot times in
/bk/dst/2026/10/02/coil/<name>/.
Scores: core cells (rho < 0.25 rho0) clustered by toroidal angle into N bins: per-loop toroidal angle, minor radius
(distance of its core from the centre circle), winding around the loop core (64-point loop of radius 2 xi in the
loop's meridional plane), and the coil's mean rotation rate vs the single-ring GP speed / r0.
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parents[2] / "01" / "weyl_bcc"))

from equilibrium.substrate.lattice import DampedLattice
from walk import A_REAL, J4, K_OF_H, K_OF_MINUS_H, DiracWalk, symbol_dirac, to_real

here = Path(__file__).parent
p = argparse.ArgumentParser()
p.add_argument("--m", type=float, default=0.5)
p.add_argument("--xi", type=float, default=3.9, help="core width (sets g = c^2 / (2 m rho0 xi^2))")
p.add_argument("--rho0", type=float, default=1.0)
p.add_argument("--N", type=int, default=8)
p.add_argument("--a", type=float, default=10.0, help="loop (minor) radius")
p.add_argument("--r0", type=float, default=30.0)
p.add_argument("--R", type=float, default=72.0)
p.add_argument("--alt", action="store_true")
p.add_argument("--T", type=int, default=1200)
p.add_argument("--every", type=int, default=20)
p.add_argument("--n_plot", type=int, default=4)
p.add_argument("--damp", type=float, default=0.05)
p.add_argument("--tag", default="")
args = p.parse_args()
c2 = 1.0 / 3.0
xi = args.xi
g = c2 / (2 * args.m * args.rho0 * xi**2)
v_ring = (c2 / args.m) / (2 * args.a) * (np.log(8 * args.a / xi) - 0.615)
name = f"coil_N{args.N}{'alt' if args.alt else ''}_a{args.a:g}_r{args.r0:g}_xi{xi:g}_m{args.m:g}_R{args.R:g}{args.tag}"

t0 = time.time()
lat = DampedLattice(radius_cells=args.R, sponge_fraction=0.25)
pos = lat.pos
x, y, z = pos.T
rho_c = np.hypot(x, y)
th_c = np.arctan2(y, x)
r_all = np.linalg.norm(pos, axis=1)
lam, V = np.linalg.eig(symbol_dirac(np.zeros(3), args.m))
om_all = -np.angle(lam)
v = V[:, int(np.argmax(om_all))]
om0 = om_all.max()
a0 = np.sqrt(args.rho0)
thetas = 2 * np.pi * np.arange(args.N) / args.N
senses = np.array([(-1) ** j if args.alt else 1 for j in range(args.N)])


def loop_field(thj, q):
    C = np.array([args.r0 * np.cos(thj), args.r0 * np.sin(thj), 0.0])
    t = np.array([-np.sin(thj), np.cos(thj), 0.0])
    X = pos - C
    s = X @ t
    rp = np.linalg.norm(X - s[:, None] * t[None, :], axis=1)
    d1, d2 = np.hypot(rp - args.a, s), np.hypot(rp + args.a, s)
    ph = q * (np.arctan2(s, rp - args.a) - np.arctan2(s, rp + args.a))
    return d1 / np.sqrt(d1**2 + xi**2) * d2 / np.sqrt(d2**2 + xi**2) * np.exp(1j * ph)


chi = a0 * np.ones(lat.N, complex)
for thj, q in zip(thetas, senses):
    chi *= loop_field(thj, q)


class GPWalk(DiracWalk):
    t = 0

    def bg(self, t):
        z_ = a0 * v * np.exp(-1j * (om0 + g * args.rho0) * t)
        return to_real(z_[None, :2])[0], to_real(z_[None, 2:])[0]

    def _hop_bg(self, psi, ks, mats, ghost):
        out = np.zeros_like(psi)
        for k, A in zip(ks, mats):
            out += np.where(self.valid[:, k][:, None], psi[self.nb[:, k]], ghost[None, :]) @ A.T
        return out

    def step(self, n_steps=1):
        for _ in range(n_steps):
            gL, gR = self.bg(self.t)
            Lh = self._hop_bg(self.L, K_OF_H, A_REAL, gL)
            Rh = self._hop_bg(self.R, K_OF_MINUS_H, A_REAL.transpose(0, 2, 1), gR)
            L1 = self.n * Lh + self.m * self.R @ J4.T
            R1 = self.m * self.L @ J4.T + self.n * Rh
            th = g * (np.sum(L1**2, 1) + np.sum(R1**2, 1))
            cth, sth = np.cos(th)[:, None], np.sin(th)[:, None]
            for P in (L1, R1):
                a_, b_ = P[:, 0::2].copy(), P[:, 1::2].copy()
                P[:, 0::2] = a_ * cth + b_ * sth
                P[:, 1::2] = b_ * cth - a_ * sth
            self.t += 1
            gL, gR = self.bg(self.t)
            self.L = gL + (L1 - gL) * self.f
            self.R = gR + (R1 - gR) * self.f


w = GPWalk(lat, m=args.m, damp=args.damp)
psi0 = chi[:, None] * v[None, :]
w.L, w.R = to_real(psi0[:, :2]), to_real(psi0[:, 2:])
inner = r_all < w.r_inner
assert args.r0 + args.a + 3 * xi < w.r_inner
print(f"{name}: N_cells={lat.N} build+seed {time.time() - t0:.0f}s; g {g:.4f} xi {xi:.2f} omega0 {om0:.4f}; "
      f"single-loop GP speed {v_ring:.4f} -> coil rotation {v_ring / args.r0:.5f} rad/tick ({2 * np.pi / args.N / (v_ring / args.r0):.0f} ticks "
      f"per loop spacing); loop spacing on the centre circle {2 * np.pi * args.r0 / args.N:.1f} cells", flush=True)
tree = cKDTree(pos)
snap_dir = Path("/bk/dst/2026/10/02/coil") / name
snap_dir.mkdir(parents=True, exist_ok=True)
np.save(snap_dir / "pos.npy", pos)


def amp():
    Lc = w.L[:, 0::2] + 1j * w.L[:, 1::2]
    Rc = w.R[:, 0::2] + 1j * w.R[:, 1::2]
    return np.concatenate([Lc, Rc], 1) @ v.conj()


def loops_state():
    """per toroidal bin (N bins centred on the current mean offset): loop angle, minor radius of its outer core,
    winding around the outer core point."""
    rho_ = w.density()
    core = (rho_ < 0.25 * args.rho0) & inner
    if not core.any():
        return None
    wgt = (args.rho0 - rho_)[core]
    thc = th_c[core]
    off = np.angle(np.sum(wgt * np.exp(1j * args.N * thc))) / args.N  # coil phase (mod 2 pi / N)
    a_ = amp()
    out = []
    for j in range(args.N):
        tj = off + 2 * np.pi * j / args.N
        sel = np.abs(np.angle(np.exp(1j * (thc - tj)))) < np.pi / args.N
        if not sel.any():
            out.append((tj, np.nan, np.nan, np.nan))
            continue
        P = pos[core][sel]
        wj = wgt[sel]
        dmin = np.hypot(np.hypot(P[:, 0], P[:, 1]) - args.r0, P[:, 2])
        a_j = np.sum(wj * dmin) / wj.sum()
        thj = np.angle(np.sum(wj * np.exp(1j * thc[sel])))
        # loop j's core passes (rho, z) = (r0 + a_j, 0) of its meridional plane heading along z: wind around it in
        # the (radial, toroidal) plane
        ps = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        er = np.array([np.cos(thj), np.sin(thj), 0.0])
        et = np.array([-np.sin(thj), np.cos(thj), 0.0])
        Cpt = (args.r0 + a_j) * er
        Q2 = Cpt[None, :] + 2 * xi * (np.cos(ps)[:, None] * er[None, :] + np.sin(ps)[:, None] * et[None, :])
        l2 = tree.query(Q2)[1]
        wind = np.sum(np.angle(a_[l2][np.r_[1:64, 0]] / a_[l2])) / (2 * np.pi)
        out.append((thj, a_j, wind, sel.sum()))
    return np.array(out), off, core.sum()


def plot(tag):
    from scipy.interpolate import griddata

    rho_ = w.density()
    a_ = amp()
    bgph = np.exp(-1j * (om0 + g * args.rho0) * w.t)
    lim = args.r0 + args.a + 4 * xi
    gg = np.linspace(-lim, lim, 401)
    G1, G2 = np.meshgrid(gg, gg)
    fig, ax = plt.subplots(1, 3, figsize=(19, 6))
    m_ = (np.abs(z) < 0.3) & (np.abs(x) < lim + 2) & (np.abs(y) < lim + 2)
    for i, (val, lab, cmap) in enumerate([(rho_, "density (xy, z = 0)", "viridis"),
                                          (np.angle(a_ / bgph), "pair angle rel. bg (xy)", "twilight")]):
        V_ = griddata((x[m_], y[m_]), val[m_], (G1, G2), method="nearest" if i else "linear")
        im = ax[i].imshow(V_, origin="lower", extent=(-lim, lim, -lim, lim), cmap=cmap)
        ax[i].set_title(lab)
        fig.colorbar(im, ax=ax[i], shrink=0.8)
    lim2 = args.a + 4 * xi
    m2 = (np.abs(y) < 0.3) & (x > 0) & (np.abs(x - args.r0) < lim2 + 2) & (np.abs(z) < lim2 + 2)
    g2 = np.linspace(-lim2, lim2, 201)
    H1, H2 = np.meshgrid(args.r0 + g2, g2)
    V_ = griddata((x[m2], z[m2]), rho_[m2], (H1, H2), method="linear")
    im = ax[2].imshow(V_, origin="lower", extent=(args.r0 - lim2, args.r0 + lim2, -lim2, lim2), cmap="viridis")
    ax[2].set_title("density, meridional plane y = 0 (x > 0)")
    fig.colorbar(im, ax=ax[2], shrink=0.8)
    fig.suptitle(f"{name}  t = {w.t}")
    fig.tight_layout()
    fig.savefig(here / f"{name}_{tag}.png", dpi=80)
    plt.close(fig)


plot("t0000")
rec = []
t0 = time.time()
plot_every = max(args.every, (args.T // args.n_plot) // args.every * args.every)
for t in range(args.T + 1):
    if t % args.every == 0:
        st = loops_state()
        if st is not None:
            L_, off, ncore = st
            rec.append(np.concatenate([[t, off, ncore], L_[:, 0], L_[:, 1], L_[:, 2]]))
            if t % plot_every == 0:
                print(f"  t {t}: core cells {ncore}, coil phase {off:+.4f}, minor radii {np.round(L_[:, 1], 1).tolist()}, "
                      f"windings {np.round(L_[:, 2], 2).tolist()}", flush=True)
        if t % plot_every == 0:
            np.savez(snap_dir / f"state_t{t:05d}.npz", t=t, L=w.L, R=w.R)
        if t and t % plot_every == 0:
            plot(f"t{t:05d}")
    if t < args.T:
        w.step()
rec = np.array(rec)
off_u = np.unwrap(rec[:, 1] * args.N) / args.N
rate = np.polyfit(rec[:, 0], off_u, 1)[0] if len(rec) > 2 else np.nan
print(f"  {args.T} ticks in {time.time() - t0:.0f}s; coil rotation {rate:+.5f} rad/tick (single-loop GP / r0: "
      f"{v_ring / args.r0:.5f})", flush=True)
np.savez(here / f"{name}_rec.npz", rec=rec, args=str(vars(args)), xi=xi, g=g, v_ring=v_ring)
