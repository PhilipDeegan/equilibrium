"""Spring coil (Philip 10-02: a coil is a spring, each turn advances toroidally -- the 10-02 meridional loops were
stellarator-like, no horizontal element): one continuous vortex strand along the torus helix
    theta_tor = s, theta_pol = N s + 2 pi j / S,  s in [0, 2 pi)
(N poloidal turns per toroidal circuit, S parallel strands = multi-start), minor radius a, centre circle r0.
Same rule as coil.py / gp.py (real-form Dirac walk + local density turn, uniform background, sponge to background).

Seed: phase = q Omega / 2, Omega = solid angle the closed strand(s) subtend (Van Oosterom-Strackee, cone to a
generic apex; Omega is defined mod 4 pi, the phase mod 2 pi), so the winding around the strand is q everywhere along it;
envelope d / sqrt(d^2 + xi^2), d = distance to the nearest strand.
Scores (co-moving: torus centre (rc, zc) = core-weighted mean (rho, z)): core cells (rho < 0.25 rho0); phase winding
along the centre circle (= N S while every turn is intact) and along a meridional circle of radius a + 3 xi around the
tube (= S, the toroidal circuit); mean minor radius of the core; helix phase arg sum w exp(i (theta_pol - N
theta_tor)) (spring advance / rotation). Full state (L, R) at plot times in /bk/dst/2026/10/02/coil/<name>/.
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
p.add_argument("--xi", type=float, default=3.9)
p.add_argument("--rho0", type=float, default=1.0)
p.add_argument("--N", type=int, default=8, help="poloidal turns per toroidal circuit")
p.add_argument("--S", type=int, default=1, help="parallel strands")
p.add_argument("--q", type=int, default=1)
p.add_argument("--a", type=float, default=10.0)
p.add_argument("--r0", type=float, default=30.0)
p.add_argument("--R", type=float, default=72.0)
p.add_argument("--T", type=int, default=1200)
p.add_argument("--every", type=int, default=20)
p.add_argument("--n_plot", type=int, default=4)
p.add_argument("--damp", type=float, default=0.05)
p.add_argument("--seg_per_turn", type=int, default=96)
p.add_argument("--tag", default="")
args = p.parse_args()
c2 = 1.0 / 3.0
xi = args.xi
g = c2 / (2 * args.m * args.rho0 * xi**2)
name = f"spring_N{args.N}_S{args.S}_q{args.q}_a{args.a:g}_r{args.r0:g}_xi{xi:g}_m{args.m:g}_R{args.R:g}{args.tag}"

t0 = time.time()
lat = DampedLattice(radius_cells=args.R, sponge_fraction=0.25)
pos = lat.pos
x, y, z = pos.T
rho_c = np.hypot(x, y)
th_c = np.arctan2(y, x)
th_p = np.arctan2(z, rho_c - args.r0)
r_all = np.linalg.norm(pos, axis=1)
lam, V = np.linalg.eig(symbol_dirac(np.zeros(3), args.m))
om_all = -np.angle(lam)
v = V[:, int(np.argmax(om_all))]
om0 = om_all.max()
a0 = np.sqrt(args.rho0)


def strand(j):
    s = np.linspace(0, 2 * np.pi, args.N * args.seg_per_turn, endpoint=False)
    tp = args.N * s + 2 * np.pi * j / args.S
    rr = args.r0 + args.a * np.cos(tp)
    return np.stack([rr * np.cos(s), rr * np.sin(s), args.a * np.sin(tp)], 1)


APEX = np.array([0.3183, -0.2718, 0.5772])


def solid_angle(P, X, chunk=3000):
    """solid angle of the closed polygon P (M, 3) seen from each X (n, 3), cone to a generic apex (no lattice cell on
    the cone), mod 4 pi."""
    A, B = P, np.roll(P, -1, 0)
    out = np.zeros(len(X))
    for i in range(0, len(X), chunk):
        x_ = X[i : i + chunk]
        a = APEX[None, None, :] - x_[:, None, :]
        b = A[None] - x_[:, None, :]
        c = B[None] - x_[:, None, :]
        la, lb, lc = np.linalg.norm(a, axis=2), np.linalg.norm(b, axis=2), np.linalg.norm(c, axis=2)
        num = np.einsum("nmk,nmk->nm", a, np.cross(b, c))
        den = (la * lb * lc + np.einsum("nmk,nmk->nm", a, b) * lc + np.einsum("nmk,nmk->nm", a, c) * lb
               + np.einsum("nmk,nmk->nm", b, c) * la)
        out[i : i + chunk] = 2 * np.arctan2(num, den).sum(1)
    return out


strands = [strand(j) for j in range(args.S)]
Om = sum(solid_angle(P, pos) for P in strands)
d = cKDTree(np.concatenate(strands)).query(pos)[0]
chi = a0 * d / np.sqrt(d**2 + xi**2) * np.exp(1j * args.q * Om / 2)


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
tree = cKDTree(pos)
ps = np.linspace(0, 2 * np.pi, 256, endpoint=False)
rm = args.a + 3 * xi


def probes(rc, zc):
    """centre circle and meridional circle (y = 0, x > 0) of the torus centred at (rho, z) = (rc, zc)."""
    pc = tree.query(np.stack([rc * np.cos(ps), rc * np.sin(ps), zc + 0 * ps], 1))[1]
    pm = tree.query(np.stack([rc + rm * np.cos(ps), 0 * ps, zc + rm * np.sin(ps)], 1))[1]
    return pc, pm


snap_dir = Path("/bk/dst/2026/10/02/coil") / name
snap_dir.mkdir(parents=True, exist_ok=True)
np.save(snap_dir / "pos.npy", pos)


print(f"{name}: N_cells={lat.N} build+seed {time.time() - t0:.0f}s; g {g:.4f} xi {xi:.2f}; strand pitch "
      f"(toroidal advance per turn) {2 * np.pi * args.r0 / args.N:.1f} cells at r0, inner-side turn spacing "
      f"{2 * np.pi * (args.r0 - args.a) / (args.N * args.S):.1f}", flush=True)


def amp():
    Lc = w.L[:, 0::2] + 1j * w.L[:, 1::2]
    Rc = w.R[:, 0::2] + 1j * w.R[:, 1::2]
    return np.concatenate([Lc, Rc], 1) @ v.conj()


def winding(a_, idx):
    return np.sum(np.angle(a_[idx][np.r_[1 : len(idx), 0]] / a_[idx])) / (2 * np.pi)


def state():
    """co-moving: the torus centre (rc, zc) = core-weighted mean (rho, z); probes and minor radius relative to it."""
    rho_ = w.density()
    a_ = amp()
    core = (rho_ < 0.25 * args.rho0) & inner
    n = core.sum()
    if not n:
        return [n, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    wt = (args.rho0 - rho_)[core]
    rc, zc = np.sum(wt * rho_c[core]) / wt.sum(), np.sum(wt * z[core]) / wt.sum()
    pc, pm = probes(rc, zc)
    wc, wm = winding(a_, pc), winding(a_, pm)
    dm = np.hypot(rho_c[core] - rc, z[core] - zc)
    thp = np.arctan2(z[core] - zc, rho_c[core] - rc)
    hel = np.angle(np.sum(wt * np.exp(1j * (thp - args.N * th_c[core]))))
    return [n, np.sum(wt * dm) / wt.sum(), hel, wc, wm, rc, zc]


def plot(tag, rc=None, zc=0.0):
    rc = args.r0 if rc is None or not np.isfinite(rc) else rc
    zc = 0.0 if not np.isfinite(zc) else zc
    from scipy.interpolate import griddata

    rho_ = w.density()
    lim = args.r0 + args.a + 4 * xi
    gg = np.linspace(-lim, lim, 401)
    G1, G2 = np.meshgrid(gg, gg)
    fig, ax = plt.subplots(1, 3, figsize=(19, 6))
    m_ = (np.abs(z) < 0.3) & (np.abs(x) < lim + 2) & (np.abs(y) < lim + 2)
    im = ax[0].imshow(griddata((x[m_], y[m_]), rho_[m_], (G1, G2), method="linear"), origin="lower",
                      extent=(-lim, lim, -lim, lim), cmap="viridis")
    ax[0].set_title("density (xy, z = 0)")
    fig.colorbar(im, ax=ax[0], shrink=0.8)
    core = (rho_ < 0.25 * args.rho0) & inner
    sc = ax[1].scatter(x[core], y[core], c=z[core] - zc, s=1, cmap="coolwarm", vmin=-args.a - xi,
                    vmax=args.a + xi)
    for P in strands:
        ax[1].plot(*P[:, :2].T, "k-", lw=0.3, alpha=0.4)
    ax[1].set_xlim(-lim, lim)
    ax[1].set_ylim(-lim, lim)
    ax[1].set_aspect("equal")
    ax[1].set_title(f"core cells (rho < 0.25), colour z - zc (zc {zc:+.1f}); grey = seed strand")
    fig.colorbar(sc, ax=ax[1], shrink=0.8)
    lim2 = args.a + 4 * xi
    m2 = (np.abs(y) < 0.3) & (x > 0) & (np.abs(x - rc) < lim2 + 2) & (np.abs(z - zc) < lim2 + 2)
    g2 = np.linspace(-lim2, lim2, 201)
    H1, H2 = np.meshgrid(rc + g2, zc + g2)
    im = ax[2].imshow(griddata((x[m2], z[m2]), rho_[m2], (H1, H2), method="linear"), origin="lower",
                      extent=(rc - lim2, rc + lim2, zc - lim2, zc + lim2), cmap="viridis")
    ax[2].set_title("density, meridional plane y = 0, centred on (rc, zc)")
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
        st = state()
        rec.append([t] + st)
        if t % (plot_every // 2) == 0:
            print(f"  t {t}: core cells {st[0]}, minor radius {st[1]:.1f}, helix phase {st[2]:+.3f}, winding centre "
                  f"circle {st[3]:+.2f}, meridional {st[4]:+.2f}, centre (rho, z) ({st[5]:.1f}, {st[6]:+.1f})", flush=True)
        if t % plot_every == 0:
            np.savez(snap_dir / f"state_t{t:05d}.npz", t=t, L=w.L, R=w.R)
        if t and t % plot_every == 0:
            plot(f"t{t:05d}", st[5], st[6])
    if t < args.T:
        w.step()
rec = np.array(rec, float)
print(f"  {args.T} ticks in {time.time() - t0:.0f}s", flush=True)
np.savez(here / f"{name}_rec.npz", rec=rec, args=str(vars(args)), xi=xi, g=g)
