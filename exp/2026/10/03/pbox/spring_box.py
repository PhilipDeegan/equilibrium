"""Periodic-box version of coil/spring_cpp.py (exp/2026/10/03/pbox/status.md): build_bcc_box_periodic of period P
(even, grid units h = 1/sqrt3; side P h = neighbour spacing), no sponge, no ghosts; seed phase = q/2 x solid angle
summed over the 27 nearest images (the 26 outer ones from a coarse polygon), minus the uniform term that removes
the mean phase jump across each face; recentre = exact periodic shift along z.

C++ walk version of spring.py (AdaptiveLattice::step_walk; same seed, scores, plots), plus --z0 (start height) and
--sponge. Spring coil (Philip 10-02: a coil is a spring, each turn advances toroidally -- the 10-02 meridional loops were
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

from equilibrium.substrate.walk import CppWalk, cpp_box, healing_g

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
p.add_argument("--P", type=int, default=346, help="box period in grid units (even); side = P / sqrt3")
p.add_argument("--img_seg", type=int, default=12, help="polygon segments per turn for the 26 outer images")
p.add_argument("--T", type=int, default=1200)
p.add_argument("--every", type=int, default=20)
p.add_argument("--n_plot", type=int, default=4)
p.add_argument("--damp", type=float, default=0.05)
p.add_argument("--seg_per_turn", type=int, default=96)
p.add_argument("--z0", type=float, default=0.0)
p.add_argument("--theta0", type=float, default=0.0, help="rotate the seed strand about z by this many degrees")
p.add_argument("--walk_xi", action="store_true", help="xi = true core width (walk dispersion n/(6m)), not the c^2 = 1/3 label")
p.add_argument("--threads", type=int, default=8)
p.add_argument("--recentre", type=int, default=0, help="every this many ticks shift the state by whole lattice vectors along z so zc ~ 0 (0 = off)")
p.add_argument("--parity", action="store_true", help="alternate with the parity-image tick (walk.py parity; aniso/status.md)")
p.add_argument("--tag", default="")
p.add_argument("--resume", default="", help="state_t*.npz to continue from (its t, z_shift); seed skipped")
p.add_argument("--snap_root", default="/bk/dst/2026/10/03/pbox")
args = p.parse_args()
c2 = 1.0 / 3.0
xi = args.xi
g = healing_g(args.m, xi, args.rho0, walk=args.walk_xi)
name = f"springbox_N{args.N}_S{args.S}_q{args.q}_a{args.a:g}_r{args.r0:g}_xi{xi:g}_m{args.m:g}_P{args.P}_z{args.z0:g}{'_wxi' if args.walk_xi else ''}{'_rc' if args.recentre else ''}{f'_th{args.theta0:g}' if args.theta0 else ''}{'_par' if args.parity else ''}{args.tag}"

run_dir = here / Path(*name.split("_"))
run_dir.mkdir(parents=True, exist_ok=True)
t0 = time.time()
alat = cpp_box(args.P, n_threads=args.threads)
side = args.P / np.sqrt(3)
pos = np.array(alat[0].leaf_pos)
x, y, z = pos.T
rho_c = np.hypot(x, y)
th_c = np.arctan2(y, x)
r_all = np.linalg.norm(pos, axis=1)
a0 = np.sqrt(args.rho0)


def strand(j, seg=None):
    s = np.linspace(0, 2 * np.pi, args.N * (seg or args.seg_per_turn), endpoint=False)
    tp = args.N * s + 2 * np.pi * j / args.S
    rr = args.r0 + args.a * np.cos(tp)
    s0 = s + np.radians(args.theta0)
    return np.stack([rr * np.cos(s0), rr * np.sin(s0), args.z0 + args.a * np.sin(tp)], 1)


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
r_inner = np.inf
w = CppWalk(alat, 0, m=args.m, g=g, rho0=args.rho0, parity=args.parity)
ix = np.rint(pos * np.sqrt(3)).astype(np.int64)
H = args.P // 2
lut = np.full((args.P,) * 3, -1, np.int64)
lut[ix[:, 0] + H, ix[:, 1] + H, ix[:, 2] + H] = np.arange(len(pos))
t_start, z_shift = 0, 0.0  # total lattice shift applied (true z = z + z_shift)
if args.resume:
    ck = np.load(args.resume)
    t_start, z_shift = int(ck["t"]), float(ck["z_shift"])
    w.set_state(ck["L"], ck["R"], t=t_start)
    del ck
else:
    coarse = [strand(j, args.img_seg) for j in range(args.S)]
    shifts = [np.array([a, b, c]) * side for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if a or b or c]

    def phase(X):
        return 0.5 * args.q * (sum(solid_angle(P, X) for P in strands)
                               + sum(solid_angle(P + d, X) for d in shifts for P in coarse))

    rng_ = np.random.default_rng(0)
    jumps = []
    for ax in range(3):  # mean jump across the face normal to ax -> uniform term
        smp = (rng_.random((400, 3)) - 0.5) * side
        smp[:, ax] = -side / 2
        hi = smp.copy()
        hi[:, ax] += side
        jump = np.angle(np.mean(np.exp(1j * (phase(hi) - phase(smp)))))
        print(f"  seed: mean phase jump across face {'xyz'[ax]} {jump:+.4f}", flush=True)
        jumps.append(jump)
    Om = 2 * (phase(pos) - sum(jumps[a] * pos[:, a] / side for a in range(3))) / args.q
    d = cKDTree(np.concatenate(strands)).query(pos)[0]
    w.seed(a0 * d / np.sqrt(d**2 + xi**2) * np.exp(1j * args.q * Om / 2))
    del Om, d
inner = np.ones(len(pos), bool)
assert max(np.abs(P).max() for P in strands) + 3 * xi < side / 2
tree = cKDTree(pos)
ps = np.linspace(0, 2 * np.pi, 256, endpoint=False)
rm = args.a + 3 * xi


def probes(rc, zc):
    """centre circle and meridional circle (y = 0, x > 0) of the torus centred at (rho, z) = (rc, zc)."""
    pc = tree.query(np.stack([rc * np.cos(ps), rc * np.sin(ps), zc + 0 * ps], 1))[1]
    pm = tree.query(np.stack([rc + rm * np.cos(ps), 0 * ps, zc + rm * np.sin(ps)], 1))[1]
    return pc, pm


snap_dir = Path(args.snap_root) / name
snap_dir.mkdir(parents=True, exist_ok=True)
np.save(snap_dir / "pos.npy", pos)


print(f"{name}: N_cells={len(pos)} build+seed {time.time() - t0:.0f}s; g {g:.4f} xi {xi:.2f}; strand pitch "
      f"(toroidal advance per turn) {2 * np.pi * args.r0 / args.N:.1f} cells at r0, inner-side turn spacing "
      f"{2 * np.pi * (args.r0 - args.a) / (args.N * args.S):.1f}", flush=True)


def amp():
    return w.amp()


def winding(a_, idx):
    return np.sum(np.angle(a_[idx][np.r_[1 : len(idx), 0]] / a_[idx])) / (2 * np.pi)


A_C = 2 / np.sqrt(3)  # conventional cube edge = lattice vector along z (hex distance 1)


def recentre(zc):
    """shift the state down by n whole lattice vectors along z (n = round(zc / A_C)): cell j takes the state of the
    cell at pos_j + n A_C z^, wrapped (periodic box). Exact on the lattice."""
    global z_shift
    nz = int(round(zc / A_C))
    if nz == 0:
        return 0
    src = lut[ix[:, 0] + H, ix[:, 1] + H, (ix[:, 2] + 2 * nz + H) % args.P]
    L, R = w.state()
    w.set_state(L[src], R[src])
    z_shift += nz * A_C
    return nz


def state():
    """co-moving: the torus centre (rc, zc) = core-weighted mean (rho, z); probes and minor radius relative to it.
    Also the deficit sum(rho0 - rho) within a + 4 xi (D_near) and a + 1.5 xi (D_core) of the centre circle."""
    rho_ = w.density()
    a_ = amp()
    core = (rho_ < 0.25 * args.rho0) & inner
    n = core.sum()
    if not n:
        return [n] + [np.nan] * 8 + [z_shift]
    wt = (args.rho0 - rho_)[core]
    rc, zc = np.sum(wt * rho_c[core]) / wt.sum(), np.sum(wt * z[core]) / wt.sum()
    pc, pm = probes(rc, zc)
    wc, wm = winding(a_, pc), winding(a_, pm)
    dm = np.hypot(rho_c[core] - rc, z[core] - zc)
    thp = np.arctan2(z[core] - zc, rho_c[core] - rc)
    hel = np.angle(np.sum(wt * np.exp(1j * (thp - args.N * th_c[core]))))
    dt_ = np.hypot(rho_c - rc, z - zc)
    Dn = np.sum((args.rho0 - rho_)[(dt_ < args.a + 4 * xi) & inner])
    Dc = np.sum((args.rho0 - rho_)[(dt_ < args.a + 1.5 * xi) & inner])
    return [n, np.sum(wt * dm) / wt.sum(), hel, wc, wm, rc, zc, Dn, Dc, z_shift]


def plot(tag, rc=None, zc=0.0):
    rc = args.r0 if rc is None or not np.isfinite(rc) else rc
    zc = 0.0 if not np.isfinite(zc) else zc
    from scipy.interpolate import griddata

    rho_ = w.density()
    lim = side / 2
    gg = np.linspace(-lim, lim, 401)
    G1, G2 = np.meshgrid(gg, gg)
    fig, ax = plt.subplots(1, 3, figsize=(19, 6))
    zp = z[np.argmin(np.abs(z - zc))]
    m_ = (np.abs(z - zp) < 0.05) & (np.abs(x) < lim + 2) & (np.abs(y) < lim + 2)
    im = ax[0].imshow(griddata((x[m_], y[m_]), rho_[m_], (G1, G2), method="linear"), origin="lower",
                      extent=(-lim, lim, -lim, lim), cmap="viridis")
    ax[0].set_title(f"density, xy lattice plane z = {zp:+.2f} (nearest zc)")
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
    fig.suptitle(f"{name}  t = {w.t}  (frame shifted by {z_shift:+.1f} in z)")
    fig.tight_layout()
    fig.savefig(run_dir / f"{tag}.png", dpi=80)
    plt.close(fig)


plot(f"t{t_start:05d}" if t_start else "t0000")
rec = []
t0 = time.time()
plot_every = max(args.every, (args.T // args.n_plot) // args.every * args.every)
for t in range(t_start, args.T + 1, args.every):
    if True:
        st = state()
        rec.append([t] + st)
        if t % (plot_every // 2) < args.every:
            print(f"  t {t}: core cells {st[0]}, minor radius {st[1]:.1f}, helix phase {st[2]:+.3f}, winding centre "
                  f"circle {st[3]:+.2f}, meridional {st[4]:+.2f}, centre (rho, z) ({st[5]:.1f}, {st[6]:+.1f}), "
                  f"D_near {st[7]:.0f} D_core {st[8]:.0f}, frame shift {st[9]:+.1f}", flush=True)
        if t % plot_every < args.every and (t > t_start or not args.resume):
            Ls, Rs = w.state()
            np.savez(snap_dir / f"state_t{t:05d}.npz", t=t, L=Ls, R=Rs, z_shift=z_shift)
        if t and t % plot_every < args.every:
            plot(f"t{t:05d}", st[5], st[6])
    if args.recentre and t > t_start and t % args.recentre < args.every and np.isfinite(st[6]):
        recentre(st[6])
    if t < args.T:
        w.step(args.every)
rec = np.array(rec, float)
print(f"  {args.T} ticks in {time.time() - t0:.0f}s", flush=True)
np.savez(run_dir / (f"rec_from{t_start}.npz" if t_start else "rec.npz"), rec=rec, args=str(vars(args)), xi=xi, g=g)
