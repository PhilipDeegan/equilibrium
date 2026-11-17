"""Two-component exclusion test (Philip 10-04: what makes a two-component structure 'hard' while waves superpose).

Periodic BCC box (cpp_box, parity walk), background = the degenerate rest pair mixed: a0 (cos b v1 + sin b v2),
v1 / v2 the two k = 0 positive-frequency rest states (same frequency arcsin m; the density turn uses the total
rho = |v1|^2 + |v2|^2, symmetric under turning one into the other). Plain vortex rings (centre on the z axis, plane
z = z_k, radius r_k, sign q_k), each either FULL (the whole state winds: both components) or in ONE component (only that
component winds and has a zero at the core; the other passes through unwound and fills it). Seed per component c:
amplitude w_c prod_k env_k exp(i sum_k q_k Omega_k / 2) over the rings acting on c, env = d / sqrt(d^2 + xi^2),
Omega = solid angle (27 images, C++), minus the mean face jump (uniform term), as in pbox/spring_box.py.

Diagnostics every --every ticks: per component c, cells with |a_c|^2 < 0.25 w_c^2 rho0 (component core) -> connected
pieces (periodic): size, centroid z, mean radius about the z axis; total-density core pieces likewise; per component the
phase-winding zeros in the y = 0 cut (cut_zeros.Cut, topological).
--ring r,z,q,comp (comp: full, 1, 2), repeatable.
"""
import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from cut_zeros import Cut, fmt_zeros
from equilibrium.substrate.walk import CppWalk, cpp_box, healing_g, solid_angle, symbol_dirac, to_complex, to_real

here = Path(__file__).parent
p = argparse.ArgumentParser()
p.add_argument("--m", type=float, default=0.8)
p.add_argument("--xi", type=float, default=3.9)
p.add_argument("--P", type=int, default=300)
p.add_argument("--mix", type=float, default=45.0, help="background angle b (deg): cos b v1 + sin b v2")
p.add_argument("--ring", action="append", required=True, help="r,z,q,comp  (comp full / 1 / 2)")
p.add_argument("--T", type=int, default=8000)
p.add_argument("--every", type=int, default=250)
p.add_argument("--n_plot", type=int, default=8)
p.add_argument("--state_every", type=int, default=4000)
p.add_argument("--threads", type=int, default=8)
p.add_argument("--tag", default="")
p.add_argument("--snap_root", default="/bk/dst/2026/10/04/pair")
args = p.parse_args()

rings = []
for s in args.ring:
    r_, z_, q_, c_ = s.split(",")
    rings.append((float(r_), float(z_), int(q_), c_))
xi, rho0, a0 = args.xi, 1.0, 1.0
g = healing_g(args.m, xi, rho0, walk=True)
b = np.radians(args.mix)
wc = {1: np.cos(b), 2: np.sin(b)}
name = "pair_" + "_".join(f"r{r:g}z{z:g}q{q}c{c}" for r, z, q, c in rings) + f"_mix{args.mix:g}_P{args.P}_xi{xi:g}{args.tag}"
run_dir = here / name
run_dir.mkdir(parents=True, exist_ok=True)
snap_dir = Path(args.snap_root) / name
snap_dir.mkdir(parents=True, exist_ok=True)

lam, V = np.linalg.eig(symbol_dirac(np.zeros(3), args.m))
pf = np.flatnonzero(-np.angle(lam) > 0)
vc = {1: V[:, pf[0]], 2: V[:, pf[1]]}

t0 = time.time()
alat = cpp_box(args.P, n_threads=args.threads)
pos = np.array(alat[0].leaf_pos)
side = args.P / np.sqrt(3)
x, y, z = pos.T
rho_c = np.hypot(x, y)
w = CppWalk(alat, 0, m=args.m, g=g, rho0=rho0, parity=True)
shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], float) * side


def circle(r, z0, n=512):
    s = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(s), r * np.sin(s), z0 + 0 * s], 1)


def phase(X, rs):
    return sum(0.5 * q * solid_angle(circle(r, z0), X, shifts=shifts, n_threads=args.threads) for r, z0, q, _ in rs)


psi = np.zeros((len(pos), 4), complex)
rng_ = np.random.default_rng(0)
for c in (1, 2):
    rs = [rg for rg in rings if rg[3] in ("full", str(c))]
    amp = np.full(len(pos), a0 * wc[c], complex)
    if rs:
        jumps = []
        for ax in range(3):
            smp = (rng_.random((400, 3)) - 0.5) * side
            smp[:, ax] = -side / 2
            hi = smp.copy()
            hi[:, ax] += side
            jumps.append(np.angle(np.mean(np.exp(1j * (phase(hi, rs) - phase(smp, rs))))))
        ph = phase(pos, rs) - sum(jumps[a] * pos[:, a] / side for a in range(3))
        env = np.ones(len(pos))
        for r, z0, q, _ in rs:
            dz = (z - z0 + side / 2) % side - side / 2
            d = np.hypot(rho_c - r, dz)
            env *= (d / np.sqrt(d**2 + xi**2)) ** abs(q)
        amp = amp * env * np.exp(1j * ph)
        print(f"  component {c}: rings {rs}, face jumps {np.round(jumps, 4)}", flush=True)
    psi += amp[:, None] * vc[c][None, :]
w.set_state(to_real(psi[:, :2]), to_real(psi[:, 2:]), t=0)
del psi
print(f"{name}: N_cells={len(pos)} build+seed {time.time() - t0:.0f}s; g {g:.4f} xi {xi}; side {side:.1f}", flush=True)

box = np.mod(pos + side / 2, side)
cut = Cut(pos)


def pieces(mask):
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    Q = box[idx]
    pr = cKDTree(Q, boxsize=side).query_pairs(1.05, output_type="ndarray")
    _, lab = connected_components(coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(Q),) * 2), directed=False)
    out = []
    for k in np.argsort(-np.bincount(lab)):
        sel = idx[lab == k]
        if len(sel) < 50:
            break
        zz = (np.angle(np.mean(np.exp(1j * box[sel, 2] / side * 2 * np.pi))) / (2 * np.pi) * side) % side - side / 2
        out.append((len(sel), zz, rho_c[sel].mean(), np.abs(((z[sel] - zz) + side / 2) % side - side / 2).std()))
    return out


def comps():
    L, R = w.state()
    ps = np.concatenate([to_complex(L), to_complex(R)], 1)
    zc = {c: cut.zeros(ps[cut.m] @ vc[c].conj())[0] for c in (1, 2)}
    return {c: np.abs(ps @ vc[c].conj()) ** 2 for c in (1, 2)}, (np.abs(ps) ** 2).sum(1), zc


def fmt(pc):
    return " | ".join(f"{n} cells z {zz:+.1f} r {rr:.1f}" for n, zz, rr, _ in pc[:4]) or "none"


def plot(tag, a2, rho):
    from scipy.interpolate import griddata

    lim = side / 2
    gg = np.linspace(-lim, lim, 301)
    G1, G2 = np.meshgrid(gg, gg)
    m_ = np.abs(y) < 0.6
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    for i, (f_, ttl) in enumerate(((a2[1], "|a1|^2"), (a2[2], "|a2|^2"), (rho, "rho total"))):
        im = ax[i].imshow(griddata((x[m_], z[m_]), f_[m_], (G1, G2), method="linear"), origin="lower",
                          extent=(-lim, lim, -lim, lim), cmap="viridis")
        ax[i].set_title(f"{ttl}, plane y ~ 0 (x, z)")
        fig.colorbar(im, ax=ax[i], shrink=0.8)
    fig.suptitle(f"{name}  t = {w.t}")
    fig.tight_layout()
    fig.savefig(run_dir / f"{tag}.png", dpi=80)
    plt.close(fig)


plot_every = max(args.every, (args.T // args.n_plot) // args.every * args.every)
rec = []
t1 = time.time()
for t in range(0, args.T + 1, args.every):
    a2, rho, zc = comps()
    p1 = pieces(a2[1] < 0.25 * wc[1] ** 2 * rho0) if wc[1] > 1e-6 else []
    p2 = pieces(a2[2] < 0.25 * wc[2] ** 2 * rho0) if wc[2] > 1e-6 else []
    pt = pieces(rho < 0.25 * rho0)
    rec.append(dict(t=t, p1=p1, p2=p2, pt=pt, z1=zc[1], z2=zc[2], rho_min=float(rho.min()), bg_sd=float(rho[rho > 0.5].std())))
    print(f"  t {t}: v1 cores [{fmt(p1)}]; v2 cores [{fmt(p2)}]; rho cores [{fmt(pt)}]; rho min {rho.min():.3f}, "
          f"bg sd {rho[rho > 0.5].std():.3f}; cut zeros v1 {fmt_zeros(zc[1])} v2 {fmt_zeros(zc[2])}", flush=True)
    if t % plot_every < args.every:
        plot(f"t{t:05d}", a2, rho)
    if t % args.state_every < args.every:
        L, R = w.state()
        np.savez(snap_dir / f"state_t{t:05d}.npz", t=t, L=L, R=R)
        np.save(snap_dir / "pos.npy", pos) if t == 0 else None
    if t < args.T:
        w.step(args.every)
print(f"  {args.T} ticks in {time.time() - t1:.0f}s", flush=True)
np.save(run_dir / "rec.npy", np.array(rec, dtype=object), allow_pickle=True)
