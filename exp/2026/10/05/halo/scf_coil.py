"""Self-consistent relaxation of a wound coil (Philip 10-05: the seed starts out of equilibrium; the loss to the
background may come from that). Periodic box, no sponge.

Cycle: evolve --window ticks; every --every ticks take the k = 0 amplitude a = psi . v*, bring it to the frame of
the window start (rotation about z by the helix-phase advance / N, shift by the zc drift, both via a periodic
nearest-cell or k-cell inverse-distance map), remove the global phase (mean a over the far region), accumulate. The mean over the window keeps
the coherent part and averages out radiation; it is renormalised to the total norm and reseeded as psi = a v.

Per cycle: shed = far-region gradient energy at the end of the window minus right after the reseed (energy the
structure gives to the background in one window); d_avg = rms change of the window mean vs the previous cycle's
over the structure region; proj = fraction of |psi|^2 outside the v projection at the window end (what each
reseed throws away); windings centre / meridional, ring radius, minor radius.
"""
import argparse
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from equilibrium.substrate.walk import CppWalk, cpp_box, healing_g, to_complex, to_real

p = argparse.ArgumentParser()
p.add_argument("--start", required=True, help="state_t*.npz (box run), or avg_c*.npy from an earlier cycle (with --t0)")
p.add_argument("--t0", type=int, default=0, help="tick for an avg_c*.npy start (keeps the parity phase)")
p.add_argument("--k", type=int, default=8, help="cells in the inverse-distance remap (1 = nearest cell)")
p.add_argument("--P", type=int, default=300)
p.add_argument("--N", type=int, default=4)
p.add_argument("--a", type=float, default=10.0)
p.add_argument("--m", type=float, default=0.8)
p.add_argument("--xi", type=float, default=3.9)
p.add_argument("--window", type=int, default=1000)
p.add_argument("--every", type=int, default=25)
p.add_argument("--cycles", type=int, default=6)
p.add_argument("--threads", type=int, default=8)
p.add_argument("--out", default="/bk/dst/2026/10/05/scf")
p.add_argument("--tag", default="")
args = p.parse_args()

g = healing_g(args.m, args.xi, 1.0, walk=True)
alat = cpp_box(args.P, n_threads=args.threads)
w = CppWalk(alat, 0, m=args.m, g=g, rho0=1.0, parity=True)
v = w.v
side = args.P / np.sqrt(3)
pos = np.array(alat[0].leaf_pos)
x, y, z = pos.T
rho_c, th_c = np.hypot(x, y), np.arctan2(y, x)
lo = pos.min(0)
tree = cKDTree(pos - lo, boxsize=side)
pairs = tree.query_pairs(1.0 + 1e-6, output_type="ndarray")
ps = np.linspace(0, 2 * np.pi, 256, endpoint=False)
name = f"scf_N{args.N}_a{args.a:g}_P{args.P}_win{args.window}_k{args.k}{args.tag}"
out = Path(args.out) / name
out.mkdir(parents=True, exist_ok=True)

if args.start.endswith(".npy"):
    psi0 = np.load(args.start)[:, None] * v[None, :]
    w.set_state(to_real(psi0[:, :2]), to_real(psi0[:, 2:]), t=args.t0)
    del psi0
else:
    ck = np.load(args.start)
    w.set_state(ck["L"], ck["R"], t=int(ck["t"]))
    del ck


def snap():
    L, R = w.state()
    psi = np.concatenate([to_complex(L), to_complex(R)], 1)
    a = psi @ v.conj()
    rho = np.sum(np.abs(psi) ** 2, 1)
    return a, rho, psi


def geom(a, rho):
    core = rho < 0.25
    wt = (1 - rho)[core]
    rc, zc = np.sum(wt * rho_c[core]) / wt.sum(), np.sum(wt * z[core]) / wt.sum()
    thp = np.arctan2(z[core] - zc, rho_c[core] - rc)
    hel = np.angle(np.sum(wt * np.exp(1j * (thp - args.N * th_c[core]))))
    mr = np.sum(wt * np.hypot(rho_c[core] - rc, z[core] - zc)) / wt.sum()
    dt = np.hypot(rho_c - rc, z - zc)
    return dict(rc=rc, zc=zc, hel=hel, mr=mr, dt=dt, core=core.sum())


def windings(a, rc, zc):
    rm = args.a + 3 * args.xi
    out_ = []
    for P_ in (np.stack([rc * np.cos(ps), rc * np.sin(ps), zc + 0 * ps], 1),
               np.stack([rc + rm * np.cos(ps), 0 * ps, zc + rm * np.sin(ps)], 1)):
        idx = tree.query(np.mod(P_ - lo, side))[1]
        out_.append(np.sum(np.angle(a[idx][np.r_[1:len(idx), 0]] / a[idx])) / (2 * np.pi))
    return out_


def far_energy(a, far):
    e = np.abs(a[pairs[:, 1]] - a[pairs[:, 0]]) ** 2 / 2
    return np.sum(e[far[pairs[:, 0]] & far[pairs[:, 1]]])


A_C = 2 / np.sqrt(3)


def aligned(a, alpha, dz, far, dt):
    """z: exact lattice shift (whole lattice vectors) everywhere; rotation about z (and the sub-lattice z rest) by a
    k-cell inverse-distance map only near the structure, blended to none between a + 4 xi and a + 8 xi (the box is
    not rotation invariant; far from the tube the field is axisymmetric)."""
    nz = int(round(dz / A_C))
    src0 = tree.query(np.mod(pos + [0, 0, nz * A_C] - lo, side))[1]
    b = a[src0]
    s_ = np.clip((args.a + 8 * args.xi - dt) / (4 * args.xi), 0, 1)
    near = s_ > 0
    ca, sa = np.cos(alpha), np.sin(alpha)
    xn, yn, zn = x[near], y[near], z[near]
    Q = np.stack([ca * xn - sa * yn, sa * xn + ca * yn, zn + dz], 1)
    dd, src = tree.query(np.mod(Q - lo, side), k=args.k, workers=args.threads)
    if args.k == 1:
        r_ = a[src]
    else:
        wk = 1.0 / np.maximum(dd, 1e-6) ** 2
        r_ = np.sum(wk * a[src], 1) / wk.sum(1)
    b[near] = s_[near] * r_ + (1 - s_[near]) * b[near]
    gph = np.mean(b[far])
    return b * np.conj(gph) / np.abs(gph)


prev = None
t_wall = time.time()
for cyc in range(args.cycles + 1):
    a, rho, psi = snap()
    G = geom(a, rho)
    far = G["dt"] > args.a + 8 * args.xi
    wc, wm = windings(a, G["rc"], G["zc"])
    E0 = far_energy(a, far)
    print(f"cycle {cyc} t {w.t}: start  rc {G['rc']:.1f} zc {G['zc']:+.1f} minor {G['mr']:.1f} core {G['core']} "
          f"windings {wc:+.2f} / {wm:+.2f}  E_far {E0:.1f}", flush=True)
    if cyc == args.cycles:
        break
    norm = np.sum(rho)
    hel0, zc0, hel_prev, unw = G["hel"], G["zc"], G["hel"], 0.0
    acc = np.zeros(len(pos), complex)
    K = 0
    for k in range(args.window // args.every):
        w.step(args.every)
        a, rho, psi = snap()
        Gk = geom(a, rho)
        d = np.angle(np.exp(1j * (Gk["hel"] - hel_prev)))
        unw += d
        hel_prev = Gk["hel"]
        acc += aligned(a, -unw / args.N, Gk["zc"] - zc0, far, G["dt"])
        K += 1
    proj = 1 - np.sum(np.abs(a) ** 2) / np.sum(rho)
    E1 = far_energy(a, far)
    wc1, wm1 = windings(a, Gk["rc"], Gk["zc"])
    avg = acc / K
    avg *= np.sqrt(norm / np.sum(np.abs(avg) ** 2))
    reg = G["dt"] < args.a + 4 * args.xi
    dav = np.nan if prev is None else np.sqrt(np.mean(np.abs(avg[reg] - prev[reg]) ** 2))
    prev = avg
    print(f"  end t {w.t}: shed (E_far end - start) {E1 - E0:+.1f}; rotation {np.degrees(-unw / args.N):+.1f} deg, "
          f"dz {Gk['zc'] - zc0:+.1f}; windings {wc1:+.2f} / {wm1:+.2f}; proj loss {proj:.2e}; d_avg {dav:.4f}; "
          f"wall {time.time() - t_wall:.0f}s", flush=True)
    np.save(out / f"avg_c{cyc}.npy", avg)
    psi_new = avg[:, None] * v[None, :]
    w.set_state(to_real(psi_new[:, :2]), to_real(psi_new[:, 2:]), t=w.t)
