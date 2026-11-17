"""Deficit budget (lam = deficit? mass ~ deficit?): D = sum(rho0 - rho) and gradient energy E = sum over hex pairs
|a_j - a_i|^2 / 2 (uniform background has E = 0 exactly), split by distance dt to the tube centre circle (rc, zc):
interior dt < a - 1.5 xi, strands |dt - a| < 1.5 xi, near halo a + 1.5 xi .. a + 4 xi, outer halo a + 4 xi .. a + 10 xi,
axis (rho_cyl < rc - a - 4 xi, |z - zc| < a + 4 xi, not in a tube region), floor = the rest (inside the sponge for a
sphere, whole box for a periodic box). (rc, zc) from the coil-only core (rho < 0.25, rho_cyl > rc0 / 2).
Gradient energy on a sphere only over cells within dt < a + 12 xi or the axis region (pairs on the full 43M-cell
lattice are too heavy); floor E is then not computed."""
import argparse
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from equilibrium.substrate.walk import rest_state, to_complex

p = argparse.ArgumentParser()
p.add_argument("--state", action="append", required=True, help="path to state_t*.npz (pos.npy beside it)")
p.add_argument("--m", type=float, default=0.8)
p.add_argument("--a", type=float, default=10.0)
p.add_argument("--r0", type=float, required=True)
p.add_argument("--xi", type=float, default=3.9)
p.add_argument("--sponge", type=float, default=0.25, help="sphere only")
p.add_argument("--census_rho", type=float, default=0.6, help="off-structure piece threshold")
p.add_argument("--census_min", type=int, default=10)
p.add_argument("--box", type=float, default=0.0, help="periodic box side (0 = sphere)")
args = p.parse_args()
v, _ = rest_state(args.m)
a_, xi = args.a, args.xi
pos_cache = {}
for f in map(Path, args.state):
    if f.parent not in pos_cache:
        pos_cache.clear()
        pos_cache[f.parent] = np.load(f.parent / "pos.npy")
    pos = pos_cache[f.parent]
    rho_c = np.hypot(pos[:, 0], pos[:, 1])
    rr = np.linalg.norm(pos, axis=1)
    inner = np.ones(len(pos), bool) if args.box else rr < (1 - args.sponge) * rr.max()
    ck = np.load(f)
    L, R = ck["L"], ck["R"]
    t = int(ck["t"])
    a = np.concatenate([to_complex(L), to_complex(R)], 1) @ v.conj()
    rho = np.sum(L**2, 1) + np.sum(R**2, 1)
    del L, R, ck
    core = (rho < 0.25) & inner
    cc = core & (rho_c > 0.5 * args.r0)
    wt = (1 - rho)[cc]
    rc, zc = np.sum(wt * rho_c[cc]) / wt.sum(), np.sum(wt * pos[cc, 2]) / wt.sum()
    dz = pos[:, 2] - zc
    if args.box:
        dz = (dz + args.box / 2) % args.box - args.box / 2
    dt = np.hypot(rho_c - rc, dz)
    regs = {"interior": dt < a_ - 1.5 * xi, "strands": np.abs(dt - a_) < 1.5 * xi,
            "near": (dt >= a_ + 1.5 * xi) & (dt < a_ + 4 * xi), "outer": (dt >= a_ + 4 * xi) & (dt < a_ + 10 * xi)}
    tube = dt < a_ + 10 * xi
    regs["axis"] = (rho_c < rc - a_ - 4 * xi) & (np.abs(dz) < a_ + 4 * xi) & ~tube
    regs["floor"] = inner & ~tube & ~regs["axis"]
    for k in regs:
        regs[k] &= inner
    sel = np.nonzero((dt < a_ + 12 * xi) | regs["axis"] | (args.box > 0))[0]
    q = np.mod(pos[sel] + args.box / 2, args.box) if args.box else pos[sel]
    if args.box:
        q[q >= args.box] = 0.0
    tr = cKDTree(q, boxsize=args.box if args.box else None)
    pairs = tr.query_pairs(1.0 + 1e-6, output_type="ndarray")
    i, j = sel[pairs[:, 0]], sel[pairs[:, 1]]
    e = np.abs(a[j] - a[i]) ** 2 / 2
    ec = np.zeros(len(pos))
    np.add.at(ec, i, e / 2)
    np.add.at(ec, j, e / 2)
    del pairs, i, j, e, tr, q
    D = {k: np.sum((1 - rho)[m]) for k, m in regs.items()}
    E = {k: np.sum(ec[m]) for k, m in regs.items()}
    Dt, Et = sum(D.values()), sum(v_ for k, v_ in E.items() if args.box or k != "floor")
    Ds = D["interior"] + D["strands"] + D["near"]
    Es = E["interior"] + E["strands"] + E["near"]
    print(f"{f.parent.name} t {t}: rc {rc:.1f} zc {zc:+.1f} coil core {cc.sum()} axis core {(core & ~cc).sum()} "
          f"floor/cell {D['floor'] / max(regs['floor'].sum(), 1):+.2e}", flush=True)
    print("   D  " + "  ".join(f"{k} {D[k]:9.0f}" for k in regs) + f"  | total {Dt:.0f}", flush=True)
    print("   E  " + "  ".join(f"{k} {E[k]:9.1f}" for k in regs) + f"  | total {Et:.1f}"
          + ("" if args.box else " (floor E not computed)"), flush=True)
    print(f"   structure (interior + strands + near): D {Ds:.0f}  E {Es:.1f}  E/D {Es / Ds:.4f}", flush=True)
    off = np.nonzero((rho < args.census_rho) & inner & (dt >= a_ + 4 * xi))[0]
    if len(off):
        qo = np.mod(pos[off] + args.box / 2, args.box) if args.box else pos[off]
        if args.box:
            qo[qo >= args.box] = 0.0
        po = cKDTree(qo, boxsize=args.box if args.box else None).query_pairs(1.0 + 1e-6, output_type="ndarray")
        G = coo_matrix((np.ones(len(po)), (po[:, 0], po[:, 1])), shape=(len(off), len(off)))
        nc, lab = connected_components(G, directed=False)
        cl = []
        for c in range(nc):
            idx = off[lab == c]
            if len(idx) < args.census_min:
                continue
            cl.append((len(idx), rho[idx].min(), np.sum((1 - rho)[idx]), np.sum(ec[idx]),
                       rho_c[idx].mean(), pos[idx, 2].mean() - zc))
        cl.sort(key=lambda c_: -c_[2])
        print(f"   census off-structure (rho < {args.census_rho}, >= {args.census_min} cells): {len(cl)} pieces, "
              f"D {sum(c_[2] for c_ in cl):.0f}, E {sum(c_[3] for c_ in cl):.1f}", flush=True)
        for c_ in cl[:8]:
            print(f"     cells {c_[0]:6d} min rho {c_[1]:.2f} D {c_[2]:8.0f} E {c_[3]:7.1f} at rho_cyl {c_[4]:6.1f} "
                  f"z-zc {c_[5]:+6.1f}", flush=True)
