"""Shell profile about the tube in a periodic box (next 2026-10-07 item 0a/b): per bin of dt (distance from the tube's
centre circle, co-moving centre (rc, zc) as in shed/quasi_energy.py regions) the mean deficit 1 - rho and the walk's
energy density e = e_q + e_nl relative to the background (quasi_energy.py local split), over cells nearer their own
tube than any of the 26 images' (own Voronoi part); plus the same means on the midplane |dt - dt_img| < 2.
Bound extent = where the profile ends; does it follow the spacing (half the tube-to-image distance)?"""
import argparse
import sys
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--dir", action="append", required=True)
p.add_argument("--P", type=int, action="append", required=True)
p.add_argument("--t", type=int, action="append", required=True)
p.add_argument("--m", type=float, default=0.8)
p.add_argument("--xi", type=float, default=3.9)
p.add_argument("--r0", type=float, default=30.0)
p.add_argument("--dr", type=float, default=2.0)
p.add_argument("--ref_mid", action="store_true", help="deficit / e relative to the midplane means (the box's own far level), not rho 1")
args = p.parse_args()

sys.argv = ["qe", "--m", str(args.m), "--r0", str(args.r0)]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shed"))
import quasi_energy as qe  # noqa: E402
from equilibrium.substrate.walk import healing_g, to_complex  # noqa: E402

g = healing_g(args.m, args.xi, 1.0, walk=True)


def tube_dist(pos, rc, zc, shift):
    x, y, z = (pos + shift).T
    return np.hypot(np.hypot(x, y) - rc, z - zc)


for d, P in zip(map(Path, args.dir), args.P):
    pos = np.load(d / "pos.npy")
    ix = (np.rint(pos * np.sqrt(3)).astype(np.int64) + P // 2) % P
    side = P / np.sqrt(3)
    for t in args.t:
        f = d / f"state_t{t:05d}.npz"
        if not f.exists():
            continue
        ck = np.load(f)
        L, R = ck["L"], ck["R"]
        psi = np.concatenate([to_complex(L), to_complex(R)], 1)
        rho = np.sum(L**2, 1) + np.sum(R**2, 1)
        del L, R, ck
        _, _, e = qe.quasi_energy(psi, ix, P, local=True)
        del psi
        e = e - qe.om0 * rho + 0.5 * g * rho * (rho - 1.0)
        rho_c = np.hypot(pos[:, 0], pos[:, 1])
        cc = (rho < 0.25) & (rho_c > 0.5 * args.r0)
        wt = (1 - rho)[cc]
        kz = 2 * np.pi / side
        rc = np.sum(wt * rho_c[cc]) / wt.sum()
        zc = np.angle(np.sum(wt * np.exp(1j * kz * pos[cc, 2]))) / kz
        pz = pos.copy()
        pz[:, 2] = (pz[:, 2] - zc + side / 2) % side - side / 2
        dt = tube_dist(pz, rc, 0.0, 0.0)
        dimg = np.full_like(dt, np.inf)
        for sx in (-1, 0, 1):
            for sy in (-1, 0, 1):
                for sz in (-1, 0, 1):
                    if sx or sy or sz:
                        dimg = np.minimum(dimg, tube_dist(pz, rc, 0.0, np.array([sx, sy, sz]) * side))
        own = dt < dimg
        mid = np.abs(dt - dimg) < 2.0
        print(f"{d.name} P {P} side {side:.1f} t {t}: rc {rc:.1f} zc {zc:+.1f}; tube-to-image gap along x "
              f"{side - 2 * rc:.1f}, along z {side:.1f}; e total {e.sum():+.2f}, deficit total {np.sum(1 - rho):.1f}",
              flush=True)
        print(f"  midplane: cells {mid.sum()}, dt {dt[mid].mean():.1f} [{dt[mid].min():.1f}, {dt[mid].max():.1f}], "
              f"<1-rho> {np.mean(1 - rho[mid]):+.3e}, <e> {np.mean(e[mid]):+.3e}", flush=True)
        if args.ref_mid:
            r_ref, e_ref = np.mean(rho[mid]), np.mean(e[mid])
            rho, e = rho + (1 - r_ref), e - e_ref
            print(f"  relative to the midplane: rho_ref {r_ref:.5f}, e_ref {e_ref:+.3e}", flush=True)
        print("    dt    <1-rho>     <e>       cum D    cum e    N_cells(own)", flush=True)
        cD, cE = 0.0, 0.0
        for b in np.arange(0, dt[own].max() + args.dr, args.dr):
            sh = own & (dt >= b) & (dt < b + args.dr)
            if sh.sum() < 50:
                continue
            cD += np.sum(1 - rho[sh])
            cE += np.sum(e[sh])
            print(f"  {b + args.dr / 2:5.1f}  {np.mean(1 - rho[sh]):+.3e}  {np.mean(e[sh]):+.3e}  {cD:8.1f}  {cE:+8.2f}  "
                  f"{sh.sum()}", flush=True)
