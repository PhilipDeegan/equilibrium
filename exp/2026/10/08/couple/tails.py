"""10-08 next item 1: which far field carries the same-sense pair coupling (excess drift ~ exp(-d / 31))?
For each saved final (q +1 pairs P 240 / 300 / 360, lone rp P 220; t 8000): per cell the deficit 1 - rho, the - branch
density |psi_-|^2 (quasi_energy.py branch_split), the phase gradient |grad arg a| and the current |Im(a* grad a)|
(a = psi . v*, central differences over the 4 hop pairs, sum_h h h^T = 4 I in grid units). Tails vs dt (distance to the
own tube centre circle, 27 images) in the own Voronoi part, split by what the cell is nearer to next: a partner tube
('->p') or an own image ('->i'); and the same fields at the equidistance surface (|dt - dt_other| < 2) per run."""
import argparse
import sys
from pathlib import Path

import numpy as np

from equilibrium.substrate.walk import HS, rest_state, to_complex

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "10/06/shed"))
sys.argv, argv_ = ["qe", "--m", "0.8"], sys.argv
import quasi_energy as qe  # noqa: E402

sys.argv = argv_
p = argparse.ArgumentParser()
p.add_argument("--state", action="append", default=[])
p.add_argument("--r0", type=float, default=30.0)
p.add_argument("--dr", type=float, default=4.0)
p.add_argument("--out", default="")
args = p.parse_args()
v, om0 = rest_state(0.8)


def circle_dist(pos, side, cx, cy, rc, zc):
    """distance to the nearest of the 27 images of the z-axis circle (centre (cx, cy, zc), radius rc)."""
    best = np.full(len(pos), np.inf)
    for sx in (-1, 0, 1):
        for sy in (-1, 0, 1):
            rr = np.hypot(pos[:, 0] - cx + sx * side, pos[:, 1] - cy + sy * side) - rc
            for sz in (-1, 0, 1):
                best = np.minimum(best, np.hypot(rr, (pos[:, 2] - zc + side / 2) % side - side / 2 + sz * side))
    return best


def centre(pos, rho, sel, cx, cy, side):
    rr = np.hypot(pos[:, 0] - cx, pos[:, 1] - cy)
    cc = sel & (rho < 0.25) & (rr > 0.5 * args.r0)
    wt = (1 - rho)[cc]
    k = 2 * np.pi / side
    return np.sum(wt * rr[cc]) / wt.sum(), np.angle(np.sum(wt * np.exp(1j * k * pos[cc, 2]))) / k


rows = []
for sp in args.state:
    sp = Path(sp)
    nm = sp.parent.name
    pos = np.load(sp.parent / "pos.npy")
    ck = np.load(sp)
    P = int(nm.split("_P")[1].split("_")[0])
    side = P / np.sqrt(3)
    pair = nm.startswith("pair")
    psi = np.concatenate([to_complex(ck["L"]), to_complex(ck["R"])], 1)
    del ck
    rho = np.sum(np.abs(psi)**2, 1)
    ix = np.rint(pos * np.sqrt(3)).astype(np.int64)
    iq = (ix + P // 2) % P
    rm = qe.branch_split(psi, iq, P)[0]
    a = psi @ v.conj()
    del psi
    lut = np.full((P,) * 3, -1, np.int64)
    lut[iq[:, 0], iq[:, 1], iq[:, 2]] = np.arange(len(pos))
    ga = np.zeros((len(pos), 3), complex)
    for h in HS[::2]:
        hp = lut[tuple(((iq + h.astype(np.int64)) % P).T)]
        hm = lut[tuple(((iq - h.astype(np.int64)) % P).T)]
        ga += (a[hp] - a[hm])[:, None] * h[None, :] / 2
    ga *= np.sqrt(3) / 4  # grid units -> cells (pos = ix / sqrt3)
    jf = np.imag(np.conj(a)[:, None] * ga)
    jm = np.linalg.norm(jf, axis=1)
    pg = jm / np.maximum(np.abs(a)**2, 1e-12)
    del ga, jf
    prim = np.ones(len(pos), bool)
    if pair:
        rho_p = np.hypot((pos[:, 0] - side / 2 + side / 2) % side - side / 2, (pos[:, 1] - side / 2 + side / 2) % side - side / 2)
        prim = np.hypot(pos[:, 0], pos[:, 1]) < rho_p
    rc, zc = centre(pos, rho, prim, 0, 0, side)
    dt = circle_dist(pos, side, 0, 0, rc, zc)
    # own images: distance to the 26 shifted circles only
    di = np.full(len(pos), np.inf)
    for sx in (-1, 0, 1):
        for sy in (-1, 0, 1):
            for sz in (-1, 0, 1):
                if sx or sy or sz:
                    rr = np.hypot(pos[:, 0] + sx * side, pos[:, 1] + sy * side) - rc
                    di = np.minimum(di, np.hypot(rr, (pos[:, 2] - zc + side / 2) % side - side / 2 + sz * side))
    dt = np.minimum(dt, np.hypot(np.hypot(pos[:, 0], pos[:, 1]) - rc, (pos[:, 2] - zc + side / 2) % side - side / 2))
    if pair:
        rcp, zcp = centre(pos, rho, ~prim, side / 2, side / 2, side)
        dp = circle_dist(pos, side, side / 2, side / 2, rcp, zcp)
    else:
        rcp = zcp = np.nan
        dp = np.full(len(pos), np.inf)
    own = (dt < di) & (dt < dp)
    top = own & (dp < di)
    toi = own & (dp >= di)
    mid = np.abs(dt - np.minimum(di, dp)) < 2
    midp = mid & (dp < di)
    print(f"{nm}: P {P} side {side:.1f} rc {rc:.1f} zc {zc:+.1f} rcp {rcp:.1f} zcp-zc {(zcp - zc) % side:.1f}; "
          f"floor (dt > 50, own) 1-rho {np.mean(1 - rho[own & (dt > 50)]) if (own & (dt > 50)).any() else np.nan:.2e}", flush=True)
    f = {"def": 1 - rho, "rm": rm, "pg": pg, "jm": jm}
    nb = int(side / args.dr) + 1
    kb = (dt / args.dr).astype(np.int64)
    for lab, msk in (("->p", top), ("->i", toi)):
        n = np.bincount(kb[msk], minlength=nb)[:nb]
        for key, fv in f.items():
            s = np.bincount(kb[msk], fv[msk], minlength=nb)[:nb]
            rows.append((nm, P, lab, key, n, np.where(n > 0, s / np.maximum(n, 1), np.nan)))
    ms = {key: (np.mean(fv[midp]) if midp.any() else np.nan, np.mean(fv[mid & ~(dp < di)])) for key, fv in f.items()}
    print("  equidistance surface  (to partner | to own image): " + "  ".join(f"{k} {a_:.3e} | {b_:.3e}" for k, (a_, b_) in ms.items()), flush=True)
    print(f"  surface dt: to partner {np.mean(dt[midp]) if midp.any() else np.nan:.1f}, to image {np.mean(dt[mid & ~(dp < di)]):.1f}", flush=True)
    del pos, rho, rm, a, jm, pg, dt, di, dp, lut, iq, ix

print(f"\nprofiles: mean per dt bin of width {args.dr} (bin centre), own Voronoi, split by the next-nearest structure")
for key in ("def", "rm", "pg", "jm"):
    print(f"\n== {key}")
    for nm, P, lab, k_, n, prof in rows:
        if k_ != key:
            continue
        cs = " ".join(f"{x:9.2e}" if np.isfinite(x) and c_ >= 50 else "        -" for x, c_ in zip(prof[2:22], n[2:22]))
        print(f"P{P:3d} {'pair' if nm.startswith('pair') else 'lone'} {lab}  {cs}")
print("bins dt: " + " ".join(f"{(i + 0.5) * args.dr:9.0f}" for i in range(2, 22)))
if args.out:
    np.savez(args.out, rows=np.array(rows, dtype=object), dr=args.dr)
