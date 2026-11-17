"""R160 (h=0.125) snapshots vs the exact retarded field of the gaussian dipole.

Outside the source, a spherically symmetric source radiates as a point source
times its Fourier transform at k: F = exp(-(ks)^2/2). After the ramp
p(t) = -I0 cos(wt)/w exactly (no static charge), so with psi = wt - kr:
  A = z F I0 sin(psi) / (4 pi c^2 r)
  u = F I0 cos(th) / (4 pi w) [k sin(psi)/r - cos(psi)/r^2]
  E = -A_t - grad u,  B = curl A        (AE: u_t = -c^2 div A, A_t = -grad u - E)
phi is half a step ahead of u and E. Per shell: relative L2 error of A, B, u, E;
instantaneous shell flux num/exact; equatorial azimuth dependence (lattice
anisotropy); pointwise error ratio h=0.25 / h=0.125 at the common t_end on
the shared (2x BCC) sites.
"""
import argparse
from pathlib import Path

import numpy as np

from equilibrium.substrate import continuous as C

C.set_weights("legacy")  # written for W = 3:1 on both legs (pre-2026-09-28; see continuous.WEIGHTS)

p = argparse.ArgumentParser()
p.add_argument("--snaps", default="7562,8823,10083,11344,12604")
p.add_argument("--threads", type=int, default=2)
args = p.parse_args()

R, lam, s, I0 = 20.0, 4.0, 0.6, 1.0
c, c2 = C.C0, C.C0SQ
w = 2 * np.pi * c / lam
k = w / c
F = np.exp(-0.5 * (k * s) ** 2)
per = 2 * np.pi / w
here = Path(__file__).resolve().parent
r_shells = np.array([3.0, 4.0, 5.0, 6.5, 8.0, 9.5])


class Rows:
    def __init__(self, pos, nb):
        self.pos, self.nb, self.N = pos, nb, len(pos)


def lattice(h, snap_dir=None):
    f = snap_dir / "nb.npy" if snap_dir else None
    if f is not None and f.exists():
        return Rows(np.load(snap_dir / "pos.npy"), np.load(f))
    lat = C.anchor_cluster(R, h, args.threads)
    out = Rows(lat.pos.copy(), lat.nb.copy())
    del lat
    if f is not None:
        assert np.array_equal(out.pos, np.load(snap_dir / "pos.npy"))
        np.save(f, out.nb)
    return out


def exact_Au(x, t):
    r = np.linalg.norm(x, axis=1)
    ps = w * t - k * r
    a = F * I0 * np.sin(ps) / (4 * np.pi * c2 * r)
    A = np.zeros_like(x)
    A[:, 2] = a
    u = F * I0 * (x[:, 2] / r) / (4 * np.pi * w) * (k * np.sin(ps) / r - np.cos(ps) / r**2)
    return A, u


def exact_EB(x, t, d=1e-4):
    At = (exact_Au(x, t + d)[0] - exact_Au(x, t - d)[0]) / (2 * d)
    gu = np.stack([(exact_Au(x + d * e, t)[1] - exact_Au(x - d * e, t)[1]) / (2 * d) for e in np.eye(3)], 1)
    g = np.stack([(exact_Au(x + d * e, t)[0] - exact_Au(x - d * e, t)[0]) / (2 * d) for e in np.eye(3)], 1)
    B = np.stack([g[:, 1, 2] - g[:, 2, 1], g[:, 2, 0] - g[:, 0, 2], g[:, 0, 1] - g[:, 1, 0]], 1)
    return -At - gu, B


def fields(lat, idx, u, phi_rows, e_rows):
    A = C.reconstruct_A_at(lat, idx, phi_rows(idx))
    B = C.curl_A_at(lat, idx, phi_rows)
    E = C.reconstruct_A_at(lat, idx, e_rows(idx))
    return A, B, u[idx], E


def rel(a, b, b90):
    """error / period-rms amplitude (b90 = exact a quarter period later)"""
    return float(np.sqrt(((a - b) ** 2).sum() / (0.5 * (b**2 + b90**2).sum())))


def exact_all(x, t, dth):
    A, _ = exact_Au(x, t + dth / 2)
    _, u = exact_Au(x, t)
    E, _ = exact_EB(x, t)
    _, B = exact_EB(x, t + dth / 2)
    return A, B, u, E


def shell_sets(lat, h):
    r = np.linalg.norm(lat.pos, axis=1)
    return [np.nonzero(np.abs(r - Rs) < h)[0] for Rs in r_shells]


snap_dir = Path("/bk/dst/2026/09/26/dipole_R160_h0.125")
h = 0.125
lat = lattice(h, snap_dir)
sets = shell_sets(lat, h)
print(f"N={lat.N}  F^2={F**2:.4f}  shells {r_shells}  cells {[len(q) for q in sets]}", flush=True)

n_az = 16
eq_num = np.zeros((len(r_shells), n_az, 2))
eq_th = np.zeros_like(eq_num)
report = []
dt = None
last = None
for sn in [int(v) for v in args.snaps.split(",")]:
    d = np.load(snap_dir / f"state_{sn:06d}.npz")
    t = float(d["t"])
    dt = t / sn
    u, phi, e = d["u"], d["phi"], d["e"]
    pr, er = (lambda j: phi[:, j].T), (lambda j: e[:, j].T)
    print(f"\nstep {sn}  t={t:.4f}  (t-r/c at r=9.5: {t - 9.5 / c:.3f}, ramp ends {per:.3f})")
    print(" r     err/amp: A        B        u        E     (P_num-P_exact)/(F^2 P_th)  P_exact/(F^2 P_th)")
    for a, idx in enumerate(sets):
        x = lat.pos[idx]
        An, Bn, un, En = fields(lat, idx, u, pr, er)
        Ae, Be, ue, Ee = exact_all(x, t, dt)
        q90 = exact_all(x, t + per / 4, dt)
        rr = np.linalg.norm(x, axis=1)
        rh = x / rr[:, None]
        Sn = c2 * (np.cross(En, Bn) * rh).sum(1)
        Se = c2 * (np.cross(Ee, Be) * rh).sum(1)
        P_th = I0**2 * w**2 / (12 * np.pi * c**3)
        # shell average of S_r times 4 pi r^2 (cells are uniform in volume)
        Pn = float((Sn * rr**2).mean() * 4 * np.pi)
        Pe = float((Se * rr**2).mean() * 4 * np.pi)
        row = dict(step=sn, t=t, r=r_shells[a], eA=rel(An, Ae, q90[0]), eB=rel(Bn, Be, q90[1]),
                   eu=rel(un, ue, q90[2]), eE=rel(En, Ee, q90[3]), Pn=Pn / P_th, Pe=Pe / P_th)
        report.append(row)
        print(f"{r_shells[a]:4.1f}  {row['eA']:.4f}  {row['eB']:.4f}  {row['eu']:.4f}  {row['eE']:.4f}"
              f"        {(Pn - Pe) / (F**2 * P_th):+.4f}              {Pe / (F**2 * P_th):+.4f}")
        eqm = np.abs(x[:, 2] / rr) < 0.1
        ia = np.clip(((np.arctan2(x[eqm, 1], x[eqm, 0]) + np.pi) / (2 * np.pi) * n_az).astype(int), 0, n_az - 1)
        for q, (vn, ve, v90) in enumerate(((En, Ee, q90[3]), (Bn, Be, q90[1]))):
            np.add.at(eq_num[a, :, q], ia, ((vn[eqm] - ve[eqm]) ** 2).sum(1))
            np.add.at(eq_th[a, :, q], ia, 0.5 * (ve[eqm] ** 2 + v90[eqm] ** 2).sum(1))
    last = (sn, t, u, phi, e)
    del d

print("\nequatorial (|cos th|<0.1) error/amplitude per azimuth bin, all snapshots:")
for a, Rs in enumerate(r_shells):
    for q, name in enumerate("EB"):
        rat = np.sqrt(eq_num[a, :, q] / eq_th[a, :, q])
        print(f"r={Rs:4.1f} {name}: min {rat.min():.4f} max {rat.max():.4f}  " + " ".join(f"{v:.3f}" for v in rat))

sn, t, u, phi, e = last
d2 = np.load(here.parent.parent / "25/dipole/dipole_h0.25_R20_cpp.npz")
pos2 = d2["pos"]
lat2 = lattice(0.25)
assert np.allclose(lat2.pos, pos2)
assert abs(t - np.ceil(t / C.stable_dt(lat)) * dt) < 1e-9 * t
print(f"\npointwise h=0.25 vs h=0.125 at t={t:.4f} (both runs end here), shared sites:")
key = lambda P: np.round(P / 0.125 * 2).astype(np.int64)
r1 = np.linalg.norm(lat.pos, axis=1)
sel1 = np.nonzero((r1 > 2.5) & (r1 < 10.0))[0]
K1 = key(lat.pos[sel1])
r2 = np.linalg.norm(pos2, axis=1)
sel2 = np.nonzero((r2 > 2.5) & (r2 < 10.0))[0]
K2 = key(pos2[sel2])
base = np.int64(4 * R / 0.125 + 8)
h1 = ((K1[:, 0] + base) * (2 * base) + (K1[:, 1] + base)) * (2 * base) + (K1[:, 2] + base)
h2 = ((K2[:, 0] + base) * (2 * base) + (K2[:, 1] + base)) * (2 * base) + (K2[:, 2] + base)
common, i1, i2 = np.intersect1d(h1, h2, return_indices=True)
i1, i2 = sel1[i1], sel2[i2]
print(f"shared sites in 2.5<r<10: {len(common)}  (h=0.25 sites there: {len(sel2)})")
u2, phi2, e2 = d2["u"], d2["phi"].T, d2["e"].T
x = lat.pos[i1]
rr = np.linalg.norm(x, axis=1)
out = {}
for tag, L, I, uu, pp, ee, dth in (("h0.125", lat, i1, u, phi, e, dt), ("h0.25", lat2, i2, u2, phi2, e2, t / np.ceil(t / C.stable_dt(lat2)))):
    pr, er = (lambda j, pp=pp: pp[:, j].T), (lambda j, ee=ee: ee[:, j].T)
    An, Bn, un, En = fields(L, I, uu, pr, er)
    Ae, _ = exact_Au(x, t + dth / 2)
    _, ue = exact_Au(x, t)
    Ee, _ = exact_EB(x, t)
    _, Be = exact_EB(x, t + dth / 2)
    out[tag] = dict(A=An - Ae, B=Bn - Be, u=un - ue, E=En - Ee, nA=Ae, nB=Be, nu=ue, nE=Ee)
rb = np.array([2.5, 4, 5.5, 7, 8.5, 10])
print(" r-bin       " + "  ".join(f"{q}: e(0.25)/e(0.125)" for q in "ABuE"))
for lo, hi in zip(rb[:-1], rb[1:]):
    m = (rr >= lo) & (rr < hi)
    ratios = []
    for q in "ABuE":
        e1 = np.sqrt((out["h0.125"][q][m] ** 2).sum())
        e2_ = np.sqrt((out["h0.25"][q][m] ** 2).sum())
        ratios.append(e2_ / e1)
    print(f"[{lo:4.1f},{hi:4.1f})   " + "   ".join(f"{v:14.2f}" for v in ratios))
np.savez(here / "r160_vs_exact.npz", report=np.array([list(r_.values()) for r_ in report]),
         report_keys=np.array(list(report[0].keys())), eq_num=eq_num, eq_th=eq_th, r_shells=r_shells)
