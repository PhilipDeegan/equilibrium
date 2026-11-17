"""11-08: + non-vector share of face content per print (anchor_residual: the 4 of 7 face DOF per cell no
smooth vector field holds; inside r < a and total) and phi dumps every --dump time to --dumpdir.
10-22 spheromak baseline in the continuous anchored-face model (first since the switch), step() + Cartan, no new
relation. Taylor spheromak (Beltrami, curl A = lam A, lam a = 4.4934) inside radius a, zero outside, sampled onto the
anchors (phi = W A(mid).d), curl-curl smoothed, div-free projected, u = 0. Periodic box.
Scorecard: E, Eu share, shape change |phi - phi0| / |phi0| (at rest, no recentring), centroid drift, Beltrami
alignment cos(A, curl A) and helicity sum A.curl A, energy inside a / a..1.5a / beyond."""
import argparse
import time
from pathlib import Path

import numpy as np
from scipy.special import spherical_jn

from equilibrium.substrate import continuous as C
from equilibrium.substrate.walk import cpp_box

C.spacing_of = lambda lat: 1.0
p = argparse.ArgumentParser()
p.add_argument("--P", type=int, default=192)
p.add_argument("--a", type=float, default=20.0, help="spheromak radius (cells)")
p.add_argument("--vmax", type=float, default=0.3, help="peak |A| of the seed")
p.add_argument("--T", type=float, default=60.0)
p.add_argument("--every", type=float, default=2.0)
p.add_argument("--smooth", type=float, default=4.938, help="curl-curl diffusion time of the seed")
p.add_argument("--eps", type=float, default=0.02)
p.add_argument("--kappa", type=float, default=1.0)
p.add_argument("--threads", type=int, default=8)
p.add_argument("--seed", default="")
p.add_argument("--save", default="")
p.add_argument("--ckpt_every", type=int, default=5)
p.add_argument("--dump", type=float, default=0.0)
p.add_argument("--dumpdir", default="")
args = p.parse_args()
side = args.P / np.sqrt(3)
W = C.W
OFF = np.array([0.13, 0.07, 0.11])
wrapx = lambda x: (x + side / 2) % side - side / 2
a_, lam = args.a, 4.4934094579 / args.a


def taylor(X):
    x = wrapx(X - OFF)
    r = np.linalg.norm(x, axis=1)
    rs = np.maximum(r, 1e-9)
    ct, st = x[:, 2] / rs, np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2) / rs
    rho = np.maximum(np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2), 1e-12)
    cp, sp = x[:, 0] / rho, x[:, 1] / rho
    k = lam * rs
    j1, j1p = spherical_jn(1, k), spherical_jn(1, k, derivative=True)
    Br = 2 * j1 / k * ct
    Bt = -(j1 / k + j1p) * st
    Bp = j1 * st
    B = np.stack([Br * st * cp + Bt * ct * cp - Bp * sp, Br * st * sp + Bt * ct * sp + Bp * cp, Br * ct - Bt * st], 1)
    return np.where((r < a_)[:, None], B, 0.0)


alat = cpp_box(args.P, n_threads=args.threads)
ac = C.AnchorCluster(alat)
pos, nb, N = ac.pos, ac.nb, ac.N
x = wrapx(pos - OFF)
r = np.linalg.norm(x, axis=1)
if args.seed and Path(args.seed).exists():
    phi = np.load(args.seed)["phi"]
else:
    phi = C.sample_A(ac, taylor)
    for it in range(int(round(args.smooth / args.eps))):
        phi = phi - args.eps * ac.curl_curl_dec(phi)
    phi = C.project_div_free(ac, phi)
    phi *= args.vmax / np.abs(C.reconstruct_A(ac, phi)).max()
    if args.seed:
        Path(args.seed).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.seed, phi=phi)
n0 = np.linalg.norm(phi)
SH = np.array([1.0, 1.5]) * a_
dt = 0.4 / float(np.sum(W**2))
nst, ev = int(round(args.T / dt)), max(1, int(round(args.every / dt)))


CH = np.array_split(np.arange(N), max(1, N // 65536))


def phiT():
    return np.asarray(ac.cl.leaf_phi)


def belt(PT):
    rows = lambda j: PT[:, j].T
    ab = aa = bb = H = 0.0
    for idx in CH:
        A = C.reconstruct_A_at(ac, idx, rows(idx))
        B = C.curl_A_at(ac, idx, rows)
        m = r[idx] < a_
        ab += np.sum(A[m] * B[m])
        aa += np.sum(A[m] ** 2)
        bb += np.sum(B[m] ** 2)
        H += np.sum(A * B)
    return ab / np.sqrt(aa * bb), H, ab / aa


def chunked(PT, PprevT, u):
    E = Eu = dd = 0.0
    sh, wx = np.zeros(3), np.zeros(3, complex)
    for idx in CH:
        ph, pp = PT[:, idx], PprevT[:, idx]
        ec = u[idx] ** 2 + 0.5 * np.sum(pp * ph, 0)
        E += ec.sum()
        Eu += np.sum(u[idx] ** 2)
        sh += np.bincount(np.searchsorted(SH, r[idx]), weights=ec, minlength=3)
        w = np.sum(ph**2, 0)
        wx += (w[:, None] * np.exp(2j * np.pi * x[idx] / side)).sum(0)
        dd += np.sum((ph - phi0T[:, idx]) ** 2)
    return E, Eu, sh / E, np.angle(wx) * side / (2 * np.pi), np.sqrt(dd) / n0


ac.u, ac.phi = np.zeros(N), phi
phi0T = np.ascontiguousarray(phi.T)
cos0, H0, lam0 = belt(phi0T)
T0 = ac.cartan_transport(phi)
print(f"P {args.P} a {a_} N {N} vmax {args.vmax} ticks {nst} dt {dt:.4e}  seed cos(A, curl A) {cos0:.4f} (lam a: "
      f"{lam0 * a_:.3f} vs 4.493)  Cartan |T| / |phi| {np.linalg.norm(T0) / n0:.3e}", flush=True)
del phi, T0
ck = args.seed.replace(".npz", "_ckpt.npz") if args.seed else ""
s0, hist, prevT = 0, [], phi0T
if ck and Path(ck).exists():
    z = np.load(ck)
    ac.phi, ac.u, s0, hist = z["phi"], z["u"], int(z["s"]), list(map(tuple, z["hist"]))
    prevT = np.ascontiguousarray(z["ph_prev"].T)
    print(f"resumed {ck} at t {s0 * dt:.2f}", flush=True)
t_wall, nprint = time.time(), 0
for s in range(s0, nst + 1):
    if s % ev == 0:
        PT, u = phiT(), ac.u
        E, Eu, sh, cen, dph = chunked(PT, prevT, u)
        cos, H, _ = belt(PT)
        ph_ = PT.T
        rs_ = C.anchor_residual(ac, ph_)
        aw_ = np.where(ac.nb >= 0, ph_ / W[None, :], 0.0)
        q_ = np.sum(rs_**2, 1)
        nv, nvin = q_.sum() / np.sum(aw_**2), q_[r < a_].sum() / np.sum(aw_[r < a_] ** 2)
        if args.dump > 0 and args.dumpdir and nprint % max(1, round(args.dump / args.every)) == 0:
            Path(args.dumpdir).mkdir(parents=True, exist_ok=True)
            np.savez(f"{args.dumpdir}/phi_t{s * dt:06.1f}.npz", phi=ph_.astype(np.float32), u=ac.u.astype(np.float32))
        del ph_, rs_, aw_
        if not (s == s0 and s0 > 0):
            hist.append((s * dt, E, Eu, dph, cos, H, *cen, *sh))
        wt = (time.time() - t_wall) / max(1, s - s0)
        print(f"  t {s * dt:6.2f}  E {E:.8e}  Eu/E {Eu / E:.3e}  |dphi|/|phi0| {dph:.4e}  cos {cos:.5f}  H/H0 {H / H0:.5f}  "
              f"nonvec {nv:.3e} in {nvin:.3e}  cen {cen[0]:+.3f} {cen[1]:+.3f} {cen[2]:+.3f}  in/shell/out {sh[0]:.4f} {sh[1]:.4f} {sh[2]:.4f}  "
              f"({wt:.2f} s/tick, ETA {wt * (nst - s) / 60:.1f} min)", flush=True)
        nprint += 1
        if ck and s > s0 and s < nst and nprint % args.ckpt_every == 0:
            np.savez(ck + ".tmp.npz", phi=PT.T, u=u, ph_prev=prevT.T, s=s, hist=np.array(hist))
            Path(ck + ".tmp.npz").replace(ck)
        del PT
        prevT = None
    if s == nst:
        break
    if (s + 1) % ev == 0 or s + 1 == nst:
        prevT = phiT()
    alat.step_cartan(0, dt, args.kappa, 2)
h = np.array(hist)
print(f"RESULT spheromak a {a_} P {args.P} vmax {args.vmax}  dE/E {h[-1, 1] / h[0, 1] - 1:+.1e}  Eu/E end {h[-1, 2] / h[-1, 1]:.3e}  "
      f"|dphi| end {h[-1, 3]:.4f}  cos {h[0, 4]:.4f}->{h[-1, 4]:.4f}  H/H0 {h[-1, 5] / h[0, 5]:.4f}  drift {h[-1, 6]:+.3f} {h[-1, 7]:+.3f} {h[-1, 8]:+.3f}  "
      f"in {h[0, 9]:.4f}->{h[-1, 9]:.4f}")
if args.save:
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.save, phi=ac.phi, u=ac.u, hist=h, args=str(vars(args)))
