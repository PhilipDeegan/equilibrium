"""10-12 villain: static Peierls-Nabarro landscape of a Villain ring. phi = W (dtheta - 2 pi n), n integer per face,
n = +-1 on faces crossing a disc (the ring = the disc edge). For fixed n, theta relaxes (linear): E_min(n) =
1/2 sum_faces |div-free part of -2 pi W n|^2. Scan disc centre / height over one lattice period; spread of E_min = the
landscape the defect line sees. Commensurate (z-normal ring moved along z, all edge faces flip together; kinked path
added) vs incommensurate (in-plane shift, tilted ring). Barrier scale: dE/dR x one layer spacing h."""
import argparse
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--P", type=int, default=72)
ap.add_argument("--R", type=float, nargs="+", default=[6.0, 10.0, 14.0])
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--cases", nargs="+", default=None)
args = ap.parse_args()
sys.argv = sys.argv[:1]

from equilibrium.substrate import continuous as C
from equilibrium.substrate.walk import cpp_box

C.spacing_of = lambda lat: 1.0
P = args.P
alat = cpp_box(P, n_threads=1)
ac = C.AnchorCluster(alat)
pos, nb, N = ac.pos, ac.nb, ac.N
side = P / np.sqrt(3)
W, D = C.W, C.D_HAT
h = 1 / np.sqrt(3)
centre = pos.mean(0) + np.array([0.13, 0.07, 0.11])
wrapx = lambda x: (x + side / 2) % side - side / 2


def n_disc(c, nrm, R, kink=None):
    nrm = nrm / np.linalg.norm(nrm)
    x = wrapx(pos - c)
    s0 = x @ nrm
    n = np.zeros((N, 14))
    for k in range(14):
        s1 = s0 + D[k] @ nrm
        up, dn = (s0 < 0) & (s1 >= 0), (s0 >= 0) & (s1 < 0)
        m = up | dn
        t = np.where(m, -s0 / np.where(m, s1 - s0, 1), 0)
        p = x + t[:, None] * D[k]
        r = np.linalg.norm(p - (p @ nrm)[:, None] * nrm, axis=1)
        n[:, k] = np.where(m & (r < R), np.where(up, 1.0, -1.0), 0.0)
    return n


E_prev = {}


def energy(n, tag):
    f = -2 * np.pi * W[None, :] * n
    out, psi = C.project_div_free(ac, f, tol=1e-9, x0=E_prev.get(tag), return_psi=True)
    E_prev[tag] = psi
    return 0.25 * np.sum(out**2)


def n_cyl(c, R, z1, alpha):
    x = wrapx(pos - c)
    n = np.zeros((N, 14))
    for k in range(14):
        d = D[k]
        a = d[0] ** 2 + d[1] ** 2
        if a == 0:
            continue
        b = 2 * (x[:, 0] * d[0] + x[:, 1] * d[1])
        cc = x[:, 0] ** 2 + x[:, 1] ** 2 - R * R
        disc = b * b - 4 * a * cc
        ok = disc > 0
        sq = np.sqrt(np.where(ok, disc, 0))
        for t, sg in (((-b - sq) / (2 * a), -1.0), ((-b + sq) / (2 * a), 1.0)):
            p = x + t[:, None] * d
            az = np.arctan2(p[:, 1], p[:, 0]) % (2 * np.pi)
            m = ok & (t >= 0) & (t < 1) & (p[:, 2] >= 0) & (p[:, 2] < z1) & (az < alpha)
            n[:, k] += np.where(m, sg, 0.0)
    return n


def kinked(c, R, dz, alpha, sgn):
    return n_disc(c, np.array([0, 0, 1.0]), R) + sgn * n_cyl(c, R, dz, alpha)


print(f"P {P} N {N} side {side:.2f} h {h:.4f}", flush=True)
t0 = time.time()
tilt = np.array([1.0, 2.0, 3.3])
lat_dir = np.cross(tilt, [0, 0, 1.0]) / np.linalg.norm(np.cross(tilt, [0, 0, 1.0]))
for R in args.R:
    Eh = [energy(n_disc(centre, np.array([0, 0, 1.0]), R + d), "r") for d in (-h, h)]
    dEdR = (Eh[1] - Eh[0]) / (2 * h)
    E0 = energy(n_disc(centre, np.array([0, 0, 1.0]), R), "r")
    print(f"R {R:5.1f}: E {E0:10.3f}  E / 2piR {E0 / (2 * np.pi * R):7.3f}  dE/dR * h {dEdR * h:8.3f}  "
          f"({time.time() - t0:.0f} s)", flush=True)
    En = [energy(kinked(centre, R, h, 2 * np.pi, g), "k") for g in (1, -1)]
    sgn = 1 if En[0] < En[1] else -1
    Ez = energy(n_disc(centre + [0, 0, h], np.array([0, 0, 1.0]), R), "k")
    print(f"   full-hop check: cyl sign {sgn}, E(disc+cyl) - E(disc at z+h) {min(En) - Ez:+.3e}", flush=True)
    s = np.linspace(0, 2 * h, args.n, endpoint=False)
    cases = {
        "z-ring moved z (commensurate)": lambda: (n_disc(centre + [0, 0, d], np.array([0, 0, 1.0]), R) for d in s),
        "z-ring moved x (in-plane)": lambda: (n_disc(centre + [d, 0, 0], np.array([0, 0, 1.0]), R) for d in s),
        "tilted ring moved along normal": lambda: (n_disc(centre + d * tilt / np.linalg.norm(tilt), tilt, R) for d in s),
        "tilted ring moved lateral": lambda: (n_disc(centre + d * lat_dir, tilt, R) for d in s),
        "z-ring kinked hop (alpha 0..2pi)": lambda: (kinked(centre, R, h, a, sgn) for a in np.linspace(0, 2 * np.pi, args.n + 1)),
    }
    for name, mk in cases.items():
        if args.cases and not any(c in name for c in args.cases):
            continue
        E = np.array([energy(n, name) for n in mk()])
        dE = E.max() - E.min()
        print(f"   {name:34s} spread {dE:8.3f}  / (dE/dR h) {dE / (dEdR * h):6.3f}  / E {dE / E0:.2e}  "
              f"per length {dE / (2 * np.pi * R):.4f}  [{' '.join(f'{e - E0:+.2f}' for e in E)}]  ({time.time() - t0:.0f} s)",
              flush=True)
