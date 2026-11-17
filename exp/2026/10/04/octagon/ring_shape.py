"""shape of a single-ring core (rho < 0.25): centroid, plane (SVD), in-plane radius r(phi) and out-of-plane height
h(phi) binned in azimuth, Fourier amplitudes (cells) of r and h for n = 1..10. usage: ring_shape.py state.npz [...]"""
import sys
from pathlib import Path

import numpy as np


NMAX = 32


def shape(pos, L, R, thr=0.25, nb=128):
    rho = (L**2).sum(1) + (R**2).sum(1)
    c = pos[rho < thr]
    cen = c.mean(0)
    u, s, vt = np.linalg.svd(c - cen, full_matrices=False)
    e1, e2, nrm = vt
    q = c - cen
    a, b, h = q @ e1, q @ e2, q @ nrm
    r, ph = np.hypot(a, b), np.arctan2(b, a)
    k = ((ph + np.pi) / (2 * np.pi) * nb).astype(int) % nb
    cnt = np.bincount(k, minlength=nb)
    rb = np.bincount(k, r, nb) / np.maximum(cnt, 1)
    hb = np.bincount(k, h, nb) / np.maximum(cnt, 1)
    phb = -np.pi + (np.arange(nb) + 0.5) * 2 * np.pi / nb
    F = lambda f, n: 2 * np.mean(f * np.exp(-1j * n * phb))
    return dict(n=len(c), cen=cen, nrm=nrm, r=rb.mean(), r_amp=[abs(F(rb, n)) for n in range(1, NMAX + 1)],
                h_amp=[abs(F(hb, n)) for n in range(1, NMAX + 1)], r8ph=np.degrees(np.angle(F(rb, 8))) / 8,
                h8ph=np.degrees(np.angle(F(hb, 8))) / 8, cnt_amp=[abs(F(cnt / cnt.mean(), n)) for n in (4, 8)])


if __name__ == "__main__":
    for f in sys.argv[1:]:
        f = Path(f)
        pos = np.load(f.parent / "pos.npy")
        ck = np.load(f)
        s = shape(pos, ck["L"], ck["R"])
        fmt = lambda v: " ".join(f"{x:.2f}" for x in v)
        top = lambda v: ", ".join(f"n{n + 1} {v[n]:.2f}" for n in np.argsort(v)[::-1][:3])
        print(f"{f.parent.name[-34:]} t {int(ck['t'])}: core {s['n']} r {s['r']:.1f}; top r_n: {top(s['r_amp'])}; top h_n: "
              f"{top(s['h_amp'])}", flush=True)
        continue
        print(f"{f.parent.name[-30:]} t {int(ck['t'])}: core {s['n']} r {s['r']:.1f} normal {np.round(s['nrm'], 3)} "
              f"z {s['cen'][2] + float(ck['z_shift']):+.1f}\n  r_n(1..10) {fmt(s['r_amp'])}\n  h_n(1..10) {fmt(s['h_amp'])}"
              f"\n  8-fold phase r {s['r8ph']:+.1f} h {s['h8ph']:+.1f} deg; cell-count 4/8-fold {fmt(s['cnt_amp'])}", flush=True)
