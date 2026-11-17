"""Transverse plane-wave packet, numpy AE (LS curl_curl_sym) vs DecAE: phase speed error (k-projection) and
pointwise residual of A (from phi) and E at the end, plus grid-scale content of E. Same setup as
tst _packet_dc_c: lam=4, s=1.2, travel 3, R=10, cylinder rho<3."""
import sys
import time
from pathlib import Path

import numpy as np

from equilibrium.substrate import continuous as C

C.set_weights("legacy")  # written for W = 3:1 on both legs (pre-2026-09-28; see continuous.WEIGHTS)
from equilibrium.substrate.lattice import SubstrateLattice

sys.path.insert(0, str(Path(__file__).parent))
from dec import DecAE  # noqa: E402

R, travel, lam, s, rho_m = 10.0, 3.0, 4.0, 1.2, 3.0
c, k = C.C0, 2 * np.pi / lam
xi0 = -travel / 2


def f(xi, d=0):
    ch = xi - xi0
    g = np.exp(-0.5 * ch**2 / s**2)
    return np.cos(k * ch) * g if d == 0 else (-k * np.sin(k * ch) - ch / s**2 * np.cos(k * ch)) * g


def run(h, n, which):
    n = np.asarray(n, float) / np.linalg.norm(n)
    ref = np.eye(3)[int(np.argmin(np.abs(n)))]
    pol = ref - (ref @ n) * n
    pol /= np.linalg.norm(pol)
    lat = SubstrateLattice(radius_cells=R, spacing=h)
    dt = C.stable_dt(lat)
    nst = int(np.ceil(travel / c / dt))
    dt = travel / c / nst
    xn = lat.pos @ n
    rho = np.linalg.norm(lat.pos - xn[:, None] * n, axis=1)
    idx = np.nonzero((rho < rho_m) & (np.abs(xn) < travel / 2 + 4 * s))[0]
    ph = np.exp(-1j * k * xn[idx])
    lat.phi = C.sample_A(lat, lambda y: f(y @ n - 0.5 * c * dt)[:, None] * pol)
    e0 = C.sample_A(lat, lambda y: (c * f(y @ n, 1))[:, None] * pol)
    ae = DecAE(lat, e0) if which == "DEC" else C.AE(lat, e0)
    t0 = time.time()
    for _ in range(nst):
        ae.step(dt)
    T = nst * dt
    out = {}
    for key, rows, tt, exf in (("A", lat.phi, T + 0.5 * dt, lambda xi: f(xi)), ("E", ae.e, T, lambda xi: c * f(xi, 1))):
        q = C.reconstruct_A_at(lat, idx, rows[idx]) @ pol
        qe = exf(xn[idx] - c * tt)
        dc = -np.angle((q * ph).sum() / (qe * ph).sum()) / (c * k * tt)
        res = np.linalg.norm(q - qe) / np.linalg.norm(qe)
        exs = C.sample_A(lat, lambda y: (exf(y @ n - c * tt))[:, None] * pol)[idx]
        face_res = np.linalg.norm(rows[idx] - exs) / np.linalg.norm(exs)
        out[key] = (dc, res, face_res)
    print(f"{which:3s} h={h} n={np.round(n, 3)} kh={k * h:.3f}: " + "  ".join(
        f"{key}: dc/c {v[0]:+.5f} cell-res {v[1]:.3e} face-res {v[2]:.3e}" for key, v in out.items())
        + f"  ({time.time() - t0:.0f}s, N={lat.N})", flush=True)


for h in (0.5, 0.25):
    for n in ([1, 0, 0], [1, 1, 1]):
        for which in ("LS", "DEC"):
            run(h, n, which)
