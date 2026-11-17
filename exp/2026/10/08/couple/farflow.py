"""10-08: signed phase gradient u = Im(a* grad a) / |a|^2 (a = psi . v*) in a saved state: box mean (smooth part, cells
with rho > 0.5), far-region mean (dt > --dt_far from every coil tube), and u_z on the axis through the primary's hole,
against the incompressible zero-mean prediction (the k = 0-dropped return flow ~ -2 pi q pi r0^2 / V)."""
import argparse
from pathlib import Path

import numpy as np

from equilibrium.substrate.walk import HS, rest_state, to_complex

p = argparse.ArgumentParser()
p.add_argument("--state", action="append", default=[])
p.add_argument("--dt_far", type=float, default=40.0)
p.add_argument("--r0", type=float, default=30.0)
args = p.parse_args()
v, _ = rest_state(0.8)
for sp in map(Path, args.state):
    nm = sp.parent.name
    P = int(nm.split("_P")[1].split("_")[0])
    side = P / np.sqrt(3)
    pos = np.load(sp.parent / "pos.npy")
    ck = np.load(sp)
    psi = np.concatenate([to_complex(ck["L"]), to_complex(ck["R"])], 1)
    del ck
    rho = np.sum(np.abs(psi)**2, 1)
    a = psi @ v.conj()
    del psi
    iq = (np.rint(pos * np.sqrt(3)).astype(np.int64) + P // 2) % P
    lut = np.full((P,) * 3, -1, np.int64)
    lut[iq[:, 0], iq[:, 1], iq[:, 2]] = np.arange(len(pos))
    u = np.zeros((len(pos), 3))
    for h in HS[::2]:
        ap = a[lut[tuple(((iq + h.astype(np.int64)) % P).T)]]
        am = a[lut[tuple(((iq - h.astype(np.int64)) % P).T)]]
        u += np.angle(ap / am)[:, None] * h[None, :] / 2
    u *= np.sqrt(3) / 4
    good = rho > 0.5
    W = lambda c: (pos[:, c] + side / 2) % side - side / 2  # noqa: E731
    x, y, z = W(0), W(1), W(2)
    cc = (rho < 0.25) & (np.hypot(x, y) > 0.5 * args.r0) & (np.hypot(x, y) < side / 2.5)
    wt = (1 - rho)[cc]
    k = 2 * np.pi / side
    rc = np.sum(wt * np.hypot(x, y)[cc]) / wt.sum()
    zc = np.angle(np.sum(wt * np.exp(1j * k * z[cc]))) / k
    dts = []
    for cx, cy, cz in ((0, 0, zc), (side / 2, side / 2, zc + side / 2)) if nm.startswith("pair") else ((0, 0, zc),):
        dx = (pos[:, 0] - cx + side / 2) % side - side / 2
        dy = (pos[:, 1] - cy + side / 2) % side - side / 2
        dz = (pos[:, 2] - cz + side / 2) % side - side / 2
        dts.append(np.hypot(np.hypot(dx, dy) - rc, dz))
    dmin = np.min(dts, 0)
    far = good & (dmin > args.dt_far)
    ax = (np.hypot(x, y) < 3) & (np.abs((z - zc + side / 2) % side - side / 2) < 3)
    ncoil = 2 if nm.startswith("pair") else 1
    u0 = -2 * np.pi * np.pi * rc**2 * ncoil / side**3
    print(f"{nm[:40]} P {P}: rc {rc:.1f}; box mean u (rho > 0.5) {u[good].mean(0).round(5)}; far (dt > {args.dt_far:g}, "
          f"{far.mean() * 100:.0f} % of cells) mean u {u[far].mean(0).round(5)} sd_z {u[far, 2].std():.4f}; hole axis u_z "
          f"{u[ax, 2].mean():+.4f}; incompressible return flow (ring, zero mean) {u0:+.5f}", flush=True)
