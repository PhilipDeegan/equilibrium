"""Ground truth on a periodic BCC box (no edge): sites of a 2n^3 integer grid with x+y+z parity (real_form.py style),
walk hops h = (+-1, +-1, +-1) by np.roll; uniform background + 1e-10 noise; rms departure vs t; growth rate."""
import sys
import numpy as np
from equilibrium.substrate.walk import A_REAL, HS, J4, healing_g, rest_state, to_real

nn = int(sys.argv[1]) if len(sys.argv) > 1 else 32
m = float(__import__("os").environ.get("M", 0.5))
n = np.sqrt(1 - m * m)
v, om0 = rest_state(m)
hs = HS.astype(int)
rng = np.random.default_rng(2)
x, y, z = np.meshgrid(*(np.arange(nn),) * 3, indexing="ij")
mask = ((x + y + z) % 2 == 0)  # one BCC sublattice of the cubic grid with hex offsets (+-1,+-1,+-1)
for xi in [float(a) for a in (sys.argv[2:] or ["2.6", "3.9", "5.2"])]:
    g = healing_g(m, xi)
    mu = om0 + g
    bL, bR = to_real(v[None, :2])[0], to_real(v[None, 2:])[0]
    L = np.broadcast_to(bL, (nn, nn, nn, 4)).copy() + 1e-10 * rng.normal(size=(nn, nn, nn, 4))
    R = np.broadcast_to(bR, (nn, nn, nn, 4)).copy() + 1e-10 * rng.normal(size=(nn, nn, nn, 4))
    out = []
    for t in range(int(__import__("os").environ.get("T", 1601))):
        if t % 100 == 0:
            th = mu * t
            c, s = np.cos(th), np.sin(th)
            def bg(b):
                o = b.copy(); o[0::2] = b[0::2] * c + b[1::2] * s; o[1::2] = b[1::2] * c - b[0::2] * s; return o
            dL, dR = L - bg(bL), R - bg(bR)
            out.append(np.sqrt(np.mean(np.sum(dL**2 + dR**2, -1)[mask])))
        Lh = sum(np.roll(L, tuple(-hh), axis=(0, 1, 2)) @ A.T for hh, A in zip(hs, A_REAL))
        Rh = sum(np.roll(R, tuple(hh), axis=(0, 1, 2)) @ A for hh, A in zip(hs, A_REAL))
        L1 = n * Lh + m * R @ J4.T
        R1 = m * L @ J4.T + n * Rh
        th = g * (np.sum(L1**2, -1) + np.sum(R1**2, -1))
        cc, ss = np.cos(th)[..., None], np.sin(th)[..., None]
        for P in (L1, R1):
            a, b = P[..., 0::2].copy(), P[..., 1::2].copy()
            P[..., 0::2] = a * cc + b * ss
            P[..., 1::2] = b * cc - a * ss
        L, R = L1, R1
    out = np.array(out)
    ts = np.arange(len(out)) * 100
    sel = (out > 3 * out[0]) & (out < 1e-2)
    rate = np.expm1(np.polyfit(ts[sel], np.log(out[sel]), 1)[0]) if sel.sum() > 2 else np.nan
    print(f"xi {xi} g {g:.4f}: rms " + " ".join(f"{d:.0e}" for d in out) + f" -> growth {rate:.2e}/tick", flush=True)
