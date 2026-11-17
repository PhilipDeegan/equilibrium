"""Long co-moving spring run: (1) damped-oscillation fit of the deficit D_near(t); (2) rigid-rotation test: density
of a later snapshot vs an earlier one rotated about z by phi (and shifted to the same zc), rms mismatch over the
structure region, vs phi = 0; phi scanned (the 4-fold symmetry makes phi mod pi/2 the meaningful quantity) and
compared with -dhel/4 (pure rotation about z would turn the helix phase theta_pol - 4 theta_tor by -4 phi)."""
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

name = "springcpp_N4_S1_q1_a10_r30_xi3.9_m0.8_R110_z0_wxi_rc_long"
a, xi = 10.0, 3.9
rec = np.load(f"{name}_rec.npz")["rec"]
t, hel, rc, zc, Dn, Dc, zs = rec[:, 0], rec[:, 3], rec[:, 6], rec[:, 7], rec[:, 8], rec[:, 9], rec[:, 10]
ok = np.isfinite(Dn)
helu = np.unwrap(hel[ok])
f = lambda t_, A, tau, P, ph, c0, c1: A * np.exp(-t_ / tau) * np.cos(2 * np.pi * t_ / P + ph) + c0 + c1 * t_
p0 = (15000, 8000, 4300, 0, 53000, 0)
sel = t[ok] >= 500
popt, pcov = curve_fit(f, t[ok][sel], Dn[ok][sel], p0=p0, maxfev=20000)
err = np.sqrt(np.diag(pcov))
print(f"D_near fit (t >= 500): amplitude {popt[0]:.0f}, decay time {popt[1]:.0f} +- {err[1]:.0f} ticks, period {popt[2]:.0f} +- "
      f"{err[2]:.0f}, settles at {popt[4]:.0f} + {popt[5]:.3f} t (drift over run {popt[5] * t.max():+.0f})")
vz = np.polyfit(t[ok], (zc + zs)[ok], 1)[0]
om_h = np.polyfit(t[ok], helu, 1)[0]
print(f"travel speed {vz:.5f} cells/tick; helix phase rate {om_h:.6f} rad/tick (period {2 * np.pi / abs(om_h):.0f} ticks); "
      f"windings all -4/-1: {np.all(rec[ok, 4] == -4) and np.all(rec[ok, 5] == -1)}")
d = Path("/bk/dst/2026/10/02/coil") / name
pos = np.load(d / "pos.npy")
tree = cKDTree(pos)
x, y, z = pos.T


def snap(tt):
    s = np.load(d / f"state_t{tt:05d}.npz")
    rho = np.sum(s["L"] ** 2, 1) + np.sum(s["R"] ** 2, 1)
    i = int(np.argmin(np.abs(t - tt)))
    return rho, rc[i], zc[i], helu[np.searchsorted(t[ok], t[i])]


def mismatch(t1, t2, phis):
    r1, rc1, zc1, h1 = snap(t1)
    r2, rc2, zc2, h2 = snap(t2)
    dt_ = np.hypot(np.hypot(x, y) - rc2, z - zc2)
    reg = dt_ < a + 4 * xi
    P = pos[reg]
    out = []
    for ph in phis:
        c, s_ = np.cos(-ph), np.sin(-ph)
        Q = np.stack([c * P[:, 0] - s_ * P[:, 1], s_ * P[:, 0] + c * P[:, 1], P[:, 2] - zc2 + zc1], 1)
        out.append(np.sqrt(np.mean((r2[reg] - r1[tree.query(Q)[1]]) ** 2)))
    return np.array(out), h2 - h1, np.sqrt(np.mean((r2[reg] - 1) ** 2))


phis = np.linspace(0, np.pi / 2, 181, endpoint=False)
print("\n t1 -> t2     rms(phi=0)  best rms   best phi   -dhel/4 mod pi/2   structure rms(rho-1)")
for t1, t2 in ((10000, 10500), (10000, 11000), (12000, 13000), (14000, 15000), (16000, 18000), (18000, 19000), (19000, 20000),
               (9000, 13500), (15500, 20000)):
    mm, dh, sr = mismatch(t1, t2, phis)
    i = int(np.argmin(mm))
    print(f" {t1:5d} -> {t2:5d}   {mm[0]:.4f}     {mm[i]:.4f}    {np.degrees(phis[i]):6.1f}    "
          f"{np.degrees((-dh / 4) % (np.pi / 2)):6.1f}            {sr:.3f}", flush=True)
