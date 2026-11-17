"""Azimuthal structure of the onset: grid-scale loop content (loop circ minus seed's) in the equatorial ring
r1 < r < r2, angular Fourier power by m (m = 4 / 8 would lock to the square / body-diagonal lattice axes)."""
import sys
import numpy as np
sys.argv += []
exec(open("exp/2026/10/24/geo/slices.py").read().split("AX = ")[0].replace("args = p.parse_args()", "args = p.parse_args(['--P', '192', '--a', '20', '--out', 'x', 's=s'])"))
B = "/bk/dst/2026/10/22/sph/"
equ = plane(2)
pts0, g0 = loops(np.load(B + "seed_a20.npz")["phi"], equ, "x", "y")
for lab, f in [("t200", "end_a20.npz"), ("t450", "end_a20_T450.npz")]:
    pts, g = loops(np.load(B + f)["phi"], equ, "x", "y")
    d = (g - g0) ** 2
    rr, th = np.hypot(pts[:, 0], pts[:, 1]), np.arctan2(pts[:, 1], pts[:, 0])
    for r1, r2 in [(0, 10), (10, 20), (20, 26)]:
        m = (rr >= r1) & (rr < r2)
        P = np.array([abs(np.sum(d[m] * np.exp(-1j * k * th[m]))) for k in range(13)]) / d[m].sum()
        c4 = np.sum(d[m] * np.exp(-4j * th[m])); print(f"   max of m4 at {np.degrees(np.angle(c4)) / 4 % 90:.1f} deg (mod 90)")
        print(f"{lab} r {r1}-{r2}: share of d {d[m].sum() / d.sum():.3f}  |c_m|/c_0 m1..12: " + " ".join(f"{v:.2f}" for v in P[1:]))
