"""Toroidal shape modes of the spring core over time (saved states): core cells (rho < 0.25) near the structure,
weights 1 - rho; A_m = |sum w r_c e^{i m th}| / sum w r_c for m = 1, 2, 4 (r_c = cylinder radius of the core cell,
th = toroidal angle), and the m = 2 orientation. Exponential vs linear growth of A_2 tells instability vs forcing."""
from pathlib import Path
import numpy as np

name = "springcpp_N4_S1_q1_a10_r30_xi3.9_m0.8_R110_z0_wxi_rc_long"
d = Path("/bk/dst/2026/10/02/coil") / name
pos = np.load(d / "pos.npy")
x, y, z = pos.T
rc_ = np.hypot(x, y)
th = np.arctan2(y, x)
inner = np.linalg.norm(pos, axis=1) < 0.75 * np.linalg.norm(pos, axis=1).max()
rows = []
for f in sorted(d.glob("state_t*.npz")):
    s = np.load(f)
    rho = np.sum(s["L"] ** 2, 1) + np.sum(s["R"] ** 2, 1)
    c = (rho < 0.25) & inner
    w = (1 - rho[c]) * rc_[c]
    A = {m: np.abs(np.sum(w * np.exp(1j * m * th[c]))) / w.sum() for m in (1, 2, 3, 4)}
    or2 = np.degrees(np.angle(np.sum(w * np.exp(2j * th[c])))) / 2
    rows.append((int(s["t"]), A[1], A[2], A[3], A[4], or2))
    print(f"t {int(s['t']):6d}: A1 {A[1]:.4f} A2 {A[2]:.4f} A3 {A[3]:.4f} A4 {A[4]:.4f}  m=2 axis {or2:+6.1f} deg", flush=True)
r = np.array(rows)
np.save("shape_modes_long.npy", r)
late = r[:, 0] >= 4000
k = np.polyfit(r[late, 0], np.log(r[late, 2]), 1)[0]
print(f"A2 log-slope (t >= 4000): {k:.2e} per tick (e-fold {1 / k:.0f} ticks)")
