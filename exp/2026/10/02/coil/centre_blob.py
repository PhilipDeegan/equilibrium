"""Central low-density region of the N 8 big-torus spring: per snapshot, density along the z axis and along x at the
spring's mid-plane, min density near the axis, and phase-winding checks: (a) along a small loop in the meridional
(x, z) plane around the density minimum (a vortex ring centred on the axis would wind 1), (b) the pair-angle change
along the axis through the structure (flow through the hole)."""
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from equilibrium.substrate.walk import rest_state, to_complex

name = "springcpp_N8_S1_q1_a20_r60_xi3.9_m0.8_R125_z0_wxi_rc_long"
d = Path("/bk/dst/2026/10/02/coil") / name
pos = np.load(d / "pos.npy")
tree = cKDTree(pos)
x, y, z = pos.T
v, om = rest_state(0.8)
mu = om + 0.0082 * 1.0
rec = np.load(f"{name}_rec.npz")["rec"] if Path(f"{name}_rec.npz").exists() else None
for tt in [int(a) for a in sys.argv[1:]]:
    s = np.load(d / f"state_t{tt:05d}.npz")
    L, R = s["L"], s["R"]
    rho = np.sum(L**2, 1) + np.sum(R**2, 1)
    amp = np.concatenate([to_complex(L), to_complex(R)], 1) @ v.conj() * np.exp(1j * mu * tt)
    ax_ = (np.hypot(x, y) < 3)
    i0 = np.argmin(np.where(ax_ & (np.abs(z) < 60), rho, 9))
    xm, zm = np.hypot(x[i0], y[i0]), z[i0]
    near = np.hypot(np.hypot(x, y), z - zm) < 12
    print(f"t {tt}: axis min rho {rho[i0]:.3f} at z {zm:+.1f}; cells rho<0.25 within 12 of it: {int(((rho < 0.25) & near).sum())}")
    zz = np.arange(-40, 41, 4.0)
    idx = tree.query(np.stack([0 * zz, 0 * zz, zm + zz], 1))[1]
    print("   rho along axis (z - zm = -40..40 step 4): " + " ".join(f"{r:.2f}" for r in rho[idx]))
    xx = np.arange(0, 41, 4.0)
    idx = tree.query(np.stack([xx, 0 * xx, zm + 0 * xx], 1))[1]
    print("   rho along x at z = zm (0..40 step 4):      " + " ".join(f"{r:.2f}" for r in rho[idx]))
    # angle change along the axis from z_m - 50 to z_m + 50
    zl = np.linspace(-50, 50, 401)
    ia = tree.query(np.stack([0 * zl, 0 * zl, zm + zl], 1))[1]
    dphi = np.sum(np.angle(amp[ia][1:] / amp[ia][:-1]))
    print(f"   pair-angle change along the axis over z_m +- 50: {dphi / (2 * np.pi):+.3f} turns")
    # winding round small meridional loops centred at (r_l, zm) for r_l = 2..10 (radius 1.5..6)
    for rl, rad in ((4, 3), (6, 4), (8, 5)):
        ps = np.linspace(0, 2 * np.pi, 128, endpoint=False)
        il = tree.query(np.stack([rl + rad * np.cos(ps), 0 * ps, zm + rad * np.sin(ps)], 1))[1]
        w = np.sum(np.angle(amp[il][np.r_[1:128, 0]] / amp[il])) / (2 * np.pi)
        print(f"   winding round meridional loop centre (x {rl}, z_m) radius {rad}: {w:+.2f}  (min rho on loop {rho[il].min():.2f})")
