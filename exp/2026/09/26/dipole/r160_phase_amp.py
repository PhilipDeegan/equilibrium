"""Split the R160 field error into phase and amplitude: fit
X_num = a X_exact(t) + b X_exact(t + T/4) on each shell; phase lag
delta = atan2(-b, a) ... per unit kr, amplitude sqrt(a^2 + b^2)."""
import sys
from pathlib import Path

import numpy as np

sys.argv_orig = list(sys.argv)
sys.argv = [sys.argv[0], "--snaps", "12604"]
src = (Path(__file__).parent / "r160_vs_exact.py").read_text().split('snap_dir = Path("/bk')[0]
exec(src)
snap_dir = Path("/bk/dst/2026/09/26/dipole_R160_h0.125")
SNAPS = [int(v) for v in (sys.argv_orig[1:] or ["10083", "12604"])]
lat = lattice(0.125, snap_dir)
sets = shell_sets(lat, 0.125)
for sn in SNAPS:
    d = np.load(snap_dir / f"state_{sn:06d}.npz")
    t = float(d["t"]); dt = t / sn
    u, phi, e = d["u"], d["phi"], d["e"]
    pr, er = (lambda j: phi[:, j].T), (lambda j: e[:, j].T)
    print(f"step {sn} t={t:.4f}")
    print(" r    kr     field  amp     phase(rad)  phase/(kr)  resid/amp after fit")
    for a, idx in enumerate(sets):
        x = lat.pos[idx]
        An, Bn, un, En = fields(lat, idx, u, pr, er)
        X0 = exact_all(x, t, dt); X9 = exact_all(x, t + per / 4, dt)
        for name, n_, e0, e9 in (("A", An, X0[0], X9[0]), ("E", En, X0[3], X9[3]), ("B", Bn, X0[1], X9[1]), ("u", un, X0[2], X9[2])):
            M = np.stack([e0.ravel(), e9.ravel()], 1)
            (ca, cb), *_ = np.linalg.lstsq(M, n_.ravel(), rcond=None)
            res = np.sqrt(((n_.ravel() - M @ [ca, cb]) ** 2).sum() / (0.5 * (e0**2 + e9**2).sum()))
            # X9 = X at psi + pi/2, so X_num = amp X(psi - delta) -> a = amp cos(delta), b = -amp sin(delta)
            dl = np.arctan2(-cb, ca)
            print(f"{r_shells[a]:4.1f} {k * r_shells[a]:5.2f}  {name}   {np.hypot(ca, cb):.4f}  {dl:+.4f}     {dl / (k * r_shells[a]):+.5f}    {res:.4f}")
    del d
