"""Exact GP (NLS) straight q = 1 vortex profile in the walk's own small-k units: xi^2 (f'' + f'/r - f/r^2) + f - f^3 = 0,
xi = the --walk_xi core width (omega = om0 + n k^2 / (6 m), g = n / (6 m rho0 xi^2)). Compared with the seed form
d / sqrt(d^2 + xi^2) and with the relaxed coil core (branch/status.md: rho 0.5 at d_core 5.25, d measured to the
rho < 0.02 core-cell set). Saves (d, rho) in core_profile.py's npz format for spring_box.py --profile."""
import argparse

import numpy as np
from scipy.integrate import solve_bvp

p = argparse.ArgumentParser()
p.add_argument("--xi", type=float, default=3.9)
p.add_argument("--save", default="")
args = p.parse_args()
L = 30.0  # r in units of xi
r = np.linspace(1e-4, L, 4000)


def ode(r, y):
    f, fp = y
    return np.vstack([fp, -fp / r + f / r**2 - f + f**3])


def bc(ya, yb):
    return np.array([ya[0] - ya[1] * r[0], yb[0] - (1 - 1 / (2 * L * L))])  # f ~ c r at 0; f -> 1 - 1/(2 r^2)


s = solve_bvp(ode, bc, r, np.vstack([r / np.sqrt(r * r + 2), 2 / (r * r + 2)**1.5]), tol=1e-8, max_nodes=200000)
assert s.success, s.message
f = s.sol(r)[0]
rho = f**2
xi = args.xi


def at(rr, val):
    return np.interp(val, rr, r)


print(f"GP f'(0) = {s.sol(1e-4)[1]:.4f} (lit. 0.5831)")
for lbl, rr in [("GP exact", rho), ("seed d/sqrt(d^2+xi^2)", r**2 / (r**2 + 1)), ("Pade d/sqrt(d^2+2xi^2)", r**2 / (r**2 + 2))]:
    r50, r02 = at(rr, 0.5) * xi, at(rr, 0.02) * xi
    print(f"{lbl:24s}: rho 0.5 at d {r50:.2f}, rho 0.02 at {r02:.2f} -> distance to the core set at rho 0.5: "
          f"{r50 - r02:.2f}; rho at 2 xi {np.interp(2, r, rr):.3f}, at 5 xi {np.interp(5, r, rr):.4f}")
print("relaxed coil (branch/status.md): rho 0.5 at d_core 5.25 (seed measured 3.3); xi_eff label 5.2")
if args.save:
    d = r * xi
    np.savez(args.save, d=d, rho=rho)
    print(f"saved {args.save} (d 0..{d[-1]:.0f})")
