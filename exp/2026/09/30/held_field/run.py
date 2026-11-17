"""Candidate in a held uniform field (next 10-01 item 3, CONDITIONAL): linear response per unit field.

Self-sourced lam rule of exp/2026/09/30/electron_survival (lam = alpha (u^2 + 1/2 sum phi^2), total local field:
candidate + held), implicit midpoint, cold vacuum, AmbientBC(T = 0) with static_phi = uniform_field(E z_hat).
Realistic fields are ~1e-17 of the field at r_e, so only the response per unit E can be read; E is scanned to check
linearity. Recorded on the departure (domain - held field), interior r < r_in:
  core / interior energy over E0, energy centroids (interior, core), u moment sum x u (interior).
E along z with the z-dipole is symmetric under (z -> -z, (u, phi) -> -(u, phi)): no centroid shift at any order.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from equilibrium.substrate.ambient_bc import AmbientBC, uniform_field
from equilibrium.substrate.lattice import _W, DampedLattice, wave_speed

out = Path(__file__).parent
bk = Path("/bk/dst/2026/09/30/held_field")
p = argparse.ArgumentParser()
p.add_argument("--R", type=float, default=12.0)
p.add_argument("--width", type=float, default=1.5)
p.add_argument("--g", type=float, default=100.0)
p.add_argument("--Es", default="0,0.001,0.003,0.01")
p.add_argument("--ic", choices=["gauss", "dipole"], required=True)
p.add_argument("--crossings", type=float, default=30.0)
p.add_argument("--axis", choices=["x", "z"], default="z", help="E direction (dipole axis is z)")
args = p.parse_args()

c = wave_speed()
alpha = args.g


def rhs(lat, u, phi):
    valid = lat.nb >= 0
    nb = np.where(valid, lat.nb, 0)
    lam = alpha * (u**2 + 0.5 * np.sum(np.where(valid, phi, 0.0) ** 2, axis=1))
    sig = 1.0 / (1.0 + lam)
    sw = np.where(valid, 0.5 * (sig[:, None] + sig[nb]) * _W[None, :], 0.0)
    return -(sw * phi).sum(axis=1), -sw * (u[nb] - u[:, None]), lam


def midpoint(lat, dt, tol=1e-13, max_iter=200):
    u0, p0 = lat.u, lat.phi
    if not (np.any(u0) or np.any(p0)):
        return
    du, dp, _ = rhs(lat, u0, p0)
    u1, p1 = u0 + dt * du, p0 + dt * dp
    for _ in range(max_iter):
        du, dp, lam = rhs(lat, 0.5 * (u0 + u1), 0.5 * (p0 + p1))
        un, pn = u0 + dt * du, p0 + dt * dp
        err = max(np.abs(un - u1).max(), np.abs(pn - p1).max())
        u1, p1 = un, pn
        if err < tol:
            break
    else:
        raise RuntimeError(f"midpoint did not converge: {err:.2e}")
    lat.u, lat.phi, lat.lam = u1, p1, lam


def e_cell(nb, u, phi):
    return 0.5 * u**2 + 0.25 * np.sum(np.where(nb >= 0, phi, 0.0) ** 2, axis=1)


bk.mkdir(parents=True, exist_ok=True)
for E in [float(x) for x in args.Es.split(",")]:
    t0 = time.time()
    lat = DampedLattice(radius_cells=args.R, sponge_fraction=0.25)
    r = np.linalg.norm(lat.pos, axis=1)
    z = lat.pos[:, 2]
    x = lat.pos
    core = r < 3.0 * args.width
    inner = r < 0.75 * args.R
    phi_E, _ = uniform_field(lat.pos, lat.nb, _W, [E, 0.0, 0.0] if args.axis == "x" else [0.0, 0.0, E])
    bc = AmbientBC(lat, 0.0, static_phi=phi_E, wave_step=midpoint)
    gss = np.exp(-(r**2) / (2 * args.width**2))
    lat.u = lat.u + (gss if args.ic == "gauss" else (z / args.width) * gss)
    lat.lam = rhs(lat, lat.u, lat.phi)[2]
    du, dphi = bc.departure()
    E0 = e_cell(lat.nb, du, dphi)[core].sum()
    dt = lat.stable_dt()
    n = int(round(args.crossings * 2 * args.R / c / dt))
    rec = []
    for s in range(n):
        bc.step(dt)
        if s % 20 == 0 or s == n - 1:
            du, dphi = bc.departure()
            e = e_cell(lat.nb, du, dphi)
            ei = e[inner]
            rec.append(((s + 1) * dt, e[core].sum() / E0, ei.sum() / E0, *(ei @ x[inner] / ei.sum()),
                        *(e[core] @ x[core] / e[core].sum()), *(du[inner] @ x[inner]),
                        float(np.mean(lat.lam[core])), float(np.mean(lat.lam[inner & ~core]))))
    rec = np.array(rec)
    name = f"{args.ic}_g{args.g:g}_E{E:g}{args.axis}"
    np.savez(out / f"rec_{name}.npz", rec=rec, cols="t Ecore/E0 Einner/E0 xc yc zc(inner) xc yc zc(core) px py pz(u, inner) lam_core lam_shell",
             E=E, E0=E0)
    np.savez(bk / f"end_{name}.npz", u=lat.u, phi=lat.phi, lam=lat.lam, static_phi=phi_E, pos=lat.pos)
    q = [len(rec) // 4, len(rec) // 2, len(rec) - 1]
    ax = {"x": 0, "z": 2}[args.axis]
    print(f"{name}: Ecore {' '.join(f'{rec[i, 1]:.3f}' for i in q)}  c_inner[E] {' '.join(f'{rec[i, 3 + ax]:+.4f}' for i in q)}  "
          f"c_core[E] {' '.join(f'{rec[i, 6 + ax]:+.4f}' for i in q)}  p_u[E] {' '.join(f'{rec[i, 9 + ax]:+.4e}' for i in q)}  "
          f"lam core/shell {rec[-1, 12]:.3e}/{rec[-1, 13]:.3e}  {time.time() - t0:.0f}s", flush=True)
