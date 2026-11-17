"""Candidate in an ambient bath with lam from the total local flux (conditional test, 2026-09-30).

Rule (held assumptions labelled):
    1 + lam = (1 + lam_far) (1 + alpha X)
  --source potential (default, Philip 09-30: lam follows the potential, 1/r like Newtonian gravity):
      X = psi, lattice Poisson G^T G psi = e - e_bath (energy density minus the bath mean: uniform background
      subtracted = Jeans swindle, ASSUMPTION), psi = 0 on the boundary cells (exterior 1/r - 1/R).
  --source flux: X = F = sqrt(1/2 sum_k phi_k^2) (cell flux magnitude, 1/r^2 exterior = the field, not potential).
  - lam_far uniform (near-Earth 5.5e-7): ASSUMPTION (Philip 09-30: far may not be uniform at electron scale).
  - local depletion from the TOTAL local flux: candidate + ambient bath together (lam = depletion proxy).
Discrete step() with face sigma, implicit midpoint (H exact for any state-dependent sigma), AmbientBC: sponge
relaxes toward an ambient bath (closed copy, same rule), bath relaxed under the rule before the candidate is
inserted. Bath set by the lam it produces: T = (lam_bath / alpha)^2 / 7. lam_bath = 0 -> cold vacuum control.
Metric: departure (domain - bath) energy in the core r < 4.5 over its initial value.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from scipy.sparse.linalg import splu

from equilibrium.substrate.ambient_bc import AmbientBC, _gradient_operator
from equilibrium.substrate.lattice import _W, DampedLattice, wave_speed

out = Path(__file__).parent
bk = Path("/bk/dst/2026/09/30/ambient_flux_lam")
p = argparse.ArgumentParser()
p.add_argument("--R", type=float, default=12.0)
p.add_argument("--width", type=float, default=1.5)
p.add_argument("--amp", type=float, default=1.0)
p.add_argument("--crossings", type=float, default=20.0)
p.add_argument("--relax", type=float, default=5.0, help="bath relaxation, crossings")
p.add_argument("--lam_far", type=float, default=5.5e-7)
p.add_argument("--alpha", type=float, required=True)
p.add_argument("--lam_baths", default="0,0.01,0.1,1")
p.add_argument("--ic", choices=["gauss", "dipole"], required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--seed_tag", action="store_true", help="append _s<seed> to output names")
p.add_argument("--tag", default="", help="extra output-name suffix")
p.add_argument("--amb_coef", type=float, default=0.2, help="lam_amb = amb_coef alpha T when --lam_amb auto")
p.add_argument("--source", choices=["potential", "flux"], default="potential")
p.add_argument("--Ts", default=None, help="potential source: bath temperatures (replaces lam_baths)")
p.add_argument("--lam_amb", default="0", help="held ambient depletion level (uniform), or 'auto' = 0.2 alpha T "
               "(twice the bath's measured sub-mean dip, -0.08 alpha T): 'inside a plasma' stand-in")
args = p.parse_args()

c = wave_speed()


_poisson = {}


def _psi(lat, src):
    key = id(lat.nb)
    if key not in _poisson:
        G, *_ = _gradient_operator(lat.nb, _W)
        L = (G.T @ G).tocsr()
        inner = np.where(~np.any(lat.nb < 0, axis=1))[0]
        _poisson[key] = (inner, splu(L[inner][:, inner].tocsc()))
    inner, lu = _poisson[key]
    psi = np.zeros(len(src))
    psi[inner] = lu.solve(src[inner])
    return psi


E_BG = 0.0
LAM_AMB = 0.0


def lam_of(lat, u, phi):
    if args.source == "flux":
        X = np.sqrt(0.5 * np.sum(np.where(lat.nb >= 0, phi, 0.0) ** 2, axis=1))
    else:
        X = _psi(lat, e_cell(lat.nb, u, phi) - E_BG)
    lam = (1.0 + args.lam_far) * (1.0 + LAM_AMB + args.alpha * X) - 1.0
    if lam.min() < 0.0:
        raise RuntimeError(f"total depletion {lam.min():.3e} < 0: raise lam_amb or lower T")
    return lam


def rhs(lat, u, phi):
    valid = lat.nb >= 0
    nb = np.where(valid, lat.nb, 0)
    lam = lam_of(lat, u, phi)
    sig = 1.0 / (1.0 + lam)
    sw = np.where(valid, 0.5 * (sig[:, None] + sig[nb]) * _W[None, :], 0.0)
    return -(sw * phi).sum(axis=1), -sw * (u[nb] - u[:, None]), lam


def midpoint(lat, dt, tol=1e-13, max_iter=200):
    u0, p0 = lat.u, lat.phi
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


def candidate(lat):
    r = np.linalg.norm(lat.pos, axis=1)
    g = np.exp(-(r**2) / (2 * args.width**2))
    return args.amp * (g if args.ic == "gauss" else (lat.pos[:, 2] / args.width) * g)


bk.mkdir(parents=True, exist_ok=True)
levels = [float(x) for x in (args.Ts if args.source == "potential" else args.lam_baths).split(",")]
for lam_b in levels:
    t0 = time.time()
    lat = DampedLattice(radius_cells=args.R, sponge_fraction=0.25)
    r = np.linalg.norm(lat.pos, axis=1)
    core = r < 3.0 * args.width
    dt = lat.stable_dt()
    T = lam_b if args.source == "potential" else (lam_b / args.alpha) ** 2 / 7.0
    E_BG = 0.0
    LAM_AMB = args.amb_coef * args.alpha * T if args.lam_amb == "auto" else float(args.lam_amb)
    bc = AmbientBC(lat, T, rng=np.random.default_rng(args.seed), wave_step=midpoint, add_to_domain=False)
    E_BG = float(e_cell(lat.nb, bc.bath.u, bc.bath.phi).mean())
    bc.bath.lam = lam_of(bc.bath, bc.bath.u, bc.bath.phi)
    for _ in range(int(round(args.relax * 2 * args.R / c / dt))):
        midpoint(bc.bath, dt)
    lat.u = bc.bath.u + candidate(lat)
    lat.phi = bc.bath.phi.copy()
    lat.lam = lam_of(lat, lat.u, lat.phi)
    bc._set_target()
    du, dphi = bc.departure()
    E0 = e_cell(lat.nb, du, dphi)[core].sum()
    lam_bath_mean = float(bc.bath.lam.mean())
    n = int(round(args.crossings * 2 * args.R / c / dt))
    rec = []
    for s in range(n):
        bc.step(dt)
        if s % 20 == 0 or s == n - 1:
            du, dphi = bc.departure()
            ed = e_cell(lat.nb, du, dphi)[core]
            e_dom = e_cell(lat.nb, lat.u, lat.phi)
            e_bath = e_cell(lat.nb, bc.bath.u, bc.bath.phi)
            rec.append(((s + 1) * dt, ed.sum() / E0, float(np.mean(lat.lam[core])), float(np.mean(bc.bath.lam[core])),
                        float(lat.lam.max()), e_dom.sum(), e_dom[core].sum(), e_bath[core].sum(), e_bath.sum(),
                        float(np.mean(1.0 / (1.0 + lat.lam[core])))))
    rec = np.array(rec)
    t = rec[:, 0]
    late = rec[t > 0.7 * t[-1]]
    name = (f"{args.source}_{args.ic}_a{args.alpha:g}_" + (f"T{T:g}" if args.source == "potential" else f"lb{lam_b:g}")
            + (f"_amb{LAM_AMB:g}" if LAM_AMB else "") + args.tag + (f"_s{args.seed}" if args.seed_tag else ""))
    np.savez(out / f"rec_{name}.npz", rec=rec,
             cols="t Edep_core/E0 lam_core lam_bath_core lam_max Etot Ecore Ecore_bath Etot_bath sigma_core",
             T=T, lam_bath_mean=lam_bath_mean, E0=E0, Ecore0=e_cell(lat.nb, candidate(lat), 0 * lat.phi)[core].sum())
    np.savez(bk / f"end_{name}.npz", u=lat.u, phi=lat.phi, lam=lat.lam, bath_u=bc.bath.u, bath_phi=bc.bath.phi,
             bath_lam=bc.bath.lam, pos=lat.pos)
    print(f"{name}: alpha_eff={args.alpha / (1 + LAM_AMB):.4g} T={T:.3e} bath lam mean {lam_bath_mean:.3e}  Edep_core late {late[:, 1].mean():.4e}  "
          f"(t=0.25/0.5/1: {rec[len(rec) // 4, 1]:.3e} {rec[len(rec) // 2, 1]:.3e} {rec[-1, 1]:.3e})  "
          f"lam core late {late[:, 2].mean():.3e} vs bath {late[:, 3].mean():.3e}  lam_max {rec[-1, 4]:.3e}  "
          f"{time.time() - t0:.0f}s", flush=True)
