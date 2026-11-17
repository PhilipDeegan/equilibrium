"""Self-sourced lam, follow-up of exp/2026/09/29/electron_survival (next items 1a, 1b).

Same rule: each step lam = eps + alpha (u^2 + 1/2 sum phi^2), discrete step(), DampedLattice rate
sponge. step() is linear in (u, phi) at fixed lam, so (alpha, amp) -> (alpha / s^2, s amp) is an
exact symmetry: only g = alpha amp^2 matters (checked by --mode scaling).

--mode long : long runs, time series of core energy, energy-weighted core sigma, core radius,
              lam_max; full state snapshots to /bk/dst/2026/09/30/electron_survival/.
--mode scan : retention vs g at amp = 1 (alpha = g), several widths.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from equilibrium.substrate.lattice import _W, DampedLattice, wave_speed

out = Path(__file__).parent
bk = Path("/bk/dst/2026/09/30/electron_survival")
p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["long", "scan", "scaling", "energy"], required=True)
p.add_argument("--R", type=float, default=12.0)
p.add_argument("--widths", default="1.5")
p.add_argument("--crossings", type=float, default=60.0)
p.add_argument("--eps", type=float, default=0.0)
p.add_argument("--gs", default="36")
p.add_argument("--ics", default="gauss,dipole")
p.add_argument("--snap_every", type=float, default=5.0, help="crossings between snapshots")
p.add_argument("--tag", default="")
p.add_argument("--dt_frac", type=float, default=1.0)
p.add_argument("--integrator", choices=["euler", "midpoint"], default="euler")
args = p.parse_args()

c = wave_speed()


def build(ic, amp, width):
    lat = DampedLattice(radius_cells=args.R, sponge_fraction=0.25)
    r = np.linalg.norm(lat.pos, axis=1)
    g = np.exp(-(r**2) / (2 * width**2))
    lat.u[:] = amp * (g if ic == "gauss" else (lat.pos[:, 2] / width) * g)
    return lat, r


def e_cell(lat):
    return 0.5 * lat.u**2 + 0.25 * np.sum(np.where(lat.nb >= 0, lat.phi, 0.0) ** 2, axis=1)


def rhs(lat, alpha, u, phi):
    valid = lat.nb >= 0
    nb = np.where(valid, lat.nb, 0)
    lam = args.eps + alpha * (u**2 + 0.5 * np.sum(np.where(valid, phi, 0.0) ** 2, axis=1))
    sig = 1.0 / (1.0 + lam)
    sw = np.where(valid, 0.5 * (sig[:, None] + sig[nb]) * _W[None, :], 0.0)
    return -(sw * phi).sum(axis=1), -sw * (u[nb] - u[:, None]), lam


def step_midpoint(lat, alpha, dt, tol=1e-13, max_iter=200):
    """Implicit midpoint: x1 = x0 + dt f(x_mid), sigma at x_mid -> H conserved exactly (quadratic H, skew J(x))."""
    u0, p0 = lat.u, lat.phi
    du, dp, _ = rhs(lat, alpha, u0, p0)
    u1, p1 = u0 + dt * du, p0 + dt * dp
    for it in range(max_iter):
        du, dp, lam = rhs(lat, alpha, 0.5 * (u0 + u1), 0.5 * (p0 + p1))
        un, pn = u0 + dt * du, p0 + dt * dp
        err = max(np.abs(un - u1).max(), np.abs(pn - p1).max())
        u1, p1 = un, pn
        if err < tol:
            break
    else:
        raise RuntimeError(f"midpoint did not converge: {err:.2e}")
    lat.u, lat.phi, lat.lam[:] = u1, p1, lam
    return it + 1


def split_core(lat, core):
    i, k = np.nonzero(lat.nb >= 0)
    j = lat.nb[i, k]
    once = i < j
    fi, fk, fj = i[once], k[once], j[once]
    w = _W[fk]
    F = len(fi)
    G = sp.csr_matrix((np.concatenate([w, -w]), (np.concatenate([np.arange(F)] * 2), np.concatenate([fj, fi]))), shape=(F, lat.N))
    f = lat.phi[fi, fk]
    fg = G @ lsqr(G, f, atol=1e-12, btol=1e-12, iter_lim=20000)[0]
    fc = core[fi] & core[fj]
    eu, eg, es = 0.5 * np.sum(lat.u[core] ** 2), 0.5 * np.sum(fg[fc] ** 2), 0.5 * np.sum((f - fg)[fc] ** 2)
    t = eu + eg + es
    return eu / t, eg / t, es / t


def run(ic, alpha, amp, width, crossings, snap_dir=None, every=20):
    lat, r = build(ic, amp, width)
    r_core = 3.0 * width
    core = r < r_core
    dt = lat.stable_dt() * args.dt_frac
    n = int(round(crossings * 2 * args.R / c / dt))
    snap_n = int(round(args.snap_every * 2 * args.R / c / dt))
    E0_core = e_cell(lat)[core].sum()
    rec = []
    for s in range(n):
        if args.integrator == "midpoint":
            step_midpoint(lat, alpha, dt)
            lat.apply_sponge(dt)
        else:
            lat.lam[:] = args.eps + alpha * 2.0 * e_cell(lat)
            lat.step(dt)
        if s % every == 0 or s == n - 1:
            e = e_cell(lat)
            ec = e[core]
            sig = 1.0 / (1.0 + lat.lam[core])
            rec.append(((s + 1) * dt, ec.sum() / E0_core, np.sum(ec * sig) / ec.sum(), sig.min(),
                        np.sqrt(np.sum(ec * r[core] ** 2) / ec.sum()), lat.lam.max(), e.sum()))
        if snap_dir is not None and ((s + 1) % snap_n == 0 or s == n - 1):
            np.savez(snap_dir / f"state_{s + 1:07d}.npz", t=(s + 1) * dt, u=lat.u, phi=lat.phi, lam=lat.lam)
    return lat, core, np.array(rec)


COLS = "t Ecore/E0 sigma_core(E-weighted) sigma_min r_rms lam_max Etot"

if args.mode == "energy":
    from equilibrium.substrate.lattice import SubstrateLattice
    for integ in ("euler", "midpoint"):
        for frac in (1.0, 0.5, 0.25):
            lat = SubstrateLattice(radius_cells=8.0)
            r = np.linalg.norm(lat.pos, axis=1)
            lat.u[:] = (lat.pos[:, 2] / 1.5) * np.exp(-(r**2) / (2 * 1.5**2))
            H0 = e_cell(lat).sum()
            dt = lat.stable_dt() * frac
            its, dev = [], 0.0
            for s in range(int(round(100 / dt))):
                if integ == "midpoint":
                    its.append(step_midpoint(lat, 100.0, dt))
                else:
                    lat.lam[:] = 100.0 * 2.0 * e_cell(lat)
                    lat.step(dt)
                dev = max(dev, abs(e_cell(lat).sum() / H0 - 1))
            print(f"{integ:8s} dt_frac={frac:5.3f}: max |H/H0 - 1| over t<100 = {dev:.3e}"
                  + (f"  iterations mean {np.mean(its):.1f} max {max(its)}" if its else ""), flush=True)

elif args.mode == "scaling":
    for alpha, amp in [(4.0, 3.0), (36.0, 1.0)]:
        lat, core, rec = run("gauss", alpha, amp, 1.5, 3.0)
        print(f"alpha={alpha} amp={amp}: Ecore/E0 {rec[-1, 1]:.12e} lam_max {rec[-1, 5]:.12e} "
              f"u/amp[0] {lat.u[0] / amp:.12e}", flush=True)

elif args.mode == "long":
    for ic in args.ics.split(","):
        for g in [float(x) for x in args.gs.split(",")]:
            for width in [float(x) for x in args.widths.split(",")]:
                name = f"{ic}_g{g:g}_w{width:g}{args.tag}"
                sd = bk / name
                sd.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                lat, core, rec = run(ic, g, 1.0, width, args.crossings, snap_dir=sd)
                np.save(sd / "pos.npy", lat.pos)
                np.savez(out / f"long_{name}.npz", rec=rec, cols=COLS)
                print(f"{name}: {time.time() - t0:.0f}s  split u/grad/sol {'/'.join(f'{x:.3f}' for x in split_core(lat, core))}",
                      flush=True)
                for q in (0.1, 0.25, 0.5, 0.75, 1.0):
                    row = rec[min(int(q * (len(rec) - 1)), len(rec) - 1)]
                    print("   " + "  ".join(f"{v:.4e}" for v in row), flush=True)

else:
    rows = []
    for ic in args.ics.split(","):
        for width in [float(x) for x in args.widths.split(",")]:
            for g in [float(x) for x in args.gs.split(",")]:
                lat, core, rec = run(ic, g, 1.0, width, args.crossings)
                t = rec[:, 0]
                late = rec[t > 0.7 * t[-1]]
                su, sg, ss = split_core(lat, core)
                row = (ic, width, g, late[:, 1].mean(), late[:, 2].mean(), late[:, 4].mean(), rec[-1, 5], su, sg, ss)
                rows.append(row)
                print(f"{ic:6s} w={width:g} g={g:6.1f}: Ecore late {row[3]:.4e}  sigma_core {row[4]:.4e}  "
                      f"r_rms {row[5]:.3f}  lam_max {row[6]:.3e}  split {su:.3f}/{sg:.3f}/{ss:.3f}", flush=True)
                np.savez(out / f"scan_{ic}_w{width:g}_g{g:g}{args.tag}.npz", rec=rec, cols=COLS,
                         u=lat.u, phi=lat.phi, lam=lat.lam, pos=lat.pos)
    np.save(out / f"scan_rows{args.tag}.npy", np.array(rows, dtype=object), allow_pickle=True)
