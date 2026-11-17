"""
outflow_tapered() discriminator: does giving the outflow BC a blended
relaxation-sink taper band (2026-09-16, [[project_bc_fn_inert_on_u]] idea 1)
fix the ring-source reflection deficit found 2026-09-15
(exp/2026/09/15/19_16_outflow_isotropic_decay_probe), given the working
theory that the ring's failure is angular/off-normal incidence, not the
zero-thickness interface itself?

Forked from that session's probe_outflow_decay.py (decay-to-quiescence
methodology, same IC/BC machinery) with two additions:

1. `outflow_taper` variant using the new `BoundaryCondition.outflow_tapered`
   C++ binding (`--taper-width`, in the same length units as R).
2. An OUTER-SHELL energy trace (r > --shell-r0) alongside the whole-domain
   one -- the whole-domain trace is a known false-negative for ring ICs
   (bc.md's caveat: a lam=1 ring is a quasi-bound structure, so >99.9% of
   its energy never leaves the near-field core and swamps the genuinely-
   radiated fraction in a whole-domain sum). The shell trace is the actual
   discriminator for the ring case; whole-domain is kept for the bump case
   and for a sanity cross-check.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import find_toroidal_ring, init_toroidal_winding

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--R", type=float, default=10.0)
parser.add_argument("--amp", type=float, default=5.0)
parser.add_argument("--sigma", type=float, default=2.0)
parser.add_argument("--c-lattice", type=float, default=np.sqrt(10.0), help="outflow's assumed local propagation speed")
parser.add_argument("--taper-width", type=float, default=3.0, help="outflow_tapered band width, same units as --R")
parser.add_argument("--n-round-trips", type=float, default=15.0)
parser.add_argument("--log-every", type=int, default=5)
parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--variants", type=str, default="outflow,outflow_taper,absorbing")
parser.add_argument("--ic", type=str, default="bump", choices=["bump", "ring", "dipole"])
parser.add_argument(
    "--dipole-sep",
    type=float,
    default=6.0,
    help="dipole IC only: pole separation along --axis (two opposite-sign Gaussian u bumps, "
    "zero net monopole moment by construction, l=1-dominated far field, no ring/winding/lam "
    "machinery at all) -- isolates whether the ring's angular reflection deficit is generic "
    "dipole-radiation physics or specific to the ring's own construction",
)
parser.add_argument("--mode", type=str, default="square", choices=["minimal", "square", "octagon"])
parser.add_argument("--hops", type=int, default=2, help="ring size -- keep small relative to R so it has real room to radiate")
parser.add_argument("--axis", default="z")
parser.add_argument("--lam-ring", type=float, default=1.0)
parser.add_argument("--shell-r0", type=float, default=None, help="outer-shell energy diagnostic inner radius; default 0.6*R")
parser.add_argument("--out-prefix", type=str, default="exp/2026/09/16/12_30_outflow_taper_probe/taper")
args = parser.parse_args()


def run_variant(variant):
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    cl = alat[0]
    _lat.build_bcc_sphere(cl, args.R, d1=1.0, u0=0.0, lam0=0.0)
    alat.rebuild(0)
    alat.set_n_threads(args.threads)

    pos = np.array(cl.leaf_pos)
    r = np.linalg.norm(pos, axis=1)

    if args.ic == "bump":
        u_ic = args.amp * np.exp(-(r**2) / (2.0 * args.sigma**2))
        cl.set_leaf_u(u_ic.copy())
        r_ring = 0.0
    elif args.ic == "dipole":
        ax_idx = {"x": 0, "y": 1, "z": 2}[args.axis]
        d = 0.5 * args.dipole_sep
        centre_plus = np.zeros(3)
        centre_plus[ax_idx] = d
        centre_minus = np.zeros(3)
        centre_minus[ax_idx] = -d
        r_plus = np.linalg.norm(pos - centre_plus, axis=1)
        r_minus = np.linalg.norm(pos - centre_minus, axis=1)
        u_ic = args.amp * np.exp(-(r_plus**2) / (2.0 * args.sigma**2)) - args.amp * np.exp(
            -(r_minus**2) / (2.0 * args.sigma**2)
        )
        cl.set_leaf_u(u_ic.copy())
        r_ring = 0.0
    else:
        ring_js, ring_faces = find_toroidal_ring(cl, axis=args.axis, mode=args.mode, hops=args.hops)
        if ring_js is None:
            raise RuntimeError(f"no ring found at R={args.R} hops={args.hops} -- shrink --hops")
        init_toroidal_winding(cl, ring_js, ring_faces, amplitude=args.amp, phi_only=True)
        lam = np.array(cl.leaf_lam)
        for j in ring_js:
            lam[j] = args.lam_ring
        cl.set_leaf_lam(lam)
        u_ic = np.array(cl.leaf_u)
        r_ring = float(np.median(np.linalg.norm(pos[ring_js], axis=1)))

    if variant == "outflow":
        cl.set_bc(_lat.BoundaryCondition.outflow(np.array(cl.leaf_nb), c_lattice=args.c_lattice))
    elif variant == "outflow_ledger":
        # Ledger-capped (passive) ghost, 2026-09-16 (bc.hpp's outflow_credit
        # doc comment) -- same ghost formula as "outflow", but a face can
        # never inject more energy over its own history than it has
        # genuinely absorbed. See doc/dst/bc.md.
        cl.set_bc(_lat.BoundaryCondition.outflow_ledger(np.array(cl.leaf_nb), c_lattice=args.c_lattice))
    elif variant == "outflow_taper":
        cl.set_bc(
            _lat.BoundaryCondition.outflow_tapered(
                np.array(cl.leaf_nb), pos, taper_width=args.taper_width, c_lattice=args.c_lattice
            )
        )
    elif variant == "absorbing":
        cl.set_bc(_lat.BoundaryCondition.absorbing())
    else:
        raise ValueError(f"unknown variant {variant}")

    vol = np.ones(cl.n_leaves)  # d1=1 uniform grid -> cell volume 1
    dt = compute_dt(cl)
    r_max = float(r.max())
    c_void = np.sqrt(10.0)
    t_round_trip = 2.0 * (r_max - r_ring) / c_void
    n_steps = int(round(args.n_round_trips * t_round_trip / dt))

    # bump IC has u_ic!=0/phi_ic=0; ring IC (phi_only) is the reverse -- track
    # u_energy+phi_energy together so the IC-normalized "energy" is nonzero
    # and comparable across both IC types.
    phi_ic = np.asarray(cl.leaf_phi)
    energy_ic = float(np.sum(u_ic**2 * vol)) + float(np.sum(phi_ic**2))
    center_idx = np.where(r < 1.5)[0]

    shell_r0 = args.shell_r0 if args.shell_r0 is not None else 0.6 * args.R
    shell_idx = np.where(r > shell_r0)[0]
    shell_energy_ic = float(np.sum(u_ic[shell_idx] ** 2 * vol[shell_idx])) + float(np.sum(phi_ic[:, shell_idx] ** 2))

    print(
        f"[{variant}/{args.ic}] N={cl.n_leaves} R={args.R} r_ring={r_ring:.2f} dt={dt:.4e} n_steps={n_steps} "
        f"t_round_trip={t_round_trip:.3f} energy_ic={energy_ic:.4e} shell_r0={shell_r0:.2f} "
        f"n_shell_cells={len(shell_idx)} shell_energy_ic={shell_energy_ic:.4e} "
        f"n_center_cells={len(center_idx)}",
        flush=True,
    )

    trace = []
    for step in range(n_steps):
        alat.step(0, dt)
        if step % args.log_every == 0 or step == n_steps - 1:
            u = np.array(cl.leaf_u)
            phi = np.asarray(cl.leaf_phi)
            t_phys = (step + 1) * dt
            u_energy = float(np.sum(u**2 * vol))
            phi_energy = float(np.sum(phi**2))
            energy = u_energy + phi_energy
            peak_u = float(np.max(np.abs(u)))
            center_rms = float(np.sqrt(np.mean(u[center_idx] ** 2))) if len(center_idx) else float("nan")
            shell_energy = float(np.sum(u[shell_idx] ** 2 * vol[shell_idx])) + float(np.sum(phi[:, shell_idx] ** 2))
            trace.append([t_phys, energy, u_energy, phi_energy, peak_u, center_rms, shell_energy])

    trace = np.array(trace)
    np.save(f"{args.out_prefix}_{args.ic}_{variant}_trace.npy", trace)

    print(f"[{variant}/{args.ic}] energy at each round trip (t, whole/IC, shell/shell_IC):", flush=True)
    for k in range(1, int(args.n_round_trips) + 1):
        t_target = k * t_round_trip
        idx = int(np.argmin(np.abs(trace[:, 0] - t_target)))
        shell_frac = trace[idx, 6] / shell_energy_ic if shell_energy_ic > 0 else float("nan")
        print(
            f"  round_trip={k:2d}  t={trace[idx,0]:7.3f}  whole_frac={trace[idx,1]/energy_ic:.4e}  "
            f"shell_frac={shell_frac:.4e}",
            flush=True,
        )

    return trace, energy_ic, t_round_trip, shell_energy_ic


def main():
    variants = args.variants.split(",")
    results = {v: run_variant(v) for v in variants}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for v, (trace, energy_ic, t_rt, shell_energy_ic) in results.items():
        ax1.semilogy(trace[:, 0], trace[:, 1] / energy_ic, label=v)
        shell_frac = trace[:, 6] / shell_energy_ic if shell_energy_ic > 0 else np.full(len(trace), np.nan)
        ax2.semilogy(trace[:, 0], shell_frac, label=v)
    t_rt_ref = next(iter(results.values()))[2]
    for ax in (ax1, ax2):
        for k in range(1, int(args.n_round_trips) + 1):
            ax.axvline(k * t_rt_ref, color="gray", lw=0.5, ls=":")
        ax.set_xlabel("t")
        ax.legend()
    ax1.set_ylabel("whole-domain energy / IC")
    ax2.set_ylabel("outer-shell energy / shell IC")
    fig.suptitle(f"outflow taper probe, R={args.R} ic={args.ic} taper_width={args.taper_width} (dotted = predicted round trips)")
    fig.tight_layout()
    out_path = f"{args.out_prefix}_{args.ic}_energy_vs_t.png"
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
