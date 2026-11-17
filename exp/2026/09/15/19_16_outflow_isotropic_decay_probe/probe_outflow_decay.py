"""
Outflow BC reflection quality via decay-to-quiescence, not a single shell-|S|
snapshot.

Adapts test_saha.py's peaked-IC methodology (TestEquilibration): put a
Gaussian u bump at the centre of a small sphere and watch it relax. Saha's
ReciprocalBC is CONSERVED (relaxes to uniform); outflow is absorbing, so the
pass condition here is "decays to the floor and stays there" -- any
lingering reflected energy shows up as a plateau/re-growth in u_energy well
after the first round trip, instead of the clean single-pass drain a real
absorber gives.

Isotropic point-source-at-centre-of-a-sphere is also the best-case incidence
angle for a first-order Mur condition (wavefront curvature == boundary
curvature everywhere, no ring-source angular structure) -- this doubles as
the isotropic-source discriminator against BCC dispersion anisotropy
([[project_bcc_dispersion_anisotropy]]) queued in `now`.

No sponge/PML annulus anywhere: outflow's own absorbing thickness is zero by
construction, satisfying the <5%-annulus constraint trivially. `absorbing`
is run alongside as the known-100%-reflective control (a Neumann wall, see
[[project_bc_fn_inert_on_u]]) to calibrate what "no drain at all" looks like
on this same curve.
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
parser.add_argument("--n-round-trips", type=float, default=15.0)
parser.add_argument("--log-every", type=int, default=5)
parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--variants", type=str, default="outflow,absorbing")
parser.add_argument("--ic", type=str, default="bump", choices=["bump", "ring"])
parser.add_argument("--mode", type=str, default="square", choices=["minimal", "square", "octagon"])
parser.add_argument("--hops", type=int, default=2, help="ring size -- keep small relative to R so it has real room to radiate")
parser.add_argument("--axis", default="z")
parser.add_argument("--lam-ring", type=float, default=1.0)
parser.add_argument("--out-prefix", type=str, default="exp/2026/09/15/19_16_outflow_isotropic_decay_probe/decay")
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

    print(
        f"[{variant}/{args.ic}] N={cl.n_leaves} R={args.R} r_ring={r_ring:.2f} dt={dt:.4e} n_steps={n_steps} "
        f"t_round_trip={t_round_trip:.3f} energy_ic={energy_ic:.4e} "
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
            trace.append([t_phys, energy, u_energy, phi_energy, peak_u, center_rms])

    trace = np.array(trace)
    np.save(f"{args.out_prefix}_{args.ic}_{variant}_trace.npy", trace)

    print(f"[{variant}/{args.ic}] energy at each round trip (t, energy/energy_ic):", flush=True)
    for k in range(1, int(args.n_round_trips) + 1):
        t_target = k * t_round_trip
        idx = int(np.argmin(np.abs(trace[:, 0] - t_target)))
        print(f"  round_trip={k:2d}  t={trace[idx,0]:7.3f}  frac={trace[idx,1]/energy_ic:.4e}", flush=True)

    return trace, energy_ic, t_round_trip


def main():
    variants = args.variants.split(",")
    results = {v: run_variant(v) for v in variants}

    fig, ax = plt.subplots(figsize=(7, 5))
    for v, (trace, u_energy_ic, t_rt) in results.items():
        ax.semilogy(trace[:, 0], trace[:, 1] / u_energy_ic, label=f"{v} (u_energy/IC)")
    t_rt_ref = next(iter(results.values()))[2]
    for k in range(1, int(args.n_round_trips) + 1):
        ax.axvline(k * t_rt_ref, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("t")
    ax.set_ylabel("u_energy / u_energy_ic")
    ax.set_title(f"Outflow decay-to-quiescence, R={args.R} ic={args.ic} (dotted = predicted round trips)")
    ax.legend()
    fig.tight_layout()
    out_path = f"{args.out_prefix}_{args.ic}_energy_vs_t.png"
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
