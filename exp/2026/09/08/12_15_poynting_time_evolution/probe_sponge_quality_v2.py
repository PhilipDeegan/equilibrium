"""
v2: the H(t)-based sponge comparison (probe_sponge_quality.py) was the
WRONG diagnostic -- total domain energy is dominated by the ring's own
near-field/reactive content that never reaches the boundary at all, so it
came back flat (99.96-99.97% retained) across every sponge config,
regardless of tuning; also 4000 steps is less than one period of the
ring's own natural oscillation (period~32), nowhere near enough cycles for
radiation loss to register in a global energy sum either way.

This version tracks the shell-averaged |S| signal instead (the same
diagnostic from probe_poynting_time_evolution.py) at shells near the
sponge (r=20, r=24) AND well inside it (r=8), over more than one full
oscillation period, for the same 4 sponge configs -- if a deeper/better-
tuned sponge genuinely reduces reflection, the outer shells should show a
cleaner (lower-amplitude, faster-decaying-after-first-pass) signal there
than the default; if all configs look equally messy, reflection likely
isn't the dominant cause of the late-time irregularity.
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from equilibrium.substrate.energy_flux import local_energy_flux_vector
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import find_toroidal_ring, init_toroidal_winding, leaf_nb_j

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--R", type=float, default=30.0)
parser.add_argument("--hops", type=int, default=4)
parser.add_argument("--amplitude", type=float, default=1.0)
parser.add_argument("--lam-ring", type=float, default=1.0)
parser.add_argument("--axis", default="z")
parser.add_argument("--n-steps", type=int, default=6500)  # > 1 period (period~32/dt~6240 steps)
parser.add_argument("--log-every", type=int, default=25)
parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--sector-half-angle", type=float, default=20.0)
parser.add_argument("--shell-width-frac", type=float, default=0.15)
parser.add_argument("--shells", type=str, default="8,20,24")


def run_config(args, sponge_fraction, damp_max, label, shells):
    ax_idx = {"x": 0, "y": 1, "z": 2}[args.axis]
    perp = [i for i in range(3) if i != ax_idx]

    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    cl = alat[0]
    _lat.build_bcc_sphere(cl, args.R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    alat.set_n_threads(args.threads)

    r_all = np.linalg.norm(np.array(cl.leaf_pos), axis=1)
    r_max = float(r_all.max())
    r_inner = r_max * (1.0 - sponge_fraction)
    cl.set_bc(_lat.BoundaryCondition.quadratic_sponge(target=0.0, damp_max=damp_max, r_all=r_all, r_inner=r_inner))

    ring_js, ring_faces = find_toroidal_ring(cl, axis=args.axis, mode="square", hops=args.hops)
    init_toroidal_winding(cl, ring_js, ring_faces, amplitude=args.amplitude, phi_only=True)
    pos = np.array(cl.leaf_pos)
    ring_pos = pos[ring_js]
    ring_centre = ring_pos.mean(axis=0)

    lam = np.array(cl.leaf_lam)
    for j in ring_js:
        lam[j] = args.lam_ring
    cl.set_leaf_lam(lam)

    dt = compute_dt(cl)
    print(f"[{label}] N={cl.n_leaves} sponge_fraction={sponge_fraction} damp_max={damp_max} r_inner={r_inner:.2f} dt={dt:.4e}", flush=True)

    nb_j = leaf_nb_j(cl)
    rel = pos - ring_centre
    ax_comp = rel[:, ax_idx]
    perp_comp = np.linalg.norm(rel[:, perp], axis=1)
    theta = np.degrees(np.arctan2(perp_comp, np.abs(ax_comp)))
    half = args.sector_half_angle
    equatorial = theta > (90.0 - half)
    r_from_centre = np.linalg.norm(rel, axis=1)
    shell_masks = [
        equatorial & (r_from_centre >= r0 - args.shell_width_frac * r0) & (r_from_centre <= r0 + args.shell_width_frac * r0)
        for r0 in shells
    ]

    trace = []
    for step in range(args.n_steps):
        do_log = (step % args.log_every == 0) or (step == args.n_steps - 1)
        phi_before = np.array(cl.leaf_phi).T.copy() if do_log else None
        alat.step(0, dt)
        alat.step_sponge(0)
        if do_log:
            u_after = np.array(cl.leaf_u)
            S3 = local_energy_flux_vector(pos, nb_j, phi_before, u_after)
            Smag = np.linalg.norm(S3, axis=1)
            t_phys = (step + 1) * dt
            row = [t_phys] + [float(np.mean(Smag[m])) if m.any() else float("nan") for m in shell_masks]
            trace.append(row)
    return np.array(trace)


def main():
    args = parser.parse_args()
    shells = [float(s) for s in args.shells.split(",")]
    configs = [
        (0.15, 0.003, "default"),
        (0.35, 0.001, "deeper_gentler"),
        (0.15, 0.02, "shallow_aggressive"),
        (0.35, 0.01, "deeper_aggressive"),
    ]
    results = {}
    for frac, damp, label in configs:
        trace = run_config(args, frac, damp, label, shells)
        np.save(f"exp/2026/09/08/12_15_poynting_time_evolution/sponge_v2_trace_{label}.npy", trace)
        results[label] = trace
        print(f"[{label}] done, {len(trace)} log points", flush=True)

    print("\n=== late-time (last third of run) outer-shell |S| comparison across configs ===")
    for label, trace in results.items():
        n = len(trace)
        late = trace[2 * n // 3 :]
        for i, r0 in enumerate(shells):
            col = late[:, i + 1]
            finite = col[np.isfinite(col) & (col > 0)]
            if finite.size:
                print(f"{label:20s} r={r0:4.0f}: mean={finite.mean():.4e} std={finite.std():.4e} max={finite.max():.4e}")
    print("done", flush=True)


if __name__ == "__main__":
    main()
