"""
Direct raw-lattice-field comparison between two sponge damp_max values, prompted
by a user observation that the rendered XZ frames "look different" between the
old (damp_max=0.003) and new (damp_max=0.01) R=150 chains at t=45 -- the earlier
shell-trace check (r=8-24, deep inside the sponge boundary) showed byte-identical
values at t=45 because those shells are far from the sponge and don't see any
effect until a reflection returns (predicted ~t=68 at R=150). But the rendered
frame covers the WHOLE slab out to r~R, including the edge/sponge region, which
the sponge damps directly and immediately as soon as the outgoing t=0 burst
reaches it (one-way light-crossing time, well before any round-trip reflection)
-- that's a real, independent mechanism the shell check didn't cover, and the
right way to check it is a direct field diff with a SHARED color scale, not two
independently-rendered (per-frame-percentile-normalized) plots.

Builds one IC, runs it twice (damp_max=A, damp_max=B) from t=0, and diffs the
raw phi/u fields and the derived |S| directly at matched steps -- avoids
depending on the old run's now-deleted checkpoints/frames entirely.
"""

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from equilibrium.substrate.energy_flux import local_energy_flux_vector
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import find_toroidal_ring, init_toroidal_winding, leaf_nb_j

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--R", type=float, default=30.0)
parser.add_argument("--hops", type=int, default=4)
parser.add_argument("--mode", type=str, default="square", choices=["square", "octagon"])
parser.add_argument("--amplitude", type=float, default=1.0)
parser.add_argument("--lam-ring", type=float, default=1.0)
parser.add_argument("--axis", default="z")
parser.add_argument("--t-end", type=float, default=10.0, help="physical time to run to (well past one-way arrival at the sponge, before round-trip)")
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-a", type=float, default=0.003, help="'old' config")
parser.add_argument("--damp-b", type=float, default=0.01, help="'new' config")
parser.add_argument("--threads", type=int, default=8)
parser.add_argument("--slab-half-thickness", type=float, default=1.5)
parser.add_argument("--out-prefix", type=str, default="exp/2026/09/08/12_15_poynting_time_evolution/sponge_field_diff")


def build_and_run(args, damp_max, n_steps, dt):
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    cl = alat[0]
    _lat.build_bcc_sphere(cl, args.R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    alat.set_n_threads(args.threads)

    r_all = np.linalg.norm(np.array(cl.leaf_pos), axis=1)
    r_max = float(r_all.max())
    r_inner = r_max * (1.0 - args.sponge)
    cl.set_bc(_lat.BoundaryCondition.quadratic_sponge(target=0.0, damp_max=damp_max, r_all=r_all, r_inner=r_inner))

    ring_js, ring_faces = find_toroidal_ring(cl, axis=args.axis, mode=args.mode, hops=args.hops)
    init_toroidal_winding(cl, ring_js, ring_faces, amplitude=args.amplitude, phi_only=True)

    lam = np.array(cl.leaf_lam)
    for j in ring_js:
        lam[j] = args.lam_ring
    cl.set_leaf_lam(lam)

    for _ in range(n_steps):
        alat.step(0, dt)
        alat.step_sponge(0)

    pos = np.array(cl.leaf_pos)
    nb_j = leaf_nb_j(cl)
    phi = np.asarray(cl.leaf_phi).T.copy()
    u = np.array(cl.leaf_u)
    S3 = local_energy_flux_vector(pos, nb_j, phi, u)
    Smag = np.linalg.norm(S3, axis=1)
    return pos, phi, u, Smag, r_inner


def main():
    args = parser.parse_args()

    # dt from a throwaway build (mesh is deterministic from R/hops/mode, same for both configs)
    alat0 = _lat.AdaptiveLattice()
    alat0.add_cluster()
    cl0 = alat0[0]
    _lat.build_bcc_sphere(cl0, args.R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat0.rebuild(0)
    dt = compute_dt(cl0)
    n_steps = int(round(args.t_end / dt))
    t_actual = n_steps * dt
    print(f"R={args.R} hops={args.hops} mode={args.mode} dt={dt:.4e} n_steps={n_steps} t={t_actual:.3f}", flush=True)

    print(f"running config A: damp_max={args.damp_a}", flush=True)
    pos_a, phi_a, u_a, Smag_a, r_inner = build_and_run(args, args.damp_a, n_steps, dt)
    print(f"running config B: damp_max={args.damp_b}", flush=True)
    pos_b, phi_b, u_b, Smag_b, _ = build_and_run(args, args.damp_b, n_steps, dt)

    assert pos_a.shape == pos_b.shape and np.allclose(pos_a, pos_b), "mesh mismatch between configs -- not directly comparable"

    dphi = phi_a - phi_b
    du = u_a - u_b
    dS = Smag_a - Smag_b
    r = np.linalg.norm(pos_a, axis=1)

    print(f"r_inner={r_inner:.2f} r_max={r.max():.2f}")
    print(f"whole-domain: max|dphi|={np.abs(dphi).max():.3e} max|du|={np.abs(du).max():.3e} max|dS|={np.abs(dS).max():.3e}")

    edge = r >= r_inner
    interior = r < r_inner * 0.5
    for name, mask in [("edge (r>=r_inner)", edge), ("deep interior (r<0.5*r_inner)", interior)]:
        if mask.any():
            print(
                f"{name}: N={mask.sum()} mean|dphi|={np.abs(dphi[mask]).mean():.3e} "
                f"mean|du|={np.abs(du[mask]).mean():.3e} mean|dS|={np.abs(dS[mask]).mean():.3e} "
                f"mean_Smag_a={Smag_a[mask][np.isfinite(Smag_a[mask])].mean():.3e} "
                f"mean_Smag_b={Smag_b[mask][np.isfinite(Smag_b[mask])].mean():.3e}"
            )

    # shared-scale slab plot: A, B, and (A-B) side by side
    slab = np.abs(pos_a[:, 1]) < args.slab_half_thickness
    x, z = pos_a[slab, 0], pos_a[slab, 2]
    Sa, Sb = Smag_a[slab], Smag_b[slab]
    finite = np.isfinite(Sa) & (Sa > 0) & np.isfinite(Sb) & (Sb > 0)
    vmin = np.log10(np.percentile(np.concatenate([Sa[finite], Sb[finite]]), 1))
    vmax = np.log10(np.percentile(np.concatenate([Sa[finite], Sb[finite]]), 99))
    ca = np.log10(np.clip(Sa, 10.0**vmin, None))
    cb = np.log10(np.clip(Sb, 10.0**vmin, None))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, c, title in [
        (axes[0], ca, f"damp_max={args.damp_a} (old)"),
        (axes[1], cb, f"damp_max={args.damp_b} (new)"),
    ]:
        sc = ax.scatter(x, z, c=c, s=3, cmap="inferno", vmin=vmin, vmax=vmax, linewidths=0)
        ax.set_title(f"{title}\nSHARED log10|S| in [{vmin:.1f},{vmax:.1f}]", fontsize=10)
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    d = Sa - Sb
    dmax = np.abs(d[np.isfinite(d)]).max() if np.isfinite(d).any() else 1.0
    sc = axes[2].scatter(x, z, c=d, s=3, cmap="RdBu_r", vmin=-dmax, vmax=dmax, linewidths=0)
    axes[2].set_title("A - B  (linear, signed)", fontsize=10)
    axes[2].set_aspect("equal")
    fig.colorbar(sc, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(f"R={args.R} t={t_actual:.2f}  r_inner={r_inner:.2f}  shared color scale across A/B panels")
    fig.tight_layout()
    out_path = f"{args.out_prefix}_R{args.R:.0f}_t{t_actual:.1f}.png"
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
