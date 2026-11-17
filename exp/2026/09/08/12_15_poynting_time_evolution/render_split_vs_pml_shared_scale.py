"""
Shared-colorscale XZ |S| comparison between the split (production, damp=0.003)
and PML (gamma=0.5) sponge variants at a matched checkpoint time -- the check
queued at the end of the 09-11/09-12 sessions to resolve Philip's "PML looks
more hectic" visual impression (a spatial-complexity claim) against the
already-confirmed amplitude finding (PML's peak |S| is measurably lower than
split's at every matched timestamp). Per-frame independent percentile
normalization (as used by probe_sponge_quality_v3.py's own frames) can make
two panels look differently "busy" even when the underlying data has the
same relative structure -- this script instead loads both raw checkpoints
directly and computes ONE shared vmin/vmax from their combined data, same
convention as probe_sponge_direct_field_diff.py.

Row-subset-only (|y|<slab-half-thickness slab), same perf-safe pattern as
probe_sponge_quality_v3.py's own plot-times frames -- never touches a
full-domain (N,14,3) temporary at N~18M.
"""

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from equilibrium.substrate.energy_flux import face_unit_vectors, local_energy_flux_vector
from tst.substrate.adaptive.adaptive_flux_tube import leaf_nb_j

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--split-h5", required=True)
parser.add_argument("--pml-h5", required=True)
parser.add_argument("--t-label", required=True, help="physical time this pair represents, for titles/filename")
parser.add_argument("--slab-half-thickness", type=float, default=1.5)
parser.add_argument(
    "--out",
    type=str,
    default="exp/2026/09/08/12_15_poynting_time_evolution/split_vs_pml_shared_scale",
)


def load_slab_Smag(path, slab_half_thickness):
    alat = _lat.AdaptiveLattice.load(path)
    cl = alat[0]
    pos = np.array(cl.leaf_pos)
    nb_j = leaf_nb_j(cl)
    u = np.array(cl.leaf_u)
    phi_full = np.asarray(cl.leaf_phi)  # view, (14,N)

    slab_mask = np.abs(pos[:, 1]) < slab_half_thickness
    slab_idx = np.where(slab_mask)[0]
    pos_s = pos[slab_idx]
    nb_s = nb_j[slab_idx]
    phi_s = phi_full[:, slab_idx].T.copy()
    u_own_s = u[slab_idx]
    n_hat_s = face_unit_vectors(pos, nb_s, pos_own=pos_s)
    S3 = local_energy_flux_vector(pos, nb_s, phi_s, u, n_hat=n_hat_s, u_own=u_own_s)
    Smag = np.linalg.norm(S3, axis=1)
    return pos_s[:, 0], pos_s[:, 2], Smag


def main():
    args = parser.parse_args()

    print(f"loading split: {args.split_h5}", flush=True)
    x_split, z_split, S_split = load_slab_Smag(args.split_h5, args.slab_half_thickness)
    print(f"loading pml: {args.pml_h5}", flush=True)
    x_pml, z_pml, S_pml = load_slab_Smag(args.pml_h5, args.slab_half_thickness)

    finite_split = np.isfinite(S_split) & (S_split > 0)
    finite_pml = np.isfinite(S_pml) & (S_pml > 0)
    combined = np.concatenate([S_split[finite_split], S_pml[finite_pml]])
    vmin = np.log10(np.percentile(combined, 1))
    vmax = np.log10(np.percentile(combined, 99))
    print(f"shared scale: log10|S| in [{vmin:.2f},{vmax:.2f}]  (from combined split+pml slab data)", flush=True)

    c_split = np.log10(np.clip(S_split, 10.0**vmin, None))
    c_pml = np.log10(np.clip(S_pml, 10.0**vmin, None))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, x, z, c, title in [
        (axes[0], x_split, z_split, c_split, "split (production, damp=0.003)"),
        (axes[1], x_pml, z_pml, c_pml, "pml (gamma=0.5)"),
    ]:
        sc = ax.scatter(x, z, c=c, s=3, cmap="inferno", vmin=vmin, vmax=vmax, linewidths=0)
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"t={args.t_label}  SHARED log10|S| in [{vmin:.1f},{vmax:.1f}] across both panels")
    fig.tight_layout()
    out_path = f"{args.out}_t{args.t_label}.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
