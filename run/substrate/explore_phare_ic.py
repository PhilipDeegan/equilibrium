"""
DST experiment: initialise from PHARE t=3 spheromak state.

Converts the PHARE EM_B.vtkhdf (all-primal, t=3) directly into DST ICs:
  - phi[i,j]  ← B(face_midpoint) · n_hat(i→j)   (nearest-neighbour from PHARE nodes)
  - lam[i]    ← lam_scale * |B|²(r_i)            (radial interpolation)

Both levels (Level0 dl=0.1, Level1 dl=0.05) are used; Level1 wins near the core.

The PHARE domain has centre (5.05, 5.05, 5.05) and spheromak R=1.0.
The DST domain is centred at the lattice origin with the PHARE R mapped to R_sph cells:
    phare_coord = centre + dst_pos / scale,  scale = R_sph / R_phare

Run:
    PYTHONPATH=$PWD python3 run/substrate/explore_phare_ic.py
    PYTHONPATH=$PWD python3 run/substrate/explore_phare_ic.py --R 8 --lam-scale 5 --steps 20000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import KDTree

sys.path.insert(0, ".")
from equilibrium.substrate.lattice import DampedLattice


def _phi_circulation(lat, centre, r_hop, dr=0.8):
    """Azimuthal phi circulation summed over faces on shell r_hop ± dr."""
    c = np.asarray(centre, dtype=float)
    nb_safe = np.where(lat.nb >= 0, lat.nb, 0)
    pos_j = lat.pos[nb_safe]
    pos_i = lat.pos[:, np.newaxis, :]
    mid = 0.5 * (pos_i + pos_j) - c
    r_mid = np.linalg.norm(mid, axis=-1)
    on_shell = lat._valid & (r_mid >= r_hop - dr) & (r_mid <= r_hop + dr)
    upper = on_shell & (np.arange(len(lat.pos))[:, None] < nb_safe)
    mx, my = mid[..., 0], mid[..., 1]
    rho = np.sqrt(mx**2 + my**2)
    rho_s = np.where(rho > 1e-10, rho, 1.0)
    diff = pos_j - pos_i
    d_len = np.linalg.norm(diff, axis=-1, keepdims=True)
    n_hat = diff / np.where(d_len > 0, d_len, 1.0)
    proj = n_hat[..., 0] * (-my / rho_s) + n_hat[..., 1] * (mx / rho_s)
    return float(np.sum(lat.phi[upper] * proj[upper]))


VTKHDF = Path("/home/deegan/git/phare/master/EM_B.vtkhdf")
PHARE_CENTRE = np.array([5.05, 5.05, 5.05])
R_PHARE = 1.0  # PHARE spheromak radius


# ---------------------------------------------------------------------------
# Load PHARE B field from vtkhdf — both levels, t=3 (step index 1)
# ---------------------------------------------------------------------------


def _load_phare_b(vtkhdf=VTKHDF, step=1):
    """
    Return (positions, B_vectors) arrays from all AMR levels at the given step.
    positions : (N, 3)  physical coords in PHARE units
    B_vectors : (N, 3)  [Bx, By, Bz] at each primal node
    Level1 nodes (finer) are appended last so KDTree queries prefer them near the core.
    """
    import h5py

    all_pos, all_B = [], []

    with h5py.File(vtkhdf, "r") as f:
        for lvl_name, dl in [("Level0", 0.1), ("Level1", 0.05)]:
            boxes_all = f[f"VTKHDF/{lvl_name}/AMRBox"][:]
            data_all = f[f"VTKHDF/{lvl_name}/PointData/data"][:]
            nbox_per = f[f"VTKHDF/Steps/{lvl_name}/NumberOfAMRBox"][:]
            pt_off = f[f"VTKHDF/Steps/{lvl_name}/PointDataOffset/data"][:]

            b_start = int(nbox_per[:step].sum())
            boxes = boxes_all[b_start : b_start + nbox_per[step]]
            n_nodes = sum(
                (b[1] - b[0] + 2) * (b[3] - b[2] + 2) * (b[5] - b[4] + 2) for b in boxes
            )
            data = data_all[pt_off[step] : pt_off[step] + n_nodes]

            ptr = 0
            for box in boxes:
                nx = box[1] - box[0] + 2
                ny = box[3] - box[2] + 2
                nz = box[5] - box[4] + 2
                n = nx * ny * nz

                xi = (box[0] + np.arange(nx)) * dl
                yj = (box[2] + np.arange(ny)) * dl
                zk = (box[4] + np.arange(nz)) * dl
                XX, YY, ZZ = np.meshgrid(xi, yj, zk, indexing="ij")

                b = data[ptr : ptr + n].reshape(nx, ny, nz, 3)

                all_pos.append(np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1))
                all_B.append(b.reshape(-1, 3))
                ptr += n

    pos = np.concatenate(all_pos).astype(np.float64)
    Bvec = np.concatenate(all_B).astype(np.float64)
    print(
        f"  Loaded {len(pos)} PHARE nodes  |B|_max={np.linalg.norm(Bvec, axis=1).max():.4f}"
    )
    return pos, Bvec


# ---------------------------------------------------------------------------
# Build radial |B|² profile (for lam)
# ---------------------------------------------------------------------------


def _radial_b2(pos, Bvec, n_bins=300):
    r = np.linalg.norm(pos - PHARE_CENTRE, axis=1)
    b2 = (Bvec**2).sum(axis=1)
    edges = np.linspace(0, r.max(), n_bins + 1)
    b2_rad = np.zeros(n_bins)
    for i in range(n_bins):
        m = (r >= edges[i]) & (r < edges[i + 1])
        if m.sum():
            b2_rad[i] = b2[m].mean()
    r_bins = 0.5 * (edges[:-1] + edges[1:])
    return r_bins, b2_rad


# ---------------------------------------------------------------------------
# DST initialisation from PHARE data
# ---------------------------------------------------------------------------


def init_phi_from_phare(lat, tree, Bvec, scale):
    """
    Set lat.phi from PHARE B field via nearest-neighbour interpolation.
    scale = R_sph_dst / R_phare  (DST cells per PHARE unit)
    """
    nb_safe = np.where(lat.nb >= 0, lat.nb, 0)

    pos_i = lat.pos[:, np.newaxis, :]  # (N, 1, 3)
    pos_j = lat.pos[nb_safe]  # (N, 14, 3)
    mid_dst = 0.5 * (pos_i + pos_j)  # (N, 14, 3) in DST cells

    # convert to PHARE physical coords
    mid_phare = PHARE_CENTRE + mid_dst / scale  # (N, 14, 3)
    shape = mid_phare.shape[:2]
    mid_flat = mid_phare.reshape(-1, 3)

    # nearest-neighbour B lookup
    _, idx = tree.query(mid_flat, workers=-1)
    B_flat = Bvec[idx]  # (N*14, 3)
    B_mid = B_flat.reshape(shape[0], shape[1], 3)

    # face unit vectors i→j
    diff = pos_j - pos_i  # (N, 14, 3)
    d_len = np.linalg.norm(diff, axis=-1, keepdims=True)
    n_hat = diff / np.where(d_len > 0, d_len, 1.0)

    phi = (B_mid * n_hat).sum(axis=-1)  # (N, 14)
    lat.phi = np.where(lat._valid, phi, 0.0)

    n_set = int(lat._valid.sum())
    print(
        f"  phi set on {n_set} faces  max={lat.phi.max():.4f}  min={lat.phi.min():.4f}"
    )
    return n_set


def init_lam_from_b2(lat, r_bins, b2_rad, scale, lam_scale=1.0, lam_floor=0.0):
    """
    Set lat.lam by interpolating the radial |B|² profile onto each cell.
    lam[i] = lam_floor + lam_scale * b2_interp(r_phare_i) / b2_peak
    """
    interp = interp1d(r_bins, b2_rad, kind="linear", bounds_error=False, fill_value=0.0)
    r_dst = np.linalg.norm(lat.pos, axis=1)
    r_phare = r_dst / scale
    b2_vals = interp(r_phare)
    b2_peak = b2_rad.max()
    lat.lam = lam_floor + lam_scale * (b2_vals / b2_peak)
    print(
        f"  lam range [{lat.lam.min():.4f}, {lat.lam.max():.4f}]"
        f"  peak at r_dst={r_dst[b2_vals.argmax()]:.1f} cells"
    )


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

R_SCAN = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 22)


def run(
    R_sph: float = 6.0,
    lam_scale: float = 5.0,
    lam_floor: float = 0.0,
    radius_cells: float = 30.0,
    run_steps: int = 20000,
    measure_every: int = 1000,
    sponge_fraction: float = 0.15,
    damp_max: float = 0.003,
    n_threads: int = 6,
    vtkhdf: Path = VTKHDF,
):
    scale = R_sph / R_PHARE

    print(
        f"\n=== DST from PHARE IC (R_sph={R_sph}, scale={scale:.1f}, lam_scale={lam_scale}) ==="
    )
    print("Loading PHARE B field …")
    pos, Bvec = _load_phare_b(vtkhdf)
    r_bins, b2_rad = _radial_b2(pos, Bvec)

    print("Building KDTree …")
    tree = KDTree(pos)

    print("Building DST lattice …")
    lat = DampedLattice(
        radius_cells=radius_cells, sponge_fraction=sponge_fraction, damp_max=damp_max
    )
    centre = np.zeros(3)
    r_all = np.linalg.norm(lat.pos, axis=1)

    print("Initialising phi from PHARE B …")
    init_phi_from_phare(lat, tree, Bvec, scale)

    print("Initialising lam from |B|² radial profile …")
    init_lam_from_b2(lat, r_bins, b2_rad, scale, lam_scale, lam_floor)

    dt = lat.stable_dt()
    print(f"  N={lat.N}  dt={dt:.6f}")
    print(
        "\n  step  |  H_total   |  u_rms_r<R  |  "
        + "  ".join(f"r={r:2d}" for r in R_SCAN)
    )
    print(f"  {'-'*90}")

    in_sph = r_all <= R_sph

    for step in range(0, run_steps + measure_every, measure_every):
        if step > 0:
            for _ in range(measure_every):
                lat.step(dt, n_threads)

        H = float(np.sum(lat.u**2) + 0.5 * np.sum(lat.phi**2))
        u_rms_core = float(np.sqrt(np.mean(lat.u[in_sph] ** 2)))
        circs = [_phi_circulation(lat, centre, r_hop=r) for r in R_SCAN]
        circ_str = "  ".join(f"{c:+.3e}" for c in circs)
        print(f"  {step:5d}  |  {H:.4e}  |  {u_rms_core:.4e}  |  {circ_str}")

    # radial u profile at end
    print("\n  Radial u profile (final):")
    for rb in range(1, int(radius_cells * 0.8), 2):
        mask = (r_all >= rb - 0.5) & (r_all < rb + 0.5)
        if mask.sum():
            um = float(lat.u[mask].mean())
            urs = float(np.sqrt(np.mean(lat.u[mask] ** 2)))
            tag = " <-- moat?" if um < -0.005 else ""
            print(f"    r={rb:3d}  u_mean={um:+.5f}  u_rms={urs:.5f}{tag}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--R", type=float, default=6.0, help="DST spheromak radius in cells"
    )
    parser.add_argument(
        "--lam-scale",
        type=float,
        default=5.0,
        help="lam peak value (lam = lam_scale * |B|²/|B|²_peak)",
    )
    parser.add_argument(
        "--lam-floor", type=float, default=0.0, help="minimum lam (vacuum background)"
    )
    parser.add_argument("--radius-cells", type=float, default=30.0)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--measure", type=int, default=1000)
    parser.add_argument("--sponge", type=float, default=0.15)
    parser.add_argument("--damp-max", type=float, default=0.003)
    parser.add_argument("--n-threads", type=int, default=6)
    parser.add_argument("--vtkhdf", default=str(VTKHDF))
    args = parser.parse_args()

    run(
        R_sph=args.R,
        lam_scale=args.lam_scale,
        lam_floor=args.lam_floor,
        radius_cells=args.radius_cells,
        run_steps=args.steps,
        measure_every=args.measure,
        sponge_fraction=args.sponge,
        damp_max=args.damp_max,
        n_threads=args.n_threads,
        vtkhdf=Path(args.vtkhdf),
    )
