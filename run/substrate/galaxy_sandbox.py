"""
DST galaxy sandbox — low resolution, fast iteration.

Each cell represents ~1-2 kpc. A "solar system" is just a chi blob at this scale.
No far field: Neumann BC (galaxy is self-contained).
No fixed_u: galactic chi distribution IS the gravitational field.

Tracks the rotation curve (mean tangential phi per radial shell) at each report
to see if DST sustains, modifies, or destroys the initial flat rotation.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, ".")

from equilibrium import lattice as _lat
from tst.substrate.adaptive import compute_dt

D1 = 1.0
L_MIN = _lat.DiscreteCell.L_MIN

# BCC face directions (matches adaptive_lattice.hpp bcc::face_dirs)
_R3I = 0.5773502691896258
FACE_DIRS = np.array(
    [
        [-_R3I, -_R3I, -_R3I],
        [-_R3I, -_R3I, +_R3I],
        [-_R3I, +_R3I, -_R3I],
        [-_R3I, +_R3I, +_R3I],
        [+_R3I, -_R3I, -_R3I],
        [+_R3I, -_R3I, +_R3I],
        [+_R3I, +_R3I, -_R3I],
        [+_R3I, +_R3I, +_R3I],
        [-1.0, 0.0, 0.0],
        [+1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, +1.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, +1.0],
    ]
)
FACE_W = np.array(
    [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="DST galaxy sandbox")
parser.add_argument("--R", type=float, default=30.0, help="domain radius (cells ≈ kpc)")
parser.add_argument(
    "--r-bulge", type=float, default=3.0, help="galactic bulge radius (cells)"
)
parser.add_argument(
    "--r-disk", type=float, default=10.0, help="disk scale radius (cells)"
)
parser.add_argument(
    "--h-disk", type=float, default=1.5, help="disk scale height (cells)"
)
parser.add_argument(
    "--lam-center",
    type=float,
    default=4.0,
    help="peak lam at galactic center (lam=4 ≈ chi=0.8)",
)
parser.add_argument(
    "--lam-disk",
    type=float,
    default=0.18,
    help="disk lam amplitude (lam=0.18 ≈ chi=0.15)",
)
parser.add_argument(
    "--v-rot", type=float, default=0.05, help="rotation speed at r=r-bulge (DST units)"
)
parser.add_argument(
    "--keplerian",
    action="store_true",
    help="v ∝ 1/√r outside bulge (solar system); default: flat (galaxy)",
)
parser.add_argument("--n-steps", type=int, default=100000)
parser.add_argument("--report", type=int, default=5000)
parser.add_argument("--threads", type=int, default=6)
parser.add_argument("--save-prefix", type=str, default="galaxy")
parser.add_argument(
    "--blobs",
    type=str,
    default="",
    help="chi blobs: 'r1:chi1,r2:chi2,...' placed at theta=0 in xy-plane",
)
parser.add_argument(
    "--blob-sigma", type=float, default=1.5, help="blob Gaussian radius (cells)"
)


def main():
    args = parser.parse_args()

    # ---------------------------------------------------------------------------
    # Chi profile — bulge (spherical) + disk (exponential sech²)
    # ---------------------------------------------------------------------------

    def galactic_lam(pos):
        r = np.linalg.norm(pos, axis=1)
        r_xy = np.linalg.norm(pos[:, :2], axis=1)
        z = np.abs(pos[:, 2])
        chi_c = args.lam_center / (1.0 + args.lam_center)
        chi_d = args.lam_disk / (1.0 + args.lam_disk)
        # Bulge: spherical exponential
        cb = chi_c * np.exp(-r / args.r_bulge)
        # Disk: radial exponential × vertical sech²
        cd = chi_d * np.exp(-r_xy / args.r_disk) / np.cosh(z / args.h_disk) ** 2
        chi = np.clip(np.maximum(cb, cd), 0.0, 1.0 - L_MIN)
        return chi / (1.0 - chi)

    def apply_galactic_lam(cl):
        pos = np.array(cl.leaf_pos)
        cl.set_leaf_lam(galactic_lam(pos))

    # Parse blobs: "r1:chi1,r2:chi2,..."
    _blobs = []
    if args.blobs:
        for tok in args.blobs.split(","):
            r_s, chi_s = tok.split(":")
            _blobs.append((float(r_s), float(chi_s)))

    def add_blobs(cl):
        if not _blobs:
            return
        pos = np.array(cl.leaf_pos)
        lam = np.array(cl.leaf_lam)
        for r_blob, chi_blob in _blobs:
            centre = np.array([r_blob, 0.0, 0.0])
            dist = np.linalg.norm(pos - centre, axis=1)
            dlam = (chi_blob / (1.0 - chi_blob)) * np.exp(
                -0.5 * (dist / args.blob_sigma) ** 2
            )
            lam = lam + dlam
        cl.set_leaf_lam(lam)

    def blob_angles(cl):
        """Return lam-centroid angle of each blob (radians)."""
        if not _blobs:
            return []
        pos = np.array(cl.leaf_pos)
        lam_act = np.array(cl.leaf_lam)
        lam_bg = galactic_lam(pos)
        chi_exc = np.maximum(lam_act - lam_bg, 0.0)  # lam-excess as weight
        angles = []
        for r_blob, chi_blob in _blobs:
            window = args.blob_sigma * 4.0
            # widen search window as blob may have drifted — use r-shell
            r_xy = np.linalg.norm(pos[:, :2], axis=1)
            mask = (np.abs(r_xy - r_blob) < r_blob * 0.5 + window) & (chi_exc > 0)
            w = chi_exc[mask]
            if w.sum() < 1e-20:
                angles.append(float("nan"))
            else:
                cx = float((w * pos[mask, 0]).sum() / w.sum())
                cy = float((w * pos[mask, 1]).sum() / w.sum())
                angles.append(float(np.degrees(np.arctan2(cy, cx))))
        return angles

    # ---------------------------------------------------------------------------
    # Rotation velocity profile — solid body inside bulge, flat outside
    # ---------------------------------------------------------------------------

    def v_rot_profile(r_xy):
        safe = np.maximum(r_xy, 1e-9)
        if args.keplerian:
            # Solid body inside bulge, Keplerian (v ∝ 1/√r) outside — solar system
            return args.v_rot * np.where(
                r_xy < args.r_bulge,
                r_xy / args.r_bulge,
                np.sqrt(args.r_bulge / safe),
            )
        else:
            # Solid body inside bulge, flat outside — galaxy
            return args.v_rot * np.where(
                r_xy < args.r_bulge,
                r_xy / args.r_bulge,
                1.0,
            )

    def apply_rotation(cl):
        """
        Set phi to encode the initial rotation curve.
        For each cell at (x,y,z), tangential direction t = (-y,x,0)/r_xy.
        phi_k += v_rot(r) * (FACE_DIRS[k] · t_hat) * amplitude
        """
        pos = np.array(cl.leaf_pos)
        r_xy = np.linalg.norm(pos[:, :2], axis=1)
        v = v_rot_profile(r_xy)

        # Unit tangential vector in xy-plane (zero for cells on z-axis)
        safe_r = np.maximum(r_xy, 1e-9)
        t_hat = np.column_stack(
            [-pos[:, 1] / safe_r, pos[:, 0] / safe_r, np.zeros(len(pos))]
        )

        phi = np.array(cl.leaf_phi)
        for k in range(14):
            proj = FACE_DIRS[k] @ t_hat.T  # (N,)
            phi[k] += v * proj
        cl.set_leaf_phi(phi)
        print(f"rotation phi      : v_rot={args.v_rot}  applied")

    # ---------------------------------------------------------------------------
    # Rotation curve diagnostic — mean tangential phi per radial shell
    # ---------------------------------------------------------------------------

    def rotation_curve(cl):
        pos = np.array(cl.leaf_pos)
        r_xy = np.linalg.norm(pos[:, :2], axis=1)
        safe = np.maximum(r_xy, 1e-9)
        t_x = -pos[:, 1] / safe
        t_y = pos[:, 0] / safe

        phi = np.array(cl.leaf_phi)
        t_phi = np.zeros(cl.n_leaves)
        for k in range(14):
            proj = FACE_DIRS[k, 0] * t_x + FACE_DIRS[k, 1] * t_y
            t_phi += FACE_W[k] * phi[k] * proj

        bins = np.arange(1.0, args.R - 1.0, 2.0)
        means = []
        for r0, r1 in zip(bins[:-1], bins[1:]):
            m = (r_xy >= r0) & (r_xy < r1)
            means.append(float(np.mean(t_phi[m])) if m.any() else float("nan"))
        return bins[:-1], means

    # ---------------------------------------------------------------------------
    # Build grid
    # ---------------------------------------------------------------------------
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    cl = alat[0]
    _lat.build_bcc_sphere(cl, args.R, d1=D1, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    alat.set_n_threads(args.threads)

    info = _lat.DSTInfo()
    info.split_hi = 0.0
    info.split_lo = 0.0
    info.max_level = 0
    cl.info = info

    apply_galactic_lam(cl)
    add_blobs(cl)
    print(
        f"galaxy            : N={cl.n_leaves}  R={args.R}  r_bulge={args.r_bulge}  r_disk={args.r_disk}"
    )
    print(f"lam               : center={args.lam_center}  disk={args.lam_disk}")

    # Neumann BC — galaxy is self-contained, no far field
    cl.set_bc(_lat.BoundaryCondition.neumann())
    print("BC                : Neumann")

    # u = 0 everywhere initially — let rotation phi drive dynamics
    # (no fixed source, no external field)

    apply_rotation(cl)

    if args.save_prefix:
        os.makedirs(".", exist_ok=True)
        alat.save(f"{args.save_prefix}_init.h5")
        print(f"saved → {args.save_prefix}_init.h5")

    # ---------------------------------------------------------------------------
    # Run
    # ---------------------------------------------------------------------------
    dt = compute_dt(cl, D1=D1)
    print(f"\n=== galaxy sandbox  N={cl.n_leaves}  dt={dt:.3e} ===")

    # Print initial rotation curve
    r_bins, v0 = rotation_curve(cl)
    header = "  ".join(f"r={r:.0f}" for r in r_bins)
    vals0 = "  ".join(f"{v:7.4f}" for v in v0)
    print("\nInitial rotation curve:")
    print(f"  r   : {header}")
    print(f"  phi : {vals0}")
    blob_hdr = "  ".join(f"blob_r{r:.0f}°" for r, _ in _blobs)
    print(
        f"\n{'step':>8}  {'v_r1':>8}  {'v_r5':>8}  {'v_r10':>8}  {'v_r15':>8}  {'L_tot':>12}  {blob_hdr}"
    )

    L0 = None

    for s in range(1, args.n_steps + 1):
        alat.step(0, dt)

        if s % args.report == 0:
            r_bins, v_curve = rotation_curve(cl)

            # Sample at a few representative radii
            def v_at(r_target):
                idx = int(np.argmin(np.abs(r_bins - r_target)))
                return v_curve[idx] if idx < len(v_curve) else float("nan")

            v1, v5, v10, v15 = v_at(1), v_at(5), v_at(10), v_at(15)

            # Total angular momentum proxy: sum of all tangential phi
            phi_arr = np.array(cl.leaf_phi)
            pos_arr = np.array(cl.leaf_pos)
            r_xy = np.linalg.norm(pos_arr[:, :2], axis=1)
            safe = np.maximum(r_xy, 1e-9)
            t_x = -pos_arr[:, 1] / safe
            t_y = pos_arr[:, 0] / safe
            L_tot = 0.0
            for k in range(14):
                proj = FACE_DIRS[k, 0] * t_x + FACE_DIRS[k, 1] * t_y
                L_tot += float(np.sum(FACE_W[k] * phi_arr[k] * proj))

            if L0 is None and abs(L_tot) > 1e-12:
                L0 = abs(L_tot)
            L_ratio = abs(L_tot) / L0 if L0 else float("nan")

            angles = blob_angles(cl)
            angle_s = "  ".join(
                f"{a:8.2f}" if np.isfinite(a) else "     nan" for a in angles
            )
            print(
                f"{s:8d}  {v1:8.4f}  {v5:8.4f}  {v10:8.4f}  {v15:8.4f}  {L_tot:12.4f}  L/L0={L_ratio:.3f}  {angle_s}",
                flush=True,
            )

            if L0 and L_ratio < 0.1:
                print("*** early exit: angular momentum collapsed ***", flush=True)
                break

    if args.save_prefix:
        alat.save(f"{args.save_prefix}_final.h5")
        print(f"saved → {args.save_prefix}_final.h5")

    print("\ndone")


if __name__ == "__main__":
    main()
