"""
DST Harris sheet — PHARE-matched IC for validation.

A double Harris sheet in a periodic box (Lz=3.5 d1, z-thin).  Two current
sheets at y = ±Ly/4 with opposing Bx(y) profiles.  The magnetic field is
carried in phi directly via a divergence-free perturbation projected onto all
14 BCC face directions.

Coordinate conventions:
  x : magnetic direction   (phi x-faces carry the B_x perturbation)
  y : reversal direction   (tanh profile, null lines at y = ±Ly/4)
  z : out-of-plane         (3 layers, periodic, ∂/∂z = 0)

BCC face ordering (bcc_geometry.hpp):
  Hex faces 0-7 : (±1,±1,±1) in lex order
  Sq  faces 8-13: -x +x -y +y -z +z

Perturbation
------------
Az = dB * cos(2π*(x+Lx/2)/Lx) * exp(-y_r²/lam²)
Bx_pert = ∂Az/∂y → projected onto phi x-faces
By_pert = -∂Az/∂x → projected onto phi y-faces

Full-wavelength (kx = 2π/Lx): By changes sign at x=0 (inflow X-point),
peaks at x=±Lx/4 (O-points). Gives 4 blobs of alternating polarity (2/sheet).
The outflow jets from x=0 drive By toward x=±Lx/4 (O-points) then x=±Lx/2.

Usage:
  PYTHONPATH=$PWD python3 tst/substrate/harris_sheet_phare.py [--Lx N --Ly N ...]
  python3 vis/substrate/harris_sheet.py harris_phare_init.h5 harris_phare_final.h5 --save harris_plot.png
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import KDTree

sys.path.insert(0, ".")
from tst.substrate.adaptive.test_lam_phi_drive import (
    step_lam_relax,
)

from equilibrium import lattice as _lat

# ---------------------------------------------------------------------------
# BCC geometry constants (mirror bcc_geometry.hpp)
# ---------------------------------------------------------------------------
HEX_DIR = np.array(
    [
        [-1, -1, -1],
        [-1, -1, +1],
        [-1, +1, -1],
        [-1, +1, +1],
        [+1, -1, -1],
        [+1, -1, +1],
        [+1, +1, -1],
        [+1, +1, +1],
    ],
    dtype=np.float64,
)
SQ_DIR = np.array(
    [[-1, 0, 0], [+1, 0, 0], [0, -1, 0], [0, +1, 0], [0, 0, -1], [0, 0, +1]],
    dtype=np.float64,
)

# x-component of each face direction (len 14)
FACE_DX = np.concatenate([HEX_DIR[:, 0], SQ_DIR[:, 0]])
# z-component of each face direction
FACE_DZ = np.concatenate([HEX_DIR[:, 2], SQ_DIR[:, 2]])

FACE_DY = np.concatenate([HEX_DIR[:, 1], SQ_DIR[:, 1]])

Z_PLUS_FACES = [k for k in range(14) if FACE_DZ[k] > 0]  # [1,3,5,7,13]
Z_MINUS_FACES = [k for k in range(14) if FACE_DZ[k] < 0]  # [0,2,4,6,12]
X_PLUS_FACES = [k for k in range(14) if FACE_DX[k] > 0]  # [4,5,6,7,9]
X_MINUS_FACES = [k for k in range(14) if FACE_DX[k] < 0]  # [0,1,2,3,8]
Y_PLUS_FACES = [k for k in range(14) if FACE_DY[k] > 0]  # [2,3,6,7,11]
Y_MINUS_FACES = [k for k in range(14) if FACE_DY[k] < 0]  # [0,1,4,5,10]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="DST Harris sheet reconnection")
parser.add_argument("--Lx", type=float, default=100.0, help="box x width (d1 units)")
parser.add_argument("--Ly", type=float, default=200.0, help="box y width (d1 units)")
parser.add_argument("--Lz", type=float, default=3.5, help="box z thickness (d1 units)")
parser.add_argument("--B0", type=float, default=0.05, help="Harris sheet amplitude")
parser.add_argument("--lam", type=float, default=6.0, help="sheet half-width λ (d1)")
parser.add_argument(
    "--u-bg", type=float, default=0.4, help="background u (pressure floor)"
)
parser.add_argument(
    "--lam-bg",
    type=float,
    default=0.0,
    help="uniform background lam (vacuum plasma density)",
)
parser.add_argument(
    "--lam-peak",
    type=float,
    default=0.0,
    help="sech² lam amplitude at each null line — models Harris density peak "
    "(PHARE uses lam_peak/lam_bg ≈ 2.5; 0 = flat lam everywhere)",
)
parser.add_argument(
    "--u-peak",
    type=float,
    default=None,
    help="u peak at current sheets (default: B0² — pressure balance for Harris sheet)",
)
parser.add_argument(
    "--flat-u",
    action="store_true",
    help="initialise u=u_bg everywhere (no sech² pressure bump); avoids phi energy runaway",
)
parser.add_argument(
    "--perturb",
    type=float,
    default=0.1,
    help="perturbation amplitude as fraction of B0 (PHARE uses 0.1)",
)
parser.add_argument("--n-steps", type=int, default=20000)
parser.add_argument("--report", type=int, default=2000)
parser.add_argument("--dt", type=float, default=0.05, help="timestep")
parser.add_argument(
    "--blowup-factor",
    type=float,
    default=50.0,
    help="abort if total phi energy (sum phi^2) grows by this factor vs step 0 (0 = disabled)",
)
parser.add_argument("--save-prefix", type=str, default="harris_phare")
parser.add_argument(
    "--plot-every",
    type=int,
    default=0,
    help="save a PNG snapshot every N steps (0 = disabled)",
)
parser.add_argument(
    "--plot-dir", type=str, default=".", help="output directory for per-step PNG plots"
)
parser.add_argument(
    "--diag-out",
    type=str,
    default=None,
    help="CSV file to write per-report-step diagnostics",
)
parser.add_argument(
    "--restart",
    type=str,
    default=None,
    help="HDF5 snapshot to restart from (loads u and phi; "
    "Lx/Ly/Lz/B0/lam are read from the file)",
)
parser.add_argument(
    "--lam-relax-rate",
    type=float,
    default=0.0,
    help="local lam relaxation rate toward alpha*|phi|  (0 = disabled)",
)
parser.add_argument(
    "--lam-alpha",
    type=float,
    default=1.0,
    help="phi→lam coupling: lam_target = alpha * |phi|",
)
parser.add_argument("--threads", type=int, default=10)
parser.add_argument("--no-amr", action="store_true", help="disable adaptive refinement")
parser.add_argument(
    "--amr-interval",
    type=int,
    default=500,
    help="steps between AMR adapt passes (default 500)",
)
parser.add_argument(
    "--gradient-amr",
    action="store_true",
    help="use gradient split criterion (cmax-cmin) instead of absolute phi — refines at field "
    "reversals (null line) rather than uniform-field bulk",
)
parser.add_argument("--split-hi", type=float, default=0.0)
parser.add_argument("--split-lo", type=float, default=0.0)
parser.add_argument("--max-level", type=int, default=0)
parser.add_argument(
    "--live",
    action="store_true",
    help="open a live matplotlib view (updated every --report steps)",
)

_POS_X = [4, 5, 6, 7, 9]
_NEG_X = [0, 1, 2, 3, 8]
_POS_Y = [2, 3, 6, 7, 11]
_NEG_Y = [0, 1, 4, 5, 10]


def _snapshot_plot(pos, phi, u, lam, step, B0, out_path):
    z = pos[:, 2]
    zmid = (z.max() + z.min()) / 2.0
    m = np.abs(z - zmid) < (z.max() - z.min()) / 6.0
    p2, phi2, u2, lam2 = pos[m], phi[:, m], u[m], lam[m]
    x, y = p2[:, 0], p2[:, 1]
    bx = phi2[_POS_X].mean(0) - phi2[_NEG_X].mean(0)
    by = phi2[_POS_Y].mean(0) - phi2[_NEG_Y].mean(0)
    phi_mag = np.sqrt(np.sum(phi2**2, axis=0))  # total phi magnitude per cell
    u_fluct = u2 - u2.mean()  # fluctuation around mean
    nr = 200
    xi = np.linspace(x.min(), x.max(), nr)
    yi = np.linspace(y.min(), y.max(), nr)
    Xg, Yg = np.meshgrid(xi, yi)
    pts = (x, y)

    def _g(v):
        return griddata(pts, v, (Xg, Yg), method="linear")

    Bx, By, Ug, Uf, Lg, Pm = _g(bx), _g(by), _g(u2), _g(u_fluct), _g(lam2), _g(phi_mag)
    dx = xi[1] - xi[0]
    dy = yi[1] - yi[0]
    Jz = np.gradient(np.nan_to_num(By), dx, axis=1) - np.gradient(
        np.nan_to_num(Bx), dy, axis=0
    )

    has_lam = lam2.max() > 0
    ncols = 7 if has_lam else 6
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), constrained_layout=True)
    fig.suptitle(f"step {step}    u_mean={u2.mean():.4f}", fontsize=10)

    def _im_div(ax, data, title, cmap="RdBu_r", vlim=None):
        vmax = (
            vlim
            if vlim is not None
            else np.nanpercentile(np.abs(np.nan_to_num(data)), 98)
        )
        vmax = max(vmax, 1e-9)
        im = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            extent=[xi[0], xi[-1], yi[0], yi[-1]],
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.8)

    def _im_pos(ax, data, title, cmap="inferno"):
        vmax = max(np.nanpercentile(np.nan_to_num(data), 98), 1e-9)
        im = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            extent=[xi[0], xi[-1], yi[0], yi[-1]],
            cmap=cmap,
            vmin=0,
            vmax=vmax,
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.8)

    _im_div(axes[0], Bx, "Bx", vlim=B0)
    _im_div(axes[1], By, "By")
    _im_div(axes[2], Jz, "Jz")
    _im_pos(axes[3], Pm, "phi_mag")
    _im_div(axes[4], Uf, "u − mean(u)")
    _im_pos(axes[5], Ug, "u", cmap="inferno")
    if has_lam:
        _im_pos(axes[6], Lg, "lam", cmap="viridis")

    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    args = parser.parse_args()

    # Pressure balance: n0 = B0²/(2μ0*T); with μ0=1, T=0.5 → n0 = B0²
    if args.u_peak is None:
        args.u_peak = args.B0**2

    D1 = 1.0
    H = D1 / np.sqrt(3.0)  # h = d1/sqrt(3), the BCC lattice step

    # ---------------------------------------------------------------------------
    # Restart: override Lx/Ly/Lz/B0/lam from the snapshot file if given.
    # ---------------------------------------------------------------------------
    step_offset = 0
    if args.restart:
        import h5py as _h5

        with _h5.File(args.restart, "r") as _f:
            for key in ("Lx", "Ly", "Lz", "B0", "lam"):
                if key in _f.attrs:
                    setattr(args, key, float(_f.attrs[key]))
            step_offset = int(_f.attrs.get("step", 0))
        print(f"Restart from {args.restart}  (step_offset={step_offset})")
        print(
            f"  Lx={args.Lx}  Ly={args.Ly}  Lz={args.Lz}  B0={args.B0}  lam={args.lam}"
        )

    # ---------------------------------------------------------------------------
    # Build box
    # ---------------------------------------------------------------------------
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    cl = alat[0]
    _lat.build_bcc_box(cl, args.Lx, args.Ly, args.Lz, D1, u0=0.0, lam0=args.lam_bg)
    alat.rebuild(0)
    alat.set_n_threads(args.threads)

    info = _lat.DSTInfo()
    info.split_hi = args.split_hi
    info.split_lo = args.split_lo
    info.max_level = args.max_level
    cl.info = info
    # gradient_amr: criterion override is applied in the step loop (see below)

    N = cl.n_leaves
    pos = cl.leaf_pos  # (N, 3)
    nb = cl.leaf_nb  # (N, 14)  global g of each neighbour, -1 = none
    gid = cl.leaf_global_indices  # (N,)     global g for each leaf j

    print(f"Box Lx={args.Lx} Ly={args.Ly} Lz={args.Lz}  N={N} cells")
    print(f"Harris: B0={args.B0}  λ={args.lam}  perturb={args.perturb}")

    # ---------------------------------------------------------------------------
    # Wire Z-periodic boundary conditions via face_ghost
    #
    # Top-layer z+ faces → corresponding bottom cell (same xy, opposite z).
    # Bottom-layer z- faces → corresponding top cell.
    # We use KDTree on xy to handle the BCC parity mismatch gracefully.
    # ---------------------------------------------------------------------------
    z_max = pos[:, 2].max()
    z_min = pos[:, 2].min()

    top_mask = pos[:, 2] > z_max - H * 0.6
    bot_mask = pos[:, 2] < z_min + H * 0.6

    top_tree = KDTree(pos[top_mask, :2])
    bot_tree = KDTree(pos[bot_mask, :2])
    top_gid = gid[top_mask]
    bot_gid = gid[bot_mask]

    def wire_z_periodic(cell_mask, face_list, partner_tree, partner_gid):
        wired = 0
        for j in np.where(cell_mask)[0]:
            g_j = int(gid[j])
            x_j, y_j = pos[j, 0], pos[j, 1]
            for k in face_list:
                if nb[j, k] >= 0:
                    continue  # has a real neighbour — not a boundary face
                if k < 8:
                    dx, dy = HEX_DIR[k, 0], HEX_DIR[k, 1]
                else:
                    dx, dy = SQ_DIR[k - 8, 0] * 2.0, SQ_DIR[k - 8, 1] * 2.0
                tgt_xy = [x_j + dx * H, y_j + dy * H]
                _, idx = partner_tree.query(tgt_xy)
                cl.wire_face(g_j, k, int(partner_gid[idx]))
                wired += 1
        return wired

    n_top = wire_z_periodic(top_mask, Z_PLUS_FACES, bot_tree, bot_gid)
    n_bot = wire_z_periodic(bot_mask, Z_MINUS_FACES, top_tree, top_gid)
    print(f"Z-periodic wiring: {n_top} top faces, {n_bot} bottom faces wired")

    # ---------------------------------------------------------------------------
    # Wire X-periodic boundary conditions (same KDTree approach, YZ plane)
    # ---------------------------------------------------------------------------
    x_max = pos[:, 0].max()
    x_min = pos[:, 0].min()

    rgt_mask = pos[:, 0] > x_max - H * 0.6
    lft_mask = pos[:, 0] < x_min + H * 0.6

    rgt_tree = KDTree(pos[rgt_mask, 1:])  # YZ coords
    lft_tree = KDTree(pos[lft_mask, 1:])
    rgt_gid = gid[rgt_mask]
    lft_gid = gid[lft_mask]

    def wire_x_periodic(cell_mask, face_list, partner_tree, partner_gid):
        wired = 0
        for j in np.where(cell_mask)[0]:
            g_j = int(gid[j])
            y_j, z_j = pos[j, 1], pos[j, 2]
            for k in face_list:
                if nb[j, k] >= 0:
                    continue
                if k < 8:
                    dy, dz = HEX_DIR[k, 1], HEX_DIR[k, 2]
                else:
                    dy, dz = SQ_DIR[k - 8, 1] * 2.0, SQ_DIR[k - 8, 2] * 2.0
                tgt_yz = [y_j + dy * H, z_j + dz * H]
                _, idx = partner_tree.query(tgt_yz)
                cl.wire_face(g_j, k, int(partner_gid[idx]))
                wired += 1
        return wired

    n_rgt = wire_x_periodic(rgt_mask, X_PLUS_FACES, lft_tree, lft_gid)
    n_lft = wire_x_periodic(lft_mask, X_MINUS_FACES, rgt_tree, rgt_gid)
    print(f"X-periodic wiring: {n_rgt} right faces, {n_lft} left faces wired")

    # ---------------------------------------------------------------------------
    # Wire Y-periodic boundary conditions (XZ plane)
    # The double-sheet Bx profile has period Ly, so Y must also be periodic.
    # ---------------------------------------------------------------------------
    y_max = pos[:, 1].max()
    y_min = pos[:, 1].min()

    top_y_mask = pos[:, 1] > y_max - H * 0.6
    bot_y_mask = pos[:, 1] < y_min + H * 0.6

    top_y_tree = KDTree(pos[top_y_mask, 0::2])  # XZ coords
    bot_y_tree = KDTree(pos[bot_y_mask, 0::2])
    top_y_gid = gid[top_y_mask]
    bot_y_gid = gid[bot_y_mask]

    def wire_y_periodic(cell_mask, face_list, partner_tree, partner_gid):
        wired = 0
        for j in np.where(cell_mask)[0]:
            g_j = int(gid[j])
            x_j, z_j = pos[j, 0], pos[j, 2]
            for k in face_list:
                if nb[j, k] >= 0:
                    continue
                if k < 8:
                    dx, dz = HEX_DIR[k, 0], HEX_DIR[k, 2]
                else:
                    dx, dz = SQ_DIR[k - 8, 0] * 2.0, SQ_DIR[k - 8, 2] * 2.0
                tgt_xz = [x_j + dx * H, z_j + dz * H]
                _, idx = partner_tree.query(tgt_xz)
                cl.wire_face(g_j, k, int(partner_gid[idx]))
                wired += 1
        return wired

    n_top_y = wire_y_periodic(top_y_mask, Y_PLUS_FACES, bot_y_tree, bot_y_gid)
    n_bot_y = wire_y_periodic(bot_y_mask, Y_MINUS_FACES, top_y_tree, top_y_gid)
    print(f"Y-periodic wiring: {n_top_y} top faces, {n_bot_y} bottom faces wired")

    # ---------------------------------------------------------------------------
    # Double Harris sheet phi initialisation  (matches PHARE harris_2d layout)
    #
    # Two current sheets at y = ±y0 (y0 = Ly/4).
    # Bx profile:  B0 * [-1 + tanh((y+y0)/λ) - tanh((y-y0)/λ)]
    #   → −B0 outside both sheets, +B0 between them, two null lines at ±y0.
    #
    # phi[i,k] = Bx(y_mid) * face_dx[k],  y_mid = (y_i + y_nb)/2.
    # ---------------------------------------------------------------------------
    y0 = args.Ly / 4.0

    def bx_profile(y):
        return args.B0 * (
            -1.0 + np.tanh((y + y0) / args.lam) - np.tanh((y - y0) / args.lam)
        )

    g_to_j: dict = {int(gid[j]): j for j in range(N)}
    phi_x_faces = [0, 2, 4, 6, 8]  # faces with dx ≤ 0

    if args.restart:
        # Load u and phi directly; skip fresh initialisation.
        import h5py as _h5

        with _h5.File(args.restart, "r") as _f:
            _u_snap = _f["u"][:]
            _phi_snap = _f["phi"][:]
        if len(_u_snap) != N:
            raise RuntimeError(
                f"Restart snapshot has {len(_u_snap)} cells but lattice has {N}. "
                "Check Lx/Ly/Lz match."
            )
        cl.set_leaf_u(_u_snap)
        cl.set_leaf_phi(_phi_snap)
        print(f"Loaded u and phi from {args.restart}")
        phi_x_all = _phi_snap[phi_x_faces, :]
    else:
        # u: flat or pressure-balanced
        y = pos[:, 1]
        if args.flat_u:
            u_init = np.full(N, args.u_bg)
        else:
            # sech² peaks at the two null lines (matches PHARE density)
            u_init = (
                args.u_bg
                + args.u_peak / np.cosh((y + y0) / args.lam) ** 2
                + args.u_peak / np.cosh((y - y0) / args.lam) ** 2
            )
        cl.set_leaf_u(u_init)

        # lam: flat background + optional sech² peak at each null line (Harris density profile).
        if args.lam_peak > 0.0:
            lam_init = (
                args.lam_bg
                + args.lam_peak / np.cosh((y + y0) / args.lam) ** 2
                + args.lam_peak / np.cosh((y - y0) / args.lam) ** 2
            )
            cl.set_leaf_lam(lam_init)

        # phi: equilibrium Bx (tanh profile) + reconnection perturbation.
        phi_init = np.zeros((14, N), dtype=np.float64)
        bx_eq = bx_profile(pos[:, 1])
        for k in range(14):
            fd = HEX_DIR[k] if k < 8 else SQ_DIR[k - 8]
            phi_init[k, :] += bx_eq * fd[0]
        if args.perturb > 0.0:
            sigma_p = args.lam
            dB = args.perturb * args.B0
            # Full-wavelength mode (kx = 2π/Lx):
            #   X-point at x=0 (centre, where By=0 and sign changes).
            #   O-points at x=±Lx/4 (By peaks, opposite polarity each side).
            # Mapping: kx_x = kx*(x + Lx/2) sends x∈[-Lx/2,+Lx/2] → [0, 2π].
            #   sin(kx_x) = 0 at x=0 (centre) and x=±Lx/2 (boundaries).
            #   sin(kx_x) = +1 at x=-Lx/4,  -1 at x=+Lx/4.
            # This gives 4 blobs of alternating polarity (2 per sheet).
            kx = 2.0 * np.pi / args.Lx
            for sign, y_null in ((+1, +y0), (-1, -y0)):
                y_r = pos[:, 1] - y_null
                env = np.exp(-(y_r**2) / sigma_p**2)
                kx_x = kx * (pos[:, 0] + args.Lx / 2.0)
                bx_p = sign * (-2.0 * y_r / sigma_p**2) * dB * np.cos(kx_x) * env
                by_p = sign * kx * dB * np.sin(kx_x) * env
                for k in range(14):
                    fd = HEX_DIR[k] if k < 8 else SQ_DIR[k - 8]
                    phi_init[k, :] += bx_p * fd[0] + by_p * fd[1]
        cl.set_leaf_phi(phi_init)
        phi_x_all = phi_init[phi_x_faces, :]

    print(
        f"phi_x (equilibrium + perturbation) range: [{phi_x_all.min():.4f}, {phi_x_all.max():.4f}]"
    )
    _lam_now = cl.leaf_lam
    print(
        f"lam range: [{_lam_now.min():.3f}, {_lam_now.max():.3f}]  mean={_lam_now.mean():.3f}"
    )

    # ---------------------------------------------------------------------------
    # dt — by_level TimeStepper: each refined cell uses dt/2^level so CFL holds
    # at all refinement levels without penalising coarse cells.
    # ---------------------------------------------------------------------------
    dt = args.dt * D1
    ts = (
        _lat.TimeStepper.by_level(dt)
        if args.max_level > 0
        else _lat.TimeStepper.fixed(dt)
    )

    # ---------------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------------
    def save_state(fname, step):
        with h5py.File(fname, "w") as f:
            f["pos"] = cl.leaf_pos
            f["u"] = cl.leaf_u
            f["phi"] = cl.leaf_phi
            f.attrs["step"] = step
            f.attrs["B0"] = args.B0
            f.attrs["lam"] = args.lam
            f.attrs["Lx"] = args.Lx
            f.attrs["Ly"] = args.Ly
            f.attrs["Lz"] = args.Lz
            f.attrs["perturb"] = args.perturb
        print(f"  saved {fname}")

    if not args.restart:
        save_state(f"{args.save_prefix}_init.h5", step_offset)

    # ---------------------------------------------------------------------------
    # Run
    # ---------------------------------------------------------------------------
    X_FACE_IDX = np.array(
        [k for k in range(14) if abs(FACE_DX[k]) > 0]
    )  # x-carrying faces

    evolver = _lat.AdaptiveLatticeEvolver(alat)

    # Initial AMR pass before the loop — phi_energy_0 must be measured post-refine
    # so the blowup check compares apples to apples (cell-count artifact cancelled).
    if not args.no_amr:
        if args.gradient_amr:
            _phi_xf = cl.leaf_phi[X_FACE_IDX, :]
            crit = _phi_xf.max(axis=0) - _phi_xf.min(axis=0)
            _p = cl.leaf_pos
            near_null = (np.abs(_p[:, 1] - y0) < 3.0 * args.lam) | (
                np.abs(_p[:, 1] + y0) < 3.0 * args.lam
            )
            crit[~near_null] = 0.0
            cl.set_criterion_max(crit)
        evolver.adapt()
        evolver.rebuild()
        pos = cl.leaf_pos
        nb = cl.leaf_nb
        gid = cl.leaf_global_indices
        g_to_j = {int(gid[j]): j for j in range(cl.n_leaves)}
    phi_energy_0 = float(np.sum(cl.leaf_phi**2))  # reference for blowup check
    _lev0 = np.array(cl.leaf_level, dtype=np.float64)
    _vol0 = (D1 * 0.5**_lev0) ** 3
    u_vol_0 = float(
        np.sum(cl.leaf_u * _vol0)
    )  # conserved: sum(u * L³) with periodic BCs
    print(
        f"\nInitial AMR: N={cl.n_leaves} cells  phi_E0={phi_energy_0:.4e}  u_vol_0={u_vol_0:.6e}"
    )
    print(f"Running {args.n_steps} steps from step {step_offset}, dt={dt:.3e}")

    # Precompute fixed geometry masks for diagnostics (null lines and X-point).
    # These are based on initial pos — valid when no_amr=True; approximate otherwise.
    _pos0 = cl.leaf_pos
    _null_mask = (np.abs(_pos0[:, 1] - y0) < args.lam) | (
        np.abs(_pos0[:, 1] + y0) < args.lam
    )
    # X-point at x=0 (centre) with full-wavelength perturbation (kx=2π/Lx).
    _xpt_mask = _null_mask & (np.abs(_pos0[:, 0]) < args.Lx * 0.1)

    _diag_f = None
    if args.diag_out:
        Path(args.diag_out).parent.mkdir(parents=True, exist_ok=True)
        _diag_f = open(args.diag_out, "w")
        _diag_f.write("step,phi_E,phi_E_ratio,phi_null_rms,phi_xpt_by,u_rms,lam_std\n")

    # Live view (optional — requires a display)
    _live = None
    if args.live:
        from vis.substrate.harris_sheet_live import HarrisLive

        _live = HarrisLive(cl, y0, args.B0, every=args.report)

    for step in range(args.n_steps):
        evolver.step(ts)
        if args.lam_relax_rate > 0.0:
            step_lam_relax(cl, dt, args.lam_relax_rate, args.lam_alpha)

        if (not args.no_amr) and step % args.amr_interval == args.amr_interval - 1:
            if args.gradient_amr:
                _phi_xf = cl.leaf_phi[X_FACE_IDX, :]
                crit = _phi_xf.max(axis=0) - _phi_xf.min(axis=0)
                _p = cl.leaf_pos
                near_null = (np.abs(_p[:, 1] - y0) < 3.0 * args.lam) | (
                    np.abs(_p[:, 1] + y0) < 3.0 * args.lam
                )
                crit[~near_null] = 0.0
                cl.set_criterion_max(crit)
            evolver.adapt()
            evolver.rebuild()
            pos = cl.leaf_pos
            nb = cl.leaf_nb
            gid = cl.leaf_global_indices
            g_to_j = {int(gid[j]): j for j in range(cl.n_leaves)}

        if _live is not None:
            _live.update(cl, step_offset + step)

        if args.plot_every > 0 and step % args.plot_every == 0:
            Path(args.plot_dir).mkdir(parents=True, exist_ok=True)
            _snapshot_plot(
                cl.leaf_pos,
                cl.leaf_phi,
                cl.leaf_u,
                cl.leaf_lam,
                step_offset + step,
                args.B0,
                os.path.join(args.plot_dir, f"step_{step_offset+step:06d}.png"),
            )

        if step % args.report == 0:
            u_now = cl.leaf_u
            phi_now = cl.leaf_phi
            lam_now = cl.leaf_lam
            u_max = float(u_now.max())
            phi_energy = float(np.sum(phi_now**2))
            _lev = np.array(cl.leaf_level, dtype=np.float64)
            _vol = (D1 * 0.5**_lev) ** 3
            u_vol = float(np.sum(u_now * _vol))
            print(
                f"  step {step_offset+step:6d}  u_max={u_max:.4e}  "
                f"phi_E={phi_energy:.4e} (×{phi_energy/max(phi_energy_0,1e-30):.2f})"
                f"  N={cl.n_leaves}"
                f"\n             u_vol={u_vol:.6e} (Δ={u_vol-u_vol_0:+.2e})"
            )

            if _diag_f is not None:
                # Remap masks to current leaf count (safe when no_amr; approximate otherwise)
                _nm = (
                    _null_mask[: cl.n_leaves]
                    if cl.n_leaves <= len(_null_mask)
                    else _null_mask
                )
                _xm = (
                    _xpt_mask[: cl.n_leaves]
                    if cl.n_leaves <= len(_xpt_mask)
                    else _xpt_mask
                )
                phi_null_rms = (
                    float(np.sqrt(np.mean(phi_now[:, _nm] ** 2))) if _nm.any() else 0.0
                )
                phi_xpt_by = (
                    float(
                        np.mean(
                            np.abs(
                                phi_now[_POS_Y][:, _xm].mean(0)
                                - phi_now[_NEG_Y][:, _xm].mean(0)
                            )
                        )
                    )
                    if _xm.any()
                    else 0.0
                )
                u_rms = float(np.sqrt(np.mean(u_now**2)))
                lam_std = float(np.std(lam_now))
                phi_E_ratio = phi_energy / max(phi_energy_0, 1e-30)
                _diag_f.write(
                    f"{step_offset+step},{phi_energy:.6e},{phi_E_ratio:.6f},"
                    f"{phi_null_rms:.6e},{phi_xpt_by:.6e},{u_rms:.6e},{lam_std:.6e}\n"
                )
                _diag_f.flush()

            if np.any(~np.isfinite(u_now)) or np.any(~np.isfinite(phi_now)):
                print("ABORT: NaN/Inf detected — saving state and stopping.")
                save_state(f"{args.save_prefix}_failed.h5", step_offset + step)
                sys.exit(1)

            if (
                args.blowup_factor > 0
                and phi_energy_0 > 0
                and phi_energy > args.blowup_factor * phi_energy_0
            ):
                print(
                    f"ABORT: phi_energy={phi_energy:.4e} exceeds {args.blowup_factor}×E0="
                    f"{args.blowup_factor*phi_energy_0:.4e} — saving state and stopping."
                )
                save_state(f"{args.save_prefix}_failed.h5", step_offset + step)
                sys.exit(1)

    if _diag_f is not None:
        _diag_f.close()
    save_state(f"{args.save_prefix}_final.h5", step_offset + args.n_steps)
    print("Done.")


if __name__ == "__main__":
    main()
