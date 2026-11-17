"""
DST Harris sheet — flux-tube-1 u initialisation (island geometry).

u is high at the O-points (x = ±Lx/2) and low at the X-point (x = 0):

    y_r  = y ∓ y0   (distance from each null line)
    u(x,y) = u_bg + u_peak * sech²(y_r/λ) * (1 - cos(2π*x/Lx)) / 2

(1−cos(2πx/Lx))/2 ∈ [0,1]: zero at X-point (x=0), maximum at O-points (x=±Lx/2).

This mirrors the magnetic island topology: u is a proxy for pressure,
which is depleted in the reconnection exhaust (X-point) and enhanced
in the closed-field-line O-points.
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from equilibrium.substrate.adaptive import wire_periodic_bcc

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

FACE_DX = np.concatenate([HEX_DIR[:, 0], SQ_DIR[:, 0]])

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="DST Harris sheet — flux-tube-1 (island) u IC"
)
parser.add_argument("--Lx", type=float, default=200.0)
parser.add_argument("--Ly", type=float, default=100.0)
parser.add_argument("--Lz", type=float, default=3.5)
parser.add_argument("--B0", type=float, default=0.05)
parser.add_argument("--lam", type=float, default=2.0, help="sheet half-width λ (d1)")
parser.add_argument(
    "--u-bg", type=float, default=0.4, help="background u (pressure floor)"
)
parser.add_argument("--u-peak", type=float, default=1.0, help="peak u at O-points")
parser.add_argument(
    "--perturb",
    type=float,
    default=0.1,
    help="perturbation amplitude as fraction of B0",
)
parser.add_argument("--n-steps", type=int, default=10000)
parser.add_argument("--report", type=int, default=500)
parser.add_argument("--dt", type=float, default=0.003)
parser.add_argument(
    "--blowup-factor",
    type=float,
    default=0.0,
    help="abort if phi_energy grows by this factor (0 = disabled)",
)
parser.add_argument("--save-prefix", type=str, default="harris_ft1")
parser.add_argument("--threads", type=int, default=10)
parser.add_argument("--no-amr", action="store_true")
parser.add_argument("--amr-interval", type=int, default=500)
parser.add_argument("--gradient-amr", action="store_true")
parser.add_argument("--split-hi", type=float, default=0.0)
parser.add_argument("--split-lo", type=float, default=0.0)
parser.add_argument("--max-level", type=int, default=0)
parser.add_argument("--live", action="store_true")


def main():
    args = parser.parse_args()

    D1 = 1.0
    H = D1 / np.sqrt(3.0)

    step_offset = 0

    # ---------------------------------------------------------------------------
    # Build box
    # ---------------------------------------------------------------------------
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    cl = alat[0]
    _lat.build_bcc_box(cl, args.Lx, args.Ly, args.Lz, D1, u0=0.0, lam0=0)
    alat.rebuild(0)
    alat.set_n_threads(args.threads)

    info = _lat.DSTInfo()
    info.split_hi = args.split_hi
    info.split_lo = args.split_lo
    info.max_level = args.max_level
    cl.info = info

    N = cl.n_leaves
    pos = cl.leaf_pos

    print(f"Box Lx={args.Lx} Ly={args.Ly} Lz={args.Lz}  N={N} cells")
    print(f"Harris: B0={args.B0}  λ={args.lam}  perturb={args.perturb}")
    print(f"Flux tube 1 (island): u_peak={args.u_peak}  λ={args.lam}")

    wire_periodic_bcc(cl, H, verbose=True)

    # ---------------------------------------------------------------------------
    # Harris sheet profile
    # ---------------------------------------------------------------------------
    y0 = args.Ly / 4.0

    phi_x_faces = [0, 2, 4, 6, 8]

    # ---------------------------------------------------------------------------
    # u initialisation — island geometry
    # u is high at O-points (x=±Lx/2) and zero at X-point (x=0)
    # ---------------------------------------------------------------------------
    xc = pos[:, 0]
    yc = pos[:, 1]
    kx_u = 2.0 * np.pi / args.Lx
    island_x = (1.0 - np.cos(kx_u * xc)) / 2.0  # 0 at x=0, 1 at x=±Lx/2
    u_init = (
        args.u_bg
        + args.u_peak / np.cosh((yc - y0) / args.lam) ** 2 * island_x
        + args.u_peak / np.cosh((yc + y0) / args.lam) ** 2 * island_x
    )
    cl.set_leaf_u(u_init)
    print(f"u: min={u_init.min():.4f}  max={u_init.max():.4f}")

    # ---------------------------------------------------------------------------
    # phi perturbation (divergence-free, seeds reconnection)
    # ---------------------------------------------------------------------------
    phi_init = np.zeros((14, N), dtype=np.float64)
    if args.perturb > 0.0:
        sigma_p = args.lam
        dB = args.perturb * args.B0
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
        f"phi_x (perturbation only) range: [{phi_x_all.min():.4f}, {phi_x_all.max():.4f}]"
    )

    dt = args.dt * D1
    ts = (
        _lat.TimeStepper.by_level(dt)
        if args.max_level > 0
        else _lat.TimeStepper.fixed(dt)
    )

    # ---------------------------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------------------------

    # ---------------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------------
    def save_state(fname, step):
        import h5py

        with h5py.File(fname, "w") as f:
            f["pos"] = cl.leaf_pos
            f["u"] = cl.leaf_u
            f["ell"] = cl.leaf_ell
            f["phi"] = cl.leaf_phi
            f.attrs["step"] = step
            f.attrs["B0"] = args.B0
            f.attrs["lam"] = args.lam
            f.attrs["Lx"] = args.Lx
            f.attrs["Ly"] = args.Ly
            f.attrs["Lz"] = args.Lz
            f.attrs["perturb"] = args.perturb
            f.attrs["u_peak"] = args.u_peak
        print(f"  saved {fname}")

    save_state(f"{args.save_prefix}_init.h5", step_offset)

    # ---------------------------------------------------------------------------
    # Run
    # ---------------------------------------------------------------------------
    X_FACE_IDX = np.array([k for k in range(14) if abs(FACE_DX[k]) > 0])

    evolver = _lat.AdaptiveLatticeEvolver(alat)

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
        wire_periodic_bcc(cl, H, verbose=True)
        pos = cl.leaf_pos

    phi_energy_0 = float(np.sum(cl.leaf_phi**2))
    _lev0 = np.array(cl.leaf_level, dtype=np.float64)
    _vol0 = (D1 * 0.5**_lev0) ** 3
    u_vol_0 = float(np.sum(cl.leaf_u * _vol0))
    print(
        f"\nInitial AMR: N={cl.n_leaves} cells  phi_E0={phi_energy_0:.4e}  u_vol_0={u_vol_0:.6e}"
    )
    print(f"Running {args.n_steps} steps from step {step_offset}, dt={dt:.3e}")

    _live = None
    if args.live:
        from vis.substrate.harris_sheet_live import HarrisLive

        _live = HarrisLive(cl, y0, args.B0, every=args.report)

    for step in range(args.n_steps):
        evolver.step(ts)

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
            wire_periodic_bcc(cl, H)

        if _live is not None:
            _live.update(cl, step_offset + step)

        if step % args.report == 0:
            phi_now = cl.leaf_phi
            u_now = cl.leaf_u
            u_max = float(u_now.max())
            u_min = float(u_now.min())
            phi_energy = float(np.sum(phi_now**2))
            _lev = np.array(cl.leaf_level, dtype=np.float64)
            _vol = (D1 * 0.5**_lev) ** 3
            u_vol = float(np.sum(u_now * _vol))
            print(
                f"  step {step_offset+step:6d}  u=[{u_min:.4f},{u_max:.4f}]  "
                f"phi_E={phi_energy:.4e} (×{phi_energy/max(phi_energy_0,1e-30):.2f})  N={cl.n_leaves}"
                f"\n             u_vol={u_vol:.6e} (Δ={u_vol-u_vol_0:+.2e})"
            )

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

    save_state(f"{args.save_prefix}_final.h5", step_offset + args.n_steps)
    print("Done.")


if __name__ == "__main__":
    main()
