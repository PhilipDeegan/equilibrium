"""
DST Harris sheet — flux-tube-0 u initialisation (2D Gaussian tube).

u is a 2D Gaussian localised in BOTH x and y at each null line:

    u(x,y) = u_bg
            + u_peak * exp(-(y-y0)²/σy² - x²/σx²)
            + u_peak * exp(-(y+y0)²/σy² - x²/σx²)

σy = lam (sheet half-width), σx = --sigma-ux (default Lx/4).

The infinite slab sech²(y) supports the global k=0 acoustic mode (visible
bars oscillating in y).  A tube localised in x cannot sustain that global
mode; acoustic energy spreads across many kx modes that chi can damp.
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

# x-component of each face direction (len 14)
FACE_DX = np.concatenate([HEX_DIR[:, 0], SQ_DIR[:, 0]])

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="DST Harris sheet — flux-tube-0 u IC")
parser.add_argument("--Lx", type=float, default=200.0, help="box x width (d1 units)")
parser.add_argument("--Ly", type=float, default=100.0, help="box y width (d1 units)")
parser.add_argument("--Lz", type=float, default=3.5, help="box z thickness (d1 units)")
parser.add_argument("--B0", type=float, default=0.05, help="Harris sheet amplitude")
parser.add_argument("--lam", type=float, default=2.0, help="sheet half-width λ (d1)")
parser.add_argument(
    "--u-bg", type=float, default=0.4, help="background u (pressure floor)"
)
parser.add_argument("--u-peak", type=float, default=1.0, help="peak u in flux tube")
parser.add_argument(
    "--sigma-ux",
    type=float,
    default=0.0,
    help="x half-width of Gaussian tube (default Lx/4)",
)
parser.add_argument(
    "--perturb",
    type=float,
    default=0.1,
    help="perturbation amplitude as fraction of B0 (PHARE uses 0.1)",
)
parser.add_argument("--n-steps", type=int, default=10000)
parser.add_argument("--report", type=int, default=500)
parser.add_argument("--dt", type=float, default=0.003, help="timestep")
parser.add_argument(
    "--blowup-factor",
    type=float,
    default=0.0,
    help="abort if phi_energy grows by this factor (0 = disabled)",
)
parser.add_argument("--save-prefix", type=str, default="harris_ft0")
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


def main():
    args = parser.parse_args()

    D1 = 1.0
    H = D1 / np.sqrt(3.0)

    if args.sigma_ux <= 0.0:
        args.sigma_ux = args.Lx / 4.0

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
    # gradient_amr: criterion override is applied in the step loop (see below)

    N = cl.n_leaves
    pos = cl.leaf_pos  # (N, 3)

    print(f"Box Lx={args.Lx} Ly={args.Ly} Lz={args.Lz}  N={N} cells")
    print(f"Harris: B0={args.B0}  λ={args.lam}  perturb={args.perturb}")
    print(
        f"Flux tube 0 (2D Gaussian): u_peak={args.u_peak}  σy={args.lam}  σx={args.sigma_ux:.1f}"
    )

    wire_periodic_bcc(cl, H, verbose=True)

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

    phi_x_faces = [0, 2, 4, 6, 8]  # faces with dx ≤ 0

    # ---------------------------------------------------------------------------
    # u initialisation — 2D Gaussian flux tube
    # ---------------------------------------------------------------------------
    xc = pos[:, 0]
    yc = pos[:, 1]
    u_init = (
        args.u_bg
        + args.u_peak
        * np.exp(-((yc - y0) ** 2) / args.lam**2 - xc**2 / args.sigma_ux**2)
        + args.u_peak
        * np.exp(-((yc + y0) ** 2) / args.lam**2 - xc**2 / args.sigma_ux**2)
    )
    cl.set_leaf_u(u_init)
    print(f"u: min={u_init.min():.4f}  max={u_init.max():.4f}")

    # ---------------------------------------------------------------------------
    # phi perturbation (divergence-free, seeds reconnection electric field)
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
            f.attrs["sigma_ux"] = args.sigma_ux
        print(f"  saved {fname}")

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
        wire_periodic_bcc(cl, H, verbose=True)
        pos = cl.leaf_pos

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

    # Live view (optional — requires a display)
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
