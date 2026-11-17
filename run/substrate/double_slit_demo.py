"""
Double-slit diffraction demo on the Kelvin cell substrate.

Runs one-slit and two-slit configurations (same driven source, same
wall/screen materials -- only the aperture count differs) and plots:
  1. Backplate dose vs y for both configurations (fringes vs envelope).
  2. Source-side u snapshot, showing the reflected wave.

The wall and backplate are ordinary background-gas cells with elevated
lam (denser material), not an imposed "u=0 here" boundary condition --
see equilibrium/substrate/double_slit.py's module docstring for why that
distinction matters (reflection, edge diffraction, and partial wall
leakage all fall out of the same sigma-gradient update rule instead of
being hand-coded). The source is a continuously driven, quasi-monochromatic
emitter (not a single pulse) -- a single pulse doesn't carry enough
oscillation cycles to the screen for a clean fringe pattern to form.

Usage:
    PYTHONPATH=$PWD python3 run/substrate/double_slit_demo.py
    PYTHONPATH=$PWD python3 run/substrate/double_slit_demo.py --n-steps 2000 --out out.png
"""

import argparse
import sys

import matplotlib
import numpy as np

sys.path.insert(0, ".")

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from equilibrium.substrate.double_slit import (
    reflected_slice,
    run,
    screen_dose_slice,
    screen_leakage,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--radius-cells", type=float, default=18.0)
    p.add_argument("--gas-lam", type=float, default=0.05)
    p.add_argument("--wall-lam", type=float, default=300.0)
    p.add_argument("--screen-lam", type=float, default=300.0)
    p.add_argument("--wall-thickness", type=float, default=1.0)
    p.add_argument("--screen-thickness", type=float, default=1.0)
    p.add_argument("--slit-separation", type=float, default=6.0)
    p.add_argument("--slit-radius", type=float, default=1.0)
    p.add_argument("--x-wall", type=float, default=0.0)
    p.add_argument("--x-screen", type=float, default=10.0)
    p.add_argument("--pulse-x", type=float, default=-5.0)
    p.add_argument("--pulse-thickness", type=float, default=1.0)
    p.add_argument("--drive-amplitude", type=float, default=1.0)
    p.add_argument("--drive-period-steps", type=float, default=250.0)
    p.add_argument("--n-steps", type=int, default=1600)
    p.add_argument("--out", default="double_slit_demo.png")
    args = p.parse_args()

    common = dict(
        radius_cells=args.radius_cells,
        gas_lam=args.gas_lam,
        x_wall=args.x_wall,
        wall_thickness=args.wall_thickness,
        wall_lam=args.wall_lam,
        slit_separation=args.slit_separation,
        slit_radius=args.slit_radius,
        x_screen=args.x_screen,
        screen_thickness=args.screen_thickness,
        screen_lam=args.screen_lam,
        pulse_x=args.pulse_x,
        pulse_thickness=args.pulse_thickness,
        drive_amplitude=args.drive_amplitude,
        drive_period_steps=args.drive_period_steps,
        n_steps=args.n_steps,
    )

    print("Running one-slit ...", flush=True)
    lat1, dose1 = run(n_slits=1, **common)
    print("Running two-slit ...", flush=True)
    lat2, dose2 = run(n_slits=2, **common)

    y1, d1 = screen_dose_slice(lat1, dose1)
    y2, d2 = screen_dose_slice(lat2, dose2)

    slit1, body1 = screen_leakage(lat1, dose1)
    slit2, body2 = screen_leakage(lat2, dose2)
    print(
        f"one-slit:  behind_slit={slit1:.3e}  behind_body={body1:.3e}  ratio={body1/slit1:.3f}"
    )
    print(
        f"two-slit:  behind_slit={slit2:.3e}  behind_body={body2:.3e}  ratio={body2/slit2:.3f}"
    )

    y_refl, u_refl = reflected_slice(
        lat2, x_source_side=args.pulse_x + 2.0, thickness=1.5
    )
    print(f"reflected wave amplitude (source side): max|u|={np.abs(u_refl).max():.3e}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(y1, d1, "o-", ms=3, label="1 slit (diffraction envelope)")
    ax.plot(y2, d2, "o-", ms=3, label="2 slits (interference)")
    ax.set_xlabel("y (backplate)")
    ax.set_ylabel("accumulated dose  ∫u² dt")
    ax.set_title("Recorded pattern at the detector backplate")
    ax.legend()

    ax = axes[1]
    ax.plot(y_refl, u_refl, "o-", ms=3, color="crimson")
    ax.set_xlabel("y (source side)")
    ax.set_ylabel("u")
    ax.set_title(
        "Reflected wave, source side of the wall\n(non-zero -- wall is not an absorbing boundary)"
    )

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
