"""
sweep_lam_lorentz.py — sweep B_phi to measure Lorentz deflection scaling.

For each B_phi value, initialises the 8-tile toroidal ring with a momentum
kick in x, seeds a phi-circulation background (vector potential A·dr),
runs N steps, and records the COM drift velocities.  Prints a comparison
table (both winding signs) and saves a summary plot.

Usage:
    PYTHONPATH=$PWD python3 run/substrate/sweep_lam_lorentz.py [options]
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium.substrate.flux_tube import find_toroidal_ring, init_toroidal_winding
from equilibrium.substrate.lattice import _K_REV, DampedLattice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--R", type=float, default=20.0)
    parser.add_argument("--axis", type=str, default="z")
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--kick-px", type=float, default=0.5)
    parser.add_argument(
        "--B-values",
        type=float,
        nargs="+",
        default=[0.0, 0.001, 0.005, 0.01, 0.02, 0.05],
    )
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=200000,
        help="steps for the B=0 profile run (needs ~10 ring cycles)",
    )
    parser.add_argument(
        "--profile-damp-max",
        type=float,
        default=0.05,
        help="sponge damp_max for profile run — hard sponge kills reflections",
    )
    parser.add_argument(
        "--profile-save",
        type=str,
        default="",
        help="path prefix to save profile start/end lattices (e.g. tmp/profile)",
    )
    parser.add_argument(
        "--window-r",
        type=float,
        default=4.0,
        help="COM window radius in units of ring_r (excludes far-field leakage)",
    )
    parser.add_argument("--dt-scale", type=float, default=0.1)
    parser.add_argument("--report", type=int, default=1000)
    parser.add_argument("--n-threads", type=int, default=1)
    parser.add_argument("--sponge", type=float, default=0.20)
    parser.add_argument("--damp-max", type=float, default=0.005)
    args = parser.parse_args()

    # Build one lattice once — reset fields between runs
    lat0 = DampedLattice(
        radius_cells=args.R, sponge_fraction=args.sponge, damp_max=args.damp_max
    )
    pos = lat0.pos
    dt = lat0.stable_dt() * args.dt_scale
    N = lat0.N

    ax_idx = {"x": 0, "y": 1, "z": 2}[args.axis]
    perp = [i for i in range(3) if i != ax_idx]

    # Precompute unit B-field phi array (B_z=1 via vector potential A=(B/2)(-y,x,0))
    # For each undirected edge (i<j): dphi = 0.5*(-y_mid*dx + x_mid*dy)
    _phi_B1 = np.zeros((N, 14), dtype=np.float64)
    _i_all = np.repeat(np.arange(N), 14)
    _f_all = np.tile(np.arange(14), N)
    _j_all = lat0.nb.ravel().astype(int)
    _valid = (_j_all >= 0) & (_j_all > _i_all)
    _iv, _fv, _jv = _i_all[_valid], _f_all[_valid], _j_all[_valid]
    _rm = 0.5 * (pos[_iv] + pos[_jv])
    _dr = pos[_jv] - pos[_iv]
    _p0, _p1 = perp[0], perp[1]
    _dp = 0.5 * (-_rm[:, _p1] * _dr[:, _p0] + _rm[:, _p0] * _dr[:, _p1])
    np.add.at(_phi_B1, (_iv, _fv), _dp)
    np.add.at(_phi_B1, (_jv, _K_REV[_fv].astype(int)), -_dp)
    print(f"phi_B1 precomputed  rms={np.sqrt(np.mean(_phi_B1**2)):.4e}\n")

    ring_tiles, ring_faces = find_toroidal_ring(lat0, axis=args.axis)
    if ring_tiles is None:
        print("ERROR: no ring found")
        sys.exit(1)

    ring_pos = pos[ring_tiles]
    ring_centre = ring_pos.mean(axis=0)
    ring_r = float(np.linalg.norm(ring_pos - ring_centre, axis=1).mean())
    N_ring = len(ring_tiles)
    K_ring = 2 * np.pi / N_ring

    print(f"N={N}  R={args.R}  dt={dt:.4e}  n_steps={args.n_steps}")
    print(f"kick_px={args.kick_px}  B values: {args.B_values}\n")

    # Window for COM: cells within window_r * ring_r of ring centre — excludes far-field leakage
    _win_r = args.window_r * ring_r
    _window = np.where(np.linalg.norm(pos - ring_centre, axis=1) < _win_r)[0]
    print(f"COM window: r < {_win_r:.2f} cells  ({len(_window)} cells)\n")

    def radial_phi_profile(r_min_fac=1.5, n_bins=25):
        """
        Bin mean phi_rms by distance from ring_centre over the final lattice state.
        Starts at r_min_fac * ring_r to exclude the ring tiles themselves.
        Returns (r_mid, mean_phi_rms, counts).
        """
        r_all = np.linalg.norm(pos - ring_centre, axis=1)
        phi_rms = np.sqrt((lat0.phi**2).mean(axis=1))
        r_min = ring_r * r_min_fac
        r_max = float(r_all.max()) * 0.80  # stay inside sponge
        edges = np.linspace(r_min, r_max, n_bins + 1)
        r_mid = 0.5 * (edges[:-1] + edges[1:])
        totals = np.zeros(n_bins)
        counts = np.zeros(n_bins, dtype=int)
        idx_all = ((r_all - r_min) / (r_max - r_min) * n_bins).astype(int)
        valid = (r_all >= r_min) & (r_all < r_max) & (idx_all >= 0) & (idx_all < n_bins)
        np.add.at(totals, idx_all[valid], phi_rms[valid])
        np.add.at(counts, idx_all[valid], 1)
        mask = counts > 0
        mean_prof = np.where(mask, totals / np.maximum(counts, 1), np.nan)
        return r_mid, mean_prof, counts

    def fit_power_law(r_mid, profile):
        """Fit profile = A * r^n via log-log least squares. Returns (A, n)."""
        mask = np.isfinite(profile) & (profile > 0) & (r_mid > 0)
        if mask.sum() < 3:
            return None, None
        lr = np.log(r_mid[mask])
        ly = np.log(profile[mask])
        coeffs = np.polyfit(lr, ly, 1)
        return float(np.exp(coeffs[1])), float(coeffs[0])

    def run_one(B_phi, sign=+1):
        """Reset lattice, seed ring + phi B-field, run, return (vx, vy, winding_final).
        sign=+1 for normal winding, sign=-1 for flipped (anti-electron).

        B field in z is seeded as the vector potential A = (B/2)(-y, x, 0):
            phi[i,f] += A(r_mid) · dr  for each undirected edge once.
        This is chiral — the sign of the curl depends on winding direction.
        """
        lat0.u[:] = 0.0
        lat0.phi[:] = 0.0
        lat0.lam[:] = 0.0

        init_toroidal_winding(
            lat0, ring_tiles, ring_faces, amplitude=sign * args.amplitude, frozen=False
        )

        # Momentum kick: phase tilt in perp[0] direction
        if args.kick_px != 0.0:
            for j, (t, f) in enumerate(zip(ring_tiles, ring_faces)):
                phase = K_ring * (j + 0.5) + args.kick_px * float(pos[t, perp[0]])
                lat0.phi[t, f] = sign * args.amplitude * np.cos(phase)
                nb_t = lat0.nb[t, f]
                if nb_t >= 0:
                    lat0.phi[nb_t, int(_K_REV[f])] = -lat0.phi[t, f]

        # B field as phi circulation: A = (B/2)(-y, x, 0)
        if B_phi != 0.0:
            lat0.phi += B_phi * _phi_B1

        def phi_com():
            prms = np.sqrt((lat0.phi[_window] ** 2).mean(axis=1))
            w = prms.sum()
            if w < 1e-30:
                return float(ring_centre[perp[0]]), float(ring_centre[perp[1]])
            return (
                float(np.dot(prms, pos[_window, perp[0]]) / w),
                float(np.dot(prms, pos[_window, perp[1]]) / w),
            )

        def winding_amp():
            total = sum(
                float(lat0.phi[t, f])
                * complex(
                    np.cos(2 * np.pi * j / N_ring), np.sin(2 * np.pi * j / N_ring)
                )
                for j, (t, f) in enumerate(zip(ring_tiles, ring_faces))
            )
            return abs(total) / (N_ring / 2)

        com_hist = []
        for s in range(1, args.n_steps + 1):
            lat0.step(dt, n_threads=args.n_threads)
            if s % args.report == 0:
                cx, cy = phi_com()
                com_hist.append((s, cx, cy))

        if len(com_hist) < 2:
            return 0.0, 0.0, winding_amp()

        s0, cx0, cy0 = com_hist[0]
        sf, cxf, cyf = com_hist[-1]
        t_elapsed = (sf - s0) * dt
        vx = (cxf - cx0) / t_elapsed
        vy = (cyf - cy0) / t_elapsed
        return vx, vy, winding_amp()

    # ---------------------------------------------------------------------------
    # Profile run — B=0, no kick: clean radial phi measurement
    # ---------------------------------------------------------------------------
    _ring_period = 2 * np.pi / 0.765  # approximate, in time units
    _profile_time = args.profile_steps * dt
    print("=== radial phi profile (B=0, kick=0) ===")
    print(
        f"  profile_steps={args.profile_steps}  total_time={_profile_time:.1f}  "
        f"ring_cycles≈{_profile_time / _ring_period:.1f}  "
        f"profile_damp_max={args.profile_damp_max}"
    )
    lat0.u[:] = 0.0
    lat0.phi[:] = 0.0
    lat0.lam[:] = 0.0
    init_toroidal_winding(
        lat0, ring_tiles, ring_faces, amplitude=args.amplitude, frozen=False
    )
    _sponge_orig = lat0._sponge.copy()
    lat0._sponge = _sponge_orig * (args.profile_damp_max / args.damp_max)
    if args.profile_save:
        import os

        os.makedirs(os.path.dirname(args.profile_save) or ".", exist_ok=True)
        lat0.save(args.profile_save + "_start.npz", step_dt=dt)
        print(f"  [saved] {args.profile_save}_start.npz")
    for _s in range(args.profile_steps):
        lat0.step(dt, n_threads=args.n_threads)
    if args.profile_save:
        lat0.save(args.profile_save + "_end.npz", step_dt=dt)
        print(f"  [saved] {args.profile_save}_end.npz")
    lat0._sponge = _sponge_orig
    r_mid_prof, phi_prof, prof_counts = radial_phi_profile()
    A_fit, n_fit = fit_power_law(r_mid_prof, phi_prof)
    print(f"  power-law fit: phi_rms ~ {A_fit:.3e} * r^{n_fit:.3f}  (dipole = -3.0)")
    print(f"  {'r':>8}  {'phi_rms':>12}  {'counts':>8}")
    for r_, p_, c_ in zip(r_mid_prof, phi_prof, prof_counts):
        if c_ > 0:
            print(f"  {r_:8.3f}  {p_:12.4e}  {c_:8d}")
    print()

    # ---------------------------------------------------------------------------
    # Sweep — both winding signs
    # ---------------------------------------------------------------------------
    def run_sweep(sign):
        label = "+1 (normal)" if sign > 0 else "-1 (flipped)"
        print(f"\n=== winding sign {label} ===")
        print(
            f"{'B_phi':>8}  {'vx':>12}  {'vy':>12}  {'Δvx':>12}  {'Δvy':>12}  {'|wind|':>8}"
        )
        results = []
        vx0, vy0 = None, None
        for B_phi in args.B_values:
            print(f"  running B_phi={B_phi:.3f} ...", end="", flush=True)
            vx, vy, wa = run_one(B_phi, sign=sign)
            if vx0 is None:
                vx0, vy0 = vx, vy
            dvx = vx - vx0
            dvy = vy - vy0
            results.append((B_phi, vx, vy, dvx, dvy, wa))
            print(
                f"\r{B_phi:8.3f}  {vx:+12.4e}  {vy:+12.4e}  {dvx:+12.4e}  {dvy:+12.4e}  {wa:8.6f}"
            )
        return results

    results_pos = run_sweep(sign=+1)
    results_neg = run_sweep(sign=-1)

    # ---------------------------------------------------------------------------
    # Sign-flip summary
    # ---------------------------------------------------------------------------
    print("\n=== sign-flip comparison at each B_phi ===")
    print(
        f"{'B_phi':>8}  {'Δvx (+1)':>14}  {'Δvx (-1)':>14}  {'Δvy (+1)':>14}  "
        f"{'Δvy (-1)':>14}  {'sum Δvx':>12}  {'sum Δvy':>12}"
    )
    for rp, rn in zip(results_pos, results_neg):
        B = rp[0]
        dvxp, dvyp = rp[3], rp[4]
        dvxn, dvyn = rn[3], rn[4]
        print(
            f"{B:8.3f}  {dvxp:+14.4e}  {dvxn:+14.4e}  {dvyp:+14.4e}  "
            f"{dvyn:+14.4e}  {dvxp+dvxn:+12.4e}  {dvyp+dvyn:+12.4e}"
        )

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        B_vals = [r[0] for r in results_pos]
        dvxp = [r[3] for r in results_pos]
        dvyp = [r[4] for r in results_pos]
        dvxn = [r[3] for r in results_neg]
        dvyn = [r[4] for r in results_neg]
        vxp = [r[1] for r in results_pos]
        vyp = [r[2] for r in results_pos]
        vxn = [r[1] for r in results_neg]
        vyn = [r[2] for r in results_neg]

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(
            f"Lorentz sweep (phi B-field)  kick_px={args.kick_px}", fontsize=10
        )

        # Panel 1: absolute velocities for both signs
        axes[0].plot(B_vals, vxp, "r-o", ms=5, label="vx (+1)")
        axes[0].plot(B_vals, vyp, "g-o", ms=5, label="vy (+1)")
        axes[0].plot(B_vals, vxn, "r--s", ms=5, label="vx (-1)")
        axes[0].plot(B_vals, vyn, "g--s", ms=5, label="vy (-1)")
        axes[0].axhline(0, color="k", lw=0.5)
        axes[0].set_xlabel("B_phi")
        axes[0].set_ylabel("drift velocity")
        axes[0].set_title("vx and vy — both winding signs")
        axes[0].legend(fontsize=7)

        # Panel 2: Δvy per sign (Lorentz: opposite sign → antisymmetric)
        axes[1].plot(B_vals, dvyp, "b-o", ms=5, label="Δvy winding +1")
        axes[1].plot(B_vals, dvyn, "m--s", ms=5, label="Δvy winding -1")
        axes[1].axhline(0, color="k", lw=0.5)
        axes[1].set_xlabel("B_phi")
        axes[1].set_ylabel("Δvy")
        axes[1].set_title("Δvy per winding sign\n(Lorentz: opposite signs)")
        axes[1].legend(fontsize=7)

        # Panel 3: sums — Lorentz gives Δvy sum→0, Δvx sum→0
        sum_dvx = [a + b for a, b in zip(dvxp, dvxn)]
        sum_dvy = [a + b for a, b in zip(dvyp, dvyn)]
        axes[2].plot(B_vals, sum_dvx, "r-o", ms=5, label="Δvx(+1)+Δvx(-1)")
        axes[2].plot(B_vals, sum_dvy, "b-o", ms=5, label="Δvy(+1)+Δvy(-1)")
        axes[2].axhline(0, color="k", lw=1, ls="--")
        axes[2].set_xlabel("B_phi")
        axes[2].set_ylabel("sum of both signs")
        axes[2].set_title(
            "sum of Δvx / Δvy\n(→ 0 = pure Lorentz, ≠ 0 = lattice artefact)"
        )
        axes[2].legend(fontsize=7)

        # Panel 4: radial phi profile at B=0
        _mask = (prof_counts > 0) & np.isfinite(phi_prof)
        axes[3].semilogy(
            r_mid_prof[_mask], phi_prof[_mask], "ko-", ms=4, label="phi_rms (B=0)"
        )
        if A_fit is not None:
            _r_fit = np.linspace(r_mid_prof[_mask].min(), r_mid_prof[_mask].max(), 100)
            axes[3].semilogy(
                _r_fit, A_fit * _r_fit**n_fit, "b--", lw=1.5, label=f"fit r^{n_fit:.2f}"
            )
            axes[3].semilogy(
                _r_fit,
                A_fit * (_r_fit / _r_fit[0]) ** (-3) * phi_prof[_mask][0],
                "r:",
                lw=1.2,
                label="r^-3 (dipole)",
            )
        axes[3].set_xlabel("r (cells)")
        axes[3].set_ylabel("mean phi_rms")
        axes[3].set_title(f"radial phi profile (B=0)\nfit: r^{n_fit:.2f}  (dipole=-3)")
        axes[3].legend(fontsize=7)

        plt.tight_layout()
        fname = "sweep_lam_lorentz.png"
        fig.savefig(fname, dpi=110)
        print(f"\n[plot] {fname}")
        plt.close(fig)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
