"""
Solar system DST simulation — multi-body chi field.

All 8 planets placed at J2000 mean longitudes (not collinear).
AMR refinement is built geometrically at init (split-on-init) — no
warm-up loop required.  Chi is proportional to mass × mean_Z.

Scale: Sun radius = R_sun_cells base cells.  All orbital radii are
compressed by a common factor so the outermost body fits inside the domain.
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")

from equilibrium.lattice import DiscreteCell

from equilibrium import lattice as _lat
from tst.substrate.adaptive import compute_dt

D1 = 1.0
L_MIN = DiscreteCell.L_MIN

# ---------------------------------------------------------------------------
# Physical data (SI: kg, km)
# mean_Z: effective mean atomic number of dominant composition
#   rocky bodies (Fe/Si) → high Z; gas/ice giants (H/He/H2O) → low Z
# orbit_angle_deg: J2000.0 mean longitude (approximate starting position)
# ---------------------------------------------------------------------------
SOLAR_DATA = {
    "Sun": {
        "mass": 1.9884e30,
        "radius_km": 696340,
        "orbit_km": 0,
        "mean_Z": 1.3,
        "orbit_angle_deg": 0.0,
    },
    "Mercury": {
        "mass": 3.285e23,
        "radius_km": 2440,
        "orbit_km": 57.9e6,
        "mean_Z": 22.0,
        "orbit_angle_deg": 252.3,
    },
    "Venus": {
        "mass": 4.867e24,
        "radius_km": 6052,
        "orbit_km": 108.2e6,
        "mean_Z": 14.0,
        "orbit_angle_deg": 182.0,
    },
    "Earth": {
        "mass": 5.972e24,
        "radius_km": 6371,
        "orbit_km": 149.6e6,
        "mean_Z": 17.0,
        "orbit_angle_deg": 100.5,
    },
    "Mars": {
        "mass": 6.390e23,
        "radius_km": 3390,
        "orbit_km": 227.9e6,
        "mean_Z": 13.0,
        "orbit_angle_deg": 355.5,
    },
    "Jupiter": {
        "mass": 1.898e27,
        "radius_km": 69911,
        "orbit_km": 778.5e6,
        "mean_Z": 2.0,
        "orbit_angle_deg": 34.4,
    },
    "Saturn": {
        "mass": 5.683e26,
        "radius_km": 58232,
        "orbit_km": 1432.0e6,
        "mean_Z": 2.0,
        "orbit_angle_deg": 50.1,
    },
    "Uranus": {
        "mass": 8.681e25,
        "radius_km": 25362,
        "orbit_km": 2872.5e6,
        "mean_Z": 3.0,
        "orbit_angle_deg": 314.1,
    },
    "Neptune": {
        "mass": 1.024e26,
        "radius_km": 24622,
        "orbit_km": 4495.1e6,
        "mean_Z": 3.5,
        "orbit_angle_deg": 304.3,
    },
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Solar system DST simulation")
parser.add_argument(
    "--R", type=float, default=150.0, help="domain radius in base cells"
)
parser.add_argument("--R-sun", type=float, default=5.0, help="Sun radius in base cells")
parser.add_argument(
    "--lam-sun", type=float, default=1.0, help="Sun lam scale (lam=1 ≈ chi=0.5)"
)
parser.add_argument("--u-sun", type=float, default=5.0, help="Sun fixed-u amplitude")
parser.add_argument(
    "--u-planet", type=float, default=0.5, help="planet u-blob amplitude"
)
parser.add_argument(
    "--sigma-planet", type=float, default=2.0, help="planet u-blob width (cells)"
)
parser.add_argument(
    "--bodies",
    type=str,
    default="Sun,Mercury,Venus,Earth,Mars,Jupiter,Saturn,Uranus,Neptune",
    help="comma-separated list of bodies to include",
)
parser.add_argument("--max-level", type=int, default=3)
parser.add_argument(
    "--split-radius-factor",
    type=float,
    default=4.0,
    help="split cells within this × body_radius of each body",
)
parser.add_argument("--n-steps", type=int, default=50000, help="main run steps")
parser.add_argument("--report", type=int, default=5000)
parser.add_argument("--threads", type=int, default=6)
parser.add_argument(
    "--sponge-weight",
    type=float,
    default=0.1,
    help="outer BC sponge weight (0=off, lower = less aggressive)",
)
parser.add_argument(
    "--save-prefix", type=str, default="", help="path prefix for .h5 checkpoints"
)
parser.add_argument(
    "--save-every", type=int, default=0, help="save every N steps (0=off)"
)
parser.add_argument("--resume", type=str, default="", help="resume from checkpoint .h5")
parser.add_argument(
    "--live", action="store_true", help="open live equatorial slice view"
)
parser.add_argument(
    "--live-every", type=int, default=500, help="update live view every N steps"
)
parser.add_argument(
    "--live-grid-n",
    type=int,
    default=300,
    help="live view interpolation grid resolution",
)


def main():
    args = parser.parse_args()

    bodies = [b.strip() for b in args.bodies.split(",")]
    assert "Sun" in bodies, "Sun must be included"

    # ---------------------------------------------------------------------------
    # Compute compressed simulation positions
    # ---------------------------------------------------------------------------
    sun_km = SOLAR_DATA["Sun"]["radius_km"]
    km_per_cell = sun_km / args.R_sun
    mass_sun = SOLAR_DATA["Sun"]["mass"]
    Z_sun = SOLAR_DATA["Sun"]["mean_Z"]

    orbit_kms = {b: SOLAR_DATA[b]["orbit_km"] for b in bodies if b != "Sun"}
    max_orbit_km = max(orbit_kms.values()) if orbit_kms else 1.0
    orbit_scale = (0.87 * args.R) / (max_orbit_km / km_per_cell)

    body_params = {}
    for name in bodies:
        d = SOLAR_DATA[name]
        orbit_cells = (d["orbit_km"] / km_per_cell) * orbit_scale
        radius_cells = max(d["radius_km"] / km_per_cell, 0.5)
        mass_Z_ratio = (d["mass"] / mass_sun) * (d["mean_Z"] / Z_sun)
        chi_param = (args.lam_sun / (1.0 + args.lam_sun)) * args.R_sun * mass_Z_ratio
        chi0 = min(chi_param / max(radius_cells, 1.0), 1.0 - L_MIN)
        angle_rad = np.radians(d["orbit_angle_deg"])
        body_params[name] = {
            "orbit_cells": orbit_cells,
            "radius_cells": radius_cells,
            "chi0": chi0,
            "mass_Z_ratio": mass_Z_ratio,
            "angle_rad": angle_rad,
            # 3-D centre in simulation space (z=0 orbital plane)
            "centre": np.array(
                [
                    orbit_cells * np.cos(angle_rad),
                    orbit_cells * np.sin(angle_rad),
                    0.0,
                ]
            ),
        }

    print("Body layout (compressed scale):")
    print(f"  km_per_cell={km_per_cell:.1f}  orbit_scale={orbit_scale:.4f}")
    print(
        f"  {'body':>10}  {'orbit_c':>9}  {'angle':>7}  {'R_c':>7}  {'chi0':>10}  {'m*Z ratio':>12}"
    )
    for name, p in body_params.items():
        print(
            f"  {name:>10}  {p['orbit_cells']:9.2f}  {np.degrees(p['angle_rad']):7.1f}°"
            f"  {p['radius_cells']:7.3f}  {p['chi0']:10.6f}  {p['mass_Z_ratio']:12.3e}"
        )

    # ---------------------------------------------------------------------------
    # Field helpers
    # ---------------------------------------------------------------------------

    def apply_lam_field(cl):
        pos = np.array(cl.leaf_pos)  # (N,3)
        chi = np.zeros(cl.n_leaves)
        for name, p in body_params.items():
            centre = p["centre"]
            R_b = p["radius_cells"]
            chi0 = p["chi0"]
            r_b = np.linalg.norm(pos - centre, axis=1)
            inside = r_b <= R_b
            outside = ~inside
            chi[inside] = np.maximum(chi[inside], chi0)
            chi[outside] = np.maximum(
                chi[outside], chi0 * R_b / np.maximum(r_b[outside], 1e-9)
            )
        chi = np.clip(chi, 0.0, 1.0 - L_MIN)
        cl.set_leaf_lam(chi / (1.0 - chi))

    def apply_sun_u(cl):
        pos = np.array(cl.leaf_pos)
        r = np.linalg.norm(pos, axis=1)
        u = np.array(cl.leaf_u)
        fu = np.zeros(cl.n_leaves, dtype=np.uint8)
        mask = r <= body_params["Sun"]["radius_cells"]
        u[mask] = args.u_sun
        fu[mask] = 1
        cl.set_leaf_u(u)
        cl.set_leaf_fixed_u(fu)

    def place_planet_blobs(cl):
        pos = np.array(cl.leaf_pos)
        u = np.array(cl.leaf_u)
        for name, p in body_params.items():
            if name == "Sun":
                continue
            r2 = np.sum((pos - p["centre"]) ** 2, axis=1)
            u += args.u_planet * np.exp(-r2 / (2.0 * args.sigma_planet**2))
        cl.set_leaf_u(u)

    def planet_centroids(cl):
        pos = np.array(cl.leaf_pos)
        u = np.array(cl.leaf_u)
        r_sun = body_params["Sun"]["radius_cells"] * 3.0
        result = {}
        for name, p in body_params.items():
            if name == "Sun":
                continue
            centre = p["centre"]
            window = max(p["orbit_cells"] * 0.5, args.sigma_planet * 5.0)
            dist_ic = np.linalg.norm(pos - centre, axis=1)
            r_from_sun = np.linalg.norm(pos, axis=1)
            mask = (dist_ic < window) & (r_from_sun > r_sun)
            w = np.maximum(u[mask], 0.0)
            wsum = float(w.sum())
            if wsum < 1e-20:
                result[name] = np.full(3, float("nan"))
            else:
                result[name] = (w[:, None] * pos[mask]).sum(axis=0) / wsum
        return result

    # ---------------------------------------------------------------------------
    # Split-on-init: refine cells geometrically near each body, level by level.
    # Replaces the warm-up AMR loop — no stepping required.
    # ---------------------------------------------------------------------------

    def split_near_bodies(alat, cl, max_level):
        """Geometrically refine cells near each body up to max_level.
        Uses split_batch so all cells at each level are split in one pass
        with a single rebuild_leaf_indices at the end — no per-cell O(N) work.
        """
        split_r_factor = args.split_radius_factor
        for level in range(max_level):
            pos = np.array(cl.leaf_pos)  # (N, 3)
            levs = np.array(cl.leaf_level, dtype=int)
            idxs = np.array(cl.leaf_global_indices, dtype=np.int32)

            # Find cells at this level that are within split_r of any body.
            at_level = levs == level
            to_split = np.zeros(cl.n_leaves, dtype=bool)
            for name, p in body_params.items():
                split_r = max(p["radius_cells"] * split_r_factor, D1 * 2.0)
                dists = np.linalg.norm(pos - p["centre"], axis=1)
                to_split |= at_level & (dists < split_r)

            gs = np.ascontiguousarray(idxs[to_split], dtype=np.int32)
            d1_at_level = D1 / float(1 << level)
            n_split = alat.split_batch(0, gs, d1_at_level)

            apply_lam_field(cl)
            apply_sun_u(cl)
            print(
                f"  split-on-init level {level}→{level+1}: "
                f"split {n_split} cells, now {cl.n_leaves} leaves",
                flush=True,
            )

    # ---------------------------------------------------------------------------
    # Build base grid
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
    info.max_level = args.max_level
    cl.info = info

    import os

    print(
        f"\nR={args.R}  R_sun={args.R_sun}  lam_sun={args.lam_sun}  max_level={args.max_level}"
    )

    if args.resume:
        print(f"resuming from {args.resume} …", flush=True)
        alat = _lat.AdaptiveLattice.load(args.resume)
        alat.set_n_threads(args.threads)
        cl = alat[0]
        print(f"  N={cl.n_leaves}", flush=True)
    else:
        # Apply chi field on level-0 grid, then split-on-init
        apply_lam_field(cl)
        apply_sun_u(cl)

        if args.max_level > 0:
            print(f"\nSplit-on-init ({args.max_level} levels) …", flush=True)
            split_near_bodies(alat, cl, args.max_level)

        # Sponge BC at outer boundary — use mild weight to avoid eating solar field
        if args.sponge_weight > 0.0:
            cl.set_bc(
                _lat.BoundaryCondition.sponge(target=0.0, weight=args.sponge_weight)
            )

        place_planet_blobs(cl)

        if args.save_prefix:
            os.makedirs(os.path.dirname(args.save_prefix) or ".", exist_ok=True)
            ckpt = f"{args.save_prefix}_init.h5"
            alat.save(ckpt)
            print(f"saved init state → {ckpt}", flush=True)

    # ---------------------------------------------------------------------------
    # Place planets and run
    # ---------------------------------------------------------------------------
    dt = compute_dt(cl, D1=D1)

    live_view = None
    if args.live:
        from vis.substrate.solar_system_live import SolarSystemLive

        live_view = SolarSystemLive(
            cl,
            body_params,
            every=args.live_every,
            grid_n=args.live_grid_n,
        )

    planet_names = [b for b in bodies if b != "Sun"]
    header = f"{'step':>8}" + "".join(
        f"  {n[:3]:>8}_r  {n[:3]:>8}_th" for n in planet_names
    )
    print(f"\n=== main run  N={cl.n_leaves}  dt={dt:.3e} ===")
    print(header)

    for s in range(1, args.n_steps + 1):
        alat.step(0, dt)
        if live_view is not None:
            live_view.update(cl, s)
        if s % args.report == 0:
            cs = planet_centroids(cl)
            row = f"{s:8d}"
            for name in planet_names:
                c = cs[name]
                if np.all(np.isfinite(c)):
                    r_c = float(np.linalg.norm(c[:2]))
                    th = float(np.degrees(np.arctan2(c[1], c[0])))
                else:
                    r_c = float("nan")
                    th = float("nan")
                row += f"  {r_c:10.3f}  {th:10.2f}"
            print(row, flush=True)
            if args.save_prefix and args.save_every and s % args.save_every == 0:
                ckpt = f"{args.save_prefix}_s{s:08d}.h5"
                alat.save(ckpt)
                print(f"  saved → {ckpt}", flush=True)

    print("\ndone")


if __name__ == "__main__":
    main()
