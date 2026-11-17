"""
DST solar system sandbox.

The solar system as nested chi islands:
  - Sun: smooth chi distribution at centre
  - Planets: smaller chi blobs at Keplerian radii (mass markers, not particles)
  - ISM: thin background chi (the outer buffer / heliopause analog)
  - Phi: Keplerian profile v ∝ 1/√r throughout — phi IS the orbit
  - Neumann BC: isolated chi island, no far field
  - No fixed_u anywhere

Tracks rotation curve (phi profile) and planet-region phi amplitude over time.
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")

from equilibrium import lattice as _lat
from tst.substrate.adaptive import compute_dt

D1 = 1.0
L_MIN = _lat.DiscreteCell.L_MIN

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
parser = argparse.ArgumentParser(description="DST solar system sandbox")
parser.add_argument("--R", type=float, default=30.0, help="domain radius (cells)")
parser.add_argument("--r-sun", type=float, default=3.0, help="sun chi scale radius")
parser.add_argument(
    "--lam-sun", type=float, default=2.3, help="peak solar lam (lam=2.3 ≈ chi=0.7)"
)
parser.add_argument(
    "--lam-ism",
    type=float,
    default=0.01,
    help="ISM background lam (small lam ≈ chi for lam<<1)",
)
# Planets: comma-separated r:chi pairs, all placed at theta=0
parser.add_argument(
    "--planets",
    type=str,
    default="5:0.05,10:0.02,15:0.01,20:0.005",
    help="planets as r:chi,... (Keplerian radii and chi mass proxy)",
)
parser.add_argument(
    "--blob-sigma", type=float, default=1.2, help="planet chi blob radius"
)
parser.add_argument(
    "--v-circ", type=float, default=0.05, help="orbital speed at r=r-sun"
)
parser.add_argument(
    "--flat-phi",
    action="store_true",
    help="use flat phi (v=const) instead of Keplerian",
)
parser.add_argument(
    "--g-eff",
    type=float,
    default=0.0,
    help="effective lam-phi coupling (scale modifier). 0=off",
)
parser.add_argument(
    "--g-every", type=int, default=10, help="apply chi advection every N steps"
)
parser.add_argument("--n-steps", type=int, default=100000)
parser.add_argument("--report", type=int, default=5000)
parser.add_argument("--threads", type=int, default=6)
parser.add_argument("--save-prefix", type=str, default="solar")


def main():
    args = parser.parse_args()

    # Parse planets
    _planets = []
    for tok in args.planets.split(","):
        r_s, chi_s = tok.split(":")
        _planets.append((float(r_s), float(chi_s)))

    # ---------------------------------------------------------------------------
    # Chi distribution
    # ---------------------------------------------------------------------------

    def solar_lam(pos):
        r = np.linalg.norm(pos, axis=1)
        chi_s = args.lam_sun / (1.0 + args.lam_sun)
        chi_i = args.lam_ism / (1.0 + args.lam_ism)
        chi = np.maximum(chi_s * np.exp(-r / args.r_sun), chi_i)
        chi = np.clip(chi, 0.0, 1.0 - L_MIN)
        return chi / (1.0 - chi)

    def planet_lam_bumps(pos):
        lam = np.zeros(len(pos))
        for r_p, chi_p in _planets:
            centre = np.array([r_p, 0.0, 0.0])
            dist = np.linalg.norm(pos - centre, axis=1)
            lam += (chi_p / (1.0 - chi_p)) * np.exp(
                -0.5 * (dist / args.blob_sigma) ** 2
            )
        return lam

    def apply_lam(cl):
        pos = np.array(cl.leaf_pos)
        lam = solar_lam(pos) + planet_lam_bumps(pos)
        cl.set_leaf_lam(lam)

    # ---------------------------------------------------------------------------
    # Keplerian phi:  v(r) = v_circ * sqrt(r_sun / r)   for r > r_sun
    #                        v_circ * (r / r_sun)          for r <= r_sun  (solid body)
    # ---------------------------------------------------------------------------

    def v_keplerian(r_xy):
        safe = np.maximum(r_xy, 1e-9)
        if args.flat_phi:
            return args.v_circ * np.where(r_xy <= args.r_sun, r_xy / args.r_sun, 1.0)
        return args.v_circ * np.where(
            r_xy <= args.r_sun,
            r_xy / args.r_sun,
            np.sqrt(args.r_sun / safe),
        )

    def apply_phi(cl):
        pos = np.array(cl.leaf_pos)
        r_xy = np.linalg.norm(pos[:, :2], axis=1)
        v = v_keplerian(r_xy)
        safe = np.maximum(r_xy, 1e-9)
        t_x = -pos[:, 1] / safe
        t_y = pos[:, 0] / safe

        phi = np.array(cl.leaf_phi)
        for k in range(14):
            proj = FACE_DIRS[k, 0] * t_x + FACE_DIRS[k, 1] * t_y
            phi[k] += v * proj
        cl.set_leaf_phi(phi)

    # ---------------------------------------------------------------------------
    # Diagnostics
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

    def planet_phi(cl):
        """Mean tangential phi in a window around each planet."""
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
        vals = []
        for r_p, _ in _planets:
            m = np.abs(r_xy - r_p) < args.blob_sigma * 2
            vals.append(float(np.mean(t_phi[m])) if m.any() else float("nan"))
        return vals

    # ---------------------------------------------------------------------------
    # Chi advection — effective phi→chi coupling (scale modifier)
    # ---------------------------------------------------------------------------

    def lam_advection(cl, g_eff, dt_eff):
        """
        Conservative lam advection in the phi velocity field.
        lam flows from cell j to neighbour j_nb when phi[k,j] > 0 (face k points from j to j_nb).
        Rate: g_eff × phi[k] × chi[j] per unit time (chi=lam/(1+lam) as advection fraction).
        Total lam is approximately conserved under small flux.
        """
        nb_g = np.array(cl.leaf_nb)
        phi_arr = np.array(cl.leaf_phi)  # (14, N)
        lam_arr = np.array(cl.leaf_lam)  # (N,)
        ell_arr = 1.0 / (1.0 + lam_arr)
        chi_arr = 1.0 - ell_arr  # χ = λ/(1+λ) as advection rate
        leaf_g = np.array(cl.leaf_global_indices)
        N = len(leaf_g)

        g_max = int(nb_g[nb_g >= 0].max()) + 1
        g2j = np.full(g_max, -1, dtype=np.int64)
        g2j[leaf_g.astype(int)] = np.arange(N, dtype=np.int64)

        dell = np.zeros(N)
        for k in range(14):
            valid = nb_g[:, k] >= 0
            nb_k = nb_g[valid, k].astype(int)
            jv = np.where(valid)[0]
            j_nb = g2j[nb_k]
            good = j_nb >= 0
            jv = jv[good]
            j_nb = j_nb[good]

            phi_k = phi_arr[k, jv]
            flux = g_eff * dt_eff * phi_k * chi_arr[jv]

            np.add.at(dell, jv, -flux)
            np.add.at(dell, j_nb, +flux)

        ell_new = np.clip(ell_arr - dell, L_MIN, 1.0 - L_MIN)
        cl.set_leaf_lam(1.0 / ell_new - 1.0)

    def planet_centroids(cl):
        """Lam-excess centroid angle for each planet blob."""
        pos = np.array(cl.leaf_pos)
        lam_act = np.array(cl.leaf_lam)
        lam_bg = solar_lam(pos)
        chi_exc = np.maximum(lam_act - lam_bg, 0.0)  # lam-excess as centroid weight
        angles = []
        for r_p, _ in _planets:
            r_xy = np.linalg.norm(pos[:, :2], axis=1)
            mask = np.abs(r_xy - r_p) < args.blob_sigma * 3
            w = chi_exc[mask]
            if w.sum() < 1e-20:
                angles.append(float("nan"))
            else:
                cx = float((w * pos[mask, 0]).sum() / w.sum())
                cy = float((w * pos[mask, 1]).sum() / w.sum())
                angles.append(float(np.degrees(np.arctan2(cy, cx))))
        return angles

    # ---------------------------------------------------------------------------
    # Build
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

    apply_lam(cl)
    apply_phi(cl)
    cl.set_bc(_lat.BoundaryCondition.neumann())

    planet_str = "  ".join(f"r={r}(chi={c})" for r, c in _planets)
    print(
        f"solar system      : R={args.R}  r_sun={args.r_sun}  lam_sun={args.lam_sun}  lam_ism={args.lam_ism}"
    )
    print(f"planets           : {planet_str}")
    print(f"v_circ            : {args.v_circ}  (Keplerian at r={args.r_sun})")
    print(f"N                 : {cl.n_leaves}  dt={compute_dt(cl, D1=D1):.3e}")
    print("BC                : Neumann")

    if args.save_prefix:
        alat.save(f"{args.save_prefix}_init.h5")
        print(f"saved → {args.save_prefix}_init.h5")

    # ---------------------------------------------------------------------------
    # Run
    # ---------------------------------------------------------------------------
    dt = compute_dt(cl, D1=D1)
    r_bins, v0 = rotation_curve(cl)
    print("\nInitial phi profile (Keplerian):")
    print("  r   : " + "  ".join(f"r={r:.0f}" for r in r_bins))
    print("  phi : " + "  ".join(f"{v:6.4f}" for v in v0))

    p_hdr = "  ".join(f"p_r{r:.0f}" for r, _ in _planets)
    cent_hdr = "  ".join(f"θ_r{r:.0f}°" for r, _ in _planets)
    if args.g_eff > 0:
        print(f"chi advection     : g_eff={args.g_eff}  every={args.g_every} steps")
    print(f"\n{'step':>8}  {'L/L0':>8}  {'L_tot':>12}  {p_hdr}  {cent_hdr}")

    L0 = None
    for s in range(1, args.n_steps + 1):
        alat.step(0, dt)
        if args.g_eff > 0 and s % args.g_every == 0:
            lam_advection(cl, args.g_eff, dt * args.g_every)
        if s % args.report == 0:
            _, v_curve = rotation_curve(cl)
            p_vals = planet_phi(cl)

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

            cents = planet_centroids(cl)
            p_str = "  ".join(f"{v:8.4f}" for v in p_vals)
            c_str = "  ".join(
                f"{a:8.2f}" if np.isfinite(a) else "     nan" for a in cents
            )
            print(
                f"{s:8d}  {L_ratio:8.4f}  {L_tot:12.4f}  {p_str}  {c_str}", flush=True
            )

            if L0 and L_ratio < 0.05:
                print("*** early exit: angular momentum collapsed ***")
                break

    if args.save_prefix:
        alat.save(f"{args.save_prefix}_final.h5")
        print(f"saved → {args.save_prefix}_final.h5")
    print("\ndone")


if __name__ == "__main__":
    main()
