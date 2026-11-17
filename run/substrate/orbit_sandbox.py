"""
DST orbit sandbox.

Planet is a winding knot (BFS 1/r additive on Sun's lam field).
Orbital motion is driven by a tangential u-asymmetry: lower u at the leading
edge drives outward phi → du > 0 at the front → lam excess advances tangentially.

Sun-growth is suppressed by setting lam = lam_background on background cells.
sigma = 1/(1+lam), so high lam slows u propagation outside active regions.

The u field is pre-initialised to the variable-sigma Laplace steady state so
initial phi ≈ 0 in the background.
"""

import argparse
import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, ".")

from equilibrium.lattice import DiscreteCell

from equilibrium import lattice as _lat
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import (
    apply_toroidal_rotation,
)

D1 = 1.0
L_MIN = DiscreteCell.L_MIN

# BCC face unit-vectors (hex×8, sq×6)
_HEX = np.array(
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
    dtype=float,
) / np.sqrt(3)
_SQ = np.array(
    [
        [-1, 0, 0],
        [+1, 0, 0],
        [0, -1, 0],
        [0, +1, 0],
        [0, 0, -1],
        [0, 0, +1],
    ],
    dtype=float,
)
FACE_DIRS = np.vstack([_HEX, _SQ])  # (14, 3)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="DST orbit sandbox")
parser.add_argument("--R", type=float, default=40.0)
parser.add_argument("--R-sun", type=float, default=5.0, help="solar radius in cells")
parser.add_argument(
    "--lam-sun", type=float, default=9.0, help="max lam at solar core (lam=9 ≈ chi=0.9)"
)
parser.add_argument(
    "--u-core",
    type=float,
    default=10.0,
    help="fixed u at solar core (self-sustaining knot source)",
)
parser.add_argument(
    "--R-core", type=float, default=0.5, help="radius of pinned solar core in cells"
)
parser.add_argument(
    "--sponge-weight",
    type=float,
    default=0.05,
    help="sponge BC weight at outer boundary",
)
parser.add_argument("--r-orbit", type=float, default=20.0)
parser.add_argument(
    "--orbit-angle",
    type=float,
    default=0.0,
    help="initial angle in orbital plane (deg)",
)
parser.add_argument(
    "--orbit-tilt",
    type=float,
    default=0.0,
    help="tilt of orbital plane from xy (deg, around y)",
)
parser.add_argument(
    "--lam-planet",
    type=float,
    default=0.43,
    help="planet knot lam at origin cell (lam=0.43 ≈ chi=0.3)",
)
parser.add_argument(
    "--v-tangential",
    type=float,
    default=1.0,
    help="dipole phi magnitude (tune empirically)",
)
parser.add_argument(
    "--v-rotation",
    type=float,
    default=0.0,
    help="toroidal phi amplitude for Sun rotation (0=off)",
)
parser.add_argument(
    "--spin-axis", type=str, default="z", help="Sun spin axis: x, y, or z"
)
parser.add_argument(
    "--lam-background",
    type=float,
    default=100.0,
    help="lam for background cells — slows u propagation outside active regions",
)
parser.add_argument("--galactic-lam", type=float, default=0.0)
parser.add_argument(
    "--galactic-tilt",
    type=float,
    default=60.0,
    help="galactic axis tilt from z toward x (deg)",
)
parser.add_argument("--max-level", type=int, default=2)
parser.add_argument("--n-steps", type=int, default=200000)
parser.add_argument("--report", type=int, default=1000)
parser.add_argument("--threads", type=int, default=6)
parser.add_argument(
    "--no-planet", action="store_true", help="Sun-only run — skip planet and phi init"
)
parser.add_argument("--save-prefix", type=str, default="")
parser.add_argument("--save-every", type=int, default=0)
parser.add_argument("--resume", type=str, default="")
parser.add_argument("--live", action="store_true")
parser.add_argument("--live-every", type=int, default=200)
parser.add_argument("--live-grid-n", type=int, default=200)


def main():
    args = parser.parse_args()

    # ---------------------------------------------------------------------------
    # Derived geometry
    # ---------------------------------------------------------------------------
    _tilt = np.radians(args.orbit_tilt)
    _angle = np.radians(args.orbit_angle)

    orbit_normal = np.array([-np.sin(_tilt), 0.0, np.cos(_tilt)])
    _e1 = np.array([np.cos(_tilt), 0.0, np.sin(_tilt)])
    _e2 = np.array([0.0, 1.0, 0.0])
    planet_centre = args.r_orbit * (np.cos(_angle) * _e1 + np.sin(_angle) * _e2)

    _gtilt = np.radians(args.galactic_tilt)
    galactic_axis = np.array([np.sin(_gtilt), 0.0, np.cos(_gtilt)])

    print(f"planet centre   : {planet_centre}")
    print(f"orbit normal    : {orbit_normal}")
    print(f"galactic axis   : {galactic_axis}")

    # ---------------------------------------------------------------------------
    # Solar lam profile — three-zone model matching current scientific understanding
    #
    #   Core        (r < 0.25 R_sun) : flat lam_max — nuclear fusion zone
    #   Radiative   (0.25–0.70 R_sun): slow exponential decay — radiation-dominated
    #   Tachocline  (~0.70 R_sun)    : kink / sharp transition — dynamo seat
    #   Convective  (0.70+ R_sun)    : fast exponential decay — convection-dominated
    # ---------------------------------------------------------------------------

    def solar_lam(r_arr):
        R = args.R_sun
        chi_0 = args.lam_sun / (1.0 + args.lam_sun)  # core chi from lam_sun
        R_core = 0.25 * R
        R_tach = 0.70 * R
        a_rad = 0.50 * R
        a_conv = 0.15 * R
        chi_at_tach = chi_0 * np.exp(-(R_tach - R_core) / a_rad)
        chi = np.where(
            r_arr <= R_core,
            chi_0,
            np.where(
                r_arr <= R_tach,
                chi_0 * np.exp(-(r_arr - R_core) / a_rad),
                chi_at_tach * np.exp(-(r_arr - R_tach) / a_conv),
            ),
        )
        return chi / np.maximum(1.0 - chi, L_MIN)

    def sun_lam_background(pos):
        """Solar lam profile at given positions — used for planet lam-excess detection."""
        r = np.linalg.norm(pos, axis=1)
        return solar_lam(r)

    def apply_solar_lam(cl):
        """Set lam from solar lam profile (+ optional galactic background)."""
        pos = np.array(cl.leaf_pos)
        r = np.linalg.norm(pos, axis=1)
        lam = solar_lam(r)
        if args.galactic_lam > 0.0:
            proj = pos @ galactic_axis
            lam_gal = args.galactic_lam * (1.0 + proj / args.R)
            lam = np.maximum(lam, np.clip(lam_gal, 0.0, 1.0 / L_MIN - 1.0))
        cl.set_leaf_lam(lam)

    def pin_solar_core(cl):
        """
        Pin u at the solar core (r < R_core) — represents the self-sustaining u source
        from the densely packed, topologically-protected knot collection in the core.
        Maintains the u gradient that drives orbital dynamics.
        Not specifically a BH — any dense enough knot collection does this.
        """
        pos = np.array(cl.leaf_pos)
        r = np.linalg.norm(pos, axis=1)
        u = np.array(cl.leaf_u)
        fu = np.zeros(cl.n_leaves, dtype=np.uint8)
        mask = r <= args.R_core
        u[mask] = args.u_core
        fu[mask] = 1
        cl.set_leaf_u(u)
        cl.set_leaf_fixed_u(fu)

    def init_u_steady(cl):
        """
        Initialize u to the variable-sigma Laplace steady state:
            ∇·(sigma ∇u) = 0   with   u(R_bh) = u_bh,  u(R) → 0 (sponge drain)

        In spherical symmetry: sigma(r) r² du/dr = C (constant flux).
        The correct DST steady state — not the Poisson potential, which assumes
        constant sigma and gives the wrong profile for variable chi.

        With BH core (fixed u source) + sponge BC (slow drain at boundary),
        this u profile is maintained at runtime without chi drift.
        """
        pos = np.array(cl.leaf_pos)
        r_arr = np.linalg.norm(pos, axis=1)

        r_bh = max(args.R_core, D1 / (4.0 * (1 << args.max_level)))
        r_grid = np.linspace(r_bh, args.R * 1.001, 4000)

        lam_g = np.where(r_grid <= args.R_sun * 1.2, 0.0, float(args.lam_background))
        sig_g = np.maximum(1.0 / (1.0 + lam_g), 1e-9)

        integrand = 1.0 / (sig_g * r_grid**2)
        cum = np.zeros(len(r_grid))
        for i in range(1, len(r_grid)):
            cum[i] = cum[i - 1] + 0.5 * (integrand[i - 1] + integrand[i]) * (
                r_grid[i] - r_grid[i - 1]
            )

        C = -args.u_core / cum[-1]  # u goes from u_bh → 0
        u_grid = np.maximum(args.u_core + C * cum, 0.0)

        u_arr = np.interp(np.maximum(r_arr, r_bh), r_grid, u_grid)
        cl.set_leaf_u(u_arr)
        pin_solar_core(cl)

    def add_planet_knot(cl):
        """Add planet winding knot (1/r BFS, additive) on top of existing lam field."""
        pos = np.array(cl.leaf_pos)
        dist = np.linalg.norm(pos - planet_centre, axis=1)
        j_origin = int(np.argmin(dist))

        nb_g = np.array(cl.leaf_nb)
        leaf_g = np.array(cl.leaf_global_indices)
        g_to_j = {int(g): j for j, g in enumerate(leaf_g)}

        # Convert lam_planet → chi_planet for sigma-space 1/r profile
        chi_planet = args.lam_planet / (1.0 + args.lam_planet)
        sigma = 1.0 / (1.0 + np.array(cl.leaf_lam))
        sigma[j_origin] = max(L_MIN, sigma[j_origin] - chi_planet)

        q = deque([(j_origin, 0)])
        visited = {j_origin: 0}
        while q:
            j, r = q.popleft()
            for k in range(14):
                g_nb = int(nb_g[j, k])
                if g_nb < 0:
                    continue
                j_nb = g_to_j.get(g_nb, -1)
                if j_nb < 0 or j_nb in visited:
                    continue
                r_nb = r + 1
                d_chi = chi_planet / r_nb
                if d_chi < 1e-5:
                    continue
                visited[j_nb] = r_nb
                q.append((j_nb, r_nb))
                sigma[j_nb] = max(L_MIN, sigma[j_nb] - d_chi)

        cl.set_leaf_lam(1.0 / sigma - 1.0)
        return j_origin

    def setup_lam(cl):
        """
        lam=0 for sun interior and planet knot — active cells, full sigma.
        lam=lam_background for vacuum — slows u propagation.
        """
        pos = np.array(cl.leaf_pos)
        r = np.linalg.norm(pos, axis=1)
        lam_arr = np.array(cl.leaf_lam)
        lam_bg = sun_lam_background(pos)
        lam_exc = np.maximum(lam_arr - lam_bg, 0.0)

        lam = np.full(cl.n_leaves, args.lam_background, dtype=np.float64)
        lam[r <= args.R_sun * 1.2] = 0  # sun interior + corona fringe
        thresh = lam_exc.max() * 0.05
        lam[lam_exc >= thresh] = 0  # planet knot cells
        cl.set_leaf_lam(lam)

    # ---------------------------------------------------------------------------
    # Tangential u-asymmetry — initial condition for moving knot
    # ---------------------------------------------------------------------------

    def apply_tangential_u(cl):
        """
        U-asymmetry initialisation for tangential lam-knot motion.

        A phi kick decays in one step as the u-gradient reasserts equilibrium phi.
        A u-asymmetry persists: pass 2 continuously recomputes phi from the u
        gradient, so a tangential u-gradient drives sustained tangential phi →
        sustained du → sustained dell → knot motion.

        For motion in +t_hat: lower u at leading edge, higher u at trailing edge.
        Pass 2 converts this to tangential phi each step, driving chi to increase
        at the front and decrease at the rear, advancing the centroid.

        v_tangential should be close to the target orbital velocity in DST
        cell/time units: v_circ = sqrt(alpha × u_c × R_c / r_orbit).
        """
        r_len = np.linalg.norm(planet_centre)
        if r_len < 1e-9 or args.v_tangential == 0.0:
            return
        r_hat = planet_centre / r_len
        t_hat = np.cross(orbit_normal, r_hat)
        t_len = np.linalg.norm(t_hat)
        if t_len < 1e-9:
            return
        t_hat /= t_len

        pos = np.array(cl.leaf_pos)
        lam_arr = np.array(cl.leaf_lam)
        lam_bg = sun_lam_background(pos)
        lam_exc = np.maximum(lam_arr - lam_bg, 0.0)
        ce_max = lam_exc.max()
        if ce_max < 1e-12:
            return
        weight = lam_exc / ce_max

        tang_disp = (pos - planet_centre) @ t_hat

        knot_mask = lam_exc > ce_max * 0.1
        s = (
            float(np.std(np.linalg.norm((pos - planet_centre)[knot_mask], axis=1)))
            if knot_mask.any()
            else 1.0
        )
        s = max(s, 0.5)

        u = np.array(cl.leaf_u)
        # Leading (tang_disp>0): u↓ → outward phi → du>0 → ell↓ → chi↑
        # Trailing (tang_disp<0): u↑ → inward phi  → du<0 → ell↑ → chi↓
        u += args.v_tangential * (-tang_disp / s) * weight
        cl.set_leaf_u(u)

    # ---------------------------------------------------------------------------
    # Lam-excess centroid (tide-locked centre of mass)
    # ---------------------------------------------------------------------------

    def planet_lam_centroid(cl, last_centre=None):
        pos = np.array(cl.leaf_pos)
        lam_arr = np.array(cl.leaf_lam)
        lam_bg = sun_lam_background(pos)
        lam_exc = np.maximum(lam_arr - lam_bg, 0.0)
        centre = last_centre if last_centre is not None else planet_centre
        r_sun = np.linalg.norm(pos, axis=1)
        dist = np.linalg.norm(pos - centre, axis=1)
        window = max(args.r_orbit * 0.6, 5.0)
        mask = (dist < window) & (r_sun > args.R_sun * 1.5)
        w = lam_exc[mask]
        wsum = float(w.sum())
        if wsum < 1e-20:
            return np.full(3, float("nan"))
        return (w[:, None] * pos[mask]).sum(axis=0) / wsum

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
    info.max_level = args.max_level
    cl.info = info

    if args.resume:
        alat = _lat.AdaptiveLattice.load(args.resume)
        alat.set_n_threads(args.threads)
        cl = alat[0]
        print(f"resumed from {args.resume}  N={cl.n_leaves}")
    else:
        apply_solar_lam(cl)

        if args.max_level > 0:
            split_r_sun = args.R_sun * 1.5  # refine sun interior + corona
            split_r_planet = max(
                (args.lam_planet / (1.0 + args.lam_planet)) * 20.0, D1 * 4.0
            )
            for level in range(args.max_level):
                pos = np.array(cl.leaf_pos)
                levs = np.array(cl.leaf_level, dtype=int)
                idxs = np.array(cl.leaf_global_indices, dtype=np.int32)
                at_level = levs == level
                to_split = at_level & (np.linalg.norm(pos, axis=1) < split_r_sun)
                to_split |= at_level & (
                    np.linalg.norm(pos - planet_centre, axis=1) < split_r_planet
                )
                gs = np.ascontiguousarray(idxs[to_split], dtype=np.int32)
                d1_l = D1 / float(1 << level)
                n = alat.split_batch(0, gs, d1_l)
                apply_solar_lam(cl)
                print(f"  split {level}→{level+1}: {n} cells, now {cl.n_leaves} leaves")

        # Sponge BC: soft absorbing boundary that maintains the u gradient from
        # the BH core.  Target=0 with small weight — the planet at r=20 in R=40
        # domain contributes only ~0.015 to boundary u, so ghost-knot effect is
        # negligible.  Galactic remote-lattice correction stacks on top.
        if args.galactic_lam > 0.0:
            R_REMOTE = 1e6
            chi_gal = args.galactic_lam / (1.0 + args.galactic_lam)
            flux = chi_gal * args.R / R_REMOTE
            axis = galactic_axis / np.linalg.norm(galactic_axis)
            bc = _lat.BoundaryCondition.sponge(target=0.0, weight=args.sponge_weight)
            bc.remote_flux = flux
            bc.remote_axis = list(axis)
            cl.set_bc(bc)
            print(f"BC sponge+remote  : weight={args.sponge_weight}  flux={flux:.3e}")
        else:
            cl.set_bc(
                _lat.BoundaryCondition.sponge(target=0.0, weight=args.sponge_weight)
            )
            print(f"BC sponge         : target=0.0  weight={args.sponge_weight}")

        # Init u to variable-sigma Laplace steady state (BH source, sponge drain).
        # This is the correct DST equilibrium — not the Poisson potential.
        init_u_steady(cl)
        print(f"solar core        : u_core={args.u_core}  R_core={args.R_core}")

        if args.v_rotation > 0.0:
            apply_toroidal_rotation(
                cl,
                centre=[0.0, 0.0, 0.0],
                R_core=args.R_sun,
                amplitude=args.v_rotation,
                spin_axis=args.spin_axis,
            )
            print(
                f"Sun rotation      : v_rot={args.v_rotation:.3g}  axis={args.spin_axis}"
            )

        if not args.no_planet:
            add_planet_knot(cl)
            setup_lam(cl)
            apply_tangential_u(cl)
        else:
            pos = np.array(cl.leaf_pos)
            r = np.linalg.norm(pos, axis=1)
            lam = np.full(cl.n_leaves, args.lam_background, dtype=np.float64)
            lam[r <= args.R_sun * 1.2] = 0
            cl.set_leaf_lam(lam)
            print("no-planet mode: sun interior active, background frozen")

        if args.save_prefix:
            os.makedirs(os.path.dirname(args.save_prefix) or ".", exist_ok=True)
            ckpt = f"{args.save_prefix}_init.h5"
            alat.save(ckpt)
            print(f"saved → {ckpt}")

    # ---------------------------------------------------------------------------
    # Run
    # ---------------------------------------------------------------------------
    dt = compute_dt(cl, D1=D1)

    live_view = None
    if args.live:
        from vis.substrate.orbit_live import OrbitLive

        live_view = OrbitLive(
            cl,
            planet_centre=planet_centre,
            R_central=args.R_sun,
            chi_central=args.lam_sun / (1.0 + args.lam_sun),
            orbit_normal=orbit_normal,
            galactic_axis=galactic_axis,
            every=args.live_every,
            grid_n=args.live_grid_n,
        )

    print(f"\n=== orbit sandbox  N={cl.n_leaves}  dt={dt:.3e} ===")
    print(
        f"{'step':>8}  {'r_xy':>8}  {'theta':>7}  {'Lz':>10}  {'|L|':>10}  {'L/L0':>8}"
    )

    # Early-exit thresholds
    L0 = None  # angular momentum at first valid measurement
    nan_streak = 0
    STOP_REASONS = []

    last_centre = None
    r_prev = None
    for s in range(1, args.n_steps + 1):
        alat.step(0, dt)
        if live_view is not None:
            live_view.update(cl, s)
        if s % args.report == 0:
            r_curr = planet_lam_centroid(cl, last_centre)
            Lz = Lmag = float("nan")
            L_ratio = float("nan")
            if r_prev is not None and np.all(np.isfinite(r_curr)):
                v_cm = (r_curr - r_prev) / (args.report * dt)
                L = np.cross(r_curr, v_cm)
                Lz = float(L[2])
                Lmag = float(np.linalg.norm(L))
                if live_view is not None:
                    live_view.record_L(s, L)
                if L0 is None and Lmag > 1e-12:
                    L0 = Lmag
                if L0 is not None:
                    L_ratio = Lmag / L0

            r_xy = (
                float(np.linalg.norm(r_curr[:2]))
                if np.all(np.isfinite(r_curr))
                else float("nan")
            )
            theta = (
                float(np.degrees(np.arctan2(r_curr[1], r_curr[0])))
                if np.all(np.isfinite(r_curr))
                else float("nan")
            )
            print(
                f"{s:8d}  {r_xy:8.3f}  {theta:7.2f}  {Lz:10.4f}  {Lmag:10.4f}  {L_ratio:8.3f}",
                flush=True,
            )

            if np.all(np.isfinite(r_curr)):
                last_centre = r_curr.copy()
                nan_streak = 0
            else:
                nan_streak += 1

            r_prev = r_curr.copy() if np.all(np.isfinite(r_curr)) else r_prev

            # --- early-exit checks ---
            if nan_streak >= 3:
                STOP_REASONS.append(
                    "planet knot dispersed (3 consecutive NaN centroids)"
                )
                break
            if np.isfinite(r_xy):
                if r_xy < args.R_sun * 1.5:
                    STOP_REASONS.append(
                        f"planet fell into sun (r_xy={r_xy:.2f} < {args.R_sun*1.5:.2f})"
                    )
                    break
                if r_xy > args.R * 0.85:
                    STOP_REASONS.append(
                        f"planet escaped domain (r_xy={r_xy:.2f} > {args.R*0.85:.2f})"
                    )
                    break
            if L0 is not None and np.isfinite(L_ratio):
                if L_ratio < 0.1:
                    STOP_REASONS.append(
                        f"angular momentum collapsed (L/L0={L_ratio:.3f})"
                    )
                    break
                if L_ratio > 10.0:
                    STOP_REASONS.append(
                        f"angular momentum diverged (L/L0={L_ratio:.3f})"
                    )
                    break

            if args.save_prefix and args.save_every and s % args.save_every == 0:
                ckpt = f"{args.save_prefix}_s{s:08d}.h5"
                alat.save(ckpt)
                print(f"  saved → {ckpt}")

    if STOP_REASONS:
        print(f"\n*** early exit: {STOP_REASONS[0]} ***", flush=True)

    if args.save_prefix and args.n_steps > 0:
        ckpt = f"{args.save_prefix}_final.h5"
        alat.save(ckpt)
        print(f"saved final → {ckpt}")

    print("\ndone")


if __name__ == "__main__":
    main()
