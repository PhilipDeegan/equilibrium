"""
Test A: does a MOVING charge generate a magnetic-like (curl(phi)) field,
with the correct chirality (sign flips with charge sign) and Biot-Savart-
like transverse orientation (circulating around the velocity axis, not
radial)?

IC: same lam-carrying "proton knot" exact-fixed-point construction as
exp/2026/07/11/19_45_proton_knot_translation/probe_proton_translation.py
(u=0, phi=0, lam(r) from set_winding -- verified elsewhere to be an exact
fixed point of the free operator set). On top of that we impose a static
"charge" -- a u-monopole via BFS hop-distance 1/r from the same origin cell
(same construction as tst/substrate/adaptive/adaptive_flux_tube.py's
set_u_background, adapted to this lattice's flat lam/nb arrays instead of
an AMR cluster object), scaled by Q. Q=0 isolates whatever curl(phi) the
knot's own translation produces with NO charge (baseline/noise floor);
Q=+1/-1 isolate the charge's contribution and its chirality.

Velocity is imparted via the same battery-shell pressure-tilt kick as the
reference script (external u gradient at r in (r_inner-2, r_inner], pinned
for n_kick steps, then released to coast).

curl(phi) = B_vec is computed via the vectorized _plaquette_vectorized/
b_vec_vectorized pattern from
exp/2026/08/18/18_45_poynting_far_zone_rerun/probe_poynting_dipole_compare.py
(itself the vectorized form of equilibrium.substrate.mhd._b_vec), evaluated
on the final (post-coast) phi field for each of the three Q runs, sampled on
a FIXED (Q-independent) shell of cells around the knot's final position.
"""

import argparse
import sys
from collections import deque

import numpy as np

sys.path.insert(0, ".")
from equilibrium.substrate.lattice import DampedLattice
from equilibrium.substrate.mhd import SQ_PLAQUETTES

_AX = {"x": 0, "y": 1, "z": 2}


def _plaquette_vectorized(phi_T, nb_j, faces):
    N = phi_T.shape[0]
    c0 = np.arange(N)
    ka, kb, kc, kd = faces
    c1 = nb_j[c0, ka]
    valid = c1 >= 0
    c1s = np.where(valid, c1, 0)
    c2 = nb_j[c1s, kb]
    valid &= c2 >= 0
    c2s = np.where(c2 >= 0, c2, 0)
    c3 = nb_j[c2s, kc]
    valid &= c3 >= 0
    c3s = np.where(c3 >= 0, c3, 0)
    close = nb_j[c3s, kd]
    valid &= close == c0
    term = phi_T[c0, ka] + phi_T[c1s, kb] + phi_T[c2s, kc] + phi_T[c3s, kd]
    return np.where(valid, term, 0.0)


def b_vec_vectorized(phi_T, nb_j):
    Bx = _plaquette_vectorized(phi_T, nb_j, SQ_PLAQUETTES["x"])
    By = _plaquette_vectorized(phi_T, nb_j, SQ_PLAQUETTES["y"])
    Bz = _plaquette_vectorized(phi_T, nb_j, SQ_PLAQUETTES["z"])
    return np.stack([Bx, By, Bz], axis=1)


def weighted_com(pos, w):
    total = w.sum()
    if total < 1e-30:
        return np.zeros(3), 0.0
    return (pos * w[:, None]).sum(axis=0) / total, float(total)


def set_u_hop_monopole(lat, origin, u_origin):
    """BFS hop-distance 1/r u background from origin -- flat-array analogue
    of adaptive_flux_tube.set_u_background, adapted to lat.nb/(N,14)."""
    nb = lat.nb
    N = lat.N
    u = np.zeros(N, dtype=np.float64)
    u[origin] = u_origin
    visited = np.zeros(N, dtype=bool)
    visited[origin] = True
    queue = deque([(origin, 0)])
    while queue:
        j, r = queue.popleft()
        for k in range(14):
            nb_j = int(nb[j, k])
            if nb_j < 0 or visited[nb_j]:
                continue
            visited[nb_j] = True
            r_nb = r + 1
            u[nb_j] = u_origin / r_nb
            queue.append((nb_j, r_nb))
    lat.u = lat.u + u


def build_lattice(args):
    lat = DampedLattice(
        radius_cells=args.R, sponge_fraction=args.sponge, damp_max=args.damp_max
    )
    pos = lat.pos
    r = np.linalg.norm(pos, axis=1)
    origin = int(np.argmin(r))
    lat.set_winding(lat.tiles[origin], chi_0=args.chi0, profile=args.profile)
    if args.Q != 0.0:
        set_u_hop_monopole(lat, origin, args.Q * args.charge_amp)
    return lat, pos, r, origin


def run_one(args):
    ax_i = _AX[args.kick_axis]
    lat, pos, r_all, origin = build_lattice(args)
    dt = lat.stable_dt()
    r_max = float(r_all.max())
    r_inner = r_max * (1.0 - args.sponge)
    battery = (r_all > r_inner - 2.0) & (r_all <= r_inner)
    tilt = args.gradient * pos[:, ax_i]

    def phys_step():
        lat.step(dt, n_threads=args.n_threads)
        if args.cross_scale > 0.0:
            lat.step_cross(dt, args.cross_scale, n_threads=args.n_threads)
        if args.force_scale > 0.0:
            lat.step_lam_force(dt, args.force_scale)

    print(
        f"[Q={args.Q:+.1f}] N={lat.N}  dt={dt:.4e}  max|u|0={np.abs(lat.u).max():.3e}",
        flush=True,
    )

    coms = []

    def record(step, phase):
        com, mass = weighted_com(pos, lat.lam)
        coms.append((step * dt, phase, com))
        if step % args.report == 0:
            print(
                f"  [Q={args.Q:+.1f}] {step:6d} {phase:>5}  "
                f"com=({com[0]:+7.3f},{com[1]:+7.3f},{com[2]:+7.3f})  "
                f"max|u|={np.abs(lat.u).max():.3e}  max|phi|={np.abs(lat.phi).max():.3e}",
                flush=True,
            )

    step = 0
    record(step, "base")
    for s in range(1, args.n_base + 1):
        phys_step()
        step += 1
    record(step, "base")

    lat.u = lat.u + tilt
    lat.fixed_u[battery] = True
    for s in range(1, args.n_kick + 1):
        phys_step()
        step += 1
        if step % args.report == 0:
            record(step, "kick")
    record(step, "kick")

    lat.fixed_u[battery] = False
    lat.u = lat.u - tilt
    for s in range(1, args.n_coast + 1):
        phys_step()
        step += 1
        if step % args.report == 0:
            record(step, "coast")
    record(step, "coast")

    def fit_v(phase):
        rows = [(t, c[ax_i]) for t, p, c in coms if p == phase]
        if len(rows) < 3:
            return 0.0
        rows = rows[len(rows) // 2 :]
        t = np.array([r[0] for r in rows])
        x = np.array([r[1] for r in rows])
        return float(np.polyfit(t, x, 1)[0])

    v_coast = fit_v("coast")
    com_final, mass_final = weighted_com(pos, lat.lam)
    B_final = b_vec_vectorized(lat.phi, lat.nb)

    print(
        f"[Q={args.Q:+.1f}] v_coast={v_coast:+.4e}  com_final="
        f"({com_final[0]:+.3f},{com_final[1]:+.3f},{com_final[2]:+.3f})  "
        f"mass_final={mass_final:.4f}",
        flush=True,
    )

    return dict(
        Q=args.Q,
        pos=pos,
        com_final=com_final,
        v_coast=v_coast,
        B_final=B_final,
        ax_i=ax_i,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--R", type=float, default=20.0)
    p.add_argument("--chi0", type=float, default=0.9)
    p.add_argument("--profile", type=str, default="1/r2")
    p.add_argument("--charge-amp", type=float, default=0.05)
    p.add_argument("--cross-scale", type=float, default=0.3)
    p.add_argument("--force-scale", type=float, default=0.3)
    p.add_argument("--gradient", type=float, default=5e-4)
    p.add_argument("--kick-axis", type=str, default="z")
    p.add_argument("--n-base", type=int, default=1000)
    p.add_argument("--n-kick", type=int, default=6000)
    p.add_argument("--n-coast", type=int, default=4000)
    p.add_argument("--report", type=int, default=250)
    p.add_argument("--sponge", type=float, default=0.15)
    p.add_argument("--damp-max", type=float, default=0.003)
    p.add_argument("--n-threads", type=int, default=8)
    p.add_argument("--shell-rmin", type=float, default=3.0)
    p.add_argument("--shell-rmax", type=float, default=9.0)
    p.add_argument(
        "--out-npz",
        type=str,
        default="exp/2026/08/20/17_32_moving_charge_curl_test/result.npz",
    )
    p.add_argument(
        "--out-png",
        type=str,
        default="exp/2026/08/20/17_32_moving_charge_curl_test/chirality.png",
    )
    args = p.parse_args()

    results = {}
    for Q in (0.0, 1.0, -1.0):
        args.Q = Q
        results[Q] = run_one(args)

    pos = results[0.0]["pos"]
    ax_i = results[0.0]["ax_i"]
    perp_axes = [i for i in range(3) if i != ax_i]
    com0 = results[0.0]["com_final"]

    # fixed (Q-independent) sampling shell around the Q=0 run's final COM,
    # away from the knot core (shell_rmin) and away from the sponge
    # (shell_rmax bound is generous relative to R -- checked against args.R)
    rel = pos - com0[None, :]
    r_perp = np.linalg.norm(rel[:, perp_axes], axis=1)
    z_rel = np.abs(rel[:, ax_i])
    shell = (
        (r_perp > args.shell_rmin)
        & (r_perp < args.shell_rmax)
        & (z_rel < args.shell_rmax)
    )
    print(f"\nshell cells: {shell.sum()}")

    B0 = results[0.0]["B_final"][shell]
    Bp = results[1.0]["B_final"][shell]
    Bm = results[-1.0]["B_final"][shell]

    diff_p = Bp - B0
    diff_m = Bm - B0

    noise = np.sqrt(np.mean(B0**2))
    sig_p = np.sqrt(np.mean(diff_p**2))
    sig_m = np.sqrt(np.mean(diff_m**2))
    print(f"noise floor rms|B0|          = {noise:.4e}")
    print(
        f"signal rms|B(Q=+1)-B(Q=0)|   = {sig_p:.4e}  ratio={sig_p/max(noise,1e-300):.2f}"
    )
    print(
        f"signal rms|B(Q=-1)-B(Q=0)|   = {sig_m:.4e}  ratio={sig_m/max(noise,1e-300):.2f}"
    )

    flat_p = diff_p.ravel()
    flat_m = diff_m.ravel()
    cos_sim = float(
        np.dot(flat_p, flat_m)
        / (np.linalg.norm(flat_p) * np.linalg.norm(flat_m) + 1e-300)
    )
    print(
        f"chirality: cos_sim(diff_p, diff_m) = {cos_sim:+.4f}  (expect ~ -1 if sign-flipping)"
    )

    # cylindrical decomposition around the velocity axis through com0
    v_hat = np.zeros(3)
    v_hat[ax_i] = (
        np.sign(results[1.0]["v_coast"]) if results[1.0]["v_coast"] != 0 else 1.0
    )
    rel_shell = rel[shell]
    r_perp_vec = rel_shell.copy()
    r_perp_vec[:, ax_i] = 0.0
    r_perp_norm = np.linalg.norm(r_perp_vec, axis=1, keepdims=True)
    r_hat = np.divide(
        r_perp_vec, r_perp_norm, out=np.zeros_like(r_perp_vec), where=r_perp_norm > 1e-9
    )
    phi_hat = np.cross(np.broadcast_to(v_hat, r_hat.shape), r_hat)

    def project(diff):
        along_v = (diff * v_hat[None, :]).sum(axis=1)
        along_r = (diff * r_hat).sum(axis=1)
        along_phi = (diff * phi_hat).sum(axis=1)
        return (
            np.sqrt(np.mean(along_v**2)),
            np.sqrt(np.mean(along_r**2)),
            np.sqrt(np.mean(along_phi**2)),
        )

    av_p, ar_p, aphi_p = project(diff_p)
    print(
        f"diff_p decomposition: |along_v|={av_p:.4e}  |along_r|={ar_p:.4e}  "
        f"|along_phi(azimuthal)|={aphi_p:.4e}"
    )
    av_m, ar_m, aphi_m = project(diff_m)
    print(
        f"diff_m decomposition: |along_v|={av_m:.4e}  |along_r|={ar_m:.4e}  "
        f"|along_phi(azimuthal)|={aphi_m:.4e}"
    )

    np.savez(
        args.out_npz,
        pos_shell=pos[shell],
        com0=com0,
        ax_i=ax_i,
        B0=B0,
        Bp=Bp,
        Bm=Bm,
        diff_p=diff_p,
        diff_m=diff_m,
        noise=noise,
        sig_p=sig_p,
        sig_m=sig_m,
        cos_sim=cos_sim,
        av_p=av_p,
        ar_p=ar_p,
        aphi_p=aphi_p,
        av_m=av_m,
        ar_m=ar_m,
        aphi_m=aphi_m,
    )
    print(f"saved {args.out_npz}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pax0, pax1 = perp_axes
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, diff, title in (
        (axes[0], diff_p, "Q=+1 minus Q=0"),
        (axes[1], diff_m, "Q=-1 minus Q=0"),
    ):
        x = rel_shell[:, pax0]
        y = rel_shell[:, ax_i]
        u_comp = diff[:, pax0]
        v_comp = diff[:, ax_i]
        ax.quiver(x, y, u_comp, v_comp, angles="xy", scale_units="xy")
        ax.set_title(title)
        ax.set_xlabel(f"axis {pax0} (relative to final COM)")
        ax.set_ylabel(f"axis {ax_i} (velocity axis, relative to final COM)")
        ax.set_aspect("equal")
    fig.suptitle("curl(phi) wake difference from Q=0 baseline, in-plane component")
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=130)
    print(f"saved {args.out_png}")


if __name__ == "__main__":
    main()
