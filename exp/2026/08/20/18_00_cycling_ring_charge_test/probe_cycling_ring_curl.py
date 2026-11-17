"""
Test A2: does a phi RING-MODE CYCLING within a knot (doc/dst.md's literal
prescribed magnetism mechanism) produce a chirality-sensitive, azimuthal
curl(phi) wake -- first at rest (cycling alone), then under translation
(cycling + coasting, directly comparable to Test A's failed static-monopole
result)?

Reuses Test A's machinery verbatim (proton-knot IC, battery-shell kick,
b_vec_vectorized curl(phi) measurement, chirality via cos_sim, cylindrical
along_v/along_r/along_phi decomposition) -- see
exp/2026/08/20/17_32_moving_charge_curl_test/probe_moving_charge_curl.py.
Only the "charge" representation changes: instead of a static u-monopole
(Q), we trace a small toroidal ring at the knot's core and seed a free
(unfrozen) traveling-wave circulation mode on it via
init_toroidal_winding(..., frozen=False) -- amplitude sign W=+1/-1/0 plays
the role Q played in Test A.
"""

import argparse
import sys
from collections import deque

import numpy as np

sys.path.insert(0, ".")
from equilibrium.substrate.lattice import DampedLattice
from equilibrium.substrate.mhd import SQ_PLAQUETTES
from equilibrium.substrate.flux_tube import (
    find_toroidal_ring,
    init_toroidal_winding,
    resonant_freq,
)

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


def field_energy(lat):
    return float(0.5 * np.sum(lat.u**2) + 0.5 * np.sum(lat.phi**2))


def build_lattice(args):
    lat = DampedLattice(
        radius_cells=args.R, sponge_fraction=args.sponge, damp_max=args.damp_max
    )
    pos = lat.pos
    r = np.linalg.norm(pos, axis=1)
    origin = int(np.argmin(r))
    lat.set_winding(lat.tiles[origin], chi_0=args.chi0, profile=args.profile)

    ring_tiles, ring_faces = find_toroidal_ring(
        lat, axis=args.kick_axis, mode="minimal"
    )
    if ring_tiles is None:
        print("ERROR: no ring found")
        sys.exit(1)

    omega = resonant_freq(len(ring_tiles), face_type="sq")
    if args.W != 0.0:
        init_toroidal_winding(
            lat, ring_tiles, ring_faces, amplitude=args.W * args.ring_amp, frozen=False
        )

    return lat, pos, r, origin, ring_tiles, omega


def run_one(args):
    ax_i = _AX[args.kick_axis]
    lat, pos, r_all, origin, ring_tiles, omega = build_lattice(args)
    dt = lat.stable_dt()
    period = 2.0 * np.pi / omega
    n_rest = int(round(args.n_periods_rest * period / dt))
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
        f"[W={args.W:+.1f}] N={lat.N}  dt={dt:.4e}  omega={omega:.4f}  period={period:.2f}  "
        f"n_rest={n_rest}  H0={field_energy(lat):.4e}",
        flush=True,
    )

    coms = []

    def record(step, phase):
        com, mass = weighted_com(pos, lat.lam)
        coms.append((step * dt, phase, com))
        if step % args.report == 0:
            print(
                f"  [W={args.W:+.1f}] {step:6d} {phase:>5}  "
                f"com=({com[0]:+7.3f},{com[1]:+7.3f},{com[2]:+7.3f})  "
                f"max|u|={np.abs(lat.u).max():.3e}  max|phi|={np.abs(lat.phi).max():.3e}  "
                f"H={field_energy(lat):.4e}",
                flush=True,
            )

    step = 0
    record(step, "base")
    for s in range(1, n_rest + 1):
        phys_step()
        step += 1
        if step % args.report == 0:
            record(step, "base")
    record(step, "base")

    H_rest = field_energy(lat)
    B_rest = b_vec_vectorized(lat.phi, lat.nb)
    com_rest, _ = weighted_com(pos, lat.lam)

    lat.u = lat.u + tilt
    lat.fixed_u[battery] = True
    for s in range(1, args.n_kick + 1):
        phys_step()
        step += 1
        if step % args.report == 0:
            record(step, "kick")
    record(step, "kick")
    H_kick = field_energy(lat)

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
    B_coast = b_vec_vectorized(lat.phi, lat.nb)
    H_coast = field_energy(lat)

    print(
        f"[W={args.W:+.1f}] v_coast={v_coast:+.4e}  com_final="
        f"({com_final[0]:+.3f},{com_final[1]:+.3f},{com_final[2]:+.3f})  "
        f"mass_final={mass_final:.4f}  H_rest={H_rest:.4e}  H_kick={H_kick:.4e}  H_coast={H_coast:.4e}",
        flush=True,
    )

    return dict(
        W=args.W,
        pos=pos,
        com_rest=com_rest,
        com_final=com_final,
        v_coast=v_coast,
        B_rest=B_rest,
        B_coast=B_coast,
        ax_i=ax_i,
        H_rest=H_rest,
        H_kick=H_kick,
        H_coast=H_coast,
    )


def analyse(
    results, pos, ax_i, com0_rest, com0_final, shell_rmin, shell_rmax, phase_label
):
    perp_axes = [i for i in range(3) if i != ax_i]
    com0 = com0_rest if phase_label == "rest" else com0_final
    rel = pos - com0[None, :]
    r_perp = np.linalg.norm(rel[:, perp_axes], axis=1)
    z_rel = np.abs(rel[:, ax_i])
    shell = (r_perp > shell_rmin) & (r_perp < shell_rmax) & (z_rel < shell_rmax)
    print(f"\n--- {phase_label} phase: shell cells = {shell.sum()} ---")

    key = "B_rest" if phase_label == "rest" else "B_coast"
    B0 = results[0.0][key][shell]
    Bp = results[1.0][key][shell]
    Bm = results[-1.0][key][shell]

    diff_p = Bp - B0
    diff_m = Bm - B0
    noise = np.sqrt(np.mean(B0**2))
    sig_p = np.sqrt(np.mean(diff_p**2))
    sig_m = np.sqrt(np.mean(diff_m**2))
    ratio_p = "N/A (baseline exactly zero)" if noise == 0.0 else f"{sig_p / noise:.2f}"
    ratio_m = "N/A (baseline exactly zero)" if noise == 0.0 else f"{sig_m / noise:.2f}"
    print(f"[{phase_label}] noise floor rms|B0|        = {noise:.4e}")
    print(f"[{phase_label}] signal rms|B(W=+1)-B(W=0)|  = {sig_p:.4e}  ratio={ratio_p}")
    print(f"[{phase_label}] signal rms|B(W=-1)-B(W=0)|  = {sig_m:.4e}  ratio={ratio_m}")

    flat_p = diff_p.ravel()
    flat_m = diff_m.ravel()
    cos_sim = float(
        np.dot(flat_p, flat_m)
        / (np.linalg.norm(flat_p) * np.linalg.norm(flat_m) + 1e-300)
    )
    print(
        f"[{phase_label}] chirality: cos_sim(diff_p, diff_m) = {cos_sim:+.4f}  (expect ~ -1 if sign-flipping)"
    )

    v_hat = np.zeros(3)
    if phase_label == "rest":
        v_hat[ax_i] = 1.0
    else:
        v_coast_sign = (
            np.sign(results[1.0]["v_coast"]) if results[1.0]["v_coast"] != 0 else 1.0
        )
        v_hat[ax_i] = v_coast_sign
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
        f"[{phase_label}] diff_p decomposition: |along_v|={av_p:.4e}  |along_r|={ar_p:.4e}  "
        f"|along_phi(azimuthal)|={aphi_p:.4e}"
    )
    av_m, ar_m, aphi_m = project(diff_m)
    print(
        f"[{phase_label}] diff_m decomposition: |along_v|={av_m:.4e}  |along_r|={ar_m:.4e}  "
        f"|along_phi(azimuthal)|={aphi_m:.4e}"
    )

    return dict(
        shell=shell,
        rel_shell=rel_shell,
        diff_p=diff_p,
        diff_m=diff_m,
        cos_sim=cos_sim,
        sig_p=sig_p,
        sig_m=sig_m,
        noise=noise,
        av_p=av_p,
        ar_p=ar_p,
        aphi_p=aphi_p,
        av_m=av_m,
        ar_m=ar_m,
        aphi_m=aphi_m,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--R", type=float, default=20.0)
    p.add_argument("--chi0", type=float, default=0.9)
    p.add_argument("--profile", type=str, default="1/r2")
    p.add_argument("--ring-amp", type=float, default=0.05)
    p.add_argument("--cross-scale", type=float, default=0.3)
    p.add_argument("--force-scale", type=float, default=0.3)
    p.add_argument("--gradient", type=float, default=5e-4)
    p.add_argument("--kick-axis", type=str, default="z")
    p.add_argument("--n-periods-rest", type=float, default=5.0)
    p.add_argument("--n-kick", type=int, default=6000)
    p.add_argument("--n-coast", type=int, default=4000)
    p.add_argument("--report", type=int, default=500)
    p.add_argument("--sponge", type=float, default=0.15)
    p.add_argument("--damp-max", type=float, default=0.003)
    p.add_argument("--n-threads", type=int, default=8)
    p.add_argument("--shell-rmin", type=float, default=3.0)
    p.add_argument("--shell-rmax", type=float, default=9.0)
    outdir = "exp/2026/08/20/18_00_cycling_ring_charge_test"
    p.add_argument("--out-npz", type=str, default=f"{outdir}/result.npz")
    p.add_argument("--out-png", type=str, default=f"{outdir}/chirality.png")
    args = p.parse_args()

    results = {}
    for W in (0.0, 1.0, -1.0):
        args.W = W
        results[W] = run_one(args)

    pos = results[0.0]["pos"]
    ax_i = results[0.0]["ax_i"]
    com0_rest = results[0.0]["com_rest"]
    com0_final = results[0.0]["com_final"]

    rest = analyse(
        results,
        pos,
        ax_i,
        com0_rest,
        com0_final,
        args.shell_rmin,
        args.shell_rmax,
        "rest",
    )
    coast = analyse(
        results,
        pos,
        ax_i,
        com0_rest,
        com0_final,
        args.shell_rmin,
        args.shell_rmax,
        "coast",
    )

    print("\n--- energy accounting (sanity, not primary pass/fail) ---")
    for W in (0.0, 1.0, -1.0):
        r = results[W]
        print(
            f"[W={W:+.1f}] H_rest={r['H_rest']:.4e}  H_kick={r['H_kick']:.4e}  H_coast={r['H_coast']:.4e}"
        )

    np.savez(
        args.out_npz,
        rest_cos_sim=rest["cos_sim"],
        rest_sig_p=rest["sig_p"],
        rest_sig_m=rest["sig_m"],
        rest_noise=rest["noise"],
        coast_cos_sim=coast["cos_sim"],
        coast_sig_p=coast["sig_p"],
        coast_sig_m=coast["sig_m"],
        coast_noise=coast["noise"],
        rest_av_p=rest["av_p"],
        rest_ar_p=rest["ar_p"],
        rest_aphi_p=rest["aphi_p"],
        coast_av_p=coast["av_p"],
        coast_ar_p=coast["ar_p"],
        coast_aphi_p=coast["aphi_p"],
    )
    print(f"saved {args.out_npz}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    perp_axes = [i for i in range(3) if i != ax_i]
    pax0 = perp_axes[0]
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for row, (data, phase_name) in enumerate(((rest, "rest"), (coast, "coast"))):
        rel_shell = data["rel_shell"]
        for col, (diff, title) in enumerate(
            (
                (data["diff_p"], f"{phase_name}: W=+1 minus W=0"),
                (data["diff_m"], f"{phase_name}: W=-1 minus W=0"),
            )
        ):
            ax = axes[row, col]
            x = rel_shell[:, pax0]
            y = rel_shell[:, ax_i]
            u_comp = diff[:, pax0]
            v_comp = diff[:, ax_i]
            ax.quiver(x, y, u_comp, v_comp, angles="xy", scale_units="xy")
            ax.set_title(title)
            ax.set_xlabel(f"axis {pax0}")
            ax.set_ylabel(f"axis {ax_i} (velocity axis)")
            ax.set_aspect("equal")
    fig.suptitle("Test A2: curl(phi) wake difference, ring cycling (rest vs coast)")
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=130)
    print(f"saved {args.out_png}")


if __name__ == "__main__":
    main()
