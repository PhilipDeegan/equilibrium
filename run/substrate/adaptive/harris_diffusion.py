#!/usr/bin/env python3
"""
Harris sheet reconnection: wave equation + explicit phi diffusion (resistivity).

Field evolution per step:
  1. alat.step()     — wave equation (C++):
                         pass 1: du  = -σ·Σ W_k·phi_k       (pressure driven by div-phi)
                         pass 2: dphi= -σ·W_k·(u_nb - u_i)  (field driven by ∇u)
  2. phi diffusion   — explicit resistivity on top:
                         dphi_k = η·σ·Σ W_j·(phi[k,nb_j] - phi[k,i])

The wave equation alone is ideal (frozen-in field lines, no reconnection).
The phi diffusion adds resistivity η that allows field lines to slip.
Together they give resistive MHD in DST form.

Use --eta-phi 0 to run the pure wave equation as a control.

Usage:
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/harris_diffusion.py
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/harris_diffusion.py \\
      --n-steps 10000 --eta-phi 0.05 --plot-dir tmp/harris_slow
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/harris_diffusion.py \\
      --eta-phi 0 --plot-dir tmp/harris_ideal   # pure wave equation, no resistivity
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from equilibrium.substrate.adaptive import wire_periodic_bcc

# ---------------------------------------------------------------------------
# BCC geometry
# ---------------------------------------------------------------------------
_HEX_DIR = np.array(
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
_SQ_DIR = np.array(
    [[-1, 0, 0], [+1, 0, 0], [0, -1, 0], [0, +1, 0], [0, 0, -1], [0, 0, +1]],
    dtype=np.float64,
)
_ALL_DIR = np.concatenate([_HEX_DIR, _SQ_DIR])
_W_FACE = np.array([3.0] * 8 + [1.0] * 6)
_B_NORM = 26.0

# ---------------------------------------------------------------------------
# IC parameters
# ---------------------------------------------------------------------------
LX, LY, LZ = 30.0, 30.0, 3.5
B0 = 0.05
LAM_W = 6.0
U_BG = 0.4
LAM_BG = 0.4

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--n-steps", type=int, default=5000)
parser.add_argument("--dt", type=float, default=0.05)
parser.add_argument(
    "--eta-phi",
    type=float,
    default=0.1,
    help="phi diffusion rate; 0 = pure wave equation (no resistivity)",
)
parser.add_argument("--perturb", type=float, default=0.1)
parser.add_argument("--report", type=int, default=500)
parser.add_argument("--plot-every", type=int, default=1000)
parser.add_argument("--plot-dir", type=str, default="tmp/harris_diff")
args = parser.parse_args()

PLOT_DIR = Path(args.plot_dir)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bx_profile(y, y0):
    return B0 * (-1.0 + np.tanh((y + y0) / LAM_W) - np.tanh((y - y0) / LAM_W))


def _reconstruct_B(phi):
    """phi (14,N) → B (N,3)."""
    return np.einsum("k,kn,kd->nd", _W_FACE, phi, _ALL_DIR) / _B_NORM


def _phi_diffuse(phi, nb, gid, g_to_j, max_g, sigma, dt, eta):
    """One explicit phi diffusion step. phi: (14,N). Returns updated (14,N)."""
    dphi = np.zeros_like(phi)
    N = phi.shape[1]
    for k in range(14):
        nb_g = nb[:, k]
        safe_g = np.where(nb_g >= 0, nb_g, gid)
        j = g_to_j[np.clip(safe_g, 0, max_g - 1)]
        safe_j = np.where(nb_g >= 0, j, np.arange(N, dtype=np.int32))
        dphi += _W_FACE[k] * sigma[None, :] * (phi[:, safe_j] - phi)
    return phi + dt * eta * dphi


def _by_xpt(pos, phi, y0):
    B = _reconstruct_B(phi)
    xm = (np.abs(pos[:, 0]) < LX * 0.15) & (np.abs(pos[:, 1] - y0) < LAM_W)
    return float(np.sqrt(np.mean(B[xm, 1] ** 2))) if xm.any() else 0.0


def _jz_null(pos, phi, y0):
    B = _reconstruct_B(phi)
    above = (pos[:, 1] > y0) & (pos[:, 1] < y0 + 1.5) & (np.abs(pos[:, 0]) < LX * 0.4)
    below = (pos[:, 1] < y0) & (pos[:, 1] > y0 - 1.5) & (np.abs(pos[:, 0]) < LX * 0.4)
    if not (above.any() and below.any()):
        return 0.0
    dy = float(pos[above, 1].mean() - pos[below, 1].mean())
    return (
        abs(float(B[above, 0].mean()) - float(B[below, 0].mean())) / dy
        if dy > 0
        else 0.0
    )


# ---------------------------------------------------------------------------
# Box builder
# ---------------------------------------------------------------------------


def _build_box():
    alat = _lat.AdaptiveLattice()
    cl = alat[0]
    _lat.build_bcc_box(cl, LX, LY, LZ, 1.0, u0=0.0, lam0=LAM_BG)
    alat.rebuild(0)
    wire_periodic_bcc(cl, 1.0 / np.sqrt(3.0))

    N = cl.n_leaves
    pos = np.array(cl.leaf_pos)
    y0 = LY / 4.0
    bx = _bx_profile(pos[:, 1], y0)

    u_ic = U_BG + (B0**2 - bx**2) / 2.0
    phi_ic = np.zeros((14, N), dtype=np.float64)
    for k in range(14):
        fd = _HEX_DIR[k] if k < 8 else _SQ_DIR[k - 8]
        phi_ic[k] += bx * fd[0]

    if args.perturb > 0:
        kx = 2.0 * np.pi / LX
        dB = args.perturb * B0
        for sign, y_null in ((+1, +y0), (-1, -y0)):
            yr = pos[:, 1] - y_null
            env = np.exp(-(yr**2) / LAM_W**2)
            kx_x = kx * (pos[:, 0] + LX / 2.0)
            bx_p = sign * (-2.0 * yr / LAM_W**2) * dB * np.cos(kx_x) * env
            by_p = sign * kx * dB * np.sin(kx_x) * env
            for k in range(14):
                fd = _HEX_DIR[k] if k < 8 else _SQ_DIR[k - 8]
                phi_ic[k] += bx_p * fd[0] + by_p * fd[1]

    cl.set_leaf_u(u_ic)
    cl.set_leaf_phi(phi_ic)

    nb = np.asarray(cl.leaf_nb, dtype=np.int32)
    gid = np.asarray(cl.leaf_global_indices, dtype=np.int32)
    max_g = int(nb.max()) + 1
    g_to_j = np.full(max_g, -1, dtype=np.int32)
    g_to_j[gid] = np.arange(N, dtype=np.int32)
    sigma = 1.0 / (1.0 + np.full(N, LAM_BG))

    return alat, cl, pos, y0, sigma, nb, gid, g_to_j, max_g


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _make_plot(pos, u, phi, y0, step, ts_step, ts_by, ts_jz, ts_phi_e):
    B = _reconstruct_B(phi)
    zmid = float(pos[:, 2].mean())
    mask = np.abs(pos[:, 2] - zmid) < 1.5
    px, py = pos[mask, 0], pos[mask, 1]

    xi = np.linspace(-LX / 2, LX / 2, 120)
    yi = np.linspace(-LY / 2, LY / 2, 120)
    Xi, Yi = np.meshgrid(xi, yi)

    def _grid(v):
        return griddata((px, py), v[mask], (Xi, Yi), method="linear")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"step={step}  t={step*args.dt:.1f}  η_φ={args.eta_phi}", fontsize=11)

    By_g = _grid(B[:, 1])
    lim = max(float(np.nanmax(np.abs(By_g))), 1e-9)
    im = axes[0].pcolormesh(
        Xi, Yi, By_g, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto"
    )
    plt.colorbar(im, ax=axes[0], label="By")
    for sign in (+1, -1):
        axes[0].axhline(sign * y0, color="k", lw=0.6, ls="--")
    axes[0].set_title("By (reconnecting component)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")

    u_g = _grid(u)
    im2 = axes[1].pcolormesh(Xi, Yi, u_g, cmap="viridis", shading="auto")
    plt.colorbar(im2, ax=axes[1], label="u")
    for sign in (+1, -1):
        axes[1].axhline(sign * y0, color="w", lw=0.6, ls="--")
    axes[1].set_title("u (pressure)")
    axes[1].set_xlabel("x")
    axes[1].set_aspect("equal")

    t_arr = np.array(ts_step) * args.dt
    if len(t_arr) > 1:
        axes[2].plot(t_arr, ts_by, label="By_xpt")
        ax2 = axes[2].twinx()
        ax2.plot(t_arr, ts_jz, color="tab:orange", ls="--")
        ax2.set_ylabel("Jz_null", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("By_xpt")
    axes[2].set_title("By_xpt (left)  Jz_null (right, dashed)")

    plt.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_DIR / f"frame_{step:06d}.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Building IC  (Lx={LX} Ly={LY} Lz={LZ})…", flush=True)
    alat, cl, pos, y0, sigma, nb, gid, g_to_j, max_g = _build_box()
    N = cl.n_leaves

    mode = f"wave + η_φ={args.eta_phi}" if args.eta_phi > 0 else "wave only (ideal)"
    tau_diff = (
        LAM_W**2 / (args.eta_phi * float(sigma[0]) * 26)
        if args.eta_phi > 0
        else float("inf")
    )
    print(
        f"  N={N}  DT={args.dt}  mode={mode}  "
        f"t_max={args.n_steps*args.dt:.0f}  τ_diff≈{tau_diff:.1f}",
        flush=True,
    )

    ts_step, ts_by, ts_jz, ts_phi_e = [], [], [], []

    phi0 = np.asarray(cl.leaf_phi, dtype=np.float64)
    by0 = _by_xpt(pos, phi0, y0)
    jz0 = _jz_null(pos, phi0, y0)
    pe0 = float(np.sum(phi0**2))
    print(
        f"\n{'step':>7}  {'t':>7}  {'By_xpt':>10}  {'Jz_null':>10}  {'phi_E':>12}",
        flush=True,
    )
    print(f"{'0':>7}  {'0.0':>7}  {by0:10.3e}  {jz0:10.3e}  {pe0:12.4e}", flush=True)
    ts_step.append(0)
    ts_by.append(by0)
    ts_jz.append(jz0)
    ts_phi_e.append(pe0)
    _make_plot(pos, np.asarray(cl.leaf_u), phi0, y0, 0, ts_step, ts_by, ts_jz, ts_phi_e)

    for step in range(1, args.n_steps + 1):
        alat.step(0, args.dt)

        if args.eta_phi > 0:
            phi = np.asarray(cl.leaf_phi, dtype=np.float64)
            phi = _phi_diffuse(
                phi, nb, gid, g_to_j, max_g, sigma, args.dt, args.eta_phi
            )
            cl.set_leaf_phi(phi)

        if step % args.report == 0:
            phi = np.asarray(cl.leaf_phi, dtype=np.float64)
            u = np.asarray(cl.leaf_u, dtype=np.float64)
            by = _by_xpt(pos, phi, y0)
            jz = _jz_null(pos, phi, y0)
            pe = float(np.sum(phi**2))
            t = step * args.dt
            print(
                f"{step:7d}  {t:7.1f}  {by:10.3e}  {jz:10.3e}  {pe:12.4e}", flush=True
            )
            ts_step.append(step)
            ts_by.append(by)
            ts_jz.append(jz)
            ts_phi_e.append(pe)
            if step % args.plot_every == 0:
                _make_plot(pos, u, phi, y0, step, ts_step, ts_by, ts_jz, ts_phi_e)

    phi = np.asarray(cl.leaf_phi, dtype=np.float64)
    u = np.asarray(cl.leaf_u, dtype=np.float64)
    _make_plot(pos, u, phi, y0, args.n_steps, ts_step, ts_by, ts_jz, ts_phi_e)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Harris reconnection — {mode}", fontsize=12)
    t_arr = np.array(ts_step) * args.dt

    axes[0, 0].plot(t_arr, ts_by)
    axes[0, 0].set_xlabel("t")
    axes[0, 0].set_ylabel("By_xpt")
    axes[0, 0].set_title("By at X-point  (↑ = reconnection)")

    axes[0, 1].plot(t_arr, ts_jz)
    axes[0, 1].set_xlabel("t")
    axes[0, 1].set_ylabel("Jz_null")
    axes[0, 1].set_title("Jz at null  (↓ = current sheet widens)")

    axes[1, 0].semilogy(t_arr, ts_phi_e)
    axes[1, 0].set_xlabel("t")
    axes[1, 0].set_ylabel("phi_E")
    axes[1, 0].set_title("Field energy (log)")

    by_arr = np.array(ts_by)
    axes[1, 1].plot(t_arr, by_arr / max(by_arr[0], 1e-30))
    axes[1, 1].axhline(1.0, color="k", ls="--", lw=0.8)
    axes[1, 1].set_xlabel("t")
    axes[1, 1].set_ylabel("By_xpt / By_xpt(0)")
    axes[1, 1].set_title("Normalised reconnection growth")

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "summary.png", dpi=130)
    plt.close(fig)
    print(f"\nSummary → {PLOT_DIR}/summary.png", flush=True)


if __name__ == "__main__":
    main()
