#!/usr/bin/env python3
"""1000-step run of Harris IC with virtual particle (macro-PIC) dynamics.

Wave equation step (C++) + virtual particle Lorentz step (Python).
No C++ virtual particle layer — everything is numpy.

Usage:
    PYTHONPATH=$PWD python3 -u run/substrate/adaptive/harris_virtual_particle.py
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt

from equilibrium import lattice as _lat
from equilibrium.substrate.adaptive import wire_periodic_bcc

# ── geometry ────────────────────────────────────────────────────────────────

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
W_FACE = np.array([3.0] * 8 + [1.0] * 6)
_ALL_DIR = np.concatenate([HEX_DIR, SQ_DIR])  # (14, 3)
_B_NORM = 26.0  # Σ_k W_k * dir_k[d]^2 = 8*3 + 2*1 = 26 per axis

# ── IC parameters ───────────────────────────────────────────────────────────

LX, LY, LZ = 30.0, 30.0, 3.5
B0 = 0.05
LAM_W = 6.0
U_BG = 0.4
LAM_BG = 0.4
DT = 0.05
PERTURB = 0.1
N_STEPS = 10000
PRINT_EVERY = 500
PLOT_OUT = Path("tmp/harris_vp")
# Current source coupling: phi[k] += dt * J_SCALE * sigma * lam * (v · dir_k)
# Dimensionless; 0 = ideal (no reconnection), ~1 = full current feedback.
J_SCALE = 1.0

# ── helpers ─────────────────────────────────────────────────────────────────


def _bx_profile(y, y0):
    return B0 * (-1.0 + np.tanh((y + y0) / LAM_W) - np.tanh((y - y0) / LAM_W))


def _reconstruct_B(phi):
    """phi (14,N) → B (N,3)."""
    return np.einsum("k,kn,kd->nd", W_FACE, phi, _ALL_DIR) / _B_NORM


def _build_box():
    alat = _lat.AdaptiveLattice()
    cl = alat[0]
    _lat.build_bcc_box(cl, LX, LY, LZ, 1.0, u0=0.0, lam0=LAM_BG)
    alat.rebuild(0)
    H = 1.0 / np.sqrt(3.0)
    wire_periodic_bcc(cl, H)

    N = cl.n_leaves
    pos = np.array(cl.leaf_pos)  # (N, 3)
    y0 = LY / 4.0
    bx = _bx_profile(pos[:, 1], y0)

    cl.set_leaf_u(U_BG + (B0**2 - bx**2) / 2.0)  # pressure-balanced

    phi = np.zeros((14, N), dtype=np.float64)
    for k in range(14):
        fd = HEX_DIR[k] if k < 8 else SQ_DIR[k - 8]
        phi[k] += bx * fd[0]

    kx = 2.0 * np.pi / LX
    dB = PERTURB * B0
    for sign, y_null in ((+1, +y0), (-1, -y0)):
        yr = pos[:, 1] - y_null
        env = np.exp(-(yr**2) / LAM_W**2)
        kx_x = kx * (pos[:, 0] + LX / 2.0)
        bx_p = sign * (-2.0 * yr / LAM_W**2) * dB * np.cos(kx_x) * env
        by_p = sign * kx * dB * np.sin(kx_x) * env
        for k in range(14):
            fd = HEX_DIR[k] if k < 8 else SQ_DIR[k - 8]
            phi[k] += bx_p * fd[0] + by_p * fd[1]

    cl.set_leaf_phi(phi)
    return alat, cl, pos, y0


def _step_virtual(cl, nb, gid, g_to_j, max_g, v3, dt):
    """Virtual particle Lorentz step + current source on phi + lam continuity.

    The DST wave equation is the field evolution (Ampere/Faraday in DST form).
    Virtual particles add a current source J = lam*v to phi AFTER the wave step:

        phi[k,i] += dt * J_scale * lam_i * (v_i · dir_k)

    This breaks the ideal (frozen-in) condition locally, enabling reconnection.
    The wave equation must run first to evolve the fields; this adds the
    non-ideal current source term that the ideal equation lacks.

    Lorentz push uses the current fields BEFORE the wave step (pre-update E, B).
    """
    N = cl.n_leaves
    u = np.asarray(cl.leaf_u, dtype=np.float64)
    phi = np.asarray(cl.leaf_phi, dtype=np.float64).copy()
    lam = np.asarray(cl.leaf_lam, dtype=np.float64)

    B = _reconstruct_B(phi)  # (N, 3)

    # E field: pressure-gradient projection (∇u, computed from current u)
    E = np.zeros((N, 3))
    for k in range(14):
        nb_g = nb[:, k]
        safe_g = np.where(nb_g >= 0, nb_g, gid)
        j = g_to_j[np.clip(safe_g, 0, max_g - 1)]
        safe_j = np.where(nb_g >= 0, j, np.arange(N, dtype=np.int32))
        du = u[safe_j] - u
        E += (W_FACE[k] * du)[:, None] * _ALL_DIR[k]

    # Lorentz acceleration
    vxB = np.cross(v3.T, B)  # (N, 3)
    v3_new = v3 + dt * (E + vxB).T  # (3, N)

    # Current source on phi: phi[k,i] += dt * J_scale * lam_i * (v_new · dir_k)
    # Attenuated by sigma (denser regions carry less current per unit lam).
    sigma = 1.0 / (1.0 + lam)
    for k in range(14):
        j_k = sigma * lam * np.einsum("d,dn->n", _ALL_DIR[k], v3_new)
        phi[k] += dt * J_SCALE * j_k
    cl.set_leaf_phi(phi)

    # Continuity: dlam = -Σ_k W_k * (lam_nb*v_nb - lam_i*v_i)·dir_k
    lam_v = lam[None, :] * v3
    dlam = np.zeros(N)
    for k in range(14):
        nb_g = nb[:, k]
        safe_g = np.where(nb_g >= 0, nb_g, gid)
        j = g_to_j[np.clip(safe_g, 0, max_g - 1)]
        safe_j = np.where(nb_g >= 0, j, np.arange(N, dtype=np.int32))
        flux = np.einsum("d,dn->n", _ALL_DIR[k], lam_v[:, safe_j] - lam_v)
        dlam -= dt * W_FACE[k] * flux

    cl.set_leaf_lam(np.clip(lam + dlam, 0.0, None))
    return v3_new


def _xy_slice(pos, vals, z_mid, dz=1.5):
    """Return mask of cells within dz of z_mid, for 2D plotting."""
    return np.abs(pos[:, 2] - z_mid) < dz


def _by_xpt(pos, phi, y0):
    """Mean |By| near the upper X-point (|x|<0.15 Lx, |y-y0|<LAM_W)."""
    B = _reconstruct_B(phi)
    xm = (np.abs(pos[:, 0]) < LX * 0.15) & (np.abs(pos[:, 1] - y0) < LAM_W)
    return float(np.sqrt(np.mean(B[xm, 1] ** 2))) if xm.any() else 0.0


def _ke(cl, v3):
    lam = np.asarray(cl.leaf_lam, dtype=np.float64)
    return 0.5 * float(np.sum(lam * np.sum(v3**2, axis=0)))


def _phi_e(cl):
    phi = np.asarray(cl.leaf_phi, dtype=np.float64)
    return float(np.sum(phi**2))


# ── plotting ─────────────────────────────────────────────────────────────────


def _make_plot(pos, cl, v3, y0, step, z_mid, out_dir):
    phi = np.asarray(cl.leaf_phi, dtype=np.float64)
    lam = np.asarray(cl.leaf_lam, dtype=np.float64)
    B = _reconstruct_B(phi)

    mask = _xy_slice(pos, lam, z_mid)
    px, py = pos[mask, 0], pos[mask, 1]
    lam_s = lam[mask]
    vx_s = v3[0, mask]
    vy_s = v3[1, mask]
    By_s = B[mask, 1]
    vmag = np.sqrt(vx_s**2 + vy_s**2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Step {step}  (t={step*DT:.1f})", fontsize=12)

    sc = axes[0].scatter(
        px, py, c=lam_s, s=4, cmap="viridis", vmin=LAM_BG * 0.5, vmax=LAM_BG * 2.0
    )
    plt.colorbar(sc, ax=axes[0], label="lam")
    axes[0].set_title("lam (mass proxy)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    # overlay null lines
    for sign in (+1, -1):
        axes[0].axhline(sign * y0, color="w", lw=0.5, ls="--")

    # velocity field (subsample for arrows)
    stride = max(1, len(px) // 300)
    axes[1].scatter(px, py, c=vmag, s=4, cmap="plasma")
    axes[1].quiver(
        px[::stride],
        py[::stride],
        vx_s[::stride],
        vy_s[::stride],
        scale=vmag.max() * 20 if vmag.max() > 0 else 1.0,
        width=0.003,
        color="k",
        alpha=0.6,
    )
    axes[1].set_title("|v| + arrows")
    axes[1].set_xlabel("x")
    for sign in (+1, -1):
        axes[1].axhline(sign * y0, color="r", lw=0.5, ls="--")

    sc2 = axes[2].scatter(
        px, py, c=By_s, s=4, cmap="RdBu_r", vmin=-0.5 * B0, vmax=0.5 * B0
    )
    plt.colorbar(sc2, ax=axes[2], label="By")
    axes[2].set_title("By (reconnecting component)")
    axes[2].set_xlabel("x")
    for sign in (+1, -1):
        axes[2].axhline(sign * y0, color="k", lw=0.5, ls="--")

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(-LX / 2, LX / 2)
        ax.set_ylim(-LY / 2, LY / 2)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"frame_{step:05d}.png", dpi=120)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    PLOT_OUT.mkdir(parents=True, exist_ok=True)

    print("Building IC …", flush=True)
    alat, cl, pos, y0 = _build_box()
    N = cl.n_leaves
    print(f"  N={N} cells", flush=True)

    # Cache neighbour lookup (static topology)
    nb = np.asarray(cl.leaf_nb, dtype=np.int32)  # (N, 14)
    gid = np.asarray(cl.leaf_global_indices, dtype=np.int32)  # (N,)
    max_g = int(nb.max()) + 1
    g_to_j = np.full(max_g, -1, dtype=np.int32)
    g_to_j[gid] = np.arange(N, dtype=np.int32)

    v3 = np.zeros((3, N), dtype=np.float64)
    z_mid = float(pos[:, 2].mean())

    # time-series buffers
    ts_step = []
    ts_by = []
    ts_ke = []
    ts_phi_e = []

    _make_plot(pos, cl, v3, y0, 0, z_mid, PLOT_OUT)

    print(f"{'step':>6}  {'By_xpt':>10}  {'KE':>12}  {'phi_E':>12}", flush=True)

    for step in range(1, N_STEPS + 1):
        alat.step(0, DT)  # wave equation: field evolution (DST Ampere/Faraday)
        v3 = _step_virtual(
            cl, nb, gid, g_to_j, max_g, v3, DT
        )  # Lorentz + J source on phi + continuity

        if step % PRINT_EVERY == 0:
            phi_now = np.asarray(cl.leaf_phi, dtype=np.float64)
            by = _by_xpt(pos, phi_now, y0)
            ke = _ke(cl, v3)
            pe = _phi_e(cl)
            ts_step.append(step)
            ts_by.append(by)
            ts_ke.append(ke)
            ts_phi_e.append(pe)
            print(f"{step:6d}  {by:10.3e}  {ke:12.4e}  {pe:12.4e}", flush=True)
            _make_plot(pos, cl, v3, y0, step, z_mid, PLOT_OUT)

    # Summary plot
    ts_step = np.array(ts_step, dtype=float) * DT
    ts_by = np.array(ts_by)
    ts_ke = np.array(ts_ke)
    ts_phi_e = np.array(ts_phi_e)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(ts_step, ts_by)
    axes[0].set(xlabel="t", ylabel="By_xpt (rms)", title="By at X-point")

    axes[1].plot(ts_step, ts_ke, label="KE")
    axes[1].set(xlabel="t", ylabel="energy", title="Kinetic energy")

    axes[2].plot(ts_step, ts_phi_e / ts_phi_e[0], label="phi_E / phi_E0")
    axes[2].axhline(1.0, color="k", lw=0.5, ls="--")
    axes[2].set(xlabel="t", ylabel="phi_E / phi_E0", title="Field energy (normalised)")

    plt.tight_layout()
    plt.savefig(PLOT_OUT / "timeseries.png", dpi=120)
    plt.close(fig)
    print(f"\nPlots written to {PLOT_OUT}/", flush=True)


if __name__ == "__main__":
    main()
