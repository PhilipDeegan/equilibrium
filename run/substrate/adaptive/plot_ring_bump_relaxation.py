"""
Plot diagnostics from a saved probe_ring_bump_relaxation.py --save-npz run.

Produces:
  1. Waterfall: ring tile index (x) vs step (y), coloured by u -- shows
     wavefront propagation / interference / recurrence directly.
  2. |u-IC|/|IC| (full-array distance from initial condition) vs step.
  3. Low-order-mode (m=1..M_LOW) Fourier amplitude vs step, compared to the
     IC's own low-mode amplitude -- the targeted recurrence diagnostic
     (a sharp bump spreads energy over ALL modes; only the low ones are
     predicted to be near-commensurate, so this is the piece expected to
     show partial recurrence even when the full-array distance doesn't).

Run:
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/plot_ring_bump_relaxation.py <npz_path> [--out-prefix PREFIX] [--m-low 10]
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("npz_path", type=str)
parser.add_argument("--out-prefix", type=str, default=None)
parser.add_argument("--m-low", type=int, default=10)
args = parser.parse_args()

prefix = args.out_prefix or args.npz_path.rsplit(".", 1)[0]

d = np.load(args.npz_path)
steps = d["steps"]
ring_u = d["ring_u"]  # (n_snapshots, N_ring)
ic = d["ic_ring_u"]
N_ring = int(d["N_ring"])
dt = float(d["dt"])

# --- 1. waterfall ------------------------------------------------------
plt.figure(figsize=(10, 6))
vmax = np.percentile(np.abs(ring_u), 99)
plt.imshow(
    ring_u,
    aspect="auto",
    origin="lower",
    extent=[0, N_ring, steps[0], steps[-1]],
    cmap="RdBu_r",
    vmin=-vmax,
    vmax=vmax,
)
plt.colorbar(label="u")
plt.xlabel("ring tile index")
plt.ylabel("step")
plt.title(f"waterfall: u(tile, step), N_ring={N_ring}")
plt.tight_layout()
plt.savefig(f"{prefix}_waterfall.png", dpi=130)
plt.close()

# --- 2. full-array distance from IC ------------------------------------
dist = np.linalg.norm(ring_u - ic[None, :], axis=1)
dist0 = np.linalg.norm(ic)
plt.figure(figsize=(9, 4))
plt.plot(steps, dist / dist0)
plt.xlabel("step")
plt.ylabel("|u - IC| / |IC|  (full array)")
plt.title("full-array distance from initial condition")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{prefix}_full_distance.png", dpi=130)
plt.close()

# --- 3. low-mode Fourier amplitude vs step -----------------------------
idx = np.arange(N_ring)
plt.figure(figsize=(9, 4))
for m in range(1, args.m_low + 1):
    basis = np.exp(-1j * 2 * np.pi * m * idx / N_ring)
    amp_ic = abs((2.0 / N_ring) * np.sum(ic * basis))
    amp_t = np.abs((2.0 / N_ring) * np.sum(ring_u * basis[None, :], axis=1))
    plt.plot(
        steps, amp_t / (amp_ic + 1e-12), alpha=0.7, label=f"m={m}" if m <= 4 else None
    )
plt.xlabel("step")
plt.ylabel("|Z_m(t)| / |Z_m(0)|")
plt.title(f"low-order mode amplitudes (m=1..{args.m_low}), relative to IC")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{prefix}_low_modes.png", dpi=130)
plt.close()

print(
    f"wrote {prefix}_waterfall.png, {prefix}_full_distance.png, {prefix}_low_modes.png"
)
