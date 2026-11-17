"""
Quantify winding-decay loss per cross_scale from --save-snapshot-dir checkpoints
(see sweep_spheromak_decay_cross_scale.py).

Pure post-hoc analysis of the saved .npz fields (phi/u/lam/pos/A_tiles/B_tiles)
— no lattice reconstruction needed. Positions come straight from the snapshot
and the sweep's IC is an unrefined (uniform) BCC sphere, so the fixed
_FACE_NORM directions used by probe_decay_compare's _toroidal_phi are exact
face-bond directions everywhere: summing a cell's toroidal-phi component over
a ring's tiles reproduces phi_ring_circulation's per-ring sum exactly, with no
neighbour-table lookup required.

Each cross_scale's checkpoints are steps along one continuous chained-restart
trajectory (decay_cs<cs>_step<n>.npz, or decay_R<R>_cs<cs>_step<n>.npz once
R-tagging was added — see sweep_spheromak_decay_cross_scale.py), so a linear
fit of log|circ_B| vs step gives a per-step decay rate independent of how the
sweep chunked --n-steps across restarts, and independent of absolute step
numbering differing between families.

Also reports a boundary-bump radius per checkpoint (radial energy-density
profile, local max restricted to a window around r_inner — see
--boundary-window) — the diagnostic behind
the R-scan discriminator: does that radius track r_inner=R*(1-sponge) as R
changes (BC-onset seam artifact) or sit at a fixed absolute radius regardless
of R (genuine relaxation size)? Groups by (R, cross_scale) so snapshots from
multiple domain sizes in the same directory don't get conflated.

Usage:
  PYTHONPATH=$PWD python3 run/substrate/adaptive/analyze_decay_loss.py
  PYTHONPATH=$PWD python3 run/substrate/adaptive/analyze_decay_loss.py \\
      tmp/decay_snapshots --out tmp/decay_loss.png
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, ".")
from run.substrate.adaptive.probe_decay_compare import _toroidal_phi

# R-tagged (current) filenames: decay_R40_cs1.0000e-02_step050000.npz
# legacy (pre-R-tagging) filenames: decay_cs1.0000e-02_step050000.npz — group
# key R falls back to None (caller must not mix untagged files from different
# domain sizes in one directory; the sweep script itself refuses to restart
# across mismatched R regardless of filename).
_FNAME_RE = re.compile(r"decay_(?:R(\d+)_)?cs([0-9.eE+-]+)_step(\d+)\.npz$")

parser = argparse.ArgumentParser()
parser.add_argument("snapshot_dir", nargs="?", default="tmp/decay_snapshots")
parser.add_argument(
    "--sponge",
    type=float,
    default=0.15,
    help="sponge fraction used at build time (r > r_max*(1-sponge) is the "
    "boundary shell for the buildup diagnostic)",
)
parser.add_argument(
    "--boundary-window",
    type=float,
    default=8.0,
    help="search for the boundary bump only in r > r_inner - this margin — "
    "restricting to a window around r_inner (not just 'outside the core') "
    "matters because the domain also has a genuine mid-radius secondary "
    "feature (a dipole-shell lobe around r~10, seen at every cross_scale "
    "including the clean ones) that would otherwise win the global-max "
    "search and mask the r_inner-specific buildup this diagnostic targets",
)
parser.add_argument("--out", type=str, default="tmp/decay_loss_vs_cross_scale.png")
parser.add_argument("--no-plot", action="store_true")


def _bump_radius(pos, u, phi, r_inner, window, nbins=60):
    """Radius of the largest local energy-density concentration within
    [r_inner - window, r_max] — restricted to a window around r_inner (not a
    global outside-the-core search) so this specifically answers "is there a
    buildup right at the sponge onset", not "where is the domain's largest
    secondary feature" (there's an unrelated mid-radius dipole-shell lobe,
    ~r=10 for R=40, present at every cross_scale, that would otherwise win).
    Returns (None, 0.0) if the window has no local bump."""
    r = np.linalg.norm(pos, axis=1)
    r_max = float(r.max())
    e_dens = u**2 + 0.5 * np.sum(phi**2, axis=0)
    bins = np.linspace(0.0, r_max, nbins + 1)
    idx = np.clip(np.digitize(r, bins) - 1, 0, nbins - 1)
    tot = np.zeros(nbins)
    cnt = np.zeros(nbins)
    np.add.at(tot, idx, e_dens)
    np.add.at(cnt, idx, 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    mean_prof = tot / np.maximum(cnt, 1)
    in_window = centers > max(r_inner - window, 0.0)
    if not np.any(in_window):
        return None, 0.0
    i = np.argmax(mean_prof[in_window])
    return float(centers[in_window][i]), float(mean_prof[in_window][i])


def _load_metrics(path, sponge, boundary_window):
    d = np.load(path)
    phi = d["phi"]  # (14, N)
    u = d["u"]  # (N,)
    lam = d["lam"]  # (N,)
    pos = d["pos"]  # (N, 3)
    A_tiles = d["A_tiles"].astype(int) if "A_tiles" in d.files else np.array([], int)
    B_tiles = d["B_tiles"].astype(int) if "B_tiles" in d.files else np.array([], int)

    phi_tor = _toroidal_phi(pos, phi.T)  # (N,) — see module docstring
    r = np.linalg.norm(pos, axis=1)
    r_max = float(r.max())
    r_inner = r_max * (1.0 - sponge)
    shell = r > r_inner
    bump_r, bump_val = _bump_radius(pos, u, phi, r_inner, boundary_window)

    return dict(
        circ_A=float(np.sum(phi_tor[A_tiles])) if A_tiles.size else 0.0,
        circ_B=float(np.sum(phi_tor[B_tiles])) if B_tiles.size else 0.0,
        phi_rms=float(np.sqrt(np.mean(phi**2))),
        H=float(np.sum(u**2) + 0.5 * np.sum(phi**2)),
        lam_A=float(np.mean(lam[A_tiles])) if A_tiles.size else 0.0,
        lam_B=float(np.mean(lam[B_tiles])) if B_tiles.size else 0.0,
        boundary_phi_frac=float(
            np.sum(phi[:, shell] ** 2) / max(np.sum(phi**2), 1e-300)
        ),
        boundary_u_frac=float(np.sum(u[shell] ** 2) / max(np.sum(u**2), 1e-300)),
        shell_cell_frac=float(np.mean(shell)),
        r_max=r_max,
        r_inner=r_inner,
        bump_r=bump_r,
        bump_val=bump_val,
    )


def collect(snapshot_dir, sponge, boundary_window):
    """(R, cross_scale) -> sorted list of (step, metrics_dict). R is None for
    legacy (pre-R-tagging) filenames."""
    groups = {}
    for path in sorted(glob.glob(os.path.join(snapshot_dir, "decay_*_step*.npz"))):
        m = _FNAME_RE.search(path)
        if not m:
            continue
        R = int(m.group(1)) if m.group(1) is not None else None
        cs = float(m.group(2))
        step = int(m.group(3))
        groups.setdefault((R, cs), []).append(
            (step, _load_metrics(path, sponge, boundary_window))
        )
    for key in groups:
        groups[key].sort(key=lambda p: p[0])
    return groups


def decay_rate(steps, circ_B):
    """Fit log|circ_B| = a - step/tau. Returns (tau_steps, frac_loss_total)."""
    mag = np.abs(np.asarray(circ_B))
    frac_loss_total = 1.0 - mag[-1] / mag[0] if mag[0] > 0 else float("nan")
    if len(steps) < 2 or mag[0] <= 0:
        return float("nan"), frac_loss_total
    slope, _ = np.polyfit(steps, np.log(np.maximum(mag, 1e-300)), 1)
    tau_steps = -1.0 / slope if slope < 0 else float("inf")
    return tau_steps, frac_loss_total


def main():
    args = parser.parse_args()
    groups = collect(args.snapshot_dir, args.sponge, args.boundary_window)
    if not groups:
        print(f"no decay_*_step*.npz snapshots found under {args.snapshot_dir}")
        return

    keys = sorted(groups, key=lambda k: (k[0] if k[0] is not None else -1, k[1]))
    hdr = (
        f"{'R':>6}  {'cross_scale':>12}  {'n_ckpt':>6}  {'steps':>21}  "
        f"{'circ_B_0':>10}  {'circ_B_f':>10}  {'frac_loss':>9}  {'tau_steps':>10}  "
        f"{'bndry_phi_0':>11}  {'bndry_phi_f':>11}  {'r_inner':>8}  "
        f"{'bump_r_0':>9}  {'bump_r_f':>9}"
    )
    print(hdr)
    print("-" * len(hdr))

    summary = {}
    for key in keys:
        R, cs = key
        pts = groups[key]
        steps = np.array([p[0] for p in pts], dtype=float)
        circ_B = np.array([p[1]["circ_B"] for p in pts])
        bnd0 = pts[0][1]["boundary_phi_frac"]
        bndf = pts[-1][1]["boundary_phi_frac"]
        r_inner = pts[-1][1]["r_inner"]
        bump_r0 = pts[0][1]["bump_r"]
        bump_rf = pts[-1][1]["bump_r"]
        tau_steps, frac_loss = decay_rate(steps, circ_B)
        step_range = f"{int(steps[0])}-{int(steps[-1])}"
        R_disp = str(R) if R is not None else "?"
        print(
            f"{R_disp:>6}  {cs:12.4e}  {len(pts):6d}  {step_range:>21}  "
            f"{circ_B[0]:10.4e}  {circ_B[-1]:10.4e}  {frac_loss:9.4f}  "
            f"{tau_steps:10.3e}  {bnd0:11.4e}  {bndf:11.4e}  {r_inner:8.2f}  "
            f"{bump_r0 if bump_r0 is not None else float('nan'):9.2f}  "
            f"{bump_rf if bump_rf is not None else float('nan'):9.2f}"
        )
        summary[key] = dict(
            tau_steps=tau_steps,
            frac_loss=frac_loss,
            bnd0=bnd0,
            bndf=bndf,
            r_inner=r_inner,
            bump_rf=bump_rf,
            steps=steps,
            circ_B=circ_B,
        )

    print(
        "\nbump_r is the radius of the largest energy-density concentration "
        "within --boundary-window of r_inner — compare bump_r_f to r_inner "
        "across R values to tell a BC-onset seam (bump_r tracks r_inner) from a genuine fixed "
        "relaxation radius (bump_r_f converges to the same value regardless of R)."
    )

    if len(keys) < 2:
        print("\nneed >=2 (R, cross_scale) groups with data to compare loss scaling")

    if args.no_plot:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cmap = plt.get_cmap("viridis")

    for i, key in enumerate(keys):
        R, cs = key
        s = summary[key]
        color = cmap(i / max(len(keys) - 1, 1))
        label = f"R={R if R is not None else '?'} cs={cs:.2e}"
        axes[0].semilogy(
            s["steps"],
            np.abs(s["circ_B"]) + 1e-30,
            "o-",
            color=color,
            label=label,
        )
        axes[1].axvline(s["r_inner"], color=color, ls=":", lw=0.8)
        if s["bump_rf"] is not None:
            axes[1].scatter([i], [s["bump_rf"]], color=color, label=label)

    axes[0].set_xlabel("step")
    axes[0].set_ylabel("|circ_B| (log)")
    axes[0].set_title("Ring B circulation vs step")
    axes[0].legend(fontsize=6)

    axes[1].set_xlabel("(R, cross_scale) group index")
    axes[1].set_ylabel("radius")
    axes[1].set_title("Final bump radius (dots) vs r_inner (dotted lines)")
    axes[1].legend(fontsize=6)

    plt.tight_layout()
    plt.savefig(args.out, dpi=130)
    print(f"\nplot saved -> {args.out}")


if __name__ == "__main__":
    main()
