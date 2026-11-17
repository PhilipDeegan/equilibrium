"""
Measure the spiral winding of the near-field u pattern in a spheromak-decay
snapshot.

The equatorial u (symlog) plot from probe_decay_compare / sweep_spheromak_
decay_cross_scale's field snapshots shows two colored arms that sweep in
angle with radius rather than sitting on a fixed axis (a static knot/moat
pair would give a fixed bipolar split — a "butterfly" with no winding). A
swept arm is the signature of a rotating near-field source combined with
finite propagation speed (same mechanism as a Parker spiral): each
successive disturbance is emitted at a different orientation and left
behind as it propagates outward, winding the far field.

This script:
  1. Slices the plane perpendicular to --axis (default z, matching the
     toroidal spheromak's axis in sweep_spheromak_decay_cross_scale.py).
  2. Bins cells by radius and computes the complex m=1 azimuthal moment of
     u in each bin: M(r) = sum_i u_i * exp(-i*theta_i). Its phase is the
     orientation of the two-arm asymmetry at that radius.
  3. Unwraps phase(r) over radial bins with amplitude above --amp-frac of
     the peak, and fits a line: phase = k*r + c. Nonzero k is the spatial
     winding rate (rad per lattice unit) — the spiral pitch.
  4. If given >=2 snapshots of the same cross_scale at different step
     counts, also reports the phase shift at matched radii between them —
     a direct (but 2*pi-aliasable) estimate of the rotation rate Omega.

Usage:
  PYTHONPATH=$PWD python3 run/substrate/adaptive/probe_spiral_pitch.py \\
      tmp/decay_snapshots/decay_cs1.0000e-02_step050000.npz

  # temporal comparison once a later snapshot of the same cross_scale exists:
  PYTHONPATH=$PWD python3 run/substrate/adaptive/probe_spiral_pitch.py \\
      tmp/decay_snapshots/decay_cs1.0000e-02_step050000.npz \\
      tmp/decay_snapshots/decay_cs1.0000e-02_step060000.npz
"""

import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, ".")

parser = argparse.ArgumentParser()
parser.add_argument(
    "files",
    nargs="+",
    help=".npz snapshots (same cross_scale, any step order) to analyze",
)
parser.add_argument(
    "--axis",
    default="z",
    choices=["x", "y", "z"],
    help="spheromak axis — the plane perpendicular to this is sliced",
)
parser.add_argument(
    "--thickness",
    type=float,
    default=2.0,
    help="half-thickness of the equatorial slice",
)
parser.add_argument("--n-r-bins", type=int, default=24)
parser.add_argument(
    "--r-min",
    type=float,
    default=None,
    help="inner radius cutoff (default: 4x mean ring radius, to skip the noisy core)",
)
parser.add_argument(
    "--r-max-frac",
    type=float,
    default=0.85,
    help="outer cutoff as a fraction of the domain's max radius, to skip the sponge-zone edge",
)
parser.add_argument(
    "--amp-frac",
    type=float,
    default=0.1,
    help="drop radial bins whose m=1 amplitude is below this fraction of the peak",
)
parser.add_argument("--plot-dir", type=str, default="tmp")
parser.add_argument("--no-plot", action="store_true")


def _step_from_name(path):
    m = re.search(r"_step(\d+)\.npz$", path)
    return int(m.group(1)) if m else None


def _cross_scale_from_name(path):
    m = re.search(r"_cs([0-9.eE+-]+)_step", os.path.basename(path))
    return float(m.group(1)) if m else None


def load(path):
    d = np.load(path)
    pos = d["pos"]
    u = d["u"]
    ring_idx = (
        np.concatenate([d["A_tiles"], d["B_tiles"]])
        if "A_tiles" in d.files
        else np.array([], dtype=int)
    )
    ring_r = (
        float(np.linalg.norm(pos[ring_idx], axis=1).mean()) if ring_idx.size else 0.0
    )
    return dict(
        path=path,
        pos=pos,
        u=u,
        ring_r=ring_r,
        step=_step_from_name(path),
        cross_scale=_cross_scale_from_name(path),
    )


def m1_phase_profile(pos, field, axis, thickness, r_min, r_max, n_bins):
    """Complex m=1 azimuthal moment of `field` in radial bins, in the plane
    perpendicular to `axis`.

    Returns (r_centers, phase, amplitude) — phase in radians (wrapped to
    (-pi, pi]), amplitude the |sum| of the m=1 moment in each bin (0 where
    the bin is empty).
    """
    ia, ib, ic = {"z": (0, 1, 2), "x": (1, 2, 0), "y": (2, 0, 1)}[axis]
    perp = pos[:, ic]
    mask = np.abs(perp) < thickness
    x, y, f = pos[mask, ia], pos[mask, ib], field[mask]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    edges = np.linspace(r_min, r_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    phase = np.full(n_bins, np.nan)
    amp = np.zeros(n_bins)
    for i in range(n_bins):
        sel = (r >= edges[i]) & (r < edges[i + 1])
        if not np.any(sel):
            continue
        moment = np.sum(f[sel] * np.exp(-1j * theta[sel]))
        amp[i] = np.abs(moment)
        phase[i] = np.angle(moment)
    return centers, phase, amp


def fit_spatial_winding(r_centers, phase, amp, amp_frac):
    keep = (amp > amp_frac * amp.max()) & ~np.isnan(phase)
    if keep.sum() < 3:
        return None
    order = np.argsort(r_centers[keep])
    r_k = r_centers[keep][order]
    ph_k = np.unwrap(phase[keep][order])
    k, c = np.polyfit(r_k, ph_k, 1)  # phase(r) = k*r + c
    resid = ph_k - (k * r_k + c)
    return dict(
        k=float(k),
        c=float(c),
        r=r_k,
        phase=ph_k,
        resid_rms=float(np.sqrt(np.mean(resid**2))),
    )


def main():
    args = parser.parse_args()
    snaps = [load(p) for p in args.files]
    snaps.sort(key=lambda s: (s["step"] if s["step"] is not None else 0))

    plane_label = {"z": "XY", "x": "YZ", "y": "ZX"}[args.axis]
    print(
        f"slicing the {plane_label} plane (perp to axis={args.axis}), thickness={args.thickness}"
    )

    profiles = []
    for s in snaps:
        r_domain_max = float(np.linalg.norm(s["pos"], axis=1).max())
        r_min = args.r_min if args.r_min is not None else max(4.0 * s["ring_r"], 3.0)
        r_max = args.r_max_frac * r_domain_max
        r_c, phase, amp = m1_phase_profile(
            s["pos"], s["u"], args.axis, args.thickness, r_min, r_max, args.n_r_bins
        )
        fit = fit_spatial_winding(r_c, phase, amp, args.amp_frac)
        profiles.append(dict(snap=s, r=r_c, phase=phase, amp=amp, fit=fit))
        cs = s["cross_scale"]
        label = (
            f"cs={cs:.4e} step={s['step']}"
            if cs is not None
            else os.path.basename(s["path"])
        )
        if fit is None:
            print(
                f"{label}: not enough high-amplitude radial bins to fit a spiral (try --amp-frac lower)"
            )
            continue
        pitch_deg = np.degrees(np.arctan2(1.0, abs(fit["k"]) * np.mean(fit["r"])))
        print(
            f"{label}: dtheta/dr = {fit['k']:+.5f} rad/unit  "
            f"(fit resid rms {fit['resid_rms']:.3f} rad, r in [{fit['r'].min():.1f}, {fit['r'].max():.1f}])  "
            f"pitch angle @ r={np.mean(fit['r']):.1f} ~= {pitch_deg:.1f} deg  ring_r={s['ring_r']:.2f}"
        )

    # temporal comparison only makes sense within a fixed cross_scale, at
    # ascending steps — group by cross_scale (falling back to a single group
    # keyed on None if the filenames don't encode it) before comparing.
    groups = {}
    for p in profiles:
        if p["snap"]["step"] is None or p["fit"] is None:
            continue
        groups.setdefault(p["snap"]["cross_scale"], []).append(p)

    for cs, group in groups.items():
        if len(group) < 2:
            continue
        group.sort(key=lambda p: p["snap"]["step"])
        cs_label = (
            f"cs={cs:.4e}" if cs is not None else "cross_scale unknown from filename"
        )
        print(f"\ntemporal comparison ({cs_label}, ascending step):")
        base = group[0]
        for p in group[1:]:
            dstep = p["snap"]["step"] - base["snap"]["step"]
            r_common = np.linspace(
                max(base["fit"]["r"].min(), p["fit"]["r"].min()),
                min(base["fit"]["r"].max(), p["fit"]["r"].max()),
                8,
            )
            dphi = (p["fit"]["k"] * r_common + p["fit"]["c"]) - (
                base["fit"]["k"] * r_common + base["fit"]["c"]
            )
            dphi_wrapped = np.angle(np.exp(1j * dphi))
            print(
                f"  step {base['snap']['step']} -> {p['snap']['step']} (dstep={dstep}): "
                f"mean phase shift wrapped to (-pi,pi] = {np.mean(dphi_wrapped):+.4f} rad "
                f"(spread {np.std(dphi_wrapped):.4f})"
            )
            print(
                "    NOTE: aliased mod 2*pi — only reliable if the true rotation is "
                "<< half a turn over this step gap. Prefer closer-spaced snapshots "
                "(smaller --n-steps between --save-snapshot-dir writes) to remove "
                "the ambiguity."
            )

    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(args.plot_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for p in profiles:
            cs = p["snap"]["cross_scale"]
            label = (
                f"cs={cs:.4e} step={p['snap']['step']}"
                if cs is not None
                else os.path.basename(p["snap"]["path"])
            )
            axes[0].plot(p["r"], p["amp"], "o-", label=label, ms=3)
            if p["fit"] is not None:
                axes[1].plot(p["fit"]["r"], p["fit"]["phase"], "o", ms=3, label=label)
                rr = np.array([p["fit"]["r"].min(), p["fit"]["r"].max()])
                axes[1].plot(rr, p["fit"]["k"] * rr + p["fit"]["c"], "--", lw=1)
        axes[0].set_xlabel("r")
        axes[0].set_ylabel("m=1 amplitude |M(r)|")
        axes[0].set_title("azimuthal m=1 amplitude of u")
        axes[1].set_xlabel("r")
        axes[1].set_ylabel("unwrapped phase (rad)")
        axes[1].set_title("spiral phase(r) + linear fit")
        for ax in axes:
            ax.legend(fontsize=7)
        plt.tight_layout()
        out = f"{args.plot_dir}/spiral_pitch_{plane_label}.png"
        plt.savefig(out, dpi=130)
        print(f"\nplot saved -> {out}")


if __name__ == "__main__":
    main()
