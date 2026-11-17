"""Per-construction, multi-perspective diagnostic gallery -- follow-up to
plot_all_dipole_ics.py.

That first gallery plotted one LINEAR-scale meridional slice per construction
in a single combined grid. Philip caught a real confound looking at it: the
bare current-loop panel (open/proton-lineage by construction -- a genuine
Ampere's-law loop, no hard cutoff) looked basically identical to the
spheromak/vecpot-dipole panels (closed/neutron-lineage, field forced exactly
zero outside R_sph*scale by the k=4.493 Taylor-state eigenvalue): both just
render as "two separated blobs, black everywhere else."

Why that happened: a current loop's energy density concentrates at the two
points where the wire crosses a meridional slice, regardless of whether the
overall field is open or closed. On a LINEAR color scale set by that near-wire
peak, an EXACTLY-zero exterior and a genuinely-nonzero-but-rapidly-decaying
exterior tail both render as indistinguishable black. "Two blobs vs one peak,
linear scale" cannot actually distinguish open from closed topology -- it's a
confounded heuristic, not a real topology test (and the same confound was
implicit in 09-21's PNG-vs-PNG visual comparison against PHARE references).

This script builds one figure PER construction (not a combined grid), each
with four panels, to make the real distinguishing test -- hard floor at
r=R_sph*scale (closed) vs a continuing decaying tail past it (open) --
directly visible and measurable:
  1. Meridional (x-z) slice, linear scale (unchanged from the first gallery,
     kept for continuity).
  2. Meridional (x-z) slice, LOG color scale, floor-clipped -- makes an
     exactly-zero exterior and a small-but-real tail visually distinct in a
     way the linear scale cannot.
  3. Equatorial (x-y, |z|<=1) slice, log color scale -- a second viewing
     angle on the same question.
  4. Radial energy-density profile (mean per shell vs r, semilogy), with a
     vertical dashed line at r=R_sph*scale=50 -- the real, quantitative
     version of the test: a hard cliff at/near that line supports
     closed/neutron-lineage, a continuing decaying tail past it supports
     open/proton-lineage.

Reuses plot_all_dipole_ics.py's cluster-building/seeding code directly
(imported as a sibling module) rather than duplicating it -- same 7
constructions, same exclusions (init_dipole_ic/init_phare_ic need a PHARE
ic_b.npz that doesn't exist in this environment; init_toroidal_winding is a
lower-level ring-seeding primitive, not itself a dipole/electron IC).
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/deegan/git/eq")
sys.path.insert(0, _HERE)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import plot_all_dipole_ics as gallery

OUT_DIR = os.path.join(_HERE, "per_ic")
os.makedirs(OUT_DIR, exist_ok=True)

R_BOUNDARY = gallery.R_SPH * gallery.SCALE  # 50.0 -- nominal construction boundary
N_RADIAL_BINS = 160


def equatorial_slice(cl, energy):
    pos = np.array(cl.leaf_pos)
    mask = np.abs(pos[:, 2]) <= gallery.SLICE_HALF_WIDTH
    return pos[mask, 0], pos[mask, 1], energy[mask]


def radial_profile(cl, energy, r_max, n_bins=N_RADIAL_BINS):
    pos = np.array(cl.leaf_pos)
    r = np.linalg.norm(pos, axis=1)
    bins = np.linspace(0.0, r_max, n_bins + 1)
    idx = np.clip(np.digitize(r, bins) - 1, 0, n_bins - 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    profile = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(n_bins):
        m = idx == i
        counts[i] = m.sum()
        if counts[i]:
            profile[i] = energy[m].mean()
    return centers, profile, counts


def _log_floor(e_max):
    return max(e_max * 1e-8, 1e-14)


def plot_construction(title, filename, seed_fn):
    print(f"--- building: {title.splitlines()[0]} ---", flush=True)
    alat, cl = gallery.build_cluster()
    pos = np.array(cl.leaf_pos)
    cell_size = np.full(cl.n_leaves, 1.0)
    try:
        seed_fn(cl, pos, cell_size)
    except Exception as exc:
        print(f"    FAILED: {exc}", flush=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, f"FAILED:\n{exc}", ha="center", va="center", color="red")
        ax.axis("off")
        fig.savefig(os.path.join(OUT_DIR, filename), dpi=130)
        plt.close(fig)
        return None

    energy = gallery.energy_density(cl)
    e_max = float(energy.max())
    floor = _log_floor(e_max)

    x_m, z_m, e_m = gallery.meridional_slice(cl, energy)
    x_e, y_e, e_e = equatorial_slice(cl, energy)
    r_c, prof, counts = radial_profile(cl, energy, gallery.PLOT_R)

    # hard-floor-vs-tail verdict: last radial bin (near r_max) still above
    # a small multiple of the numerical floor => real tail, not a hard cliff.
    valid = counts > 0
    tail_floor = max(e_max * 1e-6, 1e-13)
    beyond = r_c > R_BOUNDARY * 1.1
    beyond_valid = valid & beyond & ~np.isnan(prof)
    if beyond_valid.any():
        tail_frac_above_floor = float(np.mean(prof[beyond_valid] > tail_floor))
        tail_max_beyond = float(np.nanmax(prof[beyond_valid]))
    else:
        tail_frac_above_floor = float("nan")
        tail_max_beyond = float("nan")
    verdict = (
        "TAIL past boundary (open-lineage signature)"
        if tail_frac_above_floor > 0.5
        else "HARD FLOOR at boundary (closed-lineage signature)"
    )
    print(
        f"    e_max={e_max:.4e}  tail_frac_above_floor(beyond {R_BOUNDARY:.0f})="
        f"{tail_frac_above_floor:.3f}  tail_max_beyond={tail_max_beyond:.3e}"
        f"  -> {verdict}",
        flush=True,
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.5))
    ax_lin, ax_log, ax_eq, ax_rad = axes.ravel()

    sc = ax_lin.scatter(x_m, z_m, c=e_m, s=3, cmap="inferno", vmin=0.0, vmax=max(e_max, 1e-12))
    ax_lin.set_title("meridional (x-z), linear scale")
    ax_lin.set_xlim(-gallery.PLOT_R, gallery.PLOT_R)
    ax_lin.set_ylim(-gallery.PLOT_R, gallery.PLOT_R)
    ax_lin.set_aspect("equal")
    fig.colorbar(sc, ax=ax_lin, fraction=0.046, pad=0.04)

    e_m_clipped = np.clip(e_m, floor, None)
    sc2 = ax_log.scatter(
        x_m, z_m, c=e_m_clipped, s=3, cmap="inferno",
        norm=LogNorm(vmin=floor, vmax=max(e_max, floor * 10)),
    )
    ax_log.set_title(f"meridional (x-z), LOG scale (floor={floor:.1e})")
    ax_log.set_xlim(-gallery.PLOT_R, gallery.PLOT_R)
    ax_log.set_ylim(-gallery.PLOT_R, gallery.PLOT_R)
    ax_log.set_aspect("equal")
    circ = plt.Circle((0, 0), R_BOUNDARY, fill=False, ec="cyan", lw=1.0, ls="--")
    ax_log.add_patch(circ)
    fig.colorbar(sc2, ax=ax_log, fraction=0.046, pad=0.04)

    e_e_clipped = np.clip(e_e, floor, None)
    sc3 = ax_eq.scatter(
        x_e, y_e, c=e_e_clipped, s=3, cmap="inferno",
        norm=LogNorm(vmin=floor, vmax=max(e_max, floor * 10)),
    )
    ax_eq.set_title("equatorial (x-y), LOG scale")
    ax_eq.set_xlim(-gallery.PLOT_R, gallery.PLOT_R)
    ax_eq.set_ylim(-gallery.PLOT_R, gallery.PLOT_R)
    ax_eq.set_aspect("equal")
    circ2 = plt.Circle((0, 0), R_BOUNDARY, fill=False, ec="cyan", lw=1.0, ls="--")
    ax_eq.add_patch(circ2)
    fig.colorbar(sc3, ax=ax_eq, fraction=0.046, pad=0.04)

    prof_plot = np.where(np.isnan(prof), floor, np.clip(prof, floor, None))
    ax_rad.semilogy(r_c, prof_plot, marker=".", ms=3, lw=1.0, color="darkorange")
    ax_rad.axvline(R_BOUNDARY, color="cyan", ls="--", lw=1.2, label=f"r=R_sph*scale={R_BOUNDARY:.0f}")
    ax_rad.axhline(tail_floor, color="grey", ls=":", lw=0.8, label="tail-floor threshold")
    ax_rad.set_ylim(floor * 0.5, max(e_max, floor * 10) * 2)
    ax_rad.set_xlim(0, gallery.PLOT_R)
    ax_rad.set_title(f"radial mean energy density -> {verdict}", fontsize=9.5)
    ax_rad.set_xlabel("r (DST units)")
    ax_rad.set_ylabel("mean energy density (per-shell)")
    ax_rad.legend(fontsize=7, loc="upper right")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(OUT_DIR, filename)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"    saved {out_path}", flush=True)
    return {
        "title": title.splitlines()[0],
        "file": out_path,
        "e_max": e_max,
        "tail_frac_above_floor": tail_frac_above_floor,
        "verdict": verdict,
    }


FILE_NAMES = {
    "init_electron_dipole_ic": "init_electron_dipole_ic.png",
    "init_electron_dipole_ic_vecpot": "init_electron_dipole_ic_vecpot.png",
    "init_spheromak_ic": "init_spheromak_ic.png",
    "seed_dipole_ic": "seed_dipole_ic.png",
    "seed_curl_full_dipole_ic": "seed_curl_full_dipole_ic.png",
    "seed_curl_full_pair": "seed_curl_full_pair.png",
    "bare current loop": "bare_current_loop.png",
}


def main():
    results = []
    for title, seed_fn in gallery.PANELS:
        key = next(
            k for k in sorted(FILE_NAMES, key=len, reverse=True) if title.startswith(k)
        )
        r = plot_construction(title, FILE_NAMES[key], seed_fn)
        if r is not None:
            results.append(r)

    print("\n=== SUMMARY ===")
    for r in results:
        print(f"{r['title']:55s} tail_frac={r['tail_frac_above_floor']:.3f}  {r['verdict']}")


if __name__ == "__main__":
    main()
