"""
Supplementary check requested alongside Test A2: doc/mems/project_dst_magnetic_field.md
option (b) pass criterion -- "drive ring mode cycling and let the phi wake emerge
naturally... requires measuring the radial phi profile outside the ring and
checking for 1/r^3 dipole shape."

Same IC as probe_cycling_ring_curl.py's "rest" phase (proton-knot + toroidal
ring cycling, no kick), but instead of the fixed-shell curl(phi) chirality
check, this fits the radial falloff of local |phi| (rms over the 14 faces)
outside the ring against a power law, following the exact
radial_phi_profile/fit_power_law pattern from
exp/2026/06/15/22_57_ring_phi_dipole/sweep_lam_lorentz.py.
"""

import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium.substrate.lattice import DampedLattice
from equilibrium.substrate.flux_tube import (
    find_toroidal_ring,
    init_toroidal_winding,
    resonant_freq,
)

R = 20.0
CHI0 = 0.9
PROFILE = "1/r2"
RING_AMP = 0.05
CROSS_SCALE = 0.3
FORCE_SCALE = 0.3
KICK_AXIS = "z"
N_PERIODS_REST = 5.0
SPONGE = 0.15
DAMP_MAX = 0.003
N_THREADS = 8


def build(W):
    lat = DampedLattice(radius_cells=R, sponge_fraction=SPONGE, damp_max=DAMP_MAX)
    pos = lat.pos
    r = np.linalg.norm(pos, axis=1)
    origin = int(np.argmin(r))
    lat.set_winding(lat.tiles[origin], chi_0=CHI0, profile=PROFILE)
    ring_tiles, ring_faces = find_toroidal_ring(lat, axis=KICK_AXIS, mode="minimal")
    if ring_tiles is None:
        print("ERROR: no ring found")
        sys.exit(1)
    omega = resonant_freq(len(ring_tiles), face_type="sq")
    if W != 0.0:
        init_toroidal_winding(
            lat, ring_tiles, ring_faces, amplitude=W * RING_AMP, frozen=False
        )
    return lat, pos, ring_tiles, omega


def radial_phi_profile(lat, pos, ring_centre, ring_r, r_min_fac=1.5, n_bins=25):
    r_all = np.linalg.norm(pos - ring_centre, axis=1)
    phi_rms = np.sqrt((lat.phi**2).mean(axis=1))
    r_min = ring_r * r_min_fac
    r_max = float(r_all.max()) * 0.80
    edges = np.linspace(r_min, r_max, n_bins + 1)
    r_mid = 0.5 * (edges[:-1] + edges[1:])
    totals = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    idx_all = ((r_all - r_min) / (r_max - r_min) * n_bins).astype(int)
    valid = (r_all >= r_min) & (r_all < r_max) & (idx_all >= 0) & (idx_all < n_bins)
    np.add.at(totals, idx_all[valid], phi_rms[valid])
    np.add.at(counts, idx_all[valid], 1)
    mask = counts > 0
    mean_prof = np.where(mask, totals / np.maximum(counts, 1), np.nan)
    return r_mid, mean_prof, counts


def fit_power_law(r_mid, profile):
    mask = np.isfinite(profile) & (profile > 0) & (r_mid > 0)
    if mask.sum() < 3:
        return None, None, mask.sum()
    lr = np.log(r_mid[mask])
    ly = np.log(profile[mask])
    coeffs = np.polyfit(lr, ly, 1)
    return float(np.exp(coeffs[1])), float(coeffs[0]), mask.sum()


def run(W):
    lat, pos, ring_tiles, omega = build(W)
    dt = lat.stable_dt()
    period = 2.0 * np.pi / omega
    n_rest = int(round(N_PERIODS_REST * period / dt))
    ring_pos = pos[ring_tiles]
    ring_centre = ring_pos.mean(axis=0)
    ring_r = float(np.linalg.norm(ring_pos - ring_centre, axis=1).mean())

    print(
        f"[W={W:+.1f}] N={lat.N}  dt={dt:.4e}  omega={omega:.4f}  period={period:.2f}  "
        f"n_rest={n_rest}  ring_r={ring_r:.3f}",
        flush=True,
    )

    for s in range(1, n_rest + 1):
        lat.step(dt, n_threads=N_THREADS)
        if CROSS_SCALE > 0.0:
            lat.step_cross(dt, CROSS_SCALE, n_threads=N_THREADS)
        if FORCE_SCALE > 0.0:
            lat.step_lam_force(dt, FORCE_SCALE)
        if s % 1000 == 0:
            print(
                f"  [W={W:+.1f}] step {s}/{n_rest}  max|phi|={np.abs(lat.phi).max():.4e}",
                flush=True,
            )

    r_mid, profile, counts = radial_phi_profile(lat, pos, ring_centre, ring_r)
    A, slope, n_pts = fit_power_law(r_mid, profile)
    print(
        f"[W={W:+.1f}] power-law fit over {n_pts} bins: A={A}  slope={slope}  "
        f"(expect slope~-3 for dipole 1/r^3)",
        flush=True,
    )
    return dict(
        W=W,
        r_mid=r_mid,
        profile=profile,
        counts=counts,
        A=A,
        slope=slope,
        ring_r=ring_r,
    )


def main():
    results = {}
    for W in (1.0, -1.0):
        results[W] = run(W)

    np.savez(
        "exp/2026/08/20/18_00_cycling_ring_charge_test/radial_profile.npz",
        r_mid_p=results[1.0]["r_mid"],
        profile_p=results[1.0]["profile"],
        slope_p=results[1.0]["slope"],
        r_mid_m=results[-1.0]["r_mid"],
        profile_m=results[-1.0]["profile"],
        slope_m=results[-1.0]["slope"],
        ring_r=results[1.0]["ring_r"],
    )
    print("saved exp/2026/08/20/18_00_cycling_ring_charge_test/radial_profile.npz")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    for W, marker, color in ((1.0, "o", "tab:blue"), (-1.0, "s", "tab:red")):
        r = results[W]["r_mid"]
        p = results[W]["profile"]
        mask = np.isfinite(p) & (p > 0)
        ax.loglog(
            r[mask],
            p[mask],
            marker,
            color=color,
            label=f"W={W:+.0f} (slope={results[W]['slope']:.2f})",
        )
        if results[W]["A"] is not None:
            rr = np.linspace(r[mask].min(), r[mask].max(), 50)
            ax.loglog(
                rr,
                results[W]["A"] * rr ** results[W]["slope"],
                "--",
                color=color,
                alpha=0.5,
            )
    r_ref = np.array(
        sorted(results[1.0]["r_mid"][np.isfinite(results[1.0]["profile"])])
    )
    if len(r_ref) > 1 and results[1.0]["A"] is not None:
        ref_A = results[1.0]["A"] * r_ref[0] ** (results[1.0]["slope"] + 3.0)
        ax.loglog(r_ref, ref_A * r_ref**-3.0, "k:", label="reference 1/r^3")
    ax.set_xlabel("r (distance from ring centre, cells)")
    ax.set_ylabel("phi_rms(r)")
    ax.legend()
    ax.set_title("Radial phi falloff outside cycling ring, at rest")
    fig.tight_layout()
    out = "exp/2026/08/20/18_00_cycling_ring_charge_test/radial_profile.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
