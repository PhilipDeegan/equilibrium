"""
Vacuum isotropy test: scalar u-field diffusion on the DST Kelvin lattice.

The DST update rule is first-order in time (diffusion equation).  A scalar
perturbation of u spreads outward; isotropy of that spread is the necessary
condition for the vacuum to have no preferred direction — and therefore for
any wave-like excitation (photon, transverse mode) to propagate at the same
c in every direction.

What this test confirms
───────────────────────
1. The BCC face weights give isotropic diffusivity D = D₀ I (spherical contours).
2. The measured D matches the analytic value from the face geometry.
3. The wavefront radius at a fixed threshold is the same along the three
   crystal axes: hex [1,1,1], square [1,0,0], diagonal [1,1,0].

Analytic diffusivity (DST face weights, K=1, L=1, λ=0)
───────────────────────────────────────────────────────
The second moment of a single-step diffusion kernel:
    D = (1/6) × Σ_k FU_k × d_k²
      = (1/6) × [8 × (√3/4) × 1² + 6 × (√3/12) × (2/√3)²]
      = (1/6) × [2√3 + 2√3/3]
      = (1/6) × (8√3/3)
      = 4√3/9  ≈  0.770  (Planck units)

Usage
─────
    from equilibrium.physical.dst.vacuum_test import run
    result = run()               # quick default (radius=25, 200 steps)
    result = run(plot=False)     # headless / CI use
"""

import numpy as np

# ---------------------------------------------------------------------------
# Analytic reference
# ---------------------------------------------------------------------------


def analytic_diffusivity() -> float:
    """
    D from DST face geometry (K=1, L=1, λ=0).

    D = (1/6) Σ_k FU_k d_k²
      = (1/6)(8×(√3/4)×1 + 6×(√3/12)×(4/3))
      = 4√3/9
    """
    FU_HEX = np.sqrt(3) / 4
    FU_SQ = np.sqrt(3) / 12
    d_sq2 = (2.0 / np.sqrt(3)) ** 2  # d_sq² = 4/3
    return (8 * FU_HEX * 1.0 + 6 * FU_SQ * d_sq2) / 6.0


# ---------------------------------------------------------------------------
# Wavefront measurement
# ---------------------------------------------------------------------------


def _wavefront_radius(
    pos: np.ndarray, u: np.ndarray, direction: list, threshold: float = 0.005
) -> float:
    """
    Radius of the furthest cell with u > threshold along a given direction.

    Parameters
    ----------
    pos       : (N, 3) cell positions
    u         : (N,)  u field
    direction : unit vector (will be normalised)
    threshold : absolute u level defining the wavefront

    Returns
    -------
    Furthest radius (cell units) with u > threshold along direction, or 0.
    """
    d = np.asarray(direction, dtype=float)
    d /= np.linalg.norm(d)
    dot = pos @ d
    fwd = dot > 0
    if not fwd.any():
        return 0.0
    r = np.linalg.norm(pos[fwd], axis=1)
    above = u[fwd] > threshold
    return float(r[above].max()) if above.any() else 0.0


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------


def run(
    radius: float = 25.0,
    n_steps: int = 200,
    pulse_sigma: float = 1.5,
    snapshots: tuple = (50, 100, 200),
    plot: bool = True,
) -> dict:
    """
    Run the vacuum isotropy test on a Planck-unit Kelvin lattice.

    Parameters
    ----------
    radius      : sphere radius in cell units (Planck lengths)
    n_steps     : total steps to run (last entry in snapshots wins if larger)
    pulse_sigma : width of initial Gaussian u-pulse (cells)
    snapshots   : steps at which to record a wavefront measurement
    plot        : produce matplotlib plots

    Returns
    -------
    dict:
        D_measured    float   diffusivity from <r²(t)> fit
        D_analytic    float   from face geometry (4√3/9)
        D_ratio       float   D_measured / D_analytic  (should be ≈ 1)
        anisotropy    float   (R_max − R_min) / R_mean at last snapshot
        dt            float   stable timestep used
        records       list    one entry per snapshot
    """
    from equilibrium.cpp_dst import DSTLattice  # pylint: disable=no-name-in-module

    n_steps = max(n_steps, max(snapshots))

    lat = DSTLattice(cell_size=1.0, radius_cells=radius, K=1.0)
    lat.set_perimeter_pressure(u0=0.0, shell_width_cells=1.5)

    pos = np.array(lat.positions())  # (N, 3), copy
    r_all = np.linalg.norm(pos, axis=1)  # (N,)

    # Gaussian pulse centred at origin, normalised to peak = 1
    pulse = np.exp(-(r_all**2) / (2.0 * pulse_sigma**2))
    lat.set_u(pulse)  # writes both ping-pong buffers in C++

    dt = lat.stable_dt()
    D_analytic = analytic_diffusivity()

    # Initial second moment
    w0 = float(pulse.sum()) + 1e-30
    r2_0 = float((pulse * r_all**2).sum()) / w0

    directions = {
        "hex [111]": [1, 1, 1],
        "sq  [100]": [1, 0, 0],
        "diag[110]": [1, 1, 0],
    }

    records = []
    prev_step = 0
    for step in sorted(set(snapshots)):
        lat.run(step - prev_step, dt, n_threads=4)
        prev_step = step

        u_snap = np.array(lat.u_array())  # copy for this snapshot
        t = step * dt
        w = float(u_snap.sum()) + 1e-30
        r2 = float((u_snap * r_all**2).sum()) / w

        r_dirs = {
            name: _wavefront_radius(pos, u_snap, direction=d)
            for name, d in directions.items()
        }

        records.append({"step": step, "t": t, "r2": r2, **r_dirs})

    # Fit D from <r²(t)> = r²(0) + 6 D t
    times = np.array([rec["t"] for rec in records])
    r2s = np.array([rec["r2"] for rec in records])
    if len(times) >= 2:
        slope = float(np.polyfit(times, r2s - r2_0, 1)[0])
        D_measured = slope / 6.0
    else:
        D_measured = float("nan")

    # Isotropy at last snapshot
    last = records[-1]
    r_vals = np.array([last[name] for name in directions if last[name] > 0])
    if len(r_vals) >= 2:
        anisotropy = float((r_vals.max() - r_vals.min()) / r_vals.mean())
    else:
        anisotropy = float("nan")

    result = {
        "D_measured": D_measured,
        "D_analytic": D_analytic,
        "D_ratio": D_measured / D_analytic if D_analytic else float("nan"),
        "anisotropy": anisotropy,
        "dt": dt,
        "records": records,
    }

    _print_summary(result)

    if plot:
        _plot(result, r2_0, directions)

    return result


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _print_summary(result: dict):
    D_m = result["D_measured"]
    D_a = result["D_analytic"]
    ani = result["anisotropy"]
    print(f"  D_analytic  : {D_a:.4f}  (4√3/9)")
    print(f"  D_measured  : {D_m:.4f}  (ratio {result['D_ratio']:.4f})")
    print(f"  anisotropy  : {ani:.4f}  (0 = perfect sphere)")
    last = result["records"][-1]
    for k, v in last.items():
        if k.startswith(("hex", "sq", "diag")):
            print(f"  R {k:<12}: {v:.2f} cells")


def _plot(result: dict, r2_0: float, directions: dict):
    import matplotlib.pyplot as plt

    records = result["records"]
    times = np.array([r["t"] for r in records])
    r2s = np.array([r["r2"] for r in records])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: <r²(t)>
    ax = axes[0]
    ax.plot(times, r2s, "o-", label="measured ⟨r²⟩")
    t_fit = np.linspace(0, times[-1], 200)
    D_m, D_a = result["D_measured"], result["D_analytic"]
    ax.plot(t_fit, r2_0 + 6 * D_m * t_fit, "--", label=f"fit  D={D_m:.3f}")
    ax.plot(
        t_fit, r2_0 + 6 * D_a * t_fit, ":", color="gray", label=f"analytic D={D_a:.3f}"
    )
    ax.set_xlabel("t  (Planck time units)")
    ax.set_ylabel("⟨r²⟩  (Planck length²)")
    ax.set_title("Diffusivity (K=1, L=1, λ=0)")
    ax.legend()

    # Right: wavefront radius per crystal axis
    ax = axes[1]
    for name in directions:
        vals = [r[name] for r in records]
        ax.plot(times, vals, "o-", label=name)
    ax.set_xlabel("t  (Planck time units)")
    ax.set_ylabel("wavefront radius R  (cells)")
    ax.set_title(f"Isotropy  (anisotropy = {result['anisotropy']:.4f})")
    ax.legend()

    plt.suptitle("DST Kelvin lattice — vacuum isotropy test (Planck units)")
    plt.tight_layout()
    plt.show()
