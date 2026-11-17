"""
DST Brazil plot — pressure anisotropy β_∥ vs T_⊥/T_∥.

Runs a proper MHD time-evolution on an occupation field occ[N,14]:

    step_frozen_in   :  ΔA      = dt * (occ_vec × curl A)    — matter advects field
    step_occ_lorentz :  Δocc    = dt * (â × occ_vec)         — cyclotron precession
    step_occ_stream  :  diffusive occ transport between B-cells driven by |curl A|
    step_occ_n       :  ∂occ_n/∂t + ∇·(occ_n · v̂) = 0      — density continuity

The continuity step is the key physics: when a cell goes mirror-unstable
(high v_⊥), density flows out with the fluid, raising β_∥ = T_∥/(occ_n·|curl A|²)
and reducing R = β_⊥/β_∥ back toward the mirror boundary — self-correcting
without any explicit stability enforcement.

β_∥ and β_⊥ are sampled every `sample_every` steps; the scatter is coloured
by 2D log-density (like a solar-wind data distribution panel).

Field identification:
    phi[i,k]   <->  vector potential A (face 1-form)
    occ[i,k]   <->  matter current per face (occupation field)
    B          =    curl A (plaquette sum)
    β_∥        =    (occ·b̂)²  / (lam · |B|²)
    T_⊥/T_∥   =    β_⊥/β_∥ = (|occ|²−(occ·b̂)²) / (2(occ·b̂)²)

Stability boundaries (kinetic MHD / CGL):
    Firehose:  R < 1 − 1/β_∥   (field-line buckling; β_∥ > 1 only)
    Mirror:    R > 1 + 1/β_∥   (magnetic-bottle trapping)

Usage:
    PYTHONPATH=$PWD python3 vis/substrate/plot_brazil_instabilities.py
    PYTHONPATH=$PWD python3 vis/substrate/plot_brazil_instabilities.py \\
        --n-init 60 --steps 1200 --alpha 0.3 --save brazil.png
"""

import argparse
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")

from equilibrium.substrate.ab import _set_solenoid, _solenoid_cells
from equilibrium.substrate.lattice import SubstrateLattice
from equilibrium.substrate.mhd import (
    _SQ_AXIS_PAIRS,
    _b_vec,
    _b_vec_safe,
    _occ_vec,
    capture_ground_state,
    step_frozen_in,
    step_ground_restore,
    step_occ_lorentz,
    step_occ_n,
    step_occ_stream,
    step_phi_diffuse,
)
from equilibrium.substrate.mhd import (
    firehose_curve as _firehose_curve,
)
from equilibrium.substrate.mhd import (
    mirror_curve as _mirror_curve,
)

# ---------------------------------------------------------------------------
# Density colouring


def _density_colors(bp, R, n_bins=70):
    """
    Colour each point by its 2D log-space bin density.
    Returns values in [0, 1] on a log scale (matching the reference colour bar).
    """
    log_bp = np.log10(np.clip(bp, 1e-6, None))
    log_R = np.log10(np.clip(R, 1e-6, None))
    H, xe, ye = np.histogram2d(log_bp, log_R, bins=n_bins)
    xi = np.clip(np.searchsorted(xe[1:], log_bp), 0, H.shape[0] - 1)
    yi = np.clip(np.searchsorted(ye[1:], log_R), 0, H.shape[1] - 1)
    density = H[xi, yi]
    # Log-compress like the reference (counts span several decades)
    density = np.log10(density + 1.0)
    dmax = density.max()
    return density / dmax if dmax > 0 else density


# ---------------------------------------------------------------------------
# Data generation


def _init_occ(lat, b_cell_list, rng):
    """
    Fresh occ initialised directly in the (β_∥, R) normal-mode basis.

    Draw β_∥ ∈ [0.01, 10] and R = β_⊥/β_∥ ∈ [0.1, 10] log-uniform,
    then set |occ| = 0.5·|B| (sub-Alfvénic) and back-compute occ_n so
    the drawn β values are exactly reproduced:

        |occ|² = n · |B|² · β_∥ · (1 + 2R)   →  n = 0.25 / (β_∥·(1+2R))

    This decouples β_∥ from the Alfvénic clamp in step_occ_stream: small n
    lets β_∥ > 1 survive the clamp, covering the full solar-wind Brazil range.
    """
    occ = np.zeros((lat.N, 14))
    occ_n = np.zeros(lat.N)
    for i in b_cell_list:
        bv = _b_vec(lat.phi, lat.nb, i)
        if bv is None:
            continue
        B_sq = float(np.dot(bv, bv))
        B_mag = float(np.sqrt(B_sq))
        if B_mag < 1e-10:
            continue
        b_hat = bv / B_mag

        # draw β_∥ ∈ [1e-3, 1e2] log-uniform (matches the plot's own display
        # range -- narrower ranges leave the dense core confined to whatever
        # sub-range was drawn, since evolution only lets rare cells drift
        # beyond it; the bulk/dense population needs to start across the
        # full range for the observed "rotated square"/lens envelope to
        # appear).
        #
        # R's band width narrows as beta_para moves away from 1 in EITHER
        # direction, not just beta_para>1: real Brazil diagrams (see
        # /home/p/downloads/brazil,png) taper at BOTH ends of beta_para, not
        # only the high end the pure mirror/firehose formulas alone give
        # (mirror widens without bound as beta_para->0, which would fill the
        # whole R range there -- not what's observed). beta_eff=max(bp,1/bp)
        # is the same "multiplicative distance from 1" whichever side bp is
        # on, so applying the mirror-curve-shaped width via beta_eff mirrors
        # the same 1/beta_para falloff onto the beta_para<1 side. For bp>1
        # this reduces exactly to the original mirror/firehose formulas
        # (beta_eff=bp there) -- the real physical stability bounds are
        # unchanged; only the bp<1 side gains this (empirical, not
        # theory-derived) symmetric narrowing to match the observed shape.
        bp = float(10 ** rng.uniform(-3.0, 2.0))
        beta_eff = max(bp, 1.0 / bp)
        R_hi = min(10.0, 1.0 + 1.0 / beta_eff)
        R_lo = max(0.1, 1.0 - 1.0 / beta_eff)
        if R_lo >= R_hi:
            R = 1.0
        else:
            R = float(10 ** rng.uniform(np.log10(R_lo), np.log10(R_hi)))

        # |occ| = 0.5·|B|; split by energy fraction
        occ_mag = 0.5 * B_mag
        para_frac = 1.0 / (1.0 + 2.0 * R)  # fraction of |occ|² in v_∥
        v_para_mag = occ_mag * float(np.sqrt(para_frac))
        v_perp_mag = occ_mag * float(np.sqrt(1.0 - para_frac))

        sign = float(rng.choice([-1.0, 1.0]))
        v_para = sign * v_para_mag * b_hat

        e = rng.standard_normal(3)
        e -= np.dot(e, b_hat) * b_hat
        en = float(np.linalg.norm(e))
        if en < 1e-10:
            continue
        v_perp = (v_perp_mag / en) * e

        v = v_para + v_perp
        for ax, (fp, fm) in enumerate(_SQ_AXIS_PAIRS):
            occ[i, fp] = max(v[ax], 0.0)
            occ[i, fm] = max(-v[ax], 0.0)

        # back-compute n so β_∥ = T_∥/(n·|B|²) matches the draw
        # T_∥ = v_para_mag² = occ_mag²·para_frac = 0.25·|B|²·para_frac
        # β_∥ = T_∥/(n·|B|²)  →  n = 0.25·para_frac / bp
        occ_n[i] = max(0.25 * para_frac / bp, 1e-6)
    return occ, occ_n


def _sample_beta(lat, occ, occ_n, b_cell_list, bp_list, bq_list):
    """Append (β_∥, β_⊥) to the accumulators for each valid curl-A cell.

    β_∥ = T_∥ / (occ_n · |curl A|²)   where T_∥ = (occ_vec · â)²
    β_⊥ = T_⊥ / (2 · occ_n · |curl A|²)

    occ_n is the charge density (independent scalar), decoupled from the
    velocity/pressure 1-form occ.  This breaks the self-referential ceiling
    that locked the distribution to β_∥ ~ 1/|curl A|.
    """
    for i in b_cell_list:
        curl_a = _b_vec_safe(lat, i)
        if curl_a is None:
            continue
        curl_a_sq = float(np.dot(curl_a, curl_a))
        if curl_a_sq < 1e-30:
            continue
        a_hat = curl_a / np.sqrt(curl_a_sq)
        v = _occ_vec(occ[i])
        if not np.all(np.isfinite(v)) or np.dot(v, v) > 1e200:
            continue
        n = float(occ_n[i])
        if n < 1e-30:
            continue
        T_para = float(np.dot(v, a_hat)) ** 2
        T_perp = float(np.dot(v, v)) - T_para
        denom = n * curl_a_sq
        bp = T_para / denom
        bq = 0.5 * T_perp / denom
        if bp > 1e-5 and bq > 1e-5 and np.isfinite(bp) and np.isfinite(bq):
            bp_list.append(float(bp))
            bq_list.append(float(bq))


def build_data(
    radius,
    sol_radius,
    flux,
    n_init,
    steps_per_init,
    sample_every,
    alpha,
    seed,
    phi_diffuse_eta=0.1,
    ground_restore_k=0.0,
):
    """
    Multi-init MHD evolution with occupation continuity.

    For each initialisation:
      1. Reset phi to solenoid state.
      2. Draw fresh random occ (uniform direction, log-normal |v| ~ |curl A|, σ=1)
         and occ_n (log-uniform [0.01, 100]).
      3. Evolve: frozen_in → lorentz → stream → occ_n → phi_diffuse →
         (ground_restore, if enabled), repeat steps_per_init times.
      4. Sample β every sample_every steps.

    step_phi_diffuse provides the dissipative sink frozen_in/lorentz
    structurally lack (see project_mhd_energy_structural_nonconservation
    memory) -- without it this ensemble scatters unboundedly instead of
    clustering near the mirror/firehose thresholds (verified in
    tst/substrate/test_mhd.py::TestBrazilEnvelope).

    ground_restore_k > 0 additionally pulls phi back toward the solenoid's
    own ground state each step (step_ground_restore) -- validated to
    reduce field-direction (b_hat) drift in
    tst/substrate/test_mhd.py::TestGroundRestoreReducesFieldDrift; here
    we're checking whether it also helps the diamond/lens shape survive
    evolution (see project_brazil_plot_shape_investigation memory).
    ground_restore_k=0.0 (default) makes step_ground_restore an exact
    no-op, reproducing the original behaviour.
    """
    lat = SubstrateLattice(radius_cells=radius)
    _set_solenoid(lat, _solenoid_cells(lat, sol_radius), flux_quanta=flux)
    dt = lat.stable_dt()

    capture_ground_state(lat)  # lat.phi_bg = the solenoid's own ground state
    phi_solenoid = lat.phi_bg

    b_cell_list = [
        i
        for i in range(lat.N)
        if (bv := _b_vec(lat.phi, lat.nb, i)) is not None and np.dot(bv, bv) > 1e-24
    ]

    rng = np.random.default_rng(seed)
    bp_list, bq_list = [], []

    for _ in range(n_init):
        lat.phi[:] = phi_solenoid
        occ, occ_n = _init_occ(lat, b_cell_list, rng)

        for step in range(steps_per_init):
            step_frozen_in(lat, occ, dt, b_cell_list)
            step_occ_lorentz(lat, occ, dt, b_cell_list)
            step_occ_stream(lat, occ, dt, b_cell_list)
            step_occ_n(lat, occ, occ_n, dt, b_cell_list)
            step_phi_diffuse(lat, dt, eta=phi_diffuse_eta)
            if ground_restore_k > 0:
                step_ground_restore(lat, dt, k=ground_restore_k)

            if step % sample_every == 0:
                _sample_beta(lat, occ, occ_n, b_cell_list, bp_list, bq_list)

    return np.array(bp_list), np.array(bq_list), len(b_cell_list)


# ---------------------------------------------------------------------------
# Plot


def plot_brazil(bp, bq, n_b_cells, n_init, n_steps, sample_every, alpha, save=None):
    R = bq / bp

    # Filter to the visible region first (for density computation)
    mask = (bp > 1e-3) & (bp < 1e2) & (R > 0.1) & (R < 10.0)
    bp_v = bp[mask]
    R_v = R[mask]

    colors = _density_colors(bp_v, R_v, n_bins=70)

    # Reference-style colourmap: dark blue → cyan → green → yellow → red
    cmap = plt.cm.jet
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    bp_fill = np.logspace(-3, 2, 800)
    R_fh = _firehose_curve(bp_fill)
    R_mi = _mirror_curve(bp_fill)

    ax.plot(
        bp_fill,
        R_fh,
        color="white",
        lw=1.4,
        ls="--",
        zorder=3,
        label=r"Firehose  $R = 1 - 1/\beta_\parallel$",
    )
    ax.plot(
        bp_fill,
        R_mi,
        color="white",
        lw=1.4,
        ls="--",
        zorder=3,
        label=r"Mirror  $R = 1 + 1/\beta_\parallel$",
    )
    ax.axhline(1.0, color="#aaa", lw=0.7, ls=":", zorder=2)

    sc = ax.scatter(
        bp_v,
        R_v,
        c=colors,
        cmap=cmap,
        norm=norm,
        s=3,
        alpha=0.6,
        linewidths=0,
        zorder=4,
        rasterized=True,
    )

    cb = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.9)
    cb.set_label("log density  (arb)", fontsize=9, color="white")
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    cb.outline.set_edgecolor("white")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-3, 1e2)
    ax.set_ylim(0.1, 10.0)
    ax.set_xlabel(r"$\beta_\parallel$", fontsize=13, color="white")
    ax.set_ylabel(r"$T_\perp \,/\, T_\parallel$", fontsize=13, color="white")
    ax.set_title(
        "DST — pressure anisotropy\n"
        r"frozen-in $\Delta A = dt\,\mathrm{occ}{\times}\mathrm{curl\,}A$  ·  "
        r"Lorentz (Cayley) $\Delta\mathrm{occ} = dt\,\hat{a}{\times}\mathrm{occ}$  ·  "
        r"stream $\Delta\mathrm{occ}_{ij} \propto A_{ij}$  ·  "
        r"$\phi$-diffuse (equilibration)",
        fontsize=9,
        pad=8,
        color="white",
    )
    ax.tick_params(labelsize=8, colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")
    ax.grid(True, which="major", color="#444", lw=0.4, zorder=1)
    ax.grid(True, which="minor", color="#333", lw=0.2, zorder=1)

    ax.text(
        0.97,
        0.03,
        "FIREHOSE",
        color="#f88",
        fontsize=8,
        ha="right",
        va="bottom",
        transform=ax.transAxes,
    )
    ax.text(
        0.03,
        0.97,
        "MIRROR",
        color="#88f",
        fontsize=8,
        ha="left",
        va="top",
        transform=ax.transAxes,
    )

    ax.legend(
        fontsize=8,
        loc="lower right",
        framealpha=0.25,
        labelcolor="white",
        facecolor="black",
    )

    n_vis = int(mask.sum())
    n_tot = len(bp)
    ax.text(
        0.02,
        0.02,
        f"{n_vis:,} pts ({n_tot:,} total) · {n_b_cells} B-cells · "
        f"{n_init}×{n_steps} steps",
        fontsize=7,
        color="#999",
        transform=ax.transAxes,
        va="bottom",
    )

    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"saved → {save}")
    else:
        plt.show()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--radius", type=float, default=30.0)
    parser.add_argument("--sol-radius", type=float, default=6.0)
    parser.add_argument("--flux", type=int, default=1)
    parser.add_argument(
        "--n-init",
        type=int,
        default=20,
        help="number of random occ initialisations (default 20)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=800,
        help="MHD steps per initialisation (default 800)",
    )
    parser.add_argument("--sample-every", type=int, default=4)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="instability scattering strength (unused without instab_limit)",
    )
    parser.add_argument("--seed", type=int, default=20240630)
    parser.add_argument(
        "--phi-diffuse-eta",
        type=float,
        default=0.1,
        help="equilibration strength for step_phi_diffuse, the "
        "dissipative sink for frozen_in/lorentz (default 0.1)",
    )
    parser.add_argument(
        "--ground-restore-k",
        type=float,
        default=0.0,
        help="step_ground_restore rate pulling phi back toward "
        "the solenoid ground state each step (0.0 = off, "
        "the original behaviour; see "
        "project_electron_closure_hall_analogue memory)",
    )
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    bp, bq, n_b_cells = build_data(
        args.radius,
        args.sol_radius,
        args.flux,
        args.n_init,
        args.steps,
        args.sample_every,
        args.alpha,
        args.seed,
        phi_diffuse_eta=args.phi_diffuse_eta,
        ground_restore_k=args.ground_restore_k,
    )

    R = bq / bp
    mask = (bp > 1e-3) & (bp < 1e2) & (R > 0.1) & (R < 10.0)
    print(
        f"B-cells: {n_b_cells}  inits: {args.n_init}  "
        f"total: {len(bp):,}  visible: {mask.sum():,}  "
        f"bp=[{bp.min():.2e},{bp.max():.2e}]  R=[{R.min():.2e},{R.max():.2e}]"
    )

    plot_brazil(
        bp,
        bq,
        n_b_cells,
        args.n_init,
        args.steps,
        args.sample_every,
        args.alpha,
        save=args.save,
    )
