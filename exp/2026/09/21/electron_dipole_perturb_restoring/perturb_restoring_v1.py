"""
Perturb-and-check-restoring-force test on the real double-ring dipole
electron IC (init_electron_dipole_ic_vecpot).

Motivation (see project_chirality_mixing_two_component_mass,
project_dst_electron_dipole memories, and 09-21 conversation): every prior
free-electron soliton probe (v1-v8b, exp/2026/09/16-19) used a SINGLE-ring
IC, already flagged as structurally the wrong object to test the Dirac
chirality-mixing / "two coupled components needed for mass" argument. The
double-ring dipole IS the right structure, but it's already an EXACT static
fixed point of the plain reversible step at lam=0 (div_clean projects phi
onto the step's own divergence-free subspace, and uniform-u + div-free-phi
is fixed regardless of amplitude/profile -- a whole FAMILY of fixed points,
not an isolated one). So there is nothing to test by just running it -- any
"does it stay stable" check is vacuous. The real test is: perturb it OFF
that fixed-point family, then check whether the self-sourced lam
nonlinearity (step_lam_relax, lam_target = alpha*|phi|, already wired for
this exact IC in run/substrate/adaptive/electron_dipole.py) produces a
genuine RESTORING response, not just "doesn't grow" (the flawed
spread/growth framing that misled the single-ring v1-v4 threads).

FIRST ATTEMPT, RULED OUT (kept as a record, see git history for the v1
version that tried it): a per-cell multiplicative lobe reweighting
`phi *= (1 + eps*tanh(z/w))` -- multiplying all 14 faces of a given cell by
the SAME scalar. This does NOT break divergence-freeness: the discrete
divergence `div_i(phi) = sum_k W_k*phi[i,k]` is linear in phi with a single
per-cell multiplier, so that multiplier factors straight out of the sum --
`div_i(f_i*phi) = f_i*div_i(phi) = f_i*0 = 0` exactly, for ANY spatially
varying f, not just a uniform rescale. Confirmed directly: the "linear
control" run was frozen bit-for-bit (H and imbalance both unchanged to
printed precision over 3000 steps) -- the perturbation never left the
fixed-point family, so any apparent "restoring" behaviour from step_cross/
lam_relax on that construction was meaningless (compared against a control
that provably never moves).

Perturbation actually used: a localized additive GAUSSIAN BUMP on u,
`u = u_bg + eps*exp(-0.5*|pos-center|^2/w^2)`, centered on the dipole's own
centreline circle (x=r_axis, y=0, z=0), width w ~ 0.3*scale. This breaks
Δu=0 directly and unambiguously at t=0 -- guaranteed to source real dynamics
under plain step() too, unlike the multiplicative lobe attempt.

Diagnostic: E_bump_local(t), the field energy (u^2 + 0.5*sum(phi^2))
retained within a small FIXED window centered on the bump's initial
location (not the whole dipole envelope). The question: does the bump's
energy disperse at the same rate regardless of regime (no real interaction
with the dipole structure), or does self-sourced lam nonlinearity make it
decay SLOWER (energy retained/confined near the structure -- a genuine
attractive/restoring interaction) compared to the linear control (plain
wave-equation dispersion, expected to be regime-independent)? A secondary
diagnostic, phi_rms over the dipole's OWN envelope excluding the bump
window, checks whether the bump measurably excites the dipole's own
circulation (did it couple at all) as opposed to passing through inert.

Run:
  cd /home/p/git/eq
  python3 -u exp/2026/09/21/electron_dipole_perturb_restoring/perturb_restoring_v1.py
"""

import sys
import time

sys.path.insert(0, "/home/p/git/eq")

import numpy as np

from equilibrium import lattice as _lat
from equilibrium.substrate.adaptive import step_lam_relax
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import init_electron_dipole_ic_vecpot

R_DOMAIN = 30.0
SPH_FRAC = 0.15  # scale = R_DOMAIN * SPH_FRAC = 4.5 -- clear of the ~3-cell BCC diamond regime
SPONGE_FRACTION = 0.15
DAMP_MAX = 0.003
N_THREADS = 8

EPS_U = 0.05  # bump amplitude relative to the dipole's own phi amplitude=1.0 scale
BUMP_WIDTH_FRAC = 0.3  # bump width as a fraction of the dipole's own physical scale
N_SANITY_STEPS = 200
N_STEPS = 3000
REPORT_EVERY = 50

CROSS_SCALE = 0.01
LAM_RATE = 0.05
LAM_ALPHAS = [0.5, 1.0, 2.0]


def build_cluster():
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(N_THREADS)
    cl = alat[0]
    _lat.build_bcc_sphere(cl, R_DOMAIN, d1=1.0, u0=0.0, ell0=1.0, lam0=0.0)
    alat.rebuild(0)

    r_all = np.linalg.norm(np.array(cl.leaf_pos), axis=1)
    r_inner = float(r_all.max()) * (1.0 - SPONGE_FRACTION)
    cl.set_bc(
        _lat.BoundaryCondition.quadratic_sponge(
            target=0.0, damp_max=DAMP_MAX, r_all=r_all, r_inner=r_inner
        )
    )
    return alat, cl


def field_energy_parts(cl, mask_bump, mask_struct):
    phi_a = np.array(cl.leaf_phi)  # (14, N)
    u_a = np.array(cl.leaf_u)  # (N,)
    e = u_a**2 + 0.5 * np.sum(phi_a**2, axis=0)  # (N,)
    e_bump = float(e[mask_bump].sum())
    struct_phi_rms = float(np.sqrt(np.mean(phi_a[:, mask_struct] ** 2))) if mask_struct.any() else float("nan")
    return float(e.sum()), e_bump, struct_phi_rms


def _load(cl, phi, u, lam):
    cl.set_leaf_phi(phi.copy())
    cl.set_leaf_u(u.copy())
    cl.set_leaf_lam(lam.copy())


def _advance_one(alat, cl, dt, cross_scale, lam_rate, lam_alpha):
    alat.step(0, dt)
    alat.step_sponge(0)
    if cross_scale > 0.0:
        alat.step_cross(0, dt, cross_scale)
    if lam_rate > 0.0:
        step_lam_relax(cl, dt, lam_rate, lam_alpha)
    return np.array(cl.leaf_phi), np.array(cl.leaf_u), np.array(cl.leaf_lam)


def run_pair(label, alat, cl, phi_clean, u_clean, lam_clean, bump, mask_bump, mask_struct, dt, cross_scale, lam_rate, lam_alpha):
    """Advance TWO worlds (with-bump, reference-no-bump) under identical
    operators, diffing their fields at each report step to isolate the
    bump's own energy from the dipole's own (much larger) static field --
    see module docstring for why a single-world windowed-energy read was
    contaminated by the background."""
    phi_A, u_A, lam_A = phi_clean.copy(), u_clean + bump, lam_clean.copy()
    phi_B, u_B, lam_B = phi_clean.copy(), u_clean.copy(), lam_clean.copy()

    times, ebump_hist, struct_hist = [], [], []
    t0 = time.time()
    for step in range(N_STEPS + 1):
        if step % REPORT_EVERY == 0:
            du = u_A - u_B
            dphi = phi_A - phi_B
            de = du**2 + 0.5 * np.sum(dphi**2, axis=0)  # excess (bump-induced) energy density
            e_bump = float(de[mask_bump].sum())
            struct_rms = float(np.sqrt(np.mean(dphi[:, mask_struct] ** 2))) if mask_struct.any() else float("nan")
            times.append(step * dt)
            ebump_hist.append(e_bump)
            struct_hist.append(struct_rms)
        if step < N_STEPS:
            _load(cl, phi_A, u_A, lam_A)
            phi_A, u_A, lam_A = _advance_one(alat, cl, dt, cross_scale, lam_rate, lam_alpha)
            _load(cl, phi_B, u_B, lam_B)
            phi_B, u_B, lam_B = _advance_one(alat, cl, dt, cross_scale, lam_rate, lam_alpha)

    times = np.array(times)
    ebump_hist = np.array(ebump_hist)
    struct_hist = np.array(struct_hist)

    retained_frac = ebump_hist[-1] / ebump_hist[0] if ebump_hist[0] > 0 else float("nan")
    struct_growth = (struct_hist[-1] - struct_hist[0]) / struct_hist[0] if struct_hist[0] > 0 else float("nan")

    print(
        f"[{label:>18s}] E_bump0={ebump_hist[0]:.4e} E_bump_final={ebump_hist[-1]:.4e} "
        f"retained={retained_frac:.4f}  struct_dphi_rms_growth={struct_growth:+.4e}  "
        f"({time.time()-t0:.1f}s)",
        flush=True,
    )
    return {
        "label": label,
        "times": times,
        "E_bump": ebump_hist,
        "struct_dphi_rms": struct_hist,
        "retained_frac": float(retained_frac),
        "struct_growth": float(struct_growth),
    }


def main():
    alat, cl = build_cluster()
    centre = np.zeros(3)
    scale = R_DOMAIN * SPH_FRAC

    b_max, r_axis = init_electron_dipole_ic_vecpot(
        cl,
        R_sph=1.0,
        scale=scale,
        B0=1.0,
        lam_target=0.0,
        u_bg=0.0,
        amplitude=1.0,
        div_clean=True,
        clean_rtol=1e-8,
    )
    dt = compute_dt(cl)
    print(f"N={cl.n_leaves}  R_domain={R_DOMAIN}  scale={scale:.2f}  r_axis={r_axis:.3f}  dt={dt:.4e}", flush=True)

    pos = np.array(cl.leaf_pos)
    r_cell = np.linalg.norm((pos - centre) / scale, axis=1)
    mask_struct_full = r_cell <= 1.0
    print(f"dipole-envelope cells: {int(mask_struct_full.sum())}/{cl.n_leaves}", flush=True)

    bump_center = np.array([r_axis, 0.0, 0.0])
    bump_w = BUMP_WIDTH_FRAC * scale
    r_bump = np.linalg.norm(pos - bump_center, axis=1)
    mask_bump = r_bump <= 3.0 * bump_w  # fixed window tracking the bump's initial footprint
    mask_struct = mask_struct_full & ~mask_bump  # dipole envelope, excluding the bump's own footprint
    print(
        f"bump window cells: {int(mask_bump.sum())}  struct-excl-bump cells: {int(mask_struct.sum())}",
        flush=True,
    )

    # --- sanity check: unperturbed IC should be an exact fixed point ---
    H0, eb0, sr0 = field_energy_parts(cl, mask_bump, mask_struct)
    for _ in range(N_SANITY_STEPS):
        alat.step(0, dt)
        alat.step_sponge(0)
    H1, eb1, sr1 = field_energy_parts(cl, mask_bump, mask_struct)
    print(
        f"sanity ({N_SANITY_STEPS} steps, no perturbation, no irreversible ops): "
        f"H {H0:.6e} -> {H1:.6e} (rel {abs(H1-H0)/H0:.2e})",
        flush=True,
    )

    # rebuild clean IC (sanity run above already stepped it) and perturb
    init_electron_dipole_ic_vecpot(
        cl, R_sph=1.0, scale=scale, B0=1.0, lam_target=0.0, u_bg=0.0,
        amplitude=1.0, div_clean=True, clean_rtol=1e-8,
    )
    phi_clean = np.array(cl.leaf_phi)  # (14, N)
    u_clean = np.array(cl.leaf_u)
    lam_clean = np.array(cl.leaf_lam)

    bump = EPS_U * np.exp(-0.5 * (r_bump / bump_w) ** 2)
    e_bump0 = float((bump**2)[mask_bump].sum())
    print(f"bump-only excess energy at t=0 (eps_u={EPS_U}, w={bump_w:.2f}): E_bump={e_bump0:.4e}", flush=True)

    print(f"\n=== regime sweep (paired with-bump vs. reference), {N_STEPS} steps each ===", flush=True)
    results = []
    results.append(
        run_pair("linear (control)", alat, cl, phi_clean, u_clean, lam_clean, bump, mask_bump, mask_struct, dt, 0.0, 0.0, 0.0)
    )
    results.append(
        run_pair("step_cross only", alat, cl, phi_clean, u_clean, lam_clean, bump, mask_bump, mask_struct, dt, CROSS_SCALE, 0.0, 0.0)
    )
    for alpha in LAM_ALPHAS:
        results.append(
            run_pair(
                f"lam_relax a={alpha}", alat, cl, phi_clean, u_clean, lam_clean, bump, mask_bump, mask_struct, dt, 0.0, LAM_RATE, alpha
            )
        )

    print("\n=== summary ===", flush=True)
    lin_retained = results[0]["retained_frac"]
    print(f"linear control: E_bump retained fraction = {lin_retained:.4f}  (baseline free-dispersion rate)", flush=True)
    for r in results[1:]:
        more_retained = r["retained_frac"] > lin_retained * 1.1
        print(
            f"  {r['label']:>18s}: retained={r['retained_frac']:.4f}  struct_growth={r['struct_growth']:+.4e}  "
            f"{'MORE retained than linear (confinement signal)' if more_retained else 'no clear separation from linear'}",
            flush=True,
        )

    out = {"times": results[0]["times"]}
    for r in results:
        key = r["label"].replace(" ", "_").replace("=", "").replace("(", "").replace(")", "")
        out[f"Ebump_{key}"] = r["E_bump"]
        out[f"structdphi_{key}"] = r["struct_dphi_rms"]
    np.savez("exp/2026/09/21/electron_dipole_perturb_restoring/results_bump.npz", **out)
    print("\nsaved results_bump.npz", flush=True)


if __name__ == "__main__":
    main()
