"""
test_cross_impulse.py — immediate single-step response of each crossing coupling.

Uses deliberately imbalanced ICs so the u coupling is maximal and visible
in the first 1-5 steps without running a long simulation.

Three ICs tested:
  monopole_out  — phi[center, all 14 faces] = +A   div = +14A (source)
  monopole_in   — phi[center, all 14 faces] = -A   div = -14A (sink)
  cross_hex     — phi[center, face 0] = +A, phi[center, face 3] = +A
                  Two edge-adjacent hex faces — the literal step_cross event.

Five coupling schemes at each IC, compared side by side at every step:
  baseline  — phi only, no du
  maxwell   — du ∝ -phi·dphi   (Faraday back-EMF)
  qed       — du ∝ +phi²       (vertex amplitude, always positive)
  dst       — du ∝ -σ·Δdiv     (wave-equation consistency)
  vec_pot   — du ∝ -div(phi)   (Lorenz-gauge vector potential)

Run:
  PYTHONPATH=$PWD python3 run/substrate/cross_impulse.py
  PYTHONPATH=$PWD python3 run/substrate/cross_impulse.py --amplitude 10 --n-steps 3
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")

from equilibrium.substrate.lattice import DampedLattice

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=6.0)
parser.add_argument("--scale", type=float, default=0.01)
parser.add_argument("--sigma", type=float, default=1.0)
parser.add_argument("--n-steps", type=int, default=6)
parser.add_argument("--amplitude", type=float, default=1.0)
parser.add_argument(
    "--lam-center",
    type=float,
    default=1.0,
    help="lam at center cell for maxwell_chi (peaks at lam=1, vanishes at 0 and ∞)",
)
parser.add_argument(
    "--no-wave",
    action="store_true",
    help="Skip lat.step() — isolate the crossing du contribution only",
)

# ---------------------------------------------------------------------------
# Face topology
# ---------------------------------------------------------------------------
_K_REV = np.array([7, 6, 5, 4, 3, 2, 1, 0, 9, 8, 11, 10, 13, 12], dtype=np.intp)
_FACE_ADJ = np.full((14, 6), -1, dtype=np.int32)
_FACE_ADJ[0] = [3, 5, 6, 8, 10, 12]
_FACE_ADJ[1] = [2, 4, 7, 8, 10, 13]
_FACE_ADJ[2] = [1, 4, 7, 8, 11, 12]
_FACE_ADJ[3] = [0, 5, 6, 8, 11, 13]
_FACE_ADJ[4] = [1, 2, 7, 9, 10, 12]
_FACE_ADJ[5] = [0, 3, 6, 9, 10, 13]
_FACE_ADJ[6] = [0, 3, 5, 9, 11, 12]
_FACE_ADJ[7] = [1, 2, 4, 9, 11, 13]
_FACE_ADJ[8, :4] = [0, 1, 2, 3]
_FACE_ADJ[9, :4] = [4, 5, 6, 7]
_FACE_ADJ[10, :4] = [0, 1, 4, 5]
_FACE_ADJ[11, :4] = [2, 3, 6, 7]
_FACE_ADJ[12, :4] = [0, 2, 4, 6]
_FACE_ADJ[13, :4] = [1, 3, 5, 7]
_N_ADJ = np.array(
    [len([x for x in _FACE_ADJ[k] if x >= 0]) for k in range(14)], dtype=int
)


# ---------------------------------------------------------------------------
# Crossing kernels
# ---------------------------------------------------------------------------


def _cross_phi_delta(phi, nb, dt, scale):
    dphi = np.zeros_like(phi)
    delta = -dt * scale * phi
    for k in range(14):
        n_adj = int(_N_ADJ[k])
        mk = int(_K_REV[k])
        adj = [j for j in _FACE_ADJ[k] if j >= 0]
        valid_k = nb[:, k] >= 0
        nb_k = np.where(valid_k, nb[:, k], 0)
        dphi[:, k] += n_adj * delta[:, k]
        np.add.at(dphi, (nb_k[valid_k], mk), -(n_adj * delta[valid_k, k]))
        for j in adj:
            mj = int(_K_REV[j])
            valid_j = nb[:, j] >= 0
            nb_j = np.where(valid_j, nb[:, j], 0)
            dphi[:, j] += delta[:, k]
            np.add.at(dphi, (nb_j[valid_j], mj), -delta[valid_j, k])
    return dphi, delta


def _apply(lat, dphi, du):
    lat.phi = np.where(lat.fixed_phi, lat.phi, lat.phi + dphi)
    lat.u += du


def cross_baseline(lat, dt, scale, **_):
    dphi, _ = _cross_phi_delta(lat.phi, lat.nb, dt, scale)
    _apply(lat, dphi, np.zeros(lat.N))


def cross_maxwell(lat, dt, scale, **_):
    p = lat.phi.copy()
    dphi, _ = _cross_phi_delta(p, lat.nb, dt, scale)
    _apply(lat, dphi, -(p * dphi).sum(axis=1))


def cross_qed(lat, dt, scale, **_):
    p = lat.phi.copy()
    dphi, _ = _cross_phi_delta(p, lat.nb, dt, scale)
    _apply(lat, dphi, (p**2 * scale * dt).sum(axis=1))


def cross_dst(lat, dt, scale, sigma=1.0, **_):
    p = lat.phi.copy()
    dphi, _ = _cross_phi_delta(p, lat.nb, dt, scale)
    _apply(lat, dphi, -sigma * dphi.sum(axis=1))


def cross_vec_pot(lat, dt, scale, **_):
    p = lat.phi.copy()
    dphi, _ = _cross_phi_delta(p, lat.nb, dt, scale)
    _apply(lat, dphi, -scale * dt * p.sum(axis=1))


def cross_maxwell_chi(lat, dt, scale, **_):
    """
    Maxwell back-EMF gated by chi × sigma = lam/(1+lam)²:
      du_i = -(phi_old · dphi).sum_k × chi_i × sigma_i

    phi·dphi is phi²-dominated → always positive (Joule heating). Correct:
    field energy released by crossing always heats u. Δdiv vanishes for
    div-free fields (Harris sheet IC), making it useless there — phi·dphi
    is nonzero for any nonzero field.

    The chi × sigma gate:
      lam=0  (vacuum)     : chi×sigma → 0, no coupling  ✓
      lam=1  (electron/p) : chi×sigma = 0.25, peak       ✓
      lam→∞  (over-dense) : chi×sigma → 0, saturated    ✓
    """
    p = lat.phi.copy()
    chi_sg = lat.lam / (1.0 + lat.lam) ** 2
    dphi, _ = _cross_phi_delta(p, lat.nb, dt, scale)
    _apply(lat, dphi, -(p * dphi).sum(axis=1) * chi_sg)


SCHEMES = {
    "baseline": (cross_baseline, {}),
    "maxwell": (cross_maxwell, {}),
    "maxwell_chi": (cross_maxwell_chi, {}),
    "qed": (cross_qed, {}),
    "dst": (cross_dst, {}),
    "vec_pot": (cross_vec_pot, {}),
}


# ---------------------------------------------------------------------------
# Runner: one IC, all schemes, N steps
# ---------------------------------------------------------------------------


def run_all(lat, state0, center, neighbors, n_steps, dt, scale, sigma, no_wave):
    results = {}
    for name, (fn, kw) in SCHEMES.items():
        lat.u[:] = state0["u"]
        lat.phi[:] = state0["phi"]
        lat.lam[:] = state0["lam"]
        kw2 = dict(kw)
        if "sigma" in fn.__code__.co_varnames:
            kw2["sigma"] = sigma
        history = []
        for s in range(n_steps + 1):
            nb_u = float(lat.u[neighbors].mean()) if len(neighbors) else 0.0
            history.append(
                {
                    "step": s,
                    "u_center": float(lat.u[center]),
                    "u_nb_mean": nb_u,
                    "div_center": float(lat.phi[center].sum()),
                    "phi_rms": float(np.sqrt(np.mean(lat.phi**2))),
                    "E_u": float(0.5 * np.sum(lat.u**2)),
                }
            )
            if s < n_steps:
                if not no_wave:
                    lat.step(dt)
                fn(lat, dt, scale, **kw2)
        results[name] = history
    return results


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_ic_results(ic_name, div0, results, n_steps):
    cols = ["u_center", "u_nb_mean", "div_center", "phi_rms", "E_u"]
    col_w = 13
    name_w = 10

    print(f"\n{'═'*80}")
    print(f"  IC: {ic_name}   div(center)₀ = {div0:+.4f}")
    print(f"{'═'*80}")

    hdr = f"  {'scheme':<{name_w}}" + "".join(f"  {c:>{col_w}}" for c in cols)
    sep = "  " + "-" * (len(hdr) - 2)

    for s in range(n_steps + 1):
        print(f"\n  step {s}")
        print(hdr)
        print(sep)
        for name in SCHEMES:
            rec = results[name][s]
            row = f"  {name:<{name_w}}"
            for c in cols:
                v = rec[c]
                # Show Δ vs baseline in parens for non-baseline schemes
                if name == "baseline":
                    row += f"  {v:>{col_w}.4e}"
                else:
                    row += f"  {v:>{col_w}.4e}"
            print(row)

        # Δu at center vs baseline, shown separately for clarity
        if s > 0:
            print("\n    Δu_center vs baseline:")
            base_u = results["baseline"][s]["u_center"]
            for name in SCHEMES:
                if name == "baseline":
                    continue
                du = results[name][s]["u_center"] - base_u
                print(f"      {name:<10}  {du:+.4e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parser.parse_args()

    lat = DampedLattice(radius_cells=args.R, sponge_fraction=0.15, damp_max=0.003)
    dt = lat.stable_dt()

    dists = np.linalg.norm(lat.pos, axis=1)
    center = int(np.argmin(dists))
    nb_raw = lat.nb[center]
    neighbors = nb_raw[nb_raw >= 0]

    chi_center = args.lam_center / (1.0 + args.lam_center)
    print(f"R={args.R}  N={lat.N}  dt={dt:.4e}  scale={args.scale}  A={args.amplitude}")
    print(
        f"center cell {center}  pos={lat.pos[center].round(3)}  {len(neighbors)} neighbours"
    )
    print(
        f"lam_center={args.lam_center}  chi_center={chi_center:.4f}  "
        f"(maxwell_chi scales by chi; set --lam-center 0 for vacuum)"
    )

    A = args.amplitude

    ICs = {
        "monopole_out": lambda: np.full(14, +A),  # all faces outward: div = +14A
        "monopole_in": lambda: np.full(14, -A),  # all faces inward:  div = -14A
        "cross_hex_0_3": lambda: np.array(  # faces 0 and 3 are edge-adjacent hex faces
            [A if k in (0, 3) else 0.0 for k in range(14)]
        ),
    }

    state0_base = {"u": lat.u.copy(), "phi": lat.phi.copy(), "lam": lat.lam.copy()}
    state0_base["lam"][center] = args.lam_center  # set mass at center for chi gating

    for ic_name, ic_phi_fn in ICs.items():
        lat.u[:] = state0_base["u"]
        lat.phi[:] = state0_base["phi"]
        lat.lam[:] = state0_base["lam"]

        lat.phi[center] = ic_phi_fn()
        div0 = float(lat.phi[center].sum())
        state0 = {"u": lat.u.copy(), "phi": lat.phi.copy(), "lam": lat.lam.copy()}

        results = run_all(
            lat,
            state0,
            center,
            neighbors,
            args.n_steps,
            dt,
            args.scale,
            args.sigma,
            args.no_wave,
        )
        print_ic_results(ic_name, div0, results, args.n_steps)

    # -------------------------------------------------------------------------
    # Step-1 summary across all ICs — the most diagnostic single table
    # -------------------------------------------------------------------------
    print(f"\n{'═'*80}")
    print("  STEP-1 Δu_center SUMMARY  (scheme − baseline)")
    print(f"{'═'*80}")
    print(
        f"  {'scheme':<10}  {'monopole_out':>14}  {'monopole_in':>14}  {'cross_hex':>14}"
    )
    print("  " + "-" * 56)

    # Re-run to collect summary (results dict overwritten in the main loop above)
    summary = {ic: {} for ic in ICs}
    for ic_name, ic_phi_fn in ICs.items():
        lat.u[:] = state0_base["u"]
        lat.phi[:] = state0_base["phi"]
        lat.lam[:] = state0_base["lam"]
        lat.lam[center] = args.lam_center
        lat.phi[center] = ic_phi_fn()
        state0 = {"u": lat.u.copy(), "phi": lat.phi.copy(), "lam": lat.lam.copy()}
        res = run_all(
            lat, state0, center, neighbors, 1, dt, args.scale, args.sigma, args.no_wave
        )
        base_u1 = res["baseline"][1]["u_center"]
        for name in SCHEMES:
            summary[ic_name][name] = res[name][1]["u_center"] - base_u1

    for name in SCHEMES:
        if name == "baseline":
            continue
        row = f"  {name:<10}"
        for ic_name in ICs:
            row += f"  {summary[ic_name][name]:>+14.4e}"
        print(row)

    print()
    print("  Interpretation guide:")
    print(
        "    maxwell      — always positive: back-EMF heats u regardless of flux sign"
    )
    print(
        "    maxwell_chi  — gated by chi×sigma=lam/(1+lam)²: zero in vacuum, peaks at lam=1, zero at lam→∞"
    )
    print("    qed          — always positive, sign-blind vertex amplitude")
    print(
        "    dst          — flips with div sign: stabilising counter-force to wave step"
    )
    print("    vec_pot      — sign = -sign(div): slightly amplifies the wave step")


if __name__ == "__main__":
    main()
