"""
validate_step_cross.py — compare crossing coupling prescriptions.

When a winding crosses through a cell, phi is redistributed.  The crossing
is not free: it involves electromagnetic induction (Faraday), current flow
(Ampere/Ohm), and a Lorentz back-reaction on the potential field u.

Three explicit prescriptions are compared against the baseline (no du coupling):

  Maxwell  — energy balance via Faraday induction.
             Each face crossing transfers field energy B·dB into u:
               du_i += -Σ_k  phi[k] · dphi[k]          (first-order ΔE_B → u)
             Positive where phi drains (energy released), negative where phi
             is injected (energy absorbed).  Conserves E_B + E_u exactly.

  QED      — vertex-amplitude coupling.
             Crossing amplitude M ∝ phi (field strength at the vertex).
             Energy exchange ∝ M²:
               du_i += Σ_k  phi[k]²  · scale · dt        (always ≥ 0)
             Sign-blind: every crossing event injects energy into u regardless
             of flux direction.  Models spontaneous photon emission at the
             reconnection vertex.

  DST      — wave-equation consistency.
             Apply the wave update DU = -sigma · Δdiv(phi) immediately inside
             the crossing step, rather than waiting for the next wave tick:
               du_i += -sigma · Σ_k dphi_i[k]
             Linear in phi (like the wave step).  Removes the one-tick delay
             between topology change and pressure response.

  VecPot   — Lorenz-gauge vector potential coupling.
             From ∂φ/∂t + c²∇·A = 0 with A[k] ∝ phi[k]:
               du_i += -scale · dt · Σ_k phi_i[k]
             Responds to instantaneous div(phi), not just Δdiv from this step.
             DST only responds when crossing changes the divergence; VecPot
             also responds to pre-existing divergence at every crossing step.

IC: DampedLattice with a single toroidal ring (DC circulation, no u field).
Observables tracked at each report step:
  phi_rms       — overall field amplitude (should decay with crossing)
  circ_ring     — azimuthal phi circulation on ring tiles (topology proxy)
  u_rms         — pressure perturbation from crossing (0 in baseline)
  E_phi         — field energy  Σ phi²
  E_u           — pressure energy  Σ u²
  E_total       — E_phi + E_u  (should be conserved in Maxwell; not in others)

Run:
  PYTHONPATH=$PWD python3 run/substrate/validate_step_cross.py
  PYTHONPATH=$PWD python3 run/substrate/validate_step_cross.py --n-steps 5000 --scale 0.05
"""

import argparse
import multiprocessing as mp
import sys

import numpy as np

sys.path.insert(0, ".")

from equilibrium.substrate.flux_tube import (
    _MIRROR,
    find_toroidal_ring,
)
from equilibrium.substrate.lattice import DampedLattice

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=20.0)
parser.add_argument("--scale", type=float, default=0.01, help="crossing scale")
parser.add_argument(
    "--sigma", type=float, default=1.0, help="wave throughput σ for DST coupling"
)
parser.add_argument("--n-steps", type=int, default=2000)
parser.add_argument("--report", type=int, default=200)
parser.add_argument("--amplitude", type=float, default=1.0)
parser.add_argument(
    "--lam-ring",
    type=float,
    default=1.0,
    help="lam on ring tiles for maxwell_chi gating (0=vacuum, higher=more mass)",
)


# ---------------------------------------------------------------------------
# Face tables (mirror of lattice.py internals)
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
# Crossing implementations
# ---------------------------------------------------------------------------


def _cross_phi_delta(phi, nb, dt, scale):
    """Compute dphi and per-face delta arrays (shared by all approaches)."""
    dphi = np.zeros_like(phi)
    # delta[i,k] = -dt * scale * phi[i,k]  — signed, acts on all non-zero faces
    delta = -dt * scale * phi  # (N,14)

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


def step_cross_baseline(lat, dt, scale):
    """Current implementation — phi only, no du coupling."""
    nb = lat.nb
    dphi, _ = _cross_phi_delta(lat.phi, nb, dt, scale)
    lat.phi = np.where(lat.fixed_phi, lat.phi, lat.phi + dphi)


def step_cross_maxwell(lat, dt, scale):
    """
    Maxwell / Faraday coupling.

    Field energy E_B = ½ Σ phi² changes by:
      ΔE_B ≈ Σ_{i,k} phi[i,k] · dphi[i,k]   (first-order)

    By energy conservation this goes into u:
      du_i = -Σ_k phi[i,k] · dphi[i,k]

    Physical interpretation: when phi drains from face k, the inductance of
    the winding releases magnetic energy into the local potential (back-EMF).
    When phi is injected into adj faces, energy is drawn from u.

    Net effect: E_phi + E_u is conserved (up to sponge/BC losses).
    du is bipolar — positive where flux drains, negative where injected.
    """
    nb = lat.nb
    phi_old = lat.phi.copy()
    dphi, _ = _cross_phi_delta(phi_old, nb, dt, scale)

    # du_i = -Σ_k phi_old[i,k] · dphi[i,k]
    du = -(phi_old * dphi).sum(axis=1)  # (N,)

    lat.phi = np.where(lat.fixed_phi, lat.phi, lat.phi + dphi)
    lat.u += du


def step_cross_qed(lat, dt, scale):
    """
    QED vertex-amplitude coupling.

    The crossing amplitude at face k is M_k ∝ phi[k].
    Energy exchanged at each vertex ∝ M² = phi[k]²:
      du_i += Σ_k phi[i,k]² · scale · dt

    Physical interpretation: each crossing event emits a virtual photon
    whose energy is proportional to the squared field amplitude.  The
    energy is always injected into u (sign-blind vertex).

    Net effect: E_phi decreases, E_u always increases — not conservative
    in the classical sense.  Models dissipative reconnection (resistive).
    """
    nb = lat.nb
    phi_old = lat.phi.copy()
    dphi, _ = _cross_phi_delta(phi_old, nb, dt, scale)

    du = (phi_old**2 * scale * dt).sum(axis=1)  # (N,), always ≥ 0

    lat.phi = np.where(lat.fixed_phi, lat.phi, lat.phi + dphi)
    lat.u += du


def step_cross_dst(lat, dt, scale, sigma=1.0):
    """
    DST wave-equation consistency coupling.

    The crossing changes div(phi) by Δdiv_i = Σ_k dphi_i[k].
    The wave equation says du/dt = -sigma · div(phi).
    Rather than waiting for the next wave tick, apply the response immediately:
      du_i = -sigma · Δdiv_i

    Physical interpretation: the right-hand rule in DST — the crossing
    redirects flux, which immediately changes the local divergence and
    hence the pressure u.  No one-tick delay between topology change and
    pressure response.

    Linear in phi (like the wave step), sign-consistent with induction.
    """
    nb = lat.nb
    phi_old = lat.phi.copy()
    dphi, _ = _cross_phi_delta(phi_old, nb, dt, scale)

    # Δdiv[i] = Σ_k dphi[i,k]
    delta_div = dphi.sum(axis=1)  # (N,)
    du = -sigma * delta_div  # (N,)

    lat.phi = np.where(lat.fixed_phi, lat.phi, lat.phi + dphi)
    lat.u += du


def step_cross_maxwell_chi(lat, dt, scale):
    """
    Maxwell back-EMF gated by chi × sigma = lam/(1+lam)²:
      du_i = -(phi_old · dphi).sum_k × chi_i × sigma_i

    phi·dphi is phi²-dominated → always positive (Joule heating). This is the
    correct formula: field energy released by the crossing always heats u.
    Δdiv vanishes for div-free fields (exact Harris sheet IC), so Δdiv would
    give du=0 everywhere — phi·dphi is the right quantity for any nonzero field.

    The chi × sigma gate:
      lam=0  (vacuum)     : chi×sigma → 0, no coupling                    ✓
      lam=1  (electron/p) : chi×sigma = 0.25, maximum coupling             ✓
      lam→∞  (over-dense) : chi×sigma → 0, saturated → re-radiates as phi ✓
    """
    nb = lat.nb
    phi_old = lat.phi.copy()
    chi_sg = lat.lam / (1.0 + lat.lam) ** 2  # lam/(1+lam)²
    dphi, _ = _cross_phi_delta(phi_old, nb, dt, scale)

    du = -(phi_old * dphi).sum(axis=1) * chi_sg

    lat.phi = np.where(lat.fixed_phi, lat.phi, lat.phi + dphi)
    lat.u += du


def step_cross_vector_potential(lat, dt, scale):
    """
    Vector potential (Lorenz-gauge) coupling.

    From the Lorenz gauge condition: ∂φ/∂t + c²∇·A = 0.
    With the vector potential A[k] ∝ phi[k] (flux IS the field on each face):
      du/dt = -c²∇·A  →  du_i += -scale · dt · Σ_k phi_i[k]

    Responds to the instantaneous div(phi), not the incremental Δdiv from
    this step (contrast with DST which only responds to the crossing delta).

    Key difference from DST:
      - DST:  du ∝ Δdiv(phi) — responds only to what THIS step changes.
      - VP:   du ∝ div(phi)  — responds to total current divergence.
    For a perfectly div-free winding both give zero coupling (correct).
    Any residual divergence from imperfect ICs or prior crossings keeps
    driving u in the VP scheme even between crossings.

    Derived from Lorenz gauge, not a phenomenological choice.
    """
    nb = lat.nb
    phi_old = lat.phi.copy()
    dphi, _ = _cross_phi_delta(phi_old, nb, dt, scale)

    # Lorenz coupling: du = -scale * div(phi) * dt
    du = -scale * dt * phi_old.sum(axis=1)  # (N,), bipolar

    lat.phi = np.where(lat.fixed_phi, lat.phi, lat.phi + dphi)
    lat.u += du


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def phi_ring_circ(lat, ring_idx):
    """Azimuthal phi circulation on ring tiles."""
    if not ring_idx:
        return 0.0
    idx = np.array(sorted(ring_idx), dtype=int)
    nb_raw = lat.nb[idx]
    nb_s = np.where(nb_raw >= 0, nb_raw, 0)
    pi, pj = lat.pos[idx], lat.pos[nb_s]
    diff = pj - pi[:, np.newaxis, :]
    d_len = np.linalg.norm(diff, axis=-1, keepdims=True)
    n_hat = diff / np.where(d_len > 0, d_len, 1.0)
    xi, yi = pi[:, 0, np.newaxis], pi[:, 1, np.newaxis]
    rho = np.sqrt(xi**2 + yi**2)
    rho_s = np.where(rho > 1e-10, rho, 1.0)
    phx, phy = -yi / rho_s, xi / rho_s
    proj = n_hat[..., 0] * phx + n_hat[..., 1] * phy
    valid = lat._valid[idx]
    return float(np.sum(lat.phi[idx] * proj * valid))


def snapshot(lat, ring_idx, step, dt):
    phi_rms = float(np.sqrt(np.mean(lat.phi**2)))
    u_rms = float(np.sqrt(np.mean(lat.u**2)))
    circ = phi_ring_circ(lat, ring_idx)
    E_phi = float(0.5 * np.sum(lat.phi**2))
    E_u = float(0.5 * np.sum(lat.u**2))
    return dict(
        step=step,
        t=step * dt,
        phi_rms=phi_rms,
        u_rms=u_rms,
        circ=circ,
        E_phi=E_phi,
        E_u=E_u,
        E_total=E_phi + E_u,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run(name, step_cross_fn, lat, state0, ring_idx, n_steps, report, dt, scale, **kw):
    # Restore initial state arrays (avoids deepcopy of tile graph)
    lat.u[:] = state0["u"]
    lat.phi[:] = state0["phi"]
    lat.lam[:] = state0["lam"]
    history = [snapshot(lat, ring_idx, 0, dt)]
    for s in range(1, n_steps + 1):
        lat.step(dt)
        step_cross_fn(lat, dt, scale, **kw)
        if s % report == 0:
            history.append(snapshot(lat, ring_idx, s, dt))
    return history


# Map name → function for pickling across processes
_STEP_FNS = {
    "baseline": step_cross_baseline,
    "maxwell": step_cross_maxwell,
    "maxwell_chi": step_cross_maxwell_chi,
    "qed": step_cross_qed,
    "dst": step_cross_dst,
    "vec_pot": step_cross_vector_potential,
}


def _worker(task):
    """Top-level worker — picklable, runs in its own subprocess."""
    name, fn_name, R, state0, ring_tiles, n_steps, report, dt, scale, kw = task
    lat = DampedLattice(radius_cells=R, sponge_fraction=0.15, damp_max=0.003)
    ring_idx = set(ring_tiles)
    history = run(
        name,
        _STEP_FNS[fn_name],
        lat,
        state0,
        ring_idx,
        n_steps,
        report,
        dt,
        scale,
        **kw,
    )
    return name, history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parser.parse_args()

    # Build reference lattice
    lat0 = DampedLattice(radius_cells=args.R, sponge_fraction=0.15, damp_max=0.003)
    dt = lat0.stable_dt()

    A_tiles, A_faces = find_toroidal_ring(lat0, axis="z")
    if A_tiles is None:
        print("ERROR: ring not found — increase --R")
        sys.exit(1)

    # DC circulation IC (div-free, simple to interpret)
    N_ring = len(A_tiles)
    for j in range(N_ring):
        i = A_tiles[j]
        nxt = A_tiles[(j + 1) % N_ring]
        k = A_faces[j]
        mk = int(_MIRROR[k])
        lat0.phi[i, k] = args.amplitude
        lat0.phi[nxt, mk] = -args.amplitude

    ring_idx = set(A_tiles)
    ring_r = float(np.mean([np.linalg.norm(lat0.pos[i]) for i in ring_idx]))
    circ0 = phi_ring_circ(lat0, ring_idx)
    chi_ring = args.lam_ring / (1.0 + args.lam_ring)
    print(f"R={args.R}  N={lat0.N}  ring={len(A_tiles)} tiles  r≈{ring_r:.2f}")
    print(f"dt={dt:.4e}  scale={args.scale}  amplitude={args.amplitude}")
    print(
        f"lam_ring={args.lam_ring}  chi_ring={chi_ring:.4f}  (maxwell_chi gating on ring tiles)"
    )
    print(f"initial circ={circ0:.4f}  phi_rms={np.sqrt(np.mean(lat0.phi**2)):.4f}")
    print()

    # Set lam on ring tiles so maxwell_chi has mass to work with
    for i in A_tiles:
        lat0.lam[i] = args.lam_ring

    # Save initial state to restore between runs (avoids deepcopy of tile graph)
    state0 = {
        "u": lat0.u.copy(),
        "phi": lat0.phi.copy(),
        "lam": lat0.lam.copy(),
    }

    runs = {
        "baseline": (step_cross_baseline, {}),
        "maxwell": (step_cross_maxwell, {}),
        "maxwell_chi": (step_cross_maxwell_chi, {}),
        "qed": (step_cross_qed, {}),
        "dst": (step_cross_dst, {"sigma": args.sigma}),
        "vec_pot": (step_cross_vector_potential, {}),
    }

    tasks = [
        (
            name,
            name,
            args.R,
            state0,
            list(ring_idx),
            args.n_steps,
            args.report,
            dt,
            args.scale,
            kw,
        )
        for name, (_, kw) in runs.items()
    ]
    with mp.Pool(len(tasks)) as pool:
        results = dict(pool.map(_worker, tasks))

    # ---------------------------------------------------------------------------
    # Print comparison table
    # ---------------------------------------------------------------------------
    metrics = ["phi_rms", "u_rms", "circ", "E_phi", "E_u", "E_total"]

    def hdr():
        print(f"  {'step':>6}  {'t':>7}  " + "  ".join(f"{m:>10}" for m in metrics))
        print("  " + "-" * (15 + 12 * len(metrics)))

    for name in runs:
        print(f"=== {name.upper()} ===")
        hdr()
        for rec in results[name]:
            vals = "  ".join(f"{rec[m]:10.4e}" for m in metrics)
            print(f"  {rec['step']:6d}  {rec['t']:7.2f}  {vals}")
        print()

    # ---------------------------------------------------------------------------
    # Summary: final-state comparison
    # ---------------------------------------------------------------------------
    print("=== FINAL STATE COMPARISON ===")
    print(f"  {'':12s}  " + "  ".join(f"{m:>12}" for m in metrics))
    print("  " + "-" * (14 + 14 * len(metrics)))
    for name in runs:
        rec = results[name][-1]
        vals = "  ".join(f"{rec[m]:12.4e}" for m in metrics)
        print(f"  {name:12s}  {vals}")

    print()
    print("=== ENERGY ACCOUNTING ===")
    print(
        f"  {'':12s}  {'ΔE_phi':>12}  {'ΔE_u':>12}  {'ΔE_total':>12}  {'E_u/ΔE_phi%':>13}"
    )
    for name in runs:
        h0 = results[name][0]
        hf = results[name][-1]
        dEphi = hf["E_phi"] - h0["E_phi"]
        dEu = hf["E_u"] - h0["E_u"]
        dEtot = hf["E_total"] - h0["E_total"]
        ratio = 100 * dEu / (-dEphi + 1e-30) if dEphi < 0 else float("nan")
        print(
            f"  {name:12s}  {dEphi:12.4e}  {dEu:12.4e}  {dEtot:12.4e}  {ratio:13.2f}%"
        )

    print()
    print("Interpretation:")
    print("  Maxwell:      always heats u (back-EMF regardless of flux sign or lam).")
    print(
        "  Maxwell_chi:  same but gated by chi=lam/(1+lam) — zero in vacuum, full near mass."
    )
    print("  QED:          E_u grows independently, sign-blind (dissipative vertex).")
    print("  DST:          stabilising counter-force — flips sign with div(phi).")
    print(
        "  Baseline:     E_u = 0 — phi redistribution only, no immediate pressure effect."
    )


if __name__ == "__main__":
    main()
