"""
Is step_cross even necessary for real matter translation, or can a closed
circulation's own face-population asymmetry ("lean") already produce real
energy-centroid drift under plain step() alone?  (Philip's own live question,
`now`'s 2026-09-01 punch list item 1 -- months of step_cross-gate work,
[[project_step_cross_v2]], all failed to find a gate passing both R1/R2;
if translation doesn't need step_cross at all, that work was solving the
wrong problem.)

step_cross is NEVER called anywhere in this script -- only alat.step()/
alat.step_sponge(), same driver as probe_poynting_step_only.py. No gate,
no m0, no lam seeding by default (lam0=0 throughout, --lam-ring available
but off by default to isolate the lean mechanism from lam-gradient effects).

Method: init_toroidal_winding ring (chosen test subject per Philip, NOT the
electron double-ring or the moat), free resonant mode (phi_only=True, same
IC as the Poynting-ring work). Track two independent things over many full
periods, on a FIXED core mask (ring tiles + their 1-hop neighbours -- static
because this cluster is never re-split/coarsened, so nb_j/pos never change):

  1. local_energy_flux_vector's full 3-vector (equilibrium.substrate.
     energy_flux, verified against the real step() -- see
     project_dst_em_field_convention_crisis.md), summed over the core mask
     at each checkpoint. A genuine "lean" would show up as a persistent
     NONZERO time-averaged vector, not a zero-mean oscillation from the
     ring's own internal circulation.
  2. The core mask's own energy-weighted centroid position (H_cell =
     u^2 + 0.5*sum(phi^2), same convention as electron_positron_collision.
     py's zcom_up/zcom_dn). Net displacement over the run, compared to the
     intra-run oscillation amplitude (the noise floor).

Run TWICE, amplitude=+A and amplitude=-A (opposite winding sign) -- a real
lean-driven translation mechanism should flip direction with the winding's
sign, same chirality-check logic as the W=+1/-1 tests in
project_dst_magnetic_field / the A/A2 verifiers. If the mean flux vector
and/or centroid drift do NOT flip sign (or don't clear the noise floor at
all), that's evidence against "lean alone produces real translation" --
consistent with (not proof of) step_cross still being necessary.

IMPORTANT caveat found on the first run (baseline, phi_only=True, lam=0):
with step_cross off and lam static (no step_lam call), the update
du_i=-dt*sigma_i*sum(W*phi), dphi=-dt*sigma_face*W*(u_nb-u_i) is EXACTLY
linear in (u,phi) for ANY fixed sigma field, including a nonuniform one --
negating the whole IC negates the whole trajectory at every later time.
S=W*phi*ubar is bilinear (both factors flip -> product invariant) and
H=u^2+phi^2/2 is quadratic (invariant pointwise) under that negation. So
the +A/-A pair is a MATHEMATICALLY GUARANTEED identity (not an empirical
finding) for ANY static background (lam gradient, u-pattern IC, whatever)
as long as step_cross stays off -- it cannot by itself distinguish a real
chirality-dependent translation mechanism from "there is none". It's kept
here only as an empirical sanity check on that derivation (confirmed run:
cos_sim(disp_+,disp_-)=+1.000, cos_sim(S_mean_+,S_mean_-)=+1.000 exactly,
not the "-1 for real" the docstring above naively expected before this was
worked out) -- the real question this script answers is whether |mean
S_core| / |net displacement| clears their OWN noise floors at all, for a
single amplitude, under each background configuration.

--lam-grad seeds a static lam(pos) = max(0, lam_grad*pos[axis]) gradient
along the ring's own winding axis (perpendicular to the ring's plane) --
zero on one side, growing on the other, like an external substrate
asymmetry (Philip: "we probably need a non-zero lam background") -- tests
whether a REAL external gradient (not the ring's own dynamics) is what lean
alone needs to produce net drift, since a perfectly isotropic background
gave nothing (see baseline result above).
--seed-u sets phi_only=False in init_toroidal_winding (u_j=A*cos(K*j) on
the ring at t=0, same traveling-wave IC as the original resonant-loop
construction) instead of leaving u=0 at t=0.

Usage:
    PYTHONPATH=$PWD python3 exp/2026/09/01/19_50_step_cross_necessity/probe_ring_lean_drift.py --matrix
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from equilibrium.substrate.energy_flux import local_energy_flux_vector
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import (
    find_toroidal_ring,
    init_toroidal_winding,
    leaf_nb_j,
    winding_number,
)

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--R", type=float, default=40.0)
parser.add_argument("--hops", type=int, default=4)
parser.add_argument("--amplitude", type=float, default=1.0)
parser.add_argument("--lam-ring", type=float, default=0.0)
parser.add_argument("--lam-grad", type=float, default=0.02)
parser.add_argument("--seed-u", action="store_true")
parser.add_argument("--axis", default="z")
parser.add_argument("--n-periods", type=float, default=8.0)
parser.add_argument("--checkpoints-per-period", type=int, default=4)
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument("--threads", type=int, default=8)
parser.add_argument(
    "--matrix",
    action="store_true",
    help="run the {seed_u, lam_grad} x {off,on} 2x2 matrix instead of the single "
    "config given by --seed-u/--lam-grad",
)


def build(args, amplitude, *, seed_u, lam_grad, ax_idx):
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    cl = alat[0]
    _lat.build_bcc_sphere(cl, args.R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    alat.set_n_threads(args.threads)

    r_all = np.linalg.norm(np.array(cl.leaf_pos), axis=1)
    r_max = float(r_all.max())
    r_inner = r_max * (1.0 - args.sponge)
    cl.set_bc(
        _lat.BoundaryCondition.quadratic_sponge(
            target=0.0, damp_max=args.damp_max, r_all=r_all, r_inner=r_inner
        )
    )

    ring_js, ring_faces = find_toroidal_ring(
        cl, axis=args.axis, mode="square", hops=args.hops
    )
    if ring_js is None:
        print("ERROR: no toroidal ring found")
        sys.exit(1)
    w = winding_number(cl, ring_js, axis=args.axis)
    omega = init_toroidal_winding(
        cl, ring_js, ring_faces, amplitude=amplitude, phi_only=not seed_u
    )

    lam = np.array(cl.leaf_lam)
    if lam_grad:
        pos = np.array(cl.leaf_pos)
        lam = np.maximum(lam, lam_grad * pos[:, ax_idx])
    if args.lam_ring:
        for j in ring_js:
            lam[j] = args.lam_ring
    if lam_grad or args.lam_ring:
        cl.set_leaf_lam(lam)

    return alat, cl, ring_js, w, omega


def core_mask_from_ring(nb_j, ring_js, N):
    core = set(ring_js)
    for j in ring_js:
        core.update(int(k) for k in nb_j[j] if k >= 0)
    return np.array(sorted(core), dtype=np.int64)


def run_one(args, amplitude, label, *, seed_u=False, lam_grad=0.0):
    ax_idx = {"x": 0, "y": 1, "z": 2}[args.axis]
    alat, cl, ring_js, w, omega = build(
        args, amplitude, seed_u=seed_u, lam_grad=lam_grad, ax_idx=ax_idx
    )
    pos = np.array(cl.leaf_pos)
    nb_j = leaf_nb_j(cl)
    N = pos.shape[0]
    core = core_mask_from_ring(nb_j, ring_js, N)

    dt = compute_dt(cl)
    period = 2.0 * np.pi / omega if omega > 0 else None
    if period is None:
        print(f"ERROR: omega={omega}, no resonant period")
        sys.exit(1)
    steps_per_checkpoint = max(1, round(period / dt / args.checkpoints_per_period))
    n_checkpoints = int(args.n_periods * args.checkpoints_per_period) + 1

    print(
        f"\n=== {label}  amplitude={amplitude:+.2f} ===\n"
        f"N={cl.n_leaves}  R={args.R}  hops={args.hops}  ring_n={len(ring_js)}  "
        f"w={w:.3f}  omega={omega:.4f}  period={period:.2f}  dt={dt:.4e}  "
        f"core_n={len(core)}  steps/checkpoint={steps_per_checkpoint}  "
        f"total_steps={n_checkpoints * steps_per_checkpoint}",
        flush=True,
    )

    t_h = np.zeros(n_checkpoints)
    centroid_h = np.zeros((n_checkpoints, 3))
    S_core_h = np.zeros((n_checkpoints, 3))
    H_core_h = np.zeros(n_checkpoints)

    def measure(step_idx):
        u = np.array(cl.leaf_u)
        phi = np.array(cl.leaf_phi).T  # (N,14)
        H_cell = u**2 + 0.5 * np.sum(phi**2, axis=1)
        H_core = float(H_cell[core].sum())
        centroid = (
            (pos[core] * H_cell[core, None]).sum(axis=0) / H_core
            if H_core > 0
            else pos[core].mean(axis=0)
        )
        S3 = local_energy_flux_vector(pos, nb_j, phi, u)
        S_core = S3[core].sum(axis=0)
        idx = step_idx // steps_per_checkpoint
        t_h[idx] = step_idx * dt / period
        centroid_h[idx] = centroid
        S_core_h[idx] = S_core
        H_core_h[idx] = H_core

    measure(0)
    for cp in range(1, n_checkpoints):
        for _ in range(steps_per_checkpoint):
            alat.step(0, dt)
            alat.step_sponge(0)
        measure(cp * steps_per_checkpoint)
        if cp % max(1, n_checkpoints // 8) == 0:
            print(
                f"  t/period={t_h[cp]:6.2f}  H_core={H_core_h[cp]:.4e}  "
                f"centroid={centroid_h[cp]}",
                flush=True,
            )

    disp = centroid_h[-1] - centroid_h[0]
    osc_amp = centroid_h.std(axis=0)
    S_mean = S_core_h.mean(axis=0)
    S_std = S_core_h.std(axis=0)

    disp_mag = float(np.linalg.norm(disp))
    osc_mag = float(np.linalg.norm(osc_amp))
    S_mean_mag = float(np.linalg.norm(S_mean))
    S_std_mag = float(np.linalg.norm(S_std))

    print(f"\n--- {label} summary ---")
    print(f"centroid[0]  = {centroid_h[0]}")
    print(f"centroid[-1] = {centroid_h[-1]}")
    print(f"net displacement       = {disp}  |disp|={disp_mag:.4e}")
    print(f"intra-run osc std      = {osc_amp}  |osc|={osc_mag:.4e}  "
          f"(disp/osc = {disp_mag / max(osc_mag, 1e-300):.3f})")
    print(f"H_core[0]/H_core[-1]   = {H_core_h[0]:.4e} / {H_core_h[-1]:.4e}  "
          f"(fraction retained = {H_core_h[-1] / max(H_core_h[0], 1e-300):.4f})")
    print(f"mean S_core (period-avg lean vector) = {S_mean}  |mean|={S_mean_mag:.4e}")
    print(f"std  S_core (checkpoint-to-checkpoint noise)  = {S_std}  |std|={S_std_mag:.4e}")
    print(f"mean/std ratio (S/N of the lean signal itself) = "
          f"{S_mean_mag / max(S_std_mag, 1e-300):.3f}")
    if disp_mag > 0 and S_mean_mag > 0:
        cos_sim = float(np.dot(disp, S_mean) / (disp_mag * S_mean_mag))
        print(f"cos_sim(net displacement, mean S_core) = {cos_sim:+.3f}  "
              f"(does the lean vector predict the drift direction?)")

    return dict(disp=disp, S_mean=S_mean, osc_mag=osc_mag, S_std_mag=S_std_mag)


def run_config(args, label, *, seed_u, lam_grad):
    pos_res = run_one(args, args.amplitude, f"{label} W=+1", seed_u=seed_u, lam_grad=lam_grad)
    neg_res = run_one(args, -args.amplitude, f"{label} W=-1", seed_u=seed_u, lam_grad=lam_grad)

    disp_cos = float(
        np.dot(pos_res["disp"], neg_res["disp"])
        / max(np.linalg.norm(pos_res["disp"]) * np.linalg.norm(neg_res["disp"]), 1e-300)
    )
    S_cos = float(
        np.dot(pos_res["S_mean"], neg_res["S_mean"])
        / max(np.linalg.norm(pos_res["S_mean"]) * np.linalg.norm(neg_res["S_mean"]), 1e-300)
    )
    print(
        f"\n=== {label}: sign-flip check (mathematically guaranteed +1/+1 "
        f"for any linear, step_cross-free dynamics -- see module docstring; "
        f"kept only to empirically confirm that derivation still holds here) ==="
    )
    print(f"cos_sim(disp_+, disp_-)   = {disp_cos:+.3f}")
    print(f"cos_sim(S_mean_+, S_mean_-) = {S_cos:+.3f}")

    return dict(
        label=label,
        disp_mag=np.linalg.norm(pos_res["disp"]),
        osc_mag=pos_res["osc_mag"],
        S_mean_mag=np.linalg.norm(pos_res["S_mean"]),
        S_std_mag=pos_res["S_std_mag"],
    )


def main():
    args = parser.parse_args()

    if not args.matrix:
        run_config(args, "single", seed_u=args.seed_u, lam_grad=args.lam_grad)
        return

    configs = [
        ("baseline (phi_only, lam=0)", dict(seed_u=False, lam_grad=0.0)),
        (f"lam_grad={args.lam_grad} only", dict(seed_u=False, lam_grad=args.lam_grad)),
        ("seed_u only", dict(seed_u=True, lam_grad=0.0)),
        (f"seed_u + lam_grad={args.lam_grad}", dict(seed_u=True, lam_grad=args.lam_grad)),
    ]
    results = [run_config(args, label, **kw) for label, kw in configs]

    print("\n=== matrix summary ===")
    print(
        f"{'config':<32}{'|disp|':>12}{'|osc|':>12}{'disp/osc':>10}"
        f"{'|S_mean|':>12}{'|S_std|':>12}{'S_mean/S_std':>14}"
    )
    for r in results:
        print(
            f"{r['label']:<32}{r['disp_mag']:>12.3e}{r['osc_mag']:>12.3e}"
            f"{r['disp_mag'] / max(r['osc_mag'], 1e-300):>10.3f}"
            f"{r['S_mean_mag']:>12.3e}{r['S_std_mag']:>12.3e}"
            f"{r['S_mean_mag'] / max(r['S_std_mag'], 1e-300):>14.4f}"
        )
    print(
        "\nA config only shows real lean-driven drift if disp/osc >> 1 AND "
        "S_mean/S_std >> 1 (both well above their own noise floor). The "
        "baseline row already measured both at ~1 or below (noise) -- "
        "compare the other three rows against it, not against a fixed "
        "absolute threshold."
    )


if __name__ == "__main__":
    main()
