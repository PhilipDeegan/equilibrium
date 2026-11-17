"""
Does a better-tuned absorbing sponge kill the late-time reverberation found
in probe_poynting_time_evolution.py / probe_poynting_fine_and_snapshots.py
(equatorial radial slope swings wildly at late time, fine-resolved peak
intervals are irregular 0.04-1.4, no clean single period)? Philip's
question: since the dipole genuinely radiates outward, can we just let
modes leave the domain / be absorbed at the border properly, instead of
fighting a contaminated signal with windowed measurements?

Tracks total domain energy H(t) = sum(u^2) + 0.5*sum(phi^2) over the whole
run for several sponge configurations -- if the current sponge
(fraction=0.15, damp_max=0.003) is a poor absorber, H should plateau or
oscillate instead of decaying monotonically toward 0 as real radiated
energy actually leaves the domain (there's nowhere else for it to go once
it crosses into a properly-absorbing region). A too-strong/too-abrupt
sponge causes its own reflection (impedance mismatch) -- also checked here
by including an aggressive-damping variant.
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from tst.substrate.adaptive import compute_dt
from tst.substrate.adaptive.adaptive_flux_tube import find_toroidal_ring, init_toroidal_winding

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--R", type=float, default=30.0)
parser.add_argument("--hops", type=int, default=4)
parser.add_argument("--amplitude", type=float, default=1.0)
parser.add_argument("--lam-ring", type=float, default=1.0)
parser.add_argument("--axis", default="z")
parser.add_argument("--n-steps", type=int, default=4000)
parser.add_argument("--log-every", type=int, default=50)
parser.add_argument("--threads", type=int, default=8)


def run_config(args, sponge_fraction, damp_max, label):
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    cl = alat[0]
    _lat.build_bcc_sphere(cl, args.R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    alat.set_n_threads(args.threads)

    r_all = np.linalg.norm(np.array(cl.leaf_pos), axis=1)
    r_max = float(r_all.max())
    r_inner = r_max * (1.0 - sponge_fraction)
    cl.set_bc(_lat.BoundaryCondition.quadratic_sponge(target=0.0, damp_max=damp_max, r_all=r_all, r_inner=r_inner))

    ring_js, ring_faces = find_toroidal_ring(cl, axis=args.axis, mode="square", hops=args.hops)
    init_toroidal_winding(cl, ring_js, ring_faces, amplitude=args.amplitude, phi_only=True)
    lam = np.array(cl.leaf_lam)
    for j in ring_js:
        lam[j] = args.lam_ring
    cl.set_leaf_lam(lam)

    dt = compute_dt(cl)
    print(f"[{label}] N={cl.n_leaves} sponge_fraction={sponge_fraction} damp_max={damp_max} r_inner={r_inner:.2f} dt={dt:.4e}", flush=True)

    trace = []
    for step in range(args.n_steps):
        alat.step(0, dt)
        alat.step_sponge(0)
        if step % args.log_every == 0 or step == args.n_steps - 1:
            u = np.array(cl.leaf_u)
            phi = np.array(cl.leaf_phi)
            H = float(np.sum(u**2) + 0.5 * np.sum(phi**2))
            t_phys = (step + 1) * dt
            trace.append((t_phys, H))
    trace = np.array(trace)
    H0 = trace[0, 1] if trace[0, 1] > 0 else trace[1, 1]
    print(f"[{label}] H0={H0:.4e} H_final={trace[-1,1]:.4e} H_final/H0={trace[-1,1]/H0:.4f}", flush=True)
    return trace


def main():
    args = parser.parse_args()
    configs = [
        (0.15, 0.003, "default"),
        (0.35, 0.001, "deeper_gentler"),
        (0.15, 0.02, "shallow_aggressive"),
        (0.35, 0.01, "deeper_aggressive"),
    ]
    results = {}
    for frac, damp, label in configs:
        trace = run_config(args, frac, damp, label)
        np.save(f"exp/2026/09/08/12_15_poynting_time_evolution/sponge_trace_{label}.npy", trace)
        results[label] = trace

    print("\n=== late-time H/H0 comparison (last quarter of run) ===")
    for label, trace in results.items():
        n = len(trace)
        q = trace[3 * n // 4 :, 1]
        H0 = trace[0, 1] if trace[0, 1] > 0 else trace[1, 1]
        print(f"{label:20s}: H/H0 late-quarter mean={q.mean()/H0:.4e} std={q.std()/H0:.4e} (lower+flatter=better absorber)")
    print("done", flush=True)


if __name__ == "__main__":
    main()
