"""
Resolution / information-density convergence scan for the step_cross v2
barrier-gate R1/R2 result recorded in `now` (2026-08-21 entry).

That table (surv_e=0.955, surv_n=0.752, separation=1.27 at gate=barrier,
m0=3.0) was run at R=20, sph_frac=0.2 -> knot scale=4 cells -- ~27x coarser
than the ~110 cells/object-radius density (cells_across=220) used in the
R180L0 production spheromak studies. This scans knot scale (in cells)
upward at fixed sph_frac/sponge ratios, scaling n_steps linearly with
scale (the object's own dynamical period grows with its radius in cells
at fixed dt/cell-size -- the same confound already found+fixed in the
lockstep/swing sweep, see doc/mems/project_curl_full_dipole_spheromak.md),
to see whether the m0=3 separation holds as resolution increases.

Run:
  PYTHONPATH=$PWD python3 -u run/substrate/adaptive/probe_resolution_scan.py
"""

import argparse
import sys

sys.path.insert(0, ".")
import numpy as np

from run.substrate.adaptive.probe_cross_v2_gates import run_arm

parser = argparse.ArgumentParser()
parser.add_argument("--scales", type=str, default="4,8,16,24")
parser.add_argument("--sph-frac", type=float, default=0.2)
parser.add_argument("--gate", type=str, default="barrier")
parser.add_argument("--m0", type=float, default=3.0)
parser.add_argument("--cross-scale", type=float, default=0.01)
parser.add_argument("--base-scale", type=float, default=4.0)
parser.add_argument(
    "--base-steps",
    type=int,
    default=3000,
    help="n_steps at --base-scale, matches the original table",
)
parser.add_argument("--report", type=int, default=500)
parser.add_argument("--sponge", type=float, default=0.15)
parser.add_argument("--damp-max", type=float, default=0.003)
parser.add_argument("--n-threads", type=int, default=8)
parser.add_argument("--eng-frac", type=float, default=1e-3)
parser.add_argument("--lam-target", type=float, default=1.0)
parser.add_argument("--u-max", type=float, default=1.0)
parser.add_argument("--B0", type=float, default=1.0)
parser.add_argument("--data-out", type=str, default=None)
parser.add_argument(
    "--save-state",
    action="store_true",
    help="dump final leaf_phi/pos/lam/u for electron+neutron at each scale",
)
parser.add_argument(
    "--state-out",
    type=str,
    default="state",
    help="prefix for --save-state npz files (one per scale)",
)


def main():
    args = parser.parse_args()
    scales = [float(s) for s in args.scales.split(",")]
    rows = []
    for scale in scales:
        R = scale / args.sph_frac
        n_steps = max(1, round(args.base_steps * scale / args.base_scale))
        run_args = argparse.Namespace(
            R=R,
            sph_frac=args.sph_frac,
            B0=args.B0,
            lam_target=args.lam_target,
            u_max=args.u_max,
            cross_scale=args.cross_scale,
            n_steps=n_steps,
            report=args.report,
            sponge=args.sponge,
            damp_max=args.damp_max,
            n_threads=args.n_threads,
            eng_frac=args.eng_frac,
        )
        print(
            f"\n=== scale={scale:g} cells  R={R:.2f}  n_steps={n_steps} ===", flush=True
        )
        r_e = run_arm(
            "electron", args.gate, args.m0, run_args, scale, save_state=args.save_state
        )
        r_n = run_arm(
            "neutron", args.gate, args.m0, run_args, scale, save_state=args.save_state
        )
        se = float(r_e["hist"][-1][1])
        sn = float(r_n["hist"][-1][1])
        sep = se / sn if sn else float("nan")
        rows.append((scale, R, n_steps, se, sn, sep))
        print(
            f"scale={scale:g}  surv_e={se:.4f}  surv_n={sn:.4f}  sep={sep:.3f}",
            flush=True,
        )

        if args.save_state:
            state_path = f"{args.state_out}_scale{scale:g}.npz"
            np.savez(
                state_path,
                scale=scale,
                R=R,
                n_steps=n_steps,
                m0=args.m0,
                cross_scale=args.cross_scale,
                surv_e=se,
                surv_n=sn,
                sep=sep,
                e_phi=r_e["final_phi"],
                e_pos=r_e["final_pos"],
                e_lam=r_e["final_lam"],
                e_u=r_e["final_u"],
                n_phi=r_n["final_phi"],
                n_pos=r_n["final_pos"],
                n_lam=r_n["final_lam"],
                n_u=r_n["final_u"],
            )
            print(f"state saved -> {state_path}", flush=True)

    print(
        f"\n{'scale':>8} {'R':>8} {'n_steps':>8} {'surv_e':>8} {'surv_n':>8} {'sep':>8}"
    )
    for scale, R, n_steps, se, sn, sep in rows:
        print(f"{scale:8.1f} {R:8.2f} {n_steps:8d} {se:8.4f} {sn:8.4f} {sep:8.3f}")

    if args.data_out:
        np.save(args.data_out, np.array(rows))
        print(f"data saved -> {args.data_out}")


if __name__ == "__main__":
    main()
