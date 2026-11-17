"""
DST hydrogen ground state — proton ring stability + Bohr moat probe.

The proton ring sits at the natural BCC 8-tile position (r≈1.4 cells).
For hydrogen this is not a free parameter — it is where the ring is.

Phase 1 (warmup): chi island + u field settle without winding.
Phase 2 (ring):   inject ring at natural position, run N_RING steps,
                  track whether chi reorganises around it (stability).
Phase 3 (probe):  fork ring state, drop u-deficit at candidate moat
                  radii along +z, measure du_moat (Bohr orbit search).

Sweep --chi to find the threshold below which the ring dissolves and
above which a stable moat forms at the Bohr radius.
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")

from equilibrium import lattice as _lat
from tst.substrate.adaptive import (
    compute_dt,
    find_origin,
    setup_knot_and_origin,
)
from tst.substrate.adaptive.adaptive_flux_tube import (
    find_toroidal_ring,
    init_toroidal_winding,
    set_u_background,
)

D1 = 1.0

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="DST hydrogen ground state")
parser.add_argument("--R", type=float, default=40.0)
parser.add_argument("--chi", type=float, default=0.1)
parser.add_argument("--u-central", type=float, default=1.0)
parser.add_argument("--alpha", type=float, default=0.05)
parser.add_argument("--n-warmup", type=int, default=5000)
parser.add_argument(
    "--n-ring", type=int, default=20000, help="steps after ring injection"
)
parser.add_argument("--n-probe", type=int, default=2000, help="steps per moat probe")
parser.add_argument(
    "--u-def",
    type=float,
    default=None,
    help="moat perturbation (default: -0.1 × u_central)",
)
parser.add_argument("--probe-rmax", type=float, default=20.0, help="max probe radius")
parser.add_argument(
    "--bc",
    type=str,
    default="absorbing",
    choices=["absorbing", "sponge"],
    help="boundary condition: absorbing (u=0) or sponge (u=u_central/R, 1/r match)",
)
parser.add_argument("--sponge-weight", type=float, default=0.05)
parser.add_argument("--amplitude", type=float, default=1.0)
parser.add_argument("--report", type=int, default=1000)
parser.add_argument("--threads", type=int, default=6)
parser.add_argument("--save-prefix", type=str, default="")


def main():
    args = parser.parse_args()

    uc_alpha = args.u_central * args.alpha
    print(
        f"u_central × alpha = {uc_alpha:.4f}  ({'OK' if uc_alpha < 0.008 else 'WARN: above stability boundary'})"
    )

    # ---------------------------------------------------------------------------
    # Build lattice
    # ---------------------------------------------------------------------------
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    cl = alat[0]
    _lat.build_bcc_sphere(cl, args.R, D1, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    alat.set_n_threads(args.threads)

    info = _lat.DSTInfo()
    info.split_hi = 0.0
    info.split_lo = 0.0
    info.max_level = 0
    cl.info = info

    N = cl.n_leaves
    print(
        f"R={args.R}  N={N}  chi={args.chi}  alpha={args.alpha}  u_central={args.u_central}"
    )

    origin_j = find_origin(cl)
    setup_knot_and_origin(cl, origin_j, args.chi, u_origin=args.u_central)
    set_u_background(cl, origin_j, u_origin=args.u_central)
    if args.bc == "sponge":
        bc_target = args.u_central / args.R  # match 1/r profile at boundary
        cl.set_bc(
            _lat.BoundaryCondition.sponge(target=bc_target, weight=args.sponge_weight)
        )
        print(f"BC: sponge  target={bc_target:.5f}  weight={args.sponge_weight}")
    else:
        cl.set_bc(_lat.BoundaryCondition.absorbing())
        print("BC: absorbing")

    dt = compute_dt(cl, D1=D1)
    print(f"dt={dt:.3e}  warmup={args.n_warmup*dt:.1f}  ring={args.n_ring*dt:.1f}")

    # ---------------------------------------------------------------------------
    # Phase 1 — warmup (no ring)
    # ---------------------------------------------------------------------------
    print(f"\nwarmup {args.n_warmup} steps …", flush=True)
    for s in range(1, args.n_warmup + 1):
        alat.step(0, dt)
        if s % 1000 == 0:
            print(f"  {s}/{args.n_warmup}", flush=True)
    print("warmup done")

    # ---------------------------------------------------------------------------
    # Phase 2 — inject ring at natural BCC position, track stability
    # ---------------------------------------------------------------------------
    ring_js, ring_faces = find_toroidal_ring(cl, axis="z")
    if ring_js is None:
        print("ERROR: ring not found")
        sys.exit(1)

    ring_pos = np.array(cl.leaf_pos)[ring_js]
    ring_r = float(np.mean(np.sqrt(ring_pos[:, 0] ** 2 + ring_pos[:, 1] ** 2)))
    print(f"\nring: {len(ring_js)} tiles  r≈{ring_r:.3f}")
    init_toroidal_winding(cl, ring_js, ring_faces, amplitude=args.amplitude)

    pos_all = np.array(cl.leaf_pos)
    r_all = np.sqrt(np.sum(pos_all**2, axis=1))
    rho_all = np.sqrt(pos_all[:, 0] ** 2 + pos_all[:, 1] ** 2)

    FAIL_DISSOLVED = 1e-6
    FAIL_RUNAWAY = 5.0 * args.amplitude
    FAIL_CHI_SAT = 4.0 * args.chi

    # Radial shells for chi fluctuation accumulation
    _SHELL_R = np.arange(1.0, 20.5, 0.5)
    _chi_sum = np.zeros(len(_SHELL_R))
    _chi_sum2 = np.zeros(len(_SHELL_R))
    _chi_n = np.zeros(len(_SHELL_R), dtype=int)

    print(
        f"\n{'step':>8}  {'phi_rms':>9}  {'chi@1':>7}  {'chi@2':>7}  {'chi@4':>7}  {'chi@8':>7}  {'chi_out':>8}"
    )
    print("  " + "-" * 70)

    stop = None
    for s in range(1, args.n_ring + 1):
        alat.step(0, dt)

        chi_arr = 1.0 - np.array(cl.leaf_ell)
        for i, r in enumerate(_SHELL_R):
            m = np.abs(r_all - r) < 0.4
            if m.any():
                v = float(np.mean(chi_arr[m]))
                _chi_sum[i] += v
                _chi_sum2[i] += v * v
                _chi_n[i] += 1

        if s % args.report == 0:
            phi = np.array(cl.leaf_phi)
            phi_rms = float(np.sqrt(np.mean(phi**2)))
            outer = r_all > args.R * 0.5
            chi_out = float(np.mean(chi_arr[outer]))

            def chi_at(r):
                m = np.abs(r_all - r) < 0.8
                return float(np.mean(chi_arr[m])) if m.any() else 0.0

            print(
                f"{s:8d}  {phi_rms:9.5f}  {chi_at(1):7.4f}  {chi_at(2):7.4f}"
                f"  {chi_at(4):7.4f}  {chi_at(8):7.4f}  {chi_out:8.5f}",
                flush=True,
            )

            if phi_rms < FAIL_DISSOLVED:
                stop = "ring dissolved"
                break
            if phi_rms > FAIL_RUNAWAY:
                stop = f"runaway phi_rms={phi_rms:.2f}"
                break
            if chi_out > FAIL_CHI_SAT:
                stop = f"chi saturated outer={chi_out:.4f}"
                break

    if stop:
        print(f"  *** {stop} ***")
        sys.exit(0)

    # Chi pressure profile — mean and fluctuation amplitude at each radial shell
    print(f"\n=== chi pressure profile (mean ± std over {args.n_ring} steps) ===")
    print(f"  {'r':>5}  {'chi_mean':>10}  {'chi_std':>10}  {'std/mean':>10}")
    print("  " + "-" * 42)
    chi_pressure_r, chi_pressure_std = [], []
    for i, r in enumerate(_SHELL_R):
        if _chi_n[i] < 2:
            continue
        n = _chi_n[i]
        mean = _chi_sum[i] / n
        std = float(np.sqrt(max(_chi_sum2[i] / n - mean**2, 0.0)))
        rel = std / mean if mean > 1e-12 else float("nan")
        chi_pressure_r.append(r)
        chi_pressure_std.append(std)
        print(f"  {r:5.1f}  {mean:10.6f}  {std:10.6f}  {rel:10.4f}")

    # Pressure node: minimum chi fluctuation outside the ring
    # (where the phi pressure wave has a standing-wave node → stable orbital)
    cpr = np.array(chi_pressure_r)
    cpst = np.array(chi_pressure_std)
    out = cpr > ring_r * 1.5
    if out.any():
        node_r = float(cpr[out][np.argmin(cpst[out])])
        print(
            f"\n  pressure node r = {node_r:.2f}  ring_r = {ring_r:.3f}"
            f"  ratio = {node_r/ring_r:.2f}  (2π = {2*np.pi:.2f})"
        )

    if args.save_prefix:
        alat.save(f"{args.save_prefix}_ring.h5")

    # ---------------------------------------------------------------------------
    # Phase 3.5 — phase-shift profile
    # Sample phi_az at ring + several candidate orbit radii over N_PHASE steps,
    # compute dominant frequency and phase difference vs ring.
    # Van Allen trapping predicts phase_diff ≈ 180° at the orbit radius.
    # ---------------------------------------------------------------------------
    _HEX = np.array(
        [
            [-1, -1, -1],
            [-1, -1, +1],
            [-1, +1, -1],
            [-1, +1, +1],
            [+1, -1, -1],
            [+1, -1, +1],
            [+1, +1, -1],
            [+1, +1, +1],
        ],
        dtype=np.float64,
    )
    _SQ = np.array(
        [[-1, 0, 0], [+1, 0, 0], [0, -1, 0], [0, +1, 0], [0, 0, -1], [0, 0, +1]],
        dtype=np.float64,
    )
    _FN = np.vstack([_HEX / np.sqrt(3), _SQ])

    N_PHASE = 500
    r_phase = [ring_r] + list(np.arange(3.0, min(args.probe_rmax, 20.0) + 0.1, 2.0))
    phi_ts = {r: [] for r in r_phase}  # time series per radius

    az = np.zeros_like(pos_all)
    m = rho_all > 1e-9
    az[m, 0] = -pos_all[m, 1] / rho_all[m]
    az[m, 1] = pos_all[m, 0] / rho_all[m]

    # restore ring state before sampling
    cl.set_leaf_u(np.array(cl.leaf_u).copy())  # already at ring state
    for _ in range(N_PHASE):
        alat.step(0, dt)
        phi_now = np.array(cl.leaf_phi)
        phi_az_s = np.sum(phi_now * np.einsum("kd,nd->kn", _FN, az), axis=0)
        for r in r_phase:
            shell = np.abs(rho_all - r) < 0.8
            phi_ts[r].append(float(np.mean(phi_az_s[shell])) if shell.any() else 0.0)

    ring_sig = np.array(phi_ts[ring_r])
    N = len(ring_sig)
    freqs = np.fft.rfftfreq(N, d=dt)
    R_fft = np.fft.rfft(ring_sig)
    dom = int(np.argmax(np.abs(R_fft[1:]))) + 1
    omega_meas = freqs[dom] * 2 * np.pi
    omega_exp = 2.0 * abs(np.sin(np.pi / len(ring_js)))

    print(f"\n=== phase-shift profile  N={N_PHASE} steps ===")
    print(f"  ring ω measured = {omega_meas:.4f} rad/tick  expected = {omega_exp:.4f}")
    print(f"  {'r':>6}  {'phase_diff°':>12}  {'amp_ratio':>10}")
    print("  " + "-" * 34)
    phase_ring = float(np.angle(R_fft[dom]))
    for r in r_phase[1:]:
        sig = np.array(phi_ts[r])
        O_k = np.fft.rfft(sig)
        amp_ratio = float(np.abs(O_k[dom]) / (np.abs(R_fft[dom]) + 1e-20))
        phase_diff = float(np.degrees((np.angle(O_k[dom]) - phase_ring) % (2 * np.pi)))
        marker = " ← ~180° (Van Allen node)" if 150 < phase_diff < 210 else ""
        print(f"  {r:6.1f}  {phase_diff:12.1f}  {amp_ratio:10.4f}{marker}")

    # ---------------------------------------------------------------------------
    # Phase 3 — moat probe (Bohr orbit search)
    # ---------------------------------------------------------------------------
    u_ring = np.array(cl.leaf_u).copy()
    phi_ring = np.array(cl.leaf_phi).copy()
    ell_ring = np.array(cl.leaf_ell).copy()

    exclude = set(ring_js) | {origin_j}
    z_axis = np.array([0.0, 0.0, 1.0])
    r_probe = list(np.arange(2.0, args.probe_rmax + 0.1, 1.0))
    U_DEF = args.u_def if args.u_def is not None else -0.1 * args.u_central

    # ---------------------------------------------------------------------------
    # Phase boundary probe — phi circulation profile
    # No perturbation needed: the electron orbit is where |phi_az(r)| is minimal,
    # the phase shift boundary between proton near-field and vacuum far-field.
    # ---------------------------------------------------------------------------
    print("\n=== phi phase profile (no perturbation) ===")
    print(f"  {'r':>6}  {'|phi_az|':>10}  {'phi_az':>10}")
    print("  " + "-" * 32)

    phi_now = np.array(cl.leaf_phi)
    az = np.zeros_like(pos_all)
    m = rho_all > 1e-9
    az[m, 0] = -pos_all[m, 1] / rho_all[m]
    az[m, 1] = pos_all[m, 0] / rho_all[m]

    phi_az_all = np.sum(phi_now * np.einsum("kd,nd->kn", _FN, az), axis=0)

    phase_r, phase_v, phase_abs = [], [], []
    for r in np.arange(1.0, 20.1, 0.5):
        shell = np.abs(rho_all - r) < 0.4
        if not shell.any():
            continue
        mean_az = float(np.mean(phi_az_all[shell]))
        mean_abs = float(np.mean(np.abs(phi_az_all[shell])))
        phase_r.append(r)
        phase_v.append(mean_az)
        phase_abs.append(mean_abs)
        print(f"  {r:6.1f}  {mean_abs:10.5f}  {mean_az:+10.5f}")

    # minimum of |phi_az| outside the ring — the phase shift boundary
    phase_abs_arr = np.array(phase_abs)
    phase_r_arr = np.array(phase_r)
    outside = phase_r_arr > ring_r * 1.5
    if outside.any():
        phase_min_r = float(phase_r_arr[outside][np.argmin(phase_abs_arr[outside])])
        print(
            f"\n  phase boundary r = {phase_min_r:.2f}  ring_r = {ring_r:.3f}"
            f"  ratio = {phase_min_r/ring_r:.2f}  (2π = {2*np.pi:.2f})"
        )

    print(f"\n=== moat probe  u_def={U_DEF}  n_probe={args.n_probe} ===")
    print(f"  {'r_target':>9}  {'r_actual':>9}  {'u_init':>8}  {'du_moat':>10}")
    print("  " + "-" * 45)

    probe_results = []  # (r_actual, u_init, du)
    for r_target in r_probe:
        dists = np.linalg.norm(pos_all - r_target * z_axis, axis=1)
        j_probe = next((int(i) for i in np.argsort(dists) if int(i) not in exclude), -1)
        if j_probe < 0:
            continue

        cl.set_leaf_u(u_ring.copy())
        cl.set_leaf_phi(phi_ring.copy())
        cl.set_leaf_ell(ell_ring.copy())
        u_pert = u_ring.copy()
        u_pert[j_probe] += U_DEF
        cl.set_leaf_u(u_pert)
        u_init = float(u_ring[j_probe])

        for _ in range(args.n_probe):
            alat.step(0, dt)

        du = float(np.array(cl.leaf_u)[j_probe]) - u_init
        r_actual = float(r_all[j_probe])
        bar = "▼" * min(int(abs(du) * 100), 30) if du < U_DEF * 0.5 else ""
        spont = " *" if u_init < 0 else ""  # spontaneous moat marker
        print(
            f"  {r_target:9.1f}  {r_actual:9.3f}  {u_init:8.5f}  {du:+10.5f}  {bar}{spont}"
        )
        probe_results.append((r_actual, u_init, du))

    # Spontaneous moat: deepest naturally-occurring u-deficit (u_init minimum)
    spont_r = float("nan")
    if probe_results:
        spont_r = min(probe_results, key=lambda x: x[1])[0]

    # Outer orbit: first local du minimum OUTSIDE the inner exclusion zone.
    # Skip inner zone (r < 3 × ring_r) and spontaneous moat (u_init < 0).
    # Find the deepest negative du among outer cells with u_init > 0.
    outer_zone = [
        (r, u0, du) for r, u0, du in probe_results if r > 3.0 * ring_r and u0 > -0.001
    ]
    outer_r = float("nan")
    if outer_zone:
        best = min(outer_zone, key=lambda x: x[2])
        if best[2] < 0:
            outer_r = best[0]

    print(
        f"\n  spontaneous moat r = {spont_r:.3f}  ratio = {spont_r/ring_r:.2f}  (n=1 Bohr, 2π={2*np.pi:.2f})"
    )
    print(
        f"  outer probe orbit  r = {outer_r:.3f}  ratio = {outer_r/ring_r:.2f}  (n=2 candidate)"
    )

    if args.save_prefix:
        alat.save(f"{args.save_prefix}_probe.h5")

    print("\ndone")


if __name__ == "__main__":
    main()
