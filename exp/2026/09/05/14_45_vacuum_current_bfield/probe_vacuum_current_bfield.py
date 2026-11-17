"""
Follow-up to yesterday's exp/2026/09/04/22_15_vacuum_current_equilibrium/
probe_vacuum_current.py (two fixed-u point sources, "battery terminals",
bare vacuum -- no lam, no step_cross). That probe already established a
real, non-trivial midplane current I_mid flowing between the sources
(not just two independent radiating sources -- confirmed genuine
interaction). Philip's question (2026-09-05): should there not be a
"generated magnetic field" around that current, by Ampere's law?

This probe:
  1. Reruns the identical setup (never saved a snapshot before -- lesson
     from feedback_save_full_state_not_scalars: save phi/u/lam/pos, not
     just scalar diagnostics).
  2. Applies the ALREADY-VALIDATED discrete curl(phi)=B reconstruction
     (equilibrium.substrate.curl.b_vec_vectorized -- matches the classical
     straight-wire and current-loop B profiles exactly, same (2/sqrt(3))^2
     calibration constant both times, see tst/substrate/test_curl.py and
     test_curl_loop.py) to the settled current-carrying state.
  3. Quantifies against the CLASSICAL prediction. Note this classical
     target is NOT some nonzero Biot-Savart shape to fit -- for two POINT
     current sources/sinks radiating spherically-symmetric steady current
     into an infinite medium, each individual point source's own field is
     exactly B=0 EVERYWHERE by symmetry (a spherically-symmetric radial
     current has no preferred axis; the standard pseudovector-under-
     reflection argument forces B=0, the same reason a spherically
     symmetric mass/charge distribution can't source an external
     "gravitomagnetic"/magnetic dipole field). Superposing two such zero
     fields is still zero. So "how wrong is DST from classical" here is
     simply: how far from zero is the measured |B|, not a shape-fit R^2.
"""

import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium.substrate.curl import b_vec_vectorized
from equilibrium.substrate.lattice import DampedLattice

R = 40.0
SEP = 16.0
U0 = 1.0
N_STEPS = 20000
REPORT = 1000
N_THREADS = 8
MIDPLANE_HALF_WIDTH = 1.0
DATA_OUT = "exp/2026/09/05/14_45_vacuum_current_bfield/vacuum_current_state.npz"


def build(lat):
    pos = lat.pos
    src_a = np.array([-SEP / 2, 0.0, 0.0])
    src_b = np.array([SEP / 2, 0.0, 0.0])
    ia = int(np.argmin(np.linalg.norm(pos - src_a, axis=1)))
    ib = int(np.argmin(np.linalg.norm(pos - src_b, axis=1)))
    lat.u[ia] = U0
    lat.u[ib] = -U0
    lat.fixed_u[ia] = True
    lat.fixed_u[ib] = True
    return ia, ib


def midplane_current(lat):
    pos, nb = lat.pos, lat.nb
    near = np.abs(pos[:, 0]) <= MIDPLANE_HALF_WIDTH
    valid = nb >= 0
    safe_nb = np.where(valid, nb, 0)
    nbr_x = pos[safe_nb, 0]
    crossing = valid & near[:, None] & (
        ((pos[:, 0, None].repeat(14, axis=1) < 0) & (nbr_x > 0))
        | ((pos[:, 0, None].repeat(14, axis=1) > 0) & (nbr_x < 0))
    )
    sign = np.where(pos[:, 0, None].repeat(14, axis=1) < 0, 1.0, -1.0)
    W = np.where(np.arange(14) < 8, 3.0, 1.0)
    return float(np.sum(np.where(crossing, sign * W[None, :] * lat.phi, 0.0)))


def run():
    lat = DampedLattice(radius_cells=R, sponge_fraction=0.15, damp_max=0.003)
    ia, ib = build(lat)
    dt = lat.stable_dt()
    print(f"N={lat.N}  dt={dt:.4e}  src_a={lat.pos[ia]}  src_b={lat.pos[ib]}", flush=True)

    for step in range(N_STEPS + 1):
        if step % REPORT == 0:
            I = midplane_current(lat)
            H = float(np.sum(lat.u**2) + 0.5 * np.sum(lat.phi**2))
            print(f"  {step:6d}  I_mid={I:+.6e}  H={H:.6e}", flush=True)
        lat.step(dt, n_threads=N_THREADS)

    np.savez(
        DATA_OUT,
        pos=lat.pos, phi=lat.phi, u=lat.u, lam=lat.lam,
        ia=ia, ib=ib, R=R, SEP=SEP, U0=U0, n_steps=N_STEPS,
    )
    print(f"state saved -> {DATA_OUT}", flush=True)
    return lat, ia, ib


def analyze(lat, ia, ib):
    pos, nb = lat.pos, lat.nb
    B = b_vec_vectorized(pos, nb, lat.phi)  # (N, 3), NaN where a loop hits the boundary
    have = np.isfinite(B).all(axis=1)

    # settled/causal region: same window as yesterday's classical E-field
    # fit (both point sources' r in (2,14)) -- excludes the immediate
    # source cells and the sponge-dominated outer shell.
    src_a_pos, src_b_pos = pos[ia], pos[ib]
    r_a = np.linalg.norm(pos - src_a_pos, axis=1)
    r_b = np.linalg.norm(pos - src_b_pos, axis=1)
    settled = have & (r_a > 2) & (r_a < 14) & (r_b > 2) & (r_b < 14)

    Bmag = np.linalg.norm(B, axis=1)
    hop_sq_factor = (2.0 / np.sqrt(3.0)) ** 2  # same discrete-geometry calibration
    # calibrated so it's directly comparable to the classical target (which is
    # exactly zero here) in the same units test_curl.py/test_curl_loop.py use
    Bmag_cal = Bmag / hop_sq_factor

    print(f"\ncells with finite B: {int(have.sum())}/{len(have)}")
    print(f"settled-region cells: {int(settled.sum())}")
    print(f"classical target: B = 0 EXACTLY everywhere (two spherically-"
          f"symmetric point current sources -- pseudovector/reflection "
          f"symmetry argument, see module docstring)")
    print(f"\n|B| (raw, uncalibrated) over settled region:")
    print(f"  rms={np.sqrt(np.mean(Bmag[settled]**2)):.6e}  "
          f"max={Bmag[settled].max():.6e}  mean={Bmag[settled].mean():.6e}")
    print(f"\n|B| (calibrated, /{hop_sq_factor:.4f}) over settled region -- "
          f"directly comparable to the classical zero target:")
    print(f"  rms={np.sqrt(np.mean(Bmag_cal[settled]**2)):.6e}  "
          f"max={Bmag_cal[settled].max():.6e}  mean={Bmag_cal[settled].mean():.6e}")

    # compare against the scale of the u/phi field itself in the same region,
    # so "how wrong from classical" has a meaningful reference scale
    u_rms = float(np.sqrt(np.mean(lat.u[settled] ** 2)))
    phi_rms = float(np.sqrt(np.mean(lat.phi[settled] ** 2)))
    print(f"\nreference scales in the same region: u_rms={u_rms:.6e}  phi_rms={phi_rms:.6e}")

    # azimuthal structure check around the source-source (x) axis: does B
    # show a coherent circulating (curl-like) pattern, or is it patternless?
    y, z = pos[:, 1], pos[:, 2]
    rho = np.sqrt(y**2 + z**2)
    az_valid = settled & (rho > 0.5)
    b_azimuthal = (-B[:, 1] * z + B[:, 2] * y) / np.where(rho > 0, rho, 1.0)
    b_radial_perp = (B[:, 1] * y + B[:, 2] * z) / np.where(rho > 0, rho, 1.0)
    print(f"\naround the source-source (x) axis:")
    print(f"  B_azimuthal rms={np.sqrt(np.mean(b_azimuthal[az_valid]**2)):.6e}  "
          f"B_radial(perp) rms={np.sqrt(np.mean(b_radial_perp[az_valid]**2)):.6e}  "
          f"B_x rms={np.sqrt(np.mean(B[az_valid,0]**2)):.6e}")

    return B, Bmag_cal, settled


if __name__ == "__main__":
    lat, ia, ib = run()
    analyze(lat, ia, ib)
