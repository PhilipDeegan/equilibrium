"""
Free-electron standing-wave closure test -- no proton, no external force.

Physical picture (session discussion, 2026-09-16): a free electron isn't
orbiting anything -- it's a fixed-size closed winding whose own field
energy H (the same u^2 + 0.5*sum(phi^2) already used everywhere in this
project) plays the rest-mass role via E=hbar*omega, no separate mass
postulate needed. That frequency, combined with the substrate's own
dispersion relation, fixes a wavenumber k; closure (2*pi*r = n*lambda =
2*pi*n/k) then fixes a specific radius r = n/k for a chosen winding number
n. This does NOT assume a validated real-electron-mass-in-DST-units
conversion (none exists yet, project_derive_dont_pick_parameters) -- it
sweeps (n, r) directly and asks whether the mechanism itself (closure +
dispersion + energy) is self-consistent, not whether it matches a known
number.

For each (n, r):
  1. seed u = A*cos(n*theta)*env(rho,z) on a torus-shaped shell around
     radius r (phi=0 -- simplest possible IC, let real step() dynamics do
     the work; no companion-phi construction attempted for a curved path).
  2. evolve under the real step() update (plain SubstrateLattice, vacuum).
  3. measure the ACTUAL oscillation frequency omega_meas via FFT of the
     ring's own cos(n*theta)-projected amplitude over time.
  4. compare to the substrate's own dispersion prediction omega_pred at
     k=n/r, using the LOCAL tangential direction at theta=0 (the y-axis)
     as a representative wavevector direction -- the BCC dispersion is
     anisotropic (project_bcc_dispersion_anisotropy), so this is an
     approximation for a genuinely curved path, not an exact match.
  5. measure H (field energy) and check whether H/omega_meas is roughly
     CONSTANT across different (n, r) -- if so, that ratio is a DERIVED
     "hbar" in DST's own units, not a picked constant.

"Suggestive, not definitive" (Philip's framing) -- a handful of (n, r)
points, looking for a resonance/matching pattern, not a finished result.
"""

import sys

sys.path.insert(0, ".")

import numpy as np

from equilibrium.substrate.lattice import SubstrateLattice
from equilibrium.substrate.traveling_wave import dispersion_omega, face_offsets


def build_ring_ic(lat, n, r, width, amplitude=1.0):
    pos = lat.pos
    rho = np.hypot(pos[:, 0], pos[:, 1])
    z = pos[:, 2]
    theta = np.arctan2(pos[:, 1], pos[:, 0])
    env = np.exp(-0.5 * ((rho - r) / width) ** 2) * np.exp(-0.5 * (z / width) ** 2)
    lat.u = amplitude * env * np.cos(n * theta)
    lat.phi = np.zeros_like(lat.phi)
    return theta, env


def ring_amplitude(lat, n, theta, w):
    return float(np.sum(lat.u * np.cos(n * theta) * w) / np.sum(w**2 + 1e-300))


def field_energy(lat, mask):
    u2 = lat.u**2
    p2 = 0.5 * np.sum(lat.phi**2, axis=1)
    return float(np.sum((u2 + p2)[mask]))


def run_one(n, r, width, R_domain, n_periods_target, report_every):
    lat = SubstrateLattice(radius_cells=R_domain)
    theta, env = build_ring_ic(lat, n, r, width)
    mask = env > 0.05 * env.max()

    offsets, w = face_offsets(lat)
    K = n / r
    omega_pred = dispersion_omega(offsets, w, axis=1, K=K)  # y ~ tangential at theta=0
    dt = lat.stable_dt()
    T_pred = 2 * np.pi / omega_pred if omega_pred > 0 else 1.0
    n_steps = max(int(round(n_periods_target * T_pred / dt)), report_every * 8)

    ts, amps, energies = [], [], []
    for s in range(n_steps + 1):
        if s % report_every == 0:
            ts.append(s * dt)
            amps.append(ring_amplitude(lat, n, theta, mask))
            energies.append(field_energy(lat, mask))
        if s < n_steps:
            lat.step(dt)

    ts = np.array(ts)
    amps = np.array(amps)
    energies = np.array(energies)

    omega_meas = 0.0
    if len(ts) > 8:
        freqs = np.fft.rfftfreq(len(ts), d=(ts[1] - ts[0]))
        A = np.fft.rfft(amps - amps.mean())
        if len(A) > 1:
            dom = int(np.argmax(np.abs(A[1:]))) + 1
            omega_meas = 2 * np.pi * freqs[dom]

    H0 = energies[0] if len(energies) else 0.0
    Hend = energies[-1] if len(energies) else 0.0
    return {
        "n": n,
        "r": r,
        "width": width,
        "N": lat.pos.shape[0],
        "dt": dt,
        "n_steps": n_steps,
        "omega_pred": omega_pred,
        "omega_meas": omega_meas,
        "H0": H0,
        "H_end": Hend,
        "H_decay_frac": (H0 - Hend) / H0 if H0 else float("nan"),
        "ts": ts,
        "amps": amps,
        "energies": energies,
    }


def main():
    R_domain = 34.0
    width = 2.0
    n_periods_target = 6.0
    report_every = 10

    configs = [
        (2, 6.0),
        (2, 10.0),
        (2, 14.0),
        (4, 6.0),
        (4, 10.0),
        (4, 14.0),
        (6, 10.0),
        (6, 14.0),
        (6, 18.0),
    ]

    print(
        f"{'n':>3} {'r':>6} {'N':>8} {'steps':>7} {'w_pred':>9} {'w_meas':>9}"
        f" {'ratio':>7} {'H0':>10} {'decay%':>7} {'H0/w_meas':>11}"
    )
    results = []
    for n, r in configs:
        if r + 4 * width > R_domain * 0.8:
            print(f"{n:3d} {r:6.1f}  -- skipped (too close to domain edge)")
            continue
        res = run_one(n, r, width, R_domain, n_periods_target, report_every)
        ratio = res["omega_meas"] / res["omega_pred"] if res["omega_pred"] else float("nan")
        h_over_w = res["H0"] / res["omega_meas"] if res["omega_meas"] else float("nan")
        print(
            f"{n:3d} {r:6.1f} {res['N']:8d} {res['n_steps']:7d}"
            f" {res['omega_pred']:9.4f} {res['omega_meas']:9.4f} {ratio:7.3f}"
            f" {res['H0']:10.4e} {100*res['H_decay_frac']:7.2f} {h_over_w:11.4e}",
            flush=True,
        )
        results.append(res)

    np.savez(
        "ring_closure_results.npz",
        n=[r["n"] for r in results],
        r=[r["r"] for r in results],
        omega_pred=[r["omega_pred"] for r in results],
        omega_meas=[r["omega_meas"] for r in results],
        H0=[r["H0"] for r in results],
        H_decay_frac=[r["H_decay_frac"] for r in results],
    )
    print("\nsaved -> ring_closure_results.npz")


if __name__ == "__main__":
    main()
