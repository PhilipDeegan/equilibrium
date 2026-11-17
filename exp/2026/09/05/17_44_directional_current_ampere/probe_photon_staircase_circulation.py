"""
Direct test of the "cos/sin is a continuum approximation of a staircase"
hypothesis, using a photon (the already-validated traveling-wave IC,
exp/2026/08/19/18_30_traveling_wave_reproducer/probe_traveling_wave_shift.py)
as the minimal test object -- no lam (lam doesn't cycle, and lam requires
mass/protons we haven't validated -- Philip's correction), no wire, no
frozen "battery" loop, just a plane wave in (u, phi).

That reproducer only ever tested propagation EXACTLY along a principal
sq-face axis (z) -- the one direction that needs no "staircase": a wave
travelling along a lattice's own primary axis can hop face-to-face in a
straight line, no zigzag between different face types required. A wave
travelling in a GENERIC (off-axis) direction cannot do this -- it must
alternate between different hex/sq hops to approximate that direction on
average, exactly the "staircase, only smooth when zoomed out" claim.

Method: build the SAME companion-phi plane-wave construction (generalised
from a single axis to an arbitrary wavevector K), once for an axis-aligned
K (z) and once for a body-diagonal K (1,1,1)/sqrt(3) -- same wavelength,
same envelope, same amplitude, ONLY the direction differs. Measure genuine
discrete circulation on the 12 elementary triangles per cell
(equilibrium.substrate.discrete_curl.circulation -- the DST-native,
coordinate-free Wilson-loop measure, exact zero iff phi is some scalar's
discrete derivative, see project_dst_discrete_curl) at t=0 and after some
real step() evolution.

Two questions this settles:
  1. Is off-axis circulation systematically LARGER than on-axis (i.e. does
     the continuum sin/cos sampling genuinely fail to represent the
     staircase precisely where a real staircase should exist -- off-axis
     propagation -- while getting the on-axis case, which needs no
     staircase, essentially exactly right)?
  2. Does either case's circulation change under real step() (per the
     already-derived E_full=-grad(u)-dphi/dt=0 identity, expected to be
     EXACTLY conserved regardless of its initial value -- this checks that
     identity again, independently, on a genuinely different IC family
     than the frozen-ring test).
"""

import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium.substrate.discrete_curl import circulation
from equilibrium.substrate.lattice import SubstrateLattice, W_HEX, W_SQ

R_SPHERE = 40.0
CYL_RADIUS = 10.0
CYL_HALF_LEN = 120.0
WAVELENGTH = 16.0
ENVELOPE_WAVELENGTHS = 2.0
N_STEPS = 500
REPORT = 100


def build_faces(lat):
    c0 = int(np.argmin(np.linalg.norm(lat.pos, axis=1)))
    offsets = lat.pos[lat.nb[c0]] - lat.pos[c0]
    w = np.array([W_HEX] * 8 + [W_SQ] * 6, dtype=np.float64)
    return offsets, w


def companion_wave(lat, offsets, w, k_hat, amplitude, envelope_sigma):
    """k_hat: unit propagation direction. Generalises probe_traveling_wave_
    shift.py's single-axis companion_phi/u_field to an arbitrary direction."""
    K = 2 * np.pi / WAVELENGTH
    Kvec = K * k_hat
    theta = offsets @ Kvec  # (14,)
    sigma0 = 1.0
    omega = np.sqrt(float(np.sum(w**2 * (1 - np.cos(theta)))))

    pos = lat.pos
    proj = pos @ k_hat
    env = np.exp(-0.5 * (proj / envelope_sigma) ** 2)
    u = amplitude * env * np.cos(pos @ Kvec)

    phi = np.zeros_like(lat.phi)
    for k in range(14):
        Phi_k_amp = (2 * sigma0 * w[k] / omega) * np.sin(0.5 * theta[k])
        mid = pos + 0.5 * offsets[k][None, :]
        mid_proj = mid @ k_hat
        mid_env = np.exp(-0.5 * (mid_proj / envelope_sigma) ** 2)
        phi[:, k] = amplitude * Phi_k_amp * mid_env * np.cos(mid @ Kvec)
    return u, phi, omega


def report(lat, label, step):
    circ = circulation(lat.phi.T, lat.nb)
    phi_rms = float(np.sqrt(np.mean(lat.phi**2)))
    circ_rms = float(np.sqrt(np.mean(circ**2))) if circ.size else float("nan")
    print(f"  [{label}] step={step:5d}  n_triangles={circ.size:8d}  "
          f"phi_rms={phi_rms:.6e}  circ_rms={circ_rms:.6e}  "
          f"circ_rms/phi_rms={circ_rms/phi_rms:.6e}")
    return circ_rms


def run_case(label, k_hat, lat_builder):
    lat = lat_builder()
    offsets, w = build_faces(lat)
    envelope_sigma = ENVELOPE_WAVELENGTHS * WAVELENGTH
    u, phi, omega = companion_wave(lat, offsets, w, k_hat, 1.0, envelope_sigma)
    lat.u = u
    lat.phi = phi
    print(f"\n=== {label}  k_hat={k_hat}  omega={omega:.4f}  N={lat.N} ===")
    dt = lat.stable_dt()
    for step in range(N_STEPS + 1):
        if step % REPORT == 0:
            report(lat, label, step)
        lat.step(dt)


if __name__ == "__main__":
    # sq-axis (z) and hex-axis (1,1,1) are BOTH natural lattice axes -- the
    # lattice's own _SQ_DIR/_HEX_DIR tables are exactly the 6 coordinate
    # axes and the 8 (+-1,+-1,+-1)/sqrt3 body-diagonals respectively, so
    # neither needs a real "staircase" to propagate along. A genuinely
    # generic direction, well away from all 14 natural axes (checked:
    # 24 deg from nearest sq axis, 35 deg from nearest hex axis), is the
    # actual staircase-requiring test case.
    #
    # sq-axis propagation has NO transverse envelope variation at all (the
    # companion wave depends only on pos@k_hat), so a thin, cheap CYLINDER
    # aligned with z fits it exactly -- lets WAVELENGTH be pushed up without
    # the O(R^3) cost of an equivalently-well-resolved sphere. hex-axis and
    # the generic direction have components along all 3 coordinate axes, so
    # they still need an isotropic sphere domain -- same WAVELENGTH/envelope
    # for a fair comparison, just a different (necessarily pricier) shape.
    run_case(
        "sq-axis (z)", np.array([0.0, 0.0, 1.0]),
        lambda: SubstrateLattice.cylinder(CYL_RADIUS, CYL_HALF_LEN, axis=2),
    )
    run_case(
        "hex-axis (1,1,1)/sqrt3", np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0),
        lambda: SubstrateLattice(radius_cells=R_SPHERE),
    )
    generic = np.array([3.0, 1.3, 0.4])
    run_case(
        "generic (off all 14 axes)", generic / np.linalg.norm(generic),
        lambda: SubstrateLattice(radius_cells=R_SPHERE),
    )
