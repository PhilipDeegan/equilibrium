"""
Does a bare toroidal ring current's poloidal field match the classical
magnetic-dipole (circular current loop) B-field, via DST's already-validated
discrete curl(phi)=B (equilibrium.substrate.curl, confirmed exact against the
classical straight-wire profile in tst/substrate/test_curl.py)?

Context (now, 2026-09-02): the curl-full dipole/spheromak IC
(seed_curl_full_dipole_ic) is a NEUTRON candidate -- a CLOSED knot, poloidal
and toroidal components independently seeded and superposed into a
self-contained force-free equilibrium, zero net outward flux. The proton is
the OPEN version: per project_dst_neutron.md, "open knot (proton): u-flux
radiates outward -- creates long-range field (charge)", and per
project_proton_earth_dipole_unification.md, a real dipole field is sourced by
a current-LOOP-like circulation (Ampere's law), not by independently seeding
a poloidal component on top. This script tests exactly that: seed ONLY a
toroidal ring current's vector potential into phi (the exact classical A_phi
of a circular loop, same discrete-line-integral method the straight-wire test
used for A_z), reconstruct B via curl(phi), and check whether the POLOIDAL
dipole field appears as a derived consequence -- matching the classical
circular-loop B-field (elliptic-integral exact solution) both on-axis (pure
1/z^3 dipole falloff) and in the equatorial plane (near-field + far-field
shape) -- rather than being independently added.

Method: pure static geometry test, no time-stepping (same style as
test_curl.py). mu0*I is set to 1 -- shape/ratio only, not absolute
calibration (the straight-wire test found its own hop-distance^2 calibration
constant, not assumed here).
"""

import numpy as np
from scipy.special import ellipe, ellipk

import sys
sys.path.insert(0, ".")
from equilibrium.substrate.curl import b_vec_vectorized
from equilibrium.substrate.lattice import SubstrateLattice

R = 30.0
LOOP_A = 8.0


def _k2(rho, z, a):
    return 4.0 * a * rho / ((a + rho) ** 2 + z**2)


def _a_phi(rho, z, a):
    rho = np.asarray(rho, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    safe_rho = np.where(rho > 1e-9, rho, 1.0)
    k2 = np.clip(_k2(safe_rho, z, a), 0.0, 1.0 - 1e-12)
    K = ellipk(k2)
    E = ellipe(k2)
    val = (1.0 / np.pi) * np.sqrt(a / safe_rho) / np.sqrt(k2) * ((1.0 - k2 / 2.0) * K - E)
    # A_phi ~ rho near the axis (regular); explicit small-rho fadeout avoids
    # the ill-conditioned sqrt(a/rho) evaluation instead of trusting float
    # cancellation there.
    return np.where(rho > 1e-6, val, 0.0)


def _b_field(rho, z, a):
    rho = np.asarray(rho, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    safe_rho = np.where(rho > 1e-9, rho, 1.0)
    k2 = np.clip(_k2(safe_rho, z, a), 0.0, 1.0 - 1e-12)
    K = ellipk(k2)
    E = ellipe(k2)
    denom_pm = (a - safe_rho) ** 2 + z**2
    denom_pm = np.where(denom_pm > 1e-9, denom_pm, 1e-9)
    root = np.sqrt((a + safe_rho) ** 2 + z**2)
    prefactor = 1.0 / (2.0 * np.pi)
    b_rho = prefactor * (z / (safe_rho * root)) * (-K + (a**2 + safe_rho**2 + z**2) / denom_pm * E)
    b_z = prefactor * (1.0 / root) * (K + (a**2 - safe_rho**2 - z**2) / denom_pm * E)
    b_rho = np.where(rho > 1e-6, b_rho, 0.0)
    return b_rho, b_z


def _b_z_axis(z, a):
    return a**2 / (2.0 * (a**2 + z**2) ** 1.5)


def build_phi(lat, a):
    pos, nb = lat.pos, lat.nb
    valid = nb >= 0
    safe_nb = np.where(valid, nb, 0)
    mid = 0.5 * (pos[:, None, :] + pos[safe_nb])
    rho_mid = np.linalg.norm(mid[..., :2], axis=2)
    z_mid = mid[..., 2]
    a_phi_mid = _a_phi(rho_mid, z_mid, a)
    safe_rho_mid = np.where(rho_mid > 1e-9, rho_mid, 1.0)
    phihat_x = -mid[..., 1] / safe_rho_mid
    phihat_y = mid[..., 0] / safe_rho_mid
    diff = pos[safe_nb] - pos[:, None, :]
    a_dot_dl = a_phi_mid * (phihat_x * diff[..., 0] + phihat_y * diff[..., 1])
    return np.where(valid, a_dot_dl, 0.0)


def run():
    print(f"Building lattice R={R} ...")
    lat = SubstrateLattice(radius_cells=R)
    pos, nb = lat.pos, lat.nb
    print(f"N={lat.N}")
    lat.phi = build_phi(lat, LOOP_A)

    B = b_vec_vectorized(pos, nb, lat.phi)
    Bx, By, Bz = B[:, 0], B[:, 1], B[:, 2]
    have = np.isfinite(Bx) & np.isfinite(By) & np.isfinite(Bz)

    rho = np.linalg.norm(pos[:, :2], axis=1)
    z = pos[:, 2]
    theta = np.arctan2(pos[:, 1], pos[:, 0])
    d_loop = np.sqrt((rho - LOOP_A) ** 2 + z**2)

    b_rho_meas = Bx * np.cos(theta) + By * np.sin(theta)
    b_phi_meas = -Bx * np.sin(theta) + By * np.cos(theta)

    # --- Structural check: field of an axisymmetric current loop has NO
    # azimuthal component -- B_phi should vanish (like the wire's B_z=0).
    mask0 = have & (d_loop > 2.0) & (rho > 0.5)
    b_phi_rms = float(np.sqrt(np.mean(b_phi_meas[mask0] ** 2)))
    signal_rms = float(np.sqrt(np.mean((Bx[mask0] ** 2 + By[mask0] ** 2 + Bz[mask0] ** 2))))
    print(f"\n[axisymmetry] B_phi_rms={b_phi_rms:.4e}  |B|_rms={signal_rms:.4e}"
          f"  ratio={b_phi_rms/signal_rms:.4e}")

    # --- On-axis Bz(z): pure classical dipole falloff, zero near-field
    # complication (rho=0 exactly, no elliptic integral needed for target).
    print("\n[on-axis]  z      Bz_meas      Bz_target(mu0I=1)   ratio")
    axis_mask = have & (rho < 1.0)
    zs = np.sort(np.unique(np.round(z[axis_mask])))
    on_axis_ratios = []
    for zv in zs:
        m = axis_mask & (np.abs(z - zv) < 0.5)
        if m.sum() < 1:
            continue
        bz_m = float(np.mean(Bz[m]))
        bz_t = float(_b_z_axis(zv, LOOP_A))
        if abs(bz_t) < 1e-9:
            continue
        r_ = bz_m / bz_t
        on_axis_ratios.append(r_)
        print(f"  {zv:6.1f}  {bz_m:+.4e}  {bz_t:+.4e}  {r_:+.4f}")

    # --- Equatorial-plane Bz(rho): near AND far field shape vs the exact
    # elliptic-integral closed form.
    print("\n[equatorial z~0]  rho    Bz_meas      Bz_target   ratio")
    eq_mask = have & (np.abs(z) < 1.0) & (d_loop > 2.0)
    edges = np.linspace(0.0, R * 0.85, 20)
    eq_ratios = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = eq_mask & (rho >= lo) & (rho < hi)
        if m.sum() < 3:
            continue
        rho_c = float(np.mean(rho[m]))
        bz_m = float(np.mean(Bz[m]))
        _, bz_t_arr = _b_field(np.full(m.sum(), rho_c), np.zeros(m.sum()), LOOP_A)
        bz_t = float(np.mean(bz_t_arr))
        if abs(bz_t) < 1e-9:
            continue
        r_ = bz_m / bz_t
        eq_ratios.append(r_)
        print(f"  {rho_c:6.2f}  {bz_m:+.4e}  {bz_t:+.4e}  {r_:+.4f}")

    on_axis_ratios = np.array(on_axis_ratios)
    eq_ratios = np.array(eq_ratios)
    print(f"\n-> on-axis ratio: mean={on_axis_ratios.mean():.4f} std={on_axis_ratios.std():.4f}"
          f"  (n={len(on_axis_ratios)})")
    print(f"-> equatorial ratio: mean={eq_ratios.mean():.4f} std={eq_ratios.std():.4f}"
          f"  (n={len(eq_ratios)})")
    print(f"-> B_phi/|B| (should be ~0): {b_phi_rms/signal_rms:.4e}")


if __name__ == "__main__":
    run()
