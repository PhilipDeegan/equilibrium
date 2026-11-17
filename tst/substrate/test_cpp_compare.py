"""
C++ equivalence and AMR consistency checks.

Part 1 — Python vs C++ for step, step_lam, step_cross on the uniform lattice.
Part 2 — AMR lattice sanity:
  (a) AMR level-0 (no splits) radial u profile ≈ uniform SubstrateLattice.
  (b) AMR level-1 coarse region ≈ AMR level-0 at same physical time —
      verifies that flux correction at the coarse-fine boundary does not
      corrupt the large-scale field.
"""

import sys

import numpy as np

sys.path.insert(0, ".")

from equilibrium.substrate.lattice import (
    _FACE_ADJ,
    _W,
    LAM_MAX,
    SubstrateLattice,
    SubstrateTile,
)


def _make_lat(R=8.0, seed=42):
    rng = np.random.default_rng(seed)
    lat = SubstrateLattice(radius_cells=R)
    lat.u[:] = rng.uniform(-0.05, 0.05, lat.N)
    lat.phi[:] = rng.uniform(-5e-3, 5e-3, (lat.N, 14))
    lat.lam[:] = rng.uniform(0.0, 4.0, lat.N)
    lat.fixed_u[0] = True  # one frozen cell
    _ = lat.nb  # force cache build
    return lat


def _clone(lat):
    c = SubstrateLattice()
    c.pos = lat.pos.copy()
    c.ell = lat.ell.copy()
    c.u = lat.u.copy()
    c.phi = lat.phi.copy()
    c.lam = lat.lam.copy()
    c.fixed_u = lat.fixed_u.copy()
    c.fixed_phi = lat.fixed_phi.copy()
    c._nb_idx = lat._nb_idx.copy()
    c._dirty = False
    c.tiles = [SubstrateTile(i, c) for i in range(lat.N)]
    return c


def _step_lam_py(lat, dt, scale=1.0):
    """Reference numpy implementation (mirrors lattice.py fallback)."""
    nb_idx = lat._nb_idx
    valid = nb_idx >= 0
    face_w = np.where(valid, _W[None, :], 0.0)
    sigma = 1.0 / (1.0 + lat.lam)
    phi_sum = (face_w * np.where(valid, lat.phi, 0.0)).sum(axis=1)
    dlam = -dt * scale * sigma * lat.lam * phi_sum
    lat.lam = np.clip(np.where(lat.fixed_u, lat.lam, lat.lam + dlam), 0.0, LAM_MAX)


def _step_cross_py(lat, dt, scale=1.0):
    """Reference numpy implementation (mirrors lattice.py fallback)."""
    dphi = np.zeros_like(lat.phi)
    for k in range(14):
        inward = np.minimum(lat.phi[:, k], 0.0)
        for j in [x for x in _FACE_ADJ[k] if x >= 0]:
            dphi[:, j] -= dt * scale * inward
    lat.phi = np.where(lat.fixed_phi, lat.phi, lat.phi + dphi)


def _radial_profile(pos, u, bins, mask=None):
    r = np.linalg.norm(pos, axis=1)
    sel = mask if mask is not None else np.ones(len(u), dtype=bool)
    return np.array(
        [
            (
                float(u[(r >= b) & (r < b + 1.0) & sel].mean())
                if np.any((r >= b) & (r < b + 1.0) & sel)
                else np.nan
            )
            for b in bins
        ]
    )


def _init_amr(alat_mod, R, split_r=None):
    """Build a BCC sphere AMR lattice, optionally split cells within split_r."""
    alat = alat_mod.AdaptiveLattice()
    alat.add_cluster()
    cl = alat[0]
    alat_mod.build_bcc_sphere(cl, R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    info = alat_mod.DSTInfo()
    info.split_hi = 0.0
    info.split_lo = 0.0
    info.max_level = 2
    cl.info = info
    if split_r is not None:
        pos = np.array(cl.leaf_pos)
        r = np.linalg.norm(pos, axis=1)
        idxs = np.array(cl.leaf_global_indices, dtype=np.int32)
        gs = np.ascontiguousarray(idxs[r < split_r])
        alat.split_batch(0, gs, 1.0)
    # Seed 1/r Coulomb profile + fixed source at origin
    pos = np.array(cl.leaf_pos)
    r = np.linalg.norm(pos, axis=1)
    u = 0.1 / np.maximum(r, 1.0)
    fu = np.zeros(cl.n_leaves, dtype=np.uint8)
    fu[int(np.argmin(r))] = 1
    cl.set_leaf_u(u)
    cl.set_leaf_fixed_u(fu)
    return alat, cl


def test_amr():
    try:
        from equilibrium import lattice as _lat
    except ImportError:
        print("SKIP (AMR): equilibrium.lattice not built")
        return True

    R = 10.0
    N = 150  # steps at coarse dt
    dt = 0.004  # safe for level-0 (CFL ≈ 0.00513)
    dt_fine = dt / 2.0  # safe for level-1 (d1=0.5)
    SPLIT_R = 3.0  # cells within this radius get level-1 splits
    bins = np.arange(1.5, R - 2.0, 1.0)

    ok_all = True

    # ------------------------------------------------------------------
    # (a) AMR level-0 vs uniform SubstrateLattice
    # ------------------------------------------------------------------

    # Uniform reference: N steps at dt
    lat_ref = SubstrateLattice(radius_cells=R)
    r_ref = np.linalg.norm(lat_ref.pos, axis=1)
    lat_ref.u[:] = 0.1 / np.maximum(r_ref, 1.0)
    lat_ref.fixed_u[int(np.argmin(r_ref))] = True
    for _ in range(N):
        SubstrateLattice.step(lat_ref, dt)
    prof_ref = _radial_profile(lat_ref.pos, lat_ref.u, bins)

    # AMR level-0: same N steps at dt
    alat0, cl0 = _init_amr(_lat, R)
    for _ in range(N):
        alat0.step(0, dt)
    prof_amr0 = _radial_profile(np.array(cl0.leaf_pos), np.array(cl0.leaf_u), bins)

    valid_a = np.isfinite(prof_ref) & np.isfinite(prof_amr0)
    rel_a = float(
        np.max(
            np.abs(prof_ref[valid_a] - prof_amr0[valid_a])
            / np.maximum(np.abs(prof_ref[valid_a]), 1e-9)
        )
    )
    ok_a = rel_a < 0.20  # loose: different numerical scheme
    ok_all &= ok_a
    print(f"AMR level-0 vs uniform  max_rel={rel_a:.3f}  [{'OK' if ok_a else 'FAIL'}]")
    print(f"  bins: {[f'{b:.0f}' for b in bins]}")
    print(f"  uniform: {[f'{v:.4f}' if np.isfinite(v) else 'nan' for v in prof_ref]}")
    print(f"  AMR-L0:  {[f'{v:.4f}' if np.isfinite(v) else 'nan' for v in prof_amr0]}")

    # ------------------------------------------------------------------
    # (b) AMR level-1 coarse region vs AMR level-0 at same physical time
    #     physical time = N*dt = 2N*dt_fine
    #
    # The transition band immediately outside SPLIT_R will differ from the
    # pure level-0 run because AMR flux corrections inject fine-cell detail
    # into adjacent coarse cells.  Only the far-field (beyond the transition
    # band) should match; compare bins starting at SPLIT_R + 2.
    # ------------------------------------------------------------------

    BINS_FAR = np.arange(SPLIT_R + 2.0, R - 2.0, 1.0)  # well outside transition

    # AMR level-0 baseline at dt_fine (2N steps = same physical time as level-1)
    alat0b, cl0b = _init_amr(_lat, R)
    for _ in range(2 * N):
        alat0b.step(0, dt_fine)
    prof_amr0b = _radial_profile(
        np.array(cl0b.leaf_pos), np.array(cl0b.leaf_u), BINS_FAR
    )

    # AMR level-1: 2N steps at dt_fine, splits inside SPLIT_R
    alat1, cl1 = _init_amr(_lat, R, split_r=SPLIT_R)
    for _ in range(2 * N):
        alat1.step(0, dt_fine)

    pos1 = np.array(cl1.leaf_pos)
    u1 = np.array(cl1.leaf_u)
    level1 = np.array(cl1.leaf_level, dtype=int)
    coarse1 = level1 == 0  # leaf cells that were NOT split

    prof_amr1 = _radial_profile(pos1, u1, BINS_FAR, mask=coarse1)
    valid_b = np.isfinite(prof_amr0b) & np.isfinite(prof_amr1)

    rel_b = float(
        np.max(
            np.abs(prof_amr0b[valid_b] - prof_amr1[valid_b])
            / np.maximum(np.abs(prof_amr0b[valid_b]), 1e-9)
        )
    )
    ok_b = rel_b < 0.15
    ok_all &= ok_b
    n_coarse = int(coarse1.sum())
    n_fine = int((~coarse1).sum())
    print(
        f"\nAMR level-1 far-field vs level-0  max_rel={rel_b:.3f}  "
        f"[{'OK' if ok_b else 'FAIL'}]  ({n_coarse} coarse, {n_fine} fine leaves)"
    )
    print(f"  AMR-L0:  {[f'{v:.4f}' if np.isfinite(v) else 'nan' for v in prof_amr0b]}")
    print(f"  AMR-L1:  {[f'{v:.4f}' if np.isfinite(v) else 'nan' for v in prof_amr1]}")

    return ok_all


def test_energy_conservation_nonuniform_lam():
    """Field energy 0.5*sum(u^2) + 0.5*sum(W*phi^2) must stay conserved when
    lam (hence sigma=1/(1+lam)) varies spatially across a face -- this only
    holds if the u-update uses the same face-averaged sigma_face as the
    phi-update, not each cell's own local sigma (they coincide for uniform
    lam, which is why this went unnoticed until a probe used a nonuniform
    lam 'wall'). Caught via a spurious energy_frac=1.21 in an electron
    guide-field confinement probe (2026-07-13); see doc/dst_next.md.
    """
    lat = SubstrateLattice(radius_cells=10.0)
    r = np.linalg.norm(lat.pos, axis=1)
    lat.u = -1.0 * np.exp(-((r / 2.0) ** 2))
    lat.lam = 5.0 * np.exp(-((r - 5.0) ** 2) / (2.0 * 1.5**2))  # nonuniform shell

    dt = lat.stable_dt()

    def energy():
        valid = lat._valid
        phi_e = 0.5 * float((np.where(valid, _W[None, :], 0.0) * lat.phi**2).sum())
        u_e = 0.5 * float((lat.u**2).sum())
        return u_e + phi_e

    e0 = energy()
    for _ in range(2000):
        SubstrateLattice.step(lat, dt)
    e1 = energy()
    frac = e1 / max(e0, 1e-30)
    ok = abs(frac - 1.0) < 0.02
    print(
        f"energy conservation (nonuniform lam shell): e0={e0:.4e}  e1={e1:.4e}"
        f"  frac={frac:.4f}  [{'OK' if ok else 'FAIL'}]"
    )
    return ok


def main():
    try:
        from equilibrium import cpp as _cpp
    except ImportError:
        print("SKIP: C++ extension not built")
        sys.exit(0)

    missing = [
        n
        for n in ("substrate_step", "substrate_step_lam", "substrate_step_cross")
        if not hasattr(_cpp, n)
    ]
    if missing:
        print(f"SKIP: missing C++ symbols: {missing}")
        sys.exit(0)

    lat0 = _make_lat()
    dt = lat0.stable_dt()
    print("=== uniform substrate ===")
    print(f"N={lat0.N}  dt={dt:.4e}\n")

    ok_all = True

    # --- step() -------------------------------------------------------
    a = _clone(lat0)
    b = _clone(lat0)
    SubstrateLattice.step(a, dt)
    SubstrateLattice.step_cpp(b, dt)
    du = float(np.max(np.abs(a.u - b.u)))
    dp = float(np.max(np.abs(a.phi - b.phi)))
    ok = du < 1e-12 and dp < 1e-12
    ok_all &= ok
    print(
        f"step()       max|Δu|={du:.2e}  max|Δphi|={dp:.2e}  [{'OK' if ok else 'FAIL'}]"
    )

    # --- step_lam() ---------------------------------------------------
    a = _clone(lat0)
    b = _clone(lat0)
    _step_lam_py(a, dt)
    _cpp.substrate_step_lam(b.lam, b.phi, b.fixed_u.view(np.uint8), b._nb_idx, dt)
    dl = float(np.max(np.abs(a.lam - b.lam)))
    ok = dl < 1e-12
    ok_all &= ok
    print(f"step_lam()   max|Δlam|={dl:.2e}  [{'OK' if ok else 'FAIL'}]")

    # --- step_cross() -------------------------------------------------
    a = _clone(lat0)
    b = _clone(lat0)
    _step_cross_py(a, dt)
    _cpp.substrate_step_cross(b.phi, b.fixed_phi.view(np.uint8), dt)
    dp = float(np.max(np.abs(a.phi - b.phi)))
    ok = dp < 1e-12
    ok_all &= ok
    print(f"step_cross() max|Δphi|={dp:.2e}  [{'OK' if ok else 'FAIL'}]")

    # --- energy conservation under nonuniform lam ----------------------
    ok_e = test_energy_conservation_nonuniform_lam()
    ok_all &= ok_e

    # --- AMR lattice --------------------------------------------------
    print("\n=== AMR lattice ===")
    ok_amr = test_amr()
    ok_all &= ok_amr

    print()
    print("PASS" if ok_all else "FAIL — see above")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
