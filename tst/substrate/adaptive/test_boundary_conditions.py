"""
Boundary condition correctness tests.

Small ~10³ BCC box, two ICs, several BC contexts.

ICs
---
  flat-u  : u = u_bg everywhere, phi = 0  → completely static
  pulse   : Gaussian u peak at centre, phi = 0  → waves propagate

Periodic BC
-----------
  wire_periodic_bcc sets nb[k] = partner_g directly, so periodic faces are
  regular connected faces for pass1/pass2.  u_vol is conserved to <0.1%.

Run:
    PYTHONPATH=$PWD python3 -m unittest tst/substrate/adaptive/test_boundary_conditions.py
"""

import unittest

import numpy as np

from equilibrium import lattice as _lat
from equilibrium.substrate.adaptive import wire_periodic_bcc

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
D1 = 1.0
H = D1 / np.sqrt(3.0)
BOX = 10.0
U_BG = 1.0
U_AMP = 5.0
SIGMA2 = (2.0 * D1) ** 2
N_STEPS = 10_000
DT = 0.05

# BCC face directions — mirrors bcc_geometry.hpp
_HEX_DIR = np.array(
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
    dtype=float,
)
_SQ_DIR = np.array(
    [[-1, 0, 0], [+1, 0, 0], [0, -1, 0], [0, +1, 0], [0, 0, -1], [0, 0, +1]],
    dtype=float,
)
_ALL_DIR = np.vstack([_HEX_DIR, _SQ_DIR])  # (14, 3)


# Mirror face: for hex k<8, mirror = k^7; for sq k>=8, mirror = k^1
def _mirror(k):
    return k ^ 7 if k < 8 else (k ^ 1)


def _check_periodic_wiring(cl, BOX, H, axes="xyz"):
    """
    Validate periodic wiring set by wire_periodic_bcc.

    Checks:
      1. Antisymmetry — if nb[g][k] = p, then nb[p][mirror(k)] = g
      2. No self-loops — nb[g][k] != g

    Position is NOT checked: BCC with odd period (BOX/H not integer-even) places
    hex periodic images on the wrong sublattice, so the nearest valid cell is always
    ~H away from the nominal wrapped target — that is a lattice incommensurability,
    not a wiring error.

    Returns (antisym_errors, self_loop_errors) as lists of strings.
    """
    nb = cl.leaf_nb
    gidx = cl.leaf_global_indices
    g2j = {int(gidx[j]): j for j in range(len(gidx))}

    asym, loops = [], []
    for j in range(len(gidx)):
        g = int(gidx[j])
        for k in range(14):
            pg = int(nb[j, k])
            if pg < 0:
                continue
            if pg == g:
                loops.append(f"self-loop: g={g} k={k}")
                continue
            if pg not in g2j:
                asym.append(f"partner g={pg} not a leaf (g={g} k={k})")
                continue
            pj = g2j[pg]
            mk = _mirror(k)
            back = int(nb[pj, mk])
            if back != g:
                asym.append(
                    f"g={g} k={k} → pg={pg}, but pg.mirror={mk} → {back} (expected {g})"
                )
    return asym, loops


def _build(
    u_amp=0.0,
    wire="xyz",
    set_absorbing=False,
    dirichlet_target=None,
    set_outflow=False,
    outflow_c=None,
):
    """Return (evolver, cluster, u_ic, phi_ic)."""
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    cl = alat[0]
    _lat.build_bcc_box(cl, BOX, BOX, BOX, D1, u0=0.0, lam0=0.0)
    alat.rebuild(0)
    alat.set_n_threads(1)

    pos = cl.leaf_pos
    c = BOX / 2.0
    r2 = ((pos - c) ** 2).sum(axis=1)
    u_ic = U_BG + u_amp * np.exp(-r2 / SIGMA2)
    cl.set_leaf_u(u_ic.copy())

    if wire:
        wire_periodic_bcc(cl, H, axes=wire)
    if set_absorbing:
        cl.set_bc(_lat.BoundaryCondition.absorbing())
    if dirichlet_target is not None:
        cl.set_bc(_lat.BoundaryCondition.dirichlet(dirichlet_target))
    if set_outflow:
        kwargs = {} if outflow_c is None else {"c_lattice": outflow_c}
        cl.set_bc(_lat.BoundaryCondition.outflow(cl.leaf_nb, **kwargs))

    phi_ic = cl.leaf_phi.copy()
    evolver = _lat.AdaptiveLatticeEvolver(alat)
    return evolver, cl, u_ic.copy(), phi_ic.copy()


def _run(evolver, n=N_STEPS, dt=DT):
    ts = _lat.TimeStepper.fixed(dt)
    evolver.step(ts, n)


# ---------------------------------------------------------------------------
# flat-u tests  (u_amp=0, phi=0 everywhere)
# Flat-u + zero phi → no u gradient → dphi=0 → completely static regardless of BC.
# ---------------------------------------------------------------------------
class TestPeriodicWiring(unittest.TestCase):
    """Validate wire_periodic_bcc geometry before running dynamics."""

    def setUp(self):
        _, self.cl, _, _ = _build(u_amp=0.0, wire="xyz")

    def test_no_unwired_faces(self):
        """Every face must be connected after full periodic wiring."""
        n = int((self.cl.leaf_nb < 0).sum())
        self.assertEqual(n, 0, f"{n} faces still have nb=-1 after wire_periodic_bcc")

    def test_wiring_correctness(self):
        """Antisymmetry and no self-loops after wire_periodic_bcc."""
        asym, loops = _check_periodic_wiring(self.cl, BOX, H, axes="xyz")
        msgs = []
        if loops:
            msgs.append(f"{len(loops)} self-loops:\n" + "\n".join(loops[:5]))
        if asym:
            msgs.append(f"{len(asym)} antisymmetry errors:\n" + "\n".join(asym[:5]))
        if msgs:
            self.fail("\n".join(msgs))


class TestFlatUStatic(unittest.TestCase):

    def _assert_static(self, wire, set_absorbing=False, label=""):
        evolver, cl, u_ic, phi_ic = _build(
            u_amp=0.0, wire=wire, set_absorbing=set_absorbing
        )
        _run(evolver)
        np.testing.assert_allclose(
            cl.leaf_u,
            u_ic,
            atol=1e-12,
            err_msg=f"u drifted for flat-u [{label}]",
        )
        np.testing.assert_allclose(
            cl.leaf_phi,
            phi_ic,
            atol=1e-12,
            err_msg=f"phi drifted for flat-u [{label}]",
        )

    def test_periodic_xyz(self):
        self._assert_static(wire="xyz", label="periodic xyz")

    def test_open_no_bc(self):
        self._assert_static(wire="", label="open / no BC")

    def test_absorbing(self):
        self._assert_static(wire="", set_absorbing=True, label="absorbing")


# ---------------------------------------------------------------------------
# pulse tests  (u_amp > 0, phi = 0 initially)
# ---------------------------------------------------------------------------
class TestPulsePeriodicBC(unittest.TestCase):
    """Gaussian u pulse in a fully periodic box — no blowup, phi grows, u_vol conserved."""

    def setUp(self):
        evolver, cl, u_ic, _ = _build(u_amp=U_AMP, wire="xyz")
        vol = np.full(cl.n_leaves, D1**3)
        self.u_vol_ic = float(np.sum(u_ic * vol))
        self.phi_E_ic = float(np.sum(cl.leaf_phi**2))
        _run(evolver)
        self.u_final = cl.leaf_u.copy()
        self.phi_final = cl.leaf_phi.copy()
        self.vol = vol

    def test_no_nan(self):
        self.assertTrue(np.all(np.isfinite(self.u_final)), "u has NaN/Inf")
        self.assertTrue(np.all(np.isfinite(self.phi_final)), "phi has NaN/Inf")

    def test_phi_grew(self):
        """u gradient must have driven phi away from zero."""
        phi_E = float(np.sum(self.phi_final**2))
        self.assertGreater(phi_E, self.phi_E_ic + 1e-6, "phi did not grow from zero")

    def test_pulse_dispersed(self):
        """Peak u must decrease as the pulse disperses into waves."""
        u_max_ic = U_BG + U_AMP
        u_max_final = float(self.u_final.max())
        self.assertLess(
            u_max_final,
            u_max_ic * 0.9,
            f"u_max barely changed: IC={u_max_ic:.4f}  final={u_max_final:.4f}",
        )

    def test_u_vol_conserved(self):
        """Σ u·vol must be conserved to <0.1% for periodic BC."""
        u_vol = float(np.sum(self.u_final * self.vol))
        rel = abs(u_vol - self.u_vol_ic) / abs(self.u_vol_ic)
        self.assertLess(
            rel,
            1e-3,
            f"u_vol drifted: IC={self.u_vol_ic:.6e}  final={u_vol:.6e}  rel={rel:.2e}",
        )


class TestPulseAbsorbingBC(unittest.TestCase):
    """Gaussian u pulse with absorbing BC.

    absorbing fn-BC sets dphi=0 at all boundary faces, keeping them at their
    IC value (0).  This acts as a reflective (Dirichlet phi=0) boundary, not
    a true absorber — u_vol is conserved, no energy exits.
    The test verifies stability and phi growth from the interior u gradient.
    """

    def setUp(self):
        evolver, cl, u_ic, _ = _build(u_amp=U_AMP, wire="", set_absorbing=True)
        vol = np.full(cl.n_leaves, D1**3)
        self.u_vol_ic = float(np.sum(u_ic * vol))
        self.phi_E_ic = float(np.sum(cl.leaf_phi**2))
        _run(evolver)
        self.u_final = cl.leaf_u.copy()
        self.phi_final = cl.leaf_phi.copy()
        self.vol = vol

    def test_no_nan(self):
        self.assertTrue(np.all(np.isfinite(self.u_final)), "u has NaN/Inf")
        self.assertTrue(np.all(np.isfinite(self.phi_final)), "phi has NaN/Inf")

    def test_phi_grew(self):
        """Interior u gradient must drive phi growth even with absorbing BC."""
        phi_E = float(np.sum(self.phi_final**2))
        self.assertGreater(phi_E, self.phi_E_ic + 1e-6, "phi did not grow from zero")

    def test_u_vol_conserved(self):
        """absorbing BC with phi_ic=0 is reflective → u_vol stays constant."""
        u_vol = float(np.sum(self.u_final * self.vol))
        rel = abs(u_vol - self.u_vol_ic) / abs(self.u_vol_ic)
        self.assertLess(
            rel,
            1e-3,
            f"u_vol drifted: IC={self.u_vol_ic:.6e}  final={u_vol:.6e}  rel={rel:.2e}",
        )


class TestPulseOutflowBC(unittest.TestCase):
    """Gaussian u pulse with the new first-order Mur/Sommerfeld `outflow` BC.

    Unlike absorbing/dirichlet/remote_flux (see TestFnBasedBCInertOnU and
    [[project_bc_fn_inert_on_u]]), outflow is wired into _step_pass1 itself,
    so it should be the first BC in this file to actually drain u_vol rather
    than merely reflect. Primary concern given the exponential-instability
    history in [[project_bc_enum]]: stability first, then genuine loss.
    """

    def setUp(self):
        evolver, cl, u_ic, _ = _build(u_amp=U_AMP, wire="", set_outflow=True)
        vol = np.full(cl.n_leaves, D1**3)
        self.u_vol_ic = float(np.sum(u_ic * vol))
        self.phi_E_ic = float(np.sum(cl.leaf_phi**2))
        _run(evolver)
        self.u_final = cl.leaf_u.copy()
        self.phi_final = cl.leaf_phi.copy()
        self.vol = vol

    def test_no_nan(self):
        self.assertTrue(np.all(np.isfinite(self.u_final)), "u has NaN/Inf")
        self.assertTrue(np.all(np.isfinite(self.phi_final)), "phi has NaN/Inf")

    def test_no_blowup(self):
        """Guard against the exact exponential instability project_bc_enum
        already hit once from naively including boundary phi in pass1."""
        u_max_ic = U_BG + U_AMP
        self.assertLess(
            float(np.max(np.abs(self.u_final))),
            u_max_ic * 10,
            "u grew far beyond IC scale -- looks like the known exponential "
            "instability from including boundary phi in pass1 unguarded",
        )

    def test_phi_grew(self):
        phi_E = float(np.sum(self.phi_final**2))
        self.assertGreater(phi_E, self.phi_E_ic + 1e-6, "phi did not grow from zero")

    def test_u_vol_genuinely_drains(self):
        """The whole point: unlike absorbing/periodic, flux should actually
        leave through the boundary, so u_vol must drop by much more than
        periodic BC's <0.1% numerical drift."""
        u_vol = float(np.sum(self.u_final * self.vol))
        rel_drop = (self.u_vol_ic - u_vol) / abs(self.u_vol_ic)
        self.assertGreater(
            rel_drop,
            0.01,
            f"u_vol barely changed: IC={self.u_vol_ic:.6e} final={u_vol:.6e} "
            f"rel_drop={rel_drop:.2e} -- outflow should drain noticeably more "
            "than periodic BC's <0.1% numerical drift",
        )


class TestDirichletBC(unittest.TestCase):
    """fn boundary faces reach u on unrefined clusters (2026-09-29, _bc_face_to_u): a Dirichlet
    ghost is a fixed-u neighbour at the target (replaces the TestFnBasedBCInertOnU
    characterisation, [[project_bc_fn_inert_on_u]])."""

    def test_at_target_is_static(self):
        evolver, cl, u_ic, _ = _build(wire="", dirichlet_target=U_BG)
        _run(evolver, n=2000)
        np.testing.assert_array_equal(cl.leaf_u, u_ic)

    def test_pulls_toward_target_conserving_energy(self):
        """u = U_BG with the walls at U_BG + 1: u moves toward the target, and the energy
        measured from the target (1/2 (u - t)^2 + interior and boundary-face phi^2) is conserved
        -- a fixed-u neighbour adds none. Peaks reach ~4-5 from inward waves focusing in the box
        (measured to 60k steps: H within +-2%)."""
        target = U_BG + 1.0
        evolver, cl, u_ic, _ = _build(wire="", dirichlet_target=target)
        nb = np.asarray(cl.leaf_nb)
        inter = nb >= 0

        def H():
            u = np.asarray(cl.leaf_u)
            ph = np.asarray(cl.leaf_phi).T
            return 0.5 * np.sum((u - target) ** 2) + 0.25 * np.sum(ph[inter] ** 2) + 0.5 * np.sum(ph[~inter] ** 2)

        h0 = H()
        _run(evolver)
        u = np.asarray(cl.leaf_u)
        self.assertGreater(float(u.mean()), U_BG + 0.2)
        self.assertAlmostEqual(H() / h0, 1.0, delta=0.05)

    def test_pulse_stays_bounded(self):
        evolver, cl, u_ic, _ = _build(u_amp=U_AMP, wire="", dirichlet_target=U_BG)
        _run(evolver)
        u = np.asarray(cl.leaf_u)
        self.assertTrue(np.all(np.isfinite(u)))
        self.assertLess(float(np.abs(u - U_BG).max()), U_AMP)


if __name__ == "__main__":
    unittest.main()
