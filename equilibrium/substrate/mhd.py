"""
MHD-with-anisotropic-pressure kernels on the occupation field occ[N,14].

Extracted from vis/substrate/plot_brazil_instabilities.py (the DST Brazil-plot
pipeline) so other scripts — e.g. run/substrate/mirror_instability_dst.py —
can reuse the same physics without duplicating it.

Field identification:
    phi[i,k]   <->  vector potential A (face 1-form)
    occ[i,k]   <->  matter current per face (occupation field)
    B          =    curl A (plaquette sum)
    phi_bg[i,k]  <-> ground-state background A: the substrate's own base/
                     rest configuration, present everywhere in the domain
                     from the IC (see capture_ground_state) -- NOT a
                     boundary-imposed or far-field quantity
    beta_para  =    (occ·b_hat)²  / (lam · |B|²)
    T_perp/T_para = beta_perp/beta_para = (|occ|²−(occ·b_hat)²) / (2(occ·b_hat)²)

Stability boundaries (kinetic MHD / CGL):
    Firehose:  R < 1 − 1/beta_para   (field-line buckling; beta_para > 1 only)
    Mirror:    R > 1 + 1/beta_para   (magnetic-bottle trapping)

Electron-closure analogue (step_hall): classical Hall-MHD closes the system
with a massless electron fluid, contributing a Hall term (J×B)/(n e) and an
electron-pressure-gradient term to the generalized Ohm's law -- corrections
to pure ideal-MHD frozen-in that only matter at scales where the field
decouples from the bulk matter flow. DST doesn't import that machinery
literally (no n_e, e, or electron fluid) -- see step_hall below for the
substrate-native analogue: a field-self-sourced correction built purely from
A (not B), weighted by the existing chi=lam/(1+lam) discreteness proxy, and
referenced against a separate, non-evolving "ground state" field (phi_bg)
rather than the local instantaneous field.

Ground-state restoring force (step_ground_restore): real magnetic fields
are never isolated -- overlapping fields from nearby sources (e.g. between
protons) mean any local field structure is always superposed on, and
eventually pulled back toward, the surrounding aggregate background, not
left to drift freely the way step_frozen_in's exact rotation (proven
structurally non-conservative, see project_mhd_energy_structural_
nonconservation memory) otherwise allows. step_ground_restore is a literal
spring-like relaxation of phi toward phi_bg -- distinct from step_hall
(which only uses phi_bg's direction as an axis, never its value) and from
step_phi_diffuse (which relaxes toward the local neighborhood's current
consensus, not toward a fixed background at that cell). Universal by
design, not chi-weighted -- the physical justification (pervasive field
overlap) doesn't depend on local discreteness.
"""

import numpy as np

# ---------------------------------------------------------------------------
# BCC constants

SQ_PLAQUETTES = {
    "z": (8, 10, 9, 11),  # P_xy -> B_z
    "y": (8, 12, 9, 13),  # P_xz -> B_y
    "x": (10, 12, 11, 13),  # P_yz -> B_x
}
_SQ_AXIS_PAIRS = [(8, 9), (10, 11), (12, 13)]  # (forward, backward) for x, y, z

# step_phi_diffuse / the vector Laplacian are complete operators: they take the
# Kelvin Hodge star 3:1, not step()'s per-leg W = sqrt3.
from equilibrium.substrate.lattice import W_STAR as _W  # noqa: E402
from equilibrium.substrate.lattice import _K_REV  # noqa: E402
W_HEX, W_SQ = float(_W[0]), float(_W[8])


# ---------------------------------------------------------------------------
# Field kernels


def _plaquette_curl(phi, nb, i, faces):
    cells = [i]
    for k in faces[:-1]:
        j = int(nb[cells[-1], k])
        if j < 0:
            return None
        cells.append(j)
    if int(nb[cells[-1], faces[-1]]) != i:
        return None
    return sum(float(phi[cells[m], faces[m]]) for m in range(4))


def _b_vec(phi, nb, i):
    """B = (B_x, B_y, B_z) from plaquette sums; None if boundary."""
    b = []
    for key in ("x", "y", "z"):
        c = _plaquette_curl(phi, nb, i, SQ_PLAQUETTES[key])
        if c is None:
            return None
        b.append(c)
    return np.array(b)


def _occ_vec(occ_i):
    """Net 3D matter-current vector from a 14-element occupation row."""
    return np.array([occ_i[f] - occ_i[b] for f, b in _SQ_AXIS_PAIRS])


def _b_vec_safe(lat, i):
    """_b_vec with magnitude guard: None if any component > 1e100 or non-finite."""
    B = _b_vec(lat.phi, lat.nb, i)
    if B is None:
        return None
    if not np.all(np.isfinite(B)) or np.any(np.abs(B) > 1e100):
        return None
    return B


# ---------------------------------------------------------------------------
# MHD step kernels


def step_frozen_in(lat, occ, dt, cells=None):
    """
    Ideal-MHD frozen-in: ΔA = dt * (occ_vec × B) onto sq faces.
    Cells where B has grown super-Alfvénically large are silently skipped.
    cells: pre-filtered B-cell list for ~100× speedup (optional).
    """
    if cells is None:
        cells = range(lat.N)
    dphi = np.zeros_like(lat.phi)
    for i in cells:
        B = _b_vec_safe(lat, i)
        if B is None:
            continue
        v = _occ_vec(occ[i])
        dA = dt * np.cross(v, B)
        if not np.all(np.isfinite(dA)):
            continue
        for ax, (fp, fm) in enumerate(_SQ_AXIS_PAIRS):
            dphi[i, fp] += dA[ax]
            dphi[i, fm] -= dA[ax]
    lat.phi += dphi


def step_occ_lorentz(lat, occ, dt, cells=None):
    """
    Cyclotron precession via the Cayley transform of occ_vec around b_hat,
    not a linearized Δocc = dt*(b_hat × occ_vec) Euler step.

    The linearized version inflates |occ_vec| every step -- the textbook
    explicit-Euler-on-a-rotation error -- which compounds into an unbounded
    leak over many steps (confirmed to be the actual driver of DST's
    unrestrained mirror-instability growth, not missing physics).

    Deliberately not using sin/cos (Rodrigues' formula): those presuppose an
    exactly-periodic continuous rotation group as ground truth, which isn't
    how DST periods work -- they're persistent patterns emergent from
    discrete pressure balance, not a primitive. The Cayley transform is a
    purely algebraic (rational, no transcendental functions) map that
    preserves |v|² exactly for any generator, by a polynomial identity, with
    no periodicity assumed anywhere in the derivation:

        g     = 0.5 * dt * b_hat
        v_new = ((1-|g|²)*v + 2*(g·v)*g + 2*(g×v)) / (1+|g|²)

    Matches the small-dt precession to O(dt³); |occ_vec|² conservation is
    exact for any dt (both properties hold algebraically, not just in the
    dt→0 limit). Cells with large/non-finite B are skipped.
    cells: pre-filtered B-cell list for ~100× speedup (optional).
    """
    if cells is None:
        cells = range(lat.N)
    for i in cells:
        B = _b_vec_safe(lat, i)
        if B is None:
            continue
        B_sq = float(np.dot(B, B))
        if B_sq < 1e-30:
            continue
        b_hat = B / np.sqrt(B_sq)
        v = _occ_vec(occ[i])
        g = 0.5 * dt * b_hat
        g_sq = float(np.dot(g, g))
        v_new = (
            (1.0 - g_sq) * v + 2.0 * float(np.dot(g, v)) * g + 2.0 * np.cross(g, v)
        ) / (1.0 + g_sq)
        if not np.all(np.isfinite(v_new)):
            continue
        for ax, (fp, fm) in enumerate(_SQ_AXIS_PAIRS):
            occ[i, fp] = max(v_new[ax], 0.0)
            occ[i, fm] = max(-v_new[ax], 0.0)


def step_phi_diffuse(lat, dt, eta=0.1):
    """
    Phi equilibration: each face component relaxes toward the average of
    what it sees across all 14 neighbor directions, weighted by hex/sq face
    type -- a discrete Laplacian, i.e. isotropic pressure from every
    neighboring direction pulling a local excess back down:

        dphi[i,k] = eta * sigma_i * dt * Σ_j W_j * (phi[nb[i,j],k] - phi[i,k])

    This is the same rule as DST's u-diffusion and the Harris-reconnection
    phi-diffusion (see doc/mems/project_dst_phi_diffusion.md), not a new
    mechanism: every state change has a throughput cost (sigma(lam)), so
    relaxation toward the local neighborhood is not optional -- it's what
    the substrate does by default. Not an entropy/thermodynamic argument,
    just equilibration.

    frozen_in and step_occ_lorentz are exact rotations (reversible, no
    equilibration); this is the irreversible piece missing from the raw
    kernel set that provides an actual sink for the induction-driven field
    growth those two, by themselves, cannot dissipate (see
    project_mhd_energy_structural_nonconservation memory).

    This is an ANALOGUE of resistivity (dissipates only when there's a
    gradient/current, same qualitative role as the eta*J term in resistive
    MHD), not a direct implementation of Ohm's law -- eta here is a
    DST-native equilibration strength (chosen empirically, verified to
    behave well in tst/substrate/test_mhd.py), not a material resistivity
    value in any physical unit system, and isn't derived from V=IR or
    eta=rho/A. Don't read this kernel as "DST's resistivity coefficient";
    it's a structurally similar but independently-motivated mechanism.

    Deliberately vectorised (no per-cell Python loop, no B/curl needed) --
    applies everywhere, not just at B-cells, matching the Harris-reconnection
    reference implementation.

    Performance note: lat.nb (neighbor topology) is fixed for the lattice's
    lifetime, so the per-face gather indices are computed once and cached on
    the lattice object (_phi_diffuse_topology_cache) rather than rebuilt
    from scratch (14x np.arange + 14x np.where) on every call -- recomputing
    them every step is invisible in a short isolated benchmark but compounds
    badly over the hundreds of repeated calls a real run makes. The per-step
    update also mutates buffers in place instead of allocating a fresh
    (N,14) temporary at every intermediate step.
    """
    safe_idx, invalid = _phi_diffuse_topology(lat)
    dphi = np.zeros_like(lat.phi)
    for k in range(14):
        diff = lat.phi[safe_idx[k]]  # gather -- the one unavoidable (N,14) allocation
        diff -= lat.phi
        diff[invalid[k]] = 0.0
        diff *= _W[k]
        dphi += diff
    # one update per face, shared by both stored copies (phi_ki = -phi_ik survives; the per-cell
    # sigma_i weighting and missing boundary neighbours broke it, exp/2026/09/29): sigma at the
    # face, and the two copies' Laplacians averaged.
    sigma = 1.0 / (1.0 + lat.lam)
    valid = lat.nb >= 0
    j = np.where(valid, lat.nb, 0)
    partner = dphi[j, _K_REV[None, :]]
    sig_f = 0.5 * (sigma[:, None] + np.where(valid, sigma[j], sigma[:, None]))
    dphi = np.where(valid, sig_f * 0.5 * (dphi - partner), sigma[:, None] * dphi)
    lat.phi += dt * eta * dphi


def _phi_diffuse_topology(lat):
    """
    Cache the 14 per-face gather-index arrays (and validity masks) used by
    step_phi_diffuse. Recomputed only if lat.nb has never been cached or has
    been replaced (identity check) -- otherwise reused across every call.
    """
    cache = getattr(lat, "_phi_diffuse_topology_cache", None)
    if cache is not None and cache[0] is lat.nb:
        return cache[1]
    N = lat.N
    arange_n = np.arange(N)
    safe_idx = np.empty((14, N), dtype=np.intp)
    invalid = np.empty((14, N), dtype=bool)
    for k in range(14):
        nb_k = lat.nb[:, k]
        valid_k = nb_k >= 0
        safe_idx[k] = np.where(valid_k, nb_k, arange_n)
        invalid[k] = ~valid_k
    result = (safe_idx, invalid)
    lat._phi_diffuse_topology_cache = (lat.nb, result)
    return result


def step_occ_stream(lat, occ, dt, b_cell_list):
    """
    Lorentz-analog transport: occ streams between adjacent B-cells along A.

    For each sq face k connecting B-cell i to B-cell j, the A-flux phi[i,k]
    drives the corresponding occ component from i toward j:

        Δocc_vec[j][ax] += dt · phi[i, face] · occ_vec[i][ax]
        Δocc_vec[i][ax] -= dt · phi[i, face] · occ_vec[i][ax]

    This is the transport complement to frozen-in (A changes due to occ) and
    cyclotron (occ rotates in place due to curl A): here occ is carried between
    cells by the A field, mixing arc families and breaking the per-cell trajectories.

    Only operates between cells both in b_cell_list, so the cost is O(n_b²).
    """
    nb = lat.nb
    b_set = set(b_cell_list)
    dv = np.zeros((lat.N, 3))

    for i in b_cell_list:
        curl_a_i = _b_vec_safe(lat, i)
        if curl_a_i is None:
            continue
        v_i = _occ_vec(occ[i])
        if not np.all(np.isfinite(v_i)) or np.dot(v_i, v_i) > 1e100:
            continue
        for ax, (fp, fm) in enumerate(_SQ_AXIS_PAIRS):
            for face in (fp, fm):
                j = int(nb[i, face])
                if j < 0 or j not in b_set:
                    continue
                v_j = _occ_vec(occ[j])
                if not np.all(np.isfinite(v_j)) or np.dot(v_j, v_j) > 1e100:
                    continue
                # Diffusive exchange: flux ∝ curl_A * (v_j - v_i)
                # Anti-symmetric → conserves total occ, bounded, no amplification.
                # Mixing rate set by |curl A| — stronger field = faster diffusion.
                ca_ax = float(curl_a_i[ax])
                flux = dt * abs(ca_ax) * (v_j[ax] - v_i[ax])
                dv[i, ax] += flux
                dv[j, ax] -= flux

    for i in b_cell_list:
        curl_a_i = _b_vec_safe(lat, i)
        if curl_a_i is None:
            continue
        v_new = _occ_vec(occ[i]) + dv[i]
        if not np.all(np.isfinite(v_new)) or np.any(np.abs(v_new) > 1e100):
            continue
        for ax, (fp, fm) in enumerate(_SQ_AXIS_PAIRS):
            occ[i, fp] = max(v_new[ax], 0.0)
            occ[i, fm] = max(-v_new[ax], 0.0)


def step_occ_n(lat, occ, occ_n, dt, cells=None):
    """
    Continuity step: ∂occ_n/∂t + ∇·(occ_n · v_hat) = 0

    Density advects with the unit fluid velocity v_hat = occ_vec / |occ_vec|.
    Upwind scheme: flux from upstream cell only.  Conserves total occ_n.
    """
    if cells is None:
        cells = list(range(lat.N))
    nb = lat.nb
    cell_set = set(cells)
    docc_n = np.zeros(lat.N)
    for i in cells:
        v = _occ_vec(occ[i])
        v_sq = float(np.dot(v, v))
        if v_sq < 1e-30:
            continue
        v_hat = v / np.sqrt(v_sq)
        for ax, (fp, fm) in enumerate(_SQ_AXIS_PAIRS):
            if v_hat[ax] > 0:
                j = int(nb[i, fp])
                if j >= 0 and j in cell_set:
                    flux = dt * occ_n[i] * v_hat[ax]
                    docc_n[i] -= flux
                    docc_n[j] += flux
            elif v_hat[ax] < 0:
                j = int(nb[i, fm])
                if j >= 0 and j in cell_set:
                    flux = dt * occ_n[i] * (-v_hat[ax])
                    docc_n[i] -= flux
                    docc_n[j] += flux
    occ_n += docc_n
    np.clip(occ_n, 0.0, None, out=occ_n)


def step_occ_instab_limit(lat, occ, b_cell_list, alpha):
    """
    Soft pitch-angle scattering at instability thresholds.

    For each B-cell, computes R = beta_perp/beta_para and the local stability limits:
        R_mirror   = 1 + 1/beta_para   (mirror)
        R_firehose = 1 − 1/beta_para   (firehose, beta_para > 1 only)

    When outside a boundary the occ vector is rotated back toward stability
    proportionally to the excess:
        Mirror-unstable (R > R_mirror):    v_perp ← v_perp * (1 − α*(R−R_m))
        Firehose-unstable (R < R_firehose): v_para ← v_para * (1 − α*(Rf−R))

    After each rotation |occ_vec| is renormalised so kinetic energy is
    conserved.  This creates pile-up just inside the stability boundaries —
    the key feature of the solar-wind Brazil distribution.

    alpha: scattering strength per step (0.2–0.4 gives realistic pile-up).
    """
    for i in b_cell_list:
        B = _b_vec_safe(lat, i)
        if B is None:
            continue
        B_sq = float(np.dot(B, B))
        if B_sq < 1e-30:
            continue
        b_hat = B / np.sqrt(B_sq)
        lam_i = max(float(lat.lam[i]), 1.0)

        v = _occ_vec(occ[i])
        v_sq = float(np.dot(v, v))
        if v_sq < 1e-30:
            continue

        v_para_s = float(np.dot(v, b_hat))
        v_para = v_para_s * b_hat
        v_perp = v - v_para

        T_para = v_para_s**2
        T_perp = float(np.dot(v_perp, v_perp))
        denom = lam_i * B_sq

        if T_para < 1e-30:
            continue
        bp = T_para / denom
        bq = 0.5 * T_perp / denom
        R = bq / bp

        R_mirror = 1.0 + 1.0 / bp
        R_fh = (1.0 - 1.0 / bp) if bp > 1.0 else 0.0

        if R > R_mirror:
            excess = R - R_mirror
            v_perp_new = v_perp * max(1.0 - alpha * excess, 0.0)
            v_new = v_para + v_perp_new
        elif bp > 1.0 and R < R_fh:
            deficit = R_fh - R
            v_para_new = v_para * max(1.0 - alpha * deficit, 0.0)
            v_new = v_para_new + v_perp
        else:
            continue

        vn = float(np.linalg.norm(v_new))
        if vn < 1e-30:
            continue
        v_new = v_new / vn * np.sqrt(v_sq)  # renormalise

        for ax, (fp, fm) in enumerate(_SQ_AXIS_PAIRS):
            occ[i, fp] = max(v_new[ax], 0.0)
            occ[i, fm] = max(-v_new[ax], 0.0)


# ---------------------------------------------------------------------------
# Electron-closure analogue (Hall-term stand-in)


def capture_ground_state(lat):
    """
    Snapshot the current lat.phi as the ground-state background A-field,
    lat.phi_bg -- a second, permanently static array, separate from the
    evolving lat.phi.

    This is the substrate's own base/rest configuration -- present
    everywhere in the domain from the deterministic exogenous source (e.g.
    _set_solenoid), not a boundary-imposed or "far field" quantity (there's
    no exterior region distinct from the domain to impose it from). It IS
    still not a literal t=0 snapshot of the simulated state in one sense:
    call this right after the exogenous source is applied and before any
    noise/evolution, so lat.phi_bg captures only that deterministic
    configuration, not whatever noise/perturbation gets layered on next.

    Safe to call again later (e.g. after re-imposing a source) -- clears the
    b_hat0 cache so step_hall/step_ground_restore recompute from the new
    phi_bg.
    """
    lat.phi_bg = lat.phi.copy()
    if hasattr(lat, "_b_bg_cache"):
        del lat._b_bg_cache


def _b_bg_hat(lat, i):
    """
    Normalised background-field direction at cell i, from lat.phi_bg
    (lazily cached per-cell dict, since lat.phi_bg never changes between
    capture_ground_state calls). None if lat.phi_bg is absent, at a
    boundary, or the background field is zero there.
    """
    if not hasattr(lat, "phi_bg"):
        return None
    cache = getattr(lat, "_b_bg_cache", None)
    if cache is None:
        cache = {}
        lat._b_bg_cache = cache
    if i in cache:
        return cache[i]
    B0 = _b_vec(lat.phi_bg, lat.nb, i)
    b_hat0 = None
    if B0 is not None:
        B0_sq = float(np.dot(B0, B0))
        if B0_sq >= 1e-30 and np.all(np.isfinite(B0)):
            b_hat0 = B0 / np.sqrt(B0_sq)
    cache[i] = b_hat0
    return b_hat0


def _laplacian_vec(lat, vec_field):
    """
    Discrete vector Laplacian of a derived (N,3) field, gathered across all
    14 neighbor directions with the same hex/sq weights as step_phi_diffuse:

        Lap[i] = Σ_k W_k * (vec_field[nb[i,k]] - vec_field[i])

    Reuses step_phi_diffuse's cached neighbor-index topology (it's the same
    lat.nb-derived gather regardless of which field is being differenced).
    """
    safe_idx, invalid = _phi_diffuse_topology(lat)
    lap = np.zeros_like(vec_field)
    for k in range(14):
        diff = vec_field[safe_idx[k]] - vec_field
        diff[invalid[k]] = 0.0
        lap += _W[k] * diff
    return lap


def step_hall(lat, dt, amplitude=0.1, cells=None):
    """
    Substrate-native analogue of the electron-closure (Hall) correction to
    frozen-in induction.

    Classical Hall-MHD's induction correction has spatial shape
    curl(curl A) -- in Coulomb-like gauge this equals -Laplacian(A), so it's
    built here purely from A (never from B, per DST's preference for A as
    the fundamental quantity, B as its secondary derivative): the discrete
    vector Laplacian of phi's square-face reduction stands in for curl(B)
    without ever computing the evolving B.

    The reference axis is deliberately NOT the local, evolving field
    direction (step_occ_lorentz already precesses occ_vec around that) --
    it's the ground-state background captured in lat.phi_bg (see
    capture_ground_state), a second, non-evolving field representing the
    substrate's own base configuration (not a boundary/far-field concept --
    see feedback_ground_state_not_boundary memory). This is what makes it
    an electron-closure analogue rather than a second frozen-in: the
    correction is sourced by how the *evolving* field's own structure
    (Lap_A) misaligns with the *base* state, not by the matter current.

        dphi_hall[i] = dt * chi_i * amplitude * cross(LapA_vec[i], b_hat0[i])

    chi_i = lam_i/(1+lam_i), the existing discreteness weight (reused from
    its role in equilibrium/substrate/lattice.py's amplification feedback):
    zero in vacuum/coarse cells, growing toward 1 in dense/discrete ones --
    the same qualitative scale-dependence as the real Hall term (negligible
    at MHD scales, significant near the discrete floor). Note: scripts that
    never evolve lat.lam away from its default (vacuum, lam=0 everywhere)
    will see this term vanish identically everywhere -- that's the correct,
    documented behavior of chi (see project_lam_dual_role / chi's role as a
    density proxy), not a bug in this kernel. lat.lam must carry real
    density/discreteness information for step_hall to have any effect.

    Structured like step_frozen_in (a source term added to phi), not like
    step_occ_lorentz (a rotation of stored state) -- amplitude is an
    empirical DST-native coupling strength, analogous to phi_diffuse's eta,
    not a physical 1/(n e) Hall coefficient.

    Requires lat.phi_bg (see capture_ground_state) -- raises otherwise.
    cells: pre-filtered B-cell list for speedup (optional).
    """
    if not hasattr(lat, "phi_bg"):
        raise ValueError(
            "step_hall requires lat.phi_bg -- call capture_ground_state(lat) first"
        )
    if cells is None:
        cells = range(lat.N)
    A_vec = np.stack(
        [lat.phi[:, fp] - lat.phi[:, fm] for fp, fm in _SQ_AXIS_PAIRS], axis=1
    )
    lap_A = _laplacian_vec(lat, A_vec)
    chi = lat.lam / (1.0 + lat.lam)
    dphi = np.zeros_like(lat.phi)
    for i in cells:
        b_hat0 = _b_bg_hat(lat, i)
        if b_hat0 is None:
            continue
        lap_i = lap_A[i]
        if not np.all(np.isfinite(lap_i)):
            continue
        dA = dt * float(chi[i]) * amplitude * np.cross(lap_i, b_hat0)
        if not np.all(np.isfinite(dA)):
            continue
        for ax, (fp, fm) in enumerate(_SQ_AXIS_PAIRS):
            dphi[i, fp] += dA[ax]
            dphi[i, fm] -= dA[ax]
    lat.phi += dphi


def step_ground_restore(lat, dt, k=0.05, cells=None):
    """
    Restoring force pulling phi back toward the ground state phi_bg.

    Physical motivation: real magnetic fields are never isolated -- fields
    from nearby sources overlap pervasively (a proton's own field always
    overlaps its neighbors'), so any local field structure -- a proton, a
    mirror-instability-driven perturbation -- is always superposed on, and
    eventually pulled back toward, the surrounding aggregate background.
    step_frozen_in/step_occ_lorentz are exact rotations with no restoring
    tendency of their own (see project_mhd_energy_structural_
    nonconservation), so nothing currently resists a field direction
    drifting away from where it started -- the mechanism flagged as still
    missing in project_brazil_plot_shape_investigation.

        dphi_restore[i] = -dt * k * (phi[i] - phi_bg[i])

    A literal spring-like relaxation toward the fixed ground state --
    distinct from step_hall (uses phi_bg only as a direction/axis, never
    its value) and from step_phi_diffuse (relaxes toward the local
    neighborhood's CURRENT consensus, not toward a fixed background at
    that cell). Deliberately NOT weighted by chi=lam/(1+lam) -- the
    physical justification (pervasive field overlap) is framed as
    universal, not confined to dense/discrete cells, unlike step_hall's
    scale-dependent Hall analogue. k is an empirical DST-native restoring
    rate (like phi_diffuse's eta), not a physical spring constant.

    Requires lat.phi_bg (see capture_ground_state) -- raises otherwise.
    cells: optional cell subset; None (default) applies domain-wide
    (vectorised, matching step_phi_diffuse's whole-domain style, since the
    restoring pull isn't specific to B-cells).
    """
    if not hasattr(lat, "phi_bg"):
        raise ValueError(
            "step_ground_restore requires lat.phi_bg -- call capture_ground_state(lat) first"
        )
    if cells is None:
        lat.phi -= dt * k * (lat.phi - lat.phi_bg)
    else:
        idx = np.fromiter(cells, dtype=np.intp)
        lat.phi[idx] -= dt * k * (lat.phi[idx] - lat.phi_bg[idx])


def step_ground_restore_direction(lat, dt, k=0.05, cells=None):
    """
    Direction-only ground-state restoring force.

    step_ground_restore pulls the FULL phi vector (magnitude and direction
    together) toward phi_bg -- empirically found to also suppress
    step_phi_diffuse's intended |B|^2 dissipation (verified: at k=20, |B|^2
    barely decayed from 1.0 to ~0.8 over 250 steps vs. decaying to ~0.015
    with no restoring force), fighting an already-validated mechanism
    (project_dst_kinetic_category_question memory) rather than just fixing
    the field-direction drift it was built for.

    This variant only nudges DIRECTION, leaving magnitude free to evolve
    under frozen_in/phi_diffuse/etc. Operates on A_vec, the per-cell 3-
    vector reduction of phi's square faces (same reduction step_hall uses)
    -- not directly on B, because a local phi write's effect on B at that
    same cell is a rank-2, non-invertible map (dB_x=dA_y, dB_y=dB_z=dA_x,
    dA_z has zero local effect -- verified directly, not assumed), so an
    exact "rotate B, preserve |B|" kernel isn't achievable via local writes
    alone. Restoring A_vec's direction is an empirical stand-in, not
    guaranteed equivalent to restoring B's direction -- check both
    B-direction drift AND |B|^2 behavior when using this, don't assume.

        v = A_vec[i]; v0_hat = normalize(A_vec_bg[i]); v_hat = normalize(v)
        pull      = dt * k * (v0_hat - (v0_hat . v_hat) * v_hat)   # tangential only
        v_hat_new = normalize(v_hat + pull)
        v_new     = |v| * v_hat_new                                # |A_vec| preserved exactly

    Algebraic (no trig), same style as step_occ_lorentz's magnitude-
    preserving construction. Requires lat.phi_bg -- raises otherwise.
    """
    if not hasattr(lat, "phi_bg"):
        raise ValueError(
            "step_ground_restore_direction requires lat.phi_bg -- call "
            "capture_ground_state(lat) first"
        )
    if cells is None:
        cells = range(lat.N)
    dphi = np.zeros_like(lat.phi)
    for i in cells:
        v = np.array([lat.phi[i, fp] - lat.phi[i, fm] for fp, fm in _SQ_AXIS_PAIRS])
        v_sq = float(np.dot(v, v))
        if v_sq < 1e-30:
            continue
        v0 = np.array(
            [lat.phi_bg[i, fp] - lat.phi_bg[i, fm] for fp, fm in _SQ_AXIS_PAIRS]
        )
        v0_sq = float(np.dot(v0, v0))
        if v0_sq < 1e-30:
            continue
        v_mag = np.sqrt(v_sq)
        v_hat = v / v_mag
        v0_hat = v0 / np.sqrt(v0_sq)
        pull = dt * k * (v0_hat - float(np.dot(v0_hat, v_hat)) * v_hat)
        v_hat_new = v_hat + pull
        vhn_sq = float(np.dot(v_hat_new, v_hat_new))
        if vhn_sq < 1e-30:
            continue
        v_hat_new /= np.sqrt(vhn_sq)
        v_new = v_mag * v_hat_new
        dA = v_new - v
        if not np.all(np.isfinite(dA)):
            continue
        for ax, (fp, fm) in enumerate(_SQ_AXIS_PAIRS):
            dphi[i, fp] += dA[ax]
            dphi[i, fm] -= dA[ax]
    lat.phi += dphi


# ---------------------------------------------------------------------------
# Stability boundary curves


def firehose_curve(bp_arr):
    R = 1.0 - 1.0 / bp_arr
    R[R <= 0] = np.nan
    return R


def mirror_curve(bp_arr):
    return 1.0 + 1.0 / bp_arr
