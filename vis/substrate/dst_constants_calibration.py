"""
DST constants calibration — proton ring geometry, chi field, and speed limit.

Computes:
  - Chi field units and limits (void, BH, event horizon)
  - DST derivation of the speed limit: v_max = (1-chi)*c  (falls out of update rule)
  - Physical chi at the proton surface using Newton's G
  - SR coherence bound corrected for physical ring size
  - Orbit ratio: simulation vs physical (a_0/r_p)
  - Winding period (ell cancels — depends only on ring circumference)
  - N_k (proton mass in Planck units — sub-Planck)

Key findings:
  1. The speed limit is a tick-count constraint, not an imposed SR law.
     v_max = sigma*c = (1-chi)*c falls directly from the update rule.
     chi=0 => photon at c.  chi=1 => frozen (event horizon).  Proton lives in between.
     The proton cannot reach c because its ring tiles have chi>0 (sigma<1) — it needs
     more than one tick to propagate one cell.  This IS special relativity in DST.

  2. G is a fudge factor, not a fundamental constant.  A correct DST model should
     reproduce the physical orbit ratio (a_0/r_p = 62,892) from the update rule alone
     with no free constant.  The current simulation gives ~5.9 — a factor of ~10,600
     off.  That error is a model quality metric, not a measurement of G.  Reducing it
     requires correct ring size, emergent chi from formation energy, and correct alpha
     from the substrate constitutive relation.

  3. The SR bound chi <= 1-beta was wrong for a 1-hop ring assumption.
     For the physical ring (N_hop ~ 3e20): chi <= 1 - beta/N_hop ~ 1.  No constraint.
     chi=0.9 is compatible with LHC-speed protons at physical ring scale.
"""

import numpy as np
import scipy.constants as sc


def main():
    l_P = sc.physical_constants["Planck length"][0]
    t_P = sc.physical_constants["Planck time"][0]
    m_P = sc.physical_constants["Planck mass"][0]
    c = sc.c
    G = sc.G
    hbar = sc.hbar
    m_p = sc.m_p
    r_p = 0.8414e-15  # proton charge radius (CODATA 2018), m
    a_0 = sc.physical_constants["Bohr radius"][0]
    kappa = G / c**2  # m/kg

    print("=" * 60)
    print("DST CONSTANTS CALIBRATION")
    print("=" * 60)

    # --- DST speed limit (SR from the update rule) ---------------------------
    print("\n--- DST speed limit: v_max = (1-chi)*c ---")
    print(
        "  Update rule: each tile steps at rate sigma = ell/(1+lam) = (1-chi)/(1+lam)"
    )
    print(
        "  A pattern (proton ring) shifts one cell per tick only if sigma=1 everywhere."
    )
    print("  If chi>0 at ring tiles: sigma<1, ring needs >1 tick per cell hop => v < c")
    print("  => v_max = (1-chi)*c_void   (no SR axioms needed; falls from update rule)")
    print()
    print("  chi=0  => sigma=1 => v=c   (photon / massless, pure void propagation)")
    print("  chi>0  => sigma<1 => v<c   (proton: chi nonzero, can't reach c)")
    print("  chi=1  => sigma=0 => v=0   (event horizon, frozen)")
    print()
    print("  The proton's chi is strictly between 0 and 1.")
    print(
        "  That is why it has mass (can't reach c) and why it moves at all (not frozen)."
    )
    print(
        "  SR is not imposed — it is the tick-count constraint on pattern propagation."
    )

    # --- chi field -----------------------------------------------------------
    print("\n--- chi field: dimensionless bit-occupation ---")
    print("  chi = 1 - ell;  ell = cell size in Planck lengths")
    print("  chi=0   => void  (ell=1, sigma=1, full clock rate)")
    print("  chi=0.9 => compressed (ell=0.1, sigma=0.1)")
    print("  chi=1   => event horizon (ell=0, clock stopped) — forbidden")
    print("  L_MIN   = 1/2880  (from S2/Sgr A* periapsis, GRAVITY/ESO 2018)")
    print(f"  chi_max = 1 - 1/2880 = {1 - 1/2880:.6f}")

    print("\n--- chi gradient ---")
    print("  chi(r) = kappa * N_k / r   =>   grad(chi) = -kappa * N_k / r^2")
    print("  acceleration a = c^2 * grad(chi)  =>  Newton 1/r^2 emerges")
    print("  At event horizon (lam→∞):  |grad(chi)| -> c^2/(4*G*M) -> inf")
    print("  => tidal forces diverge; L_MIN caps this at finite value")

    # --- Physical chi at proton surface under Newton G -----------------------
    print("\n--- Physical chi at proton surface (Newton G) ---")
    chi_rp = kappa * m_p / r_p
    r_S_proton = 2 * G * m_p / c**2
    print(f"  kappa = G/c^2       = {kappa:.4e} m/kg")
    print(f"  m_p                 = {m_p:.4e} kg")
    print(f"  r_p                 = {r_p:.4e} m")
    print(f"  chi(r_p)            = kappa*m_p/r_p = {chi_rp:.4e}")
    print(f"  Schwarzschild r_p   = 2*G*m_p/c^2  = {r_S_proton:.4e} m")
    print(f"  chi(r_p) = r_S / (2*r_p)           = {r_S_proton/(2*r_p):.4e}  (same)")
    print("  => gravitational chi at proton surface is ~1e-39")
    print("     simulation chi=0.9 is NOT gravitational chi")

    # --- N_k in Planck units -------------------------------------------------
    print("\n--- Proton mass in Planck units ---")
    N_k = m_p / m_P
    print(f"  m_Planck = {m_P:.4e} kg")
    print(f"  N_k = m_p / m_Planck = {N_k:.4e}  (sub-Planck: proton < 1 Planck mass)")
    print("  => gravitational effect per cell is negligible in simulation")

    # --- SR coherence bound with correct knot size ---------------------------
    print("\n--- SR coherence bound (corrected for physical ring size) ---")
    print("  Condition: v * t_P <= N_hop * ell_ring * l_P")
    print("  => ell_ring >= beta / N_hop   => chi <= 1 - beta/N_hop")
    gamma_LHC = 7000
    beta_LHC = np.sqrt(1 - 1 / gamma_LHC**2)
    N_hop_sim = 8
    ell_ring_chi09 = 0.1
    N_hop_phys = 2 * np.pi * r_p / l_P  # physical ring circumference in hops
    chi_SR_sim = 1 - beta_LHC / N_hop_sim
    chi_SR_phys = 1 - beta_LHC / N_hop_phys
    print(f"  LHC: gamma={gamma_LHC}, 1-beta={1-beta_LHC:.3e}")
    print(
        f"  8-hop simulation ring:  chi_max = {chi_SR_sim:.4f}  (CANNOT support LHC at chi=0.9)"
    )
    print(
        f"  Physical ring (N_hop={N_hop_phys:.2e}): chi_max = {chi_SR_phys:.6f}  (chi=0.9 OK)"
    )
    print(
        "  => chi=0.9 at ring tiles is compatible with LHC protons IF ring is physical size"
    )
    print("     The 1e-8 SR bound in dst_next.md assumed N_hop=1 — it was wrong")

    # --- Winding period (ell cancels) ----------------------------------------
    print("\n--- Internal winding period ---")
    print("  Winding at c_local = ell * c_void per hop")
    print("  Physical hop size  = ell * l_P")
    print("  Period in void ticks = (N_hop * ell * l_P) / (ell * c_void)")
    print("                       = N_hop * t_P   (ell cancels!)")
    print("  => period depends only on ring circumference, not chi")
    nu_compton = m_p * c**2 / (2 * np.pi * hbar)
    T_compton = 1 / nu_compton
    T_compton_tP = T_compton / t_P
    C_phys_tP = 2 * np.pi * r_p / l_P  # ring circumference in t_P
    print(f"  Proton Compton period     = {T_compton:.4e} s  = {T_compton_tP:.4e} t_P")
    print(f"  Physical ring circumference = 2*pi*r_p/l_P = {C_phys_tP:.4e} t_P")
    print(f"  Ratio (circumference / Compton period) = {C_phys_tP/T_compton_tP:.4f}")
    print(f"  = r_p / lambda_compton_bar = r_p * m_p * c / hbar = {r_p*m_p*c/hbar:.4f}")
    print("  => ring circumference ~ 4 Compton periods (r_p ~ 4 * lambda_bar_compton)")
    print(
        f"  Simulation ring (8 hops, chi=0.9): period = {N_hop_sim/ell_ring_chi09:.0f} void ticks = {N_hop_sim/ell_ring_chi09*t_P:.3e} s"
    )

    # --- Orbit ratio: simulation vs physical ---------------------------------
    print("\n--- Orbit ratio: simulation vs physical ---")
    r_orbit_sim = 7.491  # r_phys from Sweep 14, lattice units
    r_ring_sim = 1.27  # 8-hop ring radius in lattice units
    ratio_sim = r_orbit_sim / r_ring_sim
    ratio_phys = a_0 / r_p
    discrepancy = ratio_phys / ratio_sim
    two_pi_1e4 = 2 * np.pi * 1e4
    print(f"  Physical a_0 / r_p             = {ratio_phys:.1f}")
    print(
        f"  2*pi * 1e4                     = {two_pi_1e4:.1f}  (agreement: {abs(ratio_phys-two_pi_1e4)/two_pi_1e4*100:.2f}%)"
    )
    print("  => a_0 / r_p = 2*pi * 1e4  to within 0.1%")
    print(
        f"  Simulation r_orbit / r_ring    = {ratio_sim:.1f}  (chi=0.9, alpha=0.05, Sweep 14)"
    )
    print(
        f"  simulation ratio / 2*pi        = {ratio_sim / (2*np.pi):.3f}  (≈1: geometry correct)"
    )
    print(
        f"  Discrepancy                    = {discrepancy:.0f}x  ≈ 1e4  (4 orders of magnitude, scale only)"
    )
    print("  => simulation captures the 2*pi geometry; missing 1e4 is a scale gap")
    print("     fix: adaptive mesh spanning 4 orders of magnitude in cell size")

    # --- G measurement programme ---------------------------------------------
    print("\n--- DST G-measurement programme ---")
    print("  Use empirical constants (r_p, m_p, c, a_0, hbar) — no G input.")
    print("  Treat chi_target as free parameter (set by proton formation energy).")
    print(
        "  Sweep chi_target and alpha; find pair that gives r_orbit/r_ring = a_0/r_p."
    )
    print("  Back-calculate G from chi(r_p) = G_DST/c^2 * m_p / r_p:")
    chi_needed = 0.9 * ratio_sim / ratio_phys  # rough scaling if orbit ~ 1/chi
    G_from_orbit = chi_needed * c**2 * r_p / m_p
    print(f"  Rough chi_needed (if orbit ~ 1/chi): {chi_needed:.3e}")
    print(f"  G_DST from chi_needed            = {G_from_orbit:.4e}  m^3 kg^-1 s^-2")
    print(f"  Newton G                         = {G:.4e}  m^3 kg^-1 s^-2")
    print(f"  Discrepancy G_DST/G_Newton       = {G_from_orbit/G:.3e}")
    print()
    print("  Two chi fields in DST:")
    print("  1. Topological chi: set by phi winding energy at proton formation")
    print("     => simulation chi=0.9; NOT related to G; set by creation event energy")
    print("  2. Gravitational chi: chi(r) = G*m_p/c^2/r; gives 1/r^2 force at large r")
    print("     => chi(r_p) ~ 1e-39 for a proton; negligible at simulation scale")
    print("  The orbit lives in the topological chi gradient.")
    print("  The ratio orbit/ring connects topological chi to gravitational G.")


if __name__ == "__main__":
    main()
