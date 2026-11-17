#ifndef _EQUILIBRIUM_CORE_DATA_LATTICE_CELL_DISCRETE_CELL_HPP_
#define _EQUILIBRIUM_CORE_DATA_LATTICE_CELL_DISCRETE_CELL_HPP_

#include <array>
#include <cstdint>
#include <stdexcept>

namespace equilibrium::dst::proof {

void inline lattice_step_minimum_from_gps_gravitational_data() {
  // Planck length scales with c; GPS clocks gain 45 μs/day due to GR (weaker
  // gravitational potential at orbital altitude → larger local c → larger ℓ_P).
  // The variance in ℓ_P between GPS orbit and Earth surface is the minimum
  // discrete cell-size step resolvable from that measurement.
  //   c_earth = c_orbit × (1 − 45e-6/86400)
  //   Δℓ_P   = ℓ_P_orbit − ℓ_P_earth = ℓ_P × (45e-6 / 86400) ≈ 8.42e-45
  constexpr double gps_gain_s = 45e-6;  // GPS clock gain per day [s]
  constexpr double seconds_day = 86400.0;
  constexpr double l_planck = 1.61626e-35;  // m — Planck length at GPS orbital c

  constexpr double result = l_planck * (gps_gain_s / seconds_day);

  if (result < 1e-45 or result > 1e-43) throw std::runtime_error("PROOF INVALID");
}

}  // namespace equilibrium::dst::proof

namespace equilibrium::dst {

// One discrete Kelvin cell.
// The discrete floor (minimum cell) is normalised to 1 — it is the irreducible unit
// of the substrate.  Void cells are larger by a factor of N_SAGA ≈ 2880.
// The Planck length ℓ_P is NOT the fundamental unit; it is the void cell size,
// an emergent scale derived from locally-measured G, ℏ, c (all themselves emergent).
// Derived quantities: chi = lam/(1+lam),  sigma = 1/(1+lam)
// Neighbour indices and face geometry live in the lattice, not the cell.
struct DiscreteCell {
  // L_MIN = 1: the discrete floor — smallest possible cell, irreducible substrate unit.
  // Currently anchored to S2 periapsis clock deficit ~30 s/day (GRAVITY/ESO 2018)
  // → N_SAGA=2880 floor units per void tick.  Should eventually be derived from
  // Kelvin cell geometry alone, making the S2 observation a verification.
  auto constexpr static L_MIN = 1;        // discrete floor (natural unit = 1)
  auto constexpr static N_SAGA = 2880;    // void ticks per Sgr A* tick (S2 anchor)
  auto constexpr static L_MAX = N_SAGA;   // void cell size in floor units (≈ ℓ_P_local)
  // True void is slightly larger: L_void = N_SAGA × (C_VOID/C_LOCAL) ≈ 2880.005

  // c_void: speed of light in true void. Locally we measure 299,792,458 m/s
  // (inside Milky Way potential). Anchor: Gaia DR2/RAVE escape velocity v_esc≈550 km/s at
  // solar radius (Monari et al. 2018, Bird et al. 2019) → Δc/c = v_esc²/(2c²) ≈ 1.67×10⁻⁶
  // → +502 m/s. In DST c ∝ L, so ℓ_P_void = ℓ_P_local × (c_void/c_local).
  // Placeholder — v_esc uncertainty ~30–50% on the correction term.
  auto constexpr static C_VOID = 299'792'960.0;        // m/s  [placeholder — Gaia/RAVE anchor]
  auto constexpr static C_LOCAL = 299'792'458.0;       // m/s  (SI, locally measured)
  auto constexpr static L_PLANCK_LOCAL = 1.61626e-35;  // m    (void cell size in metres)
  auto constexpr static L_PLANCK_VOID = L_PLANCK_LOCAL * (C_VOID / C_LOCAL);  // ≈ 1.61629e-35 m

  // Sgr A* reference frame — S2 periapsis, 30 s/day clock deficit (clean integer placeholder).
  // c_local = N_SAGA × C_SAGA exactly by construction; in DST c ∝ L → L_SAGA = L_MIN = 1.
  auto constexpr static L_SAGA = L_MIN;              // Sgr A* cell = floor unit (by anchor)
  auto constexpr static C_SAGA = C_LOCAL / N_SAGA;  // ≈ 104,094.603 m/s

  auto constexpr static NFACES = 14u;  // 8 hex + 6 square (Kelvin/BCC)

  double u = 0.0;    // local potential (essence)
  double lam = 0.0;  // resolution index λ  (0 = vacuum)

  std::array<double, NFACES> phi{};  // directed face flux
  std::uint16_t fixed_phi{};         // bitmask: bit k = face k frozen

  bool fixed_u = false;  // true = u pinned
};

}  // namespace equilibrium::dst

#endif  // _EQUILIBRIUM_CORE_DATA_LATTICE_CELL_DISCRETE_CELL_HPP_
