#ifndef _EQUILIBRIUM_CORE_DATA_LATTICE_CELL_ADAPTIVE_CELL_HPP_
#define _EQUILIBRIUM_CORE_DATA_LATTICE_CELL_ADAPTIVE_CELL_HPP_

#include "discrete_cell.hpp"

#include <cstddef>
#include <cstdint>

namespace equilibrium::dst::discrete {

// SoA field layout for the adaptive lattice.
//
// All per-cell data lives in flat arrays owned by AdaptiveLattice.
//   Scalar fields : arr[i]           — one value per cell, contiguous
//   Face fields   : arr[k * N + i]   — face-major, SIMD-friendly over cells for fixed k
//
// Step inner loop (vectorises over i for fixed k):
//   for k in 0..NFACES:
//     du[i] += W[k] * phi[k*N + i]     // contiguous load of phi for face k
//     u_nb   = u[nb[k*N + i]]          // gather — unavoidable, but neighbour-index
//                                      //   reuse across steps amortises the cost
//
// Scale accounting (L = cell size in Planck lengths; L=1 for a void Planck cell):
//   d_ij       = (L_i + L_j) / 2           centre-to-centre distance
//   A_ij       = L_ij^2 * A0               face area  (A0 from Kelvin geometry at L=1)
//   W_junction = W_base[k] * 2/(L_i + L_j) face weight at a scale junction
//   dt_CFL     <= min_i(L_i)               CFL limit — large void cells take large steps
//   sigma_i    = 1 / (1 + lam_i)
//
// Only leaf cells (n_children == 0) participate in the physics step.
// Interior cells hold aggregated values for AMR bookkeeping only.

namespace adaptive_cell {

constexpr std::uint32_t NFACES = DiscreteCell::NFACES;  // 14: 8 hex + 6 sq
constexpr std::uint8_t N_CHILDREN_HEX = 8;              // coarse split through hex faces
constexpr std::uint8_t N_CHILDREN_FULL = 14;            // full Kelvin split (all 14 faces)
// True self-similar BCC split: the 8 octant children (hex-direction body-center
// sites of the refined lattice) PLUS the original cell's own position, which
// becomes a genuine 9th leaf (a corner site of the refined lattice) at the
// halved scale, wired to all 8 siblings via hex faces -- not merely a
// retained coarse accumulator standing in for that volume.
constexpr std::uint8_t N_CHILDREN_HEX_CENTER = 9;
constexpr std::uint8_t CENTER_CHILD_INDEX = 8;  // index of the center child within a 9-split

// Odd (ratio-3) self-similar BCC split: 27 children on a 3x3x3 fine sublattice.
// Unlike the even-factor (ratio-2) splits above, an ODD refinement factor
// preserves the BCC corner/body sublattice parity exactly (3*even=even,
// 3*odd=odd), so every child is a genuine 14-neighbour Kelvin cell rather
// than a degenerate half-scale approximation. See bcc_defs.hpp's
// Odd27Table (built from the same nearest-centre / Voronoi assignment
// validated in tst/substrate/adaptive/probe_split27.py).
constexpr std::uint8_t N_CHILDREN_ODD27 = 27;

}  // namespace adaptive_cell

// ---------------------------------------------------------------------------
// SplitModel — compile-time selector for the split/wire_split_children code
// path. Replaces the old runtime `n_children` dispatch (assert + `if (nc==..)`
// branches) with a template parameter: each instantiation of split() /
// wire_split_children() carries exactly one model's logic, resolved and
// inlined at compile time, no runtime branching left for the model actually
// used.
//
// kDefaultSplitModel is Odd27 (ratio-3, 27 children) — the self-similar,
// parity-preserving split; Hex8/HexCenter9 remain available via an explicit
// template argument for callers that still want the legacy ratio-2 models.
// ---------------------------------------------------------------------------
enum class SplitModel : std::uint8_t { Hex8, HexCenter9, Odd27 };

template <SplitModel M>
struct split_model_traits;

template <>
struct split_model_traits<SplitModel::Hex8> {
  static constexpr std::uint8_t n_children = adaptive_cell::N_CHILDREN_HEX;
  static constexpr double ratio = 2.0;
};

template <>
struct split_model_traits<SplitModel::HexCenter9> {
  static constexpr std::uint8_t n_children = adaptive_cell::N_CHILDREN_HEX_CENTER;
  static constexpr double ratio = 2.0;
};

template <>
struct split_model_traits<SplitModel::Odd27> {
  static constexpr std::uint8_t n_children = adaptive_cell::N_CHILDREN_ODD27;
  static constexpr double ratio = 3.0;
};

inline constexpr SplitModel kDefaultSplitModel = SplitModel::Odd27;

// Strided accessor for one cell's face data in face-major SoA arrays.
// arr layout: arr[k * N + i]  for face k, cell i.
template <typename T>
struct FaceMajorCellSpan {
  T* base;             // pointer to arr[0 * N + i] = arr[i]
  std::size_t stride;  // N — number of cells in the lattice

  T& operator[](std::size_t k) noexcept { return base[k * stride]; }
  T const& operator[](std::size_t k) const noexcept { return base[k * stride]; }
};

// Non-owning view into one cell's data across all SoA arrays.
// Constructed on demand during tree traversal and initialisation — not used in the
// hot step loop (which iterates face-first over all leaf cells without this wrapper).
struct AdaptiveCellRef {
  // Scalar physics
  double& u;
  double& L;                 // scale in Planck lengths (1.0 = single Planck cell)
  double& lam;               // resolution index λ  (0 = vacuum)
  std::uint8_t& fixed_u;     // 0 = free, 1 = pinned  (uint8 for SoA alignment)
  std::uint16_t& fixed_phi;  // bitmask: bit k = face k frozen

  // Face data — strided into face-major arrays
  FaceMajorCellSpan<double> phi;
  FaceMajorCellSpan<std::int32_t> nb;

  // Tree topology
  std::int32_t& parent;       // index of parent cell  (-1 = root)
  std::int32_t& first_child;  // index of first child  (-1 = leaf)
  std::uint8_t& n_children;   // 0 = leaf, N_CHILDREN_HEX or N_CHILDREN_FULL

  std::uint8_t level;  // refinement depth, 0 = coarsest

  // Derived
  [[nodiscard]] double chi() const noexcept { return lam / (1.0 + lam); }
  [[nodiscard]] double sigma() const noexcept { return 1.0 / (1.0 + lam); }
  [[nodiscard]] double dt_limit() const noexcept { return L; }
  [[nodiscard]] bool is_leaf() const noexcept { return n_children == 0; }
  [[nodiscard]] bool is_void() const noexcept { return lam == 0.0; }
};

}  // namespace equilibrium::dst::discrete

#endif  // _EQUILIBRIUM_CORE_DATA_LATTICE_CELL_ADAPTIVE_CELL_HPP_
