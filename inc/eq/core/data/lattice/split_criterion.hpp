#ifndef _EQUILIBRIUM_CORE_DATA_LATTICE_SPLIT_CRITERION_HPP_
#define _EQUILIBRIUM_CORE_DATA_LATTICE_SPLIT_CRITERION_HPP_

// Split/coarsen criterion functor — pluggable decision rule for AMR adapt().
//
// Mirrors the BoundaryCondition pattern:
//   fn(state, hi, lo) returns:  +1 = split,  -1 = coarsen candidate,  0 = keep
//
// Coarsening requires all siblings to return -1 (same semantics as DSTInfo.split_lo).
// NaN criterion always returns 0 (new/unstepped cells are never acted on).
//
// SCALE INVARIANCE
// The same thresholds must work for protons (L ~ 5e19, u ~ 1e58) and planets
// (L ~ 1e30, u ~ ~1e80).  Raw phi values are proportional to u (NOT to L —
// because dphi ~ dt * sigma * W/L * delta_u and dt ∝ L, so phi ~ W * delta_u,
// L-independent at equilibrium).  Therefore the scale-invariant quantity is
// phi / u = relative_u, and the correct preset is relative_u.
//
// Built-in presets:
//   absolute    — threshold on raw criterion_max   (diagnostic / unit-test use only)
//   relative_u  — criterion_max / |u|              (scale-invariant, recommended)
//
// ref parameter: u floor applied in relative_u to prevent blowup in void /
// boundary cells where |u|→0.  Set to u_min of the background field
// (e.g. u_origin/R for a 1/r background at sphere boundary R).
//
// All presets are configurable via FROM(dict) with keys:
//   "type"  — preset name (default "absolute")
//   "hi"    — split threshold    (default 0.0 = disabled)
//   "lo"    — coarsen threshold  (default 0.0 = disabled)
//   "ref"   — u floor            (default 0.0)

#include "dict.hpp"
#include "eq/core/data/lattice/dst_info.hpp"  // for DSTDict typedef

#include <cmath>
#include <cstdint>
#include <string>
#include <algorithm>

namespace equilibrium::dst::discrete {

struct SplitCriterion {
  // Per-cell state passed to fn.
  struct CellState {
    double cmax;            // max |phi[k]| over active faces
    double cmin;            // min |phi[k]| over active faces
    double u;               // local potential
    double L;               // cell size (in d1 units)
    double lam;             // lambda (resolution index)
    double ref;             // u floor (copied from SplitCriterion::ref)
    std::uint8_t level;     // current refinement level
    std::uint8_t max_level; // cluster cap
  };

  // fn(state, hi, lo) → +1 = split, -1 = coarsen candidate, 0 = keep.
  int (*fn)(CellState const& s, double hi, double lo) = nullptr;
  double hi  = 0.0;  // split threshold
  double lo  = 0.0;  // coarsen threshold (0 = disabled)
  double ref = 0.0;  // u floor for relative_u; set to u at the void/boundary

  // ---------------------------------------------------------------------------
  // Predefined criterion implementations
  // ---------------------------------------------------------------------------

  // Raw criterion_max threshold — replicates DSTInfo.split_hi / split_lo.
  // Not scale-invariant; use only for diagnostics or unit tests.
  static int absolute(CellState const& s, double hi, double lo) noexcept {
    if (std::isnan(s.cmax)) return 0;
    if (hi > 0.0 && s.cmax > hi) return 1;
    if (lo > 0.0 && s.cmax < lo) return -1;
    return 0;
  }

  // max|phi| - min|phi| over active faces — fires at field reversals (large spread),
  // not in uniform-field bulk (zero spread).  Correct criterion for null-line refinement
  // in reconnection problems: the null line has phi≈0 on one side and |B0| on the other.
  static int gradient(CellState const& s, double hi, double lo) noexcept {
    if (std::isnan(s.cmax) || std::isnan(s.cmin)) return 0;
    double g = s.cmax - s.cmin;
    if (hi > 0.0 && g > hi) return 1;
    if (lo > 0.0 && g < lo) return -1;
    return 0;
  }

  // phi / max(|u|, ref).
  // phi at equilibrium is L-independent (dphi ~ W*delta_u, dt*sigma/L cancels L),
  // so phi/u is the correct dimensionless relative-flux criterion.  The same
  // hi/lo values apply at any refinement level and any physical scale.
  static int relative_u(CellState const& s, double hi, double lo) noexcept {
    if (std::isnan(s.cmax)) return 0;
    double u_eff = std::max(std::abs(s.u), s.ref);
    if (u_eff == 0.0) return 0;
    double norm = s.cmax / u_eff;
    if (hi > 0.0 && norm > hi) return 1;
    if (lo > 0.0 && norm < lo) return -1;
    return 0;
  }

  // ---------------------------------------------------------------------------
  // Factory helpers
  // ---------------------------------------------------------------------------

  static SplitCriterion make_absolute(double hi, double lo = 0.0, double ref = 0.0) {
    return {absolute, hi, lo, ref};
  }
  static SplitCriterion make_gradient(double hi, double lo = 0.0, double ref = 0.0) {
    return {gradient, hi, lo, ref};
  }
  static SplitCriterion make_relative_u(double hi, double lo = 0.0, double ref = 0.0) {
    return {relative_u, hi, lo, ref};
  }

  // ---------------------------------------------------------------------------
  // Dict-based construction — same pattern as DSTInfo::FROM(dict).
  // Keys: "type" (string), "hi" (double), "lo" (double), "ref" (double).
  // ---------------------------------------------------------------------------
  static SplitCriterion FROM(DSTDict const& dict) {
    auto type = cppdict::get_value(dict, "type", std::string{"absolute"});
    double hi  = cppdict::get_value(dict, "hi",  0.0);
    double lo  = cppdict::get_value(dict, "lo",  0.0);
    double ref = cppdict::get_value(dict, "ref", 0.0);
    if (type == "gradient")   return make_gradient(hi, lo, ref);
    if (type == "relative_u") return make_relative_u(hi, lo, ref);
    return make_absolute(hi, lo, ref);  // "absolute" or unknown
  }
};

}  // namespace equilibrium::dst::discrete

#endif  // _EQUILIBRIUM_CORE_DATA_LATTICE_SPLIT_CRITERION_HPP_
