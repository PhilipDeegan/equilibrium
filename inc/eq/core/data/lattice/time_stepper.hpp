#ifndef _EQUILIBRIUM_CORE_DATA_LATTICE_TIME_STEPPER_HPP_
#define _EQUILIBRIUM_CORE_DATA_LATTICE_TIME_STEPPER_HPP_

// TimeStepper — pluggable per-cell timestep rule for AdaptiveLatticeEvolver.
//
// Mirrors the SplitCriterion pattern:
//   fn(state, dt_base, ratio) → dt to use for this cell
//
// Built-in presets:
//   fixed     — constant dt_base for all cells (default; equivalent to scalar step)
//   by_level  — dt_base / ratio^level  (CFL-safe: each split shrinks L by `ratio`,
//               so dt_max shrinks by the same factor — ratio defaults to 3.0,
//               matching kDefaultSplitModel = SplitModel::Odd27; pass ratio=2.0
//               for a lattice built with the legacy Hex8/HexCenter9 splits)
//   cfl       — dt_base * L        (dt_base is the dt for L=1 cells; proportional to L;
//               ratio-agnostic, safe with any split model)
//
// For faces at coarse-fine interfaces, pass 2 uses min(dt_i, dt_nb) so both
// sides of the face advance at the same rate and phi antisymmetry is preserved.
//
// All presets are configurable via FROM(dict) with keys:
//   "type"  — preset name (default "fixed")
//   "dt"    — dt_base value (default 0.003)
//   "ratio" — refinement ratio for by_level (default 3.0)

#include "dict.hpp"
#include "eq/core/data/lattice/dst_info.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

namespace equilibrium::dst::discrete {

struct TimeStepper {
  struct CellState {
    double L;
    std::uint8_t level;
  };

  double (*fn)(CellState const& s, double dt_base, double ratio) = nullptr;
  double dt_base = 0.003;
  // Refinement ratio used by by_level_fn (dt_base / ratio^level) — must match
  // the ratio the lattice was actually split with (split_model_traits<Model>::
  // ratio). Unused by fixed_fn/cfl_fn. Default 3.0 matches kDefaultSplitModel
  // = SplitModel::Odd27.
  double ratio = 3.0;
  // When true, AdaptiveLatticeEvolver::step() uses subcycling: coarse cells
  // take 1 step of dt_base, level-l cells take ratio^l steps of
  // dt_base/ratio^l, so all cells end at the same physical time. Required
  // for by_level (see AdaptiveLattice::step_subcycle, which takes the SAME
  // ratio to keep substep count and per-substep dt consistent).
  bool subcycle = false;

  double operator()(double L, std::uint8_t level) const noexcept {
    if (!fn) return dt_base;
    return fn({L, level}, dt_base, ratio);
  }

  // ---------------------------------------------------------------------------
  // Predefined implementations
  // ---------------------------------------------------------------------------

  static double fixed_fn(CellState const& s, double dt_base, double ratio) noexcept {
    (void)s;
    (void)ratio;
    return dt_base;
  }

  // CFL-correct per-level: dt shrinks by `ratio` each time L shrinks by `ratio`.
  static double by_level_fn(CellState const& s, double dt_base, double ratio) noexcept {
    return dt_base / std::pow(ratio, s.level);
  }

  // CFL proportional to L: dt_base is the target dt for L=1 cells. Ratio-agnostic.
  static double cfl_fn(CellState const& s, double dt_base, double ratio) noexcept {
    (void)ratio;
    return dt_base * s.L;
  }

  // ---------------------------------------------------------------------------
  // Factory helpers
  // ---------------------------------------------------------------------------

  static TimeStepper make_fixed(double dt) { return {fixed_fn, dt, 3.0, false}; }
  static TimeStepper make_by_level(double dt_base, double ratio = 3.0) {
    return {by_level_fn, dt_base, ratio, true};
  }
  static TimeStepper make_cfl(double dt_l1) { return {cfl_fn, dt_l1, 3.0, false}; }

  // Minimum dt between two cells — used for face updates to preserve antisymmetry.
  double face_dt(double L_i, std::uint8_t level_i,
                 double L_nb, std::uint8_t level_nb) const noexcept {
    return std::min((*this)(L_i, level_i), (*this)(L_nb, level_nb));
  }

  // ---------------------------------------------------------------------------
  // Dict-based construction
  // ---------------------------------------------------------------------------

  static TimeStepper FROM(DSTDict const& dict) {
    auto type = cppdict::get_value(dict, "type", std::string{"fixed"});
    double dt    = cppdict::get_value(dict, "dt",    0.003);
    double ratio = cppdict::get_value(dict, "ratio", 3.0);
    if (type == "by_level") return make_by_level(dt, ratio);
    if (type == "cfl")      return make_cfl(dt);
    return make_fixed(dt);
  }
};

}  // namespace equilibrium::dst::discrete

#endif  // _EQUILIBRIUM_CORE_DATA_LATTICE_TIME_STEPPER_HPP_
