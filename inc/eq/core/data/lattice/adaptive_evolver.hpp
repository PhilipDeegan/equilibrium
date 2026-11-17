#ifndef _EQUILIBRIUM_CORE_DATA_LATTICE_ADAPTIVE_EVOLVER_HPP_
#define _EQUILIBRIUM_CORE_DATA_LATTICE_ADAPTIVE_EVOLVER_HPP_

#include "eq/core/data/lattice/adaptive_amr.hpp"
#include "eq/core/data/lattice/adaptive_lattice.hpp"
#include "eq/core/data/lattice/time_stepper.hpp"
#include "eq/core/phlop.hpp"

#include <cstddef>
#include <memory>

namespace equilibrium::dst::discrete {

// ---------------------------------------------------------------------------
// AdaptiveLatticeEvolver
//
// Wraps AdaptiveLattice with scope-timed methods so that phlop timing output
// has a single C++ root scope enclosing all step/adapt/rebuild work.
// Calling step/adapt/rebuild from Python individually gives disconnected timer
// roots; going through this class nests everything under one parent scope.
// ---------------------------------------------------------------------------

class AdaptiveLatticeEvolver {
 public:
  explicit AdaptiveLatticeEvolver(AdaptiveLattice& lat) : lat_{lat} {}

  void step(TimeStepper const& ts, std::size_t n_steps = 1) {
    EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLatticeEvolver::step");
    if (ts.subcycle) {
      for (std::size_t i = 0; i < n_steps; ++i) lat_.step_all_subcycle(ts.dt_base, ts.ratio);
    } else {
      for (std::size_t i = 0; i < n_steps; ++i) lat_.step_all(ts);
    }
  }

  void step(double dt, std::size_t n_steps = 1) {
    step(TimeStepper::make_fixed(dt), n_steps);
  }

  void amr_adapt() {
    EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLatticeEvolver::amr_adapt");
    ::equilibrium::dst::discrete::adapt_all(lat_);
  }

  void rebuild() {
    EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLatticeEvolver::rebuild");
    lat_.rebuild_all();
  }

 private:
  AdaptiveLattice& lat_;
};

}  // namespace equilibrium::dst::discrete

#endif  // _EQUILIBRIUM_CORE_DATA_LATTICE_ADAPTIVE_EVOLVER_HPP_
