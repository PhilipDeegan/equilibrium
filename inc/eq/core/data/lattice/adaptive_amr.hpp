#ifndef _EQUILIBRIUM_CORE_DATA_LATTICE_ADAPTIVE_AMR_HPP_
#define _EQUILIBRIUM_CORE_DATA_LATTICE_ADAPTIVE_AMR_HPP_

// AMR refinement pass: combines adaptive_lattice (physics) and bcc_geometry
// (wiring) — kept in a separate header to avoid a circular include between the
// two layers.

#include "eq/core/data/lattice/bcc_geometry.hpp"
#include "eq/core/data/lattice/adaptive_lattice.hpp"
#include "eq/core/phlop.hpp"
#include "eq/core/data/lattice/cell/adaptive_cell.hpp"
#include "eq/core/data/lattice/cluster.hpp"
#include "eq/core/data/lattice/dst_info.hpp"
#include "eq/core/data/lattice/split_criterion.hpp"

#include <algorithm>
#include <cmath>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace equilibrium::dst::discrete {

// ---------------------------------------------------------------------------
// adapt()
//
// One AMR refinement pass.  Decision source (in priority order):
//   1. cl.split_criterion (functor) — if set, all split/coarsen decisions
//      are delegated to split_criterion->fn(state, hi, lo).
//   2. cl.info.split_hi / split_lo — legacy scalar thresholds (fallback).
//
// Both decisions are taken from a snapshot of (leaf_indices, criterion_max)
// captured at entry — so splits and coarsens in the same call do not
// influence each other.  Coarsen is attempted first.  A single
// rebuild_leaf_indices is issued at the end if anything changed.
//
// Cells with NaN criterion are always kept (unstepped children after a split).
// ---------------------------------------------------------------------------

// Build a CellState from cluster + leaf index.
static inline SplitCriterion::CellState _make_cell_state(AdaptiveCellCluster& cl,
                                                          std::size_t j,
                                                          double ref = 0.0) {
  auto const g    = cl.leaf_indices[j];
  auto const cell = cl[g];
  return {cl.criterion_max[j],
          cl.criterion_min[j],
          cell.u,
          cell.L,
          cell.lam,
          ref,
          cell.level,
          cl.info.max_level};
}

inline void adapt(AdaptiveLattice& lat, AdaptiveCellCluster& cl) {
  EQ_SCOPE_TIMER("equilibrium::dst::discrete::adapt");

  // Cadence gate — callers may invoke adapt() every step; only every
  // split_interval-th call actually evaluates split/coarsen.
  auto const call_idx = cl.adapt_calls++;
  if (cl.info.split_interval > 1 && call_idx % cl.info.split_interval != 0) return;

  auto const sc          = cl.split_criterion.get();
  auto const use_functor = sc && sc->fn;

  // Fast-exit when nothing is configured.
  if (!use_functor && cl.info.split_hi <= 0.0 && cl.info.split_lo <= 0.0) return;

  bool changed = false;

  // --- Coarsen pass ---
  // Build snapshot + reverse map only when coarsening is possible.
  auto const coarsen_enabled = use_functor ? (sc->lo > 0.0) : (cl.info.split_lo > 0.0);
  if (coarsen_enabled && !cl.leaf_indices.empty()) {
    static thread_local std::vector<std::int32_t> leaves_snap;
    static thread_local std::vector<double>       cmax_snap;
    static thread_local std::vector<std::int32_t> parents;

    leaves_snap.assign(cl.leaf_indices.begin(), cl.leaf_indices.end());
    cmax_snap.assign(cl.criterion_max.begin(), cl.criterion_max.end());

    // Dense global-id -> local-leaf-index lookup: the cluster's cached map
    // (rebuilt by rebuild_leaf_indices) is still the pre-pass view here —
    // coarsen() mutates cells but never leaf_indices, so no snapshot copy is
    // needed. -1 = not a current leaf.
    auto const g_max   = cl.g_max;
    auto const& g_to_j = cl.g_to_j;

    parents.clear();
    parents.reserve(leaves_snap.size());
    for (std::int32_t g : leaves_snap) {
      auto const p = cl[g].parent;
      if (p >= 0) parents.push_back(p);
    }
    std::sort(parents.begin(), parents.end());
    parents.erase(std::unique(parents.begin(), parents.end()), parents.end());

    for (std::int32_t p : parents) {
      auto const par = cl[p];
      if (par.n_children == 0) continue;
      bool all_quiet = true;
      for (std::uint8_t c = 0; c < par.n_children; ++c) {
        auto const cg = par.first_child + c;
        auto const jn = cg <= g_max ? g_to_j[static_cast<std::size_t>(cg)] : std::int32_t{-1};
        if (jn < 0) { all_quiet = false; break; }
        auto const j = static_cast<std::size_t>(jn);
        bool quiet;
        if (use_functor) {
          quiet = (sc->fn(_make_cell_state(cl, j, sc->ref), sc->hi, sc->lo) == -1);
        } else {
          auto const cmax = cmax_snap[j];
          quiet = !std::isnan(cmax) && cmax < cl.info.split_lo;
        }
        if (!quiet) { all_quiet = false; break; }
      }
      if (all_quiet) { lat.coarsen(cl, p); changed = true; }
    }
  }

  // --- Split pass ---
  // leaf_indices and criterion_max are not modified until rebuild_leaf_indices
  // at the end, so no snapshot copy is needed.
  auto const split_enabled = use_functor ? (sc->hi > 0.0) : (cl.info.split_hi > 0.0);
  if (split_enabled) {
    auto const n = cl.leaf_indices.size();
    for (std::size_t j = 0; j < n; ++j) {
      bool want_split;
      if (use_functor) {
        want_split = (sc->fn(_make_cell_state(cl, j, sc->ref), sc->hi, sc->lo) == 1);
      } else {
        auto const cmax = cl.criterion_max[j];
        want_split = !std::isnan(cmax) && cmax > cl.info.split_hi;
      }
      if (!want_split) continue;
      auto const g = cl.leaf_indices[j];
      if (!lat.can_split(cl, g)) continue;
      auto result = lat.split(cl, g);
      bcc::wire_split_children(cl, g, result, cl[g].L);
      changed = true;
    }
  }

  if (changed) lat.rebuild_leaf_indices(cl);
}

inline void adapt_all(AdaptiveLattice& lat) {
  for (auto& cl : lat.clusters) adapt(lat, cl);
}

}  // namespace equilibrium::dst::discrete

#endif  // _EQUILIBRIUM_CORE_DATA_LATTICE_ADAPTIVE_AMR_HPP_
