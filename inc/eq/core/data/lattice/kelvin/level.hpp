#ifndef _EQUILIBRIUM_CORE_DATA_LATTICE_KELVIN_LEVEL_HPP_
#define _EQUILIBRIUM_CORE_DATA_LATTICE_KELVIN_LEVEL_HPP_

// A Kelvin lattice's level 0 for any model: the domain's sites (damn::amr::kelvin::KeyBox) cut in
// patches of a damn Kelvin hierarchy over an ExecutionModel's ranks, a PatchData of the model's per
// local patch, ghosts moved by damn, patches stepped side by side on leased threads. The model
// brings PatchData (its for_each_resource: damn::amr::kelvin::SiteField, or its own BoxData) and
// what a step does; this knows nothing of either.
//
//   make_level<PatchData>(keys, patch_cells, exec)   hierarchy, neighbours, pool; no data yet
//   allocate(level, make)       make(patch, i) for this rank's patches, i their order
//   exchange(level, names)      fill_ghosts of the named resources, over the ranks with exec.mpi()
//   for_each_patch(level, fn)   fn(i, data) for every local patch, side by side
//
// with exec.mpi() the patches are over MPI_COMM_WORLD's ranks (mpi.h at compile time)

#include "damn/amr/copy.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/kelvin/sites.hpp"
#include "damn/amr/lattice.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/amr/neighbours.hpp"
#include "damn/amr/topology.hpp"
#include "damn/core/exec/execution_model.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/core/exec/leased_pool.hpp"  // IWYU pragma: keep (made and run in the templates)

// IWYU pragma: no_include "mpi.h"
#if __has_include("mpi.h")
#define EQ_KELVIN_LEVEL_MPI 1
#include "damn/mpi/copy.hpp"
#include "damn/mpi/execution_model.hpp"
#include "damn/mpi/mpi_utils.hpp"
#include "damn/mpi/topology.hpp"
#else
#define EQ_KELVIN_LEVEL_MPI 0
#endif

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <utility>
#include <stdexcept>  // IWYU pragma: keep (builds without MPI)

namespace equilibrium::kelvin {

using Hierarchy = damn::amr::Hierarchy<3, damn::amr::Kelvin>;
using KeyBox = damn::amr::kelvin::KeyBox<3>;
using Key = damn::amr::kelvin::Key<3>;
using Site = damn::amr::kelvin::Site;
using Box3 = damn::core::Box<int, 3>;
using Point3 = damn::core::Point<int, 3>;

inline constexpr std::array<Site, 2> sites_of_cell{Site::corner, Site::body};

// exec outlives the level
template <typename PatchData>
struct Level {
  Hierarchy h{};
  KeyBox keys{};
  int rank = 0;
  damn::core::ExecutionModel* exec = nullptr;
  std::vector<damn::amr::Patch<3>> local{};  // this rank's patches, level order
  damn::amr::LevelData<3, PatchData> data{};
  damn::amr::Neighbours<3> neighbours{};
  std::unique_ptr<damn::core::LeasedPool> pool{};
};

// keys' cells cut in patches of at most patch_cells cells (2 h each) over exec's ranks
template <typename PatchData>
[[nodiscard]] Level<PatchData> make_level(KeyBox const& keys, std::array<int, 3> const& patch_cells,
                                          damn::core::ExecutionModel& exec, double d1 = 1.0);

template <typename PatchData>
void allocate(Level<PatchData>& level, auto&& make);

template <typename PatchData>
void exchange(Level<PatchData>& level, std::vector<std::string> const& names);

template <typename PatchData>
void for_each_patch(Level<PatchData>& level, auto&& fn);

template <typename PatchData>
Level<PatchData> make_level(KeyBox const& keys, std::array<int, 3> const& patch_cells,
                            damn::core::ExecutionModel& exec, double const d1) {
  keys.check();
  exec.check();
#if EQ_KELVIN_LEVEL_MPI
  damn::mpi::check(exec);
  auto const rank = exec.mpi() ? damn::mpi::rank() : 0;
  auto const topology = exec.mpi() ? damn::mpi::topology() : damn::amr::single_node();
#else
  if (exec.mpi())
    throw std::runtime_error("kelvin::make_level: mpi_ranks > 0 in a build without MPI");
  auto const rank = 0;
  auto const topology = damn::amr::single_node();
#endif

  double const h = d1 / std::sqrt(3.0);
  Level<PatchData> level;
  level.keys = keys;
  level.rank = rank;
  level.exec = &exec;
  level.h.domain = keys.cells();
  level.h.dx = {2 * h, 2 * h, 2 * h};
  level.h.origin = damn::core::Point<double, 3>{std::array{2 * level.h.domain.lower[0] * h,
                                                           2 * level.h.domain.lower[1] * h,
                                                           2 * level.h.domain.lower[2] * h}};
  level.h.periodic = keys.periodic;
  level.h.max_levels = 1;
  level.h.ratio = 3;
  damn::amr::set_level(level.h, damn::amr::uniform_level_0(
                                    level.h, patch_cells, static_cast<int>(topology.ranks.size())));
  level.local = damn::amr::local_patches(level.h.levels[0], rank);
  level.neighbours = damn::amr::make_neighbours(level.h, 0, rank, topology, 1);
  level.pool = std::make_unique<damn::core::LeasedPool>(exec);
  return level;
}

template <typename PatchData>
void allocate(Level<PatchData>& level, auto&& make) {
  std::unordered_map<std::uint32_t, std::size_t> index_of;
  for (std::size_t i = 0; i < level.local.size(); ++i) index_of[level.local[i].id] = i;
  // a rank without patches: damn's prototype, boxes only (i = local.size())
  level.data =
      damn::amr::allocate(level.h.levels[0], level.rank, [&](damn::amr::Patch<3> const& patch) {
        auto const it = index_of.find(patch.id);
        return make(patch, it == index_of.end() ? level.local.size() : it->second);
      });
}

template <typename PatchData>
void exchange(Level<PatchData>& level, std::vector<std::string> const& names) {
  if (level.exec->mpi()) {
#if EQ_KELVIN_LEVEL_MPI
    damn::mpi::copy(damn::amr::fill_ghosts, level.neighbours, names, level.data);
#else
    throw std::runtime_error("kelvin::exchange: mpi_ranks > 0 in a build without MPI");
#endif
  } else
    damn::amr::copy(damn::amr::fill_ghosts, level.neighbours, names, level.data);
}

template <typename PatchData>
void for_each_patch(Level<PatchData>& level, auto&& fn) {
  level.pool->run(level.local.size(), [&](std::size_t const i) { fn(i, level.data.data[i]); });
}

}  // namespace equilibrium::kelvin

#endif  // _EQUILIBRIUM_CORE_DATA_LATTICE_KELVIN_LEVEL_HPP_
