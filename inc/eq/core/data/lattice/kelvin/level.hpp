#ifndef EQUILIBRIUM_CORE_DATA_LATTICE_KELVIN_LEVEL_HPP
#define EQUILIBRIUM_CORE_DATA_LATTICE_KELVIN_LEVEL_HPP

// A Kelvin lattice's levels for any model: level 0 the domain's sites (damn::amr::kelvin::KeyBox)
// cut in patches of a damn Kelvin hierarchy over an ExecutionModel's ranks, each finer level boxes
// of whole coarser cells at ratio 3 (damn's clustering of tagged cells, damn's "finest"
// decomposition: a fine patch to the owners of the coarse cells under it), a PatchData of the
// model's per local patch, ghosts moved by damn, between levels along damn's level maps, patches
// stepped side by side on leased threads. The model brings PatchData (its for_each_resource:
// damn::amr::kelvin::SiteField, or its own BoxData), what a step does, and the operators refining
// and coarsening its values (damn::amr::kelvin::LinearRefine / InjectCoarsen, or its own); this
// knows nothing of them.
//
//   make_levels<PatchData>(keys, exec, shape)
//                                  level 0: hierarchy, neighbours, pool; no data yet
//   tagged_boxes(l, ln, tagged)    level ln + 1's boxes from its tagged cells (collective)
//   set_finer_level(l, ln, boxes, patch_cells)
//                                  level ln + 1 from boxes (its index space), finer ones dropped;
//                                  no data yet
//   allocate(l, ln, make)          make(patch, i) for this rank's patches of ln, i their order
//   exchange(l, ln, names)         fill_ghosts of the named resources within level ln
//   refine(l, ln, op, region, pairs[, alpha])
//                                  level ln's Interior (a new level) or LevelGhosts (its border)
//                                  from level ln - 1, (1 - alpha) coarse + alpha coarse_new
//   coarsen(l, ln, op, pairs)      level ln - 1 under level ln from it
//   for_each_patch(l, ln, fn)      fn(i, data) for every local patch of ln, side by side
//
// with exec.mpi() the patches are over MPI_COMM_WORLD's ranks (mpi.h at compile time) and every
// function but for_each_patch and allocate is collective. Stepping the levels in time is the
// model's, e.g. with damn::amr::TimeRefinement over l.h

#include "damn/amr/clustering.hpp"
#include "damn/amr/copy.hpp"
#include "damn/amr/decompose.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/inter_level.hpp"
#include "damn/amr/kelvin/inter_level.hpp"
#include "damn/amr/kelvin/sites.hpp"
#include "damn/amr/lattice.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/amr/level_map.hpp"
#include "damn/amr/neighbours.hpp"
#include "damn/amr/partition.hpp"
#include "damn/amr/topology.hpp"
#include "damn/core/exec/execution_model.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/core/exec/leased_pool.hpp"  // IWYU pragma: keep (made and run in the templates)

// IWYU pragma: no_include "mpi.h"
#if __has_include("mpi.h")
#define EQ_KELVIN_LEVEL_MPI 1
#include "damn/mpi/clustering.hpp"
#include "damn/mpi/copy.hpp"
#include "damn/mpi/execution_model.hpp"
#include "damn/mpi/inter_level.hpp"
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
#include <stdexcept>
#include <utility>

namespace equilibrium::kelvin {

using Hierarchy = damn::amr::Hierarchy<3, damn::amr::Kelvin>;
using KeyBox = damn::amr::kelvin::KeyBox<3>;
using Key = damn::amr::kelvin::Key<3>;
using Site = damn::amr::kelvin::Site;
using Box3 = damn::core::Box<int, 3>;
using Point3 = damn::core::Point<int, 3>;

inline constexpr std::array<Site, 2> sites_of_cell{Site::corner, Site::body};
inline constexpr int ratio = 3;

// one level, this rank's part: keys its sites in its own index space (h / 3^number units)
template <typename PatchData>
struct Level {
  int number = 0;
  KeyBox keys{};
  std::vector<damn::amr::Patch<3>> local{};  // this rank's patches, level order
  damn::amr::LevelData<3, PatchData> data{};
  damn::amr::Neighbours<3> neighbours{};
};

// exec outlives the levels
template <typename PatchData>
struct Levels {
  Hierarchy h{};
  int rank = 0;
  damn::amr::Topology topology{};
  damn::core::ExecutionModel* exec = nullptr;
  std::vector<Level<PatchData>> levels{};
  std::vector<damn::amr::LevelMap<3>> maps{};  // maps[ln]: levels ln and ln + 1
  std::unique_ptr<damn::core::LeasedPool> pool{};

  [[nodiscard]] Level<PatchData>& operator[](int const ln) {
    return levels.at(static_cast<std::size_t>(ln));
  }
  [[nodiscard]] Level<PatchData> const& operator[](int const ln) const {
    return levels.at(static_cast<std::size_t>(ln));
  }
};

// level 0 cut in patches of at most patch_cells cells (2 h each), hex face distance d1 (h = d1 /
//  sqrt 3), at most max_levels levels
struct Shape {
  std::array<int, 3> patch_cells{};
  double d1 = 1.0;
  int max_levels = 1;
};

// keys' cells cut as shape says over exec's ranks
template <typename PatchData>
[[nodiscard]] Levels<PatchData> make_levels(KeyBox const& keys, damn::core::ExecutionModel& exec,
                                            Shape const& shape);

template <typename PatchData>
[[nodiscard]] std::vector<Box3> tagged_boxes(Levels<PatchData> const& l, int ln,
                                             std::vector<Point3> const& tagged,
                                             damn::amr::ClusteringOptions<3> const& options = {});

template <typename PatchData>
void set_finer_level(Levels<PatchData>& l, int ln, std::vector<Box3> const& boxes,
                     std::array<int, 3> const& patch_cells);

template <typename PatchData>
void allocate(Levels<PatchData>& l, int ln, auto&& make);

template <typename PatchData>
void exchange(Levels<PatchData>& l, int ln, std::vector<std::string> const& names);

template <typename PatchData>
void refine(Levels<PatchData>& l, int ln, auto const& op, damn::amr::LevelRegion region,
            std::vector<damn::amr::LevelPair> const& pairs, double alpha = 0);

template <typename PatchData>
void coarsen(Levels<PatchData>& l, int ln, auto const& op,
             std::vector<damn::amr::LevelPair> const& pairs);

template <typename PatchData>
void for_each_patch(Levels<PatchData>& l, int ln, auto&& fn);

namespace detail {
// how far past a fine patch its level map looks for coarse patches (fine cells): its ghosts and
//  a refine stencil of one coarse cell
inline constexpr int map_reach = 2 * ratio;

[[noreturn]] inline void no_mpi(std::string const& what) {
  throw std::runtime_error("kelvin::" + what + ": mpi_ranks > 0 in a build without MPI");
}

template <typename PatchData>
Level<PatchData> make_level(Levels<PatchData> const& l, int const ln, KeyBox const& keys) {
  Level<PatchData> level;
  level.number = ln;
  level.keys = keys;
  level.local = damn::amr::local_patches(l.h.levels[static_cast<std::size_t>(ln)], l.rank);
  level.neighbours = damn::amr::make_neighbours(l.h, ln, l.rank, l.topology, 1);
  return level;
}
}  // namespace detail

template <typename PatchData>
Levels<PatchData> make_levels(KeyBox const& keys, damn::core::ExecutionModel& exec,
                              Shape const& shape) {
  auto const [patch_cells, d1, max_levels] = shape;
  keys.check();
  exec.check();
  if (max_levels < 1) throw std::invalid_argument("kelvin::make_levels: max_levels < 1");
  Levels<PatchData> l;
#if EQ_KELVIN_LEVEL_MPI
  damn::mpi::check(exec);
  l.rank = exec.mpi() ? damn::mpi::rank() : 0;
  l.topology = exec.mpi() ? damn::mpi::topology() : damn::amr::single_node();
#else
  if (exec.mpi()) detail::no_mpi("make_levels");
  l.topology = damn::amr::single_node();
#endif

  double const h = d1 / std::sqrt(3.0);
  l.exec = &exec;
  l.h.domain = keys.cells();
  l.h.dx = {2 * h, 2 * h, 2 * h};
  l.h.origin = damn::core::Point<double, 3>{std::array{
      2 * l.h.domain.lower[0] * h, 2 * l.h.domain.lower[1] * h, 2 * l.h.domain.lower[2] * h}};
  l.h.periodic = keys.periodic;
  l.h.max_levels = max_levels;
  l.h.ratio = ratio;
  damn::amr::set_level(
      l.h, damn::amr::uniform_level_0(l.h, patch_cells, static_cast<int>(l.topology.ranks.size())));
  l.levels.push_back(detail::make_level(l, 0, keys));
  l.pool = std::make_unique<damn::core::LeasedPool>(exec);
  return l;
}

template <typename PatchData>
std::vector<Box3> tagged_boxes(Levels<PatchData> const& l, int const ln,
                               std::vector<Point3> const& tagged,
                               damn::amr::ClusteringOptions<3> const& options) {
  if (l.exec->mpi()) {
#if EQ_KELVIN_LEVEL_MPI
    return damn::mpi::cluster(l.h, ln, tagged, options);
#else
    detail::no_mpi("tagged_boxes");
#endif
  }
  return damn::amr::cluster(l.h, ln, damn::amr::tagged_tiles(l.h, ln, tagged, options), options);
}

template <typename PatchData>
void set_finer_level(Levels<PatchData>& l, int const ln, std::vector<Box3> const& boxes,
                     std::array<int, 3> const& patch_cells) {
  auto const fine = ln + 1;
  if (ln < 0 or fine >= l.h.max_levels or static_cast<std::size_t>(ln) >= l.levels.size())
    throw std::invalid_argument("kelvin::set_finer_level: no level " + std::to_string(fine));
  damn::amr::Decompose<3> const in{
      .topology = l.topology,
      .weigh = [](int, std::vector<Box3> const& b) { return damn::amr::cell_weights(b); },
      .patch_size = patch_cells,
      .partition = "finest"};
  damn::amr::set_level(l.h, damn::amr::decompose_level(l.h, fine, boxes, in));
  auto const keys = damn::amr::kelvin::refined_keys(l[ln].keys, ratio);
  l.levels.resize(static_cast<std::size_t>(fine));
  l.maps.resize(static_cast<std::size_t>(ln));
  l.levels.push_back(detail::make_level(l, fine, keys));
  l.maps.push_back(damn::amr::make_level_map(l.h, ln, l.rank, l.topology, detail::map_reach));
}

template <typename PatchData>
void allocate(Levels<PatchData>& l, int const ln, auto&& make) {
  auto& level = l[ln];
  std::unordered_map<std::uint32_t, std::size_t> index_of;
  for (std::size_t i = 0; i < level.local.size(); ++i) index_of[level.local[i].id] = i;
  // a rank without patches: damn's prototype, boxes only (i = local.size())
  level.data = damn::amr::allocate(
      l.h.levels[static_cast<std::size_t>(ln)], l.rank, [&](damn::amr::Patch<3> const& patch) {
        auto const it = index_of.find(patch.id);
        return make(patch, it == index_of.end() ? level.local.size() : it->second);
      });
}

template <typename PatchData>
void exchange(Levels<PatchData>& l, int const ln, std::vector<std::string> const& names) {
  auto& level = l[ln];
  if (l.exec->mpi()) {
#if EQ_KELVIN_LEVEL_MPI
    damn::mpi::copy(damn::amr::fill_ghosts, level.neighbours, names, level.data);
#else
    detail::no_mpi("exchange");
#endif
  } else
    damn::amr::copy(damn::amr::fill_ghosts, level.neighbours, names, level.data);
}

template <typename PatchData>
void refine(Levels<PatchData>& l, int const ln, auto const& op, damn::amr::LevelRegion const region,
            std::vector<damn::amr::LevelPair> const& pairs, double const alpha) {
  auto& map = l.maps.at(static_cast<std::size_t>(ln - 1));
  auto& coarse = l[ln - 1].data;
  auto& fine = l[ln].data;
  if (l.exec->mpi()) {
#if EQ_KELVIN_LEVEL_MPI
    damn::mpi::refine(op, region, map, pairs, coarse, fine, alpha);
#else
    detail::no_mpi("refine");
#endif
  } else
    damn::amr::refine(op, region, map, pairs, coarse, fine, alpha);
}

template <typename PatchData>
void coarsen(Levels<PatchData>& l, int const ln, auto const& op,
             std::vector<damn::amr::LevelPair> const& pairs) {
  auto& map = l.maps.at(static_cast<std::size_t>(ln - 1));
  auto& coarse = l[ln - 1].data;
  auto& fine = l[ln].data;
  if (l.exec->mpi()) {
#if EQ_KELVIN_LEVEL_MPI
    damn::mpi::coarsen(op, map, pairs, fine, coarse);
#else
    detail::no_mpi("coarsen");
#endif
  } else
    damn::amr::coarsen(op, map, pairs, fine, coarse);
}

template <typename PatchData>
void for_each_patch(Levels<PatchData>& l, int const ln, auto&& fn) {
  auto& level = l[ln];
  l.pool->run(level.local.size(), [&](std::size_t const i) { fn(i, level.data.data[i]); });
}

}  // namespace equilibrium::kelvin

#endif  // EQUILIBRIUM_CORE_DATA_LATTICE_KELVIN_LEVEL_HPP
