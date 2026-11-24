#ifndef _EQUILIBRIUM_CORE_DATA_LATTICE_CONTINUOUS_DAMN_CLUSTERS_HPP_
#define _EQUILIBRIUM_CORE_DATA_LATTICE_CONTINUOUS_DAMN_CLUSTERS_HPP_

// Level 0 as several clusters, one per patch of a damn Kelvin hierarchy. A cluster holds its
// patch's sites and, as ghosts, the sites its faces reach past the patch (one cell): leaves with u
// and phi fixed, filled from their owners by damn between step_u and step_phi.
//
// eq site (ix, iy, iz), h units: damn cell floor(ix / 2) per axis, its corner site if even, its
// body site if odd. Faces never cross more than one cell, so ghosts are the patch box grown by 1.
// The sites are the keys in a box (Keys): a periodic direction's are whole cells, a bounded one's
// need not be (build_bcc_box's [-N, N]: one sublattice a layer more than the other), the domain's
// cells are those holding a site and a cell's missing site is absent (no cell, faces to it inert).
//
// An ExecutionModel says where: with mpi() the patches are over MPI_COMM_WORLD's ranks and
// damn::mpi moves the ghosts (needs mpi.h at compile time), else one process. step_all steps the
// clusters side by side on the threads it leases from it (use_threads), the main thread only if
// none are free.

#include "eq/core/data/lattice/continuous/adaptive_lattice.hpp"
#include "eq/core/data/lattice/bcc_defs.hpp"

#include "damn/amr/copy.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/lattice.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/amr/neighbours.hpp"
#include "damn/amr/topology.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/core/exec/execution_model.hpp"

#if __has_include("mpi.h")
#define EQ_DAMN_CLUSTERS_MPI 1
#include "damn/mpi/copy.hpp"
#include "damn/mpi/execution_model.hpp"
#include "damn/mpi/mpi_utils.hpp"
#include "damn/mpi/topology.hpp"
#else
#define EQ_DAMN_CLUSTERS_MPI 0
#endif

#include "BS_thread_pool.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace equilibrium::dst::continuous::clusters {

using Box3 = damn::core::Box<int, 3>;
using Point3 = damn::core::Point<int, 3>;
using Site = damn::amr::kelvin::Site;
using Hierarchy = damn::amr::Hierarchy<3, damn::amr::Kelvin>;
using Key = discrete::bcc::IntVec3;

inline constexpr std::array<Site, 2> sites_of_cell{Site::corner, Site::body};

[[nodiscard]] inline Key key_of(Point3 const& cell, Site const site) {
  auto const b = site == Site::body ? 1 : 0;
  return {2 * cell[0] + b, 2 * cell[1] + b, 2 * cell[2] + b};
}

[[nodiscard]] inline Point3 cell_of(Key const& key) {
  return Point3{std::array{damn::core::floor_div(key.x, 2), damn::core::floor_div(key.y, 2),
                           damn::core::floor_div(key.z, 2)}};
}

[[nodiscard]] inline Site site_of(Key const& key) {
  return (key.x & 1) ? Site::body : Site::corner;
}

// the domain's sites: keys lower..upper per direction (h units), periodic or bounded
struct Keys {
  std::array<int, 3> lower{}, upper{};
  std::array<bool, 3> periodic{};

  // whether key is a site: past a bounded direction's ends it is not, a periodic one's are images
  [[nodiscard]] bool has(Key const& key) const;
  [[nodiscard]] Box3 cells() const;
};

inline bool Keys::has(Key const& key) const {
  std::array const k{key.x, key.y, key.z};
  for (std::size_t d = 0; d < 3; ++d)
    if (!periodic[d] && (k[d] < lower[d] || k[d] > upper[d])) return false;
  return true;
}

inline Box3 Keys::cells() const {
  return Box3{std::array{damn::core::floor_div(lower[0], 2), damn::core::floor_div(lower[1], 2),
                         damn::core::floor_div(lower[2], 2)},
              std::array{damn::core::floor_div(upper[0], 2), damn::core::floor_div(upper[1], 2),
                         damn::core::floor_div(upper[2], 2)}};
}

// a cluster's cells (global indices) per site over box (its patch grown by 1), -1: none (past a
// bounded domain's end, or an absent site)
struct Sites {
  Box3 box{};
  std::array<std::vector<std::int32_t>, 2> g{};
  mutable double absent = 0;  // what damn reads and writes for an absent site

  [[nodiscard]] std::size_t flat(Point3 const& c) const;
  [[nodiscard]] std::int32_t& at(Site site, Point3 const& c);
  [[nodiscard]] std::int32_t at(Site site, Point3 const& c) const;
};

inline std::size_t Sites::flat(Point3 const& c) const {
  auto const s = box.shape();
  return static_cast<std::size_t>(((c[2] - box.lower[2]) * s[1] + (c[1] - box.lower[1])) * s[0] +
                                  (c[0] - box.lower[0]));
}
inline std::int32_t& Sites::at(Site const site, Point3 const& c) {
  return g[static_cast<std::size_t>(site)][flat(c)];
}
inline std::int32_t Sites::at(Site const site, Point3 const& c) const {
  return damn::core::isIn(c, box) ? g[static_cast<std::size_t>(site)][flat(c)] : -1;
}

enum class Field : std::uint8_t { u, lam };

// one field on one site of a cluster's cells, as damn BoxData over the cluster's patch
struct SiteField {
  discrete::AdaptiveCellCluster* cluster = nullptr;
  Sites const* sites = nullptr;
  Site site = Site::corner;
  Field field = Field::u;
};

[[nodiscard]] inline Box3 domain_box(SiteField const&, Box3 const& cells) { return cells; }
[[nodiscard]] inline Box3 ghost_box(SiteField const&, Box3 const& cells) {
  return damn::core::grow(cells, 1);
}
[[nodiscard]] inline double& value(SiteField const& f, Box3 const&, Point3 const& p) {
  auto const g = f.sites->at(f.site, p);
  if (g < 0) return f.sites->absent;
  auto cell = (*f.cluster)[g];
  return f.field == Field::u ? cell.u : cell.lam;
}

struct ClusterData {
  SiteField u_corner, u_body, lam_corner, lam_body;
};

inline void for_each_resource(ClusterData& d, auto&& fn) {
  fn("u_corner", d.u_corner);
  fn("u_body", d.u_body);
  fn("lam_corner", d.lam_corner);
  fn("lam_body", d.lam_body);
}

// lat's clusters, one per local patch in level order, their sites, damn's view of them; exec
// outlives it
struct Clusters {
  Hierarchy h{};
  Keys keys{};
  int rank = 0;
  damn::core::ExecutionModel* exec = nullptr;
  std::vector<Sites> sites{};
  damn::amr::LevelData<3, ClusterData> data{};
  damn::amr::Neighbours<3> neighbours{};
  std::unique_ptr<BS::thread_pool<>> pool{};  // the most threads a lease may give, idle between
};

inline std::vector<std::string> const u_names{"u_corner", "u_body"};
inline std::vector<std::string> const lam_names{"lam_corner", "lam_body"};

// ghosts' u (or lam) from their owners, any rank's with exec->mpi() (collective)
inline void exchange(Clusters& c, std::vector<std::string> const& names) {
  if (c.exec->mpi()) {
#if EQ_DAMN_CLUSTERS_MPI
    damn::mpi::copy(damn::amr::fill_ghosts, c.neighbours, names, c.data);
#else
    throw std::runtime_error("clusters::exchange: mpi_ranks > 0 in a build without MPI");
#endif
  } else
    damn::amr::copy(damn::amr::fill_ghosts, c.neighbours, names, c.data);
}

// keys' domain cut in patches of at most patch_cells cells (2 h each) over exec's ranks; lat's
// clusters replaced by this rank's patches, u0, lam0 everywhere, phi 0. lat's leaves include the
// ghosts: diagnostics take for_each_owned. A periodic direction's keys are whole cells
[[nodiscard]] inline Clusters build(AdaptiveLattice& lat, Keys const& keys,
                                    std::array<int, 3> const& patch_cells,
                                    damn::core::ExecutionModel& exec, double d1 = 1.0,
                                    double u0 = 0.0, double lam0 = 0.0);

// build_bcc_box_periodic(P)'s sites (P multiples of 4)
[[nodiscard]] inline Clusters build_periodic(AdaptiveLattice& lat, std::array<int, 3> const& P,
                                             std::array<int, 3> const& patch_cells,
                                             damn::core::ExecutionModel& exec,
                                             double const d1 = 1.0, double const u0 = 0.0,
                                             double const lam0 = 0.0) {
  for (auto const p : P)
    if (p < 4 || p % 4)
      throw std::invalid_argument(
          "clusters::build_periodic: periods are multiples of 4 (whole cells)");
  Keys const keys{{-P[0] / 2, -P[1] / 2, -P[2] / 2},
                  {P[0] / 2 - 1, P[1] / 2 - 1, P[2] / 2 - 1},
                  {true, true, true}};
  return build(lat, keys, patch_cells, exec, d1, u0, lam0);
}

// build_bcc_box's sites for sides 2 N h (h = d1 / sqrt 3): keys -N..N, bounded
[[nodiscard]] inline Clusters build_box(AdaptiveLattice& lat, std::array<int, 3> const& N,
                                        std::array<int, 3> const& patch_cells,
                                        damn::core::ExecutionModel& exec, double const d1 = 1.0,
                                        double const u0 = 0.0, double const lam0 = 0.0) {
  Keys const keys{{-N[0], -N[1], -N[2]}, {N[0], N[1], N[2]}, {false, false, false}};
  return build(lat, keys, patch_cells, exec, d1, u0, lam0);
}

// the cluster's own sites (not ghosts): fn(global index, eq key)
inline void for_each_owned(Clusters const& c, std::size_t const i, auto&& fn) {
  damn::core::for_each_point(c.data.patches[i].box, [&](auto const& cell) {
    for (auto const site : sites_of_cell)
      if (auto const g = c.sites[i].at(site, cell); g >= 0) fn(g, key_of(cell, site));
  });
}

// fn(cluster) for every cluster, side by side on the threads leased for it (one cluster per thread
//  at a time), on this thread if none are free or lat threads within a cluster itself
inline void for_each_cluster(AdaptiveLattice& lat, Clusters& c, auto&& fn) {
  auto const n = lat.clusters.size();
  auto lease = (c.pool && lat.n_threads() <= 1 && n > 1) ? c.exec->use_threads(n)
                                                         : damn::core::ThreadLease{};
  if (lease.size() == 0) {
    for (auto& cl : lat.clusters) fn(cl);
    return;
  }
  c.pool->detach_blocks<std::size_t>(
      0, n,
      [&](std::size_t const b, std::size_t const e) {
        for (auto i = b; i < e; ++i) fn(lat.clusters[i]);
      },
      lease.size());
  c.pool->wait();
}

// a step of every cluster, ghosts' u refreshed between its halves
inline void step_all(AdaptiveLattice& lat, Clusters& c, discrete::TimeStepper const& ts) {
  for_each_cluster(lat, c, [&](auto& cl) { lat.step_u(cl, ts); });
  exchange(c, u_names);
  for_each_cluster(lat, c, [&](auto& cl) { lat.step_phi(cl, ts); });
}

inline Clusters build(AdaptiveLattice& lat, Keys const& keys, std::array<int, 3> const& patch_cells,
                      damn::core::ExecutionModel& exec, double const d1, double const u0,
                      double const lam0) {
  using discrete::AdaptiveCellCluster;
  using discrete::CellSlab;
  for (std::size_t d = 0; d < 3; ++d) {
    if (keys.lower[d] > keys.upper[d]) throw std::invalid_argument("clusters::build: no keys");
    if (keys.periodic[d] && ((keys.lower[d] & 1) || !(keys.upper[d] & 1)))
      throw std::invalid_argument("clusters::build: a periodic direction's keys are whole cells");
  }

  exec.check();
#if EQ_DAMN_CLUSTERS_MPI
  damn::mpi::check(exec);
  auto const rank = exec.mpi() ? damn::mpi::rank() : 0;
  auto const topology = exec.mpi() ? damn::mpi::topology() : damn::amr::single_node();
#else
  if (exec.mpi()) throw std::runtime_error("clusters::build: mpi_ranks > 0 in a build without MPI");
  auto const rank = 0;
  auto const topology = damn::amr::single_node();
#endif

  double const h = d1 / std::sqrt(3.0);
  Clusters c;
  c.keys = keys;
  c.rank = rank;
  c.exec = &exec;
  if (auto const most = exec.schedulable(); most > 1)
    c.pool = std::make_unique<BS::thread_pool<>>(most - 1);
  c.h.domain = keys.cells();
  c.h.dx = {2 * h, 2 * h, 2 * h};
  c.h.origin = damn::core::Point<double, 3>{std::array{
      2 * c.h.domain.lower[0] * h, 2 * c.h.domain.lower[1] * h, 2 * c.h.domain.lower[2] * h}};
  c.h.periodic = keys.periodic;
  c.h.max_levels = 1;
  c.h.ratio = 3;
  damn::amr::set_level(
      c.h, damn::amr::uniform_level_0(c.h, patch_cells, static_cast<int>(topology.ranks.size())));

  auto const local = damn::amr::local_patches(c.h.levels[0], rank);
  lat.clusters.clear();
  for (std::size_t i = 0; i < local.size(); ++i) lat.clusters.emplace_back();
  c.sites.resize(local.size());

  for (std::size_t i = 0; i < local.size(); ++i) {
    auto& cl = lat.clusters[i];
    auto& sites = c.sites[i];
    sites.box = damn::core::grow(local[i].box, 1);
    for (auto& g : sites.g) g.assign(sites.box.size(), -1);

    damn::core::for_each_point(sites.box, [&](auto const& cell) {
      bool const owned = damn::core::isIn(cell, local[i].box);
      for (auto const site : sites_of_cell) {
        auto const key = key_of(cell, site);
        if (!keys.has(key)) continue;
        double const x = key.x * h, y = key.y * h, z = key.z * h;
        auto const g = cl.alloc_cell(0, AdaptiveCellCluster::zone_id(x, y, z));
        sites.at(site, cell) = g;
        auto ref = cl[g];
        ref.u = u0;
        ref.L = d1;
        ref.lam = lam0;
        ref.fixed_u = owned ? 0 : 1;
        ref.fixed_phi = owned ? 0 : static_cast<std::uint16_t>((1u << CellSlab::NF) - 1);
        ref.parent = -1;
        ref.first_child = -1;
        ref.n_children = 0;
        auto& slab = cl.slab(AdaptiveCellCluster::slab_of(g));
        auto const slot = AdaptiveCellCluster::slot_of(g);
        slab.pos[0 * CellSlab::CAPACITY + slot] = x;
        slab.pos[1 * CellSlab::CAPACITY + slot] = y;
        slab.pos[2 * CellSlab::CAPACITY + slot] = z;
        for (std::size_t k = 0; k < CellSlab::NF; ++k) {
          ref.phi[k] = 0.0;
          ref.nb[k] = -1;
        }
      }
    });

    damn::core::for_each_point(sites.box, [&](auto const& cell) {
      for (auto const site : sites_of_cell) {
        auto const g = std::as_const(sites).at(site, cell);
        if (g < 0) continue;
        auto const key = key_of(cell, site);
        auto ref = cl[g];
        for (std::size_t k = 0; k < CellSlab::NF; ++k) {
          auto const d = discrete::bcc::face_delta(k);
          Key const to{key.x + d[0], key.y + d[1], key.z + d[2]};
          ref.nb[k] = std::as_const(sites).at(site_of(to), cell_of(to));
        }
      }
    });
    lat.rebuild_leaf_indices(cl);
  }

  std::unordered_map<std::uint32_t, std::size_t> index_of;
  for (std::size_t i = 0; i < local.size(); ++i) index_of[local[i].id] = i;
  c.data = damn::amr::allocate(c.h.levels[0], rank, [&](damn::amr::Patch<3> const& patch) {
    ClusterData d;
    auto const it = index_of.find(patch.id);
    if (it == index_of.end()) return d;  // the prototype of a rank without patches: boxes only
    auto* cl = &lat.clusters[it->second];
    auto const* sites = &c.sites[it->second];
    d.u_corner = {cl, sites, Site::corner, Field::u};
    d.u_body = {cl, sites, Site::body, Field::u};
    d.lam_corner = {cl, sites, Site::corner, Field::lam};
    d.lam_body = {cl, sites, Site::body, Field::lam};
    return d;
  });
  c.neighbours = damn::amr::make_neighbours(c.h, 0, rank, topology, 1);
  return c;
}

}  // namespace equilibrium::dst::continuous::clusters

#endif  // _EQUILIBRIUM_CORE_DATA_LATTICE_CONTINUOUS_DAMN_CLUSTERS_HPP_
