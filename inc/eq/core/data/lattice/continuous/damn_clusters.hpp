#ifndef _EQUILIBRIUM_CORE_DATA_LATTICE_CONTINUOUS_DAMN_CLUSTERS_HPP_
#define _EQUILIBRIUM_CORE_DATA_LATTICE_CONTINUOUS_DAMN_CLUSTERS_HPP_

// The continuous DST model on a kelvin::Level: one AdaptiveCellCluster per patch holding its sites
// and, as ghosts, the sites its faces reach past the patch (one cell): leaves with u and phi fixed,
// filled from their owners by damn between AdaptiveLattice::step_u and step_phi. The cluster
// storage is DST's (u, lam, phi per face); its u and lam reach damn through ClusterField, a BoxData
// over a cluster's cells by site (Sites: cell and site to global index)

// IWYU pragma: no_include "mpi.h"
#include "eq/core/data/lattice/continuous/adaptive_lattice.hpp"
#include "eq/core/data/lattice/kelvin/level.hpp"
#include "eq/core/data/lattice/cell/adaptive_cell.hpp"
#include "eq/core/data/lattice/cluster.hpp"
#include "damn/amr/kelvin/sites.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/lattice.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace damn::core {
struct ExecutionModel;
}  // namespace damn::core

namespace equilibrium::dst::discrete {
struct TimeStepper;
}  // namespace equilibrium::dst::discrete

namespace equilibrium::dst::continuous::clusters {

using kelvin::Box3;
using kelvin::Key;
using kelvin::KeyBox;
using kelvin::Point3;
using kelvin::Site;
using kelvin::sites_of_cell;

// a cluster's cells (global indices) per site over box (its patch grown by 1), -1: none (past a
// bounded domain's end)
struct Sites {
  Box3 box{};
  std::array<std::vector<std::int32_t>, 2> g{};
  mutable double absent = 0;  // what damn reads and writes for a site that is not one

  [[nodiscard]] std::size_t flat(Point3 const& c) const;
  [[nodiscard]] std::int32_t& at(Site site, Point3 const& c);
  [[nodiscard]] std::int32_t at(Site site, Point3 const& c) const;
};

enum class Field : std::uint8_t { u, lam };

// one field on one site of a cluster's cells, as damn BoxData over the cluster's patch
struct ClusterField {
  discrete::AdaptiveCellCluster* cluster = nullptr;
  Sites const* sites = nullptr;
  Site site = Site::corner;
  Field field = Field::u;
};

[[nodiscard]] Box3 domain_box(ClusterField const&, Box3 const& cells);
[[nodiscard]] Box3 ghost_box(ClusterField const&, Box3 const& cells);
[[nodiscard]] double& value(ClusterField const& f, Box3 const&, Point3 const& p);

struct ClusterData {
  ClusterField u_corner, u_body, lam_corner, lam_body;
};

void for_each_resource(ClusterData& d, auto&& fn);

inline std::vector<std::string> const u_names{"u_corner", "u_body"};
inline std::vector<std::string> const lam_names{"lam_corner", "lam_body"};

// the level and its clusters' site maps (lat's clusters, one per local patch, in order)
struct Clusters {
  kelvin::Level<ClusterData> level;
  std::vector<Sites> sites{};
};

// how a domain is cut and started: patches of at most patch_cells cells (2 h each), hex face
//  distance d1 (h = d1 / sqrt 3), u0 and lam0 everywhere, phi 0
struct Setup {
  std::array<int, 3> patch_cells{};
  double d1 = 1.0, u0 = 0.0, lam0 = 0.0;
};

// keys' domain cut as setup says over exec's ranks; lat's clusters replaced by this rank's
// patches. lat's leaves include the ghosts:
// diagnostics take for_each_owned
[[nodiscard]] Clusters build(AdaptiveLattice& lat, KeyBox const& keys,
                             damn::core::ExecutionModel& exec, Setup const& setup);

// build_bcc_box_periodic(P)'s sites (P multiples of 4)
[[nodiscard]] Clusters build_periodic(AdaptiveLattice& lat, std::array<int, 3> const& P,
                                      damn::core::ExecutionModel& exec, Setup const& setup);

// build_bcc_box's sites for sides 2 N h: keys -N..N, bounded
[[nodiscard]] Clusters build_box(AdaptiveLattice& lat, std::array<int, 3> const& N,
                                 damn::core::ExecutionModel& exec, Setup const& setup);

// ghosts' u (or lam) from their owners (collective with MPI)
void exchange(Clusters& c, std::vector<std::string> const& names);

// the cluster's own sites (not ghosts): fn(global index, key)
void for_each_owned(Clusters const& c, std::size_t i, auto&& fn);

// fn(cluster) for every cluster, side by side on leased threads unless lat threads within one
void for_each_cluster(AdaptiveLattice& lat, Clusters& c, auto&& fn);

// a step of every cluster, ghosts' u refreshed between its halves
void step_all(AdaptiveLattice& lat, Clusters& c, discrete::TimeStepper const& ts);

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

inline Box3 domain_box(ClusterField const&, Box3 const& cells) { return cells; }
inline Box3 ghost_box(ClusterField const&, Box3 const& cells) { return damn::core::grow(cells, 1); }
inline double& value(ClusterField const& f, Box3 const&, Point3 const& p) {
  auto const g = f.sites->at(f.site, p);
  if (g < 0) return f.sites->absent;
  auto cell = (*f.cluster)[g];
  return f.field == Field::u ? cell.u : cell.lam;
}

void for_each_resource(ClusterData& d, auto&& fn) {
  fn("u_corner", d.u_corner);
  fn("u_body", d.u_body);
  fn("lam_corner", d.lam_corner);
  fn("lam_body", d.lam_body);
}

inline void exchange(Clusters& c, std::vector<std::string> const& names) {
  kelvin::exchange(c.level, names);
}

void for_each_owned(Clusters const& c, std::size_t const i, auto&& fn) {
  damn::core::for_each_point(c.level.local[i].box, [&](auto const& cell) {
    for (auto const site : sites_of_cell)
      if (auto const g = c.sites[i].at(site, cell); g >= 0)
        fn(g, damn::amr::kelvin::key_of(cell, site));
  });
}

void for_each_cluster(AdaptiveLattice& lat, Clusters& c, auto&& fn) {
  if (lat.n_threads() > 1) {
    for (auto& cl : lat.clusters) fn(cl);
    return;
  }
  kelvin::for_each_patch(c.level, [&](std::size_t const i, auto&) { fn(lat.clusters[i]); });
}

inline void step_all(AdaptiveLattice& lat, Clusters& c, discrete::TimeStepper const& ts) {
  for_each_cluster(lat, c, [&](auto& cl) { lat.step_u(cl, ts); });
  exchange(c, u_names);
  for_each_cluster(lat, c, [&](auto& cl) { lat.step_phi(cl, ts); });
}

inline Clusters build_periodic(AdaptiveLattice& lat, std::array<int, 3> const& P,
                               damn::core::ExecutionModel& exec, Setup const& setup) {
  for (auto const p : P)
    if (p < 4 || p % 4)
      throw std::invalid_argument(
          "clusters::build_periodic: periods are multiples of 4 (whole cells)");
  KeyBox const keys{Key{std::array{-P[0] / 2, -P[1] / 2, -P[2] / 2}},
                    Key{std::array{P[0] / 2 - 1, P[1] / 2 - 1, P[2] / 2 - 1}},
                    {true, true, true}};
  return build(lat, keys, exec, setup);
}

inline Clusters build_box(AdaptiveLattice& lat, std::array<int, 3> const& N,
                          damn::core::ExecutionModel& exec, Setup const& setup) {
  KeyBox const keys{Key{std::array{-N[0], -N[1], -N[2]}}, Key{N}, {false, false, false}};
  return build(lat, keys, exec, setup);
}

inline Clusters build(AdaptiveLattice& lat, KeyBox const& keys, damn::core::ExecutionModel& exec,
                      Setup const& setup) {
  using discrete::AdaptiveCellCluster;
  using discrete::CellSlab;

  auto const [patch_cells, d1, u0, lam0] = setup;
  Clusters c{kelvin::make_level<ClusterData>(keys, patch_cells, exec, d1), {}};
  auto const& local = c.level.local;
  double const h = d1 / std::sqrt(3.0);

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
        if (!keys.has(cell, site)) continue;
        auto const key = damn::amr::kelvin::key_of(cell, site);
        double const x = key[0] * h, y = key[1] * h, z = key[2] * h;
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

    // damn's faces are eq's (bcc::face_delta): the same 14 in the same order
    damn::core::for_each_point(sites.box, [&](auto const& cell) {
      for (auto const site : sites_of_cell) {
        auto const g = std::as_const(sites).at(site, cell);
        if (g < 0) continue;
        auto ref = cl[g];
        for (std::size_t k = 0; k < CellSlab::NF; ++k) {
          auto const [to, to_site] = damn::amr::kelvin::neighbour(cell, site, k);
          ref.nb[k] = std::as_const(sites).at(to_site, to);
        }
      }
    });
    lat.rebuild_leaf_indices(cl);
  }

  kelvin::allocate(c.level, [&](damn::amr::Patch<3> const&, std::size_t const i) {
    ClusterData d;
    if (i == local.size()) return d;  // the prototype of a rank without patches: boxes only
    auto* cl = &lat.clusters[i];
    auto const* sites = &c.sites[i];
    d.u_corner = {cl, sites, Site::corner, Field::u};
    d.u_body = {cl, sites, Site::body, Field::u};
    d.lam_corner = {cl, sites, Site::corner, Field::lam};
    d.lam_body = {cl, sites, Site::body, Field::lam};
    return d;
  });
  return c;
}

static_assert(damn::amr::kelvin::n_faces<3> == discrete::CellSlab::NF);

}  // namespace equilibrium::dst::continuous::clusters

#endif  // _EQUILIBRIUM_CORE_DATA_LATTICE_CONTINUOUS_DAMN_CLUSTERS_HPP_
