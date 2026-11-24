#ifndef EQ_TST_DAMN_CLUSTERS_TEST_HPP
#define EQ_TST_DAMN_CLUSTERS_TEST_HPP

// test_clusters.cpp's and test_clusters_mpi.cpp's: a reference run of one cluster
// (build_bcc_box_periodic, build_bcc_box), compared site by site, bit for bit, with the same sites
// as clusters built, exchanged and stepped by Ops. Checks are the including test's Tester's
// (MKN_KUL_TEST_DECLARE here: one test per file)

#include "eq/core/data/lattice/continuous/damn_clusters.hpp"
#include "eq/core/data/lattice/bcc_geometry.hpp"

#include "mkn/kul/test.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <tuple>

MKN_KUL_TEST_DECLARE();

namespace clusters_test {

using namespace equilibrium::dst;
using continuous::AdaptiveLattice;
using continuous::clusters::Key;
using continuous::clusters::KeyBox;
using discrete::CellSlab;

using RefKey = std::tuple<int, int, int>;

struct State {
  double u;
  std::array<double, CellSlab::NF> phi;
};

inline double u_init(Key const& k) {
  return 0.01 * ((7 * k[0] + 13 * k[1] + 29 * k[2]) % 17 + 17) - 0.1;
}
inline double lam_init(Key const& k) { return 0.1 * ((k[0] + 2 * k[1] + 3 * k[2]) & 3); }

// the sites of keys: periodic (build_bcc_box_periodic(P)) or bounded (build_bcc_box, sides 2 N h)
inline KeyBox periodic(std::array<int, 3> const& P) {
  return {Key{std::array{-P[0] / 2, -P[1] / 2, -P[2] / 2}},
          Key{std::array{P[0] / 2 - 1, P[1] / 2 - 1, P[2] / 2 - 1}},
          {true, true, true}};
}
inline KeyBox bounded(std::array<int, 3> const& N) {
  return {Key{std::array{-N[0], -N[1], -N[2]}}, Key{N}, {}};
}

inline double const h = 1.0 / std::sqrt(3.0);

inline Key key_at(discrete::AdaptiveCellCluster& cl, std::int32_t const g) {
  auto const& slab = cl.slab(discrete::AdaptiveCellCluster::slab_of(g));
  auto const slot = discrete::AdaptiveCellCluster::slot_of(g);
  auto r = [&](int d) {
    return static_cast<int>(std::lround(slab.pos[d * CellSlab::CAPACITY + slot] / h));
  };
  return Key{std::array{r(0), r(1), r(2)}};
}

struct Run {
  int steps = 40;
  double dt = 0.05;
};

inline std::map<RefKey, State> reference(KeyBox const& keys, Run const& run = {}) {
  AdaptiveLattice lat;
  auto& cl = lat.clusters[0];
  if (keys.periodic[0])
    discrete::bcc::build_bcc_box_periodic(cl, keys.upper[0] - keys.lower[0] + 1,
                                          keys.upper[1] - keys.lower[1] + 1,
                                          keys.upper[2] - keys.lower[2] + 1);
  else
    discrete::bcc::build_bcc_box(cl, 2 * keys.upper[0] * h, 2 * keys.upper[1] * h,
                                 2 * keys.upper[2] * h);
  lat.rebuild_leaf_indices(cl);
  for (auto const g : cl.leaf_indices) {
    auto const k = key_at(cl, g);
    cl[g].u = u_init(k);
    cl[g].lam = lam_init(k);
  }
  auto const ts = discrete::TimeStepper::make_fixed(run.dt);
  for (int s = 0; s < run.steps; ++s) lat.step(cl, ts);

  std::map<RefKey, State> out;
  for (auto const g : cl.leaf_indices) {
    auto const k = key_at(cl, g);
    State st{cl[g].u, {}};
    for (std::size_t f = 0; f < CellSlab::NF; ++f) st.phi[f] = cl[g].phi[f];
    out[{k[0], k[1], k[2]}] = st;
  }
  return out;
}

inline bool same(double const a, double const b) {
  return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

// Ops: build(lat, keys, patch_cells), exchange(c, names), step_all(lat, c, ts); this rank's sites
//  against ref, one check per comparison; the sites this rank owns, to add over the ranks
template <typename Ops>
std::size_t compare(KeyBox const& keys, std::array<int, 3> const& patch_cells,
                    std::map<RefKey, State> const& ref, Run const& run = {}) {
  AdaptiveLattice lat;
  auto c = Ops::build(lat, keys, patch_cells);
  for (std::size_t i = 0; i < lat.clusters.size(); ++i)
    continuous::clusters::for_each_owned(c, i, [&](std::int32_t const g, Key const& k) {
      auto const w = keys.wrap(k);
      lat.clusters[i][g].u = u_init(w);
      lat.clusters[i][g].lam = lam_init(w);
    });
  Ops::exchange(c, continuous::clusters::lam_names);
  Ops::exchange(c, continuous::clusters::u_names);

  auto const label = std::string{keys.periodic[0] ? "periodic" : "bounded"} + ", patch " +
                     std::to_string(patch_cells[0]) + "x" + std::to_string(patch_cells[1]) + "x" +
                     std::to_string(patch_cells[2]) + " (" + std::to_string(lat.clusters.size()) +
                     " clusters)";

  std::size_t unfilled = 0;
  for (auto& cl : lat.clusters)
    for (auto const g : cl.leaf_indices) {
      auto const w = keys.wrap(key_at(cl, g));
      unfilled += !(same(cl[g].u, u_init(w)) && same(cl[g].lam, lam_init(w)));
    }
  MKN_KUL_TEST_CHECK(unfilled == 0,
                     label + ": every ghost its owner's, " + std::to_string(unfilled) + " not");

  auto const ts = discrete::TimeStepper::make_fixed(run.dt);
  for (int s = 0; s < run.steps; ++s) Ops::step_all(lat, c, ts);

  std::size_t n = 0, missing = 0, differ = 0;
  for (std::size_t i = 0; i < lat.clusters.size(); ++i)
    continuous::clusters::for_each_owned(c, i, [&](std::int32_t const g, Key const& k) {
      auto const w = keys.wrap(k);
      auto const it = ref.find({w[0], w[1], w[2]});
      if (it == ref.end()) {
        ++missing;
        return;
      }
      ++n;
      auto cell = lat.clusters[i][g];
      bool ok = same(cell.u, it->second.u);
      for (std::size_t f = 0; f < CellSlab::NF; ++f)
        ok = ok && same(cell.phi[f], it->second.phi[f]);
      differ += !ok;
    });
  MKN_KUL_TEST_CHECK(missing == 0 && differ == 0, label + ": " + std::to_string(n) +
                                                      " sites as one cluster's, " +
                                                      std::to_string(differ) + " differ, " +
                                                      std::to_string(missing) + " not in it");
  return n;
}

}  // namespace clusters_test

#endif  // EQ_TST_DAMN_CLUSTERS_TEST_HPP
