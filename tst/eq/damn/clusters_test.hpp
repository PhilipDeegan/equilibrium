#ifndef EQ_TST_DAMN_CLUSTERS_TEST_HPP
#define EQ_TST_DAMN_CLUSTERS_TEST_HPP

// test_clusters.cpp's and test_clusters_mpi.cpp's: a reference run of one build_bcc_box_periodic
// cluster, compared site by site, bit for bit, with clusters built, exchanged and stepped by Ops

#include "eq/core/data/lattice/continuous/damn_clusters.hpp"
#include "eq/core/data/lattice/bcc_geometry.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <tuple>

using namespace equilibrium::dst;
using continuous::AdaptiveLattice;
using discrete::CellSlab;
using discrete::bcc::IntVec3;

namespace clusters_test {

int failures = 0;

void check(bool const ok, std::string const& what) {
  if (!ok) {
    ++failures;
    if (failures < 20) std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  }
}

using Key = std::tuple<int, int, int>;

struct State {
  double u;
  std::array<double, CellSlab::NF> phi;
};

double u_init(IntVec3 const& k) { return 0.01 * ((7 * k.x + 13 * k.y + 29 * k.z) % 17 + 17) - 0.1; }
double lam_init(IntVec3 const& k) { return 0.1 * ((k.x + 2 * k.y + 3 * k.z) & 3); }

// the key of a periodic box site, wrapped into [-P/2, P/2)
IntVec3 wrap(IntVec3 k, std::array<int, 3> const& P) {
  auto w = [](int i, int p) { return ((i + p / 2) % p + p) % p - p / 2; };
  return {w(k.x, P[0]), w(k.y, P[1]), w(k.z, P[2])};
}

IntVec3 key_at(discrete::AdaptiveCellCluster& cl, std::int32_t const g, double const h) {
  auto const& slab = cl.slab(discrete::AdaptiveCellCluster::slab_of(g));
  auto const slot = discrete::AdaptiveCellCluster::slot_of(g);
  auto r = [&](int d) {
    return static_cast<int>(std::lround(slab.pos[d * CellSlab::CAPACITY + slot] / h));
  };
  return {r(0), r(1), r(2)};
}

std::map<Key, State> reference(std::array<int, 3> const& P, int const steps, double const dt) {
  AdaptiveLattice lat;
  auto& cl = lat.clusters[0];
  discrete::bcc::build_bcc_box_periodic(cl, P[0], P[1], P[2]);
  double const h = 1.0 / std::sqrt(3.0);
  lat.rebuild_leaf_indices(cl);
  for (auto const g : cl.leaf_indices) {
    auto const k = key_at(cl, g, h);
    cl[g].u = u_init(k);
    cl[g].lam = lam_init(k);
  }
  auto const ts = discrete::TimeStepper::make_fixed(dt);
  for (int s = 0; s < steps; ++s) lat.step(cl, ts);

  std::map<Key, State> out;
  for (auto const g : cl.leaf_indices) {
    auto const k = key_at(cl, g, h);
    State st{cl[g].u, {}};
    for (std::size_t f = 0; f < CellSlab::NF; ++f) st.phi[f] = cl[g].phi[f];
    out[{k.x, k.y, k.z}] = st;
  }
  return out;
}

bool same(double const a, double const b) {
  return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

// Ops: build(lat, P, patch_cells), exchange(c, names), step_all(lat, c, ts); this rank's sites
//  against ref, the ones it owns: n_owned, to add over the ranks
template <typename Ops>
std::size_t compare(std::array<int, 3> const& P, std::array<int, 3> const& patch_cells,
                    int const steps, double const dt, std::map<Key, State> const& ref) {
  AdaptiveLattice lat;
  auto c = Ops::build(lat, P, patch_cells);
  for (std::size_t i = 0; i < lat.clusters.size(); ++i)
    continuous::clusters::for_each_owned(c, i, [&](std::int32_t const g, IntVec3 const& k) {
      auto const w = wrap(k, P);
      lat.clusters[i][g].u = u_init(w);
      lat.clusters[i][g].lam = lam_init(w);
    });
  Ops::exchange(c, continuous::clusters::lam_names);
  Ops::exchange(c, continuous::clusters::u_names);

  // every ghost holds its owner's values
  double const h = 1.0 / std::sqrt(3.0);
  for (auto& cl : lat.clusters)
    for (auto const g : cl.leaf_indices) {
      auto const w = wrap(key_at(cl, g, h), P);
      check(same(cl[g].u, u_init(w)) && same(cl[g].lam, lam_init(w)), "ghost filled");
    }

  auto const ts = discrete::TimeStepper::make_fixed(dt);
  for (int s = 0; s < steps; ++s) Ops::step_all(lat, c, ts);

  std::size_t n = 0;
  auto const label = "patch " + std::to_string(patch_cells[0]) + "x" +
                     std::to_string(patch_cells[1]) + "x" + std::to_string(patch_cells[2]) + " (" +
                     std::to_string(lat.clusters.size()) + " clusters)";
  for (std::size_t i = 0; i < lat.clusters.size(); ++i)
    continuous::clusters::for_each_owned(c, i, [&](std::int32_t const g, IntVec3 const& k) {
      auto const w = wrap(k, P);
      auto const it = ref.find({w.x, w.y, w.z});
      check(it != ref.end(), label + ": site in the reference");
      if (it == ref.end()) return;
      ++n;
      auto cell = lat.clusters[i][g];
      check(same(cell.u, it->second.u), label + ": u at " + std::to_string(w.x) + "," +
                                            std::to_string(w.y) + "," + std::to_string(w.z));
      for (std::size_t f = 0; f < CellSlab::NF; ++f)
        check(same(cell.phi[f], it->second.phi[f]), label + ": phi " + std::to_string(f));
    });
  std::printf("%s: %zu sites, %d failures so far\n", label.c_str(), n, failures);
  return n;
}

}  // namespace clusters_test

#endif  // EQ_TST_DAMN_CLUSTERS_TEST_HPP
