// continuous::clusters: a box (periodic, or bounded as build_bcc_box) as several clusters (damn
// patches, ghosts exchanged by damn) steps bit for bit as one cluster, u and every phi of every
// site
//   mkn clean build test -p test_damn_clusters -dKO -t 4

#include "tst/eq/damn/clusters_test.hpp"

#include <array>
#include <cstdio>
#include <string>
#include <vector>

using namespace clusters_test;

namespace {
struct Serial {
  static auto build(AdaptiveLattice& lat, Keys const& keys, std::array<int, 3> const& patch_cells) {
    return continuous::clusters::build(lat, keys, patch_cells);
  }
  static void exchange(continuous::clusters::Clusters& c, std::vector<std::string> const& names) {
    continuous::clusters::exchange(c, names);
  }
  static void step_all(AdaptiveLattice& lat, continuous::clusters::Clusters& c,
                       discrete::TimeStepper const& ts) {
    continuous::clusters::step_all(lat, c, ts);
  }
};
}  // namespace

int main() {
  int const steps = 40;
  double const dt = 0.05;

  auto const P = periodic({16, 12, 8});  // 8 x 6 x 4 cells
  auto const ref = reference(P, steps, dt);
  for (auto const& patch_cells :
       {std::array{8, 6, 4},   // one cluster, its own periodic images as ghosts
        std::array{4, 3, 2},   // 8 clusters
        std::array{3, 4, 4},   // uneven: 3, 3, 2 by 4, 2 by 4
        std::array{1, 1, 1}})  // one cell each: every neighbour a ghost
    check(compare<Serial>(P, patch_cells, steps, dt, ref) == ref.size(), "every site once");

  // keys -N..N: odd N, the lowest cell layer holds body sites only; even N, the uppermost corner
  //  sites only
  for (auto const& N : {std::array{7, 6, 5}, std::array{4, 4, 4}}) {
    auto const B = bounded(N);
    auto const ref_b = reference(B, steps, dt);
    for (auto const& patch_cells : {std::array{8, 8, 8}, std::array{3, 2, 4}, std::array{1, 1, 1}})
      check(compare<Serial>(B, patch_cells, steps, dt, ref_b) == ref_b.size(), "every site once");
  }

  if (failures) {
    std::fprintf(stderr, "%d failures\n", failures);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
