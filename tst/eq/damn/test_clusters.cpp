// continuous::clusters: a box (periodic, or bounded as build_bcc_box) as several clusters (damn
// patches, ghosts exchanged by damn) steps bit for bit as one cluster, u and every phi of every
// site, on the main thread and with the clusters side by side on leased threads
//   mkn clean build test -p test_damn_clusters -dKO -t 4 -x mpi

#include "tst/eq/damn/clusters_test.hpp"

#include "mkn/kul/fail.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <exception>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

using namespace clusters_test;

namespace {
damn::core::ExecutionModel exec;

struct Serial {
  static auto build(AdaptiveLattice& lat, KeyBox const& keys,
                    std::array<int, 3> const& patch_cells) {
    return continuous::clusters::build(lat, keys, exec, {.patch_cells = patch_cells});
  }
  static void exchange(continuous::clusters::Clusters& c, std::vector<std::string> const& names) {
    continuous::clusters::exchange(c, names);
  }
  static void step_all(AdaptiveLattice& lat, continuous::clusters::Clusters& c,
                       discrete::TimeStepper const& ts) {
    continuous::clusters::step_all(lat, c, ts);
  }
};

// 4 usable threads: floor(1.2 * 4) = 4 schedulable, the main thread and a lease of 3, the 8
//  clusters on 3 pool threads at once
void clusters_side_by_side() {
  damn::core::ExecutionModel four{.usable_threads = 4};
  AdaptiveLattice lat;
  auto c =
      continuous::clusters::build(lat, periodic({16, 12, 8}), four, {.patch_cells = {4, 3, 2}});
  std::mutex m;
  std::set<std::thread::id> ids;
  std::size_t most = 0;
  continuous::clusters::for_each_cluster(lat, c, [&](auto&) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    std::lock_guard const lock{m};
    ids.insert(std::this_thread::get_id());
    most = std::max<std::size_t>(most, four.threads_used);
  });
  MKN_KUL_TEST_CHECK(lat.clusters.size() == 8);
  MKN_KUL_TEST_CHECK(most == 4, "a lease of 3 while stepping: " + std::to_string(most));
  MKN_KUL_TEST_CHECK(ids.size() == 3 && !ids.contains(std::this_thread::get_id()),
                     "the clusters on 3 pool threads");
  MKN_KUL_TEST_CHECK(four.threads_used == 1, "the lease given back");
}
}  // namespace

int main() try {
  clusters_side_by_side();

  for (std::size_t const threads : {1, 4}) {
    exec.usable_threads = threads;

    auto const P = periodic({16, 12, 8});  // 8 x 6 x 4 cells
    auto const ref = reference(P);
    for (auto const& patch_cells :
         {std::array{8, 6, 4},   // one cluster, its own periodic images as ghosts
          std::array{4, 3, 2},   // 8 clusters
          std::array{3, 4, 4},   // uneven: 3, 3, 2 by 4, 2 by 4
          std::array{1, 1, 1}})  // one cell each: every neighbour a ghost
      MKN_KUL_TEST_CHECK(compare<Serial>(P, patch_cells, ref) == ref.size(), "every site once");

    // keys -N..N: odd N, the lowest cell layer holds body sites only; even N, the uppermost corner
    //  sites only
    for (auto const& N : {std::array{7, 6, 5}, std::array{4, 4, 4}}) {
      auto const B = bounded(N);
      auto const ref_b = reference(B);
      for (auto const& patch_cells :
           {std::array{8, 8, 8}, std::array{3, 2, 4}, std::array{1, 1, 1}})
        MKN_KUL_TEST_CHECK(compare<Serial>(B, patch_cells, ref_b) == ref_b.size(),
                           "every site once");
    }
    MKN_KUL_TEST_CHECK(exec.threads_used == 1, "every lease given back");
  }
  return 0;
} catch (std::exception const& e) {
  mkn::kul::fail(e.what());
}
