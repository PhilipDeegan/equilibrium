// continuous::clusters with mpi_ranks > 0: test_clusters.cpp's comparison with the clusters over
// MPI ranks, ghosts exchanged by damn::mpi::copy; any number of ranks, each reporting its checks
//   mkn clean build -p test_damn_clusters_mpi -dKO -t 4 -x mpi
//   mpirun -n 3 ./bin/test_damn_clusters_mpi/equilibrium

#include "tst/eq/damn/clusters_test.hpp"

#include "damn/mpi/mpi_utils.hpp"
#include "mkn/kul/fail.hpp"

#include <mpi.h>

#include <array>
#include <cstddef>
#include <exception>
#include <string>
#include <vector>

using namespace clusters_test;

namespace {
damn::core::ExecutionModel exec;

struct Mpi {
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

void run(KeyBox const& keys, std::vector<std::array<int, 3>> const& patch_sizes) {
  auto const ref = reference(keys);
  for (auto const& patch_cells : patch_sizes) {
    auto const counts = damn::mpi::collect(compare<Mpi>(keys, patch_cells, ref));
    std::size_t total = 0;
    for (auto const n : counts) total += n;
    MKN_KUL_TEST_CHECK(total == ref.size(), "every site once over the ranks");
  }
}
}  // namespace

int main(int argc, char** argv) try {
  MPI_Init(&argc, &argv);
  damn::mpi::set_ranks(exec);

  run(periodic({16, 12, 8}), {{8, 6, 4}, {4, 3, 2}, {3, 4, 4}, {1, 1, 1}});
  run(bounded({7, 6, 5}), {{8, 8, 8}, {3, 2, 4}, {1, 1, 1}});
  run(bounded({4, 4, 4}), {{3, 2, 4}});

  MPI_Finalize();
  return 0;
} catch (std::exception const& e) {
  mkn::kul::fail(e.what());
}
