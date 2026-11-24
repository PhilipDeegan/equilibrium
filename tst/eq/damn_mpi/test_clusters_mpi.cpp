// continuous::clusters::mpi: test_clusters.cpp's comparison with the clusters over MPI ranks,
// ghosts exchanged by damn::mpi::copy; any number of ranks
//   mkn clean build -p test_damn_clusters_mpi -dKO -t 4 -x mpi && mpirun -n 3
//   ./bin/test_damn_clusters_mpi/...

#include "eq/core/data/lattice/continuous/damn_clusters_mpi.hpp"
#include "tst/eq/damn/clusters_test.hpp"

#include <mpi.h>

#include <array>
#include <cstdio>
#include <string>
#include <vector>

using namespace clusters_test;

namespace {
struct Mpi {
  static auto build(AdaptiveLattice& lat, Keys const& keys, std::array<int, 3> const& patch_cells) {
    return continuous::clusters::mpi::build(lat, keys, patch_cells);
  }
  static void exchange(continuous::clusters::Clusters& c, std::vector<std::string> const& names) {
    continuous::clusters::mpi::exchange(c, names);
  }
  static void step_all(AdaptiveLattice& lat, continuous::clusters::Clusters& c,
                       discrete::TimeStepper const& ts) {
    continuous::clusters::mpi::step_all(lat, c, ts);
  }
};
}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int const steps = 40;
  double const dt = 0.05;
  auto const run = [&](Keys const& keys, auto const& patch_sizes) {
    auto const ref = reference(keys, steps, dt);
    for (auto const& patch_cells : patch_sizes) {
      long n = static_cast<long>(compare<Mpi>(keys, patch_cells, steps, dt, ref)), total = 0;
      MPI_Allreduce(&n, &total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      check(total == static_cast<long>(ref.size()), "every site once over the ranks");
    }
  };
  run(periodic({16, 12, 8}), std::vector{std::array{8, 6, 4}, std::array{4, 3, 2},
                                         std::array{3, 4, 4}, std::array{1, 1, 1}});
  run(bounded({7, 6, 5}),
      std::vector{std::array{8, 8, 8}, std::array{3, 2, 4}, std::array{1, 1, 1}});
  run(bounded({4, 4, 4}), std::vector{std::array{3, 2, 4}});

  int all = 0;
  MPI_Allreduce(&failures, &all, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) std::printf(all ? "%d failures\n" : "PASS\n", all);
  MPI_Finalize();
  return all ? 1 : 0;
}
