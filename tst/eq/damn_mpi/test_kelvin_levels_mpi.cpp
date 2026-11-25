// kelvin::Levels over MPI ranks: kelvin_levels_test.hpp's model on one and two levels gives the
// serial bits whatever the ranks; any number of ranks, each reporting its checks
//   mkn clean build -p test_kelvin_levels_mpi -dKO -t 4 -x mpi
//   mpirun -n 3 ./bin/test_kelvin_levels_mpi/equilibrium

#include "tst/eq/damn/kelvin_levels_test.hpp"

#include "damn/mpi/execution_model.hpp"
#include "damn/mpi/mpi_utils.hpp"
#include "damn/core/exec/execution_model.hpp"
#include "mkn/kul/fail.hpp"
#include "mkn/kul/test.hpp"

#include <mpi.h>

#include <array>
#include <cstddef>
#include <exception>
#include <set>
#include <string>
#include <utility>
#include <vector>

MKN_KUL_TEST_DECLARE();

using namespace kelvin_levels_test;

namespace {

// the serial bits, before MPI_Init (damn::mpi::check: mpi_ranks 0 is MPI not initialised)
Reference serial(Setup const& s) {
  damn::core::ExecutionModel exec{};
  return {owned(make_and_run(s, exec))};
}

void same_bits_over_ranks(std::string const& what, Setup s, Reference const& ref) {
  for (auto const& [coarse, fine_cells] :
       std::vector<std::pair<std::array<int, 3>, std::array<int, 3>>>{
           {{8, 8, 8}, {12, 12, 12}}, {{3, 2, 4}, {3, 6, 9}}, {{1, 1, 1}, {3, 3, 3}}}) {
    s.patch_cells = coarse;
    s.fine_patch_cells = fine_cells;
    damn::core::ExecutionModel exec{.usable_threads = 2};
    damn::mpi::set_ranks(exec);
    auto const l = make_and_run(s, exec);
    auto const got = owned(l);
    if (l.levels.size() > 1 and coarse[0] == 1 and damn::mpi::size() > 1) {
      std::set<int> owners;
      for (auto const& p : l.h.levels[1].patches) owners.insert(p.rank);
      MKN_KUL_TEST_CHECK(owners.size() > 1, what + ": level 1 over several ranks");
    }
    auto const counts = damn::mpi::collect(std::array{got.size(), ref.matching(got)});
    std::size_t sites = 0, same = 0;
    for (auto const& [n, m] : counts) {
      sites += n;
      same += m;
    }
    MKN_KUL_TEST_CHECK(sites == ref.sites.size() and same == sites,
                       what + " patches " + std::to_string(coarse[0]) +
                           ": every site once over the ranks, the serial bits");
  }
}

}  // namespace

int main(int argc, char** argv) try {
  std::vector<std::pair<std::string, Setup>> cases;
  for (auto const& [name, setup] : std::vector<std::pair<std::string, Setup>>{
           {"periodic", periodic_setup()}, {"bounded", bounded_setup()}}) {
    auto level_0 = setup;
    level_0.tagged.clear();
    cases.emplace_back(name + " level 0", level_0);
    cases.emplace_back(name + " two levels", setup);
  }
  std::vector<Reference> refs;
  for (auto const& c : cases) refs.push_back(serial(c.second));

  MPI_Init(&argc, &argv);
  for (std::size_t i = 0; i < cases.size(); ++i)
    same_bits_over_ranks(cases[i].first, cases[i].second, refs[i]);
  MPI_Finalize();
  return 0;
} catch (std::exception const& e) {
  mkn::kul::fail(e.what());
}
