// kelvin::Levels with a model of its own (kelvin_levels_test.hpp), serial:
//  - level 0 alone and with a refined level 1: the same steps from one patch a level to many, on
//    1 and 4 threads, give the same bits, periodic and bounded
//  - a linear field plus s t is the model's exact solution (the diffusion of a linear field is 0):
//    on both levels after steps it is still that, so level 1's border ghosts are refined at the
//    substeps' times; level 0's sites under level 1 its coincident fine sites

#include "tst/eq/damn/kelvin_levels_test.hpp"

#include "mkn/kul/fail.hpp"
#include "mkn/kul/test.hpp"

#include "damn/amr/kelvin/sites.hpp"
#include "damn/core/exec/execution_model.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <string>
#include <vector>

MKN_KUL_TEST_DECLARE();

using namespace kelvin_levels_test;

namespace {

// level 0's sites under level 1 hold their coincident fine site's bits (synchronized last)
void injected(Levels const& l, Sites const& sites) {
  if (l.levels.size() < 2) return;
  std::size_t under = 0, same = 0;
  for (auto const& [k, v] : sites) {
    auto const [ln, x, y, z] = k;
    if (ln != 0) continue;
    if (auto const fine = sites.find({1, kv::ratio * x, kv::ratio * y, kv::ratio * z});
        fine != sites.end()) {
      ++under;
      same += same_bits(v, fine->second);
    }
  }
  MKN_KUL_TEST_CHECK(under == std::size_t{2} * 4 * 4 * 4 and same == under,
                     "level 0 under level 1: its coincident fine sites");
}

Sites run(Setup const& s, std::size_t const threads) {
  damn::core::ExecutionModel exec{.usable_threads = threads};
  auto const l = make_and_run(s, exec);
  MKN_KUL_TEST_CHECK(exec.threads_used == 1, "every lease given back");
  auto sites = owned(l);
  injected(l, sites);
  return sites;
}

void same_bits_however_cut(std::string const& what, Setup s) {
  Reference const ref{run(s, 1)};
  std::size_t fine = 0;
  for (auto const& [k, v] : ref.sites) fine += std::get<0>(k) == 1;
  MKN_KUL_TEST_CHECK(fine == (s.tagged.empty() ? 0 : std::size_t{2} * 12 * 12 * 12),
                     what + ": level 1 over the tagged cells, tag buffer 1, tiles of 2 cells");
  for (std::size_t const threads : {1, 4})
    for (auto const& [coarse, fine_cells] :
         std::vector<std::pair<std::array<int, 3>, std::array<int, 3>>>{
             {{8, 8, 8}, {12, 12, 12}}, {{3, 2, 4}, {3, 6, 9}}, {{1, 1, 1}, {3, 3, 3}}}) {
      s.patch_cells = coarse;
      s.fine_patch_cells = fine_cells;
      auto const got = run(s, threads);
      MKN_KUL_TEST_CHECK(got.size() == ref.sites.size() and ref.matching(got) == got.size(),
                         what + " patches " + std::to_string(coarse[0]) + " threads " +
                             std::to_string(threads) + ": the one thread bits");
    }
}

void linear_stays_linear() {
  auto s = bounded_setup();
  s.init = [](dk::Key<3> const& k) { return linear(k[0], k[1], k[2]); };
  s.model = {.D = 0.02, .s = 1.0};
  s.dt = 0.1;
  s.steps = 5;
  for (auto const& [coarse, fine_cells] :
       std::vector<std::pair<std::array<int, 3>, std::array<int, 3>>>{{{8, 8, 8}, {12, 12, 12}},
                                                                      {{3, 2, 4}, {3, 6, 9}}}) {
    s.patch_cells = coarse;
    s.fine_patch_cells = fine_cells;
    auto const sites = run(s, 4);
    double const t = s.steps * s.dt;
    double worst = 0;
    std::size_t fine = 0;
    for (auto const& [k, v] : sites) {
      auto const [ln, x, y, z] = k;
      double const scale = ln == 0 ? 1.0 : 1.0 / kv::ratio;
      worst =
          std::max(worst, std::abs(v - (linear(x * scale, y * scale, z * scale) + s.model.s * t)));
      fine += ln == 1;
    }
    MKN_KUL_TEST_CHECK(fine > 0, "level 1 there");
    MKN_KUL_TEST_CHECK(worst < 1e-12, "linear + s t on both levels, patches " +
                                          std::to_string(coarse[0]) + ": off by " +
                                          std::to_string(worst));
  }
}

}  // namespace

int main() try {
  for (auto const& [name, setup] : std::vector<std::pair<std::string, Setup>>{
           {"periodic", periodic_setup()}, {"bounded", bounded_setup()}}) {
    auto level_0 = setup;
    level_0.tagged.clear();
    same_bits_however_cut(name + " level 0", level_0);
    same_bits_however_cut(name + " two levels", setup);
  }
  linear_stays_linear();
  return 0;
} catch (std::exception const& e) {
  mkn::kul::fail(e.what());
}
