// kelvin::Level with a model of its own, not DST's: a diffusion over the 14 faces kept in damn
// SiteFields, no eq cluster storage. The same steps from 1 to 192 patches, on 1 and 4 threads,
// give the same bits

#include "eq/core/data/lattice/kelvin/level.hpp"

#include "mkn/kul/fail.hpp"
#include "mkn/kul/test.hpp"

#include "damn/amr/kelvin/site_data.hpp"
#include "damn/amr/kelvin/sites.hpp"
#include "damn/core/exec/execution_model.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <map>
#include <string>
#include <tuple>
#include <vector>

MKN_KUL_TEST_DECLARE();

namespace kv = equilibrium::kelvin;
namespace dk = damn::amr::kelvin;

namespace {

struct Diffusion {
  dk::SiteField<double, 3> u, next;
};

void for_each_resource(Diffusion& d, auto&& fn) {
  dk::for_each_site_resource("u", d.u, fn);
  dk::for_each_site_resource("next", d.next, fn);
}

using Sites = std::map<std::tuple<int, int, int>, double>;

// a step: next = u + k sum over faces (u_nb - u), sites past a bounded end inert
void step(kv::Level<Diffusion>& level, double const k) {
  kv::for_each_patch(level, [&](std::size_t const i, Diffusion& d) {
    auto const& box = level.local[i].box;
    damn::core::for_each_point(box, [&](auto const& c) {
      for (auto const site : kv::sites_of_cell) {
        if (!level.keys.has(c, site)) continue;
        auto const u = value(d.u[site], box, c);
        double sum = 0;
        for (std::size_t f = 0; f < dk::n_faces<3>; ++f) {
          auto const [to, to_site] = dk::neighbour(c, site, f);
          if (level.keys.has(to, to_site)) sum += value(d.u[to_site], box, to) - u;
        }
        value(d.next[site], box, c) = u + k * sum;
      }
    });
    std::swap(d.u, d.next);
  });
  kv::exchange(level, dk::site_names({"u"}));
}

Sites run(dk::KeyBox<3> const& keys, std::array<int, 3> const& patch_cells,
          std::size_t const threads) {
  damn::core::ExecutionModel exec{.usable_threads = threads};
  auto level = kv::make_level<Diffusion>(keys, patch_cells, exec);
  kv::allocate(level, [&](damn::amr::Patch<3> const& p, std::size_t) {
    Diffusion d;
    d.u.allocate(p.box);
    d.next.allocate(p.box);
    damn::core::for_each_point(p.box, [&](auto const& c) {
      for (auto const site : kv::sites_of_cell) {
        auto const key = dk::key_of(c, site);
        value(d.u[site], p.box, c) = 0.01 * ((7 * key[0] + 13 * key[1] + 29 * key[2]) % 17 + 17);
      }
    });
    return d;
  });
  kv::exchange(level, dk::site_names({"u"}));
  for (int s = 0; s < 30; ++s) step(level, 0.02);
  MKN_KUL_TEST_CHECK(exec.threads_used == 1, "every lease given back");

  Sites out;
  for (std::size_t i = 0; i < level.local.size(); ++i) {
    auto const& box = level.local[i].box;
    damn::core::for_each_point(box, [&](auto const& c) {
      for (auto const site : kv::sites_of_cell)
        if (keys.has(c, site)) {
          auto const k = dk::key_of(c, site);
          out[{k[0], k[1], k[2]}] = value(level.data.data[i].u[site], box, c);
        }
    });
  }
  return out;
}

void compare(std::string const& what, Sites const& ref, Sites const& got) {
  bool same = ref.size() == got.size();
  for (auto const& [k, v] : ref)
    same = same and got.contains(k) and
           std::bit_cast<std::uint64_t>(got.at(k)) == std::bit_cast<std::uint64_t>(v);
  MKN_KUL_TEST_CHECK(same, what + ": the one patch, one thread bits");
}

}  // namespace

int main() try {
  dk::KeyBox<3> const periodic{
      dk::Key<3>{std::array{-8, -6, -4}}, dk::Key<3>{std::array{7, 5, 3}}, {true, true, true}};
  dk::KeyBox<3> const bounded{
      dk::Key<3>{std::array{-7, -6, -5}}, dk::Key<3>{std::array{7, 6, 5}}, {}};
  for (auto const* keys : {&periodic, &bounded}) {
    auto const ref = run(*keys, {8, 8, 8}, 1);
    for (std::size_t const threads : {1, 4})
      for (auto const patch_cells : {std::array{8, 8, 8}, std::array{3, 2, 4}, std::array{1, 1, 1}})
        compare(std::string{keys == &periodic ? "periodic " : "bounded "} +
                    std::to_string(patch_cells[0]) + " threads " + std::to_string(threads),
                ref, run(*keys, patch_cells, threads));
  }
  return 0;
} catch (std::exception const& e) {
  mkn::kul::fail(e.what());
}
