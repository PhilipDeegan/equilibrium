#ifndef EQUILIBRIUM_TST_DAMN_KELVIN_LEVELS_TEST_HPP
#define EQUILIBRIUM_TST_DAMN_KELVIN_LEVELS_TEST_HPP

// kelvin::Levels with a model of its own, not DST's: u_t = D lap u + s, a diffusion over the 14
// faces kept in damn SiteFields (a site missing a face, at a bounded end, has the source only),
// stepped with damn::amr::TimeRefinement (fine dt = coarse dt / 3); a fine level's border ghosts
// refined from the coarse level in time (LinearRefine between the coarse step's ends), the coarse
// sites under it injected after its substeps. No reflux: not conservative.
// test_kelvin_levels.cpp (serial) and test_kelvin_levels_mpi.cpp (MPI) run it

#include "eq/core/data/lattice/kelvin/level.hpp"

#include "damn/amr/clustering.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/inter_level.hpp"
#include "damn/amr/kelvin/inter_level.hpp"
#include "damn/amr/kelvin/site_data.hpp"
#include "damn/amr/kelvin/sites.hpp"
#include "damn/amr/level_map.hpp"
#include "damn/amr/time_refinement.hpp"
#include "damn/core/exec/execution_model.hpp"
#include "damn/core/geometry/box.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace kelvin_levels_test {

namespace kv = equilibrium::kelvin;
namespace dk = damn::amr::kelvin;
namespace da = damn::amr;

struct Diffusion {
  dk::SiteField<double, 3> u, next, u_old;
};

inline void for_each_resource(Diffusion& d, auto&& fn) {
  dk::for_each_site_resource("u", d.u, fn);
  dk::for_each_site_resource("next", d.next, fn);
  dk::for_each_site_resource("u_old", d.u_old, fn);
}

using Levels = kv::Levels<Diffusion>;
using Key3 = std::tuple<int, int, int, int>;  // level, key
using Sites = std::map<Key3, double>;
// u at t = 0 of a level 0 site, by key (h units)
using Init = std::function<double(dk::Key<3> const&)>;

inline std::vector<std::string> const u_names = dk::site_names({"u"});
inline std::vector<da::LevelPair> const u_pairs = da::same_level_names(u_names);
// the coarse level's u at its step's start (u_old) and end (u)
inline std::vector<da::LevelPair> const in_time{{"u_old.corner", "u.corner", "u.corner"},
                                                {"u_old.body", "u.body", "u.body"}};

struct Model {
  double D = 0.02;  // level 0: D dt / (2 h)^2 per unit dt
  double s = 0.0;
};

// a step of level ln over dt: next = u + D dt 9^ln sum over faces (u_nb - u) + s dt
inline void step(Levels& l, int const ln, Model const& m, double const dt) {
  auto& level = l[ln];
  double const k = m.D * dt * std::pow(9.0, ln);
  kv::for_each_patch(l, ln, [&](std::size_t const i, Diffusion& d) {
    auto const& box = level.local[i].box;
    damn::core::for_each_point(box, [&](auto const& c) {
      for (auto const site : kv::sites_of_cell) {
        if (!level.keys.has(c, site)) continue;
        auto const u = value(d.u[site], box, c);
        double sum = 0;
        bool all = true;
        for (std::size_t f = 0; f < dk::n_faces<3>; ++f) {
          auto const [to, to_site] = dk::neighbour(c, site, f);
          all = all and level.keys.has(to, to_site);
          if (all) sum += value(d.u[to_site], box, to) - u;
        }
        value(d.next[site], box, c) = u + (all ? k * sum : 0.0) + m.s * dt;
      }
    });
    std::swap(d.u, d.next);
  });
  kv::exchange(l, ln, u_names);
}

struct Hooks {
  Levels& l;
  Model m;

  void prepare_step(int const ln, double) {
    if (ln + 1 >= da::n_levels(l.h)) return;
    kv::for_each_patch(l, ln, [](std::size_t, Diffusion& d) {
      d.u_old.corner.v = d.u.corner.v;
      d.u_old.body.v = d.u.body.v;
    });
  }
  void advance(int const ln, da::Step const& st) {
    if (ln > 0)
      kv::refine(l, ln, dk::LinearRefine{}, da::LevelRegion::LevelGhosts, in_time,
                 (st.t - st.coarse_t) / st.coarse_dt);
    step(l, ln, m, st.dt);
  }
  void synchronize(int const ln, double) {
    kv::coarsen(l, ln, dk::InjectCoarsen{}, u_pairs);
    kv::exchange(l, ln - 1, u_names);
  }
};

struct Setup {
  dk::KeyBox<3> keys{};
  std::array<int, 3> patch_cells{}, fine_patch_cells{};
  std::vector<kv::Point3> tagged{};  // level 0 cells
  Init init{};
  Model model{};
  int steps = 0;
  double dt = 1;
};

inline da::ClusteringOptions<3> const clustering{
    .tag_buffer = 1, .nesting_buffer = 1, .tile_size = {2, 2, 2}};

inline Diffusion make(da::Patch<3> const& p) {
  Diffusion d;
  auto const nan = std::numeric_limits<double>::quiet_NaN();
  d.u.allocate(p.box, nan);
  d.next.allocate(p.box, nan);
  d.u_old.allocate(p.box, nan);
  return d;
}

// level 0 from init, level 1 over the tagged cells (none: no level 1) refined from it, steps coarse
//  steps
inline Levels make_and_run(Setup const& s, damn::core::ExecutionModel& exec) {
  bool const refined = !s.tagged.empty();
  auto l = kv::make_levels<Diffusion>(
      s.keys, exec, {.patch_cells = s.patch_cells, .max_levels = refined ? 2 : 1});
  kv::allocate(l, 0, [&](da::Patch<3> const& p, std::size_t) {
    auto d = make(p);
    damn::core::for_each_point(p.box, [&](auto const& c) {
      for (auto const site : kv::sites_of_cell)
        value(d.u[site], p.box, c) = s.init(dk::key_of(c, site));
    });
    return d;
  });
  kv::exchange(l, 0, u_names);

  if (refined) {
    // a rank's own tagged cells: those on its patches
    std::vector<kv::Point3> mine;
    for (auto const& c : s.tagged)
      for (auto const& p : l[0].local)
        if (damn::core::isIn(c, p.box)) mine.push_back(c);
    kv::set_finer_level(l, 0, kv::tagged_boxes(l, 0, mine, clustering), s.fine_patch_cells);
    kv::allocate(l, 1, [](da::Patch<3> const& p, std::size_t) { return make(p); });
    kv::refine(l, 1, dk::LinearRefine{}, da::LevelRegion::Interior, u_pairs);
    kv::exchange(l, 1, u_names);
  }

  Hooks hooks{l, s.model};
  da::TimeRefinement<3> stepping{{.time_ratio = kv::ratio}};
  for (int n = 0; n < s.steps; ++n) stepping.advance(l.h, hooks, n * s.dt, s.dt);
  return l;
}

// this rank's sites of every level, those under a finer level included
inline Sites owned(Levels const& l) {
  Sites out;
  for (auto const& level : l.levels)
    for (std::size_t i = 0; i < level.local.size(); ++i) {
      auto const& box = level.local[i].box;
      damn::core::for_each_point(box, [&](auto const& c) {
        for (auto const site : kv::sites_of_cell)
          if (level.keys.has(c, site)) {
            auto const k = dk::key_of(c, site);
            out[{level.number, k[0], k[1], k[2]}] = value(level.data.data[i].u[site], box, c);
          }
      });
    }
  return out;
}

inline bool same_bits(double const a, double const b) {
  return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

struct Reference {
  Sites sites;

  // got's sites in the reference with its bits: how many
  [[nodiscard]] std::size_t matching(Sites const& got) const {
    std::size_t n = 0;
    for (auto const& [k, v] : got) n += sites.contains(k) and same_bits(sites.at(k), v);
    return n;
  }
};

// a level 0 site's value by key: no symmetry, every key distinct
inline double hashed(dk::Key<3> const& k) {
  return 0.01 * (((7 * k[0] + 13 * k[1] + 29 * k[2]) % 17 + 17) % 17 + 17);
}

// a linear field (coarse h units)
inline double linear(double const x, double const y, double const z) {
  return 1.5 + 0.25 * x - 0.5 * y + 0.125 * z;
}

// periodic and bounded, refined around the middle
inline Setup periodic_setup() {
  return {.keys = {dk::Key<3>{std::array{-8, -6, -8}},
                   dk::Key<3>{std::array{7, 5, 7}},
                   {true, true, true}},
          .patch_cells = {8, 6, 4},
          .fine_patch_cells = {12, 12, 12},
          .tagged = {kv::Point3{std::array{-1, -1, -1}}, kv::Point3{std::array{0, 0, 0}}},
          .init = hashed,
          .model = {.D = 0.02, .s = 0},
          .steps = 4};
}

inline Setup bounded_setup() {
  auto s = periodic_setup();
  s.keys = {dk::Key<3>{std::array{-7, -6, -5}}, dk::Key<3>{std::array{7, 6, 5}}, {}};
  return s;
}

}  // namespace kelvin_levels_test

#endif  // EQUILIBRIUM_TST_DAMN_KELVIN_LEVELS_TEST_HPP
