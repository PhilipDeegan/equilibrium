// Comparison test: uniform BCC sphere vs adaptive lattice with the same
// initial conditions and info.max_level=0 must evolve identically.
// Also tests the can_split / max_level guard.

#include "eq/core/data/lattice/adaptive_lattice.hpp"
#include "eq/core/data/lattice/bcc_geometry.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>

using namespace equilibrium::dst;
using namespace equilibrium::dst::discrete;

static void check(bool cond, const char* msg) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    std::exit(1);
  }
}

static std::int32_t find_origin(AdaptiveCellCluster& cl) {
  std::int32_t origin = -1;
  double best = 1e30;
  for (std::int32_t g : cl.leaf_indices) {
    std::size_t s    = AdaptiveCellCluster::slab_of(g);
    std::size_t slot = AdaptiveCellCluster::slot_of(g);
    auto& slab = cl.slab(s);
    double r2 = slab.pos[0 * CellSlab::CAPACITY + slot] * slab.pos[0 * CellSlab::CAPACITY + slot]
              + slab.pos[1 * CellSlab::CAPACITY + slot] * slab.pos[1 * CellSlab::CAPACITY + slot]
              + slab.pos[2 * CellSlab::CAPACITY + slot] * slab.pos[2 * CellSlab::CAPACITY + slot];
    if (r2 < best) { best = r2; origin = g; }
  }
  return origin;
}

// ---------------------------------------------------------------------------
// test 1: max_level=0 blocks split; identical evolution
// ---------------------------------------------------------------------------

static void test_uniform_vs_adaptive_flat() {
  AdaptiveCellCluster cl_u, cl_a;
  bcc::build_bcc_sphere(cl_u, 3.0, 1.0, 0.0, 0);
  bcc::build_bcc_sphere(cl_a, 3.0, 1.0, 0.0, 0);
  // info.max_level=0 (default) — no splitting ever.

  std::int32_t origin = find_origin(cl_u);
  check(origin >= 0, "found origin cell");
  cl_u[origin].u = 1.0;
  cl_a[origin].u = 1.0;

  AdaptiveLattice lat;
  lat.clusters.clear();

  check(!lat.can_split(cl_a, origin), "can_split false at level 0 with max=0");

  bool threw = false;
  try {
    lat.split(cl_a, origin);
  } catch (std::logic_error const&) {
    threw = true;
  }
  check(threw, "split throws when max_level=0");

  // Wave equation: stable_dt ≈ 0.005; use dt=0.003 with 20 steps.
  double dt = 0.003;
  for (int i = 0; i < 20; ++i) lat.step(cl_u, dt);
  for (int i = 0; i < 20; ++i) lat.step(cl_a, dt);

  check(cl_u.leaf_indices.size() == cl_a.leaf_indices.size(), "same leaf count");
  for (std::size_t j = 0; j < cl_u.leaf_indices.size(); ++j) {
    double diff = std::fabs(cl_u[cl_u.leaf_indices[j]].u - cl_a[cl_a.leaf_indices[j]].u);
    check(diff < 1e-15, "u identical between uniform and flat adaptive");
  }

  std::printf("test_uniform_vs_adaptive_flat: ok, %zu leaves, u_origin=%.6f\n",
              cl_u.leaf_indices.size(), cl_u[origin].u);
}

// ---------------------------------------------------------------------------
// test 2: max_level=1 allows level-0 cells to split but not
//         the resulting level-1 children.
// ---------------------------------------------------------------------------

static void test_max_level_1() {
  AdaptiveCellCluster cl;
  bcc::build_bcc_sphere(cl, 3.0);
  cl.info.max_level = 1;

  // Find an interior cell.
  std::int32_t target = -1;
  for (std::int32_t g : cl.leaf_indices) {
    int n = 0;
    for (std::size_t k = 0; k < 14; ++k)
      if (cl[g].nb[k] >= 0) n++;
    if (n == 14) { target = g; break; }
  }
  check(target >= 0, "found interior cell");

  AdaptiveLattice lat;
  lat.clusters.clear();

  check(lat.can_split(cl, target), "can_split true at level 0 with max=1");

  auto result = lat.split(cl, target);
  bcc::wire_split_children(cl, target, result, 1.0);
  lat.rebuild_leaf_indices(cl);

  // Children are at level 1 — must not be splittable.
  for (std::uint8_t c = 0; c < 8; ++c) {
    std::int32_t cg = result.first_child + c;
    check(!lat.can_split(cl, cg), "child at level 1 cannot split (max=1)");
  }

  bool threw = false;
  try {
    lat.split(cl, result.first_child);
  } catch (std::logic_error const&) {
    threw = true;
  }
  check(threw, "split throws for level-1 child when max=1");

  std::printf("test_max_level_1: ok, %zu leaves\n", cl.leaf_indices.size());
}

// ---------------------------------------------------------------------------

int main() {
  test_uniform_vs_adaptive_flat();
  test_max_level_1();
  std::printf("ALL PASSED\n");
}
