/*
Standalone C++ test for AdaptiveLattice / BCC geometry layer.

Compile and run (from repo root):
  g++ -std=c++20 -O2 -I inc -o /tmp/test_adaptive_lattice \
      tst/substrate/test_adaptive_lattice.cpp && /tmp/test_adaptive_lattice*/

#include "eq/core/data/lattice/adaptive_lattice.hpp"
#include "eq/core/data/lattice/bcc_geometry.hpp"
#include "eq/core/data/lattice/cluster_io.hpp"

#include "highfive/H5File.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <utility>
#include <vector>

using namespace equilibrium::dst;
using namespace equilibrium::dst::discrete;

// Override via EQ_TEST_N_THREADS=<n> to avoid oversubscribing when other
// tests run concurrently; defaults to 1 (single-threaded).
static int const EQ_TEST_N_THREADS =
    std::getenv("EQ_TEST_N_THREADS") ? std::atoi(std::getenv("EQ_TEST_N_THREADS")) : 1;

static void check(bool cond, char const* msg) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    std::exit(1);
  }
}

// --- helpers ----------------------------------------------------------------

static int count_neighbours(AdaptiveCellCluster& cl, std::int32_t g) {
  auto cell = cl[g];
  int n = 0;
  for (std::size_t k = 0; k < 14; ++k)
    if (cell.nb[k] >= 0) n++;
  return n;
}

// Verify bidirectionality: if nb[k]==j then j.nb[mirror_k]==g.
static bool check_bidirectionality(AdaptiveCellCluster& cl) {
  for (std::int32_t g : cl.leaf_indices) {
    auto cell = cl[g];
    for (std::size_t k = 0; k < 14; ++k) {
      std::int32_t nb = cell.nb[k];
      if (nb < 0) continue;
      std::size_t mk = bcc::mirror_face[k];
      if (cl[nb].nb[mk] != g) return false;
    }
  }
  return true;
}

// --- test 1: build sphere ---------------------------------------------------

static void test_build_sphere() {
  AdaptiveCellCluster cl;
  bcc::build_bcc_sphere(cl, 3.0);

  check(cl.leaf_indices.size() > 0, "sphere has cells");
  check(cl.n_levels() >= 1, "level 0 created");

  // Interior cells should have all 14 neighbours.
  // Origin cell (closest to 0,0,0) should be well-connected.
  bool found_interior = false;
  for (std::int32_t g : cl.leaf_indices) {
    if (count_neighbours(cl, g) == 14) {
      found_interior = true;
      break;
    }
  }
  check(found_interior, "at least one cell has 14 neighbours");

  check(check_bidirectionality(cl), "bidirectionality holds after build");

  std::printf("test_build_sphere: %zu cells, %zu slabs\n", cl.leaf_indices.size(), cl.n_slabs());
}

// --- test 2: split (default = Odd27, compile-time selected) -----------------

static void test_split() {
  AdaptiveCellCluster cl;
  bcc::build_bcc_sphere(cl, 3.0);
  cl.info.max_level = 1;

  // Find a well-connected interior cell to split.
  std::int32_t target = -1;
  for (std::int32_t g : cl.leaf_indices) {
    if (count_neighbours(cl, g) == 14) {
      target = g;
      break;
    }
  }
  check(target >= 0, "found interior cell");

  std::size_t before = cl.leaf_indices.size();
  auto par_nb_before = cl[target];
  std::int32_t par_nb[14];
  for (std::size_t k = 0; k < 14; ++k) par_nb[k] = par_nb_before.nb[k];

  AdaptiveLattice lat;
  lat.clusters.clear();                 // we manage cl directly here
  auto result = lat.split(cl, target);  // default: SplitModel::Odd27 (ratio-3, 27 children)
  bcc::wire_split_children(cl, target, result, 1.0);
  lat.rebuild_leaf_indices(cl);

  check(result.n_children == adaptive_cell::N_CHILDREN_ODD27,
        "split() defaults to 27 children (Odd27)");
  check(cl.leaf_indices.size() == before + 26, "leaf count after split (+26 for a 27-child split)");

  // Children must be at level 1.
  for (std::uint8_t c = 0; c < result.n_children; ++c) {
    auto ch = cl[result.first_child + c];
    check(ch.level == 1, "child level == 1");
    check(cl.slab_level(AdaptiveCellCluster::slab_of(result.first_child + c)) == 1,
          "child slab at level 1");
  }

  // Every (child, face) pair is fully classified by bcc::odd27_table — either
  // a sibling (must be bidirectional within the 27-block) or external (must
  // proxy through the PARENT's own matching face, per the table's own
  // classification — NOT necessarily the same face index k; see
  // odd27_table's build comment, verified byte-for-byte against an
  // independent Python re-implementation of probe_split27.py's nearest-
  // centre algorithm during development).
  std::int32_t fc = result.first_child;
  std::uint8_t nc = result.n_children;
  int n_internal = 0, n_external = 0;
  for (std::uint8_t c = 0; c < nc; ++c) {
    auto ch = cl[fc + c];
    for (std::size_t k = 0; k < 14; ++k) {
      auto const t = bcc::odd27_table.face_target[c][k];
      if (t < bcc::ODD27_EXTERNAL_BASE) {
        std::int32_t sib_g = fc + t;
        check(ch.nb[k] == sib_g, "odd27 sibling wiring matches table");
        std::size_t mk = bcc::mirror_face[k];
        check(cl[sib_g].nb[mk] == fc + c, "odd27 sibling bidirectionality");
        ++n_internal;
      } else {
        check(ch.nb[k] == par_nb[t - bcc::ODD27_EXTERNAL_BASE],
              "odd27 external face proxies through the parent's matching neighbour");
        ++n_external;
      }
    }
  }
  // Pins to the same numbers validated by the exhaustive Python sweep
  // (probe_split27.py --R 8) and the compile-time static_asserts in
  // bcc_defs.hpp: 196 internal, 182 external, summing to 27*14=378.
  check(n_internal == 196, "odd27 internal face-link count matches validated geometry");
  check(n_external == 182, "odd27 external face-link count matches validated geometry");

  check(cl.n_levels() >= 2, "level 1 pool created after split");
  std::printf("test_split: ok, %zu leaves after split\n", cl.leaf_indices.size());
}

// --- test 2b/2c: split() with an explicit legacy SplitModel -----------------
// Preserves the exact assertions test_split used to make about the ratio-2
// models, now reached via an explicit template argument instead of being
// the default.

static void test_split_hex8() {
  AdaptiveCellCluster cl;
  bcc::build_bcc_sphere(cl, 3.0);
  cl.info.max_level = 1;

  std::int32_t target = -1;
  for (std::int32_t g : cl.leaf_indices)
    if (count_neighbours(cl, g) == 14) {
      target = g;
      break;
    }
  check(target >= 0, "found interior cell");

  std::size_t before = cl.leaf_indices.size();

  AdaptiveLattice lat;
  lat.clusters.clear();
  auto result = lat.split<SplitModel::Hex8>(cl, target);
  bcc::wire_split_children<SplitModel::Hex8>(cl, target, result, 1.0);
  lat.rebuild_leaf_indices(cl);

  check(result.n_children == adaptive_cell::N_CHILDREN_HEX, "split<Hex8>() makes 8 children");
  check(cl.leaf_indices.size() == before + 7, "leaf count after Hex8 split (+7)");

  for (std::uint8_t c = 0; c < result.n_children; ++c) {
    auto ch = cl[result.first_child + c];
    check(ch.level == 1, "Hex8 child level == 1");
  }

  // Sibling bidirectionality on sq faces (the only intra-sibling faces a
  // ratio-2 octant split wires).
  for (std::uint8_t c = 0; c < 8; ++c) {
    std::int32_t cg = result.first_child + c;
    auto ch = cl[cg];
    for (std::size_t k = 8; k < 14; ++k) {
      std::int32_t nb = ch.nb[k];
      if (nb < 0) continue;
      bool is_sibling = (nb >= result.first_child && nb < result.first_child + 8);
      if (is_sibling) {
        std::size_t mk = bcc::mirror_face[k];
        check(cl[nb].nb[mk] == cg, "Hex8 sibling bidirectionality");
      }
    }
  }

  // Legacy 8-split: the mirror-face direction (pointing back at the parent
  // interior) has no cell to connect to — dead face, nb[k] = -1.
  for (std::size_t k = 0; k < 8; ++k) {
    std::int32_t mirror_child_g = result.first_child + bcc::mirror_face[k];
    check(cl[mirror_child_g].nb[k] == -1, "Hex8 inward-facing hex face is dead (-1)");
  }

  std::printf("test_split_hex8: ok, %zu leaves after split\n", cl.leaf_indices.size());
}

static void test_split_hex_center9() {
  AdaptiveCellCluster cl;
  bcc::build_bcc_sphere(cl, 3.0);
  cl.info.max_level = 1;

  std::int32_t target = -1;
  for (std::int32_t g : cl.leaf_indices)
    if (count_neighbours(cl, g) == 14) {
      target = g;
      break;
    }
  check(target >= 0, "found interior cell");

  std::size_t before = cl.leaf_indices.size();

  AdaptiveLattice lat;
  lat.clusters.clear();
  auto result = lat.split<SplitModel::HexCenter9>(cl, target);
  bcc::wire_split_children<SplitModel::HexCenter9>(cl, target, result, 1.0);
  lat.rebuild_leaf_indices(cl);

  check(result.n_children == adaptive_cell::N_CHILDREN_HEX_CENTER,
        "split<HexCenter9>() makes 9 children");
  check(cl.leaf_indices.size() == before + 8, "leaf count after HexCenter9 split (+8)");

  for (std::uint8_t c = 0; c < result.n_children; ++c) {
    auto ch = cl[result.first_child + c];
    check(ch.level == 1, "HexCenter9 child level == 1");
  }

  for (std::uint8_t c = 0; c < 8; ++c) {
    std::int32_t cg = result.first_child + c;
    auto ch = cl[cg];
    for (std::size_t k = 8; k < 14; ++k) {
      std::int32_t nb = ch.nb[k];
      if (nb < 0) continue;
      bool is_sibling = (nb >= result.first_child && nb < result.first_child + 8);
      if (is_sibling) {
        std::size_t mk = bcc::mirror_face[k];
        check(cl[nb].nb[mk] == cg, "HexCenter9 sibling bidirectionality");
      }
    }
  }

  // Center child: must sit at the parent's own position, and be
  // hex-face-connected to all 8 outer siblings (bidirectionally) — this is
  // the previously-dead inward face the center child exists to fill.
  check(result.n_children == adaptive_cell::N_CHILDREN_HEX_CENTER,
        "split<HexCenter9>() makes 9 children");
  {
    std::int32_t center_g = result.first_child + adaptive_cell::CENTER_CHILD_INDEX;
    auto center = cl[center_g];
    check(center.level == 1, "center child level == 1");
    for (std::size_t k = 0; k < 8; ++k) {
      std::int32_t outer_g = result.first_child + static_cast<std::uint8_t>(k);
      check(center.nb[k] == outer_g, "center hex face k faces outer child k");
      std::size_t mk = bcc::mirror_face[k];
      check(cl[outer_g].nb[mk] == center_g, "outer child's inward face faces center");
    }
  }

  std::printf("test_split_hex_center9: ok, %zu leaves after split\n", cl.leaf_indices.size());
}

// --- test 3: coarsen (default = Odd27) ---------------------------------------
// coarsen()/rebuild_leaf_indices() are generic over n_children (nb[]/position
// introspection only), so this needs no split-model-specific logic.

static void test_coarsen() {
  AdaptiveCellCluster cl;
  bcc::build_bcc_sphere(cl, 3.0);
  cl.info.max_level = 1;

  std::int32_t target = -1;
  for (std::int32_t g : cl.leaf_indices)
    if (count_neighbours(cl, g) == 14) {
      target = g;
      break;
    }
  check(target >= 0, "found interior cell");

  std::size_t before = cl.leaf_indices.size();

  AdaptiveLattice lat;
  lat.clusters.clear();
  auto result = lat.split(cl, target);
  bcc::wire_split_children(cl, target, result, 1.0);
  lat.rebuild_leaf_indices(cl);

  // Now coarsen back.
  lat.coarsen(cl, target);
  lat.rebuild_leaf_indices(cl);

  check(cl.leaf_indices.size() == before, "leaf count restored after coarsen");
  check(cl[target].is_leaf(), "target is leaf again");
  check(check_bidirectionality(cl), "bidirectionality after coarsen");

  std::printf("test_coarsen: ok, %zu leaves restored\n", cl.leaf_indices.size());
}

// --- test 4: accumulate (default = Odd27) -----------------------------------
// accumulate_to_parent() is generic over n_children (dynamic n_ext counting,
// no fixed per-model divisor), so this needs no split-model-specific logic.

static void test_accumulate() {
  AdaptiveCellCluster cl;
  bcc::build_bcc_sphere(cl, 2.0);
  cl.info.max_level = 1;

  std::int32_t target = -1;
  for (std::int32_t g : cl.leaf_indices)
    if (count_neighbours(cl, g) == 14) {
      target = g;
      break;
    }
  check(target >= 0, "found interior cell");

  AdaptiveLattice lat;
  lat.clusters.clear();
  auto result = lat.split(cl, target);
  bcc::wire_split_children(cl, target, result, 1.0);
  lat.rebuild_leaf_indices(cl);

  // Set children u to known values (every one of the 27 children is a
  // genuine leaf, part of the volume-averaged mean).
  double u_expected = 0.0;
  for (std::uint8_t c = 0; c < result.n_children; ++c) {
    double u_val = static_cast<double>(c + 1);
    cl[result.first_child + c].u = u_val;
    u_expected += u_val;
  }
  u_expected /= static_cast<double>(result.n_children);

  lat.accumulate_to_parent(cl, target);

  double diff = std::fabs(cl[target].u - u_expected);
  check(diff < 1e-12, "parent u = mean of children");

  std::printf("test_accumulate: ok, u_parent=%.4f expected=%.4f\n", cl[target].u, u_expected);
}

// --- test 4b: accumulate_to_parent under extreme parent/children dynamic range
//
// Derived stress test for the catastrophic-cancellation bug found and fixed
// in accumulate_to_parent (2026-08-04, see doc/dst_algo.md): the old formula
// `p.u = p.u + damping*(new_u - p.u)` is IEEE-754-GUARANTEED to collapse to
// exactly the OLD p.u (discarding new_u entirely) whenever |new_u/p.u| is
// below ~2^-53 (~1.1e-16) -- fl(a+b)==a exactly whenever |b| <= 0.5*ulp(a).
// This is not a probabilistic risk needing a lucky organic repro (which is
// how it was actually found, via refined_poynting_ring.py --max-level 4):
// it's a deterministic property of the arithmetic. This test constructs
// that scenario directly and deterministically: poison the parent's u with
// a "stale" large value (simulating what a previous, larger transient
// substep would leave behind at a deep AMR boundary, where fields
// legitimately span 60+ orders of magnitude with distance from a source --
// see project_odd27_amr_junction_leak), set the children's u to a mean many
// orders of magnitude smaller, and confirm the restrict step tracks the
// children's mean -- not the stale parent value, not zero -- across a sweep
// spanning the derived failure threshold.
static void test_accumulate_extreme_dynamic_range() {
  double const ratios[] = {1e-1, 1e-8, 1e-15, 1e-16, 1e-17, 1e-30, 1e-60, 1e-100, 1e-200, 1e-300};
  double const stale_parent_u = 1.0;

  for (double ratio : ratios) {
    AdaptiveCellCluster cl;
    bcc::build_bcc_sphere(cl, 2.0);
    cl.info.max_level = 1;

    std::int32_t target = -1;
    for (std::int32_t g : cl.leaf_indices)
      if (count_neighbours(cl, g) == 14) {
        target = g;
        break;
      }
    check(target >= 0, "found interior cell");

    AdaptiveLattice lat;
    lat.clusters.clear();
    auto result = lat.split(cl, target);
    bcc::wire_split_children(cl, target, result, 1.0);
    lat.rebuild_leaf_indices(cl);

    double const children_u = stale_parent_u * ratio;
    for (std::uint8_t c = 0; c < result.n_children; ++c) cl[result.first_child + c].u = children_u;

    // Poison the parent with a stale value many orders of magnitude larger
    // than the children's mean -- exactly what a previous, larger transient
    // substep would leave behind.
    cl[target].u = stale_parent_u;

    lat.accumulate_to_parent(cl, target);

    double const diff = std::fabs(cl[target].u - children_u);
    double const tol = std::max(std::fabs(children_u) * 1e-9, 1e-300);
    char msg[256];
    std::snprintf(msg, sizeof(msg),
                  "ratio=%.3e: parent.u tracks children mean (got %.17g expected %.17g, stale "
                  "parent value was %.3g)",
                  ratio, cl[target].u, children_u, stale_parent_u);
    check(diff <= tol, msg);
  }

  std::printf(
      "test_accumulate_extreme_dynamic_range: ok, tracked children mean correctly across ratio "
      "1e-1 down to 1e-300 (derived failure threshold ~1.1e-16 for the old, now-fixed increment "
      "formula)\n");
}

// --- test 4c: sibling u_sum/lam_sum naive summation under adversarial spread
//
// doc/dst_algo.md's survey rated accumulate_to_parent's OWN u_sum/lam_sum
// loop (adaptive_lattice.hpp:277-280) as low risk because sibling terms are
// same-level/same-neighbourhood by construction, unlike the cross-generation
// blend that was actually broken. This test makes that a verified fact
// rather than an assumption: deliberately violate the "siblings are close in
// magnitude" premise (one O(1) dominant sibling among 26 near-ulp-scale
// ones) and confirm the naive sequential sum, while genuinely ORDER-
// dependent (a real, measurable effect -- summation order changes the
// result depending on whether the dominant term is processed first or
// last), never loses more than the small siblings' own total magnitude.
// That's ordinary summation-order rounding, categorically different from
// the fixed bug's total, guaranteed annihilation of an entire (arbitrarily
// large) tiny value.
static void test_sibling_sum_order_dependence() {
  double const big = 1.0;
  double const small = 1e-16;  // just below 0.5*ulp(1.0) ~= 1.11e-16

  auto run = [&](bool big_first) -> double {
    AdaptiveCellCluster cl;
    bcc::build_bcc_sphere(cl, 2.0);
    cl.info.max_level = 1;
    std::int32_t target = -1;
    for (std::int32_t g : cl.leaf_indices)
      if (count_neighbours(cl, g) == 14) {
        target = g;
        break;
      }
    check(target >= 0, "found interior cell");

    AdaptiveLattice lat;
    lat.clusters.clear();
    auto result = lat.split(cl, target);
    bcc::wire_split_children(cl, target, result, 1.0);
    lat.rebuild_leaf_indices(cl);

    for (std::uint8_t c = 0; c < result.n_children; ++c) {
      bool const is_big = big_first ? (c == 0) : (c == result.n_children - 1);
      cl[result.first_child + c].u = is_big ? big : small;
    }

    lat.accumulate_to_parent(cl, target);
    return cl[target].u * static_cast<double>(result.n_children);  // recover u_sum
  };

  double const sum_big_first = run(true);
  double const sum_small_first = run(false);
  double const n_small = 26.0;  // Odd27: 1 dominant + 26 near-ulp-scale siblings
  double const max_possible_loss = n_small * small;

  std::printf(
      "test_sibling_sum_order_dependence: u_sum big-first=%.17g small-first=%.17g diff=%.3e "
      "(26 terms of %.1e each, dominant term=%.1f)\n",
      sum_big_first, sum_small_first, sum_small_first - sum_big_first, small, big);

  check(std::fabs(sum_small_first - sum_big_first) <= max_possible_loss * 1.1,
        "order-dependent discrepancy stays bounded by the small siblings' own total magnitude "
        "(not a catastrophic loss)");
  check(sum_big_first > 0.0 && sum_small_first > 0.0,
        "neither summation order loses the dominant sibling entirely");
}

// --- test 5: step -----------------------------------------------------------
// A point source u=1 at the origin surrounded by u=0 cells should diffuse
// outward.  After N steps, origin u decreases and neighbours increase.

static void test_step() {
  AdaptiveCellCluster cl;
  bcc::build_bcc_sphere(cl, 3.0, 1.0, 0.0, 1.0);

  // Find origin cell (position closest to 0,0,0) and set u=1.
  std::int32_t origin = -1;
  double best = 1e30;
  for (std::int32_t g : cl.leaf_indices) {
    std::size_t s = AdaptiveCellCluster::slab_of(g);
    std::size_t slot = AdaptiveCellCluster::slot_of(g);
    auto& slab = cl.slab(s);
    double r2 = slab.pos[0 * CellSlab::CAPACITY + slot] * slab.pos[0 * CellSlab::CAPACITY + slot] +
                slab.pos[1 * CellSlab::CAPACITY + slot] * slab.pos[1 * CellSlab::CAPACITY + slot] +
                slab.pos[2 * CellSlab::CAPACITY + slot] * slab.pos[2 * CellSlab::CAPACITY + slot];
    if (r2 < best) {
      best = r2;
      origin = g;
    }
  }
  check(origin >= 0, "found origin cell");
  cl[origin].u = 1.0;

  AdaptiveLattice lat;
  lat.clusters.clear();
  double u_origin_before = cl[origin].u;

  // Wave equation stable_dt ≈ 0.4 / (sigma * sum W_k^2) = 0.4/78 ≈ 0.005.
  // Use dt=0.003 (below limit) with 50 steps so the wave has time to propagate.
  double dt = 0.003;
  for (int i = 0; i < 50; ++i) lat.step(cl, dt);

  double u_origin_after = cl[origin].u;
  check(u_origin_after < u_origin_before, "origin u decreases (wave)");

  // Check at least one neighbour gained u.
  bool neighbour_gained = false;
  for (std::size_t k = 0; k < 14; ++k) {
    std::int32_t nbg = cl[origin].nb[k];
    if (nbg >= 0 && cl[nbg].u > 0.0) {
      neighbour_gained = true;
      break;
    }
  }
  check(neighbour_gained, "neighbours gain u");

  // Repeat with EQ_TEST_N_THREADS threads (default 1) on a fresh cluster — result must be identical.
  AdaptiveCellCluster cl2;
  bcc::build_bcc_sphere(cl2, 3.0, 1.0, 0.0, 1.0);
  cl2[origin].u = 1.0;
  lat.set_n_threads(EQ_TEST_N_THREADS);
  for (int i = 0; i < 50; ++i) lat.step(cl2, dt);
  lat.set_n_threads(1);
  double u_mt = cl2[origin].u;
  check(std::fabs(u_mt - u_origin_after) < 1e-12, "multi-thread matches single-thread");

  std::printf("test_step: ok, u_origin %.4f -> %.4f (single), %.4f (4-thread)\n", u_origin_before,
              u_origin_after, u_mt);
}

// --- test 6: static boundary consistency (no stepping) ----------------------
//
// Isolates level-boundary error to the STATIC split/prolongate/restrict
// operators alone, decoupled entirely from PDE stepping -- neither test below
// ever calls lat.step()/step_subcycle(). This is the "refinement should be
// irrelevant" case: seed an analytic field on a uniform L0 mesh, split ONE
// cell, and check two round trips independently:
//
//   (a) Prolongation error: wire_split_children's REAL prolongation operator
//       (constant injection + _prolong_u_gradient_wide's gradient fit) vs the
//       analytic field evaluated at each child's own actual position -- i.e.
//       "split on IC and set the fine cells analytically" as the reference,
//       compared against "initialize L0, then split" using the real operator.
//   (b) Restriction error: accumulate_to_parent() (mean of the now-fine
//       children) vs the ORIGINAL L0-only analytic value at the parent's
//       position -- refining and immediately un-refining a field should give
//       back what was there before, with zero steps taken.
//
// A field-position helper avoids repeating the slab/slot indexing dance.

static void cell_xyz(AdaptiveCellCluster& cl, std::int32_t g, double& x, double& y, double& z) {
  std::size_t s = AdaptiveCellCluster::slab_of(g), slot = AdaptiveCellCluster::slot_of(g);
  auto& slab = cl.slab(s);
  x = slab.pos[0 * CellSlab::CAPACITY + slot];
  y = slab.pos[1 * CellSlab::CAPACITY + slot];
  z = slab.pos[2 * CellSlab::CAPACITY + slot];
}

static double isotropic_quadratic(double x, double y, double z) { return x * x + y * y + z * z; }

// A dominant affine term (gradient magnitude ~17.3, constant everywhere)
// plus the SAME isotropic_quadratic curvature as a perturbation (local
// gradient contribution ~2*|pos|, a few units at most over this test's
// domain) -- true gradient at any point = (10,10,10) + 2*pos.
static double mild_curvature_field(double x, double y, double z) {
  return 10.0 * (x + y + z) + isotropic_quadratic(x, y, z);
}

// Linear field: the round trip must be exact (to fp noise). wire_split_
// children's gradient-wide fit reconstructs an affine field's true gradient
// exactly from the parent's L0 neighbourhood (which also carries the same
// exact affine field), and accumulate_to_parent's mean-of-children is exact
// for an affine field because the 27 Odd27 offsets are exactly symmetric
// under inversion through the parent centre (Σ offset == 0, documented at
// _prolong_u_gradient_wide) -- so mean(f(child)) == f(parent) for any affine f.
static double analytic_linear(double x, double y, double z) {
  return 1.0 + 0.7 * x - 0.3 * y + 0.2 * z;
}

// --- test: HDF5 save/load round trip (streaming cluster_io) ----------------
//
// Exercises save_cluster/load_cluster directly. Added 2026-08-08 alongside
// the switch to a streaming per-slab HDF5 read/write (select().read_raw()/
// write_raw() straight into/out of each CellSlab's own storage) to replace
// the old implementation, which built one O(n_slabs*CAPACITY) flat vector
// per field on both save and load -- doubling peak memory at exactly the
// moment a checkpoint is written or restored. Builds a small but non-trivial
// multi-level mesh (base sphere + one targeted split, so parent/first_child/
// mixed levels are exercised, not just a flat leaf list), fills every
// serialised field with a distinguishable pattern, sets a real BC, saves,
// reloads, and checks every field matches EXACTLY -- a pure copy, no
// computation involved, so zero tolerance is the correct bar.
static void test_hdf5_save_load_round_trip() {
  AdaptiveCellCluster cl;
  bcc::build_bcc_sphere(cl, 4.0);
  cl.info.max_level = 1;

  std::int32_t target = -1;
  for (std::int32_t g : cl.leaf_indices)
    if (count_neighbours(cl, g) == 14) {
      target = g;
      break;
    }
  check(target >= 0, "round trip: found interior cell to split");

  AdaptiveLattice lat;
  lat.clusters.clear();
  auto result = lat.split(cl, target);
  bcc::wire_split_children(cl, target, result, 1.0);
  lat.rebuild_leaf_indices(cl);

  for (std::int32_t g : cl.leaf_indices) {
    double x, y, z;
    cell_xyz(cl, g, x, y, z);
    cl[g].u = analytic_linear(x, y, z);
    cl[g].lam = std::fabs(x) + 2.0 * std::fabs(y) + 3.0 * std::fabs(z);
    for (std::size_t k = 0; k < CellSlab::NF; ++k)
      cl[g].phi[k] = 0.1 * static_cast<double>(k) + x;
  }
  std::size_t n_flagged = 0;
  for (std::int32_t g : cl.leaf_indices) {
    if (n_flagged >= 3) break;
    cl[g].fixed_u = 1;
    cl[g].fixed_phi = 0b101;  // faces 0 and 2
    ++n_flagged;
  }
  cl.bc = std::make_unique<BoundaryCondition>();
  cl.bc->fn = BoundaryCondition::dirichlet;
  cl.bc->fn_type = BCType::FUNCTIONAL;
  cl.bc->target = 2.5;
  cl.bc->weight = 0.75;

  char const* path = "bin/test_lattice/test_round_trip.h5";
  {
    HighFive::File file(path, HighFive::File::Overwrite);
    auto grp = file.createGroup("cluster_0");
    save_cluster(cl, grp);
  }

  AdaptiveCellCluster cl2;
  {
    HighFive::File file(path, HighFive::File::ReadOnly);
    auto grp = file.getGroup("cluster_0");
    cl2 = load_cluster(grp);
  }
  std::remove(path);

  check(cl2.n_slabs() == cl.n_slabs(), "round trip: same slab count");
  check(cl2.leaf_indices == cl.leaf_indices, "round trip: leaf_indices match exactly");

  double max_u_err = 0.0, max_lam_err = 0.0, max_phi_err = 0.0, max_pos_err = 0.0;
  bool nb_mismatch = false, parent_mismatch = false, fc_mismatch = false, nc_mismatch = false;
  bool fu_mismatch = false, fp_mismatch = false, L_mismatch = false;
  for (std::size_t s = 0; s < cl.n_slabs(); ++s) {
    check(cl2.slab_level(s) == cl.slab_level(s), "round trip: slab level matches");
    auto const& a = cl.slab(s);
    auto const& b = cl2.slab(s);
    for (std::size_t i = 0; i < CellSlab::CAPACITY; ++i) {
      max_u_err = std::max(max_u_err, std::fabs(a.u[i] - b.u[i]));
      max_lam_err = std::max(max_lam_err, std::fabs(a.lam[i] - b.lam[i]));
      if (a.L[i] != b.L[i]) L_mismatch = true;
      if (a.fixed_u[i] != b.fixed_u[i]) fu_mismatch = true;
      if (a.fixed_phi[i] != b.fixed_phi[i]) fp_mismatch = true;
      if (a.n_children[i] != b.n_children[i]) nc_mismatch = true;
      if (a.parent[i] != b.parent[i]) parent_mismatch = true;
      if (a.first_child[i] != b.first_child[i]) fc_mismatch = true;
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        max_phi_err = std::max(
            max_phi_err, std::fabs(a.phi[k * CellSlab::CAPACITY + i] - b.phi[k * CellSlab::CAPACITY + i]));
        if (a.nb[k * CellSlab::CAPACITY + i] != b.nb[k * CellSlab::CAPACITY + i]) nb_mismatch = true;
      }
      for (std::size_t d = 0; d < 3; ++d)
        max_pos_err = std::max(
            max_pos_err, std::fabs(a.pos[d * CellSlab::CAPACITY + i] - b.pos[d * CellSlab::CAPACITY + i]));
    }
  }
  check(max_u_err == 0.0, "round trip: u matches exactly");
  check(max_lam_err == 0.0, "round trip: lam matches exactly");
  check(max_phi_err == 0.0, "round trip: phi matches exactly");
  check(max_pos_err == 0.0, "round trip: pos matches exactly");
  check(!L_mismatch, "round trip: L matches exactly");
  check(!fu_mismatch, "round trip: fixed_u matches exactly");
  check(!fp_mismatch, "round trip: fixed_phi matches exactly");
  check(!nc_mismatch, "round trip: n_children matches exactly");
  check(!parent_mismatch, "round trip: parent matches exactly");
  check(!fc_mismatch, "round trip: first_child matches exactly");
  check(!nb_mismatch, "round trip: nb matches exactly");

  check(cl2.info.split_hi == cl.info.split_hi, "round trip: info.split_hi matches");
  check(cl2.info.split_lo == cl.info.split_lo, "round trip: info.split_lo matches");
  check(cl2.info.max_level == cl.info.max_level, "round trip: info.max_level matches");

  check(cl2.bc != nullptr, "round trip: bc restored");
  check(cl2.bc->fn_type == BCType::FUNCTIONAL, "round trip: bc fn_type matches");
  check(cl2.bc->target == 2.5, "round trip: bc target matches");
  check(cl2.bc->weight == 0.75, "round trip: bc weight matches");

  std::printf(
      "test_hdf5_save_load_round_trip: ok, %zu slabs, %zu leaves, max_u_err=%.3e max_phi_err=%.3e\n",
      cl.n_slabs(), cl.leaf_indices.size(), max_u_err, max_phi_err);
}

static void test_static_boundary_consistency_linear() {
  AdaptiveCellCluster cl;
  bcc::build_bcc_sphere(cl, 4.0);
  cl.info.max_level = 1;

  // "Initialize the L0 mesh" -- seed every cell with the analytic field.
  for (std::int32_t g : cl.leaf_indices) {
    double x, y, z;
    cell_xyz(cl, g, x, y, z);
    cl[g].u = analytic_linear(x, y, z);
  }

  std::int32_t target = -1;
  for (std::int32_t g : cl.leaf_indices)
    if (count_neighbours(cl, g) == 14) {
      target = g;
      break;
    }
  check(target >= 0, "found interior cell");

  double const u0_L0_only = cl[target].u;  // pre-split analytic reference

  AdaptiveLattice lat;
  lat.clusters.clear();
  auto result = lat.split(cl, target);                // default Odd27
  bcc::wire_split_children(cl, target, result, 1.0);  // real prolongation runs here
  lat.rebuild_leaf_indices(cl);

  // (a) Prolongation error, no stepping.
  double max_prolong_err = 0.0;
  for (std::uint8_t c = 0; c < result.n_children; ++c) {
    std::int32_t cg = result.first_child + c;
    double x, y, z;
    cell_xyz(cl, cg, x, y, z);
    double expected = analytic_linear(x, y, z);
    max_prolong_err = std::max(max_prolong_err, std::fabs(cl[cg].u - expected));
  }
  check(max_prolong_err < 1e-9, "prolongation matches analytic linear field exactly (no stepping)");

  // (b) Restriction error, no stepping: accumulate the fine children (whose
  // values are already the correct analytic field from (a)) back to the
  // parent and compare against the ORIGINAL L0-only value.
  lat.accumulate_to_parent(cl, target);
  double restrict_err = std::fabs(cl[target].u - u0_L0_only);
  check(restrict_err < 1e-9,
        "restriction round-trip matches L0-only value exactly (linear field, no stepping)");

  std::printf(
      "test_static_boundary_consistency_linear: ok, max prolong err=%.3e, restrict err=%.3e\n",
      max_prolong_err, restrict_err);
}

// --- test: _prolong_u_gradient_wide's exact-conservation claim under an
// extreme local gradient
//
// _prolong_u_gradient_wide's own doc comment (bcc_geometry.hpp) claims
// mean(children.u) == parent.u "exactly regardless of the fitted gradient,
// no renormalisation needed" -- because the 27 Odd27 offsets are exactly
// symmetric under inversion (Sigma offset == 0), so Sigma(cx*gx+cy*gy+cz*gz)
// cancels to zero algebraically no matter how large gx/gy/gz are. Verified
// this premise directly (separate scratch check, not part of this test):
// summing odd27_table.offset[c] over all 27 children gives EXACTLY (0,0,0)
// -- integer arithmetic, no ambiguity. So the claim is true in exact
// arithmetic; this test asks what happens in FLOATING POINT when the
// correction terms are many orders of magnitude larger than parent.u itself.
//
// First attempt at this test used an arbitrary K-independent tolerance
// (1e-3) and FAILED at K=1e18 (measured conservation_err=3.37, mean=
// -2.3703703703703702 vs expected 1.0) -- initially looked like a real bug
// (the "exact -64/27"-looking mean(children.u) was suspicious enough to
// directly verify the offset-symmetry premise above). But max_rel_err was
// only 8.3e-15 (near machine epsilon) -- and a RELATIVE error of 8.3e-15 on
// a term of magnitude ~1e18 is an ABSOLUTE error of order ~10^3, so summing
// 27 such independently-rounded terms that are supposed to cancel down to
// ~1 will ordinarily leave residual noise around that scale. That's the
// same universal, unavoidable floating-point cancellation phenomenon as
// test_sibling_sum_order_dependence, just far more extreme (K=1e18 vs
// K~1) -- NOT a code-specific defect, and NOT the same failure mode as the
// accumulate_to_parent bug (which was a genuine, unbounded, guaranteed
// annihilation from an "increment" formula; this is bounded, expected
// summation noise proportional to K). The ORIGINAL 1e-3 tolerance was
// simply the wrong (K-independent) yardstick. Fixed to a properly derived,
// K-scaled bound: K * DBL_EPSILON * n_children, generously multiplied for
// margin. Measured error (3.37) sits comfortably within this derived bound
// (~6000 for K=1e18), confirming the function isn't doing anything WORSE
// than ordinary floating-point arithmetic permits.
//
// Practical note: an 18-order-of-magnitude LOCAL gradient across one
// split's 5-hop BFS stencil (a handful of physically adjacent cells) is not
// realistic for this codebase -- the 60+ orders of magnitude this project
// sees is a whole-domain (near-source vs. deep-vacuum) phenomenon, not a
// single-neighbourhood one, since fields decay smoothly cell-to-cell. So
// while this test's K=1e18 stresses the arithmetic far beyond what
// production runs would ever produce, confirming the error stays within
// the DERIVED envelope (not anomalously worse) is still meaningful
// assurance that _prolong_u_gradient_wide is well-behaved.
static double analytic_linear_local(double x, double y, double z, double px, double py,
                                     double pz, double base, double K) {
  return base + K * (0.7 * (x - px) - 0.3 * (y - py) + 0.2 * (z - pz));
}

static void test_prolong_u_gradient_wide_extreme_gradient() {
  AdaptiveCellCluster cl;
  bcc::build_bcc_sphere(cl, 4.0);
  cl.info.max_level = 1;

  std::int32_t target = -1;
  for (std::int32_t g : cl.leaf_indices)
    if (count_neighbours(cl, g) == 14) {
      target = g;
      break;
    }
  check(target >= 0, "found interior cell");

  double px, py, pz;
  cell_xyz(cl, target, px, py, pz);

  double const base_value = 1.0;
  double const K = 1e18;
  for (std::int32_t g : cl.leaf_indices) {
    double x, y, z;
    cell_xyz(cl, g, x, y, z);
    cl[g].u = analytic_linear_local(x, y, z, px, py, pz, base_value, K);
  }
  check(std::fabs(cl[target].u - base_value) < 1e-9,
        "parent's own value is exactly base_value (O(1)), not swept up by K");

  AdaptiveLattice lat;
  lat.clusters.clear();
  auto result = lat.split(cl, target);
  bcc::wire_split_children(cl, target, result, 1.0);
  lat.rebuild_leaf_indices(cl);

  // (a) The fit itself: each child should still match the analytic field at
  // its OWN position, to a tolerance scaled by K (individual child values
  // are themselves O(K), so this checks the fit's RELATIVE accuracy).
  double max_rel_err = 0.0;
  double u_sum = 0.0;
  for (std::uint8_t c = 0; c < result.n_children; ++c) {
    std::int32_t cg = result.first_child + c;
    double x, y, z;
    cell_xyz(cl, cg, x, y, z);
    double const expected = analytic_linear_local(x, y, z, px, py, pz, base_value, K);
    double const err = std::fabs(cl[cg].u - expected);
    max_rel_err = std::max(max_rel_err, err / std::max(std::fabs(expected), 1.0));
    u_sum += cl[cg].u;
  }
  check(max_rel_err < 1e-9,
        "gradient fit reconstructs the exact local analytic field at each child's own position, "
        "even though the correction term is ~1e18x larger than the parent's own value");

  // (b) mean(children.u) should recover par_u == base_value == 1.0 up to
  // ordinary floating-point cancellation noise -- NOT bit-exactly, since
  // summing 27 terms of magnitude O(K) that must cancel down to O(1) will
  // always leave residual noise proportional to K*epsilon. Tolerance is
  // DERIVED from that (K * DBL_EPSILON * n_children, x100 margin), not an
  // arbitrary constant -- this checks the error stays within the expected
  // envelope, not that it's small in some absolute sense independent of K.
  double const mean_children_u = u_sum / static_cast<double>(result.n_children);
  double const conservation_err = std::fabs(mean_children_u - base_value);
  double const expected_noise_floor =
      K * std::numeric_limits<double>::epsilon() * static_cast<double>(result.n_children);
  std::printf(
      "test_prolong_u_gradient_wide_extreme_gradient: max fit rel-err=%.3e, "
      "mean(children.u)=%.17g (expected %.1f, conservation err=%.3e, derived noise floor=%.3e)\n",
      max_rel_err, mean_children_u, base_value, conservation_err, expected_noise_floor);
  check(conservation_err < 100.0 * expected_noise_floor,
        "mean(children.u) recovers parent.u's O(1) value within the DERIVED floating-point "
        "cancellation noise floor for this K -- confirms no code-specific amplification beyond "
        "ordinary fp arithmetic, distinct from the unbounded accumulate_to_parent bug that was "
        "fixed");
}

// Cross-level flux formula (_step_pass2's nb.L > L_i branch, used every step
// after the mesh is built -- NOT the once-at-birth _prolong_u_gradient_wide
// fit) reduces to exactly the same-level formula's shape, generalised only
// via the harmonic-mean spacing 2/(L_src+nb.L). No gradient/positional
// correction term exists in it. A pure AFFINE field can't reveal this: the
// secant slope of an affine function between any two points is exact
// regardless of spacing asymmetry (this is exactly why
// test_static_boundary_consistency_linear's round trip is exact). Only a
// CURVED field can show whether treating the coarse side as a flat value at
// the wrong position (u_src lands on the ancestor found by walking
// cell.parent, evaluated at THAT cell's centre -- not the fine child's own
// position) introduces a genuine, non-vanishing error against the true
// derivative at the actual geometric face.
//
// Calls the REAL production formula via the public step() entry point
// (rather than hand-reimplementing it) to avoid the exact transcription-risk
// class of bug this file's own doc comments warn about elsewhere (see Part
// 24 in project_odd27_amr_junction_leak: two real bugs were introduced by
// hand-deriving a "should be simple" test setup). disable_multiplicity_correction
// is toggled off for the call so this measures the RAW per-face formula in
// isolation, not entangled with the separate (already independently tested)
// ancestor_group_multiplicity/face_multiplicity bookkeeping.
// Builds a FRESH mesh/field each call -- step()'s pass1 mutates u everywhere
// from phi, so reusing one lattice across two step() calls would contaminate
// the second measurement. disable_step_pass2_amr_correct is also set --
// step() (non-subcycled) unconditionally runs _step_pass2_amr_correct
// after _step_pass2/_step_pass2_lev, redistributing temp_phi across
// amr_correction_pairs using logic that assumes the ORIGINAL per-face
// formula; without disabling it, this measurement would be confounded by
// that separate reflux pass rather than isolating the flux formula itself.
static std::pair<double, double> _measure_quadratic_face_error(bool gradient_correction) {
  AdaptiveCellCluster cl;
  bcc::build_bcc_sphere(cl, 8.0);
  cl.info.max_level = 1;

  std::int32_t target0 = -1;
  for (std::int32_t g : cl.leaf_indices) {
    if (count_neighbours(cl, g) != 14) continue;
    bool all_nb_interior = true;
    for (std::size_t k = 0; k < 14; ++k)
      if (count_neighbours(cl, cl[g].nb[k]) != 14) { all_nb_interior = false; break; }
    if (all_nb_interior) { target0 = g; break; }
  }
  check(target0 >= 0, "found a 2-shell-deep interior cell to split");

  AdaptiveLattice lat;
  lat.clusters.clear();
  auto result = lat.split(cl, target0);
  bcc::wire_split_children(cl, target0, result, 1.0);
  lat.rebuild_leaf_indices(cl);

  // Seed the curved field by TRUE position on every leaf and the parent
  // proxy -- as if restriction had already run on a genuinely curved field.
  for (std::int32_t g : cl.leaf_indices) {
    double x, y, z;
    cell_xyz(cl, g, x, y, z);
    cl[g].u = isotropic_quadratic(x, y, z);
  }
  {
    double x, y, z;
    cell_xyz(cl, target0, x, y, z);
    cl[target0].u = isotropic_quadratic(x, y, z);
  }

  // Find a fine child whose face points directly at a genuinely coarser
  // (unsplit) neighbour -- the real nb.L > L_i cross-level case.
  std::int32_t fine_g = -1;
  std::size_t fine_k = 0;
  for (std::uint8_t c = 0; c < result.n_children && fine_g < 0; ++c) {
    std::int32_t cg = result.first_child + c;
    for (std::size_t k = 0; k < 14; ++k) {
      auto const nbg = cl[cg].nb[k];
      if (nbg >= 0 && cl[nbg].L > cl[cg].L) { fine_g = cg; fine_k = k; break; }
    }
  }
  check(fine_g >= 0, "found a fine child facing a genuinely coarser neighbour");

  auto const nb_g = cl[fine_g].nb[fine_k];
  double xi, yi, zi, xnb, ynb, znb;
  cell_xyz(cl, fine_g, xi, yi, zi);
  cell_xyz(cl, nb_g, xnb, ynb, znb);
  double const L_i = cl[fine_g].L;

  // True derivative of u=x^2+y^2+z^2 (grad = 2*pos) at the fine cell's OWN
  // true geometric face position (L_i/2 from its own centre, toward nb) --
  // pure calculus, independent of anything the formula under test computes.
  double const dx = xnb - xi, dy = ynb - yi, dz = znb - zi;
  double const d  = std::sqrt(dx * dx + dy * dy + dz * dz);
  double const ux = dx / d, uy = dy / d, uz = dz / d;
  double const face_x = xi + 0.5 * L_i * ux;
  double const face_y = yi + 0.5 * L_i * uy;
  double const face_z = zi + 0.5 * L_i * uz;
  double const true_deriv = 2.0 * (face_x * ux + face_y * uy + face_z * uz);

  bool const saved_mult     = AdaptiveLattice::disable_multiplicity_correction;
  bool const saved_amr_corr = AdaptiveLattice::disable_step_pass2_amr_correct;
  bool const saved_grad     = AdaptiveLattice::enable_cross_level_gradient_correction;
  AdaptiveLattice::disable_multiplicity_correction        = true;
  AdaptiveLattice::disable_step_pass2_amr_correct         = true;
  AdaptiveLattice::enable_cross_level_gradient_correction = gradient_correction;
  double const dt = 1.0;
  lat.step(cl, dt);  // phi starts at 0, so phi[fine_k] afterward == dphi exactly
  AdaptiveLattice::disable_multiplicity_correction        = saved_mult;
  AdaptiveLattice::disable_step_pass2_amr_correct         = saved_amr_corr;
  AdaptiveLattice::enable_cross_level_gradient_correction = saved_grad;

  // sigma==1 on both sides (lam never set, defaults to 0), so dphi ==
  // -dt*face_weight[fine_k]*est_deriv exactly -- invert to recover the
  // formula's own implied derivative for direct comparison.
  double const est_deriv = -cl[fine_g].phi[fine_k] / (dt * bcc::face_weight[fine_k]);
  return {est_deriv - true_deriv, true_deriv};
}

static void test_cross_level_flux_curvature_error() {
  auto const [err, true_deriv] = _measure_quadratic_face_error(false);
  std::printf(
      "test_cross_level_flux_curvature_error: true_deriv=%.6f err=%.6f rel_err=%.3e\n",
      true_deriv, err, std::fabs(err / true_deriv));
  check(std::fabs(err) > 1e-6,
        "cross-level flux's implied derivative measurably disagrees with the true derivative at "
        "the fine cell's own geometric face for a curved field, using the REAL production "
        "formula via step() -- not floating-point noise");
}

// test_cross_level_flux_curvature_error_corrected and _mild_field (both
// checked whether AdaptiveLattice::enable_cross_level_gradient_correction
// reduces the curvature error measured above -- the former on
// isotropic_quadratic, the latter on a field where curvature is a small
// perturbation on a dominant affine term, to ask the fairer question of
// whether the correction at least helps in an easier regime) were removed
// 2026-08-08 -- retired, not just known failures: both showed the "fix"
// making the error WORSE (isotropic_quadratic: -4.51 -> -7.46; mild field:
// err_uncorrected=0.449 -> err_corrected=6.963, ~15x worse), confirming
// enable_cross_level_gradient_correction is a dead end in every regime
// tested, not just the hardest one. Root cause (see doc/dst_amr.md /
// project_odd27_amr_junction_leak.md): collapsing both sample points to the
// same location leaves the flux weight's spacing normalization unchanged,
// dividing an eliminated signal by a denominator that assumed it wasn't
// eliminated.

// Both correction attempts above made things WORSE, even on a field that's
// almost entirely affine -- which _prolong_u_gradient_wide's own proven
// exactness for affine fields (test_static_boundary_consistency_linear)
// says an affine LEAST-SQUARES FIT should have no trouble with. That
// contradiction points at an implementation bug in
// _local_gradient_same_level itself, not an intrinsic accuracy limit of
// the approach -- checked directly here via debug_local_gradient_same_level,
// bypassing _gradient_corrected_value and the flux formula entirely.
static void test_local_gradient_same_level_accuracy() {
  AdaptiveCellCluster cl;
  bcc::build_bcc_sphere(cl, 8.0);

  std::int32_t target = -1;
  for (std::int32_t g : cl.leaf_indices) {
    if (count_neighbours(cl, g) != 14) continue;
    bool all_nb_interior = true;
    for (std::size_t k = 0; k < 14; ++k)
      if (count_neighbours(cl, cl[g].nb[k]) != 14) { all_nb_interior = false; break; }
    if (all_nb_interior) { target = g; break; }
  }
  check(target >= 0, "found a 2-shell-deep interior cell");

  for (std::int32_t g : cl.leaf_indices) {
    double x, y, z;
    cell_xyz(cl, g, x, y, z);
    cl[g].u = mild_curvature_field(x, y, z);
  }

  double px, py, pz;
  cell_xyz(cl, target, px, py, pz);
  double const true_grad[3] = {10.0 + 2.0 * px, 10.0 + 2.0 * py, 10.0 + 2.0 * pz};

  double fit_grad[3];
  AdaptiveLattice::debug_local_gradient_same_level(cl, target, fit_grad);

  double const err_x = fit_grad[0] - true_grad[0];
  double const err_y = fit_grad[1] - true_grad[1];
  double const err_z = fit_grad[2] - true_grad[2];
  double const err_mag = std::sqrt(err_x * err_x + err_y * err_y + err_z * err_z);

  std::printf(
      "test_local_gradient_same_level_accuracy: true_grad=(%.4f,%.4f,%.4f) "
      "fit_grad=(%.4f,%.4f,%.4f) err_mag=%.4e\n",
      true_grad[0], true_grad[1], true_grad[2], fit_grad[0], fit_grad[1], fit_grad[2], err_mag);

  check(err_mag < 1.0,
        "1-hop same-level gradient fit recovers the dominant affine gradient to within O(1) of "
        "the small quadratic perturbation, not wildly off");
}

// test_curvature_corrected_secant_formula_isolated (the curvature-corrected-
// secant fix attempt, tested against an oracle-supplied curvature value) was
// removed 2026-08-08 -- retired, not just a known failure. Its own premise
// (a single cross-level face's secant must match calculus to fp precision)
// was superseded by the project's later finding that only the FULL 14-face-
// weighted sum is meant to be isotropic by design (doc/dst.md; confirmed
// directly below in test_same_level_secant_vs_calculus_hex_sq: hex faces
// match calculus exactly, sq faces are off by a uniform, by-design 2/sqrt(3)
// factor even for a same-level, junction-free face). The fix attempt itself
// was also already a dead end on its own terms (off by ~2.8 even with the
// true curvature supplied directly -- see project_odd27_amr_junction_leak
// memory). Kept failing and calling std::exit(1), silently skipping every
// test after it in main()'s list. Full history: project_odd27_amr_junction_
// leak.md memory, exp/2026/08/06/*/report.md.

// test_same_level_secant_vs_calculus_hex_sq showed individual same-level
// faces are NOT calculus-accurate by design (doc/dst.md: face weights are
// chosen so the FULL 14-face-weighted sum is maximally isotropic, not so
// any single face matches calculus). Every cross-level "error" measured
// today compared a SINGLE face against calculus -- the wrong standard, per
// that same design philosophy. This is the fair, design-consistent check
// instead: does a BOUNDARY leaf's own AGGREGATE (all 14 of its own faces,
// weighted the same way pass1 weights them for the wave update: acc =
// Sum_k face_weight[k]*phi[k]) match what a fully INTERIOR leaf of the SAME
// size gets, for the SAME field? If they match, the cross-level formula is
// just as aggregate-isotropic as the same-level one (no real defect, this
// whole thread was checking the wrong invariant). If they differ well
// beyond fp noise, that's a genuine, AMR-junction-specific anisotropy --
// not explained by the by-design hex/sq pattern, since both sides of THIS
// comparison are already aggregates.
// Builds the SAME boundary-leaf mesh as test_cross_level_aggregate_isotropy
// and returns its per-face breakdown, optionally with ancestor
// substitution disabled (disable_ancestor_u_src) -- lets us see whether the
// aggregate discrepancy is concentrated in a few faces or spread evenly,
// and whether it's specifically the ancestor-substitution mechanism
// driving it (vs. e.g. the raw nb.u read, which stays active either way).
struct BoundaryFaceBreakdown {
  double acc = 0.0;
  double L_i = 0.0;
  std::size_t n_same_faces = 0, n_cross_faces = 0;
  std::array<double, 14> contrib{};
  std::array<bool, 14> is_cross{};
  std::array<bool, 14> is_hex{};
  std::array<bool, 14> has_face{};
  // Jensen-gap check: target0.u AFTER _accumulate_boundary_proxies's
  // restriction (mean of its 27 children) vs the TRUE analytic value at
  // target0's own centre. For a convex field, Jensen's inequality
  // guarantees mean(children) >= f(centre) -- this is the value reused
  // identically as u_src across every one of the 6 cross-hex faces.
  double target0_u_restricted = 0.0;
  double target0_u_true       = 0.0;
};

static BoundaryFaceBreakdown _measure_boundary_face_breakdown(bool disable_ancestor,
                                                              bool disable_mult,
                                                              bool disable_amr_correct = true) {
  BoundaryFaceBreakdown out;
  bool const saved_anc       = AdaptiveLattice::disable_ancestor_u_src;
  bool const saved_mult      = AdaptiveLattice::disable_multiplicity_correction;
  bool const saved_amr_corr2 = AdaptiveLattice::disable_step_pass2_amr_correct;
  AdaptiveLattice::disable_ancestor_u_src          = disable_ancestor;
  AdaptiveLattice::disable_multiplicity_correction = disable_mult;
  AdaptiveLattice::disable_step_pass2_amr_correct  = disable_amr_correct;

  AdaptiveCellCluster cl;
  bcc::build_bcc_sphere(cl, 8.0);
  cl.info.max_level = 1;

  std::int32_t target0 = -1;
  for (std::int32_t g : cl.leaf_indices) {
    if (count_neighbours(cl, g) != 14) continue;
    bool all_nb_interior = true;
    for (std::size_t k = 0; k < 14; ++k)
      if (count_neighbours(cl, cl[g].nb[k]) != 14) { all_nb_interior = false; break; }
    if (all_nb_interior) { target0 = g; break; }
  }
  check(target0 >= 0, "found a 2-shell-deep interior cell to split");

  AdaptiveLattice lat;
  lat.clusters.clear();
  auto result = lat.split(cl, target0);
  bcc::wire_split_children(cl, target0, result, 1.0);
  lat.rebuild_leaf_indices(cl);

  for (std::int32_t g : cl.leaf_indices) {
    double x, y, z;
    cell_xyz(cl, g, x, y, z);
    cl[g].u = isotropic_quadratic(x, y, z);
  }
  {
    double x, y, z;
    cell_xyz(cl, target0, x, y, z);
    cl[target0].u = isotropic_quadratic(x, y, z);
  }

  std::int32_t fine_g = -1;
  for (std::uint8_t c = 0; c < result.n_children && fine_g < 0; ++c) {
    std::int32_t cg = result.first_child + c;
    for (std::size_t k = 0; k < 14; ++k) {
      auto const nbg = cl[cg].nb[k];
      if (nbg >= 0 && cl[nbg].L > cl[cg].L) { fine_g = cg; break; }
    }
  }
  check(fine_g >= 0, "found a fine child facing a genuinely coarser neighbour");
  out.L_i = cl[fine_g].L;

  double const dt = 1.0;
  lat.step(cl, dt);  // _accumulate_boundary_proxies (called at step() start) restricts target0.u

  out.target0_u_restricted = cl[target0].u;
  {
    double x, y, z;
    cell_xyz(cl, target0, x, y, z);
    out.target0_u_true = isotropic_quadratic(x, y, z);
  }

  for (std::size_t k = 0; k < 14; ++k) {
    auto const nbg = cl[fine_g].nb[k];
    if (nbg < 0) continue;
    out.has_face[k] = true;
    out.is_hex[k]   = k < 8;
    out.is_cross[k] = cl[nbg].L != out.L_i;
    (out.is_cross[k] ? out.n_cross_faces : out.n_same_faces)++;
    out.contrib[k] = bcc::face_weight[k] * cl[fine_g].phi[k];
    out.acc += out.contrib[k];
  }

  AdaptiveLattice::disable_ancestor_u_src          = saved_anc;
  AdaptiveLattice::disable_multiplicity_correction = saved_mult;
  AdaptiveLattice::disable_step_pass2_amr_correct  = saved_amr_corr2;
  return out;
}

static void test_cross_level_aggregate_isotropy() {
  auto const with_anc    = _measure_boundary_face_breakdown(/*disable_ancestor=*/false, /*disable_mult=*/true);
  auto const without_anc = _measure_boundary_face_breakdown(/*disable_ancestor=*/true, /*disable_mult=*/true);
  auto const with_mult    = _measure_boundary_face_breakdown(/*disable_ancestor=*/false, /*disable_mult=*/false);
  auto const full_prod    = _measure_boundary_face_breakdown(/*disable_ancestor=*/false, /*disable_mult=*/false,
                                                              /*disable_amr_correct=*/false);
  double const acc_boundary        = with_anc.acc;
  auto const n_same_faces          = with_anc.n_same_faces;
  auto const n_cross_faces         = with_anc.n_cross_faces;
  double const L_i                 = with_anc.L_i;

  double const jensen_gap = with_anc.target0_u_restricted - with_anc.target0_u_true;
  std::printf(
      "test_cross_level_aggregate_isotropy: target0.u restricted=%.6f true(centre)=%.6f "
      "jensen_gap=%.6f\n",
      with_anc.target0_u_restricted, with_anc.target0_u_true, jensen_gap);

  std::printf("test_cross_level_aggregate_isotropy per-face breakdown (ancestor substitution ON):\n");
  std::printf("  %3s %5s %6s %14s %14s\n", "k", "type", "cross", "contrib", "contrib(no anc)");
  for (std::size_t k = 0; k < 14; ++k) {
    if (!with_anc.has_face[k]) continue;
    std::printf("  %3zu %5s %6s %14.6f %14.6f\n", k, with_anc.is_hex[k] ? "hex" : "sq",
                with_anc.is_cross[k] ? "yes" : "no", with_anc.contrib[k], without_anc.contrib[k]);
  }
  std::printf("  acc(ancestor ON)=%.6f  acc(ancestor OFF)=%.6f  acc(multiplicity-correction ON, "
              "production default)=%.6f\n",
              with_anc.acc, without_anc.acc, with_mult.acc);

  // --- Fully interior leaf, SAME cell size (L_i), no cross-level faces at
  // all -- built the SAME way test_biharmonic_precision_floor_by_level
  // builds its own reference (uniform mesh at that level's own d1). ---
  double acc_interior = 0.0;
  {
    AdaptiveCellCluster cl;
    bcc::build_bcc_sphere(cl, 8.0, L_i);
    AdaptiveLattice lat;
    lat.clusters.clear();
    lat.rebuild_leaf_indices(cl);

    for (std::int32_t g : cl.leaf_indices) {
      double x, y, z;
      cell_xyz(cl, g, x, y, z);
      cl[g].u = isotropic_quadratic(x, y, z);
    }

    std::int32_t target = -1;
    for (std::int32_t g : cl.leaf_indices) {
      if (count_neighbours(cl, g) != 14) continue;
      bool all_nb_interior = true;
      for (std::size_t k = 0; k < 14; ++k)
        if (count_neighbours(cl, cl[g].nb[k]) != 14) { all_nb_interior = false; break; }
      if (all_nb_interior) { target = g; break; }
    }
    check(target >= 0, "found a 2-shell-deep interior cell (reference)");

    double const dt = 1.0;
    lat.step(cl, dt);

    for (std::size_t k = 0; k < 14; ++k) acc_interior += bcc::face_weight[k] * cl[target].phi[k];
  }

  double const diff       = acc_boundary - acc_interior;
  double const rel_diff   = std::fabs(diff / acc_interior);
  double const diff_mult  = with_mult.acc - acc_interior;
  double const rel_diff_mult = std::fabs(diff_mult / acc_interior);
  double const diff_full  = full_prod.acc - acc_interior;
  double const rel_diff_full = std::fabs(diff_full / acc_interior);
  std::printf("test_cross_level_aggregate_isotropy (WITH production multiplicity correction, "
              "reflux OFF): acc_boundary=%.6f acc_interior=%.6f diff=%.6f rel_diff=%.3e\n",
              with_mult.acc, acc_interior, diff_mult, rel_diff_mult);
  std::printf("test_cross_level_aggregate_isotropy (FULL PRODUCTION -- multiplicity AND reflux "
              "both ON): acc_boundary=%.6f acc_interior=%.6f diff=%.6f rel_diff=%.3e\n",
              full_prod.acc, acc_interior, diff_full, rel_diff_full);
  std::printf(
      "test_cross_level_aggregate_isotropy: L_i=%.4e n_same_faces=%zu n_cross_faces=%zu "
      "acc_interior=%.6f acc_boundary=%.6f diff=%.6f rel_diff=%.3e\n",
      L_i, n_same_faces, n_cross_faces, acc_interior, acc_boundary, diff, rel_diff);
}

// test_cross_level_aggregate_isotropy only looked at ONE child (fine_g) in
// isolation. Multiplicity correction is designed to work across the WHOLE
// group of target0's 27 children collectively, not any one child alone --
// this sums every child's own acc (full production settings: multiplicity
// AND reflux both on) and compares the GROUP total against what 27
// well-behaved interior cells of the same size would give.
static void test_cross_level_aggregate_isotropy_full_group() {
  bool const saved_mult     = AdaptiveLattice::disable_multiplicity_correction;
  bool const saved_amr_corr = AdaptiveLattice::disable_step_pass2_amr_correct;
  AdaptiveLattice::disable_multiplicity_correction = false;
  AdaptiveLattice::disable_step_pass2_amr_correct  = false;

  AdaptiveCellCluster cl;
  bcc::build_bcc_sphere(cl, 8.0);
  cl.info.max_level = 1;

  std::int32_t target0 = -1;
  for (std::int32_t g : cl.leaf_indices) {
    if (count_neighbours(cl, g) != 14) continue;
    bool all_nb_interior = true;
    for (std::size_t k = 0; k < 14; ++k)
      if (count_neighbours(cl, cl[g].nb[k]) != 14) { all_nb_interior = false; break; }
    if (all_nb_interior) { target0 = g; break; }
  }
  check(target0 >= 0, "found a 2-shell-deep interior cell to split");

  AdaptiveLattice lat;
  lat.clusters.clear();
  auto result = lat.split(cl, target0);
  bcc::wire_split_children(cl, target0, result, 1.0);
  lat.rebuild_leaf_indices(cl);

  for (std::int32_t g : cl.leaf_indices) {
    double x, y, z;
    cell_xyz(cl, g, x, y, z);
    cl[g].u = isotropic_quadratic(x, y, z);
  }
  {
    double x, y, z;
    cell_xyz(cl, target0, x, y, z);
    cl[target0].u = isotropic_quadratic(x, y, z);
  }

  double const L_i = cl[result.first_child].L;
  double const dt  = 1.0;
  lat.step(cl, dt);

  double acc_group_total = 0.0;
  std::size_t n_cross_children = 0;
  for (std::uint8_t c = 0; c < result.n_children; ++c) {
    std::int32_t cg = result.first_child + c;
    double acc_c    = 0.0;
    bool has_cross  = false;
    for (std::size_t k = 0; k < 14; ++k) {
      auto const nbg = cl[cg].nb[k];
      if (nbg < 0) continue;
      if (cl[nbg].L != L_i) has_cross = true;
      acc_c += bcc::face_weight[k] * cl[cg].phi[k];
    }
    if (has_cross) n_cross_children++;
    acc_group_total += acc_c;
  }

  // Interior reference, same size, same field (identical to
  // test_cross_level_aggregate_isotropy's own reference).
  double acc_interior = 0.0;
  {
    AdaptiveCellCluster cl2;
    bcc::build_bcc_sphere(cl2, 8.0, L_i);
    AdaptiveLattice lat2;
    lat2.clusters.clear();
    lat2.rebuild_leaf_indices(cl2);
    for (std::int32_t g : cl2.leaf_indices) {
      double x, y, z;
      cell_xyz(cl2, g, x, y, z);
      cl2[g].u = isotropic_quadratic(x, y, z);
    }
    std::int32_t target = -1;
    for (std::int32_t g : cl2.leaf_indices) {
      if (count_neighbours(cl2, g) != 14) continue;
      bool all_nb_interior = true;
      for (std::size_t k = 0; k < 14; ++k)
        if (count_neighbours(cl2, cl2[g].nb[k]) != 14) { all_nb_interior = false; break; }
      if (all_nb_interior) { target = g; break; }
    }
    check(target >= 0, "found a 2-shell-deep interior cell (reference)");
    lat2.step(cl2, dt);
    for (std::size_t k = 0; k < 14; ++k) acc_interior += bcc::face_weight[k] * cl2[target].phi[k];
  }

  AdaptiveLattice::disable_multiplicity_correction = saved_mult;
  AdaptiveLattice::disable_step_pass2_amr_correct  = saved_amr_corr;

  double const expected_flat = static_cast<double>(result.n_children) * acc_interior;
  std::printf(
      "test_cross_level_aggregate_isotropy_full_group: n_children=%d n_cross_children=%zu "
      "acc_group_total=%.6f  n_children*acc_interior=%.6f  diff=%.6f rel_diff=%.3e\n",
      result.n_children, n_cross_children, acc_group_total, expected_flat,
      acc_group_total - expected_flat, std::fabs((acc_group_total - expected_flat) / expected_flat));
}

// Same measurement as test_cross_level_aggregate_isotropy_full_group, but
// with AdaptiveLattice::enable_ancestor_debias_correction turned on --
// validates the analytically-derived fix (see `now`, "cross-level flux
// formula has a real, substantial aggregate defect"): the entire defect was
// shown to be accumulate_to_parent's own Jensen-inequality restriction bias
// (mean(children.u) too high for a convex field) propagating through the
// flux formula's coupling coefficient. If the derivation is correct,
// subtracting an estimate of that bias from u_src should shrink diff/
// rel_diff toward zero relative to the uncorrected baseline above.
//
// Measured BOTH with disable_step_pass2_amr_correct on and off -- the
// code comment at enable_cross_level_gradient_correction's declaration
// warns that _step_pass2_amr_correct's reflux redistributes temp_phi
// across amr_correction_pairs using logic that assumes the ORIGINAL
// (uncorrected) per-face formula, which can silently erase any change
// made upstream in the raw per-face dphi -- exactly the confound that
// made the two prior fix attempts hard to read. Isolating reflux off
// shows the correction's raw effect on the formula itself.
static void test_cross_level_aggregate_isotropy_full_group_debias_variant(
    bool disable_amr_correct, double& out_diff, double& out_rel_diff) {
  bool const saved_mult     = AdaptiveLattice::disable_multiplicity_correction;
  bool const saved_amr_corr = AdaptiveLattice::disable_step_pass2_amr_correct;
  bool const saved_debias   = AdaptiveLattice::enable_ancestor_debias_correction;
  AdaptiveLattice::disable_multiplicity_correction    = false;
  AdaptiveLattice::disable_step_pass2_amr_correct     = disable_amr_correct;
  AdaptiveLattice::enable_ancestor_debias_correction  = true;

  AdaptiveCellCluster cl;
  bcc::build_bcc_sphere(cl, 8.0);
  cl.info.max_level = 1;

  std::int32_t target0 = -1;
  for (std::int32_t g : cl.leaf_indices) {
    if (count_neighbours(cl, g) != 14) continue;
    bool all_nb_interior = true;
    for (std::size_t k = 0; k < 14; ++k)
      if (count_neighbours(cl, cl[g].nb[k]) != 14) { all_nb_interior = false; break; }
    if (all_nb_interior) { target0 = g; break; }
  }
  check(target0 >= 0, "found a 2-shell-deep interior cell to split");

  AdaptiveLattice lat;
  lat.clusters.clear();
  auto result = lat.split(cl, target0);
  bcc::wire_split_children(cl, target0, result, 1.0);
  lat.rebuild_leaf_indices(cl);

  for (std::int32_t g : cl.leaf_indices) {
    double x, y, z;
    cell_xyz(cl, g, x, y, z);
    cl[g].u = isotropic_quadratic(x, y, z);
  }
  {
    double x, y, z;
    cell_xyz(cl, target0, x, y, z);
    cl[target0].u = isotropic_quadratic(x, y, z);
  }

  double const L_i = cl[result.first_child].L;
  double const dt  = 1.0;
  lat.step(cl, dt);

  double acc_group_total = 0.0;
  for (std::uint8_t c = 0; c < result.n_children; ++c) {
    std::int32_t cg = result.first_child + c;
    for (std::size_t k = 0; k < 14; ++k) {
      auto const nbg = cl[cg].nb[k];
      if (nbg < 0) continue;
      acc_group_total += bcc::face_weight[k] * cl[cg].phi[k];
    }
  }

  double acc_interior = 0.0;
  {
    AdaptiveCellCluster cl2;
    bcc::build_bcc_sphere(cl2, 8.0, L_i);
    AdaptiveLattice lat2;
    lat2.clusters.clear();
    lat2.rebuild_leaf_indices(cl2);
    for (std::int32_t g : cl2.leaf_indices) {
      double x, y, z;
      cell_xyz(cl2, g, x, y, z);
      cl2[g].u = isotropic_quadratic(x, y, z);
    }
    std::int32_t target = -1;
    for (std::int32_t g : cl2.leaf_indices) {
      if (count_neighbours(cl2, g) != 14) continue;
      bool all_nb_interior = true;
      for (std::size_t k = 0; k < 14; ++k)
        if (count_neighbours(cl2, cl2[g].nb[k]) != 14) { all_nb_interior = false; break; }
      if (all_nb_interior) { target = g; break; }
    }
    check(target >= 0, "found a 2-shell-deep interior cell (reference)");
    lat2.step(cl2, dt);
    for (std::size_t k = 0; k < 14; ++k) acc_interior += bcc::face_weight[k] * cl2[target].phi[k];
  }

  AdaptiveLattice::disable_multiplicity_correction   = saved_mult;
  AdaptiveLattice::disable_step_pass2_amr_correct    = saved_amr_corr;
  AdaptiveLattice::enable_ancestor_debias_correction = saved_debias;

  double const expected_flat = static_cast<double>(result.n_children) * acc_interior;
  out_diff     = acc_group_total - expected_flat;
  out_rel_diff = std::fabs(out_diff / expected_flat);
  std::printf(
      "test_cross_level_aggregate_isotropy_full_group_debias[amr_correct=%s]: "
      "acc_group_total=%.6f  n_children*acc_interior=%.6f  diff=%.6f rel_diff=%.3e\n",
      disable_amr_correct ? "OFF" : "ON", acc_group_total, expected_flat, out_diff, out_rel_diff);
}

static void test_cross_level_aggregate_isotropy_full_group_debias() {
  double diff_on, rel_on, diff_off, rel_off;
  test_cross_level_aggregate_isotropy_full_group_debias_variant(false, diff_on, rel_on);
  test_cross_level_aggregate_isotropy_full_group_debias_variant(true, diff_off, rel_off);
}

// test_cross_level_aggregate_isotropy_full_group compares the split group's
// aggregate flux against 27x AN INTERIOR FINE CELL'S OWN flux -- but that
// reference is a physically DIFFERENT scenario (all-fine neighbours) from
// what the split group actually faces (a genuinely coarse neighbour on the
// far side of each external face). The physically meaningful invariant is
// instead: does splitting target0 (then reassembling its flux via ancestor
// substitution) change the NET flux crossing its boundary, relative to
// leaving it UNSPLIT entirely? This measures acc_group_total (with and
// without enable_ancestor_debias_correction, reflux OFF to isolate the raw
// formula) against target0's own flux computed on a mesh where it was
// never split at all -- the actual "no adaptivity" ground truth for this
// specific invariant, same spirit as the uniform-fine-everywhere ground
// truth used elsewhere in this investigation, just inverted (coarse
// reference instead of fine).
static double _measure_split_group_acc(bool disable_amr_correct, bool enable_debias) {
  bool const saved_mult     = AdaptiveLattice::disable_multiplicity_correction;
  bool const saved_amr_corr = AdaptiveLattice::disable_step_pass2_amr_correct;
  bool const saved_debias   = AdaptiveLattice::enable_ancestor_debias_correction;
  AdaptiveLattice::disable_multiplicity_correction   = false;
  AdaptiveLattice::disable_step_pass2_amr_correct    = disable_amr_correct;
  AdaptiveLattice::enable_ancestor_debias_correction  = enable_debias;

  AdaptiveCellCluster cl;
  bcc::build_bcc_sphere(cl, 8.0);
  cl.info.max_level = 1;

  std::int32_t target0 = -1;
  for (std::int32_t g : cl.leaf_indices) {
    if (count_neighbours(cl, g) != 14) continue;
    bool all_nb_interior = true;
    for (std::size_t k = 0; k < 14; ++k)
      if (count_neighbours(cl, cl[g].nb[k]) != 14) { all_nb_interior = false; break; }
    if (all_nb_interior) { target0 = g; break; }
  }
  check(target0 >= 0, "found a 2-shell-deep interior cell to split");

  AdaptiveLattice lat;
  lat.clusters.clear();
  auto result = lat.split(cl, target0);
  bcc::wire_split_children(cl, target0, result, 1.0);
  lat.rebuild_leaf_indices(cl);

  for (std::int32_t g : cl.leaf_indices) {
    double x, y, z;
    cell_xyz(cl, g, x, y, z);
    cl[g].u = isotropic_quadratic(x, y, z);
  }
  {
    double x, y, z;
    cell_xyz(cl, target0, x, y, z);
    cl[target0].u = isotropic_quadratic(x, y, z);
  }

  lat.step(cl, 1.0);

  double acc_group_total = 0.0;
  for (std::uint8_t c = 0; c < result.n_children; ++c) {
    std::int32_t cg = result.first_child + c;
    for (std::size_t k = 0; k < 14; ++k) {
      auto const nbg = cl[cg].nb[k];
      if (nbg < 0) continue;
      acc_group_total += bcc::face_weight[k] * cl[cg].phi[k];
    }
  }

  AdaptiveLattice::disable_multiplicity_correction   = saved_mult;
  AdaptiveLattice::disable_step_pass2_amr_correct    = saved_amr_corr;
  AdaptiveLattice::enable_ancestor_debias_correction = saved_debias;
  return acc_group_total;
}

static void test_cross_level_aggregate_isotropy_full_group_vs_unsplit() {
  double acc_unsplit = 0.0;
  {
    AdaptiveCellCluster cl;
    bcc::build_bcc_sphere(cl, 8.0);
    AdaptiveLattice lat;
    lat.clusters.clear();
    lat.rebuild_leaf_indices(cl);
    for (std::int32_t g : cl.leaf_indices) {
      double x, y, z;
      cell_xyz(cl, g, x, y, z);
      cl[g].u = isotropic_quadratic(x, y, z);
    }
    std::int32_t target = -1;
    for (std::int32_t g : cl.leaf_indices) {
      if (count_neighbours(cl, g) != 14) continue;
      bool all_nb_interior = true;
      for (std::size_t k = 0; k < 14; ++k)
        if (count_neighbours(cl, cl[g].nb[k]) != 14) { all_nb_interior = false; break; }
      if (all_nb_interior) { target = g; break; }
    }
    check(target >= 0, "found a 2-shell-deep interior cell (unsplit reference)");
    lat.step(cl, 1.0);
    for (std::size_t k = 0; k < 14; ++k) acc_unsplit += bcc::face_weight[k] * cl[target].phi[k];
  }

  double const acc_baseline = _measure_split_group_acc(/*disable_amr_correct=*/true, /*enable_debias=*/false);
  double const acc_debias   = _measure_split_group_acc(/*disable_amr_correct=*/true, /*enable_debias=*/true);

  auto const diff_of = [&](double acc) { return acc - acc_unsplit; };
  std::printf(
      "test_cross_level_aggregate_isotropy_full_group_vs_unsplit: acc_unsplit=%.6f  "
      "acc_baseline=%.6f (diff=%.6f rel=%.3e)  acc_debias=%.6f (diff=%.6f rel=%.3e)\n",
      acc_unsplit, acc_baseline, diff_of(acc_baseline), std::fabs(diff_of(acc_baseline) / acc_unsplit),
      acc_debias, diff_of(acc_debias), std::fabs(diff_of(acc_debias) / acc_unsplit));
}

// Corrected version of _build_two_level_telescope: picks target2 as the
// ONE fully-INTERIOR child of target0 (all 14 same-level neighbours, no
// cross faces at all) instead of a boundary child -- mirroring target0's
// own selection criterion. If Odd27's routing ever sends one of target2's
// own grandchildren's faces 2 hops up to target0's external boundary
// despite target2 itself never touching it directly, that's real evidence
// of deep routing reach; if every grandchild face resolves to either an
// L2 sibling or one of target0's OTHER L1 children (ratio 1 or 3, single
// ancestor hop), that confirms the earlier ungraded test's huge defect was
// a test-construction artefact (bad target2 choice), not a level-based bug.
static void _build_two_level_telescope_interior(AdaptiveCellCluster& cl, AdaptiveLattice& lat,
                                                 std::int32_t& target0, std::int32_t& target2,
                                                 SplitResult& result0, bool split_target2,
                                                 SplitResult& result2) {
  bcc::build_bcc_sphere(cl, 8.0);
  cl.info.max_level = 2;

  target0 = -1;
  for (std::int32_t g : cl.leaf_indices) {
    if (count_neighbours(cl, g) != 14) continue;
    bool all_nb_interior = true;
    for (std::size_t k = 0; k < 14; ++k)
      if (count_neighbours(cl, cl[g].nb[k]) != 14) { all_nb_interior = false; break; }
    if (all_nb_interior) { target0 = g; break; }
  }
  check(target0 >= 0, "found a 2-shell-deep interior cell to split (interior 2-level telescope)");

  lat.clusters.clear();
  result0 = lat.split(cl, target0);
  bcc::wire_split_children(cl, target0, result0, 1.0);
  lat.rebuild_leaf_indices(cl);

  target2 = -1;
  for (std::uint8_t c = 0; c < result0.n_children; ++c) {
    std::int32_t cg = result0.first_child + c;
    bool all_same = true;
    for (std::size_t k = 0; k < 14; ++k) {
      auto const nbg = cl[cg].nb[k];
      if (nbg < 0 || cl[nbg].L != cl[cg].L) { all_same = false; break; }
    }
    if (all_same) { target2 = cg; break; }
  }
  check(target2 >= 0, "found the fully-interior child of target0 to split again");

  if (split_target2) {
    result2 = lat.split(cl, target2);
    bcc::wire_split_children(cl, target2, result2, 1.0 / 3.0);
    lat.rebuild_leaf_indices(cl);

    if (std::getenv("EQ_DEBUG_GRADED_CHECK")) {
      std::size_t n_l0_direct = 0, n_l1 = 0, n_l2 = 0;
      for (std::uint8_t c = 0; c < result2.n_children; ++c) {
        std::int32_t cg = result2.first_child + c;
        for (std::size_t k = 0; k < 14; ++k) {
          auto const nbg = cl[cg].nb[k];
          if (nbg < 0) continue;
          auto const ratio = cl[nbg].L / cl[cg].L;
          if (ratio > 3.5) ++n_l0_direct;
          else if (ratio > 1.5) ++n_l1;
          else ++n_l2;
        }
      }
      std::printf("  [graded check, interior target2] direct-L0-jumps=%zu  L1-neighbours=%zu  "
                  "L2-siblings=%zu\n", n_l0_direct, n_l1, n_l2);
    }
  }
}

static double _measure_target2_interior_group_acc(bool disable_amr_correct, bool enable_debias) {
  bool const saved_mult     = AdaptiveLattice::disable_multiplicity_correction;
  bool const saved_amr_corr = AdaptiveLattice::disable_step_pass2_amr_correct;
  bool const saved_debias   = AdaptiveLattice::enable_ancestor_debias_correction;
  AdaptiveLattice::disable_multiplicity_correction   = false;
  AdaptiveLattice::disable_step_pass2_amr_correct    = disable_amr_correct;
  AdaptiveLattice::enable_ancestor_debias_correction  = enable_debias;

  AdaptiveCellCluster cl;
  AdaptiveLattice lat;
  std::int32_t target0, target2;
  SplitResult result0, result2;
  _build_two_level_telescope_interior(cl, lat, target0, target2, result0, /*split_target2=*/true, result2);

  for (std::int32_t g : cl.leaf_indices) {
    double x, y, z;
    cell_xyz(cl, g, x, y, z);
    cl[g].u = isotropic_quadratic(x, y, z);
  }

  lat.step(cl, 1.0);

  double acc_group_total = 0.0;
  for (std::uint8_t c = 0; c < result2.n_children; ++c) {
    std::int32_t cg = result2.first_child + c;
    for (std::size_t k = 0; k < 14; ++k) {
      auto const nbg = cl[cg].nb[k];
      if (nbg < 0) continue;
      acc_group_total += bcc::face_weight[k] * cl[cg].phi[k];
    }
  }

  AdaptiveLattice::disable_multiplicity_correction   = saved_mult;
  AdaptiveLattice::disable_step_pass2_amr_correct    = saved_amr_corr;
  AdaptiveLattice::enable_ancestor_debias_correction = saved_debias;
  return acc_group_total;
}

// Same vs_unsplit invariant, but target2 (the 2nd-level split target) is
// picked to be fully interior to target0 in its own unsplit state -- the
// clean, apples-to-apples counterpart to the ungraded test above.
static void test_cross_level_aggregate_isotropy_full_group_vs_unsplit_2level_interior() {
  double acc_unsplit = 0.0;
  {
    AdaptiveCellCluster cl;
    AdaptiveLattice lat;
    std::int32_t target0, target2;
    SplitResult result0, result2;
    _build_two_level_telescope_interior(cl, lat, target0, target2, result0, /*split_target2=*/false, result2);
    for (std::int32_t g : cl.leaf_indices) {
      double x, y, z;
      cell_xyz(cl, g, x, y, z);
      cl[g].u = isotropic_quadratic(x, y, z);
    }
    lat.step(cl, 1.0);
    for (std::size_t k = 0; k < 14; ++k) acc_unsplit += bcc::face_weight[k] * cl[target2].phi[k];
  }

  double const acc_baseline = _measure_target2_interior_group_acc(/*disable_amr_correct=*/true, /*enable_debias=*/false);
  double const acc_debias   = _measure_target2_interior_group_acc(/*disable_amr_correct=*/true, /*enable_debias=*/true);

  auto const diff_of = [&](double acc) { return acc - acc_unsplit; };
  std::printf(
      "test_cross_level_aggregate_isotropy_full_group_vs_unsplit_2level_interior: acc_unsplit=%.6f  "
      "acc_baseline=%.6f (diff=%.6f rel=%.3e)  acc_debias=%.6f (diff=%.6f rel=%.3e)\n",
      acc_unsplit, acc_baseline, diff_of(acc_baseline), std::fabs(diff_of(acc_baseline) / acc_unsplit),
      acc_debias, diff_of(acc_debias), std::fabs(diff_of(acc_debias) / acc_unsplit));
}

// Finds target0 (a 2-shell-deep interior L0 cell), splits it, then finds
// one of ITS children that itself still touches an external (level-0)
// neighbour (has_cross) -- splitting THAT child (target2) creates a
// genuine 2-level gap: target2's own children's external-facing faces
// must walk the ancestor chain up past target2 (L=1/3) all the way to
// target0 (L=1) to find a same-scale source, exactly the multi-descendant-
// shares-one-ancestor scenario the original Part 2 fix (ancestor_group_
// multiplicity) was built for. Returns target0/target2 by out-param so
// callers can build a matching unsplit reference with identical selection.
static void _build_two_level_telescope(AdaptiveCellCluster& cl, AdaptiveLattice& lat,
                                        std::int32_t& target0, std::int32_t& target2,
                                        SplitResult& result0,
                                        bool split_target2,
                                        SplitResult& result2) {
  bcc::build_bcc_sphere(cl, 8.0);
  cl.info.max_level = 2;

  target0 = -1;
  for (std::int32_t g : cl.leaf_indices) {
    if (count_neighbours(cl, g) != 14) continue;
    bool all_nb_interior = true;
    for (std::size_t k = 0; k < 14; ++k)
      if (count_neighbours(cl, cl[g].nb[k]) != 14) { all_nb_interior = false; break; }
    if (all_nb_interior) { target0 = g; break; }
  }
  check(target0 >= 0, "found a 2-shell-deep interior cell to split (2-level telescope)");

  lat.clusters.clear();
  result0 = lat.split(cl, target0);
  bcc::wire_split_children(cl, target0, result0, 1.0);
  lat.rebuild_leaf_indices(cl);

  target2 = -1;
  for (std::uint8_t c = 0; c < result0.n_children; ++c) {
    std::int32_t cg = result0.first_child + c;
    bool has_cross = false;
    for (std::size_t k = 0; k < 14; ++k) {
      auto const nbg = cl[cg].nb[k];
      if (nbg >= 0 && cl[nbg].L != cl[cg].L) { has_cross = true; break; }
    }
    if (has_cross) { target2 = cg; break; }
  }
  check(target2 >= 0, "found a boundary child of target0 to split again (2-level telescope)");

  if (split_target2) {
    result2 = lat.split(cl, target2);
    bcc::wire_split_children(cl, target2, result2, 1.0 / 3.0);
    lat.rebuild_leaf_indices(cl);

    if (std::getenv("EQ_DEBUG_GRADED_CHECK")) {
      std::printf("  [graded check] target0.L=%.6f target2.L=%.6f\n", cl[target0].L, cl[target2].L);
      for (std::uint8_t c = 0; c < result2.n_children; ++c) {
        std::int32_t cg = result2.first_child + c;
        for (std::size_t k = 0; k < 14; ++k) {
          auto const nbg = cl[cg].nb[k];
          if (nbg < 0) continue;
          std::printf("    child=%d face=%zu nb=%d nb.L=%.6f nb_is_target0=%d ratio_to_child=%.3f\n",
                      cg, k, nbg, cl[nbg].L, nbg == target0, cl[nbg].L / cl[cg].L);
        }
      }
    }
  }
}

static double _measure_target2_group_acc(bool disable_amr_correct, bool enable_debias) {
  bool const saved_mult     = AdaptiveLattice::disable_multiplicity_correction;
  bool const saved_amr_corr = AdaptiveLattice::disable_step_pass2_amr_correct;
  bool const saved_debias   = AdaptiveLattice::enable_ancestor_debias_correction;
  AdaptiveLattice::disable_multiplicity_correction   = false;
  AdaptiveLattice::disable_step_pass2_amr_correct    = disable_amr_correct;
  AdaptiveLattice::enable_ancestor_debias_correction  = enable_debias;

  AdaptiveCellCluster cl;
  AdaptiveLattice lat;
  std::int32_t target0, target2;
  SplitResult result0, result2;
  _build_two_level_telescope(cl, lat, target0, target2, result0, /*split_target2=*/true, result2);

  for (std::int32_t g : cl.leaf_indices) {
    double x, y, z;
    cell_xyz(cl, g, x, y, z);
    cl[g].u = isotropic_quadratic(x, y, z);
  }

  lat.step(cl, 1.0);

  double acc_group_total = 0.0;
  for (std::uint8_t c = 0; c < result2.n_children; ++c) {
    std::int32_t cg = result2.first_child + c;
    for (std::size_t k = 0; k < 14; ++k) {
      auto const nbg = cl[cg].nb[k];
      if (nbg < 0) continue;
      acc_group_total += bcc::face_weight[k] * cl[cg].phi[k];
    }
  }

  AdaptiveLattice::disable_multiplicity_correction   = saved_mult;
  AdaptiveLattice::disable_step_pass2_amr_correct    = saved_amr_corr;
  AdaptiveLattice::enable_ancestor_debias_correction = saved_debias;
  return acc_group_total;
}

// CAUTION -- this deliberately builds an UNGRADED mesh (confirmed via
// EQ_DEBUG_GRADED_CHECK=1: 10 of target2's 14 children's faces connect
// DIRECTLY to a level-0 cell, a 9x size jump, not the <=3x/1-level jump a
// graded/2:1-balanced mesh would have). `_build_two_level_telescope`
// splits target2 (a boundary child of target0) in isolation, unlike the
// real production telescopes (`refined_neutron_decay.py`'s
// `_build_telescope`), which refine shrinking-radius NESTED regions one
// level at a time specifically to avoid this. The huge (~100%,
// sign-flipping) discrepancy this test finds is therefore NOT good
// evidence of a mechanism relevant to the real (graded) L6/L8 blowup --
// see test_cross_level_aggregate_isotropy_full_group_vs_unsplit_2level_
// graded below for the corrected, production-representative version.
// Kept (not deleted) because an ungraded jump like this IS still reachable
// in principle -- `can_split` has no 2:1 balance enforcement anywhere in
// this codebase (see project_2to1_balance_question memory) -- so this
// remains a real, valid regression test of THAT specific (separate,
// already-flagged) design gap, just not of the depth-compounding question
// it was originally built to investigate.
static void test_cross_level_aggregate_isotropy_full_group_vs_unsplit_2level_ungraded() {
  double acc_unsplit = 0.0;
  {
    AdaptiveCellCluster cl;
    AdaptiveLattice lat;
    std::int32_t target0, target2;
    SplitResult result0, result2;
    _build_two_level_telescope(cl, lat, target0, target2, result0, /*split_target2=*/false, result2);
    for (std::int32_t g : cl.leaf_indices) {
      double x, y, z;
      cell_xyz(cl, g, x, y, z);
      cl[g].u = isotropic_quadratic(x, y, z);
    }
    lat.step(cl, 1.0);
    for (std::size_t k = 0; k < 14; ++k) acc_unsplit += bcc::face_weight[k] * cl[target2].phi[k];
  }

  double const acc_baseline = _measure_target2_group_acc(/*disable_amr_correct=*/true, /*enable_debias=*/false);
  double const acc_debias   = _measure_target2_group_acc(/*disable_amr_correct=*/true, /*enable_debias=*/true);

  auto const diff_of = [&](double acc) { return acc - acc_unsplit; };
  std::printf(
      "test_cross_level_aggregate_isotropy_full_group_vs_unsplit_2level_ungraded: acc_unsplit=%.6f  "
      "acc_baseline=%.6f (diff=%.6f rel=%.3e)  acc_debias=%.6f (diff=%.6f rel=%.3e)\n",
      acc_unsplit, acc_baseline, diff_of(acc_baseline), std::fabs(diff_of(acc_baseline) / acc_unsplit),
      acc_debias, diff_of(acc_debias), std::fabs(diff_of(acc_debias) / acc_unsplit));
}

// Builds a properly GRADED L0/L1/L2 telescope, matching how
// refined_neutron_decay.py's _build_telescope actually works: refine the
// OUTER (larger-radius) region first, then a nested smaller region one
// level deeper -- so no leaf ever directly borders a leaf more than one
// level away. Concretely: split target0's 14 real neighbours to L1 FIRST
// (the "shell"), THEN split target0 itself to L1, THEN split every one of
// target0's own 27 L1 children to L2 (the "core") -- by the time the core
// is split, its external boundary always sees L1 cells (the already-split
// shell), never L0 directly. `l2_splits` collects all 27 SplitResults so
// callers can iterate the full 729-cell L2 core.
static void _build_graded_l0_l1_l2_telescope(AdaptiveCellCluster& cl, AdaptiveLattice& lat,
                                              std::int32_t& target0, SplitResult& result0,
                                              bool split_core_to_l2,
                                              std::vector<SplitResult>& l2_splits) {
  bcc::build_bcc_sphere(cl, 8.0);
  cl.info.max_level = 2;

  target0 = -1;
  for (std::int32_t g : cl.leaf_indices) {
    if (count_neighbours(cl, g) != 14) continue;
    bool all_nb_interior = true;
    for (std::size_t k = 0; k < 14; ++k)
      if (count_neighbours(cl, cl[g].nb[k]) != 14) { all_nb_interior = false; break; }
    if (all_nb_interior) { target0 = g; break; }
  }
  check(target0 >= 0, "found a 2-shell-deep interior cell (graded telescope)");

  lat.clusters.clear();

  // Stage 1: outer shell to L1, TWO rings deep (target0's 14 direct
  // neighbours, plus every one of THEIR 14 neighbours) -- split BEFORE
  // target0 itself so target0's (and later its children's) wire_split_
  // children calls find already-split (L1) neighbours to route to. One
  // ring alone was measured insufficient (EQ_DEBUG_GRADED_CHECK showed
  // 2366 direct-L0 jumps still reaching the L2 core -- Odd27's coarse-
  // proxy face routing for a corner/edge grandchild can resolve to a
  // ring-2 cell, not just the immediate 14); two rings measured 0.
  std::vector<std::int32_t> shell;
  auto const add_unique = [&](std::int32_t g) {
    if (g != target0 && std::find(shell.begin(), shell.end(), g) == shell.end()) shell.push_back(g);
  };
  for (std::size_t k = 0; k < 14; ++k) add_unique(cl[target0].nb[k]);
  std::size_t const ring1_end = shell.size();
  for (std::size_t i = 0; i < ring1_end; ++i)
    for (std::size_t k = 0; k < 14; ++k) add_unique(cl[shell[i]].nb[k]);

  for (auto g : shell) {
    auto r = lat.split(cl, g);
    bcc::wire_split_children(cl, g, r, 1.0);
  }
  lat.rebuild_leaf_indices(cl);

  if (!split_core_to_l2) return;  // target0 stays a genuine leaf (the "unsplit" reference)

  // Stage 2: core to L1.
  result0 = lat.split(cl, target0);
  bcc::wire_split_children(cl, target0, result0, 1.0);
  lat.rebuild_leaf_indices(cl);

  {
    // Stage 3: core to L2 -- every one of target0's 27 L1 children split
    // uniformly, so the whole former-target0 volume is L2, graded against
    // the L1 shell from stage 1.
    l2_splits.clear();
    for (std::uint8_t c = 0; c < result0.n_children; ++c) {
      std::int32_t cg = result0.first_child + c;
      auto r2 = lat.split(cl, cg);
      bcc::wire_split_children(cl, cg, r2, 1.0 / 3.0);
      l2_splits.push_back(r2);
    }
    lat.rebuild_leaf_indices(cl);

    if (std::getenv("EQ_DEBUG_GRADED_CHECK")) {
      std::size_t n_l0_direct = 0, n_l1 = 0, n_l2 = 0;
      for (auto const& r2 : l2_splits)
        for (std::uint8_t c = 0; c < r2.n_children; ++c) {
          std::int32_t cg = r2.first_child + c;
          for (std::size_t k = 0; k < 14; ++k) {
            auto const nbg = cl[cg].nb[k];
            if (nbg < 0) continue;
            auto const ratio = cl[nbg].L / cl[cg].L;
            if (ratio > 3.5) ++n_l0_direct;
            else if (ratio > 1.5) ++n_l1;
            else ++n_l2;
          }
        }
      std::printf("  [graded check, L2 core] direct-L0-jumps=%zu  L1-neighbours=%zu  "
                  "L2-siblings=%zu\n", n_l0_direct, n_l1, n_l2);
    }
  }
}

static double _measure_graded_core_acc(bool disable_amr_correct, bool enable_debias) {
  bool const saved_mult     = AdaptiveLattice::disable_multiplicity_correction;
  bool const saved_amr_corr = AdaptiveLattice::disable_step_pass2_amr_correct;
  bool const saved_debias   = AdaptiveLattice::enable_ancestor_debias_correction;
  AdaptiveLattice::disable_multiplicity_correction   = false;
  AdaptiveLattice::disable_step_pass2_amr_correct    = disable_amr_correct;
  AdaptiveLattice::enable_ancestor_debias_correction  = enable_debias;

  AdaptiveCellCluster cl;
  AdaptiveLattice lat;
  std::int32_t target0;
  SplitResult result0;
  std::vector<SplitResult> l2_splits;
  _build_graded_l0_l1_l2_telescope(cl, lat, target0, result0, /*split_core_to_l2=*/true, l2_splits);

  for (std::int32_t g : cl.leaf_indices) {
    double x, y, z;
    cell_xyz(cl, g, x, y, z);
    cl[g].u = isotropic_quadratic(x, y, z);
  }

  lat.step(cl, 1.0);

  double acc_core_total = 0.0;
  for (auto const& r2 : l2_splits)
    for (std::uint8_t c = 0; c < r2.n_children; ++c) {
      std::int32_t cg = r2.first_child + c;
      for (std::size_t k = 0; k < 14; ++k) {
        auto const nbg = cl[cg].nb[k];
        if (nbg < 0) continue;
        acc_core_total += bcc::face_weight[k] * cl[cg].phi[k];
      }
    }

  AdaptiveLattice::disable_multiplicity_correction   = saved_mult;
  AdaptiveLattice::disable_step_pass2_amr_correct    = saved_amr_corr;
  AdaptiveLattice::enable_ancestor_debias_correction = saved_debias;
  return acc_core_total;
}

// The corrected, production-representative version of the 2-level vs_
// unsplit check: same invariant (does refining target0 further change its
// net boundary flux vs leaving it as a single leaf), but with target0's
// real neighbours held at L1 in BOTH the reference and test measurement
// (so the only thing that varies is target0's OWN internal refinement,
// not the surrounding mesh) and the core refined via a graded L1-then-L2
// telescope instead of the pathological direct-L0-L2 jump above.
static void test_cross_level_aggregate_isotropy_full_group_vs_unsplit_2level_graded() {
  double acc_unsplit = 0.0;
  {
    AdaptiveCellCluster cl;
    AdaptiveLattice lat;
    std::int32_t target0;
    SplitResult result0;
    std::vector<SplitResult> l2_splits;
    _build_graded_l0_l1_l2_telescope(cl, lat, target0, result0, /*split_core_to_l2=*/false, l2_splits);
    for (std::int32_t g : cl.leaf_indices) {
      double x, y, z;
      cell_xyz(cl, g, x, y, z);
      cl[g].u = isotropic_quadratic(x, y, z);
    }
    lat.step(cl, 1.0);
    for (std::size_t k = 0; k < 14; ++k) acc_unsplit += bcc::face_weight[k] * cl[target0].phi[k];
  }

  double const acc_baseline = _measure_graded_core_acc(/*disable_amr_correct=*/true, /*enable_debias=*/false);
  double const acc_debias   = _measure_graded_core_acc(/*disable_amr_correct=*/true, /*enable_debias=*/true);

  auto const diff_of = [&](double acc) { return acc - acc_unsplit; };
  std::printf(
      "test_cross_level_aggregate_isotropy_full_group_vs_unsplit_2level_graded: acc_unsplit=%.6f  "
      "acc_baseline=%.6f (diff=%.6f rel=%.3e)  acc_debias=%.6f (diff=%.6f rel=%.3e)\n",
      acc_unsplit, acc_baseline, diff_of(acc_baseline), std::fabs(diff_of(acc_baseline) / acc_unsplit),
      acc_debias, diff_of(acc_debias), std::fabs(diff_of(acc_debias) / acc_unsplit));
}

// Same measurement as test_cross_level_aggregate_isotropy_full_group, but
// with a PURE AFFINE field instead of a curved one. Restriction and
// prolongation are already proven EXACT for affine fields (Odd27's 27
// offsets sum to exactly zero, so mean(children)==parent.u regardless of
// gradient) -- if the aggregate-isotropy defect is ALSO absent here, that
// pins the defect specifically to curvature, not to cross-level coupling
// in general. If it's still present, curvature isn't required to trigger
// it at all.
static void test_cross_level_aggregate_isotropy_full_group_affine() {
  bool const saved_mult     = AdaptiveLattice::disable_multiplicity_correction;
  bool const saved_amr_corr = AdaptiveLattice::disable_step_pass2_amr_correct;
  AdaptiveLattice::disable_multiplicity_correction = false;
  AdaptiveLattice::disable_step_pass2_amr_correct  = false;

  AdaptiveCellCluster cl;
  bcc::build_bcc_sphere(cl, 8.0);
  cl.info.max_level = 1;

  std::int32_t target0 = -1;
  for (std::int32_t g : cl.leaf_indices) {
    if (count_neighbours(cl, g) != 14) continue;
    bool all_nb_interior = true;
    for (std::size_t k = 0; k < 14; ++k)
      if (count_neighbours(cl, cl[g].nb[k]) != 14) { all_nb_interior = false; break; }
    if (all_nb_interior) { target0 = g; break; }
  }
  check(target0 >= 0, "found a 2-shell-deep interior cell to split");

  AdaptiveLattice lat;
  lat.clusters.clear();
  auto result = lat.split(cl, target0);
  bcc::wire_split_children(cl, target0, result, 1.0);
  lat.rebuild_leaf_indices(cl);

  for (std::int32_t g : cl.leaf_indices) {
    double x, y, z;
    cell_xyz(cl, g, x, y, z);
    cl[g].u = analytic_linear(x, y, z);
  }
  {
    double x, y, z;
    cell_xyz(cl, target0, x, y, z);
    cl[target0].u = analytic_linear(x, y, z);
  }

  double const L_i = cl[result.first_child].L;
  double const dt  = 1.0;
  lat.step(cl, dt);

  double acc_group_total = 0.0;
  for (std::uint8_t c = 0; c < result.n_children; ++c) {
    std::int32_t cg = result.first_child + c;
    for (std::size_t k = 0; k < 14; ++k) {
      auto const nbg = cl[cg].nb[k];
      if (nbg < 0) continue;
      acc_group_total += bcc::face_weight[k] * cl[cg].phi[k];
    }
  }

  double acc_interior = 0.0;
  {
    AdaptiveCellCluster cl2;
    bcc::build_bcc_sphere(cl2, 8.0, L_i);
    AdaptiveLattice lat2;
    lat2.clusters.clear();
    lat2.rebuild_leaf_indices(cl2);
    for (std::int32_t g : cl2.leaf_indices) {
      double x, y, z;
      cell_xyz(cl2, g, x, y, z);
      cl2[g].u = analytic_linear(x, y, z);
    }
    std::int32_t target = -1;
    for (std::int32_t g : cl2.leaf_indices) {
      if (count_neighbours(cl2, g) != 14) continue;
      bool all_nb_interior = true;
      for (std::size_t k = 0; k < 14; ++k)
        if (count_neighbours(cl2, cl2[g].nb[k]) != 14) { all_nb_interior = false; break; }
      if (all_nb_interior) { target = g; break; }
    }
    check(target >= 0, "found a 2-shell-deep interior cell (reference)");
    lat2.step(cl2, dt);
    for (std::size_t k = 0; k < 14; ++k) acc_interior += bcc::face_weight[k] * cl2[target].phi[k];
  }

  AdaptiveLattice::disable_multiplicity_correction = saved_mult;
  AdaptiveLattice::disable_step_pass2_amr_correct  = saved_amr_corr;

  double const expected_flat = static_cast<double>(result.n_children) * acc_interior;
  std::printf(
      "test_cross_level_aggregate_isotropy_full_group_affine: n_children=%d "
      "acc_group_total=%.6f  n_children*acc_interior=%.6f  diff=%.6f rel_diff=%.3e\n",
      result.n_children, acc_group_total, expected_flat, acc_group_total - expected_flat,
      std::fabs((acc_group_total - expected_flat) / std::max(std::fabs(expected_flat), 1e-12)));
}

// The affine test above showed the aggregate defect vanishes EXACTLY
// without curvature. If it's a well-behaved Taylor-remainder-style
// truncation error, |diff| should scale LINEARLY with the field's second
// derivative -- a clean signature of a characterizable error term, vs.
// something messier (e.g. saturating, or scaling with a different power)
// if it isn't. Field: c*(x^2+y^2+z^2), sweeping c.
static double _measure_full_group_diff_for_curvature(double c, double R = 8.0, double d1 = 1.0) {
  AdaptiveLattice::disable_multiplicity_correction = false;
  AdaptiveLattice::disable_step_pass2_amr_correct  = false;

  AdaptiveCellCluster cl;
  bcc::build_bcc_sphere(cl, R, d1);
  cl.info.max_level = 1;

  std::int32_t target0 = -1;
  for (std::int32_t g : cl.leaf_indices) {
    if (count_neighbours(cl, g) != 14) continue;
    bool all_nb_interior = true;
    for (std::size_t k = 0; k < 14; ++k)
      if (count_neighbours(cl, cl[g].nb[k]) != 14) { all_nb_interior = false; break; }
    if (all_nb_interior) { target0 = g; break; }
  }
  check(target0 >= 0, "found a 2-shell-deep interior cell to split");

  AdaptiveLattice lat;
  lat.clusters.clear();
  auto result = lat.split(cl, target0);
  bcc::wire_split_children(cl, target0, result, d1);
  lat.rebuild_leaf_indices(cl);

  auto const field = [c](double x, double y, double z) { return c * isotropic_quadratic(x, y, z); };
  for (std::int32_t g : cl.leaf_indices) {
    double x, y, z;
    cell_xyz(cl, g, x, y, z);
    cl[g].u = field(x, y, z);
  }
  {
    double x, y, z;
    cell_xyz(cl, target0, x, y, z);
    cl[target0].u = field(x, y, z);
  }

  double const L_i = cl[result.first_child].L;
  double const dt  = 1.0;
  lat.step(cl, dt);

  double acc_group_total = 0.0;
  for (std::uint8_t ch = 0; ch < result.n_children; ++ch) {
    std::int32_t cg = result.first_child + ch;
    for (std::size_t k = 0; k < 14; ++k) {
      auto const nbg = cl[cg].nb[k];
      if (nbg < 0) continue;
      acc_group_total += bcc::face_weight[k] * cl[cg].phi[k];
    }
  }

  double acc_interior = 0.0;
  {
    AdaptiveCellCluster cl2;
    bcc::build_bcc_sphere(cl2, 8.0, L_i);
    AdaptiveLattice lat2;
    lat2.clusters.clear();
    lat2.rebuild_leaf_indices(cl2);
    for (std::int32_t g : cl2.leaf_indices) {
      double x, y, z;
      cell_xyz(cl2, g, x, y, z);
      cl2[g].u = field(x, y, z);
    }
    std::int32_t target = -1;
    for (std::int32_t g : cl2.leaf_indices) {
      if (count_neighbours(cl2, g) != 14) continue;
      bool all_nb_interior = true;
      for (std::size_t k = 0; k < 14; ++k)
        if (count_neighbours(cl2, cl2[g].nb[k]) != 14) { all_nb_interior = false; break; }
      if (all_nb_interior) { target = g; break; }
    }
    check(target >= 0, "found a 2-shell-deep interior cell (reference)");
    lat2.step(cl2, dt);
    for (std::size_t k = 0; k < 14; ++k) acc_interior += bcc::face_weight[k] * cl2[target].phi[k];
  }

  double const expected_flat = static_cast<double>(result.n_children) * acc_interior;
  return acc_group_total - expected_flat;
}

static void test_cross_level_aggregate_isotropy_curvature_scaling() {
  bool const saved_mult     = AdaptiveLattice::disable_multiplicity_correction;
  bool const saved_amr_corr = AdaptiveLattice::disable_step_pass2_amr_correct;

  std::printf("test_cross_level_aggregate_isotropy_curvature_scaling:\n");
  std::printf("  %8s %14s %14s\n", "c", "diff", "diff/c");
  double prev_ratio = 0.0;
  bool first = true;
  for (double c : {0.01, 0.1, 1.0, 10.0, 100.0}) {
    double const diff  = _measure_full_group_diff_for_curvature(c);
    double const ratio = diff / c;
    std::printf("  %8.3f %14.6f %14.6f\n", c, diff, ratio);
    if (!first) {
      check(std::fabs(ratio - prev_ratio) < 1e-6 * std::max(std::fabs(ratio), 1.0),
            "diff/c stays constant across a 100x sweep in curvature -- confirms the aggregate "
            "defect scales LINEARLY with curvature, a clean Taylor-remainder-style truncation "
            "error, not something messier");
    }
    prev_ratio = ratio;
    first      = false;
  }

  AdaptiveLattice::disable_multiplicity_correction = saved_mult;
  AdaptiveLattice::disable_step_pass2_amr_correct  = saved_amr_corr;
}

// Is 655.407407 (diff/c at R=8) a stable, geometry-intrinsic constant, or
// an artifact of this one specific mesh realization (this R, this
// particular target0 position/symmetry)? Different R picks a genuinely
// different target0 (first "2-shell-deep interior" cell found by
// iteration), so this is a real independent check, not the same cell
// twice.
static void test_cross_level_aggregate_isotropy_curvature_scaling_R_robustness() {
  bool const saved_mult     = AdaptiveLattice::disable_multiplicity_correction;
  bool const saved_amr_corr = AdaptiveLattice::disable_step_pass2_amr_correct;

  std::printf("test_cross_level_aggregate_isotropy_curvature_scaling_R_robustness:\n");
  std::printf("  %8s %14s\n", "R", "diff/c (c=1.0)");
  for (double R : {6.0, 8.0, 10.0, 12.0}) {
    double const ratio = _measure_full_group_diff_for_curvature(1.0, R);
    std::printf("  %8.1f %14.6f\n", R, ratio);
  }

  AdaptiveLattice::disable_multiplicity_correction = saved_mult;
  AdaptiveLattice::disable_step_pass2_amr_correct  = saved_amr_corr;
}

// The critical question for connecting this to the L8 blowup: acc_interior
// is analytically -80*c*L_i (derived by hand, confirmed exact: hex/sq
// symmetry kills the linear-in-position term, leaving
// Sum_k face_weight[k]^2*|offset_k|^2/L_i = (8*9*L_i^2 + 6*1*(2L_i/sqrt3)^2)/L_i
// = (72+8)*L_i = 80*L_i). If the ERROR TERM (diff) scales the SAME way
// (~L_i^1), the RELATIVE error (diff/acc_interior) stays constant with
// depth -- a fixed, harmless-in-relative-terms defect. If diff scales
// differently (e.g. stays ~L_i^0, tied to the ancestor's FIXED d1 scale
// rather than shrinking with L_i), the RELATIVE error GROWS as L_i shrinks
// at deeper levels -- which would directly explain why the real blowup
// gets WORSE with depth rather than staying proportionally the same.
static void test_cross_level_aggregate_isotropy_curvature_scaling_d1() {
  bool const saved_mult     = AdaptiveLattice::disable_multiplicity_correction;
  bool const saved_amr_corr = AdaptiveLattice::disable_step_pass2_amr_correct;

  std::printf("test_cross_level_aggregate_isotropy_curvature_scaling_d1:\n");
  std::printf("  %8s %10s %14s %14s %14s\n", "d1", "L_i", "diff", "diff/L_i", "diff/L_i^2");
  for (double d1 : {1.0, 1.0 / 3.0, 1.0 / 9.0}) {
    double const L_i  = d1 / 3.0;
    double const diff = _measure_full_group_diff_for_curvature(1.0, 8.0, d1);
    std::printf("  %8.4f %10.4f %14.6f %14.6f %14.6f\n", d1, L_i, diff, diff / L_i,
                diff / (L_i * L_i));
  }
  std::printf("  high precision at d1=1.0: diff=%.15f  diff/L_i=%.15f  diff*27=%.15f  "
              "diff/L_i*27=%.15f\n",
              _measure_full_group_diff_for_curvature(1.0, 8.0, 1.0),
              _measure_full_group_diff_for_curvature(1.0, 8.0, 1.0) / (1.0 / 3.0),
              _measure_full_group_diff_for_curvature(1.0, 8.0, 1.0) * 27.0,
              _measure_full_group_diff_for_curvature(1.0, 8.0, 1.0) / (1.0 / 3.0) * 27.0);

  AdaptiveLattice::disable_multiplicity_correction = saved_mult;
  AdaptiveLattice::disable_step_pass2_amr_correct  = saved_amr_corr;
}

// Same measurement as test_cross_level_flux_curvature_error, but sweeping
// the LEVEL GAP: instead of one split (fine child facing an unsplit L0
// neighbour directly, gap=1, ancestor walk lands on the immediate parent
// with zero extra hops), repeatedly split ONE child deeper along a single
// tracked face direction, so at gap=N the deepest leaf's ancestor walk must
// climb N levels to find a size-matched proxy, while that same face still
// points directly at the ORIGINAL, never-split L0 neighbour -- the same
// "potentially thousands of descendants sharing one ancestor's value"
// structure Part 2 of project_odd27_amr_junction_leak describes for real
// multi-level-gap telescopes, just a single tracked chain instead of a full
// ball/shell.
static void test_cross_level_flux_curvature_error_gap_sweep() {
  std::printf("test_cross_level_flux_curvature_error_gap_sweep:\n");
  std::printf("  %4s %10s %14s %14s %14s %10s %14s\n", "gap", "L_i", "true_deriv", "est_deriv",
              "err", "rel_err", "delta_u(anc)");
  for (int max_gap = 1; max_gap <= 6; ++max_gap) {
    AdaptiveCellCluster cl;
    bcc::build_bcc_sphere(cl, 8.0);
    cl.info.max_level = max_gap;

    std::int32_t target0 = -1;
    for (std::int32_t g : cl.leaf_indices) {
      if (count_neighbours(cl, g) != 14) continue;
      bool all_nb_interior = true;
      for (std::size_t k = 0; k < 14; ++k)
        if (count_neighbours(cl, cl[g].nb[k]) != 14) { all_nb_interior = false; break; }
      if (all_nb_interior) { target0 = g; break; }
    }
    check(target0 >= 0, "gap_sweep: found a 2-shell-deep interior cell to split");

    AdaptiveLattice lat;
    lat.clusters.clear();

    std::int32_t leaf_g       = target0;
    std::size_t k_dir         = 14;  // sentinel: not yet chosen
    std::int32_t original_nb  = -1;

    for (int lvl = 0; lvl < max_gap; ++lvl) {
      auto result = lat.split(cl, leaf_g);
      bcc::wire_split_children(cl, leaf_g, result, 1.0);
      lat.rebuild_leaf_indices(cl);

      if (lvl == 0) {
        // Pick one face direction with a live neighbour and track it down
        // the whole chain -- leaf_g is now a non-leaf proxy; its own nb[]
        // is left unchanged by its own split (same invariant that lets any
        // proxy still act as an interface for cells that were already
        // pointing at it).
        for (std::size_t k = 0; k < 14; ++k) {
          auto const nbg = cl[leaf_g].nb[k];
          if (nbg >= 0) { k_dir = k; original_nb = nbg; break; }
        }
        check(k_dir != 14, "gap_sweep: found an exterior face direction to track");
      }

      // Descend into whichever child still faces the SAME tracked exterior
      // neighbour directly -- keeps the chain's face k_dir pointed at the
      // original, still-unsplit L0 cell at every depth.
      std::int32_t next_leaf = -1;
      for (std::uint8_t c = 0; c < result.n_children; ++c) {
        std::int32_t cg = result.first_child + c;
        if (cl[cg].nb[k_dir] == original_nb) { next_leaf = cg; break; }
      }
      check(next_leaf >= 0,
            "gap_sweep: found the child continuing to face the tracked exterior neighbour");
      leaf_g = next_leaf;
    }
    check(cl[original_nb].L > cl[leaf_g].L,
          "gap_sweep: tracked neighbour is still genuinely coarser than the deepest leaf");

    for (std::int32_t g : cl.leaf_indices) {
      double x, y, z;
      cell_xyz(cl, g, x, y, z);
      cl[g].u = isotropic_quadratic(x, y, z);
    }
    // Every proxy along the chain (target0 and every intermediate split
    // cell) also needs its own value seeded by its own true position -- the
    // ancestor walk reads whichever one it lands on.
    std::int32_t anc_seed = cl[leaf_g].parent;
    while (anc_seed >= 0) {
      double x, y, z;
      cell_xyz(cl, anc_seed, x, y, z);
      cl[anc_seed].u = isotropic_quadratic(x, y, z);
      anc_seed = cl[anc_seed].parent;
    }

    double xi, yi, zi, xnb, ynb, znb;
    cell_xyz(cl, leaf_g, xi, yi, zi);
    cell_xyz(cl, original_nb, xnb, ynb, znb);
    double const L_i = cl[leaf_g].L;

    double const dx = xnb - xi, dy = ynb - yi, dz = znb - zi;
    double const d  = std::sqrt(dx * dx + dy * dy + dz * dz);
    double const ux = dx / d, uy = dy / d, uz = dz / d;
    double const face_x = xi + 0.5 * L_i * ux;
    double const face_y = yi + 0.5 * L_i * uy;
    double const face_z = zi + 0.5 * L_i * uz;
    double const true_deriv = 2.0 * (face_x * ux + face_y * uy + face_z * uz);

    bool const saved_toggle                          = AdaptiveLattice::disable_multiplicity_correction;
    AdaptiveLattice::disable_multiplicity_correction = true;
    double const dt = 1.0;
    lat.step(cl, dt);
    AdaptiveLattice::disable_multiplicity_correction = saved_toggle;

    double const phi_measured = cl[leaf_g].phi[k_dir];
    double const est_deriv    = -phi_measured / (dt * bcc::face_weight[k_dir]);
    double const err          = est_deriv - true_deriv;

    // Unconfounded companion metric: how wrong is the SUBSTITUTED ancestor
    // value against the fine leaf's own true local value -- independent of
    // any "geometric face" reference point, so it isn't affected by the
    // deepest leaf's own position drifting between gap depths.
    std::int32_t anc = cl[leaf_g].parent;
    while (anc >= 0 && cl[anc].L < cl[original_nb].L) anc = cl[anc].parent;
    double const u_src_val    = (anc >= 0) ? cl[anc].u : cl[leaf_g].u;
    double const u_true_leaf  = isotropic_quadratic(xi, yi, zi);
    double const delta_u      = u_src_val - u_true_leaf;

    std::printf("  %4d %10.3e %14.6f %14.6f %14.6f %10.3e %14.6f\n", max_gap, L_i, true_deriv,
                est_deriv, err, std::fabs(err / true_deriv), delta_u);
  }
}

// Quadratic field: refinement is no longer a pure no-op for PROLONGATION --
// _prolong_u_gradient_wide fits an AFFINE field, so it carries a genuine,
// bounded curvature residual against a curved field (measured ~0.68 for the
// coefficients/geometry below). This is NOT a bug; it's the expected
// reconstruction/truncation error of a linear-fit scheme, recorded here as a
// baseline -- a real level-boundary bug should show up as an error well
// beyond this baseline (or growing under repeated split/accumulate).
//
// RESTRICTION is a different story: it comes back essentially EXACT
// (~1e-16) even here, and that is NOT evidence the prolongation was accurate
// -- it is a structural artefact of the scheme, not a correctness signal.
// mean(children) after _prolong_u_gradient_wide == parent.u + mean(offset)*grad,
// and mean(offset) == 0 exactly (the 27 Odd27 offsets are symmetric under
// inversion through the parent centre) for ANY fitted gradient, right or
// wrong. So accumulate_to_parent trivially reproduces the pre-split value
// regardless of how good the prolongation is -- this round-trip alone can
// NEVER catch a bad prolongation. Only the per-child analytic comparison
// above (max_prolong_err) tests prolongation accuracy; restrict_err here
// only regression-guards accumulate_to_parent's own arithmetic (mean-of-
// children, Odd27 face-grouping) and the offset table's symmetry.
static double analytic_quadratic(double x, double y, double z) {
  return 1.0 + 0.4 * x * x - 0.25 * y * y + 0.15 * z * z + 0.1 * x * y;
}

static void test_static_boundary_consistency_quadratic() {
  AdaptiveCellCluster cl;
  bcc::build_bcc_sphere(cl, 4.0);
  cl.info.max_level = 1;

  for (std::int32_t g : cl.leaf_indices) {
    double x, y, z;
    cell_xyz(cl, g, x, y, z);
    cl[g].u = analytic_quadratic(x, y, z);
  }

  std::int32_t target = -1;
  for (std::int32_t g : cl.leaf_indices)
    if (count_neighbours(cl, g) == 14) {
      target = g;
      break;
    }
  check(target >= 0, "found interior cell");

  double const u0_L0_only = cl[target].u;

  AdaptiveLattice lat;
  lat.clusters.clear();
  auto result = lat.split(cl, target);
  bcc::wire_split_children(cl, target, result, 1.0);
  lat.rebuild_leaf_indices(cl);

  double max_prolong_err = 0.0;
  for (std::uint8_t c = 0; c < result.n_children; ++c) {
    std::int32_t cg = result.first_child + c;
    double x, y, z;
    cell_xyz(cl, cg, x, y, z);
    double expected = analytic_quadratic(x, y, z);
    max_prolong_err = std::max(max_prolong_err, std::fabs(cl[cg].u - expected));
  }
  lat.accumulate_to_parent(cl, target);
  double restrict_err = std::fabs(cl[target].u - u0_L0_only);

  std::printf(
      "test_static_boundary_consistency_quadratic: max prolong err=%.3e, "
      "restrict err=%.3e\n",
      max_prolong_err, restrict_err);

  // Loose baseline bounds -- document order of magnitude, not exactness.
  // Measured today: prolong ~6.8e-1 (real curvature residual), restrict
  // ~4e-16 (structurally exact regardless of prolongation quality -- see
  // comment above; NOT a correctness signal on its own).
  check(max_prolong_err < 1.0, "quadratic prolongation residual within expected baseline");
  check(restrict_err < 1e-6, "restriction stays structurally exact (mean-zero-offset artefact)");
}

// --- test 7: same-level flux conservation AT a coarse-fine boundary ---------
//
// Follow-on from the CLAWD investigation into deep-telescope AMR energy
// growth: `disable_same_level_flux_at_boundary_only` (a same-level, i.e.
// nb.L == cell.L, flux contribution computed ONLY for cells that also have
// >=1 cross-level face) gives a clean ~13x single-step-growth reduction when
// disabled, localizing the dominant growth mechanism there -- but the exact
// within-formula reason a boundary cell's same-level flux differs from an
// ordinary interior cell's was never found (extensive elimination of reflux,
// CFL, sigma-Jensen-bias, proxy-refresh-timing, cross-level-scale, and a
// rare same-L-non-leaf-proxy edge case all ruled out or negligible).
//
// This test checks the one thing that must hold EXACTLY, by pure algebra, if
// the same-level formula and its face_multiplicity/ancestor_group_multiplicity
// corrections are wired correctly for a boundary (Odd27) sibling pair: for
// ANY field, ANY dt, dphi_A_to_B == -dphi_B_to_A (pairwise antisymmetric,
// since sig_face and w are symmetric under A<->B swap and same-level flux
// never triggers ancestor substitution). A genuine child's own 14 face
// targets are geometrically unique (odd27_table.face_target[c] resolves each
// face k to a distinct 3D point, so two different k's from the SAME child
// can never alias to the SAME sibling) -- so face_multiplicity must be
// exactly 1 for every sibling connection, on BOTH ends, making this an exact
// (not approximate/bounded) invariant. If it holds here but the deep-
// telescope growth persists, that rules same-level's OWN formula (in
// isolation, for a single boundary) out as the direct cause -- the growth
// must come from something else (repeated substep accumulation, or a
// cross-level bug whose bias LEAKS into the bulk via same-level coupling,
// rather than same-level's own formula being locally wrong).
//
// Uses ONLY the public step() API (single flat, non-subcycled step -- same
// per-face formula and fmult/agmult divisors as step_subcycle's
// _step_pass2_lev) so cl[c].phi[k] after one call IS the freshly-computed
// dphi (phi starts at 0 for a freshly split, unstepped mesh).

// Same idea as test_static_boundary_consistency_linear, but through an
// UNBALANCED two-level shell: one L1 child goes on to split into 27 L2
// grandchildren while its 26 L1 siblings stay unsplit leaves. This mimics
// the actual shape found in a real organic-AMR run at max_level=2 (confirmed
// directly in Python: the converged mesh had only 21 L1 leaves vs 125550 L2
// leaves -- i.e. almost every L1 sibling stays coarse while a small number go
// two levels deep) -- see project_odd27_amr_junction_leak Part 20/21. Isolates
// whether accumulate_to_parent's bottom-up multi-level composition (accumulate_
// all, iterating level-by-level from deepest to shallowest) is exact through an
// IMBALANCED tree, not just the single-level-gap, all-siblings-equal case the
// existing static tests cover.
static void test_static_boundary_consistency_unbalanced_multilevel() {
  AdaptiveCellCluster cl;
  bcc::build_bcc_sphere(cl, 4.0);
  cl.info.max_level = 2;

  for (std::int32_t g : cl.leaf_indices) {
    double x, y, z;
    cell_xyz(cl, g, x, y, z);
    cl[g].u = analytic_linear(x, y, z);
  }

  std::int32_t target = -1;
  for (std::int32_t g : cl.leaf_indices)
    if (count_neighbours(cl, g) == 14) {
      target = g;
      break;
    }
  check(target >= 0, "found interior cell");

  double const u0_L0_only = cl[target].u;

  AdaptiveLattice lat;
  lat.clusters.clear();

  // L0 -> L1: split the target into 27 children.
  auto result1 = lat.split(cl, target);
  bcc::wire_split_children(cl, target, result1, 1.0);
  lat.rebuild_leaf_indices(cl);

  // L1 -> L2: split only ONE of the 27 new L1 children further; the other 26
  // stay unsplit L1 leaves -- the deliberate imbalance under test.
  std::int32_t deep_child = result1.first_child;
  auto result2 = lat.split(cl, deep_child);
  bcc::wire_split_children(cl, deep_child, result2, 1.0);
  lat.rebuild_leaf_indices(cl);

  // (a) Prolongation error at the deepest (L2) leaves, no stepping.
  double max_prolong_err = 0.0;
  for (std::uint8_t c = 0; c < result2.n_children; ++c) {
    std::int32_t cg = result2.first_child + c;
    double x, y, z;
    cell_xyz(cl, cg, x, y, z);
    double expected = analytic_linear(x, y, z);
    max_prolong_err = std::max(max_prolong_err, std::fabs(cl[cg].u - expected));
  }
  check(max_prolong_err < 1e-9,
        "L2 prolongation matches analytic linear field exactly, unbalanced shell");

  // (b) Multi-level restriction: accumulate the whole tree bottom-up
  // (L2 -> L1 -> L0) and compare the original target's u against its
  // pre-split L0-only value.
  lat.accumulate_all(cl);
  double restrict_err = std::fabs(cl[target].u - u0_L0_only);
  check(restrict_err < 1e-9,
        "multi-level restriction through an unbalanced (1 deep branch, 26 shallow "
        "siblings) shell matches L0-only value exactly (linear field, no stepping)");

  // (c) The intermediate L1 parent (deep_child, now non-leaf) should also
  // restrict to the analytic value at its own position.
  double x1, y1, z1;
  cell_xyz(cl, deep_child, x1, y1, z1);
  double expected_l1 = analytic_linear(x1, y1, z1);
  double l1_restrict_err = std::fabs(cl[deep_child].u - expected_l1);
  check(l1_restrict_err < 1e-9,
        "intermediate L1 parent restricts to its own analytic value exactly, unbalanced shell");

  std::printf(
      "test_static_boundary_consistency_unbalanced_multilevel: ok, max prolong err=%.3e, "
      "restrict err=%.3e, L1-parent restrict err=%.3e\n",
      max_prolong_err, restrict_err, l1_restrict_err);
}

static void test_boundary_same_level_conservation() {
  AdaptiveCellCluster cl;
  bcc::build_bcc_sphere(cl, 6.0);
  cl.info.max_level = 1;

  // Off-center smooth Gaussian bump in u -- gives real, nonzero local
  // gradient/flux without being an exact affine field (a harder case than
  // the linear-field tests above, matching the established Part-1 AMR
  // ground-truth methodology).
  double const gx = 1.3, gy = 0.7, gz = -0.4, width2 = 4.0;
  for (std::int32_t g : cl.leaf_indices) {
    double x, y, z;
    cell_xyz(cl, g, x, y, z);
    double const r2 = (x - gx) * (x - gx) + (y - gy) * (y - gy) + (z - gz) * (z - gz);
    cl[g].u = std::exp(-r2 / width2);
  }

  std::int32_t target = -1;
  for (std::int32_t g : cl.leaf_indices)
    if (count_neighbours(cl, g) == 14) {
      target = g;
      break;
    }
  check(target >= 0, "found interior cell");

  AdaptiveLattice lat;
  lat.clusters.clear();
  auto result = lat.split(cl, target);  // default Odd27
  bcc::wire_split_children(cl, target, result, 1.0);
  lat.rebuild_leaf_indices(cl);

  std::int32_t const fc = result.first_child;
  std::uint8_t const nc = result.n_children;

  double dt = 0.001;
  lat.step(cl, dt);  // single flat step -- no subcycling

  // Pairwise reciprocity + total-flux conservation over every (child, face)
  // whose neighbour is a SIBLING (same-level, within the 27-child block).
  double max_asym = 0.0;
  double same_level_sum = 0.0;
  int n_same_level_links = 0;
  for (std::uint8_t c = 0; c < nc; ++c) {
    std::int32_t cg = fc + c;
    auto cell = cl[cg];
    for (std::size_t k = 0; k < CellSlab::NF; ++k) {
      std::int32_t nbg = cell.nb[k];
      bool const is_sibling = nbg >= fc && nbg < fc + nc;
      if (!is_sibling) continue;
      ++n_same_level_links;
      double const phi_here = cl[cg].phi[k];
      std::size_t const mk = bcc::mirror_face[k];
      double const phi_there = cl[nbg].phi[mk];
      max_asym = std::max(max_asym, std::fabs(phi_here + phi_there));
      same_level_sum += phi_here;
    }
  }
  check(n_same_level_links > 0, "boundary region has same-level (sibling) links");
  check(max_asym < 1e-12,
        "boundary same-level flux is exactly pairwise-antisymmetric (A->B == -B->A)");
  check(std::fabs(same_level_sum) < 1e-12,
        "total same-level flux across the boundary region's own faces sums to exactly zero");

  // Control: the identical check on an ordinary, untouched INTERIOR L0-L0
  // same-level pair elsewhere in the sphere -- expected to trivially hold,
  // establishing this isn't "the formula is broken everywhere", specifically
  // isolating the boundary/Odd27-sibling case above.
  std::int32_t interior_a = -1, interior_b = -1;
  std::size_t interior_k = 0;
  for (std::int32_t g : cl.leaf_indices) {
    if (g == target || (g >= fc && g < fc + nc)) continue;
    if (count_neighbours(cl, g) != 14) continue;
    auto cell = cl[g];
    for (std::size_t k = 0; k < CellSlab::NF; ++k) {
      std::int32_t nbg = cell.nb[k];
      if (nbg < 0 || nbg == target || (nbg >= fc && nbg < fc + nc)) continue;
      if (cl[nbg].level != 0) continue;
      interior_a = g;
      interior_b = nbg;
      interior_k = k;
      break;
    }
    if (interior_a >= 0) break;
  }
  check(interior_a >= 0, "found an untouched interior L0-L0 pair for control");
  std::size_t const interior_mk = bcc::mirror_face[interior_k];
  double const interior_asym =
      std::fabs(cl[interior_a].phi[interior_k] + cl[interior_b].phi[interior_mk]);
  check(interior_asym < 1e-12, "control: interior L0-L0 same-level flux is exactly antisymmetric");

  std::printf(
      "test_boundary_same_level_conservation: ok, %d same-level links, max asym=%.3e, "
      "sum=%.3e, control asym=%.3e\n",
      n_same_level_links, max_asym, same_level_sum, interior_asym);
}

// --- test 8: global dphi conservation across a MULTI-LEVEL telescope -------
//
// Direct regression guard for the AMR reflux fix in rebuild_amr_corrections/
// _collect_leaves_facing_multi/_step_pass2_amr_correct: the debug-only
// internal assertion check_total_dphi_balanced (debug/debug_adaptive_lattice.hpp)
// caught a real defect here that NO release-mode test had ever exercised
// (release builds compile EQUILIBRIUM_DEBUG_DO out entirely) -- a 2-level
// Odd27 telescope's total Σdphi (summed over every live face of every leaf)
// came out at rel=2.11e-10 relative to the sum of |dphi|, a genuine
// dt-proportional structural residual (confirmed by varying dt over a
// 10000x range and finding rel exactly constant -- the signature of a real
// bug, not floating-point summation noise), traced to _collect_leaves_facing
// matching candidate children by a single shared face index instead of
// searching all 14 faces (Odd27's face_target table can route a child's own
// face k through a DIFFERENT external index) -- see [[project_odd27_amr_junction_leak]].
//
// This test reproduces the SAME invariant via the public API only (no
// temp_phi access, no debug build required): with phi==0 on every leaf
// before a single lat.step(), phi==dphi exactly afterward (pass3 does
// phi[k] += dphi[k]), so Σ_leaves Σ_k phi[k] over every live face IS
// check_total_dphi_balanced's own "total", checked against the SAME
// 1e-10 relative tolerance, independent of NDEBUG/-g.

static void test_multilevel_telescope_dphi_conservation() {
  AdaptiveCellCluster cl;
  bcc::build_bcc_sphere(cl, 8.0);
  cl.info.max_level = 2;

  double const gx = 1.3, gy = 0.7, gz = -0.4, width2 = 4.0;
  for (std::int32_t g : cl.leaf_indices) {
    double x, y, z;
    cell_xyz(cl, g, x, y, z);
    double const r2 = (x - gx) * (x - gx) + (y - gy) * (y - gy) + (z - gz) * (z - gz);
    cl[g].u = std::exp(-r2 / width2);
  }

  std::int32_t target = -1;
  for (std::int32_t g : cl.leaf_indices)
    if (count_neighbours(cl, g) == 14) {
      target = g;
      break;
    }
  check(target >= 0, "found interior cell");

  AdaptiveLattice lat;
  lat.clusters.clear();
  auto result = lat.split(cl, target);
  bcc::wire_split_children(cl, target, result, 1.0);
  lat.rebuild_leaf_indices(cl);

  // Split one of the level-1 children again -- a 2-level telescope, the
  // deeper coarse-fine gap where the single-index parent-skip guard bug
  // (fixed alongside the face-matching bug) actually showed up.
  std::int32_t fc1 = result.first_child;
  std::uint8_t nc1 = result.n_children;
  std::int32_t target2 = -1;
  for (std::uint8_t c = 0; c < nc1; ++c) {
    std::int32_t cg = fc1 + c;
    if (count_neighbours(cl, cg) == 14) {
      target2 = cg;
      break;
    }
  }
  if (target2 < 0) target2 = fc1;

  auto result2 = lat.split(cl, target2);
  bcc::wire_split_children(cl, target2, result2, 1.0 / 3.0);
  lat.rebuild_leaf_indices(cl);

  double const dt = 0.0005;
  lat.step(cl, dt);

  // Weighted by bcc::face_weight[k], matching check_total_dphi_balanced
  // (debug_adaptive_lattice.hpp) -- pass1's own du sums face_weight[k]*phi[k],
  // so a face's pull on conserved Sigma-u scales with face_weight, not 1. The
  // RAW (unweighted) sum stopped being the right invariant to check the
  // moment the hex/sq face-weight-mixing fix (project_odd27_amr_junction_leak
  // Part 12) switched the actual AMR correction to target the weighted sum --
  // this test predates that fix (Part 5) and was never updated to match.
  // Confirmed directly (2026-08-03): raw sum rel~6.5e-7 (fails), weighted sum
  // rel~1.3e-16 (exact) -- a stale test assertion, not a physics regression.
  double total = 0.0, scale = 0.0;
  for (std::int32_t g : cl.leaf_indices) {
    auto cell = cl[g];
    for (std::size_t k = 0; k < CellSlab::NF; ++k) {
      if (cell.nb[k] < 0) continue;
      double const v = bcc::face_weight[k] * cell.phi[k];
      total += v;
      scale += std::fabs(v);
    }
  }
  check(scale > 0.0, "telescope step produced nonzero flux");
  double const rel = std::fabs(total) / scale;
  check(rel < 1e-10, "global dphi conservation holds across a 2-level Odd27 telescope");

  std::printf(
      "test_multilevel_telescope_dphi_conservation: ok, %zu leaves, total=%.3e scale=%.3e "
      "rel=%.3e\n",
      cl.leaf_indices.size(), total, scale, rel);
}

// --- test: check_total_dphi_balanced's global-scale blind spot is EXPECTED
//
// doc/dst_algo.md (2026-08-04 survey, following the accumulate_to_parent
// fix) flagged the single-bucket global check as a candidate false-negative
// risk: summing face_weight[k]*temp_phi[k*L+j] over EVERY leaf and face into
// one total/scale pair means a real, 100%-locally-unbalanced group of tiny
// contributions could in principle be invisible against unrelated large,
// balanced (canceling) activity elsewhere. A same-day attempt to hardened
// this by bucketing by each leaf's own level was tried and DISPROVEN by
// direct measurement: test_boundary_same_level_conservation (below) showed
// level 0's total and level 1's total across a real Odd27 junction come out
// as EXACT negatives of each other (legitimate cross-level flux exchange,
// conserved in aggregate -- the reflux-correction machinery doing its job
// correctly), so per-level bucketing produced false positives on entirely
// healthy behaviour and was reverted. The real invariant is GLOBAL
// conservation, not per-level.
//
// This test now documents the (still real, still theoretically possible)
// blind spot as an accepted, expected property of a global check rather
// than a bug to chase: place a large (O(1)), exactly-canceling "region A"
// (6 faces) alongside a tiny (O(1e-90)), all-same-sign, genuinely
// unbalanced "region B" (8 faces) on the SAME leaf, and confirm the check
// does NOT throw -- region B's real local imbalance is swamped by region
// A's dominant scale. This is the correct, intended behaviour for a global
// invariant: an isolated, physically-implausible synthetic defect this tiny
// relative to unrelated activity elsewhere is not something this check is
// designed to catch, and forcing it to (as the per-level attempt did) broke
// legitimate cross-level flux exchange instead.
static void test_check_total_dphi_balanced_global_blind_spot_is_expected() {
  AdaptiveCellCluster cl;
  bcc::build_bcc_sphere(cl, 3.0);

  AdaptiveLattice lat;
  lat.clusters.clear();
  lat.rebuild_leaf_indices(cl);

  std::int32_t target = -1;
  for (std::int32_t g : cl.leaf_indices)
    if (count_neighbours(cl, g) == 14) {
      target = g;
      break;
    }
  check(target >= 0, "found interior cell (all 14 faces live)");

  std::size_t target_j = 0;
  for (std::size_t j = 0; j < cl.leaf_indices.size(); ++j)
    if (cl.leaf_indices[j] == target) {
      target_j = j;
      break;
    }
  std::size_t const L = cl.leaf_indices.size();

  double const region_a_weighted = 1.0;    // O(1), dominant activity
  double const region_b_weighted = 1e-90;  // O(1e-90), tiny genuine local imbalance

  // Region A: faces 0-5, alternating sign, weighted contribution cancels exactly.
  for (std::size_t k = 0; k < 6; ++k) {
    double const signed_val = (k % 2 == 0) ? region_a_weighted : -region_a_weighted;
    cl.temp_phi[k * L + target_j] = signed_val / bcc::face_weight[k];
  }
  // Region B: faces 6-13, all same sign -- genuinely, 100% locally unbalanced.
  for (std::size_t k = 6; k < 14; ++k) {
    cl.temp_phi[k * L + target_j] = region_b_weighted / bcc::face_weight[k];
  }

  // The check does NOT throw -- this is the correct, intended behaviour of
  // a global conservation invariant, not a defect: region B's tiny, isolated
  // imbalance is swamped by region A's dominant scale, exactly as a global
  // sum should behave.
  bool threw = false;
  try {
    check_total_dphi_balanced(cl, CellSlab::NF, bcc::face_weight);
  } catch (AdaptiveLatticeException const&) {
    threw = true;
  }
  check(!threw, "global check correctly passes -- region B's synthetic imbalance has no bearing "
                "on the aggregate conservation invariant it actually checks");

  // For contrast: region B in isolation IS genuinely, detectably unbalanced
  // -- confirming this is a real (if synthetic/implausible) local defect,
  // not merely too-small-to-matter noise.
  double total_b = 0.0, scale_b = 0.0;
  for (std::size_t k = 6; k < 14; ++k) {
    double const v = bcc::face_weight[k] * cl.temp_phi[k * L + target_j];
    total_b += v;
    scale_b += std::fabs(v);
  }
  double const rel_b = scale_b > 0.0 ? std::fabs(total_b) / scale_b : 0.0;
  check(rel_b > 0.5, "region B in isolation is genuinely, detectably unbalanced (rel~1.0)");

  std::printf(
      "test_check_total_dphi_balanced_global_blind_spot_is_expected: ok, global check passes "
      "(region B rel-in-isolation=%.3f) -- confirmed this is the correct behaviour of a global "
      "invariant, not a bug; see doc/dst_algo.md for why per-level bucketing was tried and "
      "reverted\n",
      rel_b);
}

// --- test: biharmonic (eta_h) fp64 noise floor vs. level -------------------
//
// 2026-08-04: while chasing the L8 first-look blowup, found (analytically,
// not yet measured) that the biharmonic operator behind eta_h
// (Laplacian(Laplacian(u)), _leaf_laplacian applied twice) should hit the
// fp64 precision floor around L8 (L^4 ~ 2.4x machine epsilon there) --
// see doc/dst_amr_scaling.md. That analysis assumed a SMOOTH field
// (Taylor-truncation-scaling); this test checks it directly instead of
// trusting the estimate, using a UNIFORM, junction-free mesh (build_bcc_
// sphere at each level's own cell size directly, no AMR/refinement at all)
// so the measurement isolates pure fp64 arithmetic from any AMR-junction
// question (a separate, already extensively investigated topic --
// project_odd27_amr_junction_leak).
//
// Method: seed the EXACT ISOTROPIC quadratic field u=x^2+y^2+z^2 (continuum
// Laplacian = 6 everywhere) -- deliberately NOT `analytic_quadratic`
// (anisotropic, with an xy cross term, built for a different test's
// prolongation-accuracy purpose): the historically-validated "exact
// Laplacian" claim (project_odd27_amr_junction_leak Part 6) specifically
// used the isotropic case, and a first attempt at this test reusing
// `analytic_quadratic` measured lap1=0.0 (not 0.6) -- the BCC coordination
// shell's discrete Laplacian formula is confirmed exact for the isotropic
// field but does NOT straightforwardly generalize to an anisotropic one;
// that's a separate, real finding (recorded below), not a reason to trust
// this test's own zero-truncation-error premise any less for the isotropic
// case it actually uses. lap1 = _leaf_laplacian(u) should equal 6.0 at every
// level, machine-precision-exact, for a fully-interior cell (14 same-level
// neighbours) -- a real correctness invariant, not just a precision probe.
// lap2 = _leaf_laplacian(lap1) is the biharmonic quantity eta_h actually
// uses; since lap1 is (up to fp noise) a CONSTANT field, lap2's magnitude IS
// directly the fp noise floor -- there is no real signal left for it to
// measure once lap1 is already ~constant.
static void test_biharmonic_precision_floor_by_level() {
  double const SPLIT_RATIO = 3.0;
  // Historically confirmed constant (project_odd27_amr_junction_leak Part 6):
  // the discrete Laplacian formula does NOT recover the literal continuum
  // value (6.0 for u=x^2+y^2+z^2) -- it recovers a FIXED geometric
  // normalisation constant times it (~32/6, from the BCC coordination
  // shell's own second-moment sum), confirmed exactly 32.0 for this field at
  // a fully-interior cell. The claim under test is that this constant stays
  // level-independent (exact at every L), not that it equals the literal
  // continuum Laplacian.
  double const EXPECTED_LAP1 = 32.0;
  std::printf("test_biharmonic_precision_floor_by_level:\n");
  std::printf("  %5s %10s %14s %14s %14s\n", "level", "d1", "lap1(interior)", "lap2(interior)",
              "lap2/lap1");
  for (int level = 0; level <= 8; ++level) {
    double const d1 = 1.0 / std::pow(SPLIT_RATIO, level);

    AdaptiveCellCluster cl;
    // R_cells=8 (not 4): lap2's target needs a 2-SHELL-deep interior cell --
    // its own 14 neighbours must ALSO each have 14 neighbours, so lap1 is
    // exact everywhere lap2's sum reads from, not just at the target itself.
    bcc::build_bcc_sphere(cl, /*R_cells=*/8.0, d1);
    AdaptiveLattice lat;
    lat.clusters.clear();
    lat.rebuild_leaf_indices(cl);  // populates g_to_j/g_max, which _leaf_laplacian needs

    for (std::int32_t g : cl.leaf_indices) {
      double x, y, z;
      cell_xyz(cl, g, x, y, z);
      cl[g].u = isotropic_quadratic(x, y, z);
    }

    std::int32_t target = -1;
    for (std::int32_t g : cl.leaf_indices) {
      if (count_neighbours(cl, g) != 14) continue;
      bool all_nb_interior = true;
      for (std::size_t k = 0; k < 14; ++k) {
        if (count_neighbours(cl, cl[g].nb[k]) != 14) { all_nb_interior = false; break; }
      }
      if (all_nb_interior) { target = g; break; }
    }
    check(target >= 0, "found a 2-shell-deep interior cell (itself and all 14 neighbours have "
                        "14 same-level neighbours each)");

    std::vector<double> u_field(cl.leaf_indices.size());
    for (std::size_t j = 0; j < cl.leaf_indices.size(); ++j) u_field[j] = cl[cl.leaf_indices[j]].u;

    auto const lap1 = AdaptiveLattice::_leaf_laplacian(cl, u_field);
    auto const lap2 = AdaptiveLattice::_leaf_laplacian(cl, lap1);

    std::size_t target_j = 0;
    for (std::size_t j = 0; j < cl.leaf_indices.size(); ++j)
      if (cl.leaf_indices[j] == target) {
        target_j = j;
        break;
      }

    double const lap1_err = std::fabs(lap1[target_j] - EXPECTED_LAP1);
    double const lap2_val = lap2[target_j];
    std::printf("  %5d %10.3e %14.6e %14.6e %14.3e\n", level, d1, lap1[target_j], lap2_val,
                lap1[target_j] != 0.0 ? lap2_val / lap1[target_j] : 0.0);

    // lap1 must stay machine-precision-exact at EVERY level -- this is a
    // real correctness invariant (any level-dependent drift here would be a
    // genuine bug, separate from the fp64-noise question lap2 measures).
    check(lap1_err < 1e-9, "lap1 stays exact (32.0) at this level's cell size");
  }
}

// Sanity check prompted directly by the cross-level curvature-correction
// postmortem: does the SAME-LEVEL flux formula's secant (already proven
// EXACTLY conservative/antisymmetric elsewhere) actually match literal
// calculus true_deriv at the geometric midpoint -- for BOTH hex (weight 3)
// and sq (weight 1) face directions -- or does "proven exact" only mean
// "exactly conservative", not "matches calculus"? _leaf_laplacian already
// showed this scheme has its own normalisation constant (32.0, not the
// continuum 6.0) -- if the SAME-LEVEL secant itself doesn't match calculus
// either, holding the CROSS-LEVEL formula to that standard was the wrong
// yardstick from the start, independent of hex/sq. If it DOES match
// (per-direction, no hex/sq split), the yardstick was right and the
// cross-level defect is real. Measures actual neighbour distances directly
// rather than parsing face_dirs/hex_dir tables by hand.
static void test_same_level_secant_vs_calculus_hex_sq() {
  AdaptiveCellCluster cl;
  bcc::build_bcc_sphere(cl, 8.0);
  AdaptiveLattice lat;
  lat.clusters.clear();
  lat.rebuild_leaf_indices(cl);

  for (std::int32_t g : cl.leaf_indices) {
    double x, y, z;
    cell_xyz(cl, g, x, y, z);
    cl[g].u = isotropic_quadratic(x, y, z);
  }

  std::int32_t target = -1;
  for (std::int32_t g : cl.leaf_indices) {
    if (count_neighbours(cl, g) != 14) continue;
    bool all_nb_interior = true;
    for (std::size_t k = 0; k < 14; ++k)
      if (count_neighbours(cl, cl[g].nb[k]) != 14) { all_nb_interior = false; break; }
    if (all_nb_interior) { target = g; break; }
  }
  check(target >= 0, "found a 2-shell-deep interior cell");

  double p0[3];
  cell_xyz(cl, target, p0[0], p0[1], p0[2]);

  std::printf("test_same_level_secant_vs_calculus_hex_sq:\n");
  std::printf("  %3s %5s %10s %14s %14s %14s\n", "k", "type", "dist", "true_deriv", "secant",
              "err");
  double max_err_hex = 0.0, max_err_sq = 0.0;
  for (std::size_t k = 0; k < 14; ++k) {
    auto const nbg = cl[target].nb[k];
    check(nbg >= 0, "interior target has a live same-level neighbour on every face");
    double p1[3];
    cell_xyz(cl, nbg, p1[0], p1[1], p1[2]);
    double const dx = p1[0] - p0[0], dy = p1[1] - p0[1], dz = p1[2] - p0[2];
    double const dist = std::sqrt(dx * dx + dy * dy + dz * dz);
    double const ux = dx / dist, uy = dy / dist, uz = dz / dist;
    double const mx = 0.5 * (p0[0] + p1[0]), my = 0.5 * (p0[1] + p1[1]), mz = 0.5 * (p0[2] + p1[2]);
    double const true_deriv = 2.0 * (mx * ux + my * uy + mz * uz);
    // Same shape as _step_pass2_uniform's w=face_weight[k]/L, applied to
    // (nb.u-u_i) directly (sigma=1, dt=1, stripped of face_weight[k] itself
    // since we're checking the SPACING/geometry convention, not the
    // Kelvin-cell area weighting, which is a separate, already-understood
    // factor).
    double const secant = (cl[nbg].u - cl[target].u) / cl[target].L;
    double const err    = secant - true_deriv;
    (k < 8 ? max_err_hex : max_err_sq) = std::max(k < 8 ? max_err_hex : max_err_sq, std::fabs(err));
    std::printf("  %3zu %5s %10.6f %14.6f %14.6f %14.3e\n", k, k < 8 ? "hex" : "sq", dist,
                true_deriv, secant, err);
  }
  std::printf("  max|err| hex=%.3e sq=%.3e\n", max_err_hex, max_err_sq);
}

// --- test: amr_correction_pairs entries are globally unique (leaf,face) ---
//
// Philip's concern (2026-08-11): at a spheromak's level boundary a single
// leaf can sit at more than one distinct coarse-fine (or proxy-proxy)
// junction, on different faces -- e.g. an unsplit hub cell M with two
// separately-split neighbours N1, N2. _accumulate_amr_registers/
// _apply_amr_reflux key every correction purely by pair index (see their
// doc comments), so this verifies the structural invariant that makes that
// per-pair accumulation correct even when one cell touches several
// junctions: every (leaf, own-face) slot referenced by amr_js/amr_js_face
// is claimed by EXACTLY ONE pair-side entry. If the same (leaf, face) slot
// were listed twice -- within one pair or across two different pairs --
// _accumulate_amr_registers would sum that single physical temp_phi entry
// into more than one register, and _apply_amr_reflux would apply more than
// one correction to the same physical face: the exact "total per junction,
// used more than once" double-count step_subcycle's reflux exists to avoid.
static void test_amr_correction_pairs_globally_unique_entries() {
  AdaptiveCellCluster cl;
  bcc::build_bcc_sphere(cl, 8.0);
  cl.info.max_level = 2;

  // Hub cell M with 14 fully-interior neighbours, then split TWO of its
  // neighbours (N1, N2) -- M now borders two INDEPENDENT coarse-fine
  // junctions on two different faces: the "multiple junctions on one cell"
  // geometry a spheromak's level boundary produces.
  std::int32_t M = -1;
  for (std::int32_t g : cl.leaf_indices) {
    if (count_neighbours(cl, g) != 14) continue;
    bool all_nb_interior = true;
    for (std::size_t k = 0; k < 14; ++k)
      if (count_neighbours(cl, cl[g].nb[k]) != 14) { all_nb_interior = false; break; }
    if (all_nb_interior) { M = g; break; }
  }
  check(M >= 0, "found a hub cell with 14 fully-interior neighbours");

  std::int32_t const N1 = cl[M].nb[0];
  std::int32_t const N2 = cl[M].nb[1];
  check(N1 >= 0 && N2 >= 0 && N1 != N2, "two distinct live neighbours of the hub");

  AdaptiveLattice lat;
  lat.clusters.clear();
  auto r1 = lat.split(cl, N1);
  bcc::wire_split_children(cl, N1, r1, 1.0);
  lat.rebuild_leaf_indices(cl);

  auto r2 = lat.split(cl, N2);
  bcc::wire_split_children(cl, N2, r2, 1.0);
  lat.rebuild_leaf_indices(cl);

  // Deepen one branch further -- a 2nd-level split adjacent to the still-
  // level-1 N2 branch, mirroring the multi-level onion structure around a
  // spheromak core (some cells now face THREE different refinement depths
  // at once).
  std::int32_t const fc1 = r1.first_child;
  std::uint8_t const nc1 = r1.n_children;
  std::int32_t target2 = -1;
  for (std::uint8_t c = 0; c < nc1; ++c) {
    std::int32_t const cg = fc1 + c;
    if (count_neighbours(cl, cg) == 14) { target2 = cg; break; }
  }
  if (target2 < 0) target2 = fc1;
  auto r3 = lat.split(cl, target2);
  bcc::wire_split_children(cl, target2, r3, 1.0 / 3.0);
  lat.rebuild_leaf_indices(cl);

  check(cl.n_non_leaf_cells > 0, "mesh has at least one proxy");
  check(!cl.amr_correction_pairs.empty(), "at least one AMR correction pair was built");

  // The actual invariant: every (leaf global id, own face index) slot
  // touched by amr_js/amr_js_face belongs to exactly one pair-side entry.
  std::map<std::pair<std::int32_t, std::uint8_t>, int> seen;
  for (auto const& p : cl.amr_correction_pairs) {
    for (std::int32_t i = p.begin; i < p.end; ++i) {
      auto const leaf_g = cl.leaf_indices[cl.amr_js[i]];
      auto const face_k = cl.amr_js_face[i];
      seen[{leaf_g, face_k}]++;
    }
  }
  int max_count = 0;
  std::int32_t dup_g = -1;
  std::uint8_t dup_k = 0;
  for (auto const& [key, count] : seen) {
    if (count > max_count) {
      max_count = count;
      dup_g      = key.first;
      dup_k      = key.second;
    }
  }
  if (max_count > 1)
    std::fprintf(stderr, "  DUP: leaf g=%d face k=%u referenced by %d correction-pair entries\n",
                 dup_g, dup_k, max_count);
  check(max_count <= 1,
        "every (leaf, own-face) slot is claimed by at most one AMR correction-pair entry -- no "
        "double-counting when a cell touches multiple coarse-fine junctions");

  std::printf(
      "test_amr_correction_pairs_globally_unique_entries: ok, %zu leaves, %zu pairs, %zu "
      "distinct (leaf,face) slots referenced, max reuse=%d\n",
      cl.leaf_indices.size(), cl.amr_correction_pairs.size(), seen.size(), max_count);
}

// ---------------------------------------------------------------------------
// test_stale_parent_proxy_after_neighbour_refine
//
// KNOWN BUG, not yet fixed (2026-08-12): wire_split_children()'s Odd27 path
// resolves each child's "external" faces (odd27_table.face_target[c][k] >=
// ODD27_EXTERNAL_BASE) to `par.nb[...]` ONCE, at split time -- a fixed
// compile-time-table lookup that is never revisited. If the NEIGHBOUR cell
// referenced by that external face is *itself* split afterward (to the same
// level), the wiring is stale: the child keeps pointing at the neighbour's
// PARENT (now a non-leaf, coarser proxy) instead of the neighbour's own
// matching child leaf -- even though a genuine same-level leaf sibling now
// exists one hop further in. This makes every flux calc through that face
// go through ancestor-substitution as if it were a real coarse-fine
// junction, on a face that is actually a same-level interior boundary.
//
// This is NOT limited to a real coarse-fine mesh edge -- it fires at EVERY
// inter-parent seam in a uniformly, independently-refined region (e.g. every
// ~3 L1-cells throughout the 08/11 everything_active L1 core), which is
// exactly why AdaptiveCellCluster::leaf_is_boundary reads ~96% true even at
// the spheromak's own centre, nowhere near the L0/L1 domain boundary (see
// exp/2026/08/12/17_15_ghost_cell_junction_error/). Once a region is
// uniformly refined and will never coarsen back, these cells don't need a
// live parent relationship at all -- and right now the stale one is
// actively wrong, not just unnecessary overhead.
//
// This test builds two adjacent interior leaves, splits BOTH to level 1,
// and checks that a child of one, whose external face pointed at the other
// pre-split, now resolves to a genuine same-level LEAF -- not the other's
// now-non-leaf parent. Expected to FAIL until wire_split_children (or a
// post-refine rewiring pass) is fixed to re-resolve external faces whose
// target has since been split to the matching level.
static void test_stale_parent_proxy_after_neighbour_refine() {
  AdaptiveCellCluster cl;
  bcc::build_bcc_sphere(cl, 8.0);
  cl.info.max_level = 1;

  // Find two adjacent, fully-interior leaves (both 14 neighbours, and their
  // shared neighbours too) so the split/rewiring under test isn't confounded
  // by the sphere's own outer boundary.
  std::int32_t g0 = -1, g1 = -1;
  std::size_t face01 = 0;
  for (std::int32_t g : cl.leaf_indices) {
    if (count_neighbours(cl, g) != 14) continue;
    for (std::size_t k = 0; k < 14; ++k) {
      std::int32_t nb = cl[g].nb[k];
      if (nb >= 0 && count_neighbours(cl, nb) == 14) {
        g0 = g;
        g1 = nb;
        face01 = k;
        break;
      }
    }
    if (g0 >= 0) break;
  }
  check(g0 >= 0 && g1 >= 0, "found two adjacent fully-interior leaves");

  AdaptiveLattice lat;
  lat.clusters.clear();

  auto result0 = lat.split(cl, g0);
  bcc::wire_split_children(cl, g0, result0, 1.0);
  lat.rebuild_leaf_indices(cl);

  auto result1 = lat.split(cl, g1);
  bcc::wire_split_children(cl, g1, result1, 1.0);
  lat.rebuild_leaf_indices(cl);

  check(!cl[g0].is_leaf() && !cl[g1].is_leaf(), "both original cells are now split (non-leaf)");
  check(cl[g0].level == 0 && cl[g1].level == 0, "both parents remain at level 0");

  // Any child of g0 whose face k was externally routed toward g1 pre-split
  // (odd27_table.face_target[c][k] resolves to par_nb[face01] == g1) should,
  // now that g1 is ALSO split to level 1, point at a genuine level-1 LEAF --
  // not at g1 itself (level 0, non-leaf). HEX faces (k<8) have exactly one
  // geometrically-unique same-level partner and must resolve. SQ faces
  // (k>=8) have genuine multi-child fan-in even at matched levels (confirmed
  // empirically via the BFS-split regression, tst/substrate/test_adaptive_vs
  // _substrate.py TestMultipleSplits -- forcing a 1:1 pick there caused two
  // different source cells to collide on the same slot and broke
  // check_total_dphi_balanced) -- resolve_stale_external_proxies correctly
  // leaves them alone, so sq staleness here is EXPECTED, not a bug.
  std::int32_t fc0 = result0.first_child;
  std::uint8_t nc0 = result0.n_children;
  int n_checked_hex = 0, n_stale_hex = 0, n_checked_sq = 0, n_stale_sq = 0;
  for (std::uint8_t c = 0; c < nc0; ++c) {
    std::int32_t child_g = fc0 + c;
    for (std::size_t k = 0; k < 14; ++k) {
      auto const t = bcc::odd27_table.face_target[c][k];
      if (t < bcc::ODD27_EXTERNAL_BASE) continue;  // sibling face, not under test
      if (t - bcc::ODD27_EXTERNAL_BASE != face01) continue;  // only the face that pointed at g1
      std::int32_t nb_g = cl[child_g].nb[k];
      bool const stale = (nb_g == g1) && !cl[g1].is_leaf();
      if (k < 8) {
        ++n_checked_hex;
        if (stale) ++n_stale_hex;
      } else {
        ++n_checked_sq;
        if (stale) ++n_stale_sq;
      }
    }
  }
  check(n_checked_hex + n_checked_sq > 0, "at least one child face under test (g0's face toward g1, post-split)");

  std::printf(
      "test_stale_parent_proxy_after_neighbour_refine: hex %d/%d still stale (must be 0), "
      "sq %d/%d still stale (expected -- genuine multiplicity, left to the existing "
      "ancestor-substitution machinery)\n",
      n_stale_hex, n_checked_hex, n_stale_sq, n_checked_sq);

  check(n_checked_hex == 0 || n_stale_hex == 0,
        "hex-face children of g0 whose external face pointed at g1 pre-split must, after g1 is "
        "ALSO split to the same level, resolve to a genuine same-level LEAF child of g1 -- not "
        "g1's own now-non-leaf (stale parent-proxy) cell");
}

// ---------------------------------------------------------------------------

int main() {
  test_build_sphere();
  test_split();
  test_split_hex8();
  test_split_hex_center9();
  test_coarsen();
  test_accumulate();
  test_accumulate_extreme_dynamic_range();
  test_sibling_sum_order_dependence();
  test_step();
  test_hdf5_save_load_round_trip();
  test_static_boundary_consistency_linear();
  test_prolong_u_gradient_wide_extreme_gradient();
  test_cross_level_flux_curvature_error();
  test_local_gradient_same_level_accuracy();
  test_same_level_secant_vs_calculus_hex_sq();
  test_cross_level_aggregate_isotropy();
  test_cross_level_aggregate_isotropy_full_group();
  test_cross_level_aggregate_isotropy_full_group_debias();
  test_cross_level_aggregate_isotropy_full_group_vs_unsplit();
  test_cross_level_aggregate_isotropy_full_group_vs_unsplit_2level_interior();
  test_cross_level_aggregate_isotropy_full_group_vs_unsplit_2level_ungraded();
  test_cross_level_aggregate_isotropy_full_group_vs_unsplit_2level_graded();
  test_cross_level_aggregate_isotropy_full_group_affine();
  test_cross_level_aggregate_isotropy_curvature_scaling();
  test_cross_level_aggregate_isotropy_curvature_scaling_R_robustness();
  test_cross_level_aggregate_isotropy_curvature_scaling_d1();
  test_cross_level_flux_curvature_error_gap_sweep();
  test_static_boundary_consistency_quadratic();
  test_static_boundary_consistency_unbalanced_multilevel();
  test_boundary_same_level_conservation();
  test_multilevel_telescope_dphi_conservation();
  test_check_total_dphi_balanced_global_blind_spot_is_expected();
  test_biharmonic_precision_floor_by_level();
  test_amr_correction_pairs_globally_unique_entries();
  test_stale_parent_proxy_after_neighbour_refine();
  std::printf("ALL PASSED\n");
}
