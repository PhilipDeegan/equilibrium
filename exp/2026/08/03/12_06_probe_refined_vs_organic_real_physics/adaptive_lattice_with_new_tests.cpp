// Standalone C++ test for AdaptiveLattice / BCC geometry layer.
//
// Compile and run (from repo root):
//   g++ -std=c++20 -O2 -I inc -o /tmp/test_adaptive_lattice \
//       tst/substrate/test_adaptive_lattice.cpp && /tmp/test_adaptive_lattice

#include "eq/core/data/lattice/adaptive_lattice.hpp"
#include "eq/core/data/lattice/bcc_geometry.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>

using namespace equilibrium::dst;

static void check(bool cond, const char* msg) {
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

  std::printf("test_build_sphere: %zu cells, %zu slabs\n",
              cl.leaf_indices.size(), cl.n_slabs());
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
  lat.clusters.clear();  // we manage cl directly here
  auto result = lat.split(cl, target);  // default: SplitModel::Odd27 (ratio-3, 27 children)
  bcc::wire_split_children(cl, target, result, 1.0);
  lat.rebuild_leaf_indices(cl);

  check(result.n_children == adaptive_cell::N_CHILDREN_ODD27, "split() defaults to 27 children (Odd27)");
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
    if (count_neighbours(cl, g) == 14) { target = g; break; }
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
    if (count_neighbours(cl, g) == 14) { target = g; break; }
  check(target >= 0, "found interior cell");

  std::size_t before = cl.leaf_indices.size();

  AdaptiveLattice lat;
  lat.clusters.clear();
  auto result = lat.split<SplitModel::HexCenter9>(cl, target);
  bcc::wire_split_children<SplitModel::HexCenter9>(cl, target, result, 1.0);
  lat.rebuild_leaf_indices(cl);

  check(result.n_children == adaptive_cell::N_CHILDREN_HEX_CENTER, "split<HexCenter9>() makes 9 children");
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
  check(result.n_children == adaptive_cell::N_CHILDREN_HEX_CENTER, "split<HexCenter9>() makes 9 children");
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
    if (count_neighbours(cl, g) == 14) { target = g; break; }
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
    if (count_neighbours(cl, g) == 14) { target = g; break; }
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

  std::printf("test_accumulate: ok, u_parent=%.4f expected=%.4f\n",
              cl[target].u, u_expected);
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
    std::size_t s    = AdaptiveCellCluster::slab_of(g);
    std::size_t slot = AdaptiveCellCluster::slot_of(g);
    auto& slab = cl.slab(s);
    double r2 = slab.pos[0 * CellSlab::CAPACITY + slot] * slab.pos[0 * CellSlab::CAPACITY + slot]
              + slab.pos[1 * CellSlab::CAPACITY + slot] * slab.pos[1 * CellSlab::CAPACITY + slot]
              + slab.pos[2 * CellSlab::CAPACITY + slot] * slab.pos[2 * CellSlab::CAPACITY + slot];
    if (r2 < best) { best = r2; origin = g; }
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
    if (nbg >= 0 && cl[nbg].u > 0.0) { neighbour_gained = true; break; }
  }
  check(neighbour_gained, "neighbours gain u");

  // Repeat with 4 threads on a fresh cluster — result must be identical.
  AdaptiveCellCluster cl2;
  bcc::build_bcc_sphere(cl2, 3.0, 1.0, 0.0, 1.0);
  cl2[origin].u = 1.0;
  lat.set_n_threads(4);
  for (int i = 0; i < 50; ++i) lat.step(cl2, dt);
  lat.set_n_threads(1);
  double u_mt = cl2[origin].u;
  check(std::fabs(u_mt - u_origin_after) < 1e-12, "multi-thread matches single-thread");

  std::printf("test_step: ok, u_origin %.4f -> %.4f (single), %.4f (4-thread)\n",
              u_origin_before, u_origin_after, u_mt);
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
    if (count_neighbours(cl, g) == 14) { target = g; break; }
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
    if (count_neighbours(cl, g) == 14) { target = g; break; }
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
    if (count_neighbours(cl, g) == 14) { target = g; break; }
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
    if (count_neighbours(cl, g) == 14) { target = g; break; }
  check(target >= 0, "found interior cell");

  AdaptiveLattice lat;
  lat.clusters.clear();
  auto result = lat.split(cl, target);                // default Odd27
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
  check(max_asym < 1e-12, "boundary same-level flux is exactly pairwise-antisymmetric (A->B == -B->A)");
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
    if (count_neighbours(cl, g) == 14) { target = g; break; }
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
    if (count_neighbours(cl, cg) == 14) { target2 = cg; break; }
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

// ---------------------------------------------------------------------------

int main() {
  test_build_sphere();
  test_split();
  test_split_hex8();
  test_split_hex_center9();
  test_coarsen();
  test_accumulate();
  test_step();
  test_static_boundary_consistency_linear();
  test_static_boundary_consistency_quadratic();
  test_static_boundary_consistency_unbalanced_multilevel();
  test_boundary_same_level_conservation();
  test_multilevel_telescope_dphi_conservation();
  std::printf("ALL PASSED\n");
}
