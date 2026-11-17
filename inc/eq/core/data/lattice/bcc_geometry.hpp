#ifndef _EQUILIBRIUM_CORE_DATA_LATTICE_BCC_GEOMETRY_HPP_
#define _EQUILIBRIUM_CORE_DATA_LATTICE_BCC_GEOMETRY_HPP_

// BCC Kelvin-cell geometry layer.
//
// Face ordering (fixed, matches equilibrium/math/geo/kelvin.py):
//   Faces 0-7  : hex faces — direction (sx,sy,sz) ∈ {-1,+1}³ in lex order:
//                  0=(-,-,-)  1=(-,-,+)  2=(-,+,-)  3=(-,+,+)
//                  4=(+,-,-)  5=(+,-,+)  6=(+,+,-)  7=(+,+,+)
//   Faces 8-13 : square faces — ±x, ±y, ±z:
//                  8=(-x) 9=(+x) 10=(-y) 11=(+y) 12=(-z) 13=(+z)
//
// Geometry (d1 = hex face distance = 1 Planck unit by default):
//   h  = d1 / sqrt(3)         hex offset per axis
//   d2 = 2*h                  square face distance
//
// Split model (N_CHILDREN_HEX = 8):
//   8 children placed at (±h_child, ±h_child, ±h_child) from parent centre,
//   where h_child = (d1/2) / sqrt(3) = h/2.  Each child c sits in the octant
//   given by child_octant[c] = hex_dir[c] — one child per hex face of the parent.
//
//   Within the 8 children:
//     Square siblings: children differing in exactly one octant component are
//       at square-face distance from each other (3 siblings per child).
//     Hex faces: no sibling is at hex distance — all hex faces point outside.

#include "bcc_defs.hpp"
#include "adaptive_lattice.hpp"
#include "eq/core/phlop.hpp"
#include "cluster.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace equilibrium::dst::discrete {
namespace bcc {

// ---------------------------------------------------------------------------
// build_bcc_sphere()
//
// Populate a freshly constructed cluster with a uniform BCC sphere of void
// cells and wire all 14 neighbours.
//
// Grid: integer triples (ix,iy,iz) in units of h = d1/sqrt(3).
//   BCC: all integer triples are valid atoms (corner + body sub-lattices).
//   Hex neighbours:  Δ = (±1,±1,±1)  → physical distance = h*sqrt(3) = d1.
//   Sq  neighbours:  Δ = (±2, 0, 0) etc. → physical distance = 2*h = d2.
//   Origin = BCC atom at (0,0,0).
//
// Parameters:
//   cluster  — freshly constructed, empty cluster
//   R_cells  — sphere radius in d1 units
//   d1       — hex face distance (Planck units, default 1.0)
//   u0       — initial potential (default 0.0)
//   lam0     — initial lambda (default 0.0)
// ---------------------------------------------------------------------------
inline void build_bcc_sphere(AdaptiveCellCluster& cluster, double R_cells, double d1 = 1.0,
                             double u0 = 0.0, double lam0 = 0.0) {
  double const h = d1 / std::sqrt(3.0);
  double const R2 = R_cells * R_cells * d1 * d1;

  int imax = static_cast<int>(std::ceil(R_cells * std::sqrt(3.0))) + 1;

  PositionMap pos_to_global;
  // BCC has 2 atoms per a³ = (2h)³ cell; density = 2/(8h³) = 1/(4h³).
  pos_to_global.reserve(
      static_cast<std::size_t>((4.0 / 3.0) * 3.14159265 * R_cells * R_cells * R_cells * 0.7));

  std::vector<IntVec3> grid_pts;

  // Pass 1: allocate cells for all BCC points within radius.
  // BCC in h units: corner atoms have all-even indices; body atoms all-odd.
  // Mixed-parity indices (e.g. (1,0,0)) are face-centre positions — skip them.
  for (int ix = -imax; ix <= imax; ++ix) {
    for (int iy = -imax; iy <= imax; ++iy) {
      for (int iz = -imax; iz <= imax; ++iz) {
        if (((ix & 1) != (iy & 1)) || ((iy & 1) != (iz & 1))) continue;
        double xc = ix * h, yc = iy * h, zc = iz * h;
        if (xc * xc + yc * yc + zc * zc > R2) continue;

        IntVec3 key{ix, iy, iz};
        std::int32_t g = cluster.alloc_cell(0, AdaptiveCellCluster::zone_id(xc, yc, zc));
        pos_to_global[key] = g;
        grid_pts.push_back(key);

        auto cell = cluster[g];
        cell.u = u0;
        cell.L = d1;
        cell.lam = lam0;
        cell.fixed_u = 0;
        cell.fixed_phi = 0;
        cell.parent = -1;
        cell.first_child = -1;
        cell.n_children = 0;

        std::size_t s = AdaptiveCellCluster::slab_of(g);
        std::size_t slot = AdaptiveCellCluster::slot_of(g);
        auto& slab = cluster.slab(s);
        slab.pos[0 * CellSlab::CAPACITY + slot] = xc;
        slab.pos[1 * CellSlab::CAPACITY + slot] = yc;
        slab.pos[2 * CellSlab::CAPACITY + slot] = zc;

        for (std::size_t k = 0; k < CellSlab::NF; ++k) {
          cell.phi[k] = 0.0;
          cell.nb[k] = -1;
        }
      }
    }
  }  // iz, iy, ix

  // Pass 2: wire neighbours.
  for (IntVec3 const& pt : grid_pts) {
    std::int32_t g = pos_to_global.at(pt);
    for (std::size_t k = 0; k < 8; ++k) {
      auto& d = hex_dir[k];
      auto it = pos_to_global.find({pt.x + d[0], pt.y + d[1], pt.z + d[2]});
      cluster[g].nb[k] = (it != pos_to_global.end()) ? it->second : -1;
    }
    for (std::size_t k = 8; k < 14; ++k) {
      auto& d = sq_dir[k - 8];
      auto it = pos_to_global.find({pt.x + 2 * d[0], pt.y + 2 * d[1], pt.z + 2 * d[2]});
      cluster[g].nb[k] = (it != pos_to_global.end()) ? it->second : -1;
    }
  }

  // Pass 3: build leaf index and size scratch buffers.
  cluster.leaf_indices.clear();
  for (IntVec3 const& pt : grid_pts) cluster.leaf_indices.push_back(pos_to_global.at(pt));

  std::size_t L = cluster.leaf_indices.size();
  constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
  cluster.temp_du.assign(L, 0.0);
  cluster.temp_phi.assign(CellSlab::NF * L, 0.0);
  cluster.temp_phi_mirror.assign(CellSlab::NF * L, 0.0);
  cluster.criterion_min.assign(L, NaN);
  cluster.criterion_max.assign(L, NaN);
  // face_multiplicity/ancestor_group_multiplicity default-construct to
  // all_one=true (fresh uniform mesh: no aliasing yet) already.
}

// ---------------------------------------------------------------------------
// build_bcc_box()
//
// Populate a cluster with a uniform BCC box of void cells and wire all 14
// neighbours.  Identical to build_bcc_sphere except the bounds check is an
// axis-aligned box: x ∈ [-Lx/2, Lx/2], y ∈ [-Ly/2, Ly/2], z ∈ [-Lz/2, Lz/2].
//
// Parameters:
//   cluster       — freshly constructed, empty cluster
//   Lx, Ly, Lz   — full side lengths in d1 units
//   d1            — hex face distance (default 1.0)
//   u0, lam0      — initial field values
// ---------------------------------------------------------------------------
inline void build_bcc_box(AdaptiveCellCluster& cluster,
                          double Lx, double Ly, double Lz,
                          double d1 = 1.0, double u0 = 0.0,
                          double lam0 = 0.0) {
  double const h = d1 / std::sqrt(3.0);
  double const hx = Lx / 2.0, hy = Ly / 2.0, hz = Lz / 2.0;

  int imax_x = static_cast<int>(std::ceil(hx / h)) + 1;
  int imax_y = static_cast<int>(std::ceil(hy / h)) + 1;
  int imax_z = static_cast<int>(std::ceil(hz / h)) + 1;

  PositionMap pos_to_global;
  pos_to_global.reserve(static_cast<std::size_t>(Lx * Ly * Lz * 0.75));

  std::vector<IntVec3> grid_pts;

  for (int ix = -imax_x; ix <= imax_x; ++ix) {
    for (int iy = -imax_y; iy <= imax_y; ++iy) {
      for (int iz = -imax_z; iz <= imax_z; ++iz) {
        if (((ix & 1) != (iy & 1)) || ((iy & 1) != (iz & 1))) continue;
        double xc = ix * h, yc = iy * h, zc = iz * h;
        if (std::abs(xc) > hx || std::abs(yc) > hy || std::abs(zc) > hz) continue;

        IntVec3 key{ix, iy, iz};
        std::int32_t g = cluster.alloc_cell(0, AdaptiveCellCluster::zone_id(xc, yc, zc));
        pos_to_global[key] = g;
        grid_pts.push_back(key);

        auto cell = cluster[g];
        cell.u = u0;
        cell.L = d1;
        cell.lam = lam0;
        cell.fixed_u = 0;
        cell.fixed_phi = 0;
        cell.parent = -1;
        cell.first_child = -1;
        cell.n_children = 0;

        std::size_t s = AdaptiveCellCluster::slab_of(g);
        std::size_t slot = AdaptiveCellCluster::slot_of(g);
        auto& slab = cluster.slab(s);
        slab.pos[0 * CellSlab::CAPACITY + slot] = xc;
        slab.pos[1 * CellSlab::CAPACITY + slot] = yc;
        slab.pos[2 * CellSlab::CAPACITY + slot] = zc;

        for (std::size_t k = 0; k < CellSlab::NF; ++k) {
          cell.phi[k] = 0.0;
          cell.nb[k] = -1;
        }
      }
    }
  }

  for (IntVec3 const& pt : grid_pts) {
    std::int32_t g = pos_to_global.at(pt);
    for (std::size_t k = 0; k < 8; ++k) {
      auto& d = hex_dir[k];
      auto it = pos_to_global.find({pt.x + d[0], pt.y + d[1], pt.z + d[2]});
      cluster[g].nb[k] = (it != pos_to_global.end()) ? it->second : -1;
    }
    for (std::size_t k = 8; k < 14; ++k) {
      auto& d = sq_dir[k - 8];
      auto it = pos_to_global.find({pt.x + 2 * d[0], pt.y + 2 * d[1], pt.z + 2 * d[2]});
      cluster[g].nb[k] = (it != pos_to_global.end()) ? it->second : -1;
    }
  }

  cluster.leaf_indices.clear();
  for (IntVec3 const& pt : grid_pts) cluster.leaf_indices.push_back(pos_to_global.at(pt));

  std::size_t L = cluster.leaf_indices.size();
  constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
  cluster.temp_du.assign(L, 0.0);
  cluster.temp_phi.assign(CellSlab::NF * L, 0.0);
  cluster.temp_phi_mirror.assign(CellSlab::NF * L, 0.0);
  cluster.criterion_min.assign(L, NaN);
  cluster.criterion_max.assign(L, NaN);
  // face_multiplicity/ancestor_group_multiplicity default-construct to
  // all_one=true (fresh uniform mesh: no aliasing yet) already.
}

// ---------------------------------------------------------------------------
// build_bcc_box_periodic()
//
// Periodic BCC box: integer grid points with all-equal coordinate parity, ix in [-Px/2, Px/2 - 1] (same for y, z),
// every face wired with wrap-around mod P. P must be even (the period then maps each sublattice onto itself, hex
// and square faces alike), so the box tiles space exactly. Side length P h, h = d1 / sqrt(3).
// ---------------------------------------------------------------------------
inline void build_bcc_box_periodic(AdaptiveCellCluster& cluster, int Px, int Py, int Pz, double d1 = 1.0,
                                   double u0 = 0.0, double lam0 = 0.0) {
  if (Px < 4 || Py < 4 || Pz < 4 || (Px & 1) || (Py & 1) || (Pz & 1))
    throw std::invalid_argument("build_bcc_box_periodic: periods must be even and >= 4");
  double const h = d1 / std::sqrt(3.0);
  auto wrap = [](int i, int P) { return ((i + P / 2) % P + P) % P - P / 2; };

  PositionMap pos_to_global;
  pos_to_global.reserve(static_cast<std::size_t>(Px) * Py * Pz / 4 + 1);
  std::vector<IntVec3> grid_pts;
  for (int ix = -Px / 2; ix < Px / 2; ++ix)
    for (int iy = -Py / 2; iy < Py / 2; ++iy)
      for (int iz = -Pz / 2; iz < Pz / 2; ++iz) {
        if (((ix & 1) != (iy & 1)) || ((iy & 1) != (iz & 1))) continue;
        double xc = ix * h, yc = iy * h, zc = iz * h;
        IntVec3 key{ix, iy, iz};
        std::int32_t g = cluster.alloc_cell(0, AdaptiveCellCluster::zone_id(xc, yc, zc));
        pos_to_global[key] = g;
        grid_pts.push_back(key);
        auto cell = cluster[g];
        cell.u = u0;
        cell.L = d1;
        cell.lam = lam0;
        cell.fixed_u = 0;
        cell.fixed_phi = 0;
        cell.parent = -1;
        cell.first_child = -1;
        cell.n_children = 0;
        std::size_t s = AdaptiveCellCluster::slab_of(g);
        std::size_t slot = AdaptiveCellCluster::slot_of(g);
        auto& slab = cluster.slab(s);
        slab.pos[0 * CellSlab::CAPACITY + slot] = xc;
        slab.pos[1 * CellSlab::CAPACITY + slot] = yc;
        slab.pos[2 * CellSlab::CAPACITY + slot] = zc;
        for (std::size_t k = 0; k < CellSlab::NF; ++k) {
          cell.phi[k] = 0.0;
          cell.nb[k] = -1;
        }
      }

  for (IntVec3 const& pt : grid_pts) {
    std::int32_t g = pos_to_global.at(pt);
    for (std::size_t k = 0; k < CellSlab::NF; ++k) {
      auto const d = face_delta(k);
      IntVec3 q{wrap(pt.x + d[0], Px), wrap(pt.y + d[1], Py), wrap(pt.z + d[2], Pz)};
      cluster[g].nb[k] = pos_to_global.at(q);
    }
  }

  cluster.leaf_indices.clear();
  for (IntVec3 const& pt : grid_pts) cluster.leaf_indices.push_back(pos_to_global.at(pt));
  std::size_t L = cluster.leaf_indices.size();
  constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
  cluster.temp_du.assign(L, 0.0);
  cluster.temp_phi.assign(CellSlab::NF * L, 0.0);
  cluster.temp_phi_mirror.assign(CellSlab::NF * L, 0.0);
  cluster.criterion_min.assign(L, NaN);
  cluster.criterion_max.assign(L, NaN);
}

// ---------------------------------------------------------------------------
// _prolong_u_gradient_wide() — wide-stencil least-squares gradient
// prolongation of u, overwriting split()'s constant injection (ch.u =
// parent.u) now that child positions are known.
//
// Constant injection is a poor approximation for Odd27: children span
// offsets up to +-2 (fine h/3 units) from the parent centre, a much wider
// physical spread than the ratio-2 octant splits ever had (+-1 h_child) —
// verified (probe_split27_boundary.py) that a spatially-aware reconstruction
// cuts RMS prolongation error ~10x vs constant injection.
//
// A gradient is inherently a multi-point construct — fitting one from only
// the parent's immediate 1-ring (14 neighbours) is sensitive to any single
// noisy neighbour. This instead BFS-walks out to ORDER hops through nb[]
// (deduplicated), fitting ONE linear gradient via ordinary least squares
// (3x3 normal equations, unweighted) over every point reached, then applies
// u_child = parent.u + offset(child)·grad.
//
// Tried first: an inverse-distance-weighted BLEND of raw neighbour u-VALUES
// (rather than a linear gradient) — empirically made H-growth at AMR
// interfaces WORSE, not better (see the by_level growth sweep this was
// diagnosed from), for two reasons: (1) it isn't conservative in general
// (measured drift: mean(children.u) vs parent.u off by ~1e-3 relative for a
// smooth field — needed a post-hoc renormalisation hack to fix), and (2) it
// seeds small child-to-child value variation across all 196 new internal
// sibling faces at every split, which this scheme's dissipation-free
// symplectic wave step + reflux correction only amplifies over time. A
// LINEAR correction avoids both problems structurally: it is automatically
// mean-zero over the 27 canonical offsets (they are exactly symmetric under
// inversion through the parent centre — verified: 13 mirrored pairs + the
// self-paired centre offset), so mean(children.u) == parent.u exactly
// regardless of the fitted gradient, no renormalisation needed. Whether the
// wider stencil actually reduces H-growth vs. the original 1-ring gradient
// fit (also linear, also automatically conservative, but noisier) is the
// open empirical question this version exists to test.
//
// Model-agnostic (reads only parent_g/position + each child's own position
// via BFS through live nb[], no split-model-specific tables) — only invoked
// for Odd27 today; the legacy ratio-2 models keep split()'s constant
// injection unchanged (their existing debug invariant,
// check_state_after_split's "constant prolongation violated" assertion,
// still holds for them).
// ---------------------------------------------------------------------------
inline void _prolong_u_gradient_wide(AdaptiveCellCluster& cluster, std::int32_t parent_g,
                                     double par_u, double px, double py, double pz,
                                     std::int32_t fc, std::uint8_t nc) {
  constexpr int ORDER = 5;        // BFS hop count through nb[] for the fit stencil
  constexpr double EPS2 = 1e-20;  // guards exact position coincidence
  constexpr double DET_EPS = 1e-12;

  static thread_local std::vector<std::int32_t> seen;
  static thread_local std::vector<std::int32_t> frontier;
  static thread_local std::vector<std::int32_t> next_frontier;
  seen.clear();
  frontier.clear();
  seen.push_back(parent_g);
  frontier.push_back(parent_g);

  double Mxx = 0, Mxy = 0, Mxz = 0, Myy = 0, Myz = 0, Mzz = 0, bx = 0, by = 0, bz = 0;

  auto accumulate = [&](std::int32_t g) {
    std::size_t const s = AdaptiveCellCluster::slab_of(g);
    std::size_t const slot = AdaptiveCellCluster::slot_of(g);
    auto& slab = cluster.slab(s);
    double const x = slab.pos[0 * CellSlab::CAPACITY + slot] - px;
    double const y = slab.pos[1 * CellSlab::CAPACITY + slot] - py;
    double const z = slab.pos[2 * CellSlab::CAPACITY + slot] - pz;
    if (x * x + y * y + z * z < EPS2) return;  // coincides with parent's own position
    double const du = cluster[g].u - par_u;
    Mxx += x * x; Mxy += x * y; Mxz += x * z;
    Myy += y * y; Myz += y * z; Mzz += z * z;
    bx += x * du; by += y * du; bz += z * du;
  };

  for (int hop = 0; hop < ORDER && !frontier.empty(); ++hop) {
    next_frontier.clear();
    for (std::int32_t g : frontier) {
      auto cell = cluster[g];
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        std::int32_t const nb_g = cell.nb[k];
        if (nb_g < 0) continue;
        bool already = false;
        for (std::int32_t s : seen)
          if (s == nb_g) { already = true; break; }
        if (already) continue;
        seen.push_back(nb_g);
        next_frontier.push_back(nb_g);
        accumulate(nb_g);
      }
    }
    frontier.swap(next_frontier);
  }

  // Solve the symmetric 3x3 normal-equation system M*grad = b (Cramer's rule).
  // Falls back to grad=(0,0,0) — i.e. constant injection — if the stencil is
  // degenerate (near-boundary cell with too few/too co-planar neighbours).
  double const det = Mxx * (Myy * Mzz - Myz * Myz) - Mxy * (Mxy * Mzz - Myz * Mxz) +
                     Mxz * (Mxy * Myz - Myy * Mxz);
  double gx = 0.0, gy = 0.0, gz = 0.0;
  if (std::abs(det) > DET_EPS) {
    double const detx = bx * (Myy * Mzz - Myz * Myz) - Mxy * (by * Mzz - Myz * bz) +
                        Mxz * (by * Myz - Myy * bz);
    double const dety = Mxx * (by * Mzz - bz * Myz) - bx * (Mxy * Mzz - Myz * Mxz) +
                        Mxz * (Mxy * bz - by * Mxz);
    double const detz = Mxx * (Myy * bz - Myz * by) - Mxy * (Mxy * bz - by * Mxz) +
                        bx * (Mxy * Myz - Myy * Mxz);
    gx = detx / det;
    gy = dety / det;
    gz = detz / det;
  }

  for (std::uint8_t c = 0; c < nc; ++c) {
    std::int32_t const cg = fc + c;
    std::size_t const cs = AdaptiveCellCluster::slab_of(cg);
    std::size_t const cslot = AdaptiveCellCluster::slot_of(cg);
    auto& cslab = cluster.slab(cs);
    double const cx = cslab.pos[0 * CellSlab::CAPACITY + cslot] - px;
    double const cy = cslab.pos[1 * CellSlab::CAPACITY + cslot] - py;
    double const cz = cslab.pos[2 * CellSlab::CAPACITY + cslot] - pz;
    cluster[cg].u = par_u + cx * gx + cy * gy + cz * gz;
  }
}

// ---------------------------------------------------------------------------
// _prolong_phi_generic() — shared step-5 phi correction (see the long
// comment at its call sites below). Model-agnostic: only inspects nb[]
// contiguous-block membership (is_sibling) and dynamically counts n_ext per
// face, so it is correct for any split model/child count (8, 9, 27, ...)
// without hardcoding which/how-many children are externally-facing.
// ---------------------------------------------------------------------------
inline void _prolong_phi_generic(AdaptiveCellCluster& cluster, AdaptiveCellRef const& par,
                                 std::int32_t fc, std::uint8_t nc) {
  for (std::size_t k = 0; k < CellSlab::NF; ++k) {
    auto const parent_phi_k = par.phi[k];
    std::size_t n_ext = 0;
    for (std::uint8_t c = 0; c < nc; ++c) {
      auto const nb_ck = cluster[fc + c].nb[k];
      bool const is_sibling = nb_ck >= fc && nb_ck < fc + nc;
      if (!is_sibling && nb_ck >= 0) ++n_ext;
    }
    auto const share = n_ext > 0 ? parent_phi_k / static_cast<double>(n_ext) : 0.0;
    for (std::uint8_t c = 0; c < nc; ++c) {
      auto ch = cluster[fc + c];
      auto const nb_ck = ch.nb[k];
      bool const is_sibling = nb_ck >= fc && nb_ck < fc + nc;
      ch.phi[k] = (is_sibling || nb_ck < 0) ? 0.0 : share;
    }
  }
}

// ---------------------------------------------------------------------------
// wire_split_children()
//
// Complete the nb[] wiring that AdaptiveLattice::split<Model>() leaves to
// this layer. Must be called immediately after split<Model>() with the SAME
// Model (compile-time selected, default kDefaultSplitModel = Odd27).
//
// Odd27 (default): child c is placed at parent_centre + odd27_table.offset[c]
// * h_child (h_child = (d1/3)/sqrt(3)); nb[] wiring is a flat lookup into
// odd27_table.face_target — already resolved (at compile time) to either a
// sibling child index or the parent's own proxy face, so no per-face
// sibling/mirror tables are needed the way the ratio-2 models require.
//
// Hex8 / HexCenter9 (legacy, ratio-2): child c sits at
// parent_centre + child_octant[c] * h_child. Wiring:
//   1. Set child positions in pos[].
//   2. Square-sibling connections: child c and child flip_child(c,j) share
//      sq faces (8+2j+side) at child scale — 3 siblings per child.
//   3. Hex faces: all children inherit nb[k] = parent.nb[k] as a coarse proxy.
//      Only child k is geometrically adjacent, but all children read the same
//      coarse proxy (standard AMR coarse-fine approximation).
//      nb[k] = -1 only where the parent itself has no hex-k neighbour (global boundary).
//   4. Outer sq faces: any sq face not wired by step 2 inherits parent.nb[k].
//      nb[k] = -1 only where the parent has no sq-k neighbour (global boundary).
//   After this call, nb[k] = -1 iff the cell is at the global sphere boundary.
// ---------------------------------------------------------------------------
template <SplitModel Model = kDefaultSplitModel>
inline void wire_split_children(AdaptiveCellCluster& cluster, std::int32_t parent_g,
                                SplitResult const& result, double d1) {
  // EQ_SCOPE_TIMER("equilibrium::dst::discrete::wire_split_children")

  auto par = cluster[parent_g];
  std::int32_t fc = result.first_child;
  constexpr std::uint8_t nc = split_model_traits<Model>::n_children;

  std::size_t ps = AdaptiveCellCluster::slab_of(parent_g);
  std::size_t pslot = AdaptiveCellCluster::slot_of(parent_g);
  auto& pslab = cluster.slab(ps);
  double px = pslab.pos[0 * CellSlab::CAPACITY + pslot];
  double py = pslab.pos[1 * CellSlab::CAPACITY + pslot];
  double pz = pslab.pos[2 * CellSlab::CAPACITY + pslot];

  if constexpr (Model == SplitModel::Odd27) {
    double const h3 = (d1 / 3.0) / std::sqrt(3.0);
    for (std::uint8_t c = 0; c < nc; ++c) {
      std::int32_t cg = fc + c;
      std::size_t cs = AdaptiveCellCluster::slab_of(cg);
      std::size_t cslot = AdaptiveCellCluster::slot_of(cg);
      auto& cslab = cluster.slab(cs);
      auto const& off = odd27_table.offset[c];
      cslab.pos[0 * CellSlab::CAPACITY + cslot] = px + off[0] * h3;
      cslab.pos[1 * CellSlab::CAPACITY + cslot] = py + off[1] * h3;
      cslab.pos[2 * CellSlab::CAPACITY + cslot] = pz + off[2] * h3;

      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        auto const t = odd27_table.face_target[c][k];
        cluster[cg].nb[k] =
            (t < ODD27_EXTERNAL_BASE) ? (fc + t) : par.nb[t - ODD27_EXTERNAL_BASE];
      }
    }
    _prolong_u_gradient_wide(cluster, parent_g, par.u, px, py, pz, fc, nc);
    _prolong_phi_generic(cluster, par, fc, nc);
    return;
  }

  constexpr bool has_center = (Model == SplitModel::HexCenter9);
  double h_child = (d1 / 2.0) / std::sqrt(3.0);
  std::uint8_t const n_outer = 8;  // octant (cube-encoded) children — always 8
  std::int32_t const center_g = fc + adaptive_cell::CENTER_CHILD_INDEX;  // valid iff has_center

  // Step 1: child positions; nb[] already -1 from split().
  // Outer (octant) children sit at parent_centre + child_octant[c]*h_child.
  // The center child (9-split only) sits AT the parent's own position, at the
  // halved scale — the original cell's position becomes a genuine corner site
  // of the refined (finer) BCC lattice, not merely a retained coarse stand-in.
  for (std::uint8_t c = 0; c < n_outer; ++c) {
    std::int32_t cg = fc + c;
    std::size_t cs = AdaptiveCellCluster::slab_of(cg);
    std::size_t cslot = AdaptiveCellCluster::slot_of(cg);
    auto& cslab = cluster.slab(cs);
    auto& oct = child_octant[c];
    cslab.pos[0 * CellSlab::CAPACITY + cslot] = px + oct[0] * h_child;
    cslab.pos[1 * CellSlab::CAPACITY + cslot] = py + oct[1] * h_child;
    cslab.pos[2 * CellSlab::CAPACITY + cslot] = pz + oct[2] * h_child;
  }
  if (has_center) {
    std::size_t cs = AdaptiveCellCluster::slab_of(center_g);
    std::size_t cslot = AdaptiveCellCluster::slot_of(center_g);
    auto& cslab = cluster.slab(cs);
    cslab.pos[0 * CellSlab::CAPACITY + cslot] = px;
    cslab.pos[1 * CellSlab::CAPACITY + cslot] = py;
    cslab.pos[2 * CellSlab::CAPACITY + cslot] = pz;
  }

  // Step 2: square-sibling connections (outer children only — the center sits
  // at hex, not square, distance from every outer child; it has no sq sibling
  // within this split).
  // Children are at square-face distance along each axis when their octants
  // differ in exactly one component.  For axis j: child c connects to
  // flip_child(c,j) on sq face (8 + 2j + (oct[j]+1)/2).
  for (std::uint8_t c = 0; c < n_outer; ++c) {
    std::int32_t cg = fc + c;
    for (int j = 0; j < 3; ++j) {
      std::uint8_t sib = flip_child(c, j);
      int8_t sign = child_octant[c][j];  // +1 or -1
      // sq face in direction of the sibling: if sign == -1, sibling is in +j dir.
      std::uint8_t face = sq_face_for_axis[j][sign == -1 ? 1 : 0];
      cluster[cg].nb[face] = fc + sib;
    }
  }

  // Step 3: hex outer connections.
  // Child k is geometrically adjacent to the coarse neighbour in direction k → par.nb[k].
  // Child mirror_face[k]'s hex-k direction points back toward the parent centre:
  //   - has_center: that's a genuine, live neighbour — the center child, which
  //     itself sits exactly hex-distance away in every one of the 8 directions.
  //   - !has_center (legacy 8-split): no cell occupies that position, so the
  //     face is dead (nb[k] = -1, no active flux) — this is the gap the center
  //     child exists to close.
  // All other (non-mirror, non-center) outer children inherit par.nb[k] as a
  // coarse proxy (standard AMR coarse-fine approximation — only child k is
  // truly geometrically adjacent to par.nb[k], but this is unchanged from the
  // pre-existing approximation and not something the center child addresses).
  // nb[k] = -1 only when the parent itself has no hex-k neighbour (sphere boundary).
  for (std::size_t k = 0; k < 8; ++k) {
    std::int32_t nb_k = par.nb[k];
    for (std::uint8_t c = 0; c < n_outer; ++c) {
      if (c == bcc::mirror_face[k])
        cluster[fc + c].nb[k] = has_center ? center_g : -1;
      else
        cluster[fc + c].nb[k] = nb_k;
    }
    if (has_center) cluster[center_g].nb[k] = fc + k;  // center faces outer child k directly
  }

  // Step 4: outer sq faces — inherit parent's sq proxy neighbour.
  // Step 2 wired intra-sibling sq faces (nb[k] = fc+sib >= 0).
  // Any sq face still -1 points outside the split zone; wire to parent's nb.
  // The center child (9-split) was never touched by steps 1-3 on sq faces, so
  // it falls through here too and correctly inherits the parent's own sq
  // connectivity — it occupies the parent's old position, so its square-face
  // (next-nearest, same-sublattice) neighbours are the parent's own.
  // nb[k] = -1 only when parent has no sq-k neighbour (sphere boundary).
  for (std::size_t k = 8; k < CellSlab::NF; ++k)
    for (std::uint8_t c = 0; c < nc; ++c)
      if (cluster[fc + c].nb[k] < 0)
        cluster[fc + c].nb[k] = par.nb[k];

  // Step 5: correct split()'s constant-prolongation of phi (adaptive_lattice.hpp
  // split(): ch.phi[k] = parent.phi[k] verbatim, for every child) now that nb[]
  // wiring above tells us, per face, which children are actually the live
  // fan-out that the sum convention (phi_coarse = -Σ phi_fine, see
  // _step_cross_mirror_proxy / _step_pass2_amr_correct) sums over — and which
  // are dead, or sibling-interior (incl. center-facing) faces that must not
  // inherit the parent's outward flux at all: a brand-new interior face has no
  // flux history yet, so it starts at 0 and gets a real value from the very
  // next step's ordinary nb[]-based pass. See _prolong_phi_generic above.
  _prolong_phi_generic(cluster, par, fc, nc);
}

}  // namespace bcc
}  // namespace equilibrium::dst::discrete

#endif  // _EQUILIBRIUM_CORE_DATA_LATTICE_BCC_GEOMETRY_HPP_
