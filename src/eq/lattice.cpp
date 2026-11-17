
#include "eq/core/data/lattice/adaptive_amr.hpp"
#include "eq/core/data/lattice/adaptive_evolver.hpp"
#include "eq/core/data/lattice/adaptive_lattice.hpp"
#include "eq/core/data/lattice/bc.hpp"
#include "eq/core/data/lattice/split_criterion.hpp"
#include "eq/core/data/lattice/time_stepper.hpp"
#include "eq/core/data/lattice/bcc_geometry.hpp"
#include "eq/core/data/lattice/cell/discrete_cell.hpp"
#include "eq/core/data/lattice/cluster_io.hpp"
#include "eq/core/data/lattice/continuous/adaptive_lattice.hpp"
#include "eq/core/data/lattice/dst_info.hpp"
#include "eq/core/math/solid_angle.hpp"
#include "eq/core/phlop.hpp"

#include "highfive/H5File.hpp"

#include "mkn/kul/env.hpp"
#include "mkn/kul/os.hpp"
#include "phlop/timing/mpi_scope_timer.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace nb = nanobind;
using namespace equilibrium::dst;
using namespace equilibrium::dst::discrete;

// ---------------------------------------------------------------------------
// Numpy array helpers — pack SoA slab data into contiguous leaf-ordered arrays
// ---------------------------------------------------------------------------

using F64_1D = nb::ndarray<nb::numpy, double, nb::ndim<1>>;
using F64_2D = nb::ndarray<nb::numpy, double, nb::ndim<2>>;
using F64_2D_C = nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::c_contig>;
using I32_1D = nb::ndarray<nb::numpy, std::int32_t, nb::ndim<1>>;
using I32_2D = nb::ndarray<nb::numpy, std::int32_t, nb::ndim<2>>;
using I64_1D = nb::ndarray<nb::numpy, std::int64_t, nb::ndim<1>>;
using U8_1D = nb::ndarray<nb::numpy, std::uint8_t, nb::ndim<1>>;
using U16_1D = nb::ndarray<nb::numpy, std::uint16_t, nb::ndim<1>>;

template <typename T>
static nb::ndarray<nb::numpy, T, nb::ndim<1>> make_1d(std::size_t N) {
  T* data = new T[N];
  nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
  return nb::ndarray<nb::numpy, T, nb::ndim<1>>(data, {N}, owner);
}

template <typename T>
static nb::ndarray<nb::numpy, T, nb::ndim<2>> make_2d(std::size_t rows, std::size_t cols) {
  T* data = new T[rows * cols];
  nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
  std::size_t shape[2] = {rows, cols};
  return nb::ndarray<nb::numpy, T, nb::ndim<2>>(data, 2, shape, owner);
}

static F64_1D cluster_leaf_u(AdaptiveCellCluster& cl) {
  std::size_t N = cl.leaf_indices.size();
  auto arr = make_1d<double>(N);
  for (std::size_t j = 0; j < N; ++j) arr.data()[j] = cl[cl.leaf_indices[j]].u;
  return arr;
}

static void cluster_set_leaf_u(AdaptiveCellCluster& cl, F64_1D arr) {
  if (arr.shape(0) != cl.leaf_indices.size()) throw std::runtime_error("set_leaf_u: size mismatch");
  for (std::size_t j = 0; j < cl.leaf_indices.size(); ++j) cl[cl.leaf_indices[j]].u = arr.data()[j];
}

// leaf_ell: shim — returns 1/(1+lam) for backward compatibility
static F64_1D cluster_leaf_ell(AdaptiveCellCluster& cl) {
  std::size_t N = cl.leaf_indices.size();
  auto arr = make_1d<double>(N);
  for (std::size_t j = 0; j < N; ++j) {
    double lam = cl[cl.leaf_indices[j]].lam;
    arr.data()[j] = 1.0 / (1.0 + lam);
  }
  return arr;
}

// set_leaf_ell: shim — converts ell → lam = 1/ell - 1, clamped to [0, LAM_MAX]
static void cluster_set_leaf_ell(AdaptiveCellCluster& cl, F64_1D arr) {
  if (arr.shape(0) != cl.leaf_indices.size())
    throw std::runtime_error("set_leaf_ell: size mismatch");
  constexpr double LAM_MAX = DiscreteCell::N_SAGA - 1.0;
  for (std::size_t j = 0; j < cl.leaf_indices.size(); ++j) {
    double ell = arr.data()[j];
    double lam = (ell > 0.0) ? std::clamp(1.0 / ell - 1.0, 0.0, LAM_MAX) : LAM_MAX;
    cl[cl.leaf_indices[j]].lam = lam;
  }
}

static F64_1D cluster_leaf_lam(AdaptiveCellCluster& cl) {
  std::size_t N = cl.leaf_indices.size();
  auto arr = make_1d<double>(N);
  for (std::size_t j = 0; j < N; ++j) arr.data()[j] = cl[cl.leaf_indices[j]].lam;
  return arr;
}

static void cluster_set_leaf_lam(AdaptiveCellCluster& cl, F64_1D arr) {
  if (arr.shape(0) != cl.leaf_indices.size())
    throw std::runtime_error("set_leaf_lam: size mismatch");
  for (std::size_t j = 0; j < cl.leaf_indices.size(); ++j)
    cl[cl.leaf_indices[j]].lam = arr.data()[j];
}

static F64_1D cluster_leaf_L(AdaptiveCellCluster& cl) {
  std::size_t N = cl.leaf_indices.size();
  auto arr = make_1d<double>(N);
  for (std::size_t j = 0; j < N; ++j) arr.data()[j] = cl[cl.leaf_indices[j]].L;
  return arr;
}

static void cluster_set_leaf_L(AdaptiveCellCluster& cl, F64_1D arr) {
  if (arr.shape(0) != cl.leaf_indices.size()) throw std::runtime_error("set_leaf_L: size mismatch");
  for (std::size_t j = 0; j < cl.leaf_indices.size(); ++j) cl[cl.leaf_indices[j]].L = arr.data()[j];
}

static U8_1D cluster_leaf_fixed_u(AdaptiveCellCluster& cl) {
  std::size_t N = cl.leaf_indices.size();
  auto arr = make_1d<std::uint8_t>(N);
  for (std::size_t j = 0; j < N; ++j) arr.data()[j] = cl[cl.leaf_indices[j]].fixed_u;
  return arr;
}

static void cluster_set_leaf_fixed_u(AdaptiveCellCluster& cl, U8_1D arr) {
  if (arr.shape(0) != cl.leaf_indices.size())
    throw std::runtime_error("set_leaf_fixed_u: size mismatch");
  for (std::size_t j = 0; j < cl.leaf_indices.size(); ++j)
    cl[cl.leaf_indices[j]].fixed_u = arr.data()[j];
}

// leaf_level: refinement level for each leaf j — (N,) uint8
static U8_1D cluster_leaf_level(AdaptiveCellCluster& cl) {
  std::size_t N = cl.leaf_indices.size();
  auto arr = make_1d<std::uint8_t>(N);
  for (std::size_t j = 0; j < N; ++j) arr.data()[j] = cl[cl.leaf_indices[j]].level;
  return arr;
}

// leaf_global_indices: global g for each leaf j — (N,) int32
static I32_1D cluster_leaf_global_indices(AdaptiveCellCluster& cl) {
  std::size_t N = cl.leaf_indices.size();
  auto arr = make_1d<std::int32_t>(N);
  std::copy(cl.leaf_indices.begin(), cl.leaf_indices.end(), arr.data());
  return arr;
}

// leaf_ancestry: cross-run-stable identity per leaf — (root L0 gid << 32) |
// sentinel-encoded child-index path (3 bits per level, leaf→root order).
// Geometry cannot identify a leaf across runs: adjacent split parents place
// children on the shared face midpoint, so two distinct leaves coincide in
// both position and level. Root L0 gids are deterministic for a given build,
// and the path is independent of split order — unlike child gids.
static I64_1D cluster_leaf_ancestry(AdaptiveCellCluster& cl) {
  std::size_t N = cl.leaf_indices.size();
  auto arr = make_1d<std::int64_t>(N);
  for (std::size_t j = 0; j < N; ++j) {
    std::int64_t path = 1;
    std::int32_t g = cl.leaf_indices[j];
    while (true) {
      auto const p = cl[g].parent;
      if (p < 0) break;
      path = (path << 3) | (g - cl[p].first_child);
      g = p;
    }
    arr.data()[j] = (static_cast<std::int64_t>(g) << 32) | path;
  }
  return arr;
}

// leaf_nb: shape (N, 14), global g of each face-k neighbour (−1 = no neighbour)
static I32_2D cluster_leaf_nb(AdaptiveCellCluster& cl) {
  std::size_t N = cl.leaf_indices.size();
  auto arr = make_2d<std::int32_t>(N, CellSlab::NF);
  std::int32_t* data = arr.data();
  for (std::size_t j = 0; j < N; ++j) {
    auto cell = cl[cl.leaf_indices[j]];
    for (std::size_t k = 0; k < CellSlab::NF; ++k) data[j * CellSlab::NF + k] = cell.nb[k];
  }
  return arr;
}

// leaf_effective_w2: per-leaf Σ(effective weight)² across all 14 faces,
// matching _step_pass2_lev's actual w/multiplicity computation exactly —
// temporary diagnostic (2026-07-19) to compare a boundary cell's true
// conductance against W2_SUM (the flat reference value _compute_dt_l0
// assumes), isolating whether boundary cells violate the local CFL bound
// the global dt was chosen to satisfy.
static F64_1D cluster_leaf_effective_w2(AdaptiveCellCluster& cl) {
  std::size_t N = cl.leaf_indices.size();
  auto arr = make_1d<double>(N);
  for (std::size_t j = 0; j < N; ++j) {
    auto const g    = cl.leaf_indices[j];
    auto const cell = cl[g];
    auto const L_i  = cell.L;
    double sum2 = 0.0;
    for (std::size_t k = 0; k < CellSlab::NF; ++k) {
      auto const nbg = cell.nb[k];
      if (nbg < 0) continue;
      auto nb = cl[nbg];
      double L_src = L_i;
      if (nb.L > L_i) {
        std::int32_t anc = cell.parent;
        while (anc >= 0 && cl[anc].L < nb.L) anc = cl[anc].parent;
        if (anc >= 0) L_src = cl[anc].L;
      }
      double w = bcc::face_weight[k] * 2.0 / (L_src + nb.L);
      w /= cl.face_multiplicity.get(k, N, j);
      w /= cl.ancestor_group_multiplicity.get(k, N, j);
      sum2 += w * w;
    }
    arr.data()[j] = sum2;
  }
  return arr;
}

// leaf_is_boundary / leaf_n_same_level_faces / leaf_n_cross_level_faces:
// temporary diagnostic (2026-08-04) -- boundary flag and same/cross-level
// face counts per leaf, matching _step_pass2_lev's own cell_is_boundary
// definition exactly (>=1 face whose neighbour has a different L). Used to
// test the "reduced same-level neighbour count" candidate mechanism for the
// deep-telescope H-growth (doc/dst_amr_scaling.md) -- comparing boundary
// vs. interior cells' leaf_effective_w2 (already exposed) now that boundary
// status can be determined directly instead of reconstructed in Python.
static U8_1D cluster_leaf_is_boundary(AdaptiveCellCluster& cl) {
  std::size_t N = cl.leaf_indices.size();
  auto arr = make_1d<std::uint8_t>(N);
  for (std::size_t j = 0; j < N; ++j) {
    auto const cell = cl[cl.leaf_indices[j]];
    std::uint8_t b = 0;
    for (std::size_t k = 0; k < CellSlab::NF; ++k) {
      auto const nbg = cell.nb[k];
      if (nbg >= 0 && cl[nbg].L != cell.L) { b = 1; break; }
    }
    arr.data()[j] = b;
  }
  return arr;
}

static U8_1D cluster_leaf_n_same_level_faces(AdaptiveCellCluster& cl) {
  std::size_t N = cl.leaf_indices.size();
  auto arr = make_1d<std::uint8_t>(N);
  for (std::size_t j = 0; j < N; ++j) {
    auto const cell = cl[cl.leaf_indices[j]];
    std::uint8_t n = 0;
    for (std::size_t k = 0; k < CellSlab::NF; ++k) {
      auto const nbg = cell.nb[k];
      if (nbg >= 0 && cl[nbg].L == cell.L) ++n;
    }
    arr.data()[j] = n;
  }
  return arr;
}

// leaf_face_multiplicity: shape (14, N), face-major, matches leaf_phi's layout.
// Temporary diagnostic (2026-08-04): direct per-face inspection of the
// face_multiplicity divisor (see rebuild_face_multiplicity's doc comment) --
// checking whether it is ever != 1 on a SAME-LEVEL face (nb.L == cell.L) of a
// boundary cell, which would mean the intra-leaf duplicate-face correction
// (designed for cross-level Odd27 coarse-proxy aliasing) is also, perhaps
// unintentionally, rescaling a genuine same-level exchange -- a candidate
// mechanism for the still-open deep-telescope H-growth (project_odd27_amr_
// junction_leak). See doc/dst_amr_scaling.md.
static F64_2D cluster_leaf_face_multiplicity(AdaptiveCellCluster& cl) {
  std::size_t N = cl.leaf_indices.size();
  auto arr = make_2d<double>(CellSlab::NF, N);
  double* data = arr.data();
  for (std::size_t j = 0; j < N; ++j)
    for (std::size_t k = 0; k < CellSlab::NF; ++k) data[k * N + j] = cl.face_multiplicity.get(k, N, j);
  return arr;
}

// leaf_ancestor_group_multiplicity: shape (14, N), face-major. Same rationale
// as leaf_face_multiplicity above, for the OTHER divisor
// (rebuild_ancestor_group_multiplicity).
static F64_2D cluster_leaf_ancestor_group_multiplicity(AdaptiveCellCluster& cl) {
  std::size_t N = cl.leaf_indices.size();
  auto arr = make_2d<double>(CellSlab::NF, N);
  double* data = arr.data();
  for (std::size_t j = 0; j < N; ++j)
    for (std::size_t k = 0; k < CellSlab::NF; ++k)
      data[k * N + j] = cl.ancestor_group_multiplicity.get(k, N, j);
  return arr;
}

// debug_check_boundary_same_level_multiplicity: temporary diagnostic
// (2026-08-04) -- extends the leaf_face_multiplicity/leaf_ancestor_group_
// multiplicity check to same-L neighbours that are NON-LEAF coarse proxies
// (a leaf-to-leaf-only Python-side check found zero anomalies among 164502
// connections, but skipped ~97320 same-L connections whose neighbour has
// children -- exactly the coarse-proxy-routing case Odd27's own duplication
// bug (project_odd27_amr_junction_leak Part 1) was originally found in).
// Prints a summary + up to 20 example anomalies to stdout. Not a permanent
// feature -- remove once the deep-telescope H-growth investigation
// (doc/dst_amr_scaling.md) concludes.
static void cluster_debug_check_boundary_same_level_multiplicity(AdaptiveCellCluster& cl) {
  std::size_t const L = cl.leaf_indices.size();
  long n_checked = 0, n_anomalies = 0, n_examples = 0;
  std::map<int, long> checked_by_level, anomaly_by_level;
  for (std::size_t j = 0; j < L; ++j) {
    auto const g    = cl.leaf_indices[j];
    auto const cell = cl[g];
    auto const L_i  = cell.L;
    for (std::size_t k = 0; k < CellSlab::NF; ++k) {
      auto const nbg = cell.nb[k];
      if (nbg < 0) continue;
      auto const nb = cl[nbg];
      if (nb.L != L_i) continue;  // only same-SIZE faces (leaf or non-leaf proxy)
      ++n_checked;
      checked_by_level[cell.level] += 1;
      auto const fm  = cl.face_multiplicity.get(k, L, j);
      auto const agm = cl.ancestor_group_multiplicity.get(k, L, j);
      if (std::abs(fm - 1.0) > 1e-9 || std::abs(agm - 1.0) > 1e-9) {
        ++n_anomalies;
        anomaly_by_level[cell.level] += 1;
        if (n_examples < 20) {
          std::printf("  anomaly: leaf_j=%zu g=%d k=%zu nbg=%d nb_is_leaf=%d level=%d fmult=%.6g agmult=%.6g\n",
                      j, g, k, nbg, nb.n_children == 0, static_cast<int>(cell.level), fm, agm);
          ++n_examples;
        }
      }
    }
  }
  std::printf("debug_check_boundary_same_level_multiplicity: n_checked=%ld n_anomalies=%ld\n",
              n_checked, n_anomalies);
  for (auto const& [lvl, n] : checked_by_level)
    std::printf("  level %d: checked=%ld anomalies=%ld\n", lvl, n, anomaly_by_level[lvl]);
}

// leaf_fixed_phi: fixed_phi bitmask — (N,) uint16
static U16_1D cluster_leaf_fixed_phi(AdaptiveCellCluster& cl) {
  std::size_t N = cl.leaf_indices.size();
  auto arr = make_1d<std::uint16_t>(N);
  for (std::size_t j = 0; j < N; ++j) arr.data()[j] = cl[cl.leaf_indices[j]].fixed_phi;
  return arr;
}

static void cluster_set_leaf_fixed_phi(AdaptiveCellCluster& cl, U16_1D arr) {
  if (arr.shape(0) != cl.leaf_indices.size())
    throw std::runtime_error("set_leaf_fixed_phi: size mismatch");
  for (std::size_t j = 0; j < cl.leaf_indices.size(); ++j)
    cl[cl.leaf_indices[j]].fixed_phi = arr.data()[j];
}

// pos: shape (N, 3), row-major
static F64_2D cluster_leaf_pos(AdaptiveCellCluster& cl) {
  std::size_t N = cl.leaf_indices.size();
  auto arr = make_2d<double>(N, 3);
  double* data = arr.data();
  for (std::size_t j = 0; j < N; ++j) {
    std::int32_t g = cl.leaf_indices[j];
    std::size_t s = AdaptiveCellCluster::slab_of(g);
    std::size_t slot = AdaptiveCellCluster::slot_of(g);
    auto& slab = cl.slab(s);
    data[j * 3 + 0] = slab.pos[0 * CellSlab::CAPACITY + slot];
    data[j * 3 + 1] = slab.pos[1 * CellSlab::CAPACITY + slot];
    data[j * 3 + 2] = slab.pos[2 * CellSlab::CAPACITY + slot];
  }
  return arr;
}

static F64_1D cluster_criterion_min(AdaptiveCellCluster& cl) {
  std::size_t N = cl.criterion_min.size();
  auto arr = make_1d<double>(N);
  std::copy(cl.criterion_min.begin(), cl.criterion_min.end(), arr.data());
  return arr;
}

static F64_1D cluster_criterion_max(AdaptiveCellCluster& cl) {
  std::size_t N = cl.criterion_max.size();
  auto arr = make_1d<double>(N);
  std::copy(cl.criterion_max.begin(), cl.criterion_max.end(), arr.data());
  return arr;
}

static void cluster_set_criterion_max(AdaptiveCellCluster& cl, F64_1D arr) {
  std::size_t N = cl.criterion_max.size();
  if (arr.size() != N) throw std::runtime_error("set_criterion_max: size mismatch");
  std::copy(arr.data(), arr.data() + N, cl.criterion_max.begin());
}

// phi: shape (14, N), face-major  — matches temp_phi layout
static F64_2D cluster_leaf_phi(AdaptiveCellCluster& cl) {
  std::size_t N = cl.leaf_indices.size();
  auto arr = make_2d<double>(CellSlab::NF, N);
  double* data = arr.data();
  for (std::size_t j = 0; j < N; ++j) {
    auto cell = cl[cl.leaf_indices[j]];
    for (std::size_t k = 0; k < CellSlab::NF; ++k) data[k * N + j] = cell.phi[k];
  }
  return arr;
}

static void cluster_set_leaf_phi(AdaptiveCellCluster& cl, F64_2D arr) {
  std::size_t N = cl.leaf_indices.size();
  if (arr.shape(0) != CellSlab::NF || arr.shape(1) != N)
    throw std::runtime_error("set_leaf_phi: expected shape (14, N)");
  double const* data = arr.data();
  for (std::size_t j = 0; j < N; ++j) {
    auto cell = cl[cl.leaf_indices[j]];
    for (std::size_t k = 0; k < CellSlab::NF; ++k) cell.phi[k] = data[k * N + j];
  }
}

// step_cross gate string -> CrossGate + m0, matching the Python
// SubstrateLattice.step_cross(gate="v1"|"barrier", m0=...) convention
// (equilibrium/substrate/lattice.py) -- "barrier" requires m0, no default
// (see doc/dst_next.md sec 8).
static std::pair<CrossGate, double> parse_cross_gate(std::string const& gate, nb::object m0) {
  if (gate == "v1") return {CrossGate::V1, 0.0};
  if (gate == "barrier") {
    if (m0.is_none())
      throw std::invalid_argument(
          "gate='barrier' requires m0 (no default -- see doc/dst_next.md sec 8 "
          "for per-knot calibration)");
    return {CrossGate::Barrier, nb::cast<double>(m0)};
  }
  throw std::invalid_argument("unknown gate '" + gate + "'");
}

// ---------------------------------------------------------------------------

struct Dictator {
  DSTDict dict;
};

static void bind_discrete(nb::module_& m) {
  nb::class_<BoundaryCondition>(m, "BoundaryCondition")
      .def(nb::init<>())
      // Face ghost wiring: redirect a specific boundary face to a real cell.
      // level must match the refinement level of cell_g; call sort_face_ghosts() after wiring.
      .def("wire_face", &BoundaryCondition::wire_face, nb::arg("cell_g"), nb::arg("face_k"),
           nb::arg("ghost_g"), nb::arg("level"))
      .def("unwire_face", &BoundaryCondition::unwire_face, nb::arg("cell_g"), nb::arg("face_k"),
           nb::arg("level"))
      .def("sort_face_ghosts", &BoundaryCondition::sort_face_ghosts)
      .def("clear_face_ghosts", &BoundaryCondition::clear_face_ghosts)
      // Cluster-wide fallback — set one of the built-in C++ function pointers.
      .def_static("absorbing",
                  []() {
                    BoundaryCondition bc;
                    bc.fn      = BoundaryCondition::absorbing;
                    bc.fn_type = BCType::ABSORBING;
                    return bc;
                  })
      .def_static(
          "dirichlet",
          [](double value) {
            BoundaryCondition bc;
            bc.fn      = BoundaryCondition::dirichlet;
            bc.fn_type = BCType::FUNCTIONAL;
            bc.target  = value;
            return bc;
          },
          nb::arg("value"))
      .def_static(
          "sponge",
          [](double target, double weight) {
            BoundaryCondition bc;
            bc.fn      = BoundaryCondition::sponge;
            bc.fn_type = BCType::FUNCTIONAL;
            bc.target  = target;
            bc.weight  = weight;
            return bc;
          },
          nb::arg("target") = 0.0, nb::arg("weight") = 0.5)
      .def_static(
          "quadratic_sponge",
          [](double target, std::optional<double> damp_max, F64_1D r_all, double r_inner,
             std::optional<double> gamma_max, double power) {
            if (damp_max.has_value() == gamma_max.has_value())
              throw std::invalid_argument("quadratic_sponge: give exactly one of damp_max, gamma_max");
            BoundaryCondition bc;
            bc.target = target;
            std::size_t N = r_all.shape(0);
            double r_max = 0.0;
            for (std::size_t j = 0; j < N; ++j)
              if (r_all(j) > r_max) r_max = r_all(j);
            double span = r_max - r_inner;
            auto& prof = gamma_max ? bc.vol_rate : bc.vol_damp;
            prof.resize(N);
            for (std::size_t j = 0; j < N; ++j) {
              double t = (span > 0.0 && r_all(j) > r_inner) ? (r_all(j) - r_inner) / span : 0.0;
              if (t > 1.0) t = 1.0;
              prof[j] = gamma_max ? *gamma_max * std::pow(t, power) : 1.0 - *damp_max * std::pow(t, power);
            }
            return bc;
          },
          nb::arg("target") = 0.0, nb::arg("damp_max") = nb::none(), nb::arg("r_all"),
          nb::arg("r_inner"), nb::arg("gamma_max") = nb::none(), nb::arg("power") = 2.0,
          "Volumetric sponge over r > r_inner, profile t^power: gamma_max = rate (1/time, "
          "step_sponge(i, dt)); damp_max = legacy per-call fraction. Design rule "
          "(exp/2026/09/29/bc_design): gamma_max ~ 10 c / (r_max - r_inner).")
      .def_static(
          "quadratic_sponge_integrated",
          [](double target, double gamma_max, F64_1D r_all, double r_inner) {
            BoundaryCondition bc;
            bc.target = target;
            std::size_t N = r_all.shape(0);
            double r_max = 0.0;
            for (std::size_t j = 0; j < N; ++j)
              if (r_all(j) > r_max) r_max = r_all(j);
            double span = r_max - r_inner;
            bc.gamma.resize(N);
            for (std::size_t j = 0; j < N; ++j) {
              double t = (span > 0.0 && r_all(j) > r_inner) ? (r_all(j) - r_inner) / span : 0.0;
              if (t > 1.0) t = 1.0;
              bc.gamma[j] = gamma_max * t * t;
            }
            return bc;
          },
          nb::arg("target") = 0.0, nb::arg("gamma_max") = 0.003, nb::arg("r_all"),
          nb::arg("r_inner"))
      .def_static(
          "quadratic_sponge_directional",
          [](double target, double gamma_max, F64_2D pos_all, double r_inner) {
            BoundaryCondition bc;
            bc.target = target;
            std::size_t N = pos_all.shape(0);
            double r_max  = 0.0;
            std::vector<double> r_all(N);
            for (std::size_t j = 0; j < N; ++j) {
              double x = pos_all(j, 0), y = pos_all(j, 1), z = pos_all(j, 2);
              double r = std::sqrt(x * x + y * y + z * z);
              r_all[j] = r;
              if (r > r_max) r_max = r;
            }
            double span = r_max - r_inner;
            // Canonical unit face directions -- level/position-independent (BCC
            // shares the same 14 directions everywhere): hex faces (±1,±1,±1)
            // normalized by sqrt(3), sq faces already unit (±1,0,0) etc.
            std::array<std::array<double, 3>, CellSlab::NF> unit_dir{};
            for (std::size_t k = 0; k < 8; ++k)
              for (int i = 0; i < 3; ++i) unit_dir[k][i] = bcc::hex_dir[k][i] / std::sqrt(3.0);
            for (std::size_t k = 8; k < CellSlab::NF; ++k)
              for (int i = 0; i < 3; ++i) unit_dir[k][i] = static_cast<double>(bcc::sq_dir[k - 8][i]);

            bc.gamma_face.assign(CellSlab::NF * N, 0.0);
            for (std::size_t j = 0; j < N; ++j) {
              double const r = r_all[j];
              double t = (span > 0.0 && r > r_inner) ? (r - r_inner) / span : 0.0;
              if (t > 1.0) t = 1.0;
              if (t <= 0.0) continue;
              double const r_hat[3] = {r > 0.0 ? pos_all(j, 0) / r : 0.0,
                                        r > 0.0 ? pos_all(j, 1) / r : 0.0,
                                        r > 0.0 ? pos_all(j, 2) / r : 0.0};
              for (std::size_t k = 0; k < CellSlab::NF; ++k) {
                double const cos_ang = unit_dir[k][0] * r_hat[0] + unit_dir[k][1] * r_hat[1] +
                                        unit_dir[k][2] * r_hat[2];
                bc.gamma_face[k * N + j] = gamma_max * t * t * cos_ang * cos_ang;
              }
            }
            return bc;
          },
          nb::arg("target") = 0.0, nb::arg("gamma_max") = 0.003, nb::arg("pos_all"),
          nb::arg("r_inner"))
      .def_static(
          "quadratic_sponge_pml",
          [](double target, double gamma_max, F64_2D pos_all, double r_inner) {
            BoundaryCondition bc;
            bc.target = target;
            std::size_t N = pos_all.shape(0);
            double r_max  = 0.0;
            std::vector<double> r_all(N);
            for (std::size_t j = 0; j < N; ++j) {
              double x = pos_all(j, 0), y = pos_all(j, 1), z = pos_all(j, 2);
              double r = std::sqrt(x * x + y * y + z * z);
              r_all[j] = r;
              if (r > r_max) r_max = r;
            }
            double span = r_max - r_inner;
            std::array<std::array<double, 3>, CellSlab::NF> unit_dir{};
            for (std::size_t k = 0; k < 8; ++k)
              for (int i = 0; i < 3; ++i) unit_dir[k][i] = bcc::hex_dir[k][i] / std::sqrt(3.0);
            for (std::size_t k = 8; k < CellSlab::NF; ++k)
              for (int i = 0; i < 3; ++i) unit_dir[k][i] = static_cast<double>(bcc::sq_dir[k - 8][i]);

            bc.gamma.assign(N, 0.0);
            bc.proj_r.assign(CellSlab::NF * N, 0.0);
            bc.u_aux.assign(N, 0.0);
            bc.phi_aux.assign(CellSlab::NF * N, 0.0);
            for (std::size_t j = 0; j < N; ++j) {
              double const r = r_all[j];
              double t = (span > 0.0 && r > r_inner) ? (r - r_inner) / span : 0.0;
              if (t > 1.0) t = 1.0;
              bc.gamma[j] = gamma_max * t * t;
              double const r_hat[3] = {r > 0.0 ? pos_all(j, 0) / r : 0.0,
                                        r > 0.0 ? pos_all(j, 1) / r : 0.0,
                                        r > 0.0 ? pos_all(j, 2) / r : 0.0};
              for (std::size_t k = 0; k < CellSlab::NF; ++k) {
                double const cos_ang = unit_dir[k][0] * r_hat[0] + unit_dir[k][1] * r_hat[1] +
                                        unit_dir[k][2] * r_hat[2];
                bc.proj_r[k * N + j] = cos_ang * cos_ang;
              }
            }
            return bc;
          },
          nb::arg("target") = 0.0, nb::arg("gamma_max") = 0.003, nb::arg("pos_all"),
          nb::arg("r_inner"))
      .def_static(
          "outflow",
          [](I32_2D nb_all, double c_lattice) {
            BoundaryCondition bc;
            std::size_t N = nb_all.shape(0);
            bc.outflow_active.assign(CellSlab::NF * N, 0);
            bc.outflow_ghost.assign(CellSlab::NF * N, 0.0);
            bc.outflow_c_lattice = c_lattice;
            for (std::size_t j = 0; j < N; ++j)
              for (std::size_t k = 0; k < CellSlab::NF; ++k)
                if (nb_all(j, k) < 0) bc.outflow_active[k * N + j] = 1;
            return bc;
          },
          nb::arg("nb_all"), nb::arg("c_lattice") = bcc::wave_speed())
      .def_static(
          "outflow_tapered",
          [](I32_2D nb_all, F64_2D pos_all, double taper_width, double c_lattice) {
            BoundaryCondition bc;
            std::size_t N = nb_all.shape(0);
            bc.outflow_active.assign(CellSlab::NF * N, 0);
            bc.outflow_ghost.assign(CellSlab::NF * N, 0.0);
            bc.outflow_c_lattice = c_lattice;
            for (std::size_t j = 0; j < N; ++j)
              for (std::size_t k = 0; k < CellSlab::NF; ++k)
                if (nb_all(j, k) < 0) bc.outflow_active[k * N + j] = 1;

            double r_max = 0.0;
            std::vector<double> r_all(N);
            for (std::size_t j = 0; j < N; ++j) {
              double x = pos_all(j, 0), y = pos_all(j, 1), z = pos_all(j, 2);
              double r = std::sqrt(x * x + y * y + z * z);
              r_all[j] = r;
              if (r > r_max) r_max = r;
            }
            bc.outflow_taper_weight.assign(N, 0.0);
            bc.outflow_taper_ghost.assign(N, 0.0);
            if (taper_width > 0.0) {
              for (std::size_t j = 0; j < N; ++j) {
                // Skip true boundary leaves (already fully handled by the
                // exact ghost mechanism via outflow_active) -- applying the
                // taper sink there too double-counts absorption at r_max and
                // was confirmed empirically to make even the easy isotropic
                // case worse (2026-09-16 sanity check).
                bool is_boundary = false;
                for (std::size_t k = 0; k < CellSlab::NF; ++k)
                  if (nb_all(j, k) < 0) { is_boundary = true; break; }
                if (is_boundary) continue;
                double t = (r_all[j] - (r_max - taper_width)) / taper_width;
                if (t < 0.0) t = 0.0;
                if (t > 1.0) t = 1.0;
                bc.outflow_taper_weight[j] = t * t;
              }
            }
            return bc;
          },
          nb::arg("nb_all"), nb::arg("pos_all"), nb::arg("taper_width") = 0.0,
          nb::arg("c_lattice") = bcc::wave_speed())
      .def_static("neumann",
                  []() {
                    BoundaryCondition bc;
                    // galactic_flux=0, fn=nullptr → dphi=0 on all boundary faces
                    return bc;
                  })
      .def_static(
          "remote",
          [](double flux, nb::ndarray<double, nb::shape<3>> axis) {
            BoundaryCondition bc;
            bc.remote_flux = flux;
            bc.remote_axis = {axis(0), axis(1), axis(2)};
            bc.fn_type     = BCType::MPI_REMOTE;
            return bc;
          },
          nb::arg("flux"), nb::arg("axis"))
      .def_rw("remote_flux", &BoundaryCondition::remote_flux)
      .def_prop_rw(
          "remote_axis",
          [](BoundaryCondition const& bc) {
            return std::vector<double>(bc.remote_axis.begin(), bc.remote_axis.end());
          },
          [](BoundaryCondition& bc, nb::ndarray<double, nb::shape<3>> v) {
            bc.remote_axis = {v(0), v(1), v(2)};
          });

  nb::class_<SplitCriterion>(m, "SplitCriterion")
      .def(nb::init<>())
      .def_rw("hi", &SplitCriterion::hi)
      .def_rw("lo", &SplitCriterion::lo)
      .def_rw("ref", &SplitCriterion::ref)
      // Preset factories — mirror BoundaryCondition::absorbing() etc.
      .def_static(
          "absolute",
          [](double hi, double lo, double ref) {
            return SplitCriterion::make_absolute(hi, lo, ref);
          },
          nb::arg("hi"), nb::arg("lo") = 0.0, nb::arg("ref") = 0.0)
      .def_static(
          "gradient",
          [](double hi, double lo, double ref) {
            return SplitCriterion::make_gradient(hi, lo, ref);
          },
          nb::arg("hi"), nb::arg("lo") = 0.0, nb::arg("ref") = 0.0)
      .def_static(
          "relative_u",
          [](double hi, double lo, double ref) {
            return SplitCriterion::make_relative_u(hi, lo, ref);
          },
          nb::arg("hi"), nb::arg("lo") = 0.0, nb::arg("ref") = 0.0)
      // Dict-based construction — same pattern as cluster.configure(dictator).
      .def_static(
          "from_dict", [](Dictator const& d) { return SplitCriterion::FROM(d.dict); },
          nb::arg("dictator"));

  nb::class_<DSTInfo>(m, "DSTInfo")
      .def(nb::init<>())
      .def_rw("split_hi", &DSTInfo::split_hi)
      .def_rw("split_lo", &DSTInfo::split_lo)
      .def_rw("max_level", &DSTInfo::max_level)
      .def_rw("split_interval", &DSTInfo::split_interval);

  nb::class_<AdaptiveCellCluster>(m, "AdaptiveCellCluster")
      .def_rw("info", &AdaptiveCellCluster::info)
      .def(
          "configure",
          [](AdaptiveCellCluster& cl, Dictator const& d) { cl.info = DSTInfo::FROM(d.dict); },
          nb::arg("dictator"))
      .def_prop_ro("n_leaves", [](AdaptiveCellCluster& cl) { return cl.leaf_indices.size(); })
      // leaf data — read (take_ownership: capsule-backed arrays must not also
      // be reference_internal since they already have an owner)
      .def_prop_ro("leaf_level", cluster_leaf_level, nb::rv_policy::take_ownership)
      .def_prop_ro("leaf_global_indices", cluster_leaf_global_indices,
                   nb::rv_policy::take_ownership)
      .def_prop_ro("leaf_ancestry", cluster_leaf_ancestry, nb::rv_policy::take_ownership)
      .def_prop_ro("leaf_u", cluster_leaf_u, nb::rv_policy::take_ownership)
      .def_prop_ro("leaf_ell", cluster_leaf_ell, nb::rv_policy::take_ownership)
      .def_prop_ro("leaf_lam", cluster_leaf_lam, nb::rv_policy::take_ownership)
      .def_prop_ro("leaf_L", cluster_leaf_L, nb::rv_policy::take_ownership)
      .def_prop_ro("leaf_fixed_u", cluster_leaf_fixed_u, nb::rv_policy::take_ownership)
      .def_prop_ro("leaf_nb", cluster_leaf_nb, nb::rv_policy::take_ownership)
      .def_prop_ro("leaf_effective_w2", cluster_leaf_effective_w2, nb::rv_policy::take_ownership)
      .def_prop_ro("leaf_is_boundary", cluster_leaf_is_boundary, nb::rv_policy::take_ownership)
      .def_prop_ro("leaf_n_same_level_faces", cluster_leaf_n_same_level_faces,
                   nb::rv_policy::take_ownership)
      .def_prop_ro("leaf_face_multiplicity", cluster_leaf_face_multiplicity,
                   nb::rv_policy::take_ownership)
      .def_prop_ro("leaf_ancestor_group_multiplicity", cluster_leaf_ancestor_group_multiplicity,
                   nb::rv_policy::take_ownership)
      .def("debug_check_boundary_same_level_multiplicity",
           cluster_debug_check_boundary_same_level_multiplicity)
      .def_prop_ro("leaf_fixed_phi", cluster_leaf_fixed_phi, nb::rv_policy::take_ownership)
      .def_prop_ro("leaf_pos", cluster_leaf_pos, nb::rv_policy::take_ownership)
      .def_prop_ro("leaf_phi", cluster_leaf_phi, nb::rv_policy::take_ownership)
      .def_prop_ro("criterion_min", cluster_criterion_min, nb::rv_policy::take_ownership)
      .def_prop_ro("criterion_max", cluster_criterion_max, nb::rv_policy::take_ownership)
      .def("set_criterion_max", cluster_set_criterion_max, nb::arg("arr"))
      // leaf data — write
      .def("set_leaf_u", cluster_set_leaf_u, nb::arg("arr"))
      .def("set_leaf_ell", cluster_set_leaf_ell, nb::arg("arr"))
      .def("set_leaf_lam", cluster_set_leaf_lam, nb::arg("arr"))
      .def("set_leaf_L", cluster_set_leaf_L, nb::arg("arr"))
      .def("set_leaf_fixed_u", cluster_set_leaf_fixed_u, nb::arg("arr"))
      .def("set_leaf_phi", cluster_set_leaf_phi, nb::arg("arr"))
      .def("set_leaf_fixed_phi", cluster_set_leaf_fixed_phi, nb::arg("arr"))
      // boundary leaf positions (indices into leaf_u / leaf_pos etc.)
      .def_prop_ro(
          "boundary_leaf_positions",
          [](AdaptiveCellCluster& cl) -> I64_1D {
            std::size_t N = cl.boundary_leaf_positions.size();
            auto arr = make_1d<std::int64_t>(N);
            for (std::size_t i = 0; i < N; ++i)
              arr.data()[i] = static_cast<std::int64_t>(cl.boundary_leaf_positions[i]);
            return arr;
          },
          nb::rv_policy::take_ownership)
      // BC configuration
      .def(
          "set_bc",
          [](AdaptiveCellCluster& cl, BoundaryCondition& bc) {
            cl.bc = std::make_unique<BoundaryCondition>(bc);
          },
          nb::arg("bc"))
      .def("clear_bc", [](AdaptiveCellCluster& cl) { cl.bc.reset(); })
      // Split criterion — pluggable scale-invariant refinement decision.
      .def(
          "set_split_criterion",
          [](AdaptiveCellCluster& cl, SplitCriterion& sc) {
            cl.split_criterion = std::make_unique<SplitCriterion>(sc);
          },
          nb::arg("criterion"))
      .def("clear_split_criterion", [](AdaptiveCellCluster& cl) { cl.split_criterion.reset(); })
      // Convenience: wire/unwire a single face ghost without touching bc.fn.
      // Level is looked up automatically from the cluster; call sort_face_ghosts() after wiring.
      .def(
          "set_nb",
          [](AdaptiveCellCluster& cl, std::int32_t g, std::size_t k, std::int32_t partner_g) {
            cl[g].nb[k] = partner_g;
          },
          nb::arg("cell_g"), nb::arg("face_k"), nb::arg("partner_g"))
      .def(
          "wire_face",
          [](AdaptiveCellCluster& cl, std::int32_t g, std::size_t k, std::int32_t ghost_g) {
            if (!cl.bc) cl.bc = std::make_unique<BoundaryCondition>();
            cl.bc->wire_face(g, k, ghost_g, cl[g].level);
          },
          nb::arg("cell_g"), nb::arg("face_k"), nb::arg("ghost_g"))
      .def(
          "unwire_face",
          [](AdaptiveCellCluster& cl, std::int32_t g, std::size_t k) {
            if (cl.bc) cl.bc->unwire_face(g, k, cl[g].level);
          },
          nb::arg("cell_g"), nb::arg("face_k"))
      .def(
          "sort_face_ghosts",
          [](AdaptiveCellCluster& cl) {
            if (cl.bc) cl.bc->sort_face_ghosts();
          });

  nb::class_<AdaptiveLattice>(m, "AdaptiveLattice")
      .def(nb::init<>())
      .def_static("set_disable_ancestor_u_src",
                  [](bool v) { AdaptiveLattice::disable_ancestor_u_src = v; })
      .def_static("get_disable_ancestor_u_src",
                  []() { return AdaptiveLattice::disable_ancestor_u_src; })
      .def_static("set_disable_multiplicity_correction",
                  [](bool v) { AdaptiveLattice::disable_multiplicity_correction = v; })
      .def_static("get_disable_multiplicity_correction",
                  []() { return AdaptiveLattice::disable_multiplicity_correction; })
      .def_static("set_enable_ancestor_debias_correction",
                  [](bool v) { AdaptiveLattice::enable_ancestor_debias_correction = v; })
      .def_static("get_enable_ancestor_debias_correction",
                  []() { return AdaptiveLattice::enable_ancestor_debias_correction; })
      .def_static("set_disable_step_pass2_amr_correct",
                  [](bool v) { AdaptiveLattice::disable_step_pass2_amr_correct = v; })
      .def_static("get_disable_step_pass2_amr_correct",
                  []() { return AdaptiveLattice::disable_step_pass2_amr_correct; })
      .def_static("set_use_recursive_mean_sigma",
                  [](bool v) { AdaptiveLattice::use_recursive_mean_sigma = v; })
      .def_static("get_use_recursive_mean_sigma",
                  []() { return AdaptiveLattice::use_recursive_mean_sigma; })
      .def_static("set_step_cross_level_scale_power",
                  [](double v) { AdaptiveLattice::step_cross_level_scale_power = v; })
      .def_static("get_step_cross_level_scale_power",
                  []() { return AdaptiveLattice::step_cross_level_scale_power; })
      .def_static("set_cross_level_flux_scale",
                  [](double v) { AdaptiveLattice::cross_level_flux_scale = v; })
      .def_static("get_cross_level_flux_scale",
                  []() { return AdaptiveLattice::cross_level_flux_scale; })
      .def_prop_ro("n_clusters", [](AdaptiveLattice& lat) { return lat.clusters.size(); })
      .def("add_cluster",
           [](AdaptiveLattice& lat) {
             lat.clusters.emplace_back();
             return lat.clusters.size() - 1;  // return index of new cluster
           })
      .def(
          "__getitem__",
          [](AdaptiveLattice& lat, std::size_t i) -> AdaptiveCellCluster& {
            if (i >= lat.clusters.size()) throw std::out_of_range("cluster index out of range");
            return lat.clusters[i];
          },
          nb::rv_policy::reference_internal)
      .def(
          "set_n_threads", [](AdaptiveLattice& lat, std::size_t n) { lat.set_n_threads(n); },
          nb::arg("n"))
      .def(
          "step",
          [](AdaptiveLattice& lat, std::size_t i, double dt) {
            if (i >= lat.clusters.size()) throw std::out_of_range("cluster index out of range");
            nb::gil_scoped_release release;
            lat.step(lat.clusters[i], dt);
          },
          nb::arg("i"), nb::arg("dt"))
      .def(
          "step_all",
          [](AdaptiveLattice& lat, double dt) {
            nb::gil_scoped_release release;
            lat.step_all(dt);
          },
          nb::arg("dt"))
      .def(
          "step_damped",
          [](AdaptiveLattice& lat, std::size_t i, double dt) {
            if (i >= lat.clusters.size()) throw std::out_of_range("cluster index out of range");
            nb::gil_scoped_release release;
            lat.step_damped(lat.clusters[i], dt);
          },
          nb::arg("i"), nb::arg("dt"))
      .def(
          "step_all_damped",
          [](AdaptiveLattice& lat, double dt) {
            nb::gil_scoped_release release;
            lat.step_all_damped(dt);
          },
          nb::arg("dt"))
      .def(
          "step_dir_damped",
          [](AdaptiveLattice& lat, std::size_t i, double dt) {
            if (i >= lat.clusters.size()) throw std::out_of_range("cluster index out of range");
            nb::gil_scoped_release release;
            lat.step_dir_damped(lat.clusters[i], dt);
          },
          nb::arg("i"), nb::arg("dt"))
      .def(
          "step_all_dir_damped",
          [](AdaptiveLattice& lat, double dt) {
            nb::gil_scoped_release release;
            lat.step_all_dir_damped(dt);
          },
          nb::arg("dt"))
      .def(
          "step_pml",
          [](AdaptiveLattice& lat, std::size_t i, double dt) {
            if (i >= lat.clusters.size()) throw std::out_of_range("cluster index out of range");
            nb::gil_scoped_release release;
            lat.step_pml(lat.clusters[i], dt);
          },
          nb::arg("i"), nb::arg("dt"))
      .def(
          "step_all_pml",
          [](AdaptiveLattice& lat, double dt) {
            nb::gil_scoped_release release;
            lat.step_all_pml(dt);
          },
          nb::arg("dt"))
      .def(
          "step_lam",
          [](AdaptiveLattice& lat, std::size_t i, double dt, double scale) {
            if (i >= lat.clusters.size()) throw std::out_of_range("cluster index out of range");
            nb::gil_scoped_release release;
            lat.step_lam(lat.clusters[i], dt, scale);
          },
          nb::arg("i"), nb::arg("dt"), nb::arg("scale") = 1.0)
      .def(
          "step_lam_all",
          [](AdaptiveLattice& lat, double dt, double scale) {
            nb::gil_scoped_release release;
            lat.step_lam_all(dt, scale);
          },
          nb::arg("dt"), nb::arg("scale") = 1.0)
      .def(
          "step_cross",
          [](AdaptiveLattice& lat, std::size_t i, double dt, double scale,
             std::string const& gate, nb::object m0) {
            if (i >= lat.clusters.size()) throw std::out_of_range("cluster index out of range");
            auto const [g, m] = parse_cross_gate(gate, m0);
            nb::gil_scoped_release release;
            lat.step_cross(lat.clusters[i], dt, scale, g, m);
          },
          nb::arg("i"), nb::arg("dt"), nb::arg("scale") = 1.0, nb::arg("gate") = "v1",
          nb::arg("m0") = nb::none())
      .def(
          "step_cross_all",
          [](AdaptiveLattice& lat, double dt, double scale, std::string const& gate,
             nb::object m0) {
            auto const [g, m] = parse_cross_gate(gate, m0);
            nb::gil_scoped_release release;
            lat.step_cross_all(dt, scale, g, m);
          },
          nb::arg("dt"), nb::arg("scale") = 1.0, nb::arg("gate") = "v1", nb::arg("m0") = nb::none())
      .def(
          "step_sponge",
          [](AdaptiveLattice& lat, std::size_t i, double dt) {
            if (i >= lat.clusters.size()) throw std::out_of_range("cluster index out of range");
            nb::gil_scoped_release release;
            lat.step_sponge(lat.clusters[i], dt);
          },
          nb::arg("i"), nb::arg("dt") = 0.0)
      .def(
          "step_sponge_all",
          [](AdaptiveLattice& lat, double dt) {
            nb::gil_scoped_release release;
            lat.step_sponge_all(dt);
          },
          nb::arg("dt") = 0.0)
      .def(
          "step_boundary_hyper_resistivity",
          [](AdaptiveLattice& lat, std::size_t i, double dt, double eta_h, double ratio,
             bool imex) {
            if (i >= lat.clusters.size()) throw std::out_of_range("cluster index out of range");
            nb::gil_scoped_release release;
            lat.step_boundary_hyper_resistivity(lat.clusters[i], dt, eta_h, ratio, imex);
          },
          nb::arg("i"), nb::arg("dt"), nb::arg("eta_h"), nb::arg("ratio") = 3.0,
          nb::arg("imex") = false)
      .def(
          "step_boundary_hyper_resistivity_all",
          [](AdaptiveLattice& lat, double dt, double eta_h, double ratio, bool imex) {
            nb::gil_scoped_release release;
            lat.step_boundary_hyper_resistivity_all(dt, eta_h, ratio, imex);
          },
          nb::arg("dt"), nb::arg("eta_h"), nb::arg("ratio") = 3.0, nb::arg("imex") = false)
      .def(
          "step_boundary_hyper_resistivity_per_level",
          [](AdaptiveLattice& lat, std::size_t i, double dt,
             std::vector<double> const& eta_h_by_level, bool imex) {
            if (i >= lat.clusters.size()) throw std::out_of_range("cluster index out of range");
            nb::gil_scoped_release release;
            lat.step_boundary_hyper_resistivity_per_level(lat.clusters[i], dt, eta_h_by_level,
                                                            imex);
          },
          nb::arg("i"), nb::arg("dt"), nb::arg("eta_h_by_level"), nb::arg("imex") = false)
      .def(
          "step_boundary_hyper_resistivity_per_level_all",
          [](AdaptiveLattice& lat, double dt, std::vector<double> const& eta_h_by_level,
             bool imex) {
            nb::gil_scoped_release release;
            lat.step_boundary_hyper_resistivity_per_level_all(dt, eta_h_by_level, imex);
          },
          nb::arg("dt"), nb::arg("eta_h_by_level"), nb::arg("imex") = false)
      .def(
          "step_boundary_hyper_resistivity_ratio",
          [](AdaptiveLattice& lat, std::size_t i, std::vector<double> const& ratio_by_level) {
            if (i >= lat.clusters.size()) throw std::out_of_range("cluster index out of range");
            nb::gil_scoped_release release;
            lat.step_boundary_hyper_resistivity_ratio(lat.clusters[i], ratio_by_level);
          },
          nb::arg("i"), nb::arg("ratio_by_level"))
      .def(
          "step_boundary_hyper_resistivity_ratio_all",
          [](AdaptiveLattice& lat, std::vector<double> const& ratio_by_level) {
            nb::gil_scoped_release release;
            lat.step_boundary_hyper_resistivity_ratio_all(ratio_by_level);
          },
          nb::arg("ratio_by_level"))
      .def(
          "rebuild",
          [](AdaptiveLattice& lat, std::size_t i) {
            if (i >= lat.clusters.size()) throw std::out_of_range("cluster index out of range");
            lat.rebuild_leaf_indices(lat.clusters[i]);
          },
          nb::arg("i"))
      .def("rebuild_all", &AdaptiveLattice::rebuild_all)
      .def(
          "adapt",
          [](AdaptiveLattice& lat, std::size_t i) {
            if (i >= lat.clusters.size()) throw std::out_of_range("cluster index out of range");
            nb::gil_scoped_release release;
            adapt(lat, lat.clusters[i]);
          },
          nb::arg("i"))
      .def("adapt_all",
           [](AdaptiveLattice& lat) {
             nb::gil_scoped_release release;
             adapt_all(lat);
           })
      .def(
          "split",
          [](AdaptiveLattice& lat, std::size_t i, std::int32_t cell_g, double d1) {
            if (i >= lat.clusters.size()) throw std::out_of_range("cluster index out of range");
            auto& cl = lat.clusters[i];
            auto result = lat.split(cl, cell_g);
            bcc::wire_split_children(cl, cell_g, result, d1);
            lat.rebuild_leaf_indices(cl);
            return result.first_child;
          },
          nb::arg("i"), nb::arg("cell_g"), nb::arg("d1") = 1.0)
      .def(
          "split_batch",
          [](AdaptiveLattice& lat, std::size_t i, I32_1D gs, double d1) {
            if (i >= lat.clusters.size()) throw std::out_of_range("cluster index out of range");
            auto& cl = lat.clusters[i];
            std::size_t n_split = 0;
            for (std::size_t k = 0; k < gs.shape(0); ++k) {
              std::int32_t g = gs.data()[k];
              if (!lat.can_split(cl, g)) continue;
              auto result = lat.split(cl, g);
              bcc::wire_split_children(cl, g, result, d1);
              ++n_split;
            }
            if (n_split > 0) lat.rebuild_leaf_indices(cl);
            return n_split;
          },
          nb::arg("i"), nb::arg("gs"), nb::arg("d1") = 1.0)
      .def(
          "coarsen",
          [](AdaptiveLattice& lat, std::size_t i, std::int32_t cell_g) {
            if (i >= lat.clusters.size()) throw std::out_of_range("cluster index out of range");
            auto& cl = lat.clusters[i];
            lat.coarsen(cl, cell_g);
            lat.rebuild_leaf_indices(cl);
          },
          nb::arg("i"), nb::arg("cell_g"))
      .def(
          "can_split",
          [](AdaptiveLattice& lat, std::size_t i, std::int32_t cell_g) {
            if (i >= lat.clusters.size()) throw std::out_of_range("cluster index out of range");
            return lat.can_split(lat.clusters[i], cell_g);
          },
          nb::arg("i"), nb::arg("cell_g"))
      .def(
          "save",
          [](AdaptiveLattice const& lat, std::string const& path) {
            HighFive::File file(path, HighFive::File::Overwrite);
            file.createAttribute("n_clusters", lat.clusters.size());
            for (std::size_t i = 0; i < lat.clusters.size(); ++i) {
              auto grp = file.createGroup("cluster_" + std::to_string(i));
              save_cluster(lat.clusters[i], grp);
            }
          },
          nb::arg("path"))
      .def_static(
          "load",
          [](std::string const& path) {
            HighFive::File file(path, HighFive::File::ReadOnly);
            std::size_t nc = 0;
            file.getAttribute("n_clusters").read(nc);
            auto lat = AdaptiveLattice{};
            lat.clusters.clear();
            for (std::size_t i = 0; i < nc; ++i) {
              auto grp = file.getGroup("cluster_" + std::to_string(i));
              lat.clusters.push_back(load_cluster(grp));
              lat.rebuild_leaf_indices(lat.clusters.back());
            }
            return lat;
          },
          nb::arg("path"));

  nb::class_<TimeStepper>(m, "TimeStepper")
      .def(nb::init<>())
      .def_rw("dt_base", &TimeStepper::dt_base)
      .def_rw("ratio", &TimeStepper::ratio)
      .def_static(
          "fixed", [](double dt) { return TimeStepper::make_fixed(dt); }, nb::arg("dt"))
      .def_static(
          "by_level",
          [](double dt_base, double ratio) { return TimeStepper::make_by_level(dt_base, ratio); },
          nb::arg("dt_base"), nb::arg("ratio") = 3.0)
      .def_static(
          "cfl", [](double dt_l1) { return TimeStepper::make_cfl(dt_l1); }, nb::arg("dt_l1"))
      .def_static(
          "from_dict", [](Dictator const& d) { return TimeStepper::FROM(d.dict); },
          nb::arg("dictator"));

  nb::class_<AdaptiveLatticeEvolver>(m, "AdaptiveLatticeEvolver")
      .def(nb::init<AdaptiveLattice&>(), nb::arg("lat"))
      .def("step", nb::overload_cast<double, std::size_t>(&AdaptiveLatticeEvolver::step),
           nb::arg("dt"), nb::arg("n_steps") = 1, nb::call_guard<nb::gil_scoped_release>())
      .def("step",
           nb::overload_cast<TimeStepper const&, std::size_t>(&AdaptiveLatticeEvolver::step),
           nb::arg("ts"), nb::arg("n_steps") = 1, nb::call_guard<nb::gil_scoped_release>())
      .def("adapt", &AdaptiveLatticeEvolver::amr_adapt, nb::call_guard<nb::gil_scoped_release>())
      .def("rebuild", &AdaptiveLatticeEvolver::rebuild);

  nb::class_<DiscreteCell>(m, "DiscreteCell")
      .def(nb::init<>())
      .def_rw("u", &DiscreteCell::u)
      .def_rw("lam", &DiscreteCell::lam)
      .def_rw("fixed_u", &DiscreteCell::fixed_u)
      .def_prop_ro_static("L_MIN", [](nb::handle) { return DiscreteCell::L_MIN; })
      .def_prop_ro_static("L_MAX", [](nb::handle) { return DiscreteCell::L_MAX; })
      .def_prop_ro_static("C_VOID", [](nb::handle) { return DiscreteCell::C_VOID; })
      .def_prop_ro_static("C_LOCAL", [](nb::handle) { return DiscreteCell::C_LOCAL; })
      .def_prop_ro_static("L_PLANCK_LOCAL", [](nb::handle) { return DiscreteCell::L_PLANCK_LOCAL; })
      .def_prop_ro_static("L_PLANCK_VOID", [](nb::handle) { return DiscreteCell::L_PLANCK_VOID; })
      .def_prop_ro_static("N_SAGA", [](nb::handle) { return DiscreteCell::N_SAGA; })
      .def_prop_ro_static("L_SAGA", [](nb::handle) { return DiscreteCell::L_SAGA; })
      .def_prop_ro_static("C_SAGA", [](nb::handle) { return DiscreteCell::C_SAGA; })
      .def_prop_ro_static("NFACES", [](nb::handle) { return DiscreteCell::NFACES; });

  // ell0 accepted but ignored for backward compatibility — ell is no longer a stored field.
  m.def(
      "build_bcc_sphere",
      [](AdaptiveCellCluster& cl, double R, double d1, double u0, double /*ell0*/, double lam0) {
        bcc::build_bcc_sphere(cl, R, d1, u0, lam0);
      },
      nb::arg("cluster"), nb::arg("R_cells"), nb::arg("d1") = 1.0, nb::arg("u0") = 0.0,
      nb::arg("ell0") = 1.0, nb::arg("lam0") = 0.0);
  m.def(
      "build_bcc_box",
      [](AdaptiveCellCluster& cl, double Lx, double Ly, double Lz, double d1, double u0,
         double /*ell0*/, double lam0) { bcc::build_bcc_box(cl, Lx, Ly, Lz, d1, u0, lam0); },
      nb::arg("cluster"), nb::arg("Lx"), nb::arg("Ly"), nb::arg("Lz"), nb::arg("d1") = 1.0,
      nb::arg("u0") = 0.0, nb::arg("ell0") = 1.0, nb::arg("lam0") = 0.0);
  m.def(
      "build_bcc_box_periodic",
      [](AdaptiveCellCluster& cl, int Px, int Py, int Pz, double d1, double u0, double lam0) {
        bcc::build_bcc_box_periodic(cl, Px, Py, Pz, d1, u0, lam0);
      },
      nb::arg("cluster"), nb::arg("Px"), nb::arg("Py"), nb::arg("Pz"), nb::arg("d1") = 1.0, nb::arg("u0") = 0.0,
      nb::arg("lam0") = 0.0);
  m.def(
      "solid_angle",
      [](F64_2D_C P, F64_2D_C X, F64_1D apex, std::optional<F64_2D_C> shifts, std::size_t n_threads) {
        if (P.shape(1) != 3 || X.shape(1) != 3 || apex.shape(0) != 3)
          throw std::invalid_argument("solid_angle: P (M, 3), X (N, 3), apex (3,)");
        if (shifts && shifts->shape(1) != 3) throw std::invalid_argument("solid_angle: shifts (S, 3)");
        double const zero[3] = {0, 0, 0};
        double const a[3] = {apex(0), apex(1), apex(2)};
        double const* sh = shifts ? shifts->data() : zero;
        std::size_t const S = shifts ? shifts->shape(0) : 1;
        auto out = make_1d<double>(X.shape(0));
        {
          nb::gil_scoped_release release;
          equilibrium::math::solid_angle(P.data(), P.shape(0), X.data(), X.shape(0), a, sh, S, out.data(), n_threads);
        }
        return out;
      },
      nb::arg("P"), nb::arg("X"), nb::arg("apex"), nb::arg("shifts") = nb::none(), nb::arg("n_threads") = 8,
      "solid angle of the closed polygon P (summed over P + each shift) seen from each X, cone to apex, mod 4 pi");

  nb::class_<Dictator>(m, "Dictator")
      .def(nb::init<>())
      .def("add_bool",
           [](Dictator& d, std::string const& p, bool v) { cppdict::add(p, std::move(v), d.dict); })
      .def("add_int",
           [](Dictator& d, std::string const& p, int v) { cppdict::add(p, std::move(v), d.dict); })
      .def("add_double", [](Dictator& d, std::string const& p,
                            double v) { cppdict::add(p, std::move(v), d.dict); })
      .def("add_string", [](Dictator& d, std::string const& p,
                            std::string v) { cppdict::add(p, std::move(v), d.dict); })
      .def("add_init_fn", [](Dictator& d, std::string const& p, InitFn v) {
        cppdict::add(p, std::move(v), d.dict);
      });

  m.def("start_phlop", []() {
    EQUILIBRIUM_WITH_PHLOP({
      mkn::kul::Dir{".phlop"}.mk();
      phlop::scope_timer().file_name(".phlop/timings.txt").init();
    })
  });
  m.def("stop_phlop", []() { EQUILIBRIUM_WITH_PHLOP({ phlop::shutdown_scope_timer(); }) });
}

static void bind_continuous(nb::module_& m, nb::module_& d) {
  namespace ct = equilibrium::dst::continuous;
  using F64_C = nb::ndarray<nb::numpy, double, nb::c_contig>;
  using I32_C = nb::ndarray<nb::numpy, std::int32_t, nb::c_contig>;
  nb::register_exception_translator([](std::exception_ptr const& p, void*) {
    try {
      std::rethrow_exception(p);
    } catch (ct::not_implemented const& e) {
      PyErr_SetString(PyExc_NotImplementedError, e.what());
    }
  });

  for (char const* name : {"BoundaryCondition", "TimeStepper", "AdaptiveCellCluster",
                           "build_bcc_sphere", "build_bcc_box", "build_bcc_box_periodic", "solid_angle", "SplitCriterion", "DSTInfo"})
    m.attr(name) = d.attr(name);

  auto cluster_at = [](ct::AdaptiveLattice& lat, std::size_t i) -> AdaptiveCellCluster& {
    if (i >= lat.clusters.size()) throw std::out_of_range("cluster index out of range");
    return lat.clusters[i];
  };

  nb::class_<ct::AdaptiveLattice>(m, "AdaptiveLattice")
      .def(nb::init<>())
      .def_prop_ro("n_clusters", [](ct::AdaptiveLattice& lat) { return lat.clusters.size(); })
      .def("add_cluster",
           [](ct::AdaptiveLattice& lat) {
             lat.clusters.emplace_back();
             return lat.clusters.size() - 1;
           })
      .def("__getitem__", cluster_at, nb::rv_policy::reference_internal)
      .def("set_n_threads", &ct::AdaptiveLattice::set_n_threads, nb::arg("n"))
      .def(
          "step",
          [=](ct::AdaptiveLattice& lat, std::size_t i, double dt) { lat.step(cluster_at(lat, i), dt); },
          nb::arg("i"), nb::arg("dt"))
      .def(
          "step_all", [](ct::AdaptiveLattice& lat, double dt) { lat.step_all(dt); }, nb::arg("dt"))
      .def(
          "step_damped",
          [=](ct::AdaptiveLattice& lat, std::size_t i, double dt) {
            lat.step_damped(cluster_at(lat, i), dt);
          },
          nb::arg("i"), nb::arg("dt"))
      .def(
          "step_dir_damped",
          [=](ct::AdaptiveLattice& lat, std::size_t i, double dt) {
            lat.step_dir_damped(cluster_at(lat, i), dt);
          },
          nb::arg("i"), nb::arg("dt"))
      .def(
          "step_pml",
          [=](ct::AdaptiveLattice& lat, std::size_t i, double dt) {
            lat.step_pml(cluster_at(lat, i), dt);
          },
          nb::arg("i"), nb::arg("dt"))
      .def(
          "step_lam",
          [=](ct::AdaptiveLattice& lat, std::size_t i, double dt, double scale) {
            lat.step_lam(cluster_at(lat, i), dt, scale);
          },
          nb::arg("i"), nb::arg("dt"), nb::arg("scale") = 1.0)
      .def(
          "step_sponge",
          [=](ct::AdaptiveLattice& lat, std::size_t i, double dt) { lat.step_sponge(cluster_at(lat, i), dt); },
          nb::arg("i"), nb::arg("dt") = 0.0)
      .def("step_sponge_all", &ct::AdaptiveLattice::step_sponge_all, nb::arg("dt") = 0.0)
      .def(
          "rebuild",
          [=](ct::AdaptiveLattice& lat, std::size_t i) {
            lat.rebuild_leaf_indices(cluster_at(lat, i));
          },
          nb::arg("i"))
      .def("rebuild_all", &ct::AdaptiveLattice::rebuild_all)
      .def(
          "adapt", [=](ct::AdaptiveLattice& lat, std::size_t i) { lat.adapt(cluster_at(lat, i)); },
          nb::arg("i"))
      .def("adapt_all", &ct::AdaptiveLattice::adapt_all)
      .def(
          "split",
          [=](ct::AdaptiveLattice& lat, std::size_t i, std::int32_t cell_g) {
            return lat.split(cluster_at(lat, i), cell_g);
          },
          nb::arg("i"), nb::arg("cell_g"))
      .def(
          "coarsen",
          [=](ct::AdaptiveLattice& lat, std::size_t i, std::int32_t cell_g) {
            lat.coarsen(cluster_at(lat, i), cell_g);
          },
          nb::arg("i"), nb::arg("cell_g"))
      .def(
          "can_split",
          [=](ct::AdaptiveLattice& lat, std::size_t i, std::int32_t cell_g) {
            return lat.can_split(cluster_at(lat, i), cell_g);
          },
          nb::arg("i"), nb::arg("cell_g"))
      .def(
          "step_ae",
          [=](ct::AdaptiveLattice& lat, std::size_t i, double dt, std::optional<F64_2D_C> J, double scale) {
            auto& cl = cluster_at(lat, i);
            double const* Jp = nullptr;
            if (J) {
              if (J->shape(0) != CellSlab::NF || J->shape(1) != cl.leaf_indices.size())
                throw std::runtime_error("step_ae: J expected shape (14, N)");
              Jp = J->data();
            }
            lat.step_ae(cl, dt, Jp, scale);
          },
          nb::arg("i"), nb::arg("dt"), nb::arg("J") = nb::none(), nb::arg("scale") = 1.0)
      .def(
          "curl_curl_sym",
          [=](ct::AdaptiveLattice& lat, std::size_t i, F64_2D_C phi) {
            auto& cl = cluster_at(lat, i);
            auto const N = cl.leaf_indices.size();
            if (phi.shape(0) != CellSlab::NF || phi.shape(1) != N)
              throw std::runtime_error("curl_curl_sym: phi expected shape (14, N)");
            auto out = make_2d<double>(CellSlab::NF, N);
            lat.curl_curl_sym(cl, phi.data(), out.data());
            return out;
          },
          nb::arg("i"), nb::arg("phi"))
      .def(
          "curl_curl_dec",
          [=](ct::AdaptiveLattice& lat, std::size_t i, F64_2D_C phi, std::string const& star) {
            auto& cl = cluster_at(lat, i);
            auto const N = cl.leaf_indices.size();
            if (phi.shape(0) != CellSlab::NF || phi.shape(1) != N)
              throw std::runtime_error("curl_curl_dec: phi expected shape (14, N)");
            auto out = make_2d<double>(CellSlab::NF, N);
            if (star != "dec" && star != "dec_geo") throw std::runtime_error("curl_curl_dec: star 'dec' or 'dec_geo'");
            lat.curl_curl_dec(cl, phi.data(), out.data(),
                              star == "dec" ? ct::AdaptiveLattice::Curl::dec : ct::AdaptiveLattice::Curl::dec_geo);
            return out;
          },
          nb::arg("i"), nb::arg("phi"), nb::arg("star") = "dec")
      .def(
          "step_cartan",
          [=](ct::AdaptiveLattice& lat, std::size_t i, double dt, double kappa, std::size_t n_iter) {
            auto& cl = cluster_at(lat, i);
            nb::gil_scoped_release release;
            lat.step_cartan(cl, dt, kappa, n_iter);
          },
          nb::arg("i"), nb::arg("dt"), nb::arg("kappa") = 1.0, nb::arg("n_iter") = 3)
      .def("set_cartan_sigma", &ct::AdaptiveLattice::set_cartan_sigma, nb::arg("on"))
      .def("cartan_sigma", &ct::AdaptiveLattice::cartan_sigma)
      .def("set_cell_capacity", &ct::AdaptiveLattice::set_cell_capacity, nb::arg("S_c"), nb::arg("p") = 1)
      .def("cell_capacity", &ct::AdaptiveLattice::cell_capacity)
      .def("set_cartan_vcap", &ct::AdaptiveLattice::set_cartan_vcap, nb::arg("v"))
      .def("cartan_vcap", &ct::AdaptiveLattice::cartan_vcap)
      .def(
          "cartan_transport",
          [=](ct::AdaptiveLattice& lat, std::size_t i, F64_2D_C phi) {
            auto& cl = cluster_at(lat, i);
            auto const N = cl.leaf_indices.size();
            if (phi.shape(0) != CellSlab::NF || phi.shape(1) != N)
              throw std::runtime_error("cartan_transport: phi expected shape (14, N)");
            auto out = make_2d<double>(CellSlab::NF, N);
            lat.cartan_transport(cl, phi.data(), out.data());
            return out;
          },
          nb::arg("i"), nb::arg("phi"))
      .def(
          "set_curl",
          [](ct::AdaptiveLattice& lat, std::string const& c) {
            if (c == "ls")
              lat.set_curl(ct::AdaptiveLattice::Curl::ls);
            else if (c == "dec")
              lat.set_curl(ct::AdaptiveLattice::Curl::dec);
            else if (c == "dec_geo")
              lat.set_curl(ct::AdaptiveLattice::Curl::dec_geo);
            else
              throw std::runtime_error("set_curl: expected 'ls', 'dec' or 'dec_geo'");
          },
          nb::arg("curl"))
      .def_prop_ro("curl", [](ct::AdaptiveLattice const& lat) {
        auto const c = lat.curl();
        return std::string(c == ct::AdaptiveLattice::Curl::dec       ? "dec"
                           : c == ct::AdaptiveLattice::Curl::dec_geo ? "dec_geo"
                                                                     : "ls");
      })
      .def(
          "leaf_e",
          [=](ct::AdaptiveLattice& lat, std::size_t i) {
            auto& an = lat.anchors(cluster_at(lat, i));
            auto const N = an.e.size() / CellSlab::NF;
            auto out = make_2d<double>(CellSlab::NF, N);
            std::copy(an.e.begin(), an.e.end(), out.data());
            return out;
          },
          nb::arg("i"))
      .def(
          "set_leaf_e",
          [=](ct::AdaptiveLattice& lat, std::size_t i, F64_2D_C e) {
            auto& an = lat.anchors(cluster_at(lat, i));
            auto const N = an.e.size() / CellSlab::NF;
            if (e.shape(0) != CellSlab::NF || e.shape(1) != N)
              throw std::runtime_error("set_leaf_e: expected shape (14, N)");
            std::copy(e.data(), e.data() + an.e.size(), an.e.begin());
          },
          nb::arg("i"), nb::arg("e"))
      .def(
          "set_face_medium",
          [=](ct::AdaptiveLattice& lat, std::size_t i, F64_2D_C dsig, F64_2D_C dn) {
            auto& cl = cluster_at(lat, i);
            auto const N = cl.leaf_indices.size();
            for (auto const* a : {&dsig, &dn})
              if (a->shape(0) != CellSlab::NF || a->shape(1) != N)
                throw std::runtime_error("set_face_medium: expected shape (14, N)");
            lat.set_face_medium(cl, dsig.data(), dn.data());
          },
          nb::arg("i"), nb::arg("dsig"), nb::arg("dn"))
      .def(
          "set_cartan_hold",
          [=](ct::AdaptiveLattice& lat, std::size_t i, F64_1D kappa) {
            auto& cl = cluster_at(lat, i);
            if (kappa.shape(0) != cl.leaf_indices.size()) throw std::runtime_error("set_cartan_hold: expected shape (N,)");
            std::vector<double> k(kappa.shape(0));
            for (std::size_t j = 0; j < k.size(); ++j) k[j] = kappa(j);
            lat.set_cartan_hold(cl, k.data());
          },
          nb::arg("i"), nb::arg("kappa"))
      .def(
          "clear_cartan_hold",
          [=](ct::AdaptiveLattice& lat, std::size_t i) { lat.clear_cartan_hold(cluster_at(lat, i)); }, nb::arg("i"))
      .def(
          "clear_face_medium",
          [=](ct::AdaptiveLattice& lat, std::size_t i) { lat.clear_face_medium(cluster_at(lat, i)); },
          nb::arg("i"))
      .def(
          "set_walk",
          [=](ct::AdaptiveLattice& lat, std::size_t i, F64_C AL, F64_C AR, I32_C kL, I32_C kR, double n, double m,
              double g, F64_C bgL, F64_C bgR, double mu) {
            if (AL.size() != 8 * 16 || AR.size() != 8 * 16) throw std::runtime_error("set_walk: AL, AR expected (8, 4, 4)");
            if (kL.size() != 8 || kR.size() != 8) throw std::runtime_error("set_walk: kL, kR expected (8,)");
            if (bgL.size() != 4 || bgR.size() != 4) throw std::runtime_error("set_walk: bgL, bgR expected (4,)");
            lat.set_walk(cluster_at(lat, i), AL.data(), AR.data(), kL.data(), kR.data(), n, m, g, bgL.data(),
                         bgR.data(), mu);
          },
          nb::arg("i"), nb::arg("AL"), nb::arg("AR"), nb::arg("kL"), nb::arg("kR"), nb::arg("n"), nb::arg("m"),
          nb::arg("g"), nb::arg("bgL"), nb::arg("bgR"), nb::arg("mu"))
      .def(
          "set_walk_parity",
          [=](ct::AdaptiveLattice& lat, std::size_t i, bool parity) { lat.set_walk_parity(cluster_at(lat, i), parity); },
          nb::arg("i"), nb::arg("parity"))
      .def(
          "set_walk_clock",
          [=](ct::AdaptiveLattice& lat, std::size_t i, double beta, double om0, double rho0) {
            lat.set_walk_clock(cluster_at(lat, i), beta, om0, rho0);
          },
          nb::arg("i"), nb::arg("beta"), nb::arg("om0"), nb::arg("rho0"))
      .def(
          "set_walk_state",
          [=](ct::AdaptiveLattice& lat, std::size_t i, F64_C L, F64_C R, std::int64_t t) {
            auto& cl = cluster_at(lat, i);
            auto& w = lat.walk(cl);
            if (L.size() != w.L.size() || R.size() != w.R.size())
              throw std::runtime_error("set_walk_state: L, R expected (N, 4)");
            std::copy(L.data(), L.data() + w.L.size(), w.L.begin());
            std::copy(R.data(), R.data() + w.R.size(), w.R.begin());
            if (t >= 0) w.t = t;
          },
          nb::arg("i"), nb::arg("L"), nb::arg("R"), nb::arg("t") = -1)
      .def(
          "set_walk_sponge",
          [=](ct::AdaptiveLattice& lat, std::size_t i, F64_C f) {
            auto& cl = cluster_at(lat, i);
            if (f.size() != cl.leaf_indices.size()) throw std::runtime_error("set_walk_sponge: f expected (N,)");
            lat.set_walk_sponge(cl, f.data());
          },
          nb::arg("i"), nb::arg("f"))
      .def(
          "step_walk",
          [=](ct::AdaptiveLattice& lat, std::size_t i, std::size_t n_steps) {
            auto& cl = cluster_at(lat, i);
            nb::gil_scoped_release rel;
            lat.step_walk(cl, n_steps);
          },
          nb::arg("i"), nb::arg("n_steps") = 1)
      .def(
          "walk_L",
          [=](ct::AdaptiveLattice& lat, std::size_t i) {
            auto& w = lat.walk(cluster_at(lat, i));
            auto out = make_2d<double>(w.L.size() / 4, 4);
            std::copy(w.L.begin(), w.L.end(), out.data());
            return out;
          },
          nb::arg("i"))
      .def(
          "walk_R",
          [=](ct::AdaptiveLattice& lat, std::size_t i) {
            auto& w = lat.walk(cluster_at(lat, i));
            auto out = make_2d<double>(w.R.size() / 4, 4);
            std::copy(w.R.begin(), w.R.end(), out.data());
            return out;
          },
          nb::arg("i"))
      .def(
          "walk_density",
          [=](ct::AdaptiveLattice& lat, std::size_t i) {
            auto& w = lat.walk(cluster_at(lat, i));
            auto const N = w.L.size() / 4;
            auto out = make_1d<double>(N);
            for (std::size_t j = 0; j < N; ++j) {
              double s = 0.0;
              for (int r = 0; r < 4; ++r) s += w.L[4 * j + r] * w.L[4 * j + r] + w.R[4 * j + r] * w.R[4 * j + r];
              out.data()[j] = s;
            }
            return out;
          },
          nb::arg("i"))
      .def(
          "walk_t", [=](ct::AdaptiveLattice& lat, std::size_t i) { return lat.walk(cluster_at(lat, i)).t; },
          nb::arg("i"))
      .def(
          "leaf_nb_j",
          [=](ct::AdaptiveLattice& lat, std::size_t i) {
            auto& an = lat.anchors(cluster_at(lat, i));
            auto const N = an.nbj.size() / CellSlab::NF;
            auto out = make_2d<std::int32_t>(N, CellSlab::NF);
            std::copy(an.nbj.begin(), an.nbj.end(), out.data());
            return out;
          },
          nb::arg("i"))
      .def(
          "set_weights",
          [](ct::AdaptiveLattice& lat, std::string const& w) {
            if (w == "iso")
              lat.set_weights(ct::anchor::Weights::iso);
            else if (w == "discrete")
              lat.set_weights(ct::anchor::Weights::discrete);
            else if (w == "legacy")
              lat.set_weights(ct::anchor::Weights::legacy);
            else
              throw std::runtime_error("set_weights: expected 'iso', 'discrete' or 'legacy'");
          },
          nb::arg("weights"))
      .def_prop_ro("weights",
                   [](ct::AdaptiveLattice const& lat) {
                     auto const w = lat.weights();
                     return std::string(w == ct::anchor::Weights::iso        ? "iso"
                                        : w == ct::anchor::Weights::discrete ? "discrete"
                                                                             : "legacy");
                   })
      .def_prop_ro("face_weights",
                   [](ct::AdaptiveLattice const& lat) {
                     auto const w = ct::anchor::face_weights(lat.weights());
                     return std::vector<double>(w.begin(), w.end());
                   })
      .def_prop_ro("C0SQ", &ct::AdaptiveLattice::C0SQ);

  nb::class_<ct::AdaptiveLatticeEvolver>(m, "AdaptiveLatticeEvolver")
      .def(nb::init<ct::AdaptiveLattice&>(), nb::arg("lat"), nb::keep_alive<1, 2>())
      .def("step", nb::overload_cast<double, std::size_t>(&ct::AdaptiveLatticeEvolver::step),
           nb::arg("dt"), nb::arg("n_steps") = 1)
      .def("step",
           nb::overload_cast<TimeStepper const&, std::size_t>(&ct::AdaptiveLatticeEvolver::step),
           nb::arg("ts"), nb::arg("n_steps") = 1)
      .def("adapt", &ct::AdaptiveLatticeEvolver::amr_adapt)
      .def("rebuild", &ct::AdaptiveLatticeEvolver::rebuild);
}

NB_MODULE(lattice, m) {
  auto d = m.def_submodule("discrete", "Current AMR model: discrete face-flux rule (reference)");
  bind_discrete(d);
  for (auto item : nb::borrow<nb::dict>(d.attr("__dict__")))
    if (!nb::cast<std::string>(item.first).starts_with("__")) m.attr(item.first) = item.second;
  auto c = m.def_submodule("continuous", "Continuous model: faces as anchors of a continuum field");
  bind_continuous(c, d);
}
