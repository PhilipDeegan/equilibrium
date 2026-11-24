#ifndef _EQUILIBRIUM_CORE_DATA_LATTICE_CONTINUOUS_ADAPTIVE_LATTICE_HPP_
#define _EQUILIBRIUM_CORE_DATA_LATTICE_CONTINUOUS_ADAPTIVE_LATTICE_HPP_

// Continuous model: cells and faces are anchors sampling a continuum field,
// not the sites of flux transfer. Refinement is better sampling of the same
// field and results must converge under it.
//
// Reference: equilibrium::dst::discrete. Each member names its discrete
// analogue; any deviation from it carries its justification here.
//
// Shared with discrete, unchanged (storage / geometry / config, no dynamics):
//   AdaptiveCellCluster  per-cell and per-face-anchor sample storage
//   build_bcc_sphere/box level-0 anchor positions and neighbour table
//   BoundaryCondition    BC description; how it acts is defined per model
//   TimeStepper          dt rule
//
// Discrete functions with no continuous stub (undecided, discrete-only for now):
//   step_cross, step_cross_all   gated per-face drain+mirror kernel
//   step_boundary_hyper_resistivity*  fix-up keyed on a binary coarse/fine
//                                     boundary-leaf classification
//   defragment, save/load        storage housekeeping; add when needed
//   static diagnostic toggles    they switch discrete junction corrections

#include "eq/core/data/lattice/adaptive_lattice.hpp"

#include "eq/core/data/lattice/cluster.hpp"
#include "eq/core/data/lattice/time_stepper.hpp"
#include "BS_thread_pool.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <initializer_list>

namespace equilibrium::dst::continuous {

using discrete::AdaptiveCellCluster;
using discrete::CellSlab;
using discrete::TimeStepper;

struct not_implemented : std::logic_error {
  explicit not_implemented(std::string const& what, std::string const& ref)
      : std::logic_error("continuous::" + what + " not implemented (discrete reference: discrete::" +
                         ref + ")") {}
};

// Anchor operators of equilibrium/substrate/continuous.py, level 0 (uniform
// L = h, the hex-neighbour distance). Hop vectors in units of h: hex faces
// |d| = 1, square faces |d| = 2/sqrt3.
namespace anchor {

constexpr std::size_t NF = CellSlab::NF;
using discrete::bcc::mirror_face;
using FaceW = std::array<double, NF>;

// Face weights W_k of the anchor sampling phi_k = W_k A.d_k (per leg: step's
// u sector is DEC with Hodge star *1 ~ W^2). The continuous model approximates
// the discrete lattice's long-wave limit, so its coarse-cell error is pure
// discretisation error; its Kelvin cells are there to avoid a preferred axis.
//   iso       sqrt2 : 1 -> star 2:1, the only hex:square ratio whose 2nd-order
//             error is direction independent in both the u and the transverse
//             sector (exp/2026/09/28/hodge_star, tst/substrate/test_hodge_star.py)
//   discrete  discrete::bcc::face_weight, sqrt3 : 1 -> star 3:1 (Kelvin
//             geometry, since 2026-09-28): step() == discrete step (plumbing)
//   legacy    3 : 1 -> star 9:1, both models before 2026-09-28 (old runs)
enum class Weights { iso, discrete, legacy };
inline FaceW face_weights(Weights w) {
  FaceW r{};
  for (std::size_t k = 0; k < NF; ++k)
    r[k] = w == Weights::discrete ? discrete::bcc::face_weight[k]
           : w == Weights::legacy ? (k < 8 ? 3.0 : 1.0)
                                  : (k < 8 ? std::sqrt(2.0) : 1.0);
  return r;
}

inline std::array<std::array<double, 3>, NF> const& hops() {
  static auto const d = [] {
    std::array<std::array<double, 3>, NF> r{};
    for (std::size_t k = 0; k < NF; ++k) {
      auto const s = k < 8 ? 1.0 : 2.0 / std::sqrt(3.0);
      for (std::size_t e = 0; e < 3; ++e) r[k][e] = s * discrete::bcc::face_dirs[k][e];
    }
    return r;
  }();
  return d;
}

// C0SQ = 1/2 sum_k W_k^2 d_kx^2 (= c^2): 4 (iso), 16/3 (discrete), 40/3 (legacy).
inline double c0sq(FaceW const& w) {
  double s = 0.0;
  for (std::size_t k = 0; k < NF; ++k) s += w[k] * w[k] * hops()[k][0] * hops()[k][0];
  return 0.5 * s;
}

// Delaunay triangles of the BCC lattice as templates based at a cell:
// i -k-> n -m-> l with d_s = d_k + d_m, s the +x/+y/+z square hop, so each
// triangle is counted once (12 per cell, all hex-hex-square).
struct Tri {
  std::uint8_t s, k, m;
};
inline std::array<Tri, 12> const& tris() {
  static auto const t = [] {
    std::array<Tri, 12> r{};
    std::size_t n = 0;
    auto const& d = hops();
    for (std::uint8_t s : {9, 11, 13})
      for (std::uint8_t k = 0; k < 8; ++k)
        for (std::uint8_t m = 0; m < 8; ++m) {
          bool eq = true;
          for (std::size_t e = 0; e < 3; ++e) eq = eq && std::abs(d[k][e] + d[m][e] - d[s][e]) < 1e-12;
          if (eq) r.at(n++) = {s, k, m};
        }
    if (n != 12) throw std::logic_error("continuous::anchor::tris: expected 12 triangle templates");
    return r;
  }();
  return t;
}

// Moore-Penrose inverse of a symmetric PSD 3x3 (cyclic Jacobi), eigenvalues
// below 1e-30 * max treated as zero (numpy pinv's rcond=1e-15 on singular values).
inline std::array<std::array<double, 3>, 3> pinv_sym3(std::array<std::array<double, 3>, 3> a) {
  std::array<std::array<double, 3>, 3> v{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
  for (int sweep = 0; sweep < 50; ++sweep) {
    double off = std::abs(a[0][1]) + std::abs(a[0][2]) + std::abs(a[1][2]);
    if (off < 1e-300) break;
    for (int p = 0; p < 2; ++p)
      for (int q = p + 1; q < 3; ++q) {
        if (a[p][q] == 0.0) continue;
        double const th = 0.5 * (a[q][q] - a[p][p]) / a[p][q];
        double const t  = std::copysign(1.0, th) / (std::abs(th) + std::sqrt(th * th + 1.0));
        double const c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
        for (int r = 0; r < 3; ++r) {
          double const arp = a[r][p], arq = a[r][q];
          a[r][p] = c * arp - s * arq;
          a[r][q] = s * arp + c * arq;
        }
        for (int r = 0; r < 3; ++r) {
          double const apr = a[p][r], aqr = a[q][r];
          a[p][r] = c * apr - s * aqr;
          a[q][r] = s * apr + c * aqr;
        }
        for (int r = 0; r < 3; ++r) {
          double const vrp = v[r][p], vrq = v[r][q];
          v[r][p] = c * vrp - s * vrq;
          v[r][q] = s * vrp + c * vrq;
        }
      }
  }
  double lmax = std::max({a[0][0], a[1][1], a[2][2]});
  std::array<std::array<double, 3>, 3> out{};
  for (int m = 0; m < 3; ++m) {
    if (!(a[m][m] > 1e-30 * lmax)) continue;
    double const inv = 1.0 / a[m][m];
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) out[r][c] += v[r][m] * inv * v[c][m];
  }
  return out;
}

// Per-leaf state of the continuous model, indexed by leaf position j. The
// reconstruction operators depend only on which faces are present, so they are
// stored once per face-presence pattern (every interior cell shares one).
struct ClusterAnchors {
  double h = 0.0;
  FaceW w{};                      // face weights (set by build)
  double c0sq = 0.0;              // c^2 = C0SQ(w)
  std::vector<std::int32_t> nbj;  // L*NF, leaf index of face-k neighbour, -1 = none
  std::vector<std::uint16_t> op;  // L, operator index of leaf j
  std::vector<double> P;          // n_op*3*NF, pinv(D) over present faces (A from phi/W)
  std::vector<double> Pw;         // n_op*3*NF, pinv(diag(W) D): Euclidean adjoint of sampling
  std::vector<double> e;          // NF*L face-major, EXPERIMENTAL E at the anchors
  std::vector<double> tri;        // L*12, *2 x circulation per triangle template (DEC curl)
  std::vector<double> cart;       // L*12*3, per-triangle edge contributions (cartan_transport)
  std::vector<double> cref, cph0, cmid, cd;  // NF*L scratch (step_cartan)
  std::vector<double> gcap;                   // L, G'(S_j) of the cell capacity (set_cell_capacity)
  std::vector<double> sig;                    // L scratch, sigma of each leaf (step, no medium)
  // W_k sigma_f / (h n_f) at the face anchors, NF*L face-major; empty = no
  // medium (sigma_f from lam, n_f = 1).
  std::vector<double> coef;
  std::vector<double> fsig;  // sigma_f at the face anchors, NF*L face-major (set with coef)
  std::vector<double> hold;  // L, per-leaf hold kappa of the Cartan vorticity (set_cartan_hold); empty = off
  std::vector<double> omb;   // L*12 scratch: raw triangle loop sums (NaN = no triangle), with hold
  std::vector<double> A, B;  // L*3 scratch

  // Two-component BCC walk (equilibrium/substrate/walk.py): 8 reals per leaf, L and R halves of two pairs each,
  // leaf-major (L[4 j + r]); L1 / R1 the second buffer. AL / AR: the 8 hops' real 4x4 (row-major), kL / kR the face
  // of each hop (L gathers from x + h, R from x - h). Cleared by build.
  struct Walk {
    bool set = false;
    std::array<double, 8 * 16> AL{}, AR{};
    std::array<std::int32_t, 8> kL{}, kR{};
    double n = 1.0, m = 0.0, g = 0.0, mu = 0.0;
    std::array<double, 4> bgL{}, bgR{};  // background at t = 0; at t: every pair turned by -mu t
    std::int64_t t = 0;
    bool parity = false;  // odd t: L and R exchange AL / AR (parity image tick)
    double beta = 0.0, om0 = 0.0, rho0 = 1.0;  // local clock sigma = 1 + beta (rho0 - rho1) / rho0 (0 = off)
    std::vector<double> L, R, L1, R1, f;  // f: per-leaf sponge factor (empty = none)
  } walk;

  bool medium() const { return !coef.empty(); }
  double const* p(std::size_t j) const { return &P[op[j] * 3 * NF]; }
  double const* pw(std::size_t j) const { return &Pw[op[j] * 3 * NF]; }

  void build(AdaptiveCellCluster& cl, FaceW const& weights) {
    auto const L = cl.leaf_indices.size();
    h = L ? cl[cl.leaf_indices[0]].L : 0.0;
    w    = weights;
    c0sq = anchor::c0sq(w);
    nbj.assign(L * NF, -1);
    op.assign(L, 0);
    P.clear();
    Pw.clear();
    e.assign(NF * L, 0.0);
    coef.clear();
    fsig.clear();
    hold.clear();
    tri.clear();
    A.assign(L * 3, 0.0);
    B.assign(L * 3, 0.0);
    walk = Walk{};
    std::unordered_map<std::uint16_t, std::uint16_t> ops;
    auto const& d = hops();
    for (std::size_t j = 0; j < L; ++j) {
      auto const cell = cl[cl.leaf_indices[j]];
      std::uint16_t mask = 0;
      for (std::size_t k = 0; k < NF; ++k) {
        auto const g = cell.nb[k];
        if (g < 0) continue;
        nbj[j * NF + k] = cl.g_to_j[g];
        mask |= std::uint16_t(1u << k);
      }
      auto const [it, fresh] = ops.try_emplace(mask, static_cast<std::uint16_t>(ops.size()));
      op[j] = it->second;
      if (!fresh) continue;
      std::array<std::array<double, 3>, 3> M{}, Mw{};
      for (std::size_t k = 0; k < NF; ++k) {
        if (!(mask & (1u << k))) continue;
        auto const w2 = w[k] * w[k];
        for (int r = 0; r < 3; ++r)
          for (int c = 0; c < 3; ++c) {
            M[r][c] += d[k][r] * d[k][c];
            Mw[r][c] += w2 * d[k][r] * d[k][c];
          }
      }
      auto const Mi = pinv_sym3(M), Mwi = pinv_sym3(Mw);
      auto const o = P.size();
      P.resize(o + 3 * NF, 0.0);
      Pw.resize(o + 3 * NF, 0.0);
      for (std::size_t k = 0; k < NF; ++k) {
        if (!(mask & (1u << k))) continue;
        for (int r = 0; r < 3; ++r) {
          double pp = 0.0, pwr = 0.0;
          for (int c = 0; c < 3; ++c) {
            pp += Mi[r][c] * d[k][c];
            pwr += Mwi[r][c] * d[k][c];
          }
          P[o + r * NF + k]  = pp;
          Pw[o + r * NF + k] = pwr * w[k];
        }
      }
    }
  }
};
}  // namespace anchor

struct AdaptiveLattice {
  std::vector<AdaptiveCellCluster> clusters;

  AdaptiveLattice() { clusters.emplace_back(); }
  AdaptiveLattice(AdaptiveLattice const&) = delete;
  AdaptiveLattice& operator=(AdaptiveLattice const&) = delete;
  AdaptiveLattice(AdaptiveLattice&&) = default;
  AdaptiveLattice& operator=(AdaptiveLattice&&) = default;

  // discrete: set_n_threads. Same pool.
  void set_n_threads(std::size_t n) {
    n_threads_ = n;
    if (n <= 1)
      pool_.reset();
    else
      pool_ = std::make_unique<BS::thread_pool<>>(n);
  }
  std::size_t n_threads() const { return n_threads_; }

  anchor::ClusterAnchors& anchors(AdaptiveCellCluster& cl) {
    auto const i = static_cast<std::size_t>(&cl - clusters.data());
    if (i >= anchors_.size() || anchors_[i].nbj.size() != cl.leaf_indices.size() * anchor::NF)
      throw std::logic_error("continuous: rebuild the cluster before stepping");
    auto& an = anchors_[i];
    for (auto const g : cl.leaf_indices)
      if (cl[g].L != an.h) throw std::logic_error("continuous: coarse cells need uniform L (set a medium with set_face_medium)");
    return an;
  }

  // discrete: rebuild_leaf_indices. Same leaf list, g_to_j, temp buffers and
  // boundary-leaf partition. Deviation: no rebuild_amr_corrections /
  // face_multiplicity / ancestor_group_multiplicity / sibling_groups /
  // resolve_stale_external_proxies -- those are discrete junction flux
  // bookkeeping; refinement here must come from continuum prolong/restrict.
  // Until those exist a refined cluster is rejected.
  void rebuild_leaf_indices(AdaptiveCellCluster& cluster) {
    cluster.leaf_indices.clear();
    cluster.n_non_leaf_cells = 0;
    for (std::size_t s = 0; s < cluster.n_slabs(); ++s) {
      auto& slab = cluster.slab(s);
      for (std::size_t i = 0; i < CellSlab::CAPACITY; ++i) {
        if (slab.parent[i] == -2) continue;
        if (slab.n_children[i] == 0)
          cluster.leaf_indices.push_back(AdaptiveCellCluster::global(s, i));
        else
          ++cluster.n_non_leaf_cells;
      }
    }
    if (cluster.n_non_leaf_cells)
      throw not_implemented("rebuild_leaf_indices on a refined cluster",
                            "rebuild_leaf_indices");
    auto const L = cluster.leaf_indices.size();
    cluster.g_max = L ? *std::max_element(cluster.leaf_indices.begin(), cluster.leaf_indices.end())
                      : -1;
    cluster.g_to_j.assign(static_cast<std::size_t>(cluster.g_max + 1), -1);
    for (std::size_t j = 0; j < L; ++j)
      cluster.g_to_j[cluster.leaf_indices[j]] = static_cast<std::int32_t>(j);
    // discrete step scratch (the shared builders allocate it); unused here
    std::vector<double>().swap(cluster.temp_du);
    std::vector<double>().swap(cluster.temp_phi);
    std::vector<double>().swap(cluster.temp_phi_mirror);
    constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
    cluster.criterion_min.resize(L, NaN);
    cluster.criterion_max.resize(L, NaN);
    cluster.boundary_leaf_positions.clear();
    for (std::size_t j = 0; j < L; ++j) {
      auto cell = cluster[cluster.leaf_indices[j]];
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        if (cell.nb[k] < 0) {
          cluster.boundary_leaf_positions.push_back(j);
          break;
        }
      }
    }
    auto const i = static_cast<std::size_t>(&cluster - clusters.data());
    if (anchors_.size() <= i) anchors_.resize(clusters.size());
    anchors_[i].build(cluster, anchor::face_weights(weights_));
  }

  // Face weights (anchor::Weights). Rebuilds the anchors of every cluster
  // already built, which clears their E and face medium.
  void set_weights(anchor::Weights w) {
    weights_ = w;
    for (std::size_t i = 0; i < anchors_.size() && i < clusters.size(); ++i)
      if (!anchors_[i].nbj.empty()) anchors_[i].build(clusters[i], anchor::face_weights(w));
  }
  anchor::Weights weights() const { return weights_; }
  double C0SQ() const { return anchor::c0sq(anchor::face_weights(weights_)); }

  void rebuild_all() {
    for (auto& c : clusters) rebuild_leaf_indices(c);
  }

  // discrete: step. With Weights::discrete, level 0 and no medium this is
  // discrete::step's uniform path (pass 1, _step_pass2_uniform, pass 3); that
  // identity is a plumbing check, not a requirement (the continuous model is
  // continuous at every h; default Weights::iso). Justification: under
  // the anchor mapping phi_k = W_k A(mid_k).d_k it converges at 2nd order to
  // u_t = -sigma C0SQ div A, A_t = -sigma grad u (continuous.py, measured
  // 2026-09-24), i.e. it is already a consistent discretisation of that
  // continuum system -- the E = 0 sector. Face coefficient W sigma_f / (h n_f),
  // the same in both passes (phi is one antisymmetric DOF per face).
  // sigma_f, n_f are the medium at the face anchor (set_face_medium); the
  // coarse cells' own size h stays uniform -- the medium is not a property of
  // the sampling. Unset: sigma_f = midpoint interpolation of the anchor
  // samples 1/(1+lam), n_f = 1 (= discrete). Other deviations: no AMR
  // corrections (level 0 only) and no face BCs (ghost/fn/remote/outflow are
  // discrete ghost-face operations; the continuum boundary is undecided) --
  // throws. The volumetric sponge (step_sponge: vol_rate / vol_damp) is allowed.
  void step(AdaptiveCellCluster& cl, TimeStepper const& ts) {
    EQ_SCOPE("equilibrium::dst::continuous::AdaptiveLattice::step");
    step_u(cl, ts);
    step_phi(cl, ts);
  }

  // step's two halves: u from phi, then phi from the new u. Clusters whose faces reach each other's
  // sites exchange u between them (continuous/damn_clusters.hpp)
  void step_u(AdaptiveCellCluster& cl, TimeStepper const& ts) {
    _require_no_face_bc(cl, "step");
    auto& an = anchors(cl);
    if (an.medium())
      for (auto const g : cl.leaf_indices)
        if (cl[g].lam != 0.0) throw std::logic_error("continuous::step: lam must be 0 with a face medium");
    auto const L = cl.leaf_indices.size();
    // pass 2 reads only u, so phi is updated in place (discrete's pass 3 fused)
    double const* g = nullptr;
    if (cap_S_ > 0.0) _cap_g(cl, an, nullptr), g = an.gcap.data();
    if (!an.medium()) {
      an.sig.resize(L);
      _par(L, [&](std::size_t b, std::size_t e) {
        for (std::size_t j = b; j < e; ++j) an.sig[j] = cl[cl.leaf_indices[j]].sigma();
      });
    }
    _par(L, [&](std::size_t b, std::size_t e) {
      EQ_SCOPE("equilibrium::dst::continuous::AdaptiveLattice::_pass1", time, perf_light);
      _pass1(cl, an, ts, b, e, g);
    });
  }

  void step_phi(AdaptiveCellCluster& cl, TimeStepper const& ts) {
    auto& an = anchors(cl);
    _par(cl.leaf_indices.size(), [&](std::size_t b, std::size_t e) {
      EQ_SCOPE("equilibrium::dst::continuous::AdaptiveLattice::_pass2", time, perf_light);
      _pass2(cl, an, ts, b, e);
    });
  }

  // No discrete analogue. dsig, dn: NF*L face-major deviations of sigma and
  // n = L/h at the face anchors, from a continuum field sampled there. Both
  // sides of a face are one anchor: they must agree (to round-off of the
  // midpoint evaluation) and are then stored as one value. Cleared by rebuild.
  void set_face_medium(AdaptiveCellCluster& cl, double const* dsig, double const* dn) {
    auto& an = anchors(cl);
    auto const L = cl.leaf_indices.size(), NF = anchor::NF;
    auto const inv_h = 1.0 / an.h;
    std::vector<double> coef(NF * L, 0.0), fsig(NF * L, 1.0);
    for (std::size_t j = 0; j < L; ++j)
      for (std::size_t k = 0; k < NF; ++k) {
        auto const n = an.nbj[j * NF + k];
        if (n < 0 || static_cast<std::size_t>(n) < j) continue;
        auto const f = k * L + j, m = anchor::mirror_face[k] * L + static_cast<std::size_t>(n);
        for (auto const* v : {dsig, dn})
          if (std::abs(v[f] - v[m]) > 1e-12 * (1.0 + std::abs(v[f])))
            throw std::runtime_error("set_face_medium: face anchor values differ across the face");
        coef[f] = coef[m] = (1.0 + dsig[f]) / (1.0 + dn[f]) * an.w[k] * inv_h;
        fsig[f] = fsig[m] = 1.0 + dsig[f];
      }
    an.coef = std::move(coef);
    an.fsig = std::move(fsig);
  }
  void clear_face_medium(AdaptiveCellCluster& cl) {
    anchors(cl).coef.clear();
    anchors(cl).fsig.clear();
  }

  void step(AdaptiveCellCluster& cl, double dt) { step(cl, TimeStepper::make_fixed(dt)); }
  void step_all(TimeStepper const& ts) {
    for (auto& c : clusters) step(c, ts);
  }
  void step_all(double dt) { step_all(TimeStepper::make_fixed(dt)); }

  // discrete: step_damped / step_dir_damped / step_pml. Analogue: the same
  // absorbing layers applied to the continuum update.
  void step_damped(AdaptiveCellCluster&, double) {
    throw not_implemented("step_damped", "step_damped");
  }
  void step_dir_damped(AdaptiveCellCluster&, double) {
    throw not_implemented("step_dir_damped", "step_dir_damped");
  }
  void step_pml(AdaptiveCellCluster&, double) { throw not_implemented("step_pml", "step_pml"); }

  // No discrete analogue. EXPERIMENTAL E field at the anchors, port of
  // continuous.AE (numpy, the reference; must agree to round-off):
  //   u_t = -c^2 div A,  A_t = -grad u - E,  E_t = c^2 curl curl A - J
  // curl curl = curl_curl_dec (default, set_curl(Curl::dec)) or the LS
  // reference curl_curl_sym (C0SQ antisym(K^T K phi), set_curl(Curl::ls)). Order: u (old phi), E (old phi), phi
  // (new u, new E). lam must be 0 (sigma placement with E undecided). J: NF*L
  // face-major or nullptr, applied as J_scale * J.
  void step_ae(AdaptiveCellCluster& cl, double dt, double const* J = nullptr, double J_scale = 1.0) {
    EQ_SCOPE("equilibrium::dst::continuous::AdaptiveLattice::step_ae");
    _require_no_face_bc(cl, "step_ae");
    auto& an = anchors(cl);
    auto const L  = cl.leaf_indices.size();
    auto const NF = anchor::NF;
    auto const dl = dt / an.h;
    for (auto const g : cl.leaf_indices)
      if (cl[g].lam != 0.0) throw std::logic_error("continuous::step_ae: lam must be 0");
    if (an.medium()) throw std::logic_error("continuous::step_ae: face medium not supported yet");
    _par(L, [&](std::size_t b, std::size_t e) {
      for (std::size_t j = b; j < e; ++j) {
        auto cell = cl[cl.leaf_indices[j]];
        double acc = 0.0;
        for (std::size_t k = 0; k < NF; ++k)
          if (an.nbj[j * NF + k] >= 0) acc += an.w[k] * cell.phi[k];
        if (!cell.fixed_u) cell.u += -dl * acc;
      }
    });
    bool const dec = curl_ != Curl::ls, geo = curl_ == Curl::dec_geo;
    if (dec)
      _dec_tri(cl, an, nullptr, geo);
    else
      _curl_curl_Q(cl, an, nullptr);
    _par(L, [&](std::size_t b, std::size_t e) {
      for (std::size_t j = b; j < e; ++j)
        for (std::size_t k = 0; k < NF; ++k) {
          auto& ek = an.e[k * L + j];
          if (an.nbj[j * NF + k] < 0) {
            ek = 0.0;
            continue;
          }
          double de = dt * an.c0sq * (dec ? _dec_face(an, j, k, geo) : _curl_curl_face(an, j, k));
          if (J) de -= dt * J_scale * J[k * L + j];
          ek += de;
        }
    });
    _par(L, [&](std::size_t b, std::size_t e) {
      for (std::size_t j = b; j < e; ++j) {
        auto cell = cl[cl.leaf_indices[j]];
        auto const u_i = cell.u;
        for (std::size_t k = 0; k < NF; ++k) {
          auto const n = an.nbj[j * NF + k];
          if (n < 0 || (cell.fixed_phi & (1u << k))) continue;
          auto const u_n = cl[cl.leaf_indices[static_cast<std::size_t>(n)]].u;
          cell.phi[k] += -dl * an.w[k] * (u_n - u_i) - dt * an.e[k * L + j];
        }
      }
    });
  }

  // continuous.curl_curl_sym: out = C0SQ antisym(K^T K phi), NF*L face-major.
  // phi = nullptr reads the cluster's own phi.
  void curl_curl_sym(AdaptiveCellCluster& cl, double const* phi, double* out) {
    auto& an = anchors(cl);
    auto const L = cl.leaf_indices.size();
    _curl_curl_Q(cl, an, phi);
    _par(L, [&](std::size_t b, std::size_t e) {
      for (std::size_t j = b; j < e; ++j)
        for (std::size_t k = 0; k < anchor::NF; ++k)
          out[k * L + j] = an.nbj[j * anchor::NF + k] < 0 ? 0.0 : _curl_curl_face(an, j, k);
    });
  }

  // DEC curl curl on the BCC Delaunay / Kelvin complex, port of
  // continuous.curl_curl_dec: circulation of the edge integrals a = h phi / W
  // around each Delaunay triangle; out = (W/h)^2 antisym(d1^T *2 d1 a) / *1
  // with d1 in the h/W-scaled form, a_c = 2h/sqrt3, *1 = r a_c/8 (hex) :
  // a_c/8 (square), *2 = s/a_c. In a = phi/W the operator is plain DEC times
  // C0SQ, so c_T^2 = C0SQ 4s/(r+1): s = (r+1)/4 gives c_T = c for any r.
  // The output is exactly divergence-free in sum_k (*1_k / W_k) out_k.
  // dec: r = (W_hex/W_sq)^2 (the u sector's own star: 2 iso, 9 discrete), so
  // that is step()'s sum_k W_k (no transverse -> u leak). dec_geo: r = 3,
  // s = 1 (Kelvin geometry of the coarse cell), which matches sum_k W_k only
  // if W_hex/W_sq = sqrt3, so it leaks into u under both Weights. curl(grad)
  // = 0 exactly for both. NF*L face-major.
  enum class Curl { ls, dec, dec_geo };
  void set_curl(Curl c) { curl_ = c; }
  Curl curl() const { return curl_; }

  void curl_curl_dec(AdaptiveCellCluster& cl, double const* phi, double* out, Curl star = Curl::dec) {
    auto& an = anchors(cl);
    auto const L   = cl.leaf_indices.size();
    auto const geo = star == Curl::dec_geo;
    _dec_tri(cl, an, phi, geo);
    _par(L, [&](std::size_t b, std::size_t e) {
      for (std::size_t j = b; j < e; ++j)
        for (std::size_t k = 0; k < anchor::NF; ++k)
          out[k * L + j] = an.nbj[j * anchor::NF + k] < 0 ? 0.0 : _dec_face(an, j, k, geo);
    });
  }

  // continuous.cartan_transport: self-transport -i_v(da) on the Delaunay
  // triangles (exp/2026/10/13/cartan). a = phi / W; per triangle i -k-> n -m->
  // l with loop a1 = a_ik, a2 = a_nm, a3 = -a_is, omega = a1 + a2 + a3: edge k
  // += omega (a2 - a3), edge m at n += omega (a3 - a1), edge s -= omega (a1 -
  // a2); out = 3/4 / (h W_k) antisym(gathered). Antisymmetric per triangle in
  // v = W phi, so sum phi out = 0 exactly; zero on a face whose triangles all
  // have omega = 0. phi = nullptr reads the cluster's own phi. NF*L face-major.
  // set_cartan_sigma(true) with a face medium: each triangle's three
  // contributions scale by the mean sigma_f of its edges (circulation carried
  // slower in slower cells); still antisymmetric per triangle, so sum phi out
  // = 0 is kept.
  // set_cartan_hold(kappa per leaf): context-blended vorticity. Each triangle's
  // loop sum om becomes kh om + (1 - kh) <om>, kh = mean kappa of its three
  // cells, <om> = mean of the same triangle template's om over the cell and its
  // face neighbours (the pattern as read over a cell of reach). A scalar per
  // triangle, so sum dH/dphi . out = 0 is kept; kappa = 1 == off. Cleared by
  // rebuild (exp/2026/10/22/depression/contextual_hold.md).
  // set_cartan_vcap(v > 0): smooth carrier bound, each triangle's terms x
  // 1 / sqrt(1 + <a^2> / v^2) (<a^2> = mean of its three edge a^2); a scalar
  // per triangle as with sigma, so sum phi out = 0 is kept. 0 = off.
  // set_cell_capacity(S_c > 0, p): finite holding capacity. Store per cell
  // S_j = 1/4 sum_k phi_jk^2, energy sum_j G(S_j) with G' = 1 + (S/S_c)^p
  // (G(S) = S + S_c/(p+1) (S/S_c)^(p+1)), so dH/dphi_f = phi_f (G'_j + G'_n)/2.
  // step(): pass 1 drives u by that instead of phi. cartan_transport: the
  // triangle's velocity slot takes b = a (G'_j + G'_n)/2 (dH/da), omega stays
  // the loop sum of a, so sum dH/dphi . out = 0 per triangle. 0 = off (== plain).
  void set_cell_capacity(double S_c, int p = 1) {
    if (p < 1) throw std::invalid_argument("set_cell_capacity: p >= 1");
    cap_S_ = S_c, cap_p_ = p;
  }
  double cell_capacity() const { return cap_S_; }
  void set_cartan_vcap(double v) { cartan_vcap_ = v; }
  double cartan_vcap() const { return cartan_vcap_; }
  void set_cartan_sigma(bool on) { cartan_sigma_ = on; }
  bool cartan_sigma() const { return cartan_sigma_; }
  void set_cartan_hold(AdaptiveCellCluster& cl, double const* kappa) {
    auto& an = anchors(cl);
    if (!kappa) return an.hold.clear();
    an.hold.assign(kappa, kappa + cl.leaf_indices.size());
  }
  void clear_cartan_hold(AdaptiveCellCluster& cl) { anchors(cl).hold.clear(); }

  void cartan_transport(AdaptiveCellCluster& cl, double const* phi, double* out) {
    EQ_SCOPE("equilibrium::dst::continuous::AdaptiveLattice::cartan_transport");
    auto& an = anchors(cl);
    auto const L  = cl.leaf_indices.size();
    auto const NF = anchor::NF;
    an.cart.resize(L * 36);
    bool const hold = !an.hold.empty();
    bool const   sig = cartan_sigma_ && !an.fsig.empty();
    double const iv2 = cartan_vcap_ > 0.0 ? 1.0 / (3.0 * cartan_vcap_ * cartan_vcap_) : 0.0;
    bool const   cap = cap_S_ > 0.0;
    if (cap) _cap_g(cl, an, phi);
    auto const& g = an.gcap;
    auto phi_at = [&](std::size_t j, std::size_t k) {
      return phi ? phi[k * L + j] : cl[cl.leaf_indices[j]].phi[k];
    };
    auto const& w = an.w;
    std::array<double, anchor::NF> iw;
    for (std::size_t k = 0; k < NF; ++k) iw[k] = 1.0 / w[k];
    auto const& tr = anchor::tris();
    if (hold) {
      an.omb.resize(L * 12);
      _par(L, [&](std::size_t b, std::size_t e) {
        for (std::size_t j = b; j < e; ++j)
          for (std::size_t t = 0; t < 12; ++t) {
            auto const [s, k, m] = tr[t];
            auto const n = an.nbj[j * NF + k], l = an.nbj[j * NF + s];
            an.omb[j * 12 + t] = (n >= 0 && l >= 0 && an.nbj[static_cast<std::size_t>(n) * NF + m] == l)
                                     ? phi_at(j, k) * iw[k] + phi_at(static_cast<std::size_t>(n), m) * iw[m] -
                                           phi_at(j, s) * iw[s]
                                     : std::numeric_limits<double>::quiet_NaN();
          }
      });
    }
    _par(L, [&](std::size_t b, std::size_t e) {
      EQ_SCOPE("equilibrium::dst::continuous::AdaptiveLattice::cartan_tri", time, perf_light);
      for (std::size_t j = b; j < e; ++j) {
        EQ_MCA_BEGIN("cartan_tri");
        for (std::size_t t = 0; t < 12; ++t) {
          auto const [s, k, m] = tr[t];
          auto const n = an.nbj[j * NF + k], l = an.nbj[j * NF + s];
          double* c = &an.cart[j * 36 + t * 3];
          c[0] = c[1] = c[2] = 0.0;
          if (n >= 0 && l >= 0 && an.nbj[static_cast<std::size_t>(n) * NF + m] == l) {
            double const a1 = phi_at(j, k) * iw[k], a2 = phi_at(static_cast<std::size_t>(n), m) * iw[m],
                         a3 = -phi_at(j, s) * iw[s];
            double om = a1 + a2 + a3;
            if (hold) {
              double avg = an.omb[j * 12 + t];
              int    cnt = 1;
              for (std::size_t q = 0; q < NF; ++q) {
                auto const g = an.nbj[j * NF + q];
                if (g < 0) continue;
                auto const v = an.omb[static_cast<std::size_t>(g) * 12 + t];
                if (v == v) avg += v, ++cnt;
              }
              double const kh = (an.hold[j] + an.hold[static_cast<std::size_t>(n)] + an.hold[static_cast<std::size_t>(l)]) / 3.0;
              om = kh * om + (1.0 - kh) * avg / cnt;
            }
            if (sig)
              om *= (an.fsig[k * L + j] + an.fsig[m * L + static_cast<std::size_t>(n)] + an.fsig[s * L + j]) / 3.0;
            if (iv2 > 0.0) om /= std::sqrt(1.0 + (a1 * a1 + a2 * a2 + a3 * a3) * iv2);
            double b1 = a1, b2 = a2, b3 = a3;
            if (cap) {
              auto const nn = static_cast<std::size_t>(n), ll = static_cast<std::size_t>(l);
              b1 *= 0.5 * (g[j] + g[nn]), b2 *= 0.5 * (g[nn] + g[ll]), b3 *= 0.5 * (g[j] + g[ll]);
            }
            c[0] = om * (b2 - b3);
            c[1] = om * (b3 - b1);
            c[2] = -om * (b1 - b2);
          }
        }
        EQ_MCA_END("cartan_tri");
      }
    });
    std::array<double, anchor::NF> fcoef;
    for (std::size_t k = 0; k < NF; ++k) fcoef[k] = 0.75 / (an.h * w[k]);
    auto const& ct = _cart_tab();
    _par(L, [&](std::size_t b, std::size_t e) {
      EQ_SCOPE("equilibrium::dst::continuous::AdaptiveLattice::cartan_face", time, perf_light);
      for (std::size_t j = b; j < e; ++j) {
        EQ_MCA_BEGIN("cartan_face");
        for (std::size_t k = 0; k < NF; ++k) {
          auto const n = an.nbj[j * NF + k];
          out[k * L + j] = n < 0 ? 0.0
                                 : fcoef[k] * (_cart_sum(an, ct, k, j) -
                                               _cart_sum(an, ct, anchor::mirror_face[k], static_cast<std::size_t>(n)));
        }
        EQ_MCA_END("cartan_face");
      }
    });
  }

  // No discrete analogue. step() then the transport, implicit midpoint as in
  // exp/2026/10/13/cartan/vill.py: ref = phi before step(), ph0 after, d =
  // T((ref + ph0) / 2), n_iter times d = T((ref + ph0 + a d) / 2), phi = ph0 +
  // a d with a = kappa dt.
  void step_cartan(AdaptiveCellCluster& cl, double dt, double kappa, std::size_t n_iter) {
    EQ_SCOPE("equilibrium::dst::continuous::AdaptiveLattice::step_cartan");
    auto& an = anchors(cl);
    auto const L = cl.leaf_indices.size(), NF = anchor::NF;
    auto load = [&](std::vector<double>& v) {
      v.resize(NF * L);
      _par(L, [&](std::size_t b, std::size_t e) {
        for (std::size_t j = b; j < e; ++j)
          for (std::size_t k = 0; k < NF; ++k) v[k * L + j] = cl[cl.leaf_indices[j]].phi[k];
      });
    };
    load(an.cref);
    step(cl, dt);
    load(an.cph0);
    an.cmid.resize(NF * L), an.cd.assign(NF * L, 0.0);
    double const a = kappa * dt;
    for (std::size_t it = 0; it <= n_iter; ++it) {
      _par(L, [&](std::size_t b, std::size_t e) {
        EQ_MCA_BEGIN("cartan_mid");
        for (std::size_t k = 0; k < NF; ++k)
          for (std::size_t j = b; j < e; ++j) {
            auto const f = k * L + j;
            an.cmid[f] = 0.5 * (an.cref[f] + an.cph0[f] + a * an.cd[f]);
          }
        EQ_MCA_END("cartan_mid");
      });
      cartan_transport(cl, an.cmid.data(), an.cd.data());
    }
    _par(L, [&](std::size_t b, std::size_t e) {
      for (std::size_t j = b; j < e; ++j)
        for (std::size_t k = 0; k < NF; ++k) {
          auto const f = k * L + j;
          cl[cl.leaf_indices[j]].phi[k] = an.cph0[f] + a * an.cd[f];
        }
    });
  }

  // discrete: step_lam. Analogue: continuum lam evolution; the discrete clamp
  // to N_SAGA-1 is a substrate limit and is expected to carry over as a bound.
  void step_lam(AdaptiveCellCluster&, double, double) {
    throw not_implemented("step_lam", "step_lam");
  }

  // No discrete analogue. Two-component BCC walk with a local density turn, level 0 (equilibrium/substrate/walk.py
  // is the reference and must agree to round-off): per leaf
  //   L1 = n sum_h A_h L(x + h) + m J R,  R1 = m J L + n sum_h A_h^T R(x - h)  (AL / AR as given)
  //   every pair turns by -g (|L1|^2 + |R1|^2);  sponge: X = bg(t+1) + (X1 - bg(t+1)) f
  // Missing neighbours read bg(t). parity: odd t uses AR for L and AL for R (parity image tick).
  // Not a continuum anchor rule: it lives on the level-0 cells as they are.
  void set_walk(AdaptiveCellCluster& cl, double const* AL, double const* AR, std::int32_t const* kL,
                std::int32_t const* kR, double n, double m, double g, double const* bgL, double const* bgR,
                double mu) {
    auto& w = anchors(cl).walk;
    auto const L = cl.leaf_indices.size();
    std::copy(AL, AL + 8 * 16, w.AL.begin());
    std::copy(AR, AR + 8 * 16, w.AR.begin());
    for (std::size_t h = 0; h < 8; ++h) {
      if (kL[h] < 0 || kL[h] >= 8 || kR[h] < 0 || kR[h] >= 8)
        throw std::invalid_argument("set_walk: hop faces must be hex faces (0..7)");
      w.kL[h] = kL[h];
      w.kR[h] = kR[h];
    }
    std::copy(bgL, bgL + 4, w.bgL.begin());
    std::copy(bgR, bgR + 4, w.bgR.begin());
    w.n = n, w.m = m, w.g = g, w.mu = mu, w.t = 0, w.beta = 0.0;
    w.L.assign(4 * L, 0.0), w.R.assign(4 * L, 0.0), w.L1.assign(4 * L, 0.0), w.R1.assign(4 * L, 0.0);
    w.f.clear();
    w.set = true;
  }
  anchor::ClusterAnchors::Walk& walk(AdaptiveCellCluster& cl) {
    auto& w = anchors(cl).walk;
    if (!w.set) throw std::logic_error("continuous: set_walk before using the walk");
    return w;
  }
  void set_walk_sponge(AdaptiveCellCluster& cl, double const* f) {
    auto& w = walk(cl);
    w.f.assign(f, f + cl.leaf_indices.size());
  }

  void set_walk_parity(AdaptiveCellCluster& cl, bool parity) { walk(cl).parity = parity; }

  // unused outside TestWalkClock since 10-08 (walk = reified approximation); removal pending
  // local clock (cell.L / L0 as a function of depletion): sigma = 1 + beta (rho0 - rho1) / rho0 per cell after the hop;
  // the on-site rates scale by sigma: an extra mass-coin turn (sigma - 1) om0 (exact rest-frequency shift) and the
  // density turn g sigma rho1. Per-cell rotations keep each cell's norm; sigma is read from rho1, which they preserve.
  // The hop (speed) is not scaled.
  void set_walk_clock(AdaptiveCellCluster& cl, double beta, double om0, double rho0) {
    auto& w = walk(cl);
    if (!(rho0 > 0.0)) throw std::invalid_argument("set_walk_clock: rho0 > 0");
    w.beta = beta, w.om0 = om0, w.rho0 = rho0;
  }

  void step_walk(AdaptiveCellCluster& cl, std::size_t n_steps = 1) {
    auto& an = anchors(cl);
    auto& w  = walk(cl);
    auto const L = cl.leaf_indices.size(), NF = anchor::NF;
    for (std::size_t s = 0; s < n_steps; ++s) {
      bool const swap_ab = w.parity && (w.t % 2 != 0);
      double const* AL = swap_ab ? w.AR.data() : w.AL.data();
      double const* AR = swap_ab ? w.AL.data() : w.AR.data();
      std::array<double, 4> gL0, gR0, gL1, gR1;
      _turn_pairs(w.bgL.data(), w.mu * static_cast<double>(w.t), gL0.data());
      _turn_pairs(w.bgR.data(), w.mu * static_cast<double>(w.t), gR0.data());
      _turn_pairs(w.bgL.data(), w.mu * static_cast<double>(w.t + 1), gL1.data());
      _turn_pairs(w.bgR.data(), w.mu * static_cast<double>(w.t + 1), gR1.data());
      bool const sponge = !w.f.empty();
      _par(L, [&](std::size_t b, std::size_t e) {
        for (std::size_t j = b; j < e; ++j) {
          double Lh[4] = {0, 0, 0, 0}, Rh[4] = {0, 0, 0, 0};
          for (std::size_t h = 0; h < 8; ++h) {
            auto const nl = an.nbj[j * NF + static_cast<std::size_t>(w.kL[h])];
            auto const nr = an.nbj[j * NF + static_cast<std::size_t>(w.kR[h])];
            double const* sl = nl >= 0 ? &w.L[4 * static_cast<std::size_t>(nl)] : gL0.data();
            double const* sr = nr >= 0 ? &w.R[4 * static_cast<std::size_t>(nr)] : gR0.data();
            double const* al = &AL[16 * h];
            double const* ar = &AR[16 * h];
            for (int r = 0; r < 4; ++r) {
              Lh[r] += al[4 * r] * sl[0] + al[4 * r + 1] * sl[1] + al[4 * r + 2] * sl[2] + al[4 * r + 3] * sl[3];
              Rh[r] += ar[4 * r] * sr[0] + ar[4 * r + 1] * sr[1] + ar[4 * r + 2] * sr[2] + ar[4 * r + 3] * sr[3];
            }
          }
          double const* Lj = &w.L[4 * j];
          double const* Rj = &w.R[4 * j];
          double JL[4] = {-Lj[1], Lj[0], -Lj[3], Lj[2]}, JR[4] = {-Rj[1], Rj[0], -Rj[3], Rj[2]};
          double* L1 = &w.L1[4 * j];
          double* R1 = &w.R1[4 * j];
          double rho = 0.0;
          for (int r = 0; r < 4; ++r) {
            L1[r] = w.n * Lh[r] + w.m * JR[r];
            R1[r] = w.m * JL[r] + w.n * Rh[r];
            rho += L1[r] * L1[r] + R1[r] * R1[r];
          }
          double sig = 1.0;
          if (w.beta != 0.0) {
            sig = 1.0 + w.beta * (w.rho0 - rho) / w.rho0;
            double const ph = (sig - 1.0) * w.om0, c = std::cos(ph), sn = std::sin(ph);
            double const JL1[4] = {-L1[1], L1[0], -L1[3], L1[2]}, JR1[4] = {-R1[1], R1[0], -R1[3], R1[2]};
            for (int r = 0; r < 4; ++r) {
              L1[r] = c * L1[r] + sn * JR1[r];
              R1[r] = sn * JL1[r] + c * R1[r];
            }
          }
          double tmp[4];
          _turn_pairs(L1, w.g * sig * rho, tmp);
          std::copy(tmp, tmp + 4, L1);
          _turn_pairs(R1, w.g * sig * rho, tmp);
          std::copy(tmp, tmp + 4, R1);
          if (sponge) {
            auto const f = w.f[j];
            for (int r = 0; r < 4; ++r) {
              L1[r] = gL1[r] + (L1[r] - gL1[r]) * f;
              R1[r] = gR1[r] + (R1[r] - gR1[r]) * f;
            }
          }
        }
      });
      std::swap(w.L, w.L1);
      std::swap(w.R, w.R1);
      ++w.t;
    }
  }

  // discrete: step_sponge, same rule (bc.hpp: u toward target per leaf, one factor per face for
  // both copies), extended to E at the anchors when it has been allocated.
  void step_sponge(AdaptiveCellCluster& cl, double dt = 0.0) {
    if (!cl.bc || !cl.bc->has_sponge()) return;
    if (!cl.bc->vol_rate.empty() && !(dt > 0.0))
      throw std::invalid_argument("continuous::step_sponge: rate sponge needs dt > 0");
    auto& an      = anchors(cl);
    auto const& bc = *cl.bc;
    auto const L  = cl.leaf_indices.size();
    auto const NF = anchor::NF;
    bool const has_e = an.e.size() == NF * L;
    _par(L, [&](std::size_t b, std::size_t e) {
      for (std::size_t j = b; j < e; ++j) {
        auto cell = cl[cl.leaf_indices[j]];
        if (!cell.fixed_u) cell.u = bc.target + (cell.u - bc.target) * bc.sponge_leaf(j, dt);
        for (std::size_t k = 0; k < NF; ++k) {
          auto const f = bc.sponge_face(j, an.nbj[j * NF + k], dt);
          if (!(cell.fixed_phi & (1u << k))) cell.phi[k] *= f;
          if (has_e) an.e[k * L + j] *= f;
        }
      }
    });
  }
  void step_sponge_all(double dt = 0.0) {
    for (auto& c : clusters) step_sponge(c, dt);
  }

  // discrete: split + bcc::wire_split_children. Analogue: children are new
  // anchors sampling the parent's continuum reconstruction (prolong), not an
  // integer division of face flux among external faces.
  bool can_split(AdaptiveCellCluster& cluster, std::int32_t cell_g) {
    return cluster[cell_g].level < cluster.info.max_level;
  }
  std::int32_t split(AdaptiveCellCluster&, std::int32_t) { throw not_implemented("split", "split"); }

  // discrete: coarsen / accumulate_to_parent. Analogue: parent sample is the
  // continuum average of the children's reconstruction (restrict); must be
  // consistent with prolong (discrete's mean-restrict vs n_ext-divide-prolong
  // are not inverses).
  void coarsen(AdaptiveCellCluster&, std::int32_t) { throw not_implemented("coarsen", "coarsen"); }
  void prolong(AdaptiveCellCluster&, std::int32_t) {
    throw not_implemented("prolong", "_prolong_u_gradient_wide / _prolong_phi_generic");
  }
  void restrict_to_parent(AdaptiveCellCluster&, std::int32_t) {
    throw not_implemented("restrict_to_parent", "accumulate_to_parent");
  }

  // discrete: adapt (threshold split/coarsen on max|phi_k|). Analogue: a
  // refinement criterion on the reconstruction's resolution error.
  void adapt(AdaptiveCellCluster&) { throw not_implemented("adapt", "adapt"); }
  void adapt_all() {
    for (auto& c : clusters) adapt(c);
  }

  // discrete: step_subcycle + _apply_amr_reflux. Analogue: time refinement
  // with conservation enforced on the continuum flux through the coarse/fine
  // surface.
  void step_subcycle(double, double) { throw not_implemented("step_subcycle", "step_all_subcycle"); }

 private:
  template <typename F>
  void _par(std::size_t n, F&& f) {
    if (!pool_) {
      f(std::size_t{0}, n);
      return;
    }
    pool_->detach_blocks<std::size_t>(0, n, f);
    pool_->wait();
  }

  // every pair (a, b) of x (4 reals) turned by -th: (a cos + b sin, b cos - a sin) (e^{-i th} on a + i b).
  static void _turn_pairs(double const* x, double th, double* out) {
    auto const c = std::cos(th), s = std::sin(th);
    for (int p = 0; p < 4; p += 2) {
      auto const a = x[p], b = x[p + 1];
      out[p]     = a * c + b * s;
      out[p + 1] = b * c - a * s;
    }
  }

  static void _require_no_face_bc(AdaptiveCellCluster& cl, std::string const& what) {
    if (!cl.bc) return;
    auto const& bc = *cl.bc;
    if (bc.fn || bc.has_face_ghosts() || bc.remote_flux != 0.0 || !bc.outflow_active.empty() ||
        !bc.outflow_taper_weight.empty() || !bc.gamma.empty() || !bc.gamma_face.empty())
      throw not_implemented(what + " with a face BC", "_step_pass2_bc");
  }

  // W_k sigma_f / (h n_f) on face k of leaf j (neighbour n).
  static double _coef(AdaptiveCellCluster& cl, anchor::ClusterAnchors const& an, std::size_t j,
                      std::size_t k, std::int32_t n, double inv_h) {
    if (an.medium()) return an.coef[k * cl.leaf_indices.size() + j];
    return 0.5 * (an.sig[j] + an.sig[static_cast<std::size_t>(n)]) * an.w[k] * inv_h;
  }

  static void _pass1(AdaptiveCellCluster& cl, anchor::ClusterAnchors const& an, TimeStepper const& ts,
                     std::size_t b, std::size_t e, double const* g = nullptr) {
    auto const inv_h = 1.0 / an.h;
    for (std::size_t j = b; j < e; ++j) {
      EQ_MCA_BEGIN("continuous_pass1");
      auto cell       = cl[cl.leaf_indices[j]];
      auto const dt   = ts(cell.L, cell.level);
      double acc      = 0.0;
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        auto const n = an.nbj[j * CellSlab::NF + k];
        if (n >= 0)
          acc += _coef(cl, an, j, k, n, inv_h) * cell.phi[k] *
                 (g ? 0.5 * (g[j] + g[static_cast<std::size_t>(n)]) : 1.0);
      }
      if (!cell.fixed_u) cell.u += -dt * acc;
      EQ_MCA_END("continuous_pass1");
    }
  }

  static void _pass2(AdaptiveCellCluster& cl, anchor::ClusterAnchors const& an, TimeStepper const& ts,
                     std::size_t b, std::size_t e) {
    auto const inv_h = 1.0 / an.h;
    for (std::size_t j = b; j < e; ++j) {
      EQ_MCA_BEGIN("continuous_pass2");
      auto cell      = cl[cl.leaf_indices[j]];
      auto const dt  = ts(cell.L, cell.level);
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        auto const n = an.nbj[j * CellSlab::NF + k];
        if (n < 0 || (cell.fixed_phi & (1u << k))) continue;
        cell.phi[k] += -dt * _coef(cl, an, j, k, n, inv_h) *
                       (cl[cl.leaf_indices[static_cast<std::size_t>(n)]].u - cell.u);
      }
      EQ_MCA_END("continuous_pass2");
    }
  }

  // G = [B]_x, the antisymmetric gradient curl_K_adjoint spreads to the faces.
  static std::array<std::array<double, 3>, 3> _cross(double const* Bj) {
    std::array<std::array<double, 3>, 3> G{};
    G[1][2] += Bj[0];
    G[2][1] -= Bj[0];
    G[2][0] += Bj[1];
    G[0][2] -= Bj[1];
    G[0][1] += Bj[2];
    G[1][0] -= Bj[2];
    return G;
  }

  // Stages of curl_curl_sym up to Q = K^T K phi reconstructed per leaf (in
  // an.A): A = Pw phi, B = curl A, Q_j = sum_k (dA_nb,mirror - dA_j,k) with
  // dA_j,k = P_j[:,k] . [B_j]_x / h computed on the fly.
  void _curl_curl_Q(AdaptiveCellCluster& cl, anchor::ClusterAnchors& an, double const* phi) {
    auto const L  = cl.leaf_indices.size();
    auto const NF = anchor::NF;
    auto const inv_h = 1.0 / an.h;
    auto phi_at = [&](std::size_t j, std::size_t k) {
      return phi ? phi[k * L + j] : cl[cl.leaf_indices[j]].phi[k];
    };
    _par(L, [&](std::size_t b, std::size_t e) {
      for (std::size_t j = b; j < e; ++j) {
        auto const* pw = an.pw(j);
        for (int r = 0; r < 3; ++r) {
          double a = 0.0;
          for (std::size_t k = 0; k < NF; ++k)
            if (an.nbj[j * NF + k] >= 0) a += pw[r * NF + k] * phi_at(j, k);
          an.A[j * 3 + r] = a;
        }
      }
    });
    _par(L, [&](std::size_t b, std::size_t e) {
      for (std::size_t j = b; j < e; ++j) {
        auto const* pj = an.p(j);
        double G[3][3] = {};
        for (std::size_t k = 0; k < NF; ++k) {
          auto const n = an.nbj[j * NF + k];
          if (n < 0) continue;
          for (int c = 0; c < 3; ++c) {
            auto const dA = an.A[static_cast<std::size_t>(n) * 3 + c] - an.A[j * 3 + c];
            for (int r = 0; r < 3; ++r) G[r][c] += pj[r * NF + k] * dA;
          }
        }
        an.B[j * 3 + 0] = (G[1][2] - G[2][1]) * inv_h;
        an.B[j * 3 + 1] = (G[2][0] - G[0][2]) * inv_h;
        an.B[j * 3 + 2] = (G[0][1] - G[1][0]) * inv_h;
      }
    });
    // Scatter A[nb] += dA as a gather: level-0 wiring is symmetric,
    // nb[nb[j][k]][mirror k] = j. A is dead after B, so Q reuses it.
    _par(L, [&](std::size_t b, std::size_t e) {
      for (std::size_t j = b; j < e; ++j) {
        auto const* pj = an.p(j);
        auto const Gj  = _cross(&an.B[j * 3]);
        double Q[3]    = {};
        for (std::size_t k = 0; k < NF; ++k) {
          auto const n = an.nbj[j * NF + k];
          if (n < 0) continue;
          auto const nn = static_cast<std::size_t>(n);
          auto const* pn = an.p(nn);
          auto const Gn  = _cross(&an.B[nn * 3]);
          auto const km  = anchor::mirror_face[k];
          for (int c = 0; c < 3; ++c) {
            double own = 0.0, back = 0.0;
            for (int r = 0; r < 3; ++r) {
              own += pj[r * NF + k] * Gj[r][c];
              back += pn[r * NF + km] * Gn[r][c];
            }
            Q[c] += back * inv_h - own * inv_h;
          }
        }
        for (int c = 0; c < 3; ++c) an.A[j * 3 + c] = Q[c];
      }
    });
  }

  // C0SQ (Pw_j[:,k] . Q_j - Pw_n[:,mirror k] . Q_n), after _curl_curl_Q.
  static double _curl_curl_face(anchor::ClusterAnchors const& an, std::size_t j, std::size_t k) {
    auto const NF = anchor::NF;
    auto const n  = static_cast<std::size_t>(an.nbj[j * NF + k]);
    auto const km = anchor::mirror_face[k];
    auto const* pj = an.pw(j);
    auto const* pn = an.pw(n);
    double yj = 0.0, yn = 0.0;
    for (int r = 0; r < 3; ++r) {
      yj += pj[r * NF + k] * an.A[j * 3 + r];
      yn += pn[r * NF + km] * an.A[n * 3 + r];
    }
    return an.c0sq * (yj - yn);
  }

  // DEC star (r, s): *1 = r a_c/8 (hex) : a_c/8 (square), *2 = s/a_c.
  static std::array<double, 2> _dec_star(anchor::ClusterAnchors const& an, bool geo) {
    if (geo) return {3.0, 1.0};
    auto const r = an.w[0] * an.w[0] / (an.w[8] * an.w[8]);
    return {r, 0.25 * (r + 1.0)};
  }

  // *2 x circulation per triangle template, 0 where a triangle is missing.
  void _dec_tri(AdaptiveCellCluster& cl, anchor::ClusterAnchors& an, double const* phi, bool geo) {
    auto const L  = cl.leaf_indices.size();
    auto const NF = anchor::NF;
    auto const h  = an.h;
    auto const s2 = _dec_star(an, geo)[1] * std::sqrt(3.0) / (2.0 * h);
    an.tri.resize(L * 12);
    auto phi_at = [&](std::size_t j, std::size_t k) {
      return phi ? phi[k * L + j] : cl[cl.leaf_indices[j]].phi[k];
    };
    auto const& w = an.w;
    _par(L, [&](std::size_t b, std::size_t e) {
      for (std::size_t j = b; j < e; ++j)
        for (std::size_t t = 0; t < 12; ++t) {
          auto const [s, k, m] = anchor::tris()[t];
          auto const n = an.nbj[j * NF + k], l = an.nbj[j * NF + s];
          double c = 0.0;
          if (n >= 0 && l >= 0 && an.nbj[static_cast<std::size_t>(n) * NF + m] == l)
            c = s2 * h *
                (phi_at(j, k) / w[k] + phi_at(static_cast<std::size_t>(n), m) / w[m] - phi_at(j, s) / w[s]);
          an.tri[j * 12 + t] = c;
        }
    });
  }

  // d1^T gathered onto the face copy (j, k), without the h/W_k factor: first
  // hex edge of a triangle based at j, its second edge (based one hop back
  // along mirror k'), or its square edge (subtracted).
  static double _dec_sum(anchor::ClusterAnchors const& an, std::size_t j, std::size_t k) {
    auto const NF = anchor::NF;
    double y = 0.0;
    for (std::size_t t = 0; t < 12; ++t) {
      auto const [s, kk, m] = anchor::tris()[t];
      if (kk == k) y += an.tri[j * 12 + t];
      if (s == k) y -= an.tri[j * 12 + t];
      if (m == k) {
        auto const b = an.nbj[j * NF + anchor::mirror_face[kk]];
        if (b >= 0) y += an.tri[static_cast<std::size_t>(b) * 12 + t];
      }
    }
    return y;
  }

  // cartan_transport contributions gathered onto the face copy (j, k), as
  // _dec_sum, from a per-face table: (triangle t, slot, hop to its base or -1).
  struct CartG {
    std::uint8_t t, slot;
    std::int8_t via;
  };
  static std::array<std::vector<CartG>, anchor::NF> const& _cart_table() {
    static auto const g = [] {
      std::array<std::vector<CartG>, anchor::NF> r;
      for (std::uint8_t t = 0; t < 12; ++t) {
        auto const [s, kk, m] = anchor::tris()[t];
        r[kk].push_back({t, 0, -1});
        r[s].push_back({t, 2, -1});
        r[m].push_back({t, 1, static_cast<std::int8_t>(anchor::mirror_face[kk])});
      }
      return r;
    }();
    return g;
  }
  struct CartTab {
    std::array<std::array<CartG, 6>, anchor::NF> e{};
    std::array<std::uint8_t, anchor::NF> n{};
  };
  static CartTab const& _cart_tab() {
    static auto const t = [] {
      CartTab r;
      auto const& v = _cart_table();
      for (std::size_t k = 0; k < anchor::NF; ++k) {
        if (v[k].size() > r.e[k].size()) throw std::logic_error("_cart_tab: more than 6 entries on a face");
        r.n[k] = static_cast<std::uint8_t>(v[k].size());
        std::copy(v[k].begin(), v[k].end(), r.e[k].begin());
      }
      return r;
    }();
    return t;
  }
  // an.gcap[j] = G'(S_j) = 1 + (S_j/S_c)^p, S_j = 1/4 sum_k phi_jk^2.
  void _cap_g(AdaptiveCellCluster& cl, anchor::ClusterAnchors& an, double const* phi) {
    auto const L = cl.leaf_indices.size(), NF = anchor::NF;
    an.gcap.resize(L);
    double const inv = 1.0 / cap_S_;
    _par(L, [&](std::size_t b, std::size_t e) {
      for (std::size_t j = b; j < e; ++j) {
        double S = 0.0;
        for (std::size_t k = 0; k < NF; ++k) {
          if (an.nbj[j * NF + k] < 0) continue;
          double const v = phi ? phi[k * L + j] : cl[cl.leaf_indices[j]].phi[k];
          S += v * v;
        }
        double const x = 0.25 * S * inv;
        double xp = x;
        for (int i = 1; i < cap_p_; ++i) xp *= x;
        an.gcap[j] = 1.0 + xp;
      }
    });
  }

  static double _cart_sum(anchor::ClusterAnchors const& an, CartTab const& tab, std::size_t k, std::size_t j) {
    double y = 0.0;
    for (std::size_t i = 0; i < tab.n[k]; ++i) {
      auto const& g = tab.e[k][i];
      std::int64_t b = static_cast<std::int64_t>(j);
      if (g.via >= 0) b = an.nbj[j * anchor::NF + static_cast<std::size_t>(g.via)];
      if (b >= 0) y += an.cart[static_cast<std::size_t>(b) * 36 + g.t * 3u + g.slot];
    }
    return y;
  }

  // (W_k/h) / *1_k (y_j,k - y_n,mirror k), after _dec_tri with the same star.
  static double _dec_face(anchor::ClusterAnchors const& an, std::size_t j, std::size_t k, bool geo) {
    auto const n  = static_cast<std::size_t>(an.nbj[j * anchor::NF + k]);
    auto const ac = 2.0 * an.h / std::sqrt(3.0);
    auto const s1 = k < 8 ? _dec_star(an, geo)[0] * ac / 8.0 : ac / 8.0;
    return an.w[k] / an.h / s1 * (_dec_sum(an, j, k) - _dec_sum(an, n, anchor::mirror_face[k]));
  }

  Curl curl_ = Curl::dec;
  bool cartan_sigma_ = false;
  double cap_S_ = 0.0;
  int cap_p_ = 1;
  double cartan_vcap_ = 0.0;
  anchor::Weights weights_ = anchor::Weights::iso;
  std::size_t n_threads_ = 1;
  std::unique_ptr<BS::thread_pool<>> pool_;
  std::vector<anchor::ClusterAnchors> anchors_;
};

// discrete: AdaptiveLatticeEvolver. Same dispatch on TimeStepper.
class AdaptiveLatticeEvolver {
 public:
  explicit AdaptiveLatticeEvolver(AdaptiveLattice& lat) : lat_{lat} {}

  void step(TimeStepper const& ts, std::size_t n_steps = 1) {
    for (std::size_t i = 0; i < n_steps; ++i) {
      if (ts.subcycle)
        lat_.step_subcycle(ts.dt_base, ts.ratio);
      else
        lat_.step_all(ts);
    }
  }
  void step(double dt, std::size_t n_steps = 1) { step(TimeStepper::make_fixed(dt), n_steps); }
  void amr_adapt() { lat_.adapt_all(); }
  void rebuild() { lat_.rebuild_all(); }

 private:
  AdaptiveLattice& lat_;
};

}  // namespace equilibrium::dst::continuous

#endif  // _EQUILIBRIUM_CORE_DATA_LATTICE_CONTINUOUS_ADAPTIVE_LATTICE_HPP_
