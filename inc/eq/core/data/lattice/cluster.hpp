#ifndef _EQUILIBRIUM_CORE_DATA_LATTICE_CLUSTER_HPP_
#define _EQUILIBRIUM_CORE_DATA_LATTICE_CLUSTER_HPP_

#include "eq/core/data/lattice/bc.hpp"
#include "eq/core/data/lattice/dst_info.hpp"
#include "eq/core/data/lattice/split_criterion.hpp"
#include "eq/core/data/lattice/cell/adaptive_cell.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <unordered_map>
#include <vector>
#include <array>
#include <memory>
#include <utility>

namespace equilibrium::dst::discrete {

struct RemoteCellRef {
  std::int32_t cluster_id;
  std::int32_t local_idx;
};

struct SendEntry {
  std::int32_t local_idx;
  std::int32_t remote_ghost_idx;
};

struct ClusterSendBatch {
  std::int32_t remote_cluster_id;
  std::vector<SendEntry> entries;
};

// ---------------------------------------------------------------------------
// MultiplicityTable — storage for face_multiplicity/ancestor_group_multiplicity.
//
// Both tables are 1.0 (no-op) for the overwhelming majority of (leaf, face)
// entries — only cells adjacent to a coarse-fine AMR junction ever deviate,
// and on some meshes (measured: a real deep-telescope spheromak checkpoint)
// face_multiplicity deviates nowhere at all. `all_one` short-circuits that
// common case to a single bool + an empty vector instead of an NF*L-sized
// allocation. When at least one entry deviates, falls back to a dense float
// array — double precision isn't needed here: these are one-shot post-hoc
// divisors (Σface_weight² ratios of small integers), never accumulated —
// half the size of the std::vector<double> this replaces.
// ---------------------------------------------------------------------------
struct MultiplicityTable {
  bool all_one = true;
  std::vector<float> data;  // NF * L when !all_one, empty when all_one

  [[nodiscard]] double get(std::size_t k, std::size_t L, std::size_t j) const noexcept {
    return all_one ? 1.0 : static_cast<double>(data[k * L + j]);
  }

  // Rebuild from a freshly computed dense double buffer (NF*L, default 1.0
  // for no-op entries). Collapses to all_one when every entry is exactly 1.0.
  void rebuild_from(std::vector<double> const& dense) {
    all_one = std::all_of(dense.begin(), dense.end(), [](double v) { return v == 1.0; });
    if (all_one) {
      data = decltype(data){};
    } else {
      data.assign(dense.begin(), dense.end());
    }
  }
};

// ---------------------------------------------------------------------------
// CellSlab — one fixed-size SoA block.
//
// Global cell index:  global = slab_idx * CAPACITY + slot
// Fast decomposition: slab   = global >> CAPACITY_LOG2
//                     slot   = global &  CAPACITY_MASK
//
// Face-major layout:  phi[k * CAPACITY + slot],  nb[k * CAPACITY + slot]
// Coord-major layout: pos[d * CAPACITY + slot]
//
// The slab is one aligned_alloc — no realloc, no pointer invalidation when
// the chain grows.  All pointers remain valid for the lifetime of the slab.
// ---------------------------------------------------------------------------

struct CellSlab {
  static constexpr std::size_t CAPACITY_LOG2 = 12;
  static constexpr std::size_t CAPACITY = 1u << CAPACITY_LOG2;  // 4096 cells
  static constexpr std::size_t CAPACITY_MASK = CAPACITY - 1;
  static constexpr std::size_t ALIGN = 32;  // AVX
  static constexpr std::size_t NF = adaptive_cell::NFACES;

  double* u = nullptr;                  // [CAPACITY]
  double* L = nullptr;                  // [CAPACITY]
  double* lam = nullptr;                // [CAPACITY]
  std::uint8_t* fixed_u = nullptr;      // [CAPACITY]
  std::uint16_t* fixed_phi = nullptr;   // [CAPACITY]
  double* phi = nullptr;                // [NF * CAPACITY]
  std::int32_t* nb = nullptr;           // [NF * CAPACITY]
  std::int32_t* parent = nullptr;       // [CAPACITY]
  std::int32_t* first_child = nullptr;  // [CAPACITY]
  std::uint8_t* n_children = nullptr;   // [CAPACITY]
  double* pos = nullptr;                // [3 * CAPACITY]

  CellSlab() { _init(); }
  ~CellSlab() { std::free(slab_mem_); }

  CellSlab(CellSlab const&) = delete;
  CellSlab& operator=(CellSlab const&) = delete;
  CellSlab(CellSlab&& o) noexcept { _steal(o); }
  CellSlab& operator=(CellSlab&& o) noexcept {
    if (this != &o) {
      std::free(slab_mem_);
      _steal(o);
    }
    return *this;
  }

 private:
  void* slab_mem_ = nullptr;

  static std::size_t _bytes() noexcept {
    auto a = [](std::size_t b) { return (b + ALIGN - 1) & ~(ALIGN - 1); };
    constexpr std::size_t C = CAPACITY;
    return a(C * 8)         // u
           + a(C * 8)       // L
           + a(C * 8)       // lam
           + a(C * 1)       // fixed_u
           + a(C * 2)       // fixed_phi
           + a(NF * C * 8)  // phi
           + a(NF * C * 4)  // nb
           + a(C * 4)       // parent
           + a(C * 4)       // first_child
           + a(C * 1)       // n_children
           + a(3 * C * 8);  // pos
  }

  void _init() {
    // EQ_SCOPE_TIMER("equilibrium::dst::discrete::CellSlab::init"); // noisey

    slab_mem_ = std::aligned_alloc(ALIGN, _bytes());
    auto* p = static_cast<std::uint8_t*>(slab_mem_);
    auto take = [&]<typename T>(std::size_t n) -> T* {
      auto* ptr = reinterpret_cast<T*>(p);
      p += (n * sizeof(T) + ALIGN - 1) & ~(ALIGN - 1);
      return ptr;
    };
    constexpr std::size_t C = CAPACITY;
    u = take.template operator()<double>(C);
    L = take.template operator()<double>(C);
    lam = take.template operator()<double>(C);
    fixed_u = take.template operator()<std::uint8_t>(C);
    fixed_phi = take.template operator()<std::uint16_t>(C);
    phi = take.template operator()<double>(NF * C);
    nb = take.template operator()<std::int32_t>(NF * C);
    parent = take.template operator()<std::int32_t>(C);
    first_child = take.template operator()<std::int32_t>(C);
    n_children = take.template operator()<std::uint8_t>(C);
    pos = take.template operator()<double>(3 * C);
    _defaults(0, C);
  }

  void _defaults(std::size_t from, std::size_t to) {
    constexpr std::size_t C = CAPACITY;
    for (std::size_t i = from; i < to; ++i) {
      u[i] = 0.0;
      L[i] = 1.0;
      lam[i] = 0.0;
      fixed_u[i] = 0;
      fixed_phi[i] = 0;
      parent[i] = -2;  // unallocated sentinel; caller sets -1 (root) or parent_g
      first_child[i] = -1;
      n_children[i] = 0;
    }
    for (std::size_t k = 0; k < NF; ++k) {
      for (std::size_t i = from; i < to; ++i) {
        phi[k * C + i] = 0.0;
        nb[k * C + i] = -1;
      }
    }
    for (std::size_t d = 0; d < 3; ++d) {
      for (std::size_t i = from; i < to; ++i) {
        pos[d * C + i] = 0.0;
      }
    }
  }

  void _steal(CellSlab& o) noexcept {
    slab_mem_ = o.slab_mem_;
    o.slab_mem_ = nullptr;
    u = o.u;
    o.u = nullptr;
    L = o.L;
    o.L = nullptr;
    lam = o.lam;
    o.lam = nullptr;
    fixed_u = o.fixed_u;
    o.fixed_u = nullptr;
    fixed_phi = o.fixed_phi;
    o.fixed_phi = nullptr;
    phi = o.phi;
    o.phi = nullptr;
    nb = o.nb;
    o.nb = nullptr;
    parent = o.parent;
    o.parent = nullptr;
    first_child = o.first_child;
    o.first_child = nullptr;
    n_children = o.n_children;
    o.n_children = nullptr;
    pos = o.pos;
    o.pos = nullptr;
  }
};

// ---------------------------------------------------------------------------
// AdaptiveCellCluster — chain of CellSlabs with per-refinement-level pools.
//
// Global index encoding (unchanged):
//   slab  = global >> CellSlab::CAPACITY_LOG2
//   slot  = global &  CellSlab::CAPACITY_MASK
//
// Level pools:
//   Each slab belongs to exactly one refinement level, tracked via slab_level_[s].
//   level_slabs_[L] lists all slab indices belonging to level L.
//   Each level has its own freelists (freelist_1/8/14) and bump pointer.
//
//   alloc_cell(level)       — allocate from level L's pool
//   alloc_children(n, level)— allocate n contiguous slots from level L's pool
//   split() targets level parent.level + 1 for children.
//   accumulate_all() iterates level_slabs_ from max level down (no heuristic).
// ---------------------------------------------------------------------------
struct AdaptiveCellCluster {
  static constexpr std::size_t CAP = CellSlab::CAPACITY;
  static constexpr std::size_t LOG2 = CellSlab::CAPACITY_LOG2;
  static constexpr std::size_t MASK = CellSlab::CAPACITY_MASK;

  AdaptiveCellCluster() = default;
  AdaptiveCellCluster(AdaptiveCellCluster const&) = delete;
  AdaptiveCellCluster& operator=(AdaptiveCellCluster const&) = delete;
  AdaptiveCellCluster(AdaptiveCellCluster&&) = default;
  AdaptiveCellCluster& operator=(AdaptiveCellCluster&&) = default;

  [[nodiscard]] static std::size_t slab_of(std::int32_t g) noexcept {
    return static_cast<std::size_t>(g) >> LOG2;
  }
  [[nodiscard]] static std::size_t slot_of(std::int32_t g) noexcept {
    return static_cast<std::size_t>(g) & MASK;
  }
  [[nodiscard]] static std::int32_t global(std::size_t s, std::size_t slot) noexcept {
    return static_cast<std::int32_t>((s << LOG2) | slot);
  }

  DSTInfo info;

  // Compact leaf index list; rebuilt after any split/coarsen.
  std::vector<std::int32_t> leaf_indices;

  // Inverse map: g_to_j[g] = position of global index g in leaf_indices,
  // -1 for non-leaves.  g_max = max global leaf index (-1 when no leaves).
  // Rebuilt with leaf_indices; valid until the next split/coarsen/defragment.
  std::vector<std::int32_t> g_to_j;
  std::int32_t g_max = -1;

  // Step scratch buffers — dense over leaf_indices.
  // temp_du[j]        = du   for leaf_indices[j]
  // temp_phi[k*L + j] = dphi on face k for leaf_indices[j]  (face-major)
  std::vector<double> temp_du;
  std::vector<double> temp_phi;

  // step_cross-only scratch: mirror corrections computed from the frozen
  // post-local temp_phi (see AdaptiveLattice::_step_cross_mirror's doc
  // comment) accumulate here instead of directly into temp_phi, so that
  // reading another leaf's already-aggregated local value never races
  // against a write into that same slot. Merged into temp_phi with one
  // elementwise add before _step_cross_apply.
  std::vector<double> temp_phi_mirror;

  // face_multiplicity[k*L + j] = divisor applied to leaf j's face-k dphi so
  // that its OWN duplicate faces sharing the same nb[] target (>= 1; 1 means
  // unique/no duplication) collectively reduce to the single term the
  // external target's own reverse face contributes. A real ratio (Σ dup
  // face_weight[k]² / target's own face_weight²), not a flat count — see
  // rebuild_face_multiplicity's comment. Only computed for nb.L <= cell.L;
  // left at 1 (no-op) when nb.L > cell.L, since ancestor_group_multiplicity
  // below already covers that case over the WHOLE sharing group.
  // Rebuilt by AdaptiveLattice::rebuild_leaf_indices whenever n_non_leaf_cells
  // > 0 (a uniform mesh has no coarse-proxy aliasing, so multiplicity is
  // always 1 there — left unallocated/empty in that case).
  //
  // Exists because Odd27's coarse-proxy approximation (an externally-facing
  // child's face proxies through par.nb[j] for whichever parent face j its
  // nearest-centre assignment resolves to — see bcc_defs.hpp's Odd27Table)
  // can send MULTIPLE of a single child's own 14 faces to the exact same
  // external target: confirmed directly (2026-07-19) that 74% of children in
  // a uniform single-level refinement have at least one duplicated target,
  // always with multiplicity exactly 4 when it occurs (3 hex + 1 sq face,
  // all resolving to the same neighbour — a child near the parent's outer
  // edge has that one neighbour dominate a whole octant of its own angular
  // space). Without correction, each of those faces independently computes
  // a dphi proportional to its OWN face_weight (same nb/u_src, but hex/sq
  // weights differ — 3 hex + 1 sq is NOT four identical contributions), so a
  // child's own u update — which sums W[k]*phi[k] over ALL its faces in
  // pass1 — gets pulled toward that one neighbour more than once. This is
  // upstream of and distinct from the AMR reflux mechanism (which only
  // balances the aggregate across a coarse-fine junction group, and cannot
  // see or fix an individual child's own inflated pull) — dividing each
  // duplicated face's dphi by this ratio at the point it's computed removes
  // the inflation at its source.
  MultiplicityTable face_multiplicity;

  // ancestor_group_multiplicity[k*L + j] = the divisor applied to leaf j's
  // face-k dphi so that the WHOLE group of leaves sharing the same
  // (ancestor, external neighbour) pair — that j's own face k resolves to
  // via the coarse-proxy ancestor walk in _step_pass2_lev/_step_pass2 —
  // collectively reduces to the single term the neighbour's own side
  // contributes (>= 1; 1 means no sharing, i.e. an exact match already).
  //
  // A real ratio, not a leaf count: Σ face_weight[k]² over every entry in the
  // group, divided by the neighbour's own face_weight² for the face pointing
  // back at the ancestor. A flat entry count is only correct when every
  // entry in the group shares one face class (hex, weight 3, vs sq, weight
  // 1); Odd27's coarse-proxy routing frequently mixes both classes within one
  // group, so a plain count under/over-corrects Σu conservation whenever a
  // group is mixed — see rebuild_ancestor_group_multiplicity's comment.
  //
  // Distinct from face_multiplicity above: that one corrects a SINGLE leaf's
  // own multiple faces aliasing to the same target. This corrects MULTIPLE
  // DIFFERENT leaves (siblings/cousins under a shared ancestor) each treating
  // one shared (ancestor, neighbour) flux as their own private exchange —
  // found 2026-07-19 investigating a deep-telescope (multi-level-gap) AMR
  // runaway in refined_neutron_decay.py: when a fine leaf's neighbour is more
  // than one level coarser, _step_pass2_lev substitutes u_src = the nearest
  // ancestor whose L >= the neighbour's L (since that ancestor is volume-
  // averaged bottom-up every substep via accumulate_all, and pairing it with
  // the correspondingly larger face distance L_src is the dimensionally
  // correct way to compare across a multi-level gap) — but every boundary-
  // facing descendant under that same ancestor independently applies that
  // SAME ancestor-derived dphi to its own u update, over-representing the
  // single physical (ancestor, neighbour) connection by however many
  // descendants share it. Dividing by this ratio at the point dphi is
  // computed removes that over-representation, the same way face_multiplicity
  // removes the single-leaf duplicate-face inflation.
  MultiplicityTable ancestor_group_multiplicity;

  // Refinement criterion — updated at end of each step.
  // criterion_min[j] = min |phi[k]| over active faces of leaf j
  // criterion_max[j] = max |phi[k]| over active faces of leaf j
  // Refine if criterion_max > high_threshold (peak flux large → gradient present).
  // Coarsen if criterion_max < low_threshold (all faces quiet → cell can grow).
  std::vector<double> criterion_min;
  std::vector<double> criterion_max;

  // Ghost + send infrastructure (MPI prep).
  std::vector<RemoteCellRef> ghost_map;
  std::vector<ClusterSendBatch> send_schedule;

  // Boundary condition dispatch.  nullptr = open BC everywhere (fast path).
  std::unique_ptr<BoundaryCondition> bc;

  // Split/coarsen criterion functor.  nullptr = fall back to DSTInfo.split_hi/lo.
  std::unique_ptr<SplitCriterion> split_criterion;

  // Number of adapt() calls seen so far — drives DSTInfo.split_interval cadence.
  // Lives on the cluster (not DSTInfo) so reassigning cl.info mid-run doesn't reset it.
  std::uint64_t adapt_calls = 0;

  // Positions (indices into leaf_indices) of leaves that have at least one
  // face with no local neighbour (nb[k] < 0).  Rebuilt by rebuild_leaf_indices.
  // Only these cells enter the BC dispatch pass; all others take the fast path.
  std::vector<std::size_t> boundary_leaf_positions;

  // Count of active cells that have children (non-leaves). Set by
  // rebuild_leaf_indices. Used to skip _accumulate_boundary_proxies
  // fast-path when the lattice is uniform (all cells at the same level).
  std::size_t n_non_leaf_cells = 0;

  // Pre-computed AMR flux correction pairs. Rebuilt by AdaptiveLattice::rebuild_leaf_indices.
  // Empty when n_non_leaf_cells == 0.
  // Each pair balances temp_phi across one coarse-fine or proxy-proxy junction:
  //   side A: entries in [A_begin, A_end) of (amr_js, amr_js_face) — leaf j index + that
  //           leaf's OWN face index connecting to this junction (per-entry, not shared).
  //   side B: entries in [B_begin, B_end), same (leaf j, face) pairing.
  // amr_js[i]/amr_js_face[i] together give: leaf j = amr_js[i], face = amr_js_face[i].
  // amr_js[begin..mid) / amr_js_face[begin..mid) = A-side
  // amr_js[mid..end)   / amr_js_face[mid..end)   = B-side
  // a_only=true  (coarse-fine): error shared across A side only; B is read-only reference.
  // a_only=false (proxy-proxy): error shared across all (end-begin) cells.
  //
  // Per-entry face indices (rather than one shared face_k/mirror_k per side)
  // are REQUIRED for correctness under Odd27: a descendant's own face k, when
  // external, does not always proxy through the SAME index k on the far side
  // (true only by construction for the legacy ratio-2 octant splits) — a
  // single child can even have more than one of its own faces routing to the
  // same external target (up to 4-way, see face_multiplicity's doc comment).
  // A single-shared-face-per-side version of this struct was tried once
  // (2026-07-19) and reverted — see [[project_odd27_amr_junction_leak]] — but
  // that was BEFORE face_multiplicity/ancestor_group_multiplicity existed to
  // normalize a child's per-face contribution by how many of its faces route
  // to the same target; with those fixes in place (confirmed present, see
  // rebuild_face_multiplicity/rebuild_ancestor_group_multiplicity), this
  // per-entry version is the one that should be correct, verified via the
  // exact check_total_dphi_balanced invariant (not the noisy H-growth
  // comparison the earlier attempt was judged against).
  struct AmrCorrectionPair {
    bool a_only;
    std::int32_t begin, mid, end;
  };
  std::vector<AmrCorrectionPair> amr_correction_pairs;
  std::vector<std::int32_t> amr_js;
  std::vector<std::uint8_t> amr_js_face;

  // All-leaf sibling groups: each entry is the first leaf-j index j0 of a group
  // of 8 consecutive leaves that are the children of a single parent.
  // Their global indices are first_child..first_child+7 (contiguous in the slab);
  // their leaf_indices positions are j0..j0+7 (contiguous by rebuild order).
  // Used by _step_pass2_siblings for vectorised intra-group sq-face updates.
  struct SiblingGroup {
    std::size_t j0;            // first position in leaf_indices
    std::int32_t first_child;  // global index of child 0
  };
  std::vector<SiblingGroup> sibling_groups;

  // --- Allocation ---

  // ---------------------------------------------------------------------------
  // Spatial zone ID — Morton-encoded quantised position for zone-aware slab
  // allocation.  Cells with the same zone_id at a given level are allocated
  // into the same slab(s), keeping geometric neighbours cache-local.
  //
  // Zone width (ZONE_WIDTH) is in physical units (d1 = 1).  All 14 BCC
  // neighbours lie within radius ~1.16; ZONE_WIDTH >> 2.32 guarantees
  // all neighbours of any interior cell share the same zone_id.
  // ---------------------------------------------------------------------------
  static constexpr double ZONE_WIDTH = 32.0;
  static constexpr uint32_t ZONE_OFFSET = 512u;  // handles coords down to −512×ZONE_WIDTH

  [[nodiscard]] static uint32_t zone_id(double x, double y, double z) noexcept {
    auto qi = [](double v) -> uint32_t {
      return static_cast<uint32_t>(static_cast<int>(v / ZONE_WIDTH) + ZONE_OFFSET);
    };
    return _morton3(qi(x) & 0x3FFu, qi(y) & 0x3FFu, qi(z) & 0x3FFu);
  }

  // Allocate one cell at refinement level `lv`, routed to spatial zone `zid`.
  [[nodiscard]] std::int32_t alloc_cell(std::uint8_t lv = 0, uint32_t zid = 0) {
    _ensure_level(lv);
    auto& fl = level_freelist_1_[lv];
    if (!fl.empty()) {
      auto g = fl.back();
      fl.pop_back();
      ++slab_live_[slab_of(g)];
      return g;
    }
    return _bump(1, lv, zid);
  }

  // Select the freelist bucket matching a given children-block size.
  [[nodiscard]] std::vector<std::int32_t>& _freelist_for(std::size_t n, std::uint8_t lv) {
    assert(n == adaptive_cell::N_CHILDREN_HEX || n == adaptive_cell::N_CHILDREN_HEX_CENTER ||
           n == adaptive_cell::N_CHILDREN_ODD27 || n == adaptive_cell::N_CHILDREN_FULL);
    if (n == adaptive_cell::N_CHILDREN_HEX) return level_freelist_8_[lv];
    if (n == adaptive_cell::N_CHILDREN_HEX_CENTER) return level_freelist_9_[lv];
    if (n == adaptive_cell::N_CHILDREN_ODD27) return level_freelist_27_[lv];
    return level_freelist_14_[lv];
  }

  // Allocate n contiguous slots at refinement level `lv`, routed to zone `zid`.
  [[nodiscard]] std::int32_t alloc_children(std::size_t n, std::uint8_t lv, uint32_t zid = 0) {
    _ensure_level(lv);
    auto& fl = _freelist_for(n, lv);
    if (!fl.empty()) {
      auto g = fl.back();
      fl.pop_back();
      slab_live_[slab_of(g)] += static_cast<std::uint32_t>(n);
      return g;
    }
    return _bump(n, lv, zid);
  }

  // Free a single cell.
  void free_cell(std::int32_t g) {
    slabs_[slab_of(g)].parent[slot_of(g)] = -2;
    std::uint8_t lv = slab_level_[slab_of(g)];
    level_freelist_1_[lv].push_back(g);
    _dec_live(slab_of(g));
  }

  // Free a contiguous block of n children starting at fc.
  void free_children(std::int32_t fc, std::uint8_t n) {
    for (std::uint8_t c = 0; c < n; ++c) slabs_[slab_of(fc + c)].parent[slot_of(fc + c)] = -2;
    std::uint8_t lv = slab_level_[slab_of(fc)];
    _freelist_for(n, lv).push_back(fc);
    _dec_live(slab_of(fc), n);
  }

  // Collect all global indices in the subtree rooted at `root` (breadth-first).
  std::vector<std::int32_t> collect_subtree(std::int32_t root) {
    std::vector<std::int32_t> out;
    out.push_back(root);
    for (std::size_t i = 0; i < out.size(); ++i) {
      auto& s = slabs_[slab_of(out[i])];
      std::size_t slot = slot_of(out[i]);
      std::int32_t fc = s.first_child[slot];
      std::uint8_t nc = s.n_children[slot];
      for (std::uint8_t c = 0; c < nc; ++c) out.push_back(fc + c);
    }
    return out;
  }

  // Move a subtree into `dest`, using bidirectionality to patch all nb[] pointers.
  // mirror[k] = opposing face index on neighbour k (BCC geometry constant).
  void migrate_subtree(std::int32_t root, AdaptiveCellCluster& dest,
                       std::array<std::uint8_t, CellSlab::NF> const& mirror) {
    auto old_indices = collect_subtree(root);

    std::unordered_map<std::int32_t, std::int32_t> remap;
    remap.reserve(old_indices.size());
    for (std::int32_t old_g : old_indices) {
      auto const nc = slabs_[slab_of(old_g)].n_children[slot_of(old_g)];
      auto const lv = slab_level_[slab_of(old_g)];
      auto const new_g = (nc == 0) ? dest.alloc_cell(lv) : dest.alloc_children(nc, lv);
      remap[old_g] = new_g;
      if (nc > 0) {
        std::int32_t old_fc = slabs_[slab_of(old_g)].first_child[slot_of(old_g)];
        for (std::uint8_t c = 1; c < nc; ++c) remap[old_fc + c] = new_g + c;
      }
    }

    for (std::int32_t old_g : old_indices) {
      auto src = (*this)[old_g];
      auto dst = dest[remap.at(old_g)];
      dst.u = src.u;
      dst.L = src.L;
      dst.lam = src.lam;
      dst.fixed_u = src.fixed_u;
      dst.fixed_phi = src.fixed_phi;
      dst.n_children = src.n_children;
      auto remap_if = [&](std::int32_t g) {
        auto it = remap.find(g);
        return it != remap.end() ? it->second : g;
      };
      dst.parent = remap_if(src.parent);
      dst.first_child = (src.n_children > 0) ? remap.at(src.first_child) : -1;
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        dst.phi[k] = src.phi[k];
        dst.nb[k] = remap_if(src.nb[k]);
      }
    }

    for (std::int32_t old_g : old_indices) {
      std::int32_t new_g = remap.at(old_g);
      auto& src_slab = slabs_[slab_of(old_g)];
      std::size_t src_slot = slot_of(old_g);
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        std::int32_t outside = src_slab.nb[k * CAP + src_slot];
        if (outside < 0 || remap.count(outside)) continue;
        auto& nb_slab = slabs_[slab_of(outside)];
        nb_slab.nb[mirror[k] * CAP + slot_of(outside)] = new_g;
      }
    }

    for (std::int32_t old_g : old_indices) free_cell(old_g);
  }

  // Move one cell to a new slot (used by defragment).
  // Patches all nb[] back-references using bidirectionality.
  std::int32_t migrate_cell(std::int32_t old_g,
                            std::array<std::uint8_t, CellSlab::NF> const& mirror) {
    auto const lv = slab_level_[slab_of(old_g)];
    auto const new_g = alloc_cell(lv);
    auto src = (*this)[old_g];
    auto dst = (*this)[new_g];
    dst.u = src.u;
    dst.L = src.L;
    dst.lam = src.lam;
    dst.fixed_u = src.fixed_u;
    dst.fixed_phi = src.fixed_phi;
    dst.parent = src.parent;
    dst.first_child = src.first_child;
    dst.n_children = src.n_children;
    for (std::size_t k = 0; k < CellSlab::NF; ++k) {
      dst.phi[k] = src.phi[k];
      dst.nb[k] = src.nb[k];
    }
    for (std::size_t k = 0; k < CellSlab::NF; ++k) {
      std::int32_t nbg = src.nb[k];
      if (nbg < 0) continue;
      (*this)[nbg].nb[mirror[k]] = new_g;
    }
    if (src.parent >= 0) {
      auto par = (*this)[src.parent];
      std::int32_t fc = par.first_child;
      std::uint8_t nc = par.n_children;
      for (std::uint8_t c = 0; c < nc; ++c) {
        if (fc + c == old_g) {
          if (c == 0) par.first_child = new_g;
          break;
        }
      }
    }
    if (src.n_children > 0) {
      for (std::uint8_t c = 0; c < src.n_children; ++c) (*this)[src.first_child + c].parent = new_g;
    }
    free_cell(old_g);
    return new_g;
  }

  // --- Accessors ---

  [[nodiscard]] AdaptiveCellRef operator[](std::int32_t g) noexcept {
    auto const si = slab_of(g);
    auto& s = slabs_[si];
    auto const i = slot_of(g);
    return AdaptiveCellRef{
        .u = s.u[i],
        .L = s.L[i],
        .lam = s.lam[i],
        .fixed_u = s.fixed_u[i],
        .fixed_phi = s.fixed_phi[i],
        .phi = {s.phi + i, CAP},
        .nb = {s.nb + i, CAP},
        .parent = s.parent[i],
        .first_child = s.first_child[i],
        .n_children = s.n_children[i],
        .level = slab_level(si),
    };
  }

  [[nodiscard]] CellSlab& slab(std::size_t s) noexcept { return slabs_[s]; }
  [[nodiscard]] CellSlab const& slab(std::size_t s) const noexcept { return slabs_[s]; }

  [[nodiscard]] std::size_t n_slabs() const noexcept { return slabs_.size(); }
  [[nodiscard]] std::size_t n_levels() const noexcept { return level_slabs_.size(); }

  // Slab indices for level L — for level-ordered traversal in accumulate_all.
  [[nodiscard]] std::vector<std::size_t> const& level_slabs(std::uint8_t lv) const noexcept {
    return level_slabs_[lv];
  }

  [[nodiscard]] double slab_occupancy(std::size_t s) const noexcept {
    return s < slab_live_.size() ? static_cast<double>(slab_live_[s]) / CAP : 0.0;
  }

  [[nodiscard]] std::uint8_t slab_level(std::size_t s) const noexcept {
    return s < slab_level_.size() ? slab_level_[s] : 0;
  }

  // ---------------------------------------------------------------------------
  // restore() — reconstruct a cluster from flat slab arrays (HDF5 checkpoint).
  //
  // Flat layout: field[s * C + slot]  for scalar fields,
  //              field[k * NS * C + s * C + slot]  for face/coord fields.
  //
  // All cell arrays must have length n_slabs * CAPACITY.
  // phi / nb : length NF * n_slabs * CAPACITY.
  // pos      : length 3  * n_slabs * CAPACITY.
  //
  // Streaming variant: the caller (load_cluster) reads each slab's fields
  // directly off disk into the slab's own storage — no O(n_slabs*CAPACITY)
  // intermediate buffer. Usage per slab: restore_add_slab(lv) to get a
  // writable CellSlab&, fill every field (parent LAST, since
  // restore_finalize_slab scans it), then restore_finalize_slab(s). Once
  // every slab is done, restore_finalize(leaf_idx, info) installs the leaf
  // list and sizes the step scratch buffers.
  //
  // After restore, new allocations create fresh slabs without touching the
  // restored cells (the zone-aware bump allocator only reuses empty slabs
  // registered via level_empty_slabs_, which restore_finalize_slab does for
  // any slab that turns out to be fully dead). Dead slots (parent == -2) are
  // pushed to level_freelist_1_ so they can be recycled.
  // ---------------------------------------------------------------------------
  [[nodiscard]] CellSlab& restore_add_slab(std::uint8_t lv) {
    _ensure_level(lv);
    slabs_.emplace_back();
    std::size_t s = slabs_.size() - 1;
    slab_live_.push_back(0);
    slab_level_.push_back(lv);
    level_slabs_[lv].push_back(s);
    return slabs_[s];
  }

  void restore_finalize_slab(std::size_t s) {
    std::uint8_t lv = slab_level_[s];
    for (std::size_t i = 0; i < CAP; ++i) {
      if (slabs_[s].parent[i] == -2)
        level_freelist_1_[lv].push_back(global(s, i));
      else
        ++slab_live_[s];
    }
    if (slab_live_[s] == 0) level_empty_slabs_[lv].push_back(s);
  }

  void restore_finalize(std::vector<std::int32_t> leaf_idx, DSTInfo info_in) {
    leaf_indices = std::move(leaf_idx);
    temp_du.assign(leaf_indices.size(), 0.0);
    temp_phi.assign(CellSlab::NF * leaf_indices.size(), 0.0);
    temp_phi_mirror.assign(CellSlab::NF * leaf_indices.size(), 0.0);
    criterion_min.assign(leaf_indices.size(), 0.0);
    criterion_max.assign(leaf_indices.size(), 0.0);
    // face_multiplicity/ancestor_group_multiplicity default-construct to
    // all_one=true (empty backing storage) already — recomputed for real by
    // the next rebuild_leaf_indices.
    info = std::move(info_in);
  }

 private:
  std::vector<CellSlab> slabs_;
  std::vector<std::uint32_t> slab_live_;
  std::vector<std::uint8_t> slab_level_;

  // Per-level: which slabs belong to level L, empty-slab reuse pool, and freelists.
  std::vector<std::vector<std::size_t>> level_slabs_;
  std::vector<std::vector<std::size_t>> level_empty_slabs_;
  std::vector<std::vector<std::int32_t>> level_freelist_1_;
  std::vector<std::vector<std::int32_t>> level_freelist_8_;
  std::vector<std::vector<std::int32_t>> level_freelist_9_;
  std::vector<std::vector<std::int32_t>> level_freelist_27_;
  std::vector<std::vector<std::int32_t>> level_freelist_14_;
  // Per-zone bump state: zone_id → active slab index / next slot.
  // Replaces the single level_next_slot_ with per-zone tracking so cells
  // sharing a spatial zone are packed into the same slab(s).
  std::vector<std::unordered_map<uint32_t, std::size_t>> level_zone_slab_;
  std::vector<std::unordered_map<uint32_t, std::size_t>> level_zone_slot_;

  void _ensure_level(std::uint8_t lv) {
    if (lv < level_slabs_.size()) return;
    std::size_t needed = static_cast<std::size_t>(lv) + 1;
    level_slabs_.resize(needed);
    level_empty_slabs_.resize(needed);
    level_freelist_1_.resize(needed);
    level_freelist_8_.resize(needed);
    level_freelist_9_.resize(needed);
    level_freelist_27_.resize(needed);
    level_freelist_14_.resize(needed);
    level_zone_slab_.resize(needed);
    level_zone_slot_.resize(needed);
  }

  // Allocate a fresh slab for level lv, register it, return its index.
  std::size_t _new_slab(std::uint8_t lv) {
    auto& empty = level_empty_slabs_[lv];
    if (!empty.empty()) {
      std::size_t s = empty.back();
      empty.pop_back();
      return s;
    }
    slabs_.emplace_back();
    std::size_t s = slabs_.size() - 1;
    slab_live_.push_back(0);
    slab_level_.push_back(lv);
    level_slabs_[lv].push_back(s);
    return s;
  }

  // Zone-aware bump allocator.  Cells with the same (lv, zid) are packed into
  // the same slab(s), so geometric neighbours are cache-local during stepping.
  [[nodiscard]] std::int32_t _bump(std::size_t n, std::uint8_t lv, uint32_t zid) {
    auto& zslab = level_zone_slab_[lv];
    auto& zslot = level_zone_slot_[lv];

    auto sit = zslab.find(zid);
    auto nit = zslot.find(zid);
    bool need_new = (sit == zslab.end()) || (nit->second + n > CAP);

    if (need_new) {
      std::size_t s = _new_slab(lv);
      zslab[zid] = s;
      zslot[zid] = 0;
      sit = zslab.find(zid);
      nit = zslot.find(zid);
    }

    std::size_t s = sit->second;
    std::size_t slot = nit->second;
    nit->second += n;
    slab_live_[s] += static_cast<std::uint32_t>(n);
    return global(s, slot);
  }

  // Morton bit-spread and encode — maps (x,y,z) each 10-bit to a 30-bit code.
  [[nodiscard]] static uint32_t _spread(uint32_t v) noexcept {
    v &= 0x000003ffu;
    v = (v | (v << 16u)) & 0x030000ffu;
    v = (v | (v << 8u)) & 0x0300f00fu;
    v = (v | (v << 4u)) & 0x030c30c3u;
    v = (v | (v << 2u)) & 0x09249249u;
    return v;
  }
  [[nodiscard]] static uint32_t _morton3(uint32_t x, uint32_t y, uint32_t z) noexcept {
    return _spread(x) | (_spread(y) << 1u) | (_spread(z) << 2u);
  }

  void _dec_live(std::size_t s, std::size_t n = 1) {
    slab_live_[s] -= static_cast<std::uint32_t>(n);
    if (slab_live_[s] == 0) {
      std::uint8_t lv = slab_level_[s];
      level_empty_slabs_[lv].push_back(s);
    }
  }
};

}  // namespace equilibrium::dst::discrete

#endif  // _EQUILIBRIUM_CORE_DATA_LATTICE_CLUSTER_HPP_
