#ifndef _EQUILIBRIUM_CORE_DATA_LATTICE_ADAPTIVE_LATTICE_HPP_
#define _EQUILIBRIUM_CORE_DATA_LATTICE_ADAPTIVE_LATTICE_HPP_

#include "eq/core/phlop.hpp"
#include "eq/core/debug.hpp"
#include "eq/core/mca.hpp"
#include "eq/core/data/lattice/bcc_defs.hpp"
#include "eq/core/data/lattice/cluster.hpp"
#include "eq/core/data/lattice/time_stepper.hpp"
#include "eq/core/data/lattice/debug/debug_adaptive_lattice.hpp"
#include "BS_thread_pool.hpp"
#include "eq/core/data/lattice/bc.hpp"
#include "eq/core/data/lattice/cell/adaptive_cell.hpp"
#include "eq/core/data/lattice/cell/discrete_cell.hpp"
#include "eq/core/data/lattice/dst_info.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <utility>
#include <variant>

namespace equilibrium::dst::discrete {

// ---------------------------------------------------------------------------
// BCC Kelvin geometry constants
//
// Face ordering (fixed, matches the Python SubstrateLattice convention):
//   Faces 0-7  : hex faces    (through truncated-octahedron hexagonal faces)
//   Faces 8-13 : square faces (through square faces)
//
// mirror_face[k] = face index on the neighbour that points back to this cell.
// child_face_mask[k] = bitmask of which children (0..7) lie on face k's side.
//   Used during split to wire intra-children nb[] for the face-k direction.
// ---------------------------------------------------------------------------
namespace bcc {

// Each face k pairs with mirror_face[k] on the neighbour.
// Hex face ordering (from kelvin.py): sx,sy,sz ∈ {-1,+1}³ in lex order:
//   0=(-,-,-) 1=(-,-,+) 2=(-,+,-) 3=(-,+,+) 4=(+,-,-) 5=(+,-,+) 6=(+,+,-) 7=(+,+,+)
// Face k mirrors to the face with all signs flipped: 0↔7, 1↔6, 2↔5, 3↔4.
constexpr std::array<std::uint8_t, 14> mirror_face = {
    7, 6, 5,  4,  3,  2, 1, 0,  // hex pairs  (0↔7, 1↔6, 2↔5, 3↔4)
    9, 8, 11, 10, 13, 12        // sq  pairs  (8↔9, 10↔11, 12↔13)
};

// Number of children sharing each face of the parent in a hex-split (8 children).
// Split model: child c placed at octant hex_dir[c] (1 child per hex face).
// Hex faces (0-7): 1 child per face — the unique child in that octant.
// Square faces (8-13): 4 children per face — the 4 children with matching sq-axis sign.
constexpr std::array<std::uint8_t, 14> children_per_face = {
    1, 1, 1, 1, 1, 1, 1, 1,  // hex: 1 child per face (child c == face c)
    4, 4, 4, 4, 4, 4         // sq:  4 children per face (half the octants per axis)
};

// Face weights. The Kelvin geometry fixes the Hodge star *1 = |dual face| /
// |link| = 3:1 (hex:sq): the weight of a complete operator (a Laplacian, a
// diffusion) -- face_star. step() applies W on both legs (du = -sigma sum W
// phi, dphi = -sigma W du), so its per-leg weight is W = sqrt(star), W_HEX =
// sqrt3, and phi-divergence readers (lam, step_cross drive) use W too.
// Before 2026-09-28 W_HEX was 3 on both legs (star 9:1). exp/2026/09/28/hodge_star.
constexpr double W_HEX = 1.7320508075688772;  // sqrt3
constexpr double W_SQ = 1.0;
constexpr std::array<double, 14> face_weight = {
    W_HEX, W_HEX, W_HEX, W_HEX, W_HEX, W_HEX, W_HEX, W_HEX, W_SQ, W_SQ, W_SQ, W_SQ, W_SQ, W_SQ,
};
constexpr std::array<double, 14> face_star = {3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1};  // W^2

// Unit outward vectors for all 14 faces (matches Python FACE_DIRS ordering).
// Hex 0-7: (±1,±1,±1)/√3 in lex sign order; sq 8-13: axis-aligned ±x/±y/±z.
constexpr double R3I = 0.5773502691896258;  // 1/√3
constexpr std::array<std::array<double, 3>, 14> face_dirs = {{
    {-R3I, -R3I, -R3I}, {-R3I, -R3I, +R3I}, {-R3I, +R3I, -R3I}, {-R3I, +R3I, +R3I},
    {+R3I, -R3I, -R3I}, {+R3I, -R3I, +R3I}, {+R3I, +R3I, -R3I}, {+R3I, +R3I, +R3I},
    {-1.0,  0.0,  0.0}, {+1.0,  0.0,  0.0},
    { 0.0, -1.0,  0.0}, { 0.0, +1.0,  0.0},
    { 0.0,  0.0, -1.0}, { 0.0,  0.0, +1.0},
}};

// Long-wave speed of step() in link lengths per tick at L = 1 (d1 = 1 hex, 2/sqrt3 square):
// c^2 = 1/2 sum_k W_k^2 (d_k . x)^2 (= 16/3 now, c = 2.309; numpy lattice.wave_speed).
inline double wave_speed() {
  double c2 = 0.0;
  for (std::size_t k = 0; k < 14; ++k) {
    double const len = k < 8 ? 1.0 : 2.0 * R3I;
    double const dx  = len * face_dirs[k][0];
    c2 += 0.5 * face_weight[k] * face_weight[k] * dx * dx;
  }
  return std::sqrt(c2);
}

// Face adjacency for step_cross: edge-adjacent faces on the Kelvin cell.
// face_adj[k][m] = m-th face edge-adjacent to face k; -1 = unused (sq faces only use 0-3).
// Matches Python _FACE_ADJ in equilibrium/substrate/lattice.py and FACE_ADJ in substrate.hpp.
constexpr std::array<std::array<int, 6>, 14> face_adj = {{
    {{ 3,  5,  6,  8, 10, 12}},  // 0 (-,-,-)
    {{ 2,  4,  7,  8, 10, 13}},  // 1 (-,-,+)
    {{ 1,  4,  7,  8, 11, 12}},  // 2 (-,+,-)
    {{ 0,  5,  6,  8, 11, 13}},  // 3 (-,+,+)
    {{ 1,  2,  7,  9, 10, 12}},  // 4 (+,-,-)
    {{ 0,  3,  6,  9, 10, 13}},  // 5 (+,-,+)
    {{ 0,  3,  5,  9, 11, 12}},  // 6 (+,+,-)
    {{ 1,  2,  4,  9, 11, 13}},  // 7 (+,+,+)
    {{ 0,  1,  2,  3, -1, -1}},  // 8  -x sq
    {{ 4,  5,  6,  7, -1, -1}},  // 9  +x sq
    {{ 0,  1,  4,  5, -1, -1}},  // 10 -y sq
    {{ 2,  3,  6,  7, -1, -1}},  // 11 +y sq
    {{ 0,  2,  4,  6, -1, -1}},  // 12 -z sq
    {{ 1,  3,  5,  7, -1, -1}},  // 13 +z sq
}};

}  // namespace bcc

// step_cross gate — matches Python SubstrateLattice.step_cross(gate=...) and
// eq::substrate_step_cross / eq::substrate_step_cross_barrier (substrate.hpp).
//   V1:      continuous, always-fires crossing (original behaviour).
//   Barrier: stateless, instantaneous, all-or-nothing per cell — pooled
//            per-cell drive Σ_k W_k·max(-phi[k],0) gated against m0 (no
//            default; see doc/dst_next.md sec 8 for calibration).
enum class CrossGate : std::uint8_t { V1 = 0, Barrier = 1 };

// ---------------------------------------------------------------------------
// Split result — the set of children produced by one split operation.
// ---------------------------------------------------------------------------
struct SplitResult {
  std::int32_t first_child;                        // global index of first child
  std::uint8_t n_children;                         // N_CHILDREN_HEX or N_CHILDREN_FULL
  std::array<std::int32_t, 14> child_nb_global{};  // for each parent face k: which
                                                   // child (first_child+offset) sits
                                                   // on that face's interior side
};

// ---------------------------------------------------------------------------
// AdaptiveLattice
//
// Holds one cluster per compute unit (thread / MPI rank).
// Single-machine use: clusters.size() == 1.
//
// Coarse-fine interface protocol (called every step):
//   1. step all leaf cells in all clusters
//   2. accumulate_to_parent() for all non-leaf cells (bottom-up)
//      — refreshes parent.u / parent.phi from children for coarse neighbours
//   3. (MPI) exchange ghost cells
// ---------------------------------------------------------------------------
struct AdaptiveLattice {
  std::vector<AdaptiveCellCluster> clusters;

  AdaptiveLattice() { clusters.emplace_back(); }
  AdaptiveLattice(AdaptiveLattice const&) = delete;
  AdaptiveLattice& operator=(AdaptiveLattice const&) = delete;
  AdaptiveLattice(AdaptiveLattice&&) = default;
  AdaptiveLattice& operator=(AdaptiveLattice&&) = default;

  // Set the number of worker threads.  n <= 1 disables the pool (single-thread path).
  void set_n_threads(std::size_t n) {
    if (n <= 1) {
      pool_.reset();
      return;
    }
    pool_ = std::make_unique<BS::thread_pool<>>(n);
  }

  // True if cell_g is a leaf whose level is below cluster.info.max_level.
  bool can_split(AdaptiveCellCluster& cluster, std::int32_t cell_g) {
    return cluster[cell_g].level < cluster.info.max_level;
  }

  // -----------------------------------------------------------------------
  // split()
  //
  // Split leaf cell `cell_g` in `cluster` into `Model`'s children (compile-
  // time selected via split_model_traits<Model> — n_children/ratio are
  // resolved and inlined per instantiation, no runtime dispatch). Defaults
  // to kDefaultSplitModel (Odd27, ratio-3, 27 children); pass an explicit
  // SplitModel (Hex8/HexCenter9) to use a legacy ratio-2 model instead.
  // Children inherit the parent's physics (constant prolongation).
  // Parent becomes a non-leaf interior node — coarse neighbours' nb[] are
  // unchanged; they continue to address the parent as an interface proxy.
  //
  // The caller is responsible for:
  //   - Filling child positions (geometry layer, BCC sub-lattice offsets).
  //   - Wiring intra-children nb[] (geometry layer, wire_split_children<Model>
  //     — must be called with the SAME Model).
  //   - Calling rebuild_leaf_indices() after all splits in a refinement pass.
  //
  // Returns SplitResult so the geometry layer can complete the nb[] wiring.
  // -----------------------------------------------------------------------
  template <SplitModel Model = kDefaultSplitModel>
  SplitResult split(AdaptiveCellCluster& cluster, std::int32_t cell_g) {
    constexpr std::uint8_t n_children = split_model_traits<Model>::n_children;
    constexpr double ratio = split_model_traits<Model>::ratio;

    auto parent = cluster[cell_g];
    assert(parent.is_leaf());
    if (parent.level >= cluster.info.max_level)
      throw std::logic_error("split: cell level >= max_level");

    // Allocate contiguous child slots at the next refinement level,
    // routed to the parent's spatial zone for cache locality.
    std::uint8_t child_level = static_cast<std::uint8_t>(parent.level + 1);
    auto const ps = AdaptiveCellCluster::slab_of(cell_g);
    auto const pslot = AdaptiveCellCluster::slot_of(cell_g);
    auto& pslab = cluster.slab(ps);
    auto const px = pslab.pos[0 * CellSlab::CAPACITY + pslot];
    auto const py = pslab.pos[1 * CellSlab::CAPACITY + pslot];
    auto const pz = pslab.pos[2 * CellSlab::CAPACITY + pslot];
    auto const first_child =
        cluster.alloc_children(n_children, child_level, AdaptiveCellCluster::zone_id(px, py, pz));

    // Initialise children from parent (constant prolongation).
    // L shrinks by `ratio` (2 for the legacy hex splits, 3 for Odd27); u, lam,
    // phi inherited; nb[] left as -1 for geometry layer.
    auto const child_L = parent.L / ratio;
    for (std::uint8_t c = 0; c < n_children; ++c) {
      auto ch = cluster[first_child + c];
      ch.u = parent.u;
      ch.L = child_L;
      ch.lam = parent.lam;
      ch.fixed_u = parent.fixed_u;
      ch.fixed_phi = 0;  // children start with no frozen faces
      ch.parent = cell_g;
      ch.first_child = -1;
      ch.n_children = 0;
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        ch.phi[k] = parent.phi[k];  // geometry layer scales by area ratio
        ch.nb[k] = -1;              // filled by geometry layer
      }
    }

    // Mark parent as interior proxy.
    parent.first_child = first_child;
    parent.n_children = n_children;

    EQUILIBRIUM_DEBUG_DO({  //
      check_state_after_split(cluster, cell_g, first_child, n_children, ratio);
    })

    // child_nb_global[k] = global index of the child on face-k's interior side.
    // Filled by the geometry layer; returned here as -1 placeholders.
    SplitResult result;
    result.first_child = first_child;
    result.n_children = n_children;
    result.child_nb_global.fill(-1);
    return result;
  }

  // -----------------------------------------------------------------------
  // accumulate_to_parent()
  //
  // Restrict children's u and phi up to the parent interior node.
  // Called bottom-up after each step so coarse neighbours see a consistent
  // flux at the coarse-fine interface.
  //
  // u_parent = mean(u_children) — volume average over ALL children, including
  // the center child (9-split) if present: it's a genuine leaf occupying real
  // volume, not bookkeeping.
  //
  // phi_parent[k] = mean over the children whose face k is externally-facing
  // (nb[k] not a sibling) — computed dynamically per face rather than a fixed
  // children_per_face[k]/nc-1 divisor, since which children (and how many)
  // face externally on a given face depends on the split model: e.g. hex face
  // k has 7 externally-approximating children under the legacy 8-split but
  // still only those same 7 under the 9-split (the 8th, mirror_face[k], now
  // faces the center — an internal face, excluded here exactly as it was
  // implicitly excluded before via the old nb[k]=-1 dead-face convention).
  // The center child itself never contributes to a hex-face mean (its hex
  // faces are all internal, to its 8 siblings) but does contribute to square
  // faces, where it inherits the parent's own sq connectivity.
  // -----------------------------------------------------------------------
  void accumulate_to_parent(AdaptiveCellCluster& cluster, std::int32_t parent_g) {
    EQUILIBRIUM_DEBUG_DO({  //
      check_state_priot_accumulate_to_parent_start(cluster, parent_g);
    })

    auto p = cluster[parent_g];
    assert(!p.is_leaf());

    auto const nc = p.n_children;
    auto const fc = p.first_child;

    double u_sum = 0.0;
    double lam_sum = 0.0;
    for (std::uint8_t c = 0; c < nc; ++c) {
      u_sum += cluster[fc + c].u;
      lam_sum += cluster[fc + c].lam;
    }
    p.lam = lam_sum / nc;
    p.u   = u_sum / nc;

    // Face-average phi: mean over the children externally-facing on each
    // PARENT face — grouped by WHICH of the parent's own neighbours a
    // child's external face actually connects to (matched by global index
    // against par.nb[]), NOT by the child's own face index.
    //
    // Those two coincide for the legacy octant splits (Hex8/HexCenter9): an
    // externally-facing child's own face k always proxies through par.nb[k]
    // — same index on both sides, by construction of the octant/mirror
    // tables. That assumption is FALSE in general for Odd27: a child's own
    // face k can route (via odd27_table.face_target) to a DIFFERENT parent
    // face k' (e.g. verified directly: the child at canonical offset
    // [-2,-2,0]'s own face 2 proxies through the parent's face 0, not its
    // face 2). Grouping by the child's own index would silently blend flux
    // belonging to different physical neighbours into the wrong parent
    // face. Matching by actual global neighbour index is correct for every
    // model (degenerates to the old index-matching behaviour for the
    // legacy splits, since there k_child == k_parent always holds).
    std::array<double, CellSlab::NF> phi_sum{};
    std::array<std::size_t, CellSlab::NF> n_ext{};
    for (std::uint8_t c = 0; c < nc; ++c) {
      auto ch = cluster[fc + c];
      for (std::size_t k_child = 0; k_child < CellSlab::NF; ++k_child) {
        auto const nb_ck = ch.nb[k_child];
        bool const is_sibling = nb_ck >= fc && nb_ck < fc + nc;
        if (is_sibling || nb_ck < 0) continue;
        for (std::size_t k_parent = 0; k_parent < CellSlab::NF; ++k_parent) {
          if (p.nb[k_parent] == nb_ck) {
            phi_sum[k_parent] += ch.phi[k_child];
            ++n_ext[k_parent];
            break;
          }
        }
      }
    }
    for (std::size_t k = 0; k < CellSlab::NF; ++k)
      p.phi[k] = n_ext[k] > 0 ? phi_sum[k] / static_cast<double>(n_ext[k]) : 0.0;

    EQUILIBRIUM_DEBUG_DO({  //
      check_state_priot_accumulate_to_parent_end(cluster, parent_g);
    })
  }

  // Accumulate the full tree bottom-up: finest level first, coarsest last.
  // Children are in level L+1 pools; iterating levels in descending order
  // guarantees children are processed before their parents.
  void accumulate_all(AdaptiveCellCluster& cluster) {
    for (std::ptrdiff_t lv = static_cast<std::ptrdiff_t>(cluster.n_levels()) - 1; lv >= 0; --lv) {
      for (std::size_t s : cluster.level_slabs(static_cast<std::uint8_t>(lv))) {
        auto& slab = cluster.slab(s);
        for (std::size_t i = 0; i < CellSlab::CAPACITY; ++i) {
          if (slab.n_children[i] > 0 && slab.parent[i] != -2) {
            accumulate_to_parent(cluster, AdaptiveCellCluster::global(s, i));
          }
        }
      }
    }
  }

  // -----------------------------------------------------------------------
  // coarsen()
  //
  // Merge all children of `parent_g` back into the parent, making it a leaf.
  // Only valid when every child is itself a leaf (no grandchildren).
  //
  // Protocol (uses bidirectionality — no global scan needed):
  //   1. Accumulate children → parent (final physics state).
  //   2. For each child: scan its nb[]; any nb that is NOT a sibling has a
  //      back-reference on mirror_face[k] that still points to the child.
  //      Patch it to point to parent_g instead.
  //   3. Free child slots (sets freed sentinel parent=-2).
  //   4. Clear parent's first_child / n_children → parent becomes leaf.
  // -----------------------------------------------------------------------
  void coarsen(AdaptiveCellCluster& cluster, std::int32_t parent_g) {
    auto p = cluster[parent_g];
    assert(!p.is_leaf());

    auto const fc = p.first_child;
    auto const nc = p.n_children;

    // All children must be leaves.
    for (std::uint8_t c = 0; c < nc; ++c) assert(cluster[fc + c].is_leaf());

    // 1. Final accumulation — parent gets the children's averaged state.
    accumulate_to_parent(cluster, parent_g);

    // 2. Patch back-references from intra-level (non-sibling) neighbours.
    //    Each child knows its own neighbours; siblings are fc..fc+nc-1.
    //
    //    mirror_face[k] is only the correct reciprocal face on `nbg` when a
    //    child's external face k genuinely corresponds to the SAME face index
    //    on the parent — true by construction for the ratio-2 octant splits
    //    (Hex8/HexCenter9: external child face k always proxies through
    //    par.nb[k]), but NOT true in general for Odd27 (odd27_table can map
    //    a child's face k to a DIFFERENT parent face — see its build comment
    //    / the byte-for-byte cross-check against probe_split27.py). Guard
    //    the write with `outside.nb[mk] == child_g`: wire_split_children
    //    never modifies any array outside [fc, fc+nc) in the first place, so
    //    an external neighbour's nb[] already still points at parent_g
    //    (unchanged since before the split) for every current model — this
    //    guard makes that explicit rather than blindly overwriting whatever
    //    unrelated face mirror_face[k] happens to name on `nbg`.
    for (std::uint8_t c = 0; c < nc; ++c) {
      auto const child_g = fc + c;
      auto ch = cluster[child_g];
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        auto const nbg = ch.nb[k];
        if (nbg < 0) continue;                     // boundary / unset
        if (nbg >= fc && nbg < fc + nc) continue;  // sibling — no patch needed
        // nbg is an outside cell (same or coarser level).
        auto outside = cluster[nbg];
        auto const mk = bcc::mirror_face[k];
        if (outside.nb[mk] == child_g) outside.nb[mk] = parent_g;
      }
    }

    // 3. Free child slots.
    for (std::uint8_t c = 0; c < nc; ++c) cluster.free_cell(fc + c);

    // 4. Parent becomes a leaf.
    p.first_child = -1;
    p.n_children = 0;

    EQUILIBRIUM_DEBUG_DO({  //
      check_state_after_coarsen(cluster, parent_g);
    })
  }

  // -----------------------------------------------------------------------
  // resolve_stale_external_proxies()
  //
  // wire_split_children()'s Odd27 path resolves each child's "external"
  // faces (odd27_table.face_target[c][k] >= ODD27_EXTERNAL_BASE) to
  // `par.nb[...]` ONCE, at split time -- a fixed compile-time-table lookup
  // that is never revisited afterward. If the neighbour referenced by that
  // external face is LATER split too (independently, e.g. two adjacent
  // parent cells both refined to the same level by refine_region_to_
  // convergence), the wiring goes stale: the child keeps pointing at the
  // neighbour's now-non-leaf PARENT instead of the neighbour's own matching
  // child leaf, even though a genuine same-level leaf sibling now exists one
  // hop further in. Every consumer of nb[] (the wave-step flux formula,
  // leaf_is_boundary, step_cross's mirroring) then treats this as a real
  // coarse-fine junction -- averaging over the stale proxy's descendants --
  // when it's actually a plain same-level interior face. This is not
  // specific to hex or sq faces; fixed at the source here rather than
  // patched per-consumer.
  //
  // Walks every leaf's external (non-leaf-neighbour) faces and, using the
  // same position-descent primitive _step_cross_mirror_proxy already uses
  // for exactly this situation (_descend_to), checks whether the neighbour
  // has since been split down to a genuine LEAF at this leaf's own level. If
  // so, rewires both directions directly to that leaf -- no more proxy. If
  // descent instead stops at a coarser leaf (genuine partial refinement) or
  // another non-leaf (neighbour refined deeper than this level), that IS a
  // real multiplicity/coarse-fine junction -- left untouched.
  //
  // Called from rebuild_leaf_indices(), before the multiplicity tables are
  // rebuilt, so every downstream consumer sees corrected topology.
  // ALL 14 FACES (both hex k<8 and sq k>=8): in Odd27, at a one-level gap
  // (both sides split exactly once, the common uniform-region-refinement
  // case), EVERY direction -- hex or sq -- has exactly one geometrically-
  // unique child per parent, confirmed directly by position (2026-08-14):
  // built a cell + one hex neighbour + one sq neighbour, split all three,
  // and for every one of the still-stale (parent-routed) sq connections,
  // `_descend_to`'s nearest-position search landed on a DISTINCT child of
  // the split neighbour at exact (0.0) position match -- the same clean 1:1
  // structure as hex, not a fan-in.
  //
  // The previous HEX-ONLY restriction was based on a DIFFERENT topology
  // (2026-08-12, BFS-split regression repro,
  // tst/substrate/test_adaptive_vs_substrate.py TestMultipleSplits): deep
  // INCREMENTAL single-cell splitting, which manufactures genuine >1-level
  // gaps (already excluded below by the `cell.level - 1` check regardless of
  // face type) -- every observed sq-face collision in that test came from
  // that multi-level-gap topology, not from sq-ness itself. Restricting to
  // hex was fixing the wrong axis of the problem; the real invariant this
  // function depends on is "exactly one level of gap", which is face-type-
  // agnostic and already enforced independently of k. Left disabled for sq
  // on an untested assumption -- now extended to all 14 faces under the
  // same one-level-gap + collision-safety guard.
  //
  // Two-pass, collision-safe regardless: even at a one-level gap, a deeply-
  // nested "corner" child can have multiple directions whose greedy
  // _descend_to search all land on the SAME single target (confirmed
  // empirically 2026-08-12, BFS-split regression -- three different hex
  // directions off one source cell resolving to the identical target is
  // geometrically real, not a search bug: the coarser side's child grid
  // genuinely doesn't resolve those directions separately at this depth).
  // Applying such a resolution blindly makes two different source cells
  // overwrite the same target's back-reference, breaking
  // check_total_dphi_balanced -- this is what the contested-slot mechanism
  // below guards against, for hex and sq alike.
  //
  // Each candidate (g,k,part) makes TWO writes: a "direct" slot (g,k)=part
  // and a "mirror" slot (part,mirror_face[k])=g. A naive per-candidate
  // uniqueness check only catches two candidates colliding on the SAME kind
  // of write (e.g. two mirror writes) -- it misses one candidate's direct
  // write landing on the exact slot ANOTHER candidate's mirror write also
  // targets (confirmed empirically too: PART's own candidate, resolving its
  // face mirror_face[k] toward some X != g, writes its direct slot
  // (part,mirror_face[k])=X -- the same physical slot G's candidate's
  // mirror write targets with value g). So: accumulate every write (both
  // kinds, from every candidate) into ONE slot->value map first; any slot
  // that ever receives two DIFFERENT values is contested. Only commit a
  // candidate if NEITHER of its two slots ended up contested. Contested
  // slots are left exactly as they were pre-pass -- no worse than before,
  // safely deferred to the existing ancestor-substitution machinery.
  static void resolve_stale_external_proxies(AdaptiveCellCluster& cluster) {
    if (disable_resolve_stale_proxy_neighbour) return;
    struct Candidate {
      std::int32_t g;
      std::size_t k;
      std::int32_t part;
    };
    using Slot = std::pair<std::int32_t, std::size_t>;
    std::vector<Candidate> candidates;
    std::map<Slot, std::int32_t> slot_value;
    std::set<Slot> slot_contested;

    auto claim = [&](Slot const& slot, std::int32_t value) {
      auto it = slot_value.find(slot);
      if (it == slot_value.end())
        slot_value.emplace(slot, value);
      else if (it->second != value)
        slot_contested.insert(slot);
    };

    for (std::int32_t g : cluster.leaf_indices) {
      auto const cell = cluster[g];
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        auto const nbg = cell.nb[k];
        // Exactly a one-level gap only: _descend_to's greedy nearest-per-
        // level search is unambiguous for a single hop (searches one
        // parent's children directly), but is NOT guaranteed correct across
        // 2+ levels -- confirmed empirically (2026-08-12, same BFS-split
        // regression). Restricting to cluster[nbg].level == cell.level - 1
        // still fully covers the motivating case (a genuine single
        // L0/L1-style gap) while avoiding the deeper, unverified
        // multi-level case -- left to the existing ancestor-substitution
        // machinery, for any face whose gap is deeper than one level.
        if (nbg < 0 || cell.level == 0 || cluster[nbg].n_children == 0 ||
            cluster[nbg].level != static_cast<std::uint8_t>(cell.level - 1))
          continue;
        double tp[3];
        _neighbour_pos(cluster, g, k, tp);
        auto const part = _descend_to(cluster, nbg, cell.level, tp);
        if (part < 0 || cluster[part].level != cell.level || cluster[part].n_children != 0)
          continue;
        candidates.push_back({g, k, part});
        auto const mk = static_cast<std::size_t>(bcc::mirror_face[k]);
        claim({g, k}, part);
        claim({part, mk}, g);
      }
    }

    [[maybe_unused]] int n_committed = 0, n_contested = 0;
    for (auto const& c : candidates) {
      auto const mk = static_cast<std::size_t>(bcc::mirror_face[c.k]);
      if (slot_contested.count({c.g, c.k}) || slot_contested.count({c.part, mk})) {
        ++n_contested;
        continue;
      }
      cluster[c.g].nb[c.k]    = c.part;
      cluster[c.part].nb[mk] = c.g;
      ++n_committed;
    }
    EQUILIBRIUM_DEBUG_DO({
      if (std::getenv("EQ_DEBUG_STALE_PROXY") && (n_committed > 0 || n_contested > 0)) {
        std::fprintf(stderr, "resolve_stale_external_proxies: committed=%d contested(skipped)=%d\n",
                     n_committed, n_contested);
      }
    })
  }

  // -----------------------------------------------------------------------
  // Rebuild leaf_indices and resize step temp buffers.
  // Call after any split/coarsen pass before the next step.
  // -----------------------------------------------------------------------
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
    resolve_stale_external_proxies(cluster);
    auto const L = cluster.leaf_indices.size();
    cluster.g_max = L ? *std::max_element(cluster.leaf_indices.begin(),
                                          cluster.leaf_indices.end())
                      : -1;
    cluster.g_to_j.assign(static_cast<std::size_t>(cluster.g_max + 1), -1);
    for (std::size_t j = 0; j < L; ++j)
      cluster.g_to_j[cluster.leaf_indices[j]] = static_cast<std::int32_t>(j);
    cluster.temp_du.resize(L, 0.0);
    cluster.temp_phi.resize(CellSlab::NF * L, 0.0);
    cluster.temp_phi_mirror.resize(CellSlab::NF * L, 0.0);
    constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
    cluster.criterion_min.resize(L, NaN);
    cluster.criterion_max.resize(L, NaN);

    // Partition boundary leaves: any leaf with at least one face missing a
    // local neighbour.  Only these enter the BC dispatch pass (see step()).
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

    rebuild_amr_corrections(cluster);
    rebuild_face_multiplicity(cluster);
    rebuild_ancestor_group_multiplicity(cluster);

    // Identify all-leaf sibling groups for the vectorised sibling pass.
    cluster.sibling_groups.clear();
    for (std::size_t s = 0; s < cluster.n_slabs(); ++s) {
      auto& slab = cluster.slab(s);
      for (std::size_t i = 0; i < CellSlab::CAPACITY; ++i) {
        if (slab.parent[i] == -2 || slab.n_children[i] != adaptive_cell::N_CHILDREN_HEX) continue;
        auto const fc = slab.first_child[i];
        if (fc < 0) continue;
        // Check all 8 children are leaves.
        bool all_leaf = true;
        for (int c = 0; c < 8 && all_leaf; ++c)
          all_leaf = cluster[fc + c].is_leaf();
        if (!all_leaf) continue;
        // Find j0: first child's position in leaf_indices (sorted by global index).
        auto it = std::lower_bound(cluster.leaf_indices.begin(), cluster.leaf_indices.end(), fc);
        if (it == cluster.leaf_indices.end() || *it != fc) continue;
        std::size_t j0 = static_cast<std::size_t>(it - cluster.leaf_indices.begin());
        cluster.sibling_groups.push_back({j0, fc});
      }
    }
  }

  void rebuild_all() {
    for (auto& c : clusters) rebuild_leaf_indices(c);
  }

  // -----------------------------------------------------------------------
  // defragment()
  //
  // Compact sparse slabs by migrating their live cells to other slabs.
  // When a slab's occupancy drops below `threshold` (default 25%), all its
  // live cells are moved out — the slab then reaches live_count==0 and is
  // pushed to empty_slabs_ for reuse by future alloc_children calls.
  //
  // Children are always moved as a contiguous block (alloc_children) so the
  // first_child + c indexing invariant is preserved.  Parents are moved
  // individually — their children's parent[] pointers are updated in place.
  //
  // Bidirectionality handles nb[] fixup: each migrated cell scans its 14
  // neighbours and patches mirror-face back-references directly.
  //
  // Call after a coarsen pass (when fragmentation is highest).
  // Must be followed by rebuild_leaf_indices() since global indices change.
  // -----------------------------------------------------------------------
  void defragment(AdaptiveCellCluster& cluster,
                  std::array<std::uint8_t, CellSlab::NF> const& mirror, double threshold = 0.25) {
    EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::defragment");

    auto const ns = cluster.n_slabs();
    for (std::size_t s = 0; s < ns; ++s) {
      if (cluster.slab_occupancy(s) >= threshold) continue;
      if (cluster.slab_occupancy(s) == 0.0) continue;  // already empty

      // Collect live cells in this slab.
      std::vector<std::int32_t> live;
      auto& slab = cluster.slab(s);
      for (std::size_t i = 0; i < CellSlab::CAPACITY; ++i) {
        if (slab.parent[i] != -2)  // -2 = freed sentinel
          live.push_back(AdaptiveCellCluster::global(s, i));
      }

      // Move non-leaf cells' children blocks first (children must remain
      // contiguous).  A non-leaf's children may be in any slab — only
      // move the non-leaf cell itself here; children are handled when
      // their own slab is defragmented (or stay put if healthy).
      // Leaf cells and non-leaf cells are migrated individually.
      for (std::int32_t g : live) {
        if (AdaptiveCellCluster::slab_of(g) != s) continue;  // already moved
        cluster.migrate_cell(g, mirror);
      }
      // Slab s should now be empty (live_count==0) and pushed to empty_slabs_.
    }
  }

  // -----------------------------------------------------------------------
  // step()
  //
  // One DST update tick — symplectic Euler wave equation, matching
  // SubstrateLattice.step() in equilibrium/substrate/lattice.py.
  //
  // Physics (3-pass, phi is a persistent wave field):
  //   Pass 1 — u update (reads phi[], writes slab.u directly):
  //     du[i]   = -dt · sigma_i · Σ_k W[k] · phi[k][i]
  //     u_new   = u + du   (written to slab so pass 2 can read neighbours)
  //   Pass 2 — dphi from u_new (reads slab.u, writes temp_phi):
  //     dphi[k] = -dt · sigma_face · W[k] · 2/(L_i+L_nb) · (u_new_nb − u_new_i)
  //   Pass 3 — accumulate phi (reads temp_phi, writes slab.phi):
  //     phi[k] += dphi[k]
  //
  // Pool workers each take a contiguous leaf range per pass; the calling
  // thread handles the single-threaded BC pass between pass 2 and pass 3.
  // -----------------------------------------------------------------------
  void step(AdaptiveCellCluster& cluster, TimeStepper const& ts) {
    EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::step");

    _accumulate_boundary_proxies(cluster);
    auto const L       = cluster.leaf_indices.size();
    auto const B       = cluster.boundary_leaf_positions.size();
    auto const has_bc  = cluster.bc && B > 0;
    auto const uniform = (cluster.n_non_leaf_cells == 0);

    if (!pool_) {
      _step_pass1(cluster, ts, 0, L);
      if (uniform)
        _step_pass2_uniform(cluster, ts, 0, L);
      else
        _step_pass2(cluster, ts, 0, L);
      if (has_bc) _step_pass2_bc(cluster, ts, 0, B);
      if (!disable_step_pass2_amr_correct) _step_pass2_amr_correct(cluster);
      _step_pass2_siblings(cluster, ts.dt_base);
      // check_total_dphi_balanced's invariant is only guaranteed to hold
      // once _step_pass2_amr_correct has actually run -- that's the pass
      // responsible for zeroing the global weighted-dphi imbalance in the
      // first place. Several intentional diagnostic tests set
      // disable_step_pass2_amr_correct=true to isolate the raw flux formula
      // from reflux's redistribution (see its own doc comment); asserting
      // the invariant anyway in that mode was checking a postcondition
      // outside its precondition, tripping on every debug (-gW, non -g 0)
      // build and aborting the whole process before any later test could
      // run (found 2026-08-05, `now`).
      EQUILIBRIUM_DEBUG_DO({  //
        if (!disable_step_pass2_amr_correct)
          check_total_dphi_balanced(cluster, CellSlab::NF, bcc::face_weight);
      })
      _step_pass3(cluster, 0, L);
      return;
    }

    {
      EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::_pass1_dispatch");
      pool_->detach_blocks<std::size_t>(
          0, L, [&](std::size_t b, std::size_t e) { _step_pass1(cluster, ts, b, e); });
      pool_->wait();
    }

    {
      EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::_pass2_dispatch");
      if (uniform)
        pool_->detach_blocks<std::size_t>(
            0, L, [&](std::size_t b, std::size_t e) { _step_pass2_uniform(cluster, ts, b, e); });
      else
        pool_->detach_blocks<std::size_t>(
            0, L, [&](std::size_t b, std::size_t e) { _step_pass2(cluster, ts, b, e); });
      pool_->wait();
    }

    if (has_bc) {
      EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::_pass2_bc_dispatch");
      pool_->detach_blocks<std::size_t>(
          0, B, [&](std::size_t b, std::size_t e) { _step_pass2_bc(cluster, ts, b, e); });
      pool_->wait();
    }

    if (!disable_step_pass2_amr_correct) _step_pass2_amr_correct(cluster);
    _step_pass2_siblings(cluster, ts.dt_base);
    // See the matching comment in the !pool_ branch above -- the invariant
    // only holds once _step_pass2_amr_correct has run.
    EQUILIBRIUM_DEBUG_DO({  //
      if (!disable_step_pass2_amr_correct)
        check_total_dphi_balanced(cluster, CellSlab::NF, bcc::face_weight);
    })

    {
      EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::_pass3_dispatch");
      pool_->detach_blocks<std::size_t>(
          0, L, [&](std::size_t b, std::size_t e) { _step_pass3(cluster, b, e); });
      pool_->wait();
    }
  }

  void step(AdaptiveCellCluster& cluster, double dt) {
    step(cluster, TimeStepper::make_fixed(dt));
  }

  // Damped variant of step() — folds a `gamma`-based sponge (see bc.hpp)
  // into pass1/pass3 instead of requiring a separate step_sponge() call.
  // A new function rather than a flag on step() so the split (existing,
  // vol_damp-based) and integrated (gamma-based) sponge behaviours can be
  // A/B compared directly — see [[project_poynting_perf_r150_farfield]].
  // Everything except pass1/pass3 (pass2, BC, AMR correction, siblings) is
  // identical to step() since dphi's flux formula never depended on the old
  // phi value in the first place — only the state the decay acts on differs.
  void step_damped(AdaptiveCellCluster& cluster, TimeStepper const& ts) {
    EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::step_damped");

    _accumulate_boundary_proxies(cluster);
    auto const L       = cluster.leaf_indices.size();
    auto const B       = cluster.boundary_leaf_positions.size();
    auto const has_bc  = cluster.bc && B > 0;
    auto const uniform = (cluster.n_non_leaf_cells == 0);

    if (!pool_) {
      _step_pass1_damped(cluster, ts, 0, L);
      if (uniform)
        _step_pass2_uniform(cluster, ts, 0, L);
      else
        _step_pass2(cluster, ts, 0, L);
      if (has_bc) _step_pass2_bc(cluster, ts, 0, B);
      if (!disable_step_pass2_amr_correct) _step_pass2_amr_correct(cluster);
      _step_pass2_siblings(cluster, ts.dt_base);
      EQUILIBRIUM_DEBUG_DO({  //
        if (!disable_step_pass2_amr_correct)
          check_total_dphi_balanced(cluster, CellSlab::NF, bcc::face_weight);
      })
      _step_pass3_damped(cluster, ts, 0, L);
      return;
    }

    {
      EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::_pass1_damped_dispatch");
      pool_->detach_blocks<std::size_t>(
          0, L, [&](std::size_t b, std::size_t e) { _step_pass1_damped(cluster, ts, b, e); });
      pool_->wait();
    }

    {
      EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::_pass2_dispatch");
      if (uniform)
        pool_->detach_blocks<std::size_t>(
            0, L, [&](std::size_t b, std::size_t e) { _step_pass2_uniform(cluster, ts, b, e); });
      else
        pool_->detach_blocks<std::size_t>(
            0, L, [&](std::size_t b, std::size_t e) { _step_pass2(cluster, ts, b, e); });
      pool_->wait();
    }

    if (has_bc) {
      EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::_pass2_bc_dispatch");
      pool_->detach_blocks<std::size_t>(
          0, B, [&](std::size_t b, std::size_t e) { _step_pass2_bc(cluster, ts, b, e); });
      pool_->wait();
    }

    if (!disable_step_pass2_amr_correct) _step_pass2_amr_correct(cluster);
    _step_pass2_siblings(cluster, ts.dt_base);
    EQUILIBRIUM_DEBUG_DO({  //
      if (!disable_step_pass2_amr_correct)
        check_total_dphi_balanced(cluster, CellSlab::NF, bcc::face_weight);
    })

    {
      EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::_pass3_damped_dispatch");
      pool_->detach_blocks<std::size_t>(
          0, L, [&](std::size_t b, std::size_t e) { _step_pass3_damped(cluster, ts, b, e); });
      pool_->wait();
    }
  }

  void step_damped(AdaptiveCellCluster& cluster, double dt) {
    step_damped(cluster, TimeStepper::make_fixed(dt));
  }

  // Directional (PML-lite) variant of step() — see bc.hpp's `gamma_face`
  // doc comment. pass1/pass2 are the plain, undamped originals (u is never
  // touched); only pass3 differs (_step_pass3_dir).
  void step_dir_damped(AdaptiveCellCluster& cluster, TimeStepper const& ts) {
    EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::step_dir_damped");

    _accumulate_boundary_proxies(cluster);
    auto const L       = cluster.leaf_indices.size();
    auto const B       = cluster.boundary_leaf_positions.size();
    auto const has_bc  = cluster.bc && B > 0;
    auto const uniform = (cluster.n_non_leaf_cells == 0);

    if (!pool_) {
      _step_pass1(cluster, ts, 0, L);
      if (uniform)
        _step_pass2_uniform(cluster, ts, 0, L);
      else
        _step_pass2(cluster, ts, 0, L);
      if (has_bc) _step_pass2_bc(cluster, ts, 0, B);
      if (!disable_step_pass2_amr_correct) _step_pass2_amr_correct(cluster);
      _step_pass2_siblings(cluster, ts.dt_base);
      EQUILIBRIUM_DEBUG_DO({  //
        if (!disable_step_pass2_amr_correct)
          check_total_dphi_balanced(cluster, CellSlab::NF, bcc::face_weight);
      })
      _step_pass3_dir(cluster, ts, 0, L);
      return;
    }

    {
      EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::_pass1_dispatch");
      pool_->detach_blocks<std::size_t>(
          0, L, [&](std::size_t b, std::size_t e) { _step_pass1(cluster, ts, b, e); });
      pool_->wait();
    }

    {
      EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::_pass2_dispatch");
      if (uniform)
        pool_->detach_blocks<std::size_t>(
            0, L, [&](std::size_t b, std::size_t e) { _step_pass2_uniform(cluster, ts, b, e); });
      else
        pool_->detach_blocks<std::size_t>(
            0, L, [&](std::size_t b, std::size_t e) { _step_pass2(cluster, ts, b, e); });
      pool_->wait();
    }

    if (has_bc) {
      EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::_pass2_bc_dispatch");
      pool_->detach_blocks<std::size_t>(
          0, B, [&](std::size_t b, std::size_t e) { _step_pass2_bc(cluster, ts, b, e); });
      pool_->wait();
    }

    if (!disable_step_pass2_amr_correct) _step_pass2_amr_correct(cluster);
    _step_pass2_siblings(cluster, ts.dt_base);
    EQUILIBRIUM_DEBUG_DO({  //
      if (!disable_step_pass2_amr_correct)
        check_total_dphi_balanced(cluster, CellSlab::NF, bcc::face_weight);
    })

    {
      EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::_pass3_dir_dispatch");
      pool_->detach_blocks<std::size_t>(
          0, L, [&](std::size_t b, std::size_t e) { _step_pass3_dir(cluster, ts, b, e); });
      pool_->wait();
    }
  }

  void step_dir_damped(AdaptiveCellCluster& cluster, double dt) {
    step_dir_damped(cluster, TimeStepper::make_fixed(dt));
  }

  // Split-field PML variant of step() — see bc.hpp's proj_r/u_aux/phi_aux
  // doc comment. Only pass1/pass3 differ (_step_pass1_pml/_step_pass3_pml);
  // pass2 is the plain original (dphi's flux formula is unaffected, same as
  // the earlier damped/dir_damped variants).
  void step_pml(AdaptiveCellCluster& cluster, TimeStepper const& ts) {
    EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::step_pml");

    _accumulate_boundary_proxies(cluster);
    auto const L       = cluster.leaf_indices.size();
    auto const B       = cluster.boundary_leaf_positions.size();
    auto const has_bc  = cluster.bc && B > 0;
    auto const uniform = (cluster.n_non_leaf_cells == 0);

    if (!pool_) {
      _step_pass1_pml(cluster, ts, 0, L);
      if (uniform)
        _step_pass2_uniform(cluster, ts, 0, L);
      else
        _step_pass2(cluster, ts, 0, L);
      if (has_bc) _step_pass2_bc(cluster, ts, 0, B);
      if (!disable_step_pass2_amr_correct) _step_pass2_amr_correct(cluster);
      _step_pass2_siblings(cluster, ts.dt_base);
      EQUILIBRIUM_DEBUG_DO({  //
        if (!disable_step_pass2_amr_correct)
          check_total_dphi_balanced(cluster, CellSlab::NF, bcc::face_weight);
      })
      _step_pass3_pml(cluster, ts, 0, L);
      return;
    }

    {
      EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::_pass1_pml_dispatch");
      pool_->detach_blocks<std::size_t>(
          0, L, [&](std::size_t b, std::size_t e) { _step_pass1_pml(cluster, ts, b, e); });
      pool_->wait();
    }

    {
      EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::_pass2_dispatch");
      if (uniform)
        pool_->detach_blocks<std::size_t>(
            0, L, [&](std::size_t b, std::size_t e) { _step_pass2_uniform(cluster, ts, b, e); });
      else
        pool_->detach_blocks<std::size_t>(
            0, L, [&](std::size_t b, std::size_t e) { _step_pass2(cluster, ts, b, e); });
      pool_->wait();
    }

    if (has_bc) {
      EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::_pass2_bc_dispatch");
      pool_->detach_blocks<std::size_t>(
          0, B, [&](std::size_t b, std::size_t e) { _step_pass2_bc(cluster, ts, b, e); });
      pool_->wait();
    }

    if (!disable_step_pass2_amr_correct) _step_pass2_amr_correct(cluster);
    _step_pass2_siblings(cluster, ts.dt_base);
    EQUILIBRIUM_DEBUG_DO({  //
      if (!disable_step_pass2_amr_correct)
        check_total_dphi_balanced(cluster, CellSlab::NF, bcc::face_weight);
    })

    {
      EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::_pass3_pml_dispatch");
      pool_->detach_blocks<std::size_t>(
          0, L, [&](std::size_t b, std::size_t e) { _step_pass3_pml(cluster, ts, b, e); });
      pool_->wait();
    }
  }

  void step_pml(AdaptiveCellCluster& cluster, double dt) {
    step_pml(cluster, TimeStepper::make_fixed(dt));
  }

  void step_all(TimeStepper const& ts) {
    for (auto& c : clusters) step(c, ts);
  }

  void step_all(double dt) {
    step_all(TimeStepper::make_fixed(dt));
  }

  void step_all_damped(TimeStepper const& ts) {
    for (auto& c : clusters) step_damped(c, ts);
  }

  void step_all_damped(double dt) {
    step_all_damped(TimeStepper::make_fixed(dt));
  }

  void step_all_dir_damped(TimeStepper const& ts) {
    for (auto& c : clusters) step_dir_damped(c, ts);
  }

  void step_all_dir_damped(double dt) {
    step_all_dir_damped(TimeStepper::make_fixed(dt));
  }

  void step_all_pml(TimeStepper const& ts) {
    for (auto& c : clusters) step_pml(c, ts);
  }

  void step_all_pml(double dt) {
    step_all_pml(TimeStepper::make_fixed(dt));
  }

  // -----------------------------------------------------------------------
  // step_subcycle() — proper subcycling for by-level timesteppers.
  //
  // All cells end at the same physical time t + dt_base.
  // Coarse-first ordering: level l takes ratio^l sub-steps of dt_base/ratio^l
  // — `ratio` MUST match the refinement ratio the lattice was actually split
  // with (split_model_traits<Model>::ratio; defaults to 3.0 = Odd27).
  //   level 0: 1 step         of dt_base           (coarse steps first, sees fine at t)
  //   level 1: ratio steps    of dt_base/ratio      (fine sees fresh coarse after step)
  //   level 2: ratio^2 steps  of dt_base/ratio^2
  //   ...
  //
  // AMR junction correction uses register-based reflux, not the plain
  // _step_pass2_amr_correct (that assumes both sides of a junction were
  // just computed together, true for step()'s single dt_base pass but not
  // here — a junction's two sides generally run in DIFFERENT (lev, sub)
  // iterations at different cadences). Every substep accumulates its
  // pair sums into amr_reg_a/amr_reg_b via _accumulate_amr_registers
  // (each side still applies its own honest per-substep temp_phi via
  // pass3, unmodified); once the whole outer step (all levels) is done,
  // _apply_amr_reflux applies ONE correction per pair straight to phi,
  // using the totals accumulated over the full dt_base step. See
  // _accumulate_amr_registers's comment for why the old lev==0-only
  // single-shot correction was both a double-count (level-0/1 junctions)
  // and a silent no-op (any junction not touching level 0).
  //
  // NOTE: BCs from cluster.bc are not yet filtered by level — skipped here.
  // Harris sheet uses wired periodic faces (no cluster.bc) so this is fine.
  // -----------------------------------------------------------------------
  void step_subcycle(AdaptiveCellCluster& cluster, double dt_base, double ratio = 3.0) {
    EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::step_subcycle");

    // Find the max leaf level in this cluster.
    uint8_t max_lev = 0;
    for (auto g : cluster.leaf_indices)
      max_lev = std::max(max_lev, cluster[g].level);

    if (max_lev == 0) {
      step(cluster, TimeStepper::make_fixed(dt_base));
      return;
    }
    if (auto const* bc = cluster.bc.get()) {
      bool const inert_fn = !bc->fn || bc->fn == BoundaryCondition::absorbing;
      if (!inert_fn || bc->has_face_ghosts() || bc->remote_flux != 0.0 || !bc->outflow_active.empty() ||
          !bc->outflow_taper_weight.empty())
        throw std::logic_error(
            "step_subcycle: face BCs (fn / ghosts / remote / outflow) are not applied on refined "
            "clusters; use step() or a volumetric sponge (step_sponge)");
    }

    auto const ratio_i = static_cast<std::size_t>(std::llround(ratio));

    // Flux registers for the reflux-at-end-of-outer-step AMR correction —
    // see _accumulate_amr_registers / _apply_amr_reflux. Reset every call
    // (one call == one full dt_base outer step).
    std::vector<double> amr_reg_a(cluster.amr_correction_pairs.size(), 0.0);
    std::vector<double> amr_reg_b(cluster.amr_correction_pairs.size(), 0.0);

    if (std::getenv("EQ_DEBUG_FLAT_SUBCYCLE")) {
      for (uint8_t lev = 0; lev <= max_lev; ++lev) {
        std::size_t n_sub = 1;
        for (uint8_t i = 0; i < lev; ++i) n_sub *= ratio_i;
        auto const dt_lev = dt_base / double(n_sub);
        for (std::size_t sub = 0; sub < n_sub; ++sub)
          _step_subcycle_one(cluster, dt_lev, lev, amr_reg_a, amr_reg_b);
      }
    } else {
      _step_subcycle_recurse(cluster, 0, dt_base, max_lev, ratio_i, amr_reg_a, amr_reg_b);
    }

    EQUILIBRIUM_DEBUG_DO({  //
      print_amr_reflux_magnitude(cluster, amr_reg_a, amr_reg_b, "step_subcycle");
    })
    _apply_amr_reflux(cluster, amr_reg_a, amr_reg_b);
  }

 private:
  // One atomic (level, single dt_lev step) update — refresh coarse-fine
  // proxies, pass1/pass2/siblings/pass3, accumulate the AMR reflux
  // registers. Shared body for every node of _step_subcycle_recurse below.
  void _step_subcycle_one(AdaptiveCellCluster& cluster, double dt_lev, std::uint8_t lev,
                          std::vector<double>& reg_a, std::vector<double>& reg_b) {
    auto const L = cluster.leaf_indices.size();
    std::fill(cluster.temp_phi.begin(), cluster.temp_phi.end(), 0.0);
    _accumulate_boundary_proxies(cluster);

    if (!pool_) {
      _step_pass1_lev(cluster, dt_lev, lev, 0, L);
      _step_pass2_lev(cluster, dt_lev, lev, 0, L);
      _step_pass2_siblings(cluster, dt_lev, static_cast<std::int8_t>(lev));
      _accumulate_amr_registers(cluster, reg_a, reg_b);
      _step_pass3(cluster, 0, L);
    } else {
      pool_->detach_blocks<std::size_t>(0, L, [&, lev, dt_lev](std::size_t b, std::size_t e) {
        _step_pass1_lev(cluster, dt_lev, lev, b, e);
      });
      pool_->wait();
      pool_->detach_blocks<std::size_t>(0, L, [&, lev, dt_lev](std::size_t b, std::size_t e) {
        _step_pass2_lev(cluster, dt_lev, lev, b, e);
      });
      pool_->wait();
      _step_pass2_siblings(cluster, dt_lev, static_cast<std::int8_t>(lev));
      _accumulate_amr_registers(cluster, reg_a, reg_b);
      pool_->detach_blocks<std::size_t>(0, L, [&](std::size_t b, std::size_t e) {
        _step_pass3(cluster, b, e);
      });
      pool_->wait();
    }
  }

  // Recursive Berger-Oliger-style subcycle: level `lev` takes ONE step of
  // dt_lev, then — if finer levels exist — recurses into level lev+1
  // `ratio_i` times, each covering dt_lev/ratio_i. This is properly NESTED/
  // interleaved: level lev+1's substeps happen strictly within the span of
  // ONE specific level-lev substep at a time, unlike the old flat "finish
  // level lev entirely (all its substeps), then start lev+1" ordering. That
  // old ordering meant every finer level saw its coarser neighbour's FINAL
  // (fully end-of-outer-step) state throughout ALL of its own substeps —
  // effectively "double-stepping" the coarse side once per fine substep
  // instead of apportioning its contribution across them. Nesting properly
  // shrinks the "how far in the future is my coarse neighbour" gap from the
  // full dt_base (old scheme, at every level) down to just dt_lev at the
  // immediate parent level, recursively smaller at each successive level —
  // found 2026-07-19 investigating a deep-telescope (L6/L8) AMR runaway that
  // persisted after fixing two separate, real, but insufficient bugs (see
  // project_odd27_amr_junction_leak memory for the full investigation).
  // Temporary diagnostic (2026-07-19): H = sum(u^2) + 0.5*sum(phi^2) over
  // leaves only, matching the Python-side H() used throughout the
  // deep-telescope investigation -- split into leaves with at least one
  // cross-level (boundary-adjacent) face vs. leaves whose every face is
  // same-level (fully interior), isolating whether growth concentrates at
  // the boundary or is spread uniformly. Gated by EQ_DEBUG_H_TRACE so it
  // costs nothing unless explicitly enabled.
  static void _leaf_H_split(AdaptiveCellCluster& cl, double& h_boundary, double& h_interior,
                            std::size_t& n_boundary, std::size_t& n_interior) {
    h_boundary = 0.0;
    h_interior = 0.0;
    n_boundary = 0;
    n_interior = 0;
    for (auto g : cl.leaf_indices) {
      auto cell = cl[g];
      bool is_boundary = false;
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        auto const nbg = cell.nb[k];
        if (nbg >= 0 && cl[nbg].L != cell.L) { is_boundary = true; break; }
      }
      double h = cell.u * cell.u;
      for (std::size_t k = 0; k < CellSlab::NF; ++k) h += 0.5 * cell.phi[k] * cell.phi[k];
      if (is_boundary) { h_boundary += h; ++n_boundary; }
      else { h_interior += h; ++n_interior; }
    }
  }

  void _step_subcycle_recurse(AdaptiveCellCluster& cluster, std::uint8_t lev, double dt_lev,
                              std::uint8_t max_lev, std::size_t ratio_i,
                              std::vector<double>& reg_a, std::vector<double>& reg_b) {
    EQUILIBRIUM_DEBUG_DO({
      if (std::getenv("EQ_DEBUG_SUBCYCLE_ORDER"))
        std::fprintf(stderr, "L%d ", static_cast<int>(lev));
    })
    _step_subcycle_one(cluster, dt_lev, lev, reg_a, reg_b);
    EQUILIBRIUM_DEBUG_DO({
      if (std::getenv("EQ_DEBUG_H_TRACE")) {
        static thread_local long call_idx = 0;
        double h_b, h_i;
        std::size_t n_b, n_i;
        _leaf_H_split(cluster, h_b, h_i, n_b, n_i);
        std::fprintf(stderr, "H_TRACE call=%ld level=%d H_boundary=%.17g (n=%zu) H_interior=%.17g (n=%zu)\n",
                     call_idx++, static_cast<int>(lev), h_b, n_b, h_i, n_i);
      }
    })
    if (lev < max_lev) {
      auto const dt_next = dt_lev / static_cast<double>(ratio_i);
      for (std::size_t sub = 0; sub < ratio_i; ++sub)
        _step_subcycle_recurse(cluster, static_cast<std::uint8_t>(lev + 1), dt_next, max_lev,
                               ratio_i, reg_a, reg_b);
    }
  }

 public:

  // Diagnostic toggle (2026-08-05): step()'s (non-subcycled path)
  // _step_pass2_amr_correct call has no existing gate -- needed to test
  // enable_cross_level_gradient_correction's effect on _step_pass2/_lev's
  // OWN output in isolation, since _step_pass2_amr_correct redistributes
  // temp_phi across amr_correction_pairs afterward using logic that assumes
  // the original (uncorrected) per-face formula, confounding any direct
  // before/after comparison of the flux formula itself.
  static inline bool disable_step_pass2_amr_correct = false;

  // Temporary diagnostic toggle (2026-07-19): when true, _step_pass2_lev
  // always uses a fine leaf's OWN u as the flux source, even when its
  // neighbour is coarser -- bypassing the ancestor-walk substitution that
  // replaces u_i with a shared, multi-descendant-averaged ancestor u.
  // Not a permanent feature -- remove once the root cause is found.
  static inline bool disable_ancestor_u_src = false;

  // Diagnostic toggle (2026-08-12): when true, disables the stale-proxy
  // resolution added to _step_pass2_lev/_step_pass2 (see the comment at its
  // call site) -- restores the pre-fix behaviour of always ancestor-
  // substituting/averaging when nb[k] is non-leaf, even when the neighbour
  // has since been split to a genuine same-level match. Default false (fix
  // active) -- kept as an escape hatch for isolating its effect, matching
  // the file's existing diagnostic-toggle convention.
  static inline bool disable_resolve_stale_proxy_neighbour = false;

  // Temporary diagnostic toggle (2026-07-19): when true, skip both
  // face_multiplicity and ancestor_group_multiplicity divisors entirely
  // (treat every face as multiplicity 1), isolating whether the counting
  // correction itself is a factor in the deep-telescope growth.
  // Not a permanent feature -- remove once the root cause is found.
  static inline bool disable_multiplicity_correction = false;

  // Temporary diagnostic toggle (2026-07-19): when true, sig_face uses a
  // true recursive mean-of-sigma over a non-leaf neighbour's leaf
  // descendants (_mean_sigma_recursive) instead of nb.sigma() (which is
  // sigma(mean(lam)) -- Jensen-biased relative to mean(sigma(lam)) since
  // sigma(lam)=1/(1+lam) is convex). Isolating whether this asymmetry
  // (a proxy neighbour's conductance being biased relative to a genuine
  // leaf neighbour's) is a factor in the deep-telescope growth.
  // Not a permanent feature -- remove once the root cause is found.
  static inline bool use_recursive_mean_sigma = false;

  // Experimental (2026-08-05): when true, both sides of a cross-level
  // face's (nb.u - u_src) difference are extrapolated via
  // _gradient_corrected_value (cheap 1-hop, same-level-only least-squares
  // fit) to the fine cell's own actual geometric face position, instead of
  // using each side's raw centre value unmodified. Motivated by
  // test_cross_level_flux_curvature_error[_gap_sweep] in
  // tst/eq/core/data/lattice/adaptive_lattice.cpp, which measured this raw
  // formula disagreeing with the true derivative at the geometric face by
  // ~56% for a single-level gap, growing with gap depth. Off by default --
  // not yet validated against the real base-update L8 blowup, only against
  // the static curvature-error tests. See doc/dst_algo.md.
  static inline bool enable_cross_level_gradient_correction = false;

  // Test-only hook exposing the private _local_gradient_same_level fit
  // directly, so its own accuracy can be checked in isolation from the
  // full cross-level flux formula (same pattern as _leaf_laplacian being
  // public for test_biharmonic_precision_floor_by_level).
  static void debug_local_gradient_same_level(AdaptiveCellCluster& cl, std::int32_t g,
                                              double out[3]) {
    _local_gradient_same_level(cl, g, out);
  }

  // Temporary diagnostic toggle (2026-07-19): when true, coarse-fine proxy
  // means are refreshed ONCE at the start of the whole outer (dt_base) step
  // instead of every substep -- breaks the self-referential feedback where
  // a boundary cell's own just-updated u feeds into its ancestor's mean
  // Temporary diagnostic (2026-07-19): scales the magnitude of dphi for
  // cross-level (boundary) faces specifically -- 1.0 (default) matches
  // original behaviour.
  // Not a permanent feature -- remove once the root cause is found.
  static inline double cross_level_flux_scale = 1.0;

  // (2026-08-05): de-biases any proxy (non-leaf) cell's `.u` when read for
  // flux purposes, by subtracting an estimate of accumulate_to_parent's own
  // Jensen-inequality restriction bias -- mean(children.u) is provably too
  // HIGH for any locally-convex field, by exactly (16/9)*L_child^2*curvature
  // at the proxy, a closed-form result derived analytically and confirmed
  // to fully explain test_cross_level_aggregate_isotropy_full_group's
  // discrepancy (see `now`, "cross-level flux formula has a real,
  // substantial aggregate defect"). curvature is estimated via
  // _ancestor_curvature_estimate (reuses _leaf_laplacian's own already-
  // verified-exact per-cell formula, applied directly to the proxy instead
  // of through the leaf-array/g_to_j machinery). Off by default -- unlike
  // the two prior failed attempts (enable_cross_level_gradient_correction,
  // the curvature-corrected-secant idea), this one targets the VALUE bias
  // itself rather than the spacing/weight side of the formula.
  //
  // Originally implemented as an inline patch to u_src, applied ONLY on the
  // ancestor-substitution branch (nb.L > L_i) -- confirmed (via `now`'s
  // "reflux survival" investigation) to be invisible to (a) a same-level
  // neighbour Q's own read of a proxy P's .u (Q.L == P.L, so Q never enters
  // the ancestor-substitution branch at all) and (b) _step_pass2_amr_
  // correct's reflux, which forces the fine side's aggregate dphi to
  // exactly negate Q's own (unpatched, still-biased) dphi -- silently
  // erasing the correction's entire effect whenever reflux is enabled.
  // Fixed by moving the correction to a single read-time helper
  // (_debiased_proxy_u, below) used at EVERY point a proxy's .u feeds a
  // flux calculation -- including Q's same-level read -- so the correction
  // propagates through the shared reference value both sides of a junction
  // ultimately agree on, and survives reflux's antisymmetry enforcement
  // rather than being computed on one side and discarded.
  static inline bool enable_ancestor_debias_correction = false;

  // Temporary diagnostic toggle (2026-08-15): exponent applied to `cell.L` in
  // `_step_cross_local`'s per-face `delta` (default 1.0 = original
  // `cell.L^1` behaviour). That factor was designed to convert a FLAT,
  // shared L0-calibrated `dt` (the same numeric value used across every
  // level of one AMR mesh) into an effective per-cell dt (`dt_L0 * cell.L`
  // ~= the cell's own CFL-stable dt) -- see the comment at its use site. It
  // assumes the caller always passes that flat L0-reference `dt`. Callers
  // that instead pass a dt ALREADY matched to their own uniform mesh's cell
  // size (`_compute_dt_l0(cl, D1=cell_size)`, e.g. `pure_l1size_control`/
  // `tst/substrate/adaptive/test_step_cross_per_level.py`) have no per-mesh
  // level variation for `cell.L^1` to correct -- multiplying by it there
  // double-counts the D1 scaling already baked into `dt`.
  //
  // Measured 2026-08-15 (`test_fractional_loss_level_independent`,
  // D1 1.0 vs 1/3, cross_scale=1e-3): power=1 (default) gives
  // ratio(L1/L0)=0.111 =~ (1/3)^2 -- dt and cell.L^1 both carrying a factor
  // of the D1 ratio, squaring it. power=0 (factor removed entirely) gives
  // ratio=0.333 =~ (1/3)^1 exactly -- confirms only ONE of the two D1
  // factors was removed (dt's own), not both. Neither is 1 (level-
  // independent); reaching that requires cell.L^(-1) (cancelling dt's D1
  // dependence too), untested as of this comment -- NOT a permanent
  // feature; if some power measurably gives ratio~1, the real fix is to
  // make the exponent conditional on the calling convention (an explicit
  // `L_ref` parameter), not to hardcode a different constant power (the
  // flat-dt/AMR case still needs power=1).
  static inline double step_cross_level_scale_power = 1.0;

  // Discrete Laplacian at a single (possibly non-leaf) cell `g`, restricted
  // to same-level, live neighbours -- same face_weight/1/L^2 formula as
  // _leaf_laplacian, but evaluated directly off cl[g].u instead of through
  // an arbitrary field array + leaf_indices/g_to_j indirection, so it can be
  // applied to a non-leaf ancestor proxy (whose own .u is kept current by
  // accumulate_to_parent, but which has no leaf_indices slot of its own).
  // Returns the SAME normalisation as _leaf_laplacian for a fully-symmetric
  // 14-same-level-neighbour cell (raw output 32.0 for u=x^2+y^2+z^2) -- divide
  // by 32.0 to recover the field's own curvature constant `c` (as in
  // u=c*(x^2+y^2+z^2)). Boundary ancestors with fewer live same-level
  // neighbours will under-estimate; same acceptable-heuristic caveat as
  // _leaf_laplacian itself (see its doc comment).
  static double _ancestor_curvature_estimate(AdaptiveCellCluster& cl, std::int32_t g) {
    auto const cell = cl[g];
    double acc = 0.0;
    for (std::size_t k = 0; k < CellSlab::NF; ++k) {
      auto const nbg = cell.nb[k];
      if (nbg < 0) continue;
      auto const nb = cl[nbg];
      if (nb.L != cell.L || nb.n_children > 0) continue;
      acc += bcc::face_star[k] * (nb.u - cell.u) / (cell.L * cell.L);
    }
    return acc;
  }

  // Single read-time entry point for a (possibly non-leaf) cell's debiased
  // `.u` -- a no-op (returns cell.u unchanged) for genuine leaves, so this
  // is strictly scoped to proxy cells and cannot affect ordinary interior/
  // uniform-mesh physics. See enable_ancestor_debias_correction's doc
  // comment for why every flux-facing read of a proxy's .u (not just the
  // ancestor-substitution branch) must go through this same helper.
  static double _debiased_proxy_u(AdaptiveCellCluster& cl, std::int32_t g) {
    auto const cell = cl[g];
    if (cell.n_children == 0) return cell.u;
    auto const L_child = cl[cell.first_child].L;
    auto const c_est   = _ancestor_curvature_estimate(cl, g) / 32.0;
    return cell.u - (16.0 / 9.0) * L_child * L_child * c_est;
  }

  void step_all_subcycle(double dt_base, double ratio = 3.0) {
    for (auto& c : clusters) step_subcycle(c, dt_base, ratio);
  }

  // Phi-driven lam transport — matches Python SubstrateLattice.step_lam().
  // Call after step() when winding transport is needed.
  void step_lam(AdaptiveCellCluster& cluster, double dt, double scale = 1.0) {
    auto const L = cluster.leaf_indices.size();
    if (!pool_) {
      _step_pass_lam(cluster, dt, scale, 0, L);
      return;
    }
    pool_->detach_blocks<std::size_t>(
        0, L, [&](std::size_t b, std::size_t e) { _step_pass_lam(cluster, dt, scale, b, e); });
    pool_->wait();
  }

  void step_lam_all(double dt, double scale = 1.0) {
    for (auto& c : clusters) step_lam(c, dt, scale);
  }

  // Conservative face-adjacent phi crossing — matches eq::substrate_step_cross
  // (inc/eq/substrate.hpp), the uniform C++ path. Three phases:
  //   1a (parallel): local drain + redistribute into own temp_phi slots
  //   1b (parallel): mirror corrections into neighbour temp_phi slots
  //   2  (parallel): apply temp_phi to phi
  // 1b's cross-cell writes go through _temp_phi_sub (atomic): on a uniform
  // lattice every slot has exactly one writer (the unique cell across that
  // face) so sums stay bit-for-bit deterministic; at AMR junctions a coarse
  // receiver takes writes from several fine siblings (and proxy distribution
  // fans out), where accumulation order — and thus rounding — depends on
  // thread schedule.
  //
  // Only INWARD phi (phi[k] < 0) drives redistribution — matches the uniform
  // path exactly. Each physical face crossing is processed once, from the
  // receiving cell's perspective only; processing both signs (as this used to)
  // double-counts the same crossing from both adjoining cells — confirmed via
  // tst/substrate/test_adaptive_vs_substrate.py::TestStepCrossSignConvention.
  //
  // delta is scaled by sigma(lam)^2 = (1/(1+lam))^2 per source cell — same
  // resistive throttle as the uniform C++ path, so a high-lam knot resists
  // topology-opening here too.
  //
  // AMR coarse-fine junctions follow the sum convention phi_coarse =
  // -Σ phi_fine (see _step_pass2_amr_correct): a fine cell mirrors 1:1 into
  // its coarser leaf neighbour, and a coarse cell whose nb is a refinement
  // proxy distributes its mirror across the proxy's leaf descendants
  // (_step_cross_mirror_proxy), so Σphi is conserved and every face pair
  // stays consistent across refinement boundaries.  Mixed fine-face signs
  // mean the coarse aggregate and individual fine faces can both fire on the
  // same junction in one call — each firing mirrors its own update, so the
  // pair books stay balanced.
  //
  // gate=CrossGate::Barrier (m0 required, no default — see doc/dst_next.md
  // sec 8): the only change vs V1 is an extra per-cell early-out inside
  // _step_cross_local/_step_cross_mirror — the pooled per-cell drive is
  // compared against m0 and the cell's crossing is skipped entirely (all 14
  // deltas 0) if it doesn't clear the threshold. The drain/mirror/apply
  // machinery, refinement-proxy distribution, and periodic-ghost handling
  // are gate-agnostic and unchanged — matches
  // equilibrium/substrate/lattice.py step_cross(gate="barrier") /
  // eq::substrate_step_cross_barrier (substrate.hpp) bit-for-bit on a
  // single-cluster (non-AMR, non-periodic) lattice.
  void step_cross(AdaptiveCellCluster& cluster, double dt, double scale = 1.0,
                   CrossGate gate = CrossGate::V1, double m0 = 0.0) {
    auto const L = cluster.leaf_indices.size();
    if (!pool_) {
      std::fill(cluster.temp_phi.begin(), cluster.temp_phi.end(), 0.0);
      std::fill(cluster.temp_phi_mirror.begin(), cluster.temp_phi_mirror.end(), 0.0);
      _step_cross_local(cluster, dt, scale, 0, L, gate, m0);
      _step_cross_mirror(cluster, cluster.g_to_j, cluster.g_max, dt, scale, L, gate, m0, 0, L);
      _step_cross_merge_mirror(cluster, 0, L);
      _step_cross_apply(cluster, 0, L);
      return;
    }
    auto* tp = cluster.temp_phi.data();
    auto* tpm = cluster.temp_phi_mirror.data();
    pool_->detach_blocks<std::size_t>(
        0, cluster.temp_phi.size(),
        [&](std::size_t b, std::size_t e) { std::fill(tp + b, tp + e, 0.0); });
    pool_->detach_blocks<std::size_t>(
        0, cluster.temp_phi_mirror.size(),
        [&](std::size_t b, std::size_t e) { std::fill(tpm + b, tpm + e, 0.0); });
    pool_->wait();
    // Firing cells cluster spatially — and zone-aware slab allocation makes
    // them contiguous in j — so one-block-per-thread partitioning hands most
    // of the real work to a single thread.  Oversubscribe blocks and let the
    // pool's queue balance them.
    auto const n_blocks = 16 * pool_->get_thread_count();
    pool_->detach_blocks<std::size_t>(0, L, [&](std::size_t b, std::size_t e) {
      _step_cross_local(cluster, dt, scale, b, e, gate, m0);
    }, n_blocks);
    pool_->wait();
    pool_->detach_blocks<std::size_t>(0, L, [&](std::size_t b, std::size_t e) {
      _step_cross_mirror(cluster, cluster.g_to_j, cluster.g_max, dt, scale, L, gate, m0, b, e);
    }, n_blocks);
    pool_->wait();
    pool_->detach_blocks<std::size_t>(0, L, [&](std::size_t b, std::size_t e) {
      _step_cross_merge_mirror(cluster, b, e);
    }, n_blocks);
    pool_->wait();
    pool_->detach_blocks<std::size_t>(
        0, L, [&](std::size_t b, std::size_t e) { _step_cross_apply(cluster, b, e); });
    pool_->wait();
  }

  void step_cross_all(double dt, double scale = 1.0, CrossGate gate = CrossGate::V1,
                       double m0 = 0.0) {
    for (auto& c : clusters) step_cross(c, dt, scale, gate, m0);
  }

  // Graduated volumetric sponge (quadratic_sponge BC). Call after step() / step_cross() /
  // step_lam(). dt is required for a rate sponge (vol_rate); ignored by the legacy vol_damp.
  void step_sponge(AdaptiveCellCluster& cluster, double dt = 0.0) {
    if (!cluster.bc || !cluster.bc->has_sponge()) return;
    if (!cluster.bc->vol_rate.empty() && !(dt > 0.0))
      throw std::invalid_argument("step_sponge: rate sponge needs dt > 0");
    auto const L = cluster.leaf_indices.size();
    if (!pool_) {
      _step_sponge_pass(cluster, 0, L, dt);
      return;
    }
    pool_->detach_blocks<std::size_t>(
        0, L, [&](std::size_t b, std::size_t e) { _step_sponge_pass(cluster, b, e, dt); });
    pool_->wait();
  }

  void step_sponge_all(double dt = 0.0) {
    for (auto& c : clusters) step_sponge(c, dt);
  }

  // -----------------------------------------------------------------------
  // step_boundary_hyper_resistivity() — k^4-selective damping at coarse-fine
  // (AMR) boundary cells only, the discrete analogue of the hyper-resistivity
  // term (-eta_h * Laplacian(J), here Laplacian(Laplacian(u))) added to
  // Ohm's law in PIC/MHD codes to remove grid-scale noise without touching
  // resolved (low-k) physics. Motivated by disable_same_level_flux_at_
  // boundary_only's finding that a boundary cell's own same-level flux acts
  // as the dominant conduit for a large, bounded-but-persistent H-growth
  // oscillation (see [[project_odd27_amr_junction_leak]]) whose root cause
  // is still open -- unlike the earlier per-formula diagnostic toggles, this
  // is a genuine physical mitigation (selectively absorb what the coarse
  // side can't resolve) rather than a bug hunt, so it's exposed as a real,
  // opt-in feature (eta_h defaults to 0.0, exact no-op) rather than a
  // temporary diagnostic.
  //
  // _leaf_laplacian(u) approximates the per-leaf discrete Laplacian via the
  // SAME face_weight/distance-scaling used by the primary wave step, but
  // restricted to leaf-to-leaf connections (a non-leaf proxy neighbour is
  // skipped -- a coarse proxy has no single well-defined local curvature of
  // its own to contribute here; this is a heuristic damping term, not the
  // primary physics, so the approximation is acceptable) and WITHOUT the
  // sigma/conductance factor (eta_h is an independent numerical parameter,
  // not coupled to the substrate's physical resistivity). Applying it twice
  // gives a discrete biharmonic-like quantity, maximally sensitive to the
  // highest-frequency (nearest-neighbour-alternating) component and
  // negligible for smooth, long-wavelength structure -- unlike a flat
  // relax-toward-parent sponge, which would damp genuine low-k differences
  // between a boundary cell and its coarse ancestor just as much as noise.
  //
  // Only applied to "boundary" leaves (>=1 cross-level face) -- interior
  // cells (away from any AMR junction) are left untouched.
  // Verified empirically (2026-07-21, static test against the EXACT field
  // u=x^2+y^2+z^2, whose continuum Laplacian is exactly 6 everywhere, no
  // truncation error): dividing by cell.L ONCE (the original version of
  // this function) gives a raw output that scales PROPORTIONAL TO L, not
  // constant -- 32 at L=1 (an unrefined L0 cell), 10.667 at L=1/3 (its
  // Odd27 child), i.e. exactly raw/L = 32 in BOTH cases. A genuine discrete
  // curvature (second-derivative) estimator must be L-INDEPENDENT (up to a
  // fixed geometric normalisation constant, here ~32/6, from the specific
  // BCC coordination shell) -- this is a basic Taylor-expansion fact, true
  // on ANY reasonably regular point set (square, hex, BCC...), NOT a
  // "square grid" assumption and not a refinement artefact; refinement just
  // means using each cell's own already-correctly-shrunk L, which was
  // already happening — the bug was the MISSING extra power of 1/L in the
  // formula itself. Dividing by cell.L*cell.L (not cell.L) makes the raw
  // output constant across cell size, confirmed by the same test. Original
  // (single /cell.L) version was copied from the primary flux formula's own
  // convention (a legitimately first-derivative-like quantity, correctly
  // needing only 1/L) without checking that THIS function's job — estimating
  // curvature, a second derivative — has a different dimensional
  // requirement; matching the primary formula's convention was the wrong
  // reference point.
  static std::vector<double> _leaf_laplacian(AdaptiveCellCluster& cl,
                                             std::vector<double> const& field) {
    auto const L = cl.leaf_indices.size();
    std::vector<double> out(L, 0.0);
    auto const& g_to_j = cl.g_to_j;
    auto const g_max   = cl.g_max;
    for (std::size_t j = 0; j < L; ++j) {
      auto const g    = cl.leaf_indices[j];
      auto const cell = cl[g];
      double acc = 0.0;
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        auto const nbg = cell.nb[k];
        if (nbg < 0 || nbg > g_max) continue;
        auto const j_nb = g_to_j[nbg];
        if (j_nb < 0) continue;  // non-leaf proxy neighbour: skip (see doc comment)
        acc += bcc::face_star[k] * (field[static_cast<std::size_t>(j_nb)] - field[j]) /
               (cell.L * cell.L);
      }
      out[j] = acc;
    }
    return out;
  }

  // Per-leaf sum of face_weight over LIVE leaf-to-leaf connections only
  // (same restriction as _leaf_laplacian) — the "degree" each leaf's own
  // Laplacian diagonal depends on. Needed for the IMEX diagonal-implicit
  // treatment below: D_i = (W_sum_i/L_i^2)^2 is the coefficient of u_i
  // itself in L2(u)_i = Laplacian(Laplacian(u))_i, derived by expanding
  // Laplacian(u)_i = -(W_sum_i/L_i^2)*u_i + sum_k W[k]/L_i^2*u_nb (the
  // corrected, properly-dimensioned /L^2 form — see _leaf_laplacian's doc
  // comment) and substituting into a second application of the same
  // operator — the only term that multiplies u_i (not some neighbour's u)
  // is (W_sum_i/L_i^2)*(W_sum_i/L_i^2)*u_i.
  static std::vector<double> _leaf_wsum(AdaptiveCellCluster& cl) {
    auto const L = cl.leaf_indices.size();
    std::vector<double> out(L, 0.0);
    auto const& g_to_j = cl.g_to_j;
    auto const g_max   = cl.g_max;
    for (std::size_t j = 0; j < L; ++j) {
      auto const g    = cl.leaf_indices[j];
      auto const cell = cl[g];
      double wsum = 0.0;
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        auto const nbg = cell.nb[k];
        if (nbg < 0 || nbg > g_max) continue;
        if (g_to_j[nbg] < 0) continue;
        wsum += bcc::face_star[k];
      }
      out[j] = wsum;
    }
    return out;
  }

  // eta_h is a BASE coefficient, scaled down per cell by 1/ratio^(4*level).
  // PHARE's ohm.hpp spatial_hyperresistive_ uses `1/4^levelNumber` (i.e.
  // 1/ratio^2 per level for ratio=2) -- but that compensates a SINGLE
  // Laplacian(J) term (second-order, needing 1/L^2 per level). This
  // function's L2(u) = Laplacian(Laplacian(u)) applies the (now correctly
  // /L^2-scaled, see _leaf_laplacian's doc comment) operator TWICE, making
  // it fourth-order (a genuine discrete biharmonic), which needs 1/L^4 per
  // level to compensate — i.e. 1/ratio^(4*level), not PHARE's 1/ratio^(2*level)
  // (copying PHARE's exponent directly, without checking it was derived for
  // a different-order operator, was an earlier mistake in this file's
  // history — see [[project_odd27_amr_junction_leak]]).
  //
  // A single flat coefficient has to be tiny enough to stay explicitly
  // stable at the FINEST level present, which starves it of any real effect
  // at shallower boundary levels (confirmed empirically: flat eta_h=1e-6
  // was already at the edge of instability at level 6, giving almost no
  // margin and no per-level tunability). Scaling by level lets a much
  // larger, more physically meaningful base coefficient stay stable
  // everywhere: shallow (large-L) boundary levels get a proportionally
  // larger effective coefficient, deep (small-L) levels automatically get a
  // correspondingly smaller one, tracking the actual explicit-stability
  // limit at each scale.
  //
  // Even with per-level scaling, the pure-explicit form (imex=false) was
  // found empirically to only DELAY the underlying explicit-instability
  // blowup, not eliminate it (stable base coefficients up to ~0.5 diverge
  // by ~step 16-22 of a 25-step run) — expected for an explicit biharmonic
  // term, whose stable timestep scales as ~1/L^4, far stricter than the
  // primary wave step's own CFL.
  //
  // imex=true switches to a diagonal-implicit (IMEX Euler / Jacobi-implicit)
  // treatment: the update is u_new = u_old - dt*eta_h_level*L2(u_new), which
  // for a general (non-diagonal) L2 would need a full linear solve — instead
  // this treats ONLY the self (diagonal) term implicitly and everything else
  // (the neighbour-coupling part) explicitly from u_old, a standard, cheap
  // way to remove exactly the stiffest (highest-wavenumber, most
  // self-dominated) part of a discrete biharmonic operator's instability
  // without needing sparse linear algebra:
  //   L2(u)_i = D_i*u_i + offdiag_i(u)   (D_i = (W_sum_i/L_i^2)^2, see _leaf_wsum)
  //   u_new_i - u_old_i = -dt*eta_h_level*(D_i*u_new_i + offdiag_i(u_old))
  //   u_new_i = (u_old_i - dt*eta_h_level*offdiag_i(u_old)) / (1 + dt*eta_h_level*D_i)
  // The division by (1 + dt*eta_h_level*D_i) is unconditionally stable with
  // respect to the diagonal part for ANY dt*eta_h_level*D_i >= 0 (matching
  // how implicit Euler is unconditionally stable for a plain decay ODE);
  // the remaining explicit off-diagonal coupling may still have some
  // residual stability constraint, but should be far less restrictive than
  // leaving the (typically dominant) diagonal term explicit too.
  // Shared core: eta_h_level_for(level) supplies the per-cell coefficient,
  // letting step_boundary_hyper_resistivity's base+exponent formula and
  // step_boundary_hyper_resistivity_per_level's direct per-level array share
  // one implementation.
  template <typename EtaHFn>
  void _step_boundary_hyper_resistivity_impl(AdaptiveCellCluster& cluster, double dt,
                                              EtaHFn&& eta_h_level_for, bool imex) {
    if (cluster.n_non_leaf_cells == 0) return;
    EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::step_boundary_hyper_resistivity");

    auto const L = cluster.leaf_indices.size();
    std::vector<double> u0(L);
    for (std::size_t j = 0; j < L; ++j) u0[j] = cluster[cluster.leaf_indices[j]].u;

    auto const lap1 = _leaf_laplacian(cluster, u0);
    auto const lap2 = _leaf_laplacian(cluster, lap1);
    std::vector<double> wsum;
    if (imex) wsum = _leaf_wsum(cluster);

    for (std::size_t j = 0; j < L; ++j) {
      auto const g    = cluster.leaf_indices[j];
      auto cell       = cluster[g];
      bool is_boundary = false;
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        auto const nbg = cell.nb[k];
        if (nbg >= 0 && cluster[nbg].L != cell.L) { is_boundary = true; break; }
      }
      if (!is_boundary || cell.fixed_u) continue;
      auto const eta_h_level = eta_h_level_for(cell.level);
      if (eta_h_level == 0.0) continue;
      if (imex) {
        auto const wl              = wsum[j] / (cell.L * cell.L);
        auto const D               = wl * wl;
        auto const explicit_offdiag = lap2[j] - D * u0[j];
        cell.u = (u0[j] - dt * eta_h_level * explicit_offdiag) /
                 (1.0 + dt * eta_h_level * D);
      } else {
        cell.u -= dt * eta_h_level * lap2[j];
      }
    }
  }

  void step_boundary_hyper_resistivity(AdaptiveCellCluster& cluster, double dt, double eta_h,
                                       double ratio = 3.0, bool imex = false) {
    if (eta_h == 0.0) return;
    _step_boundary_hyper_resistivity_impl(
        cluster, dt,
        [&](std::uint8_t level) { return eta_h / std::pow(ratio, 4.0 * level); }, imex);
  }

  void step_boundary_hyper_resistivity_all(double dt, double eta_h, double ratio = 3.0,
                                           bool imex = false) {
    for (auto& c : clusters) step_boundary_hyper_resistivity(c, dt, eta_h, ratio, imex);
  }

  // Per-level-tunable variant: eta_h_by_level[level] is used DIRECTLY, no
  // formula/scaling applied. Needed because a single base+exponent formula
  // cannot serve both shallow and deep boundary levels at once -- their
  // useful coefficient ranges differ by ~ratio^(4*max_level), confirmed
  // empirically to span ~11 orders of magnitude at ratio=3/L6 (see
  // project_odd27_amr_junction_leak Part 10). Levels at or beyond the
  // array's size get eta_h=0 (no correction, matching the default no-op).
  void step_boundary_hyper_resistivity_per_level(AdaptiveCellCluster& cluster, double dt,
                                                  std::vector<double> const& eta_h_by_level,
                                                  bool imex = false) {
    if (eta_h_by_level.empty()) return;
    _step_boundary_hyper_resistivity_impl(
        cluster, dt,
        [&](std::uint8_t level) {
          return static_cast<std::size_t>(level) < eta_h_by_level.size() ? eta_h_by_level[level]
                                                                          : 0.0;
        },
        imex);
  }

  void step_boundary_hyper_resistivity_per_level_all(double dt,
                                                      std::vector<double> const& eta_h_by_level,
                                                      bool imex = false) {
    for (auto& c : clusters)
      step_boundary_hyper_resistivity_per_level(c, dt, eta_h_by_level, imex);
  }

  // Ratio-parameterized variant: raw eta_h isn't directly comparable across
  // levels because the IMEX diagonal coefficient D_i = (W_sum_i/L_i^2)^2
  // itself grows as ~1/L^4 -- what actually governs the update's behaviour
  // is the dimensionless product s = dt*eta_h_level*D_i, not eta_h_level in
  // isolation (confirmed empirically at L8: eta_h_level values spanning
  // 1e-17 to 1e-13 corresponded to s ranging from negligible to ~856, with
  // the useful window sitting at s~8-10 and s>~100 causing INCREASING
  // instability, not just saturation -- see project_odd27_amr_junction_leak
  // Part 10's "dangerous?" discussion). Substituting eta_h_level =
  // s/(dt*D_i) into the IMEX update
  //   u_new = (u_old - dt*eta_h_level*offdiag) / (1 + dt*eta_h_level*D)
  // simplifies to the dt-independent
  //   u_new = (u_old - s*offdiag/D) / (1 + s)
  // -- s=0 is a no-op, s->inf is full replacement of the cell's own value
  // by its neighbour-geometric-extrapolation. Since D_i already carries the
  // full per-level 1/L^4 dependence, a SINGLE s may generalize across all
  // levels without per-level tuning -- this is what ratio_by_level (still
  // indexable per level, for generality/comparison) is meant to test.
  void step_boundary_hyper_resistivity_ratio(AdaptiveCellCluster& cluster,
                                              std::vector<double> const& ratio_by_level) {
    if (ratio_by_level.empty() || cluster.n_non_leaf_cells == 0) return;
    EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::step_boundary_hyper_resistivity_ratio");

    auto const L = cluster.leaf_indices.size();
    std::vector<double> u0(L);
    for (std::size_t j = 0; j < L; ++j) u0[j] = cluster[cluster.leaf_indices[j]].u;

    auto const lap1 = _leaf_laplacian(cluster, u0);
    auto const lap2 = _leaf_laplacian(cluster, lap1);
    auto const wsum = _leaf_wsum(cluster);

    for (std::size_t j = 0; j < L; ++j) {
      auto const g    = cluster.leaf_indices[j];
      auto cell       = cluster[g];
      bool is_boundary = false;
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        auto const nbg = cell.nb[k];
        if (nbg >= 0 && cluster[nbg].L != cell.L) { is_boundary = true; break; }
      }
      if (!is_boundary || cell.fixed_u) continue;
      auto const s = static_cast<std::size_t>(cell.level) < ratio_by_level.size()
                         ? ratio_by_level[cell.level]
                         : 0.0;
      if (s == 0.0) continue;
      auto const wl = wsum[j] / (cell.L * cell.L);
      auto const D  = wl * wl;
      if (D == 0.0) continue;
      auto const explicit_offdiag = lap2[j] - D * u0[j];
      cell.u = (u0[j] - s * explicit_offdiag / D) / (1.0 + s);
    }
  }

  void step_boundary_hyper_resistivity_ratio_all(std::vector<double> const& ratio_by_level) {
    for (auto& c : clusters) step_boundary_hyper_resistivity_ratio(c, ratio_by_level);
  }

  // BC dispatch pass — parallelisable over boundary_leaf_positions[begin, end).
  // Only processes boundary leaves (those with at least one nb[k] < 0).
  // Adds the gradient-BC contribution to temp_phi for boundary faces.
  // temp_phi[k*L+j] was initialised to 0 by _step_pass2's main loop; += preserves any prior contribution.
  //   Priority 1 — face_ghost:   treat the ghost cell as a real neighbour.
  //   Priority 2 — remote_flux:  directional Neumann; ghost u = u_i + flux*proj.
  //   Priority 3 — fn:           cluster-wide function pointer (sponge, dirichlet…).
  // Each worker writes only its own temp_phi[k*L+j] slots — no conflicts.
  static void _step_pass2_bc(AdaptiveCellCluster& cl, TimeStepper const& ts,
                             std::size_t begin, std::size_t end) {
    EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::_step_pass2_bc");

    auto const L           = cl.leaf_indices.size();
    auto const has_ghosts  = cl.bc->has_face_ghosts();
    auto const has_galactic = cl.bc->remote_flux != 0.0;
    auto const has_outflow = !cl.bc->outflow_active.empty();
    for (std::size_t bi = begin; bi < end; ++bi) {
      auto const j    = cl.boundary_leaf_positions[bi];
      auto const g    = cl.leaf_indices[j];
      auto const cell = cl[g];
      auto const dt   = ts(cell.L, cell.level);
      auto const u_i  = cell.u;
      auto const sig_i = cell.sigma();
      auto const L_i  = cell.L;
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        if (cell.nb[k] >= 0) continue;             // regular face; already handled
        if (cell.fixed_phi & (1u << k)) continue;  // frozen face

        double dphi = 0.0;

        // Priority 1: face ghost — real cell used as neighbour
        if (has_ghosts) {
          auto const ghost_g = cl.bc->find_face_ghost(g, k, cell.level);
          if (ghost_g >= 0) {
            auto const ghost    = cl[ghost_g];
            auto const dt_face  = ts.face_dt(L_i, cell.level, ghost.L, ghost.level);
            auto const sig_face = 0.5 * (sig_i + ghost.sigma());
            auto const w        = bcc::face_weight[k] * 2.0 / (L_i + ghost.L);
            dphi = -dt_face * sig_face * w * (ghost.u - u_i);
            cl.temp_phi[k * L + j] += dphi;
            continue;
          }
        }

        // Priority 2: directional Neumann — far-field remote-lattice gradient.
        // ghost u = u_i + remote_flux * dot(face_dir[k], remote_axis).
        // Perpendicular faces (proj=0) get dphi=0 (pure Neumann) automatically.
        if (has_galactic) {
          auto const& ax = cl.bc->remote_axis;
          auto const& fd = bcc::face_dirs[k];
          auto const proj    = fd[0]*ax[0] + fd[1]*ax[1] + fd[2]*ax[2];
          auto const u_ghost = u_i + cl.bc->remote_flux * proj;
          auto const w       = bcc::face_weight[k] / L_i;
          dphi           = -dt * sig_i * w * (u_ghost - u_i);
          cl.temp_phi[k * L + j] += dphi;
          continue;
        }

        // Priority 3: cluster-wide function pointer (sponge, dirichlet, absorbing…)
        if (cl.bc->fn) {
          auto const u_ghost = cl.bc->fn(u_i, cl.bc->target, cl.bc->weight);
          auto const w       = bcc::face_weight[k] / L_i;  // ghost same L as cell
          dphi = -dt * sig_i * w * (u_ghost - u_i);
          cl.temp_phi[k * L + j] += dphi;
          continue;
        }

        // Priority 4: outflow — first-order Mur/Sommerfeld one-way ghost (see
        // bc.hpp's outflow_active doc comment). ALSO summed into u via
        // _step_pass1 (unlike priorities 1-3), which is why it's gated
        // separately rather than folded into the fn dispatch above.
        if (has_outflow && cl.bc->outflow_active[k * L + j]) {
          auto const c_i = cl.bc->outflow_c.empty() ? sig_i * cl.bc->outflow_c_lattice
                                                     : cl.bc->outflow_c[j];
          auto const r          = (L_i - c_i * dt) / (L_i + c_i * dt);
          auto const u_i_old    = u_i - cl.temp_du[j];  // pass1 already applied du this step
          auto const ghost_prev = cl.bc->outflow_ghost[k * L + j];
          auto const u_ghost    = u_i_old + r * (ghost_prev - u_i);
          cl.bc->outflow_ghost[k * L + j] = u_ghost;
          auto const w = bcc::face_weight[k] / L_i;
          dphi = -dt * sig_i * w * (u_ghost - u_i);
        }

        cl.temp_phi[k * L + j] += dphi;
      }
    }
  }

  // Build face_multiplicity from scratch. Called from rebuild_leaf_indices.
  // See AdaptiveCellCluster::face_multiplicity's doc comment for why this
  // exists (Odd27's coarse-proxy approximation can send several of one
  // leaf's own faces to the exact same external target).
  //
  // Only computed for nb.L <= cell.L (same-level/non-substituted targets):
  // when nb.L > cell.L, ancestor_group_multiplicity's group sum (over ALL
  // (leaf, face) entries sharing that (ancestor, neighbour) pairing, which
  // already includes every one of THIS leaf's own duplicate faces to that
  // same target) is the complete correction — applying both would divide
  // the same redundancy twice. Left at the default 1 (no-op) there.
  //
  // A weight²-ratio, not a flat count, for the same reason
  // ancestor_group_multiplicity isn't a flat count either: Σ_dup
  // face_weight[k]² over the duplicate group, divided by the external
  // target's own face_weight² for whichever face points back at this leaf —
  // a flat count under/over-corrects whenever the duplicate group mixes hex
  // (weight 3) and sq (weight 1) faces, which Odd27's routing does often.
  static void rebuild_face_multiplicity(AdaptiveCellCluster& cl) {
    auto const L = cl.leaf_indices.size();
    std::vector<double> dense(CellSlab::NF * L, 1.0);
    if (cl.n_non_leaf_cells == 0 || L == 0) {
      cl.face_multiplicity.rebuild_from(dense);  // uniform mesh: no coarse-proxy aliasing possible
      return;
    }

    for (std::size_t j = 0; j < L; ++j) {
      auto const g    = cl.leaf_indices[j];
      auto const cell = cl[g];
      std::array<std::int32_t, CellSlab::NF> nbs{};
      for (std::size_t k = 0; k < CellSlab::NF; ++k) nbs[k] = cell.nb[k];
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        if (nbs[k] < 0 || cl[nbs[k]].L > cell.L) continue;
        double w2_sum = 0.0;
        for (std::size_t k2 = 0; k2 < CellSlab::NF; ++k2)
          if (nbs[k2] == nbs[k]) w2_sum += bcc::face_weight[k2] * bcc::face_weight[k2];
        auto const target_cell = cl[nbs[k]];
        double target_w2       = bcc::face_weight[k] * bcc::face_weight[k];  // fallback: reverse face not found
        for (std::size_t kk = 0; kk < CellSlab::NF; ++kk) {
          if (target_cell.nb[kk] == g) {
            target_w2 = bcc::face_weight[kk] * bcc::face_weight[kk];
            break;
          }
        }
        dense[k * L + j] = w2_sum / target_w2;
      }
    }
    cl.face_multiplicity.rebuild_from(dense);
  }

  // Build ancestor_group_multiplicity from scratch. Called from
  // rebuild_leaf_indices. See AdaptiveCellCluster::ancestor_group_multiplicity's
  // doc comment for why this exists (multiple leaves at the same level, all
  // descending from the same ancestor, independently substitute that
  // ancestor's u for their own when facing a much-coarser neighbour, so each
  // treats the same single (ancestor, neighbour) flux as its own private
  // exchange).
  static void rebuild_ancestor_group_multiplicity(AdaptiveCellCluster& cl) {
    auto const L = cl.leaf_indices.size();
    std::vector<double> dense(CellSlab::NF * L, 1.0);
    if (cl.n_non_leaf_cells == 0 || L == 0) {
      cl.ancestor_group_multiplicity.rebuild_from(dense);  // uniform mesh: no coarse-fine gap possible
      return;
    }

    EQUILIBRIUM_DEBUG_DO({
      if (std::getenv("EQ_DEBUG_SAMELEVEL_COUNT")) {
        long n_cross = 0, n_same_proxy = 0, n_same_leaf = 0;
        for (std::size_t j = 0; j < L; ++j) {
          auto const g    = cl.leaf_indices[j];
          auto const cell = cl[g];
          for (std::size_t k = 0; k < CellSlab::NF; ++k) {
            auto const nbg = cell.nb[k];
            if (nbg < 0) continue;
            auto nb = cl[nbg];
            if (nb.L > cell.L) ++n_cross;
            else if (nb.L == cell.L && nb.n_children > 0) ++n_same_proxy;
            else if (nb.L == cell.L) ++n_same_leaf;
          }
        }
        std::fprintf(stderr, "SAMELEVEL_COUNT n_cross=%ld n_same_proxy=%ld n_same_leaf=%ld\n",
                     n_cross, n_same_proxy, n_same_leaf);
      }
    })

    // Σ face_weight[k]² over every (leaf j, face k) entry sharing a given
    // (level, ancestor, neighbour) triple — NOT a plain entry count. Pass1's
    // contribution from a face is face_weight[k]·phi[k], and phi[k] itself is
    // proportional to face_weight[k] (see _step_pass2's `w`), so a face's
    // pull on Σu scales as face_weight[k]², not 1. A flat entry count is only
    // a correct divisor when every entry in the group shares one face class;
    // Odd27's coarse-proxy routing frequently mixes hex (weight 3) and sq
    // (weight 1) faces in the SAME group (confirmed directly: groups of
    // sibling children facing one external neighbour routinely split across
    // both classes), so counting entries instead of summing face_weight²
    // under- or over-corrects whenever a group is mixed.
    std::map<std::tuple<std::uint8_t, std::int32_t, std::int32_t>, double> weight2_sum;
    auto const find_pairing = [&](std::size_t j) {
      auto const g    = cl.leaf_indices[j];
      auto const cell = cl[g];
      std::array<std::pair<std::int32_t, std::int32_t>, CellSlab::NF> pairing{};
      pairing.fill({-1, -1});
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        auto const nbg = cell.nb[k];
        if (nbg < 0) continue;
        auto nb = cl[nbg];
        if (nb.L <= cell.L) continue;  // not a coarse-neighbour case: no substitution happens
        std::int32_t anc = cell.parent;
        while (anc >= 0 && cl[anc].L < nb.L) anc = cl[anc].parent;
        if (anc < 0) continue;
        pairing[k] = {anc, nbg};
        weight2_sum[{cell.level, anc, nbg}] += bcc::face_weight[k] * bcc::face_weight[k];
      }
      return pairing;
    };

    std::vector<std::array<std::pair<std::int32_t, std::int32_t>, CellSlab::NF>> pairings(L);
    for (std::size_t j = 0; j < L; ++j) pairings[j] = find_pairing(j);

    // The coarse neighbour's OWN face weight for a (ancestor, nbg) pair — the
    // single term nbg itself contributes on its side of the junction (nbg's
    // nb[] is untouched by the ancestor's split, so exactly one of its 14
    // faces still points at `anc`). Memoised per distinct (anc, nbg): this is
    // what the fine-side group's total must reduce to for Σu to balance.
    std::map<std::pair<std::int32_t, std::int32_t>, double> coarse_weight2;
    auto const coarse_face_weight2 = [&](std::int32_t anc, std::int32_t nbg) {
      auto const key = std::pair{anc, nbg};
      auto it        = coarse_weight2.find(key);
      if (it != coarse_weight2.end()) return it->second;
      auto const nb_cell = cl[nbg];
      double w2           = 1.0;  // fallback: no-op divisor if the reverse face is ever not found
      for (std::size_t kk = 0; kk < CellSlab::NF; ++kk) {
        if (nb_cell.nb[kk] == anc) {
          w2 = bcc::face_weight[kk] * bcc::face_weight[kk];
          break;
        }
      }
      coarse_weight2.emplace(key, w2);
      return w2;
    };

    for (std::size_t j = 0; j < L; ++j) {
      auto const g    = cl.leaf_indices[j];
      auto const cell = cl[g];
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        auto const [anc, nbg] = pairings[j][k];
        if (anc < 0) continue;
        auto const total = weight2_sum[{cell.level, anc, nbg}];
        auto const target = coarse_face_weight2(anc, nbg);
        dense[k * L + j] = total / target;
      }
    }
    cl.ancestor_group_multiplicity.rebuild_from(dense);
  }

  // Build amr_correction_pairs from scratch. Called from rebuild_leaf_indices.
  // Runs the slab scan + BFS once so _step_pass2_amr_correct is a flat loop.
  static void rebuild_amr_corrections(AdaptiveCellCluster& cl) {
    cl.amr_correction_pairs.clear();
    cl.amr_js.clear();
    cl.amr_js_face.clear();
    if (cl.n_non_leaf_cells == 0) return;

    auto const L = cl.leaf_indices.size();
    if (L == 0) return;
    auto const g_max = cl.g_max;
    auto const& g_to_j = cl.g_to_j;

    static thread_local std::vector<std::int32_t> tmp_g, tmp2_g;
    static thread_local std::vector<std::uint8_t> tmp_k, tmp2_k;

    // Does any of g's own 14 faces point at target_g? Used for the parent-
    // skip guard below in place of a single fixed-index comparison: under
    // Odd27 a child's face k does not necessarily route through the SAME
    // index k on its own parent (see _collect_leaves_facing_multi), so
    // checking only cl[par].nb[k] can miss that the parent DOES connect to
    // the same target via a different face — which would silently let a
    // child build its own (redundant) correction pair on top of the
    // parent's already-recursive one, double-applying the reflux.
    auto const has_face_to = [&](std::int32_t g, std::int32_t target_g) {
      auto const cell = cl[g];
      for (std::size_t kk = 0; kk < CellSlab::NF; ++kk)
        if (cell.nb[kk] == target_g) return true;
      return false;
    };

    for (std::size_t s = 0; s < cl.n_slabs(); ++s) {
      auto& slab = cl.slab(s);
      for (std::size_t i = 0; i < CellSlab::CAPACITY; ++i) {
        if (slab.parent[i] == -2 || slab.n_children[i] == 0) continue;
        auto const P_g = AdaptiveCellCluster::global(s, i);
        auto P = cl[P_g];

        // Coarse-fine: P (proxy) facing Q (leaf).
        for (std::size_t k = 0; k < CellSlab::NF; ++k) {
          auto const Q_g = P.nb[k];
          if (Q_g < 0 || Q_g > g_max) continue;
          auto const j_Q = g_to_j[Q_g];
          if (j_Q < 0) continue;
          auto const par = P.parent;
          if (par >= 0 && cl[par].n_children > 0 && has_face_to(par, Q_g)) continue;

          tmp_g.clear();
          tmp_k.clear();
          _collect_leaves_facing_multi(cl, P_g, Q_g, g_max, g_to_j, tmp_g, tmp_k);
          if (tmp_g.empty()) continue;

          auto& pair = cl.amr_correction_pairs.emplace_back();
          pair.a_only = true;
          pair.begin = static_cast<std::int32_t>(cl.amr_js.size());
          for (std::size_t ii = 0; ii < tmp_g.size(); ++ii) {
            cl.amr_js.push_back(g_to_j[tmp_g[ii]]);
            cl.amr_js_face.push_back(tmp_k[ii]);
          }
          pair.mid = static_cast<std::int32_t>(cl.amr_js.size());
          cl.amr_js.push_back(j_Q);
          cl.amr_js_face.push_back(bcc::mirror_face[k]);
          pair.end = static_cast<std::int32_t>(cl.amr_js.size());
        }

        // Proxy-proxy: P and P2 both non-leaf.
        for (std::size_t k = 0; k < CellSlab::NF; ++k) {
          auto const P2_g = P.nb[k];
          if (P2_g < 0) continue;
          auto const p2_is_proxy = (P2_g > g_max) || (g_to_j[P2_g] < 0);
          if (!p2_is_proxy) continue;
          if (P2_g <= P_g) continue;
          auto const par = P.parent;
          if (par >= 0 && cl[par].n_children > 0 && has_face_to(par, P2_g)) continue;
          auto P2 = cl[P2_g];
          if (P2.n_children == 0) continue;
          auto const par2 = P2.parent;
          if (par2 >= 0 && cl[par2].n_children > 0 && has_face_to(par2, P_g)) continue;

          tmp_g.clear();
          tmp_k.clear();
          tmp2_g.clear();
          tmp2_k.clear();
          _collect_leaves_facing_multi(cl, P_g, P2_g, g_max, g_to_j, tmp_g, tmp_k);
          _collect_leaves_facing_multi(cl, P2_g, P_g, g_max, g_to_j, tmp2_g, tmp2_k);
          if (tmp_g.empty() && tmp2_g.empty()) continue;

          auto& pair = cl.amr_correction_pairs.emplace_back();
          pair.a_only = false;
          pair.begin = static_cast<std::int32_t>(cl.amr_js.size());
          for (std::size_t ii = 0; ii < tmp_g.size(); ++ii) {
            cl.amr_js.push_back(g_to_j[tmp_g[ii]]);
            cl.amr_js_face.push_back(tmp_k[ii]);
          }
          pair.mid = static_cast<std::int32_t>(cl.amr_js.size());
          for (std::size_t ii = 0; ii < tmp2_g.size(); ++ii) {
            cl.amr_js.push_back(g_to_j[tmp2_g[ii]]);
            cl.amr_js_face.push_back(tmp2_k[ii]);
          }
          pair.end = static_cast<std::int32_t>(cl.amr_js.size());
        }
      }
    }
  }

  // Collect all (leaf descendant, own-face-index) pairs of proxy_g's subtree
  // where the leaf's OWN face points at target_g — i.e. every physical
  // connection into target_g, not just the ones that happen to share the
  // same face index as the top-level proxy-to-target direction. Recurses
  // through sub-proxies so multi-level interfaces are fully covered; a node
  // is only descended into if at least one of ITS OWN faces points at
  // target_g (geometrically sound: refining a node further only subdivides
  // its own interior/faces, it cannot create new adjacency to a region the
  // node wasn't already touching).
  //
  // Supersedes the old single-face_k _collect_leaves_facing for the
  // wave-step AMR correction path (rebuild_amr_corrections): under Odd27 a
  // descendant's own face k, when external, does not always proxy through
  // the SAME index k on the far side (a single child can even have more than
  // one of its own faces routing to the same external target, up to 4-way —
  // see face_multiplicity's doc comment), so matching by a single shared
  // face_k silently drops some real connections. A per-entry version was
  // tried once (2026-07-19) and reverted for being judged against the noisy
  // H-growth metric BEFORE face_multiplicity/ancestor_group_multiplicity
  // existed to normalize the underlying double-coupling; with those in
  // place, this is re-verified via the exact check_total_dphi_balanced
  // invariant instead. See [[project_odd27_amr_junction_leak]].
  static void _collect_leaves_facing_multi(AdaptiveCellCluster& cl, std::int32_t proxy_g,
                                           std::int32_t target_g, std::int32_t g_max,
                                           std::vector<std::int32_t> const& g_to_j,
                                           std::vector<std::int32_t>& out_g,
                                           std::vector<std::uint8_t>& out_k) {
    static thread_local std::vector<std::int32_t> stk;
    stk.clear();
    stk.push_back(proxy_g);
    while (!stk.empty()) {
      auto const pg = stk.back();
      stk.pop_back();
      auto const P  = cl[pg];
      auto const fc = P.first_child;
      auto const nc = P.n_children;
      for (std::uint8_t c = 0; c < nc; ++c) {
        auto const cg = fc + c;
        auto const cell = cl[cg];
        bool any_match = false;
        for (std::size_t k = 0; k < CellSlab::NF; ++k) {
          if (cell.nb[k] != target_g) continue;
          any_match = true;
          if (cell.n_children == 0) {
            if (cg <= g_max && g_to_j[cg] >= 0) {
              out_g.push_back(cg);
              out_k.push_back(static_cast<std::uint8_t>(k));
            }
          }
        }
        if (any_match && cell.n_children > 0) stk.push_back(cg);
      }
    }
  }

  // Collect all leaf descendants of proxy_g whose nb[face_k] == target_g.
  // Recurses through sub-proxies so multi-level interfaces are fully covered.
  //
  // KNOWN LIMITATION (Odd27): assumes a descendant's own face_k is the only
  // way it can route to target_g — false in general for Odd27 (see
  // _collect_leaves_facing_multi above, now used instead for the wave-step
  // AMR correction path). Still used as-is by _step_cross_mirror_proxy,
  // which reads/writes a single fixed face index `mk` across every collected
  // descendant by design (step_cross's own mirroring convention, a separate
  // mechanism from the wave-step reflux this file's AMR correction handles)
  // — left unchanged here; step_cross's AMR behaviour is a separate, already
  // known-deferred question (see [[project_step_cross]]).
  static void _collect_leaves_facing(AdaptiveCellCluster& cl, std::int32_t proxy_g,
                                     std::size_t face_k, std::int32_t target_g, std::int32_t g_max,
                                     std::vector<std::int32_t> const& g_to_j,
                                     std::vector<std::int32_t>& out) {
    static thread_local std::vector<std::int32_t> stk;
    stk.clear();
    stk.push_back(proxy_g);
    while (!stk.empty()) {
      auto const pg = stk.back();
      stk.pop_back();
      auto const P  = cl[pg];
      auto const fc = P.first_child;
      auto const nc = P.n_children;
      for (std::uint8_t c = 0; c < nc; ++c) {
        auto const cg = fc + c;
        if (cl[cg].nb[face_k] != target_g) continue;
        if (cl[cg].n_children == 0) {
          if (cg <= g_max && g_to_j[cg] >= 0) out.push_back(cg);
        } else {
          stk.push_back(cg);
        }
      }
    }
  }

  // Flux conservation correction — applied between pass 2 and pass 3.
  //
  // At every coarse-fine junction the proxy P is read by both sides:
  //   • fine leaf descendants of P use distance 2/(L_fine + L_coarse)
  //   • coarse neighbor Q uses distance 2/(L_Q + L_P)
  // Because L_fine ≠ L_Q the dphi pair does not cancel — net u is created.
  //
  // For each proxy P at any refinement depth:
  //   1. Coarse-fine: P.nb[k] = Q (leaf).  Collect all leaf descendants of P
  //      facing Q; balance their dphi sum against Q's dphi on mirror_k.
  //      Parent-skip guard prevents double-correction when an ancestor proxy
  //      already covers this junction.
  //   2. Proxy-proxy: P.nb[k] = P2 (both proxies).  Collect leaf descendants
  //      of P facing P2 and leaf descendants of P2 facing P; balance the pair.
  //      Deduplicated by P_g < P2_g; parent-skip guards on both sides.
  //
  // What must actually reach zero is the face_weight²-WEIGHTED sum, not the
  // raw phi sum: pass1's du is -dt·sigma·Σ_k face_weight[k]·phi[k], so a
  // face's pull on Σu scales with face_weight[k], not 1. Odd27 face-routing
  // routinely mixes hex (weight 3) and sq (weight 1) faces within one
  // correction group, so zeroing the raw sum (as before) does not zero the
  // weighted sum whenever a group is mixed — same root cause as
  // ancestor_group_multiplicity/face_multiplicity's flat-count bug (see
  // their doc comments). Fixed by weighting both the error and the
  // per-entry correction by 1/face_weight[jf[i]] (equal weighted-impact
  // redistribution, matching how face_multiplicity/ancestor_group_
  // multiplicity redistribute).
  static void _step_pass2_amr_correct(AdaptiveCellCluster& cl) {
    if (cl.n_non_leaf_cells == 0) return;
    EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::_step_pass2_amr_correct");

    auto const L   = cl.leaf_indices.size();
    auto const* js = cl.amr_js.data();
    auto const* jf = cl.amr_js_face.data();

    for (auto const& p : cl.amr_correction_pairs) {
      double werror = 0.0;
      for (std::int32_t i = p.begin; i < p.end; ++i)
        werror += bcc::face_weight[jf[i]] * cl.temp_phi[jf[i] * L + js[i]];
      if (werror == 0.0) continue;
      if (p.a_only) {
        // Coarse-fine: redistribute error across fine (A) side only, each
        // entry's own-phi correction scaled by 1/face_weight so every
        // entry's WEIGHTED contribution changes by the same amount.
        auto const n = static_cast<double>(p.mid - p.begin);
        auto const c = -werror / n;
        for (std::int32_t i = p.begin; i < p.mid; ++i)
          cl.temp_phi[jf[i] * L + js[i]] += c / bcc::face_weight[jf[i]];
      } else {
        // Proxy-proxy: redistribute across all cells on both sides.
        auto const n = static_cast<double>(p.end - p.begin);
        auto const c = -werror / n;
        for (std::int32_t i = p.begin; i < p.end; ++i)
          cl.temp_phi[jf[i] * L + js[i]] += c / bcc::face_weight[jf[i]];
      }
    }
  }

  // -----------------------------------------------------------------------
  // Register-based reflux for step_subcycle().
  //
  // _step_pass2_amr_correct (above) assumes both sides of a junction were
  // just computed in the SAME pass — true for step()'s single dt_base pass,
  // false under subcycling: a coarse-fine pair's Q-side (level lev_Q) and
  // fine side (level lev_Q+1, or deeper) are computed in DIFFERENT (lev, sub)
  // iterations of step_subcycle's loop, each at that level's own cadence.
  // Calling the plain correction once at lev==0 either (a) fires before the
  // fine side has run at all this outer step — reading a real sum_B against
  // a not-yet-computed sum_A==0, injecting a "placeholder" onto the fine
  // side's temp_phi that pass3 applies immediately — and then the fine side
  // ALSO computes and applies its own genuine contribution later at its own
  // lev, double-counting the same physical face; or (b) never fires at all
  // for junctions where neither side is at level 0 (e.g. a level-1/level-2
  // interface, or a proxy-proxy pair between two once-split root cells),
  // leaving those faces permanently uncorrected.
  //
  // Fix: accumulate each pair's sum_A/sum_B across every substep of the
  // whole outer step (every side applies its own honestly-computed temp_phi
  // via pass3 at its own cadence, unmodified), then apply ONE reflux
  // correction directly to phi after the last (finest) level has completed
  // — standard AMR "reflux at the end of the coarse timestep" pattern.
  // -----------------------------------------------------------------------
  static void _accumulate_amr_registers(AdaptiveCellCluster& cl, std::vector<double>& reg_a,
                                        std::vector<double>& reg_b) {
    if (cl.amr_correction_pairs.empty()) return;
    auto const L   = cl.leaf_indices.size();
    auto const* js = cl.amr_js.data();
    auto const* jf = cl.amr_js_face.data();
    for (std::size_t pi = 0; pi < cl.amr_correction_pairs.size(); ++pi) {
      auto const& p = cl.amr_correction_pairs[pi];
      double sum_A = 0.0;
      for (std::int32_t i = p.begin; i < p.mid; ++i) sum_A += cl.temp_phi[jf[i] * L + js[i]];
      double sum_B = 0.0;
      for (std::int32_t i = p.mid; i < p.end; ++i) sum_B += cl.temp_phi[jf[i] * L + js[i]];
      reg_a[pi] += sum_A;
      reg_b[pi] += sum_B;
    }
  }

  // Apply the whole-outer-step accumulated correction directly to phi.
  // temp_phi has already been fully consumed (by pass3, every substep) by
  // the time this runs, so the correction is written straight to cl phi.
  //
  // Distributes the junction's aggregate correction across each side's cells
  // weighted by that cell's own |phi| on the relevant face — NOT a flat
  // equal split. An equal split is exactly zero-sum (cancels the junction's
  // aggregate imbalance) but spatially blind: applied identically regardless
  // of each cell's own local state, it can systematically misallocate the
  // correction when a junction group spans a geometrically diverse set of
  // fine cells (as Odd27's wider per-split spread does), smearing structure
  // near the junction over many steps without ever showing up in the
  // aggregate imbalance check. Weighting by |phi| — same distribution
  // principle _step_cross_mirror_proxy already uses for an analogous
  // "spread a correction across several fine faces" problem — puts more of
  // the correction on cells that were already carrying more flux there,
  // falling back to an equal split only when every candidate face is at
  // exactly zero (matches _step_cross_mirror_proxy's own fallback).
  static void _apply_amr_reflux(AdaptiveCellCluster& cl, std::vector<double> const& reg_a,
                                std::vector<double> const& reg_b) {
    if (cl.amr_correction_pairs.empty()) return;
    auto const* js = cl.amr_js.data();
    auto const* jf = cl.amr_js_face.data();

    auto distribute = [&](std::int32_t begin, std::int32_t end, double error, double wsum,
                          std::int32_t n_free) {
      for (std::int32_t i = begin; i < end; ++i) {
        auto cell = cl[cl.leaf_indices[js[i]]];
        auto const face_k = jf[i];
        if (cell.fixed_phi & (1u << face_k)) continue;
        double const w = wsum > 0.0 ? std::abs(cell.phi[face_k]) / wsum
                                    : (n_free > 0 ? 1.0 / static_cast<double>(n_free) : 0.0);
        cell.phi[face_k] += -error * w;
      }
    };

    for (std::size_t pi = 0; pi < cl.amr_correction_pairs.size(); ++pi) {
      auto const& p     = cl.amr_correction_pairs[pi];
      auto const error  = reg_a[pi] + reg_b[pi];
      if (error == 0.0) continue;
      if (p.a_only) {
        double wsum = 0.0;
        std::int32_t n_free = 0;
        for (std::int32_t i = p.begin; i < p.mid; ++i) {
          auto cell = cl[cl.leaf_indices[js[i]]];
          if (cell.fixed_phi & (1u << jf[i])) continue;
          wsum += std::abs(cell.phi[jf[i]]);
          ++n_free;
        }
        distribute(p.begin, p.mid, error, wsum, n_free);
      } else {
        double wsum = 0.0;
        std::int32_t n_free = 0;
        for (std::int32_t i = p.begin; i < p.mid; ++i) {
          auto cell = cl[cl.leaf_indices[js[i]]];
          if (cell.fixed_phi & (1u << jf[i])) continue;
          wsum += std::abs(cell.phi[jf[i]]);
          ++n_free;
        }
        for (std::int32_t i = p.mid; i < p.end; ++i) {
          auto cell = cl[cl.leaf_indices[js[i]]];
          if (cell.fixed_phi & (1u << jf[i])) continue;
          wsum += std::abs(cell.phi[jf[i]]);
          ++n_free;
        }
        distribute(p.begin, p.mid, error, wsum, n_free);
        distribute(p.mid, p.end, error, wsum, n_free);
      }
    }
  }

  // Vectorised sq-face update for all-leaf sibling groups.
  //
  // Gathers the 8 sibling u values into a stack array, computes the 12 internal
  // sq-face dphi values (3 axes × 4 pairs each), and writes them to temp_phi.
  // Called after _step_pass2 / _step_pass2_lev so it overwrites only the 12
  // intra-group entries that the main pass handled via indirect nb[] lookup.
  //
  // The stack gather lets the compiler vectorise the compute even though the
  // SoA u[] values are in scattered slab slots.  The temp_phi writes land on
  // j0..j0+7 — contiguous by rebuild order — enabling AVX2 stores.
  // level_exact < 0 (default): process every sibling group unconditionally —
  // used by the non-subcycling step() path, which has no per-level cadence.
  // level_exact >= 0: used by step_subcycle(), which calls this once per
  // (level, substep) with that level's own dt_lev; without the filter, a
  // group at a level other than level_exact would get its 12 intra-group
  // temp_phi entries overwritten (and then permanently applied by the
  // following _step_pass3) with a dt sized for the WRONG level's cadence —
  // e.g. a level-1-only mesh still runs step_subcycle's lev=0 iteration once
  // (dt_lev=dt_base, matching the coarsest possible level even though no
  // leaf is actually level 0), which would otherwise apply a full dt_base
  // sibling correction on top of the two correct dt_base/2 corrections lev=1
  // itself applies — double-counting the sibling exchange 2x per outer step.
  static void _step_pass2_siblings(AdaptiveCellCluster& cl, double dt,
                                   std::int8_t level_exact = -1) {
    if (cl.sibling_groups.empty()) return;
    EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::_step_pass2_siblings");

    auto const L = cl.leaf_indices.size();

    for (auto const& sg : cl.sibling_groups) {
      auto const fc     = sg.first_child;
      auto const j0     = sg.j0;
      auto const cell_L = cl[fc].L;  // all siblings same L
      if (level_exact >= 0 && cl[fc].level != static_cast<std::uint8_t>(level_exact)) continue;

      // Gather u and sigma into stack arrays — 8 reads, then pure register math.
      double u8[8], s8[8];
      for (int c = 0; c < 8; ++c) {
        auto ch = cl[fc + c];
        u8[c]  = ch.u;
        s8[c]  = ch.sigma();
      }

      // 3 axes × 4 pairs = 12 internal sq-face pairs.
      for (int j = 0; j < 3; ++j) {
        auto const f_neg = bcc::sq_face_for_axis[j][0];
        auto const f_pos = bcc::sq_face_for_axis[j][1];
        auto const w     = bcc::face_weight[f_neg] * 2.0 / (cell_L + cell_L);

        // iterate the 4 children with negative axis-j component (bit (2-j) == 0)
        auto const bit = 2 - j;
        for (int c = 0; c < 8; ++c) {
          if (c & (1 << bit)) continue;
          auto const sib   = bcc::flip_child(static_cast<std::uint8_t>(c), j);
          auto const sig_f = 0.5 * (s8[c] + s8[sib]);
          auto const dphi  = -dt * sig_f * w * (u8[sib] - u8[c]);
          // c (negative side) → f_pos face points toward sib
          cl.temp_phi[f_pos * L + j0 + c]   =  dphi;
          // sib (positive side) → f_neg face points toward c
          cl.temp_phi[f_neg * L + j0 + sib] = -dphi;
        }
      }
    }
  }

 private:
  // Level-filtered pass 1: only update u for cells at level == level_exact.
  // Called by step_subcycle().  temp_phi must be zeroed before this call.
  static void _step_pass1_lev(AdaptiveCellCluster& cl, double dt,
                               uint8_t level_exact, std::size_t begin, std::size_t end) {
    EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::_step_pass1_lev");
    auto const L = cl.leaf_indices.size();
    for (std::size_t j = begin; j < end; ++j) {
      auto const g    = cl.leaf_indices[j];
      auto const cell = cl[g];
      if (cell.level != level_exact) continue;
      double acc = 0.0;
      // Periodic faces wired via set_nb appear as nb[k] >= 0 — no special casing.
      // Uses the SAME per-face coefficient (`_face_flux_coefficient`,
      // including ancestor-substitution L_src) that `_step_pass2_lev` used to
      // build this phi[k] value in the first place — a mismatched coefficient
      // here breaks both H and Σu conservation at real junctions (see `now`,
      // 2026-08-14). nbg_r is always just nbg at this call site — see
      // `_resolve_face_neighbour`.
      //
      // Multiplicity is deliberately NOT reapplied here: pass2 divides dphi
      // by face_multiplicity/ancestor_group_multiplicity when constructing
      // phi[k], so phi[k] is already this face's correctly-scaled private
      // share of a fan-in junction's total flux. Dividing again here would
      // double-count that correction (confirmed empirically 2026-08-15:
      // applying it in pass1 too made the coarse-fine/common-denominator
      // drift WORSE — larger and sign-flipped — not better).
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        auto const nbg = cell.nb[k];
        if (nbg >= 0) {
          auto const L_src = _ancestor_L_src(cl, cell, nbg, /*respect_disable_gate=*/true);
          auto const coef  = _face_flux_coefficient(cl, cell, k, nbg, L_src,
                                                     /*apply_multiplicity=*/false, L, j);
          acc += coef * cell.phi[k];
        }
      }
      auto const du = -dt * acc;
      cl.temp_du[j] = du;
      if (!cell.fixed_u) cell.u += du;
    }
  }

  // Temporary diagnostic (2026-07-19): true recursive mean-of-sigma over a
  // non-leaf cell's leaf descendants -- Jensen-correct, unlike nb.sigma()
  // which computes sigma(mean(lam)) from the bottom-up-averaged lam field.
  // Used only when use_recursive_mean_sigma is set; testing whether this
  // asymmetry (a proxy neighbour's sig_face contribution being biased
  // relative to a genuine-leaf neighbour's) is a factor in the deep-
  // telescope growth. Not a permanent feature.
  static double _mean_sigma_recursive(AdaptiveCellCluster& cl, std::int32_t g) {
    auto cell = cl[g];
    if (cell.n_children == 0) return cell.sigma();
    double sum = 0.0;
    for (std::uint8_t c = 0; c < cell.n_children; ++c)
      sum += _mean_sigma_recursive(cl, cell.first_child + c);
    return sum / cell.n_children;
  }

  // Resolve a face's runtime neighbour target, mirroring the identical stale-
  // proxy substitution `_step_pass2` (flat) performs before computing dphi --
  // pass1 must land on the SAME nbg_r so its per-face coefficient (see
  // `_face_flux_coefficient` below) pairs with the phi[k] value pass2 stored
  // for that exact target, not some other stale one. Hex faces only (k<8):
  // sq faces have genuine multi-child fan-in even once both sides are split
  // to the same level (see `resolve_stale_external_proxies`'s own comment).
  // `_step_pass2_lev`/`_step_pass1_lev` never call this -- resolution there
  // is fully handled ahead of time by `resolve_stale_external_proxies`
  // (called from rebuild_leaf_indices), so nbg_r is always just nbg.
  static std::int32_t _resolve_face_neighbour(AdaptiveCellCluster& cl, std::int32_t g,
                                               AdaptiveCellRef const& cell, std::size_t k,
                                               std::int32_t nbg) {
    auto nbg_r = nbg;
    if (!disable_resolve_stale_proxy_neighbour && k < 8 && cell.level > 0 &&
        cl[nbg].n_children > 0 && cl[nbg].level == static_cast<std::uint8_t>(cell.level - 1)) {
      double tp[3];
      _neighbour_pos(cl, g, k, tp);
      auto const part = _descend_to(cl, nbg, cell.level, tp);
      if (part >= 0 && cl[part].level == cell.level && cl[part].n_children == 0) nbg_r = part;
    }
    return nbg_r;
  }

  // Ancestor-substitution's L: when the neighbour is coarser, walk this
  // cell's own parent chain to the ancestor at the neighbour's level -- the
  // SAME ancestor pass2 substitutes u_src from. Only L is needed here (not
  // u_src itself): pass1 uses the persistent phi[k] state directly, so only
  // the coefficient (which depends on L_src, not u_src) must match pass2's.
  // `respect_disable_gate` mirrors whichever pass2 variant this pairs with --
  // `_step_pass2_lev` gates this walk on `disable_ancestor_u_src`, the flat
  // `_step_pass2` does not (a pre-existing asymmetry between the two
  // variants, left as-is rather than changed as a side effect of this fix).
  static double _ancestor_L_src(AdaptiveCellCluster& cl, AdaptiveCellRef const& cell,
                                 std::int32_t nbg_r, bool respect_disable_gate) {
    auto const nb = cl[nbg_r];
    double L_src = cell.L;
    if ((!respect_disable_gate || !disable_ancestor_u_src) && nb.L > cell.L) {
      auto anc = cell.parent;
      while (anc >= 0 && cl[anc].L < nb.L) anc = cl[anc].parent;
      if (anc >= 0) L_src = cl[anc].L;
    }
    return L_src;
  }

  // Per-face flux coefficient: sig_face * w (face_weight[k] folded into w),
  // after the multiplicity divisor and cross_level_flux_scale that pass2
  // applies to dphi. Pass2 multiplies this against (nb_u_for_flux - u_src)
  // to get dphi; pass1 multiplies it directly against the persistent phi[k]
  // state to get its contribution to du. Sharing this one computation is
  // what keeps pass1/pass2 a matched symplectic pair at every junction
  // topology (direct same-level, ancestor-substituted, or multiplicity
  // fan-in) instead of just direct same-level faces -- see `now`, 2026-08-14.
  static double _face_flux_coefficient(AdaptiveCellCluster& cl, AdaptiveCellRef const& cell,
                                        std::size_t k, std::int32_t nbg_r, double L_src,
                                        bool apply_multiplicity, std::size_t L, std::size_t j) {
    auto const nb        = cl[nbg_r];
    auto const nb_sigma   = (use_recursive_mean_sigma && nb.n_children > 0)
                                ? _mean_sigma_recursive(cl, nbg_r)
                                : nb.sigma();
    auto const sig_face = 0.5 * (cell.sigma() + nb_sigma);
    auto const w         = bcc::face_weight[k] * 2.0 / (L_src + nb.L);
    double coef = sig_face * w;
    if (!disable_multiplicity_correction && apply_multiplicity) {
      coef /= cl.face_multiplicity.get(k, L, j);
      coef /= cl.ancestor_group_multiplicity.get(k, L, j);
    }
    if (cross_level_flux_scale != 1.0 && nb.L != cell.L) coef *= cross_level_flux_scale;
    return coef;
  }

  // Level-filtered pass 2: compute dphi only for cells at level == level_exact.
  // Non-level cells are skipped; their temp_phi remains 0 (set by caller).
  static void _step_pass2_lev(AdaptiveCellCluster& cl, double dt,
                               uint8_t level_exact, std::size_t begin, std::size_t end) {
    EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::_step_pass2_lev");
    auto const L = cl.leaf_indices.size();
    for (std::size_t j = begin; j < end; ++j) {
      auto const g    = cl.leaf_indices[j];
      auto const cell = cl[g];
      if (cell.level != level_exact) continue;
      auto const u_i   = cell.u;
      auto const L_i   = cell.L;
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        double dphi = 0.0;
        auto const nbg = cell.nb[k];
        if (nbg >= 0 && !(cell.fixed_phi & (1u << k))) {
          // NOTE (2026-08-12): a per-face runtime resolution used to live
          // here (mirroring resolve_stale_external_proxies below), but it
          // lacked that function's collision detection -- a stateless,
          // per-(g,k) check can't see whether some OTHER cell already claims
          // the same target this step, so it could reintroduce exactly the
          // ambiguous-partner bug the persistent, collision-safe pass exists
          // to avoid. Removed; resolve_stale_external_proxies (called from
          // rebuild_leaf_indices, before every step and on every checkpoint
          // load) is now the sole mechanism -- by the time this flux pass
          // runs, any face it could safely resolve already has been, so
          // nbg_r is always just nbg here.
          auto const nbg_r = nbg;
          auto nb = cl[nbg_r];
          double u_src = u_i;
          double L_src = L_i;
          std::int32_t anc = -1;
          if (!disable_ancestor_u_src && nb.L > L_i) {
            anc = cell.parent;
            while (anc >= 0 && cl[anc].L < nb.L) anc = cl[anc].parent;
            if (anc >= 0) {
              u_src = enable_ancestor_debias_correction ? _debiased_proxy_u(cl, anc) : cl[anc].u;
              L_src = cl[anc].L;
            }
          }
          auto nb_u_for_flux = (enable_ancestor_debias_correction && nb.n_children > 0)
                                   ? _debiased_proxy_u(cl, nbg_r)
                                   : nb.u;
          if (enable_cross_level_gradient_correction && nb.L != L_i) {
            double p0[3], p1[3];
            _cell_pos(cl, g, p0);
            _cell_pos(cl, nbg_r, p1);
            auto const fdx = p1[0] - p0[0], fdy = p1[1] - p0[1], fdz = p1[2] - p0[2];
            auto const flen = std::sqrt(fdx * fdx + fdy * fdy + fdz * fdz);
            double const face[3] = {p0[0] + 0.5 * L_i * fdx / flen,
                                    p0[1] + 0.5 * L_i * fdy / flen,
                                    p0[2] + 0.5 * L_i * fdz / flen};
            nb_u_for_flux = _gradient_corrected_value(cl, nbg_r, face);
            if (anc >= 0) u_src = _gradient_corrected_value(cl, anc, face);
          }
          // face_multiplicity/ancestor_group_multiplicity were built from the
          // STORED (possibly stale) nb[k] topology -- they correct for real
          // fan-in at a genuine coarse-fine junction (several fine faces
          // sharing one coarse target). Once stale-proxy resolution above has
          // found a genuine same-level 1:1 partner, this face is not that
          // junction anymore (multiplicity 1 by construction) -- applying the
          // stale divisor here would attenuate a now-correct flux by a factor
          // that reflects the OLD wiring, not this face's real fan-in.
          auto const coef = _face_flux_coefficient(cl, cell, k, nbg_r, L_src, nbg_r == nbg, L, j);
          dphi = -dt * coef * (nb_u_for_flux - u_src);
          EQUILIBRIUM_DEBUG_DO({
            if (auto const* trace_env = std::getenv("EQ_DEBUG_CELL_TRACE"); trace_env && g == std::atoi(trace_env)) {
              std::fprintf(stderr,
                  "CELL_TRACE g=%d k=%zu nbg=%d nb.L=%.6g L_i=%.6g same_level=%d u_i=%.10g "
                  "nb.u=%.10g u_src=%.10g coef=%.10g dphi=%.10g fmult=%d agmult=%d\n",
                  g, k, nbg, nb.L, L_i, nb.L == L_i, u_i, nb.u, u_src, coef, dphi,
                  static_cast<int>(cl.face_multiplicity.get(k, L, j)),
                  static_cast<int>(cl.ancestor_group_multiplicity.get(k, L, j)));
            }
          })
        }
        cl.temp_phi[k * L + j] = dphi;
      }
    }
  }

  // Refresh all non-leaf cells (coarse-fine interface proxies) so their u/phi
  // are current before the step passes read them as neighbours.  Iterates
  // level_slabs finest-to-coarsest so children are always accumulated before
  // their parents.  No allocation; early-exits when the lattice is uniform.
  void _accumulate_boundary_proxies(AdaptiveCellCluster& cluster) {
    if (cluster.n_non_leaf_cells == 0) return;
    EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::_accumulate_boundary_proxies");
    accumulate_all(cluster);
  }

  // Lam transport: Δlam = -dt·scale·σ·lam·Σ W[k]·phi[k]
  // Single pass — each cell reads its own lam and phi only (no cross-cell dependency).
  // fixed_u cells are skipped (lam frozen, matching Python step_lam behaviour).
  // lam clamped to [0, LAM_MAX] after update.
  static void _step_pass_lam(AdaptiveCellCluster& cl, double dt, double scale,
                              std::size_t begin, std::size_t end) {
    constexpr double LAM_MAX = DiscreteCell::N_SAGA - 1.0;
    for (std::size_t j = begin; j < end; ++j) {
      auto const g    = cl.leaf_indices[j];
      auto cell = cl[g];
      if (cell.fixed_u) continue;
      auto const li = cell.lam;
      if (li == 0.0) continue;
      auto const si = 1.0 / (1.0 + li);
      double acc = 0.0;
      for (std::size_t k = 0; k < CellSlab::NF; ++k)
        if (cell.nb[k] >= 0) acc += bcc::face_weight[k] * cell.phi[k];
      auto const frac = -dt * scale * si * acc;  // dlam / lam
      auto const nl = li + frac * li;
      cell.lam = nl < 0.0 ? 0.0 : (nl > LAM_MAX ? LAM_MAX : nl);
    }
  }

  // Leaf index of face k's neighbour, -1 if none or not a leaf. Face-shared BC quantities are
  // formed from both copies (j, k) and (jn, mirror_face[k]) so phi_ki = -phi_ik survives.
  template <typename Cell>
  static std::int32_t _leaf_nb(AdaptiveCellCluster const& cl, Cell const& cell, std::size_t k) {
    auto const nbg = cell.nb[k];
    return (nbg >= 0 && nbg <= cl.g_max) ? cl.g_to_j[static_cast<std::size_t>(nbg)] : std::int32_t{-1};
  }

  static double _face_mean(std::vector<double> const& per_leaf, std::size_t j, std::int32_t jn) {
    return jn < 0 ? per_leaf[j] : 0.5 * (per_leaf[j] + per_leaf[static_cast<std::size_t>(jn)]);
  }

  // per_face is k*L+j; the partner copy is (mirror_face[k], jn).
  static double _face_mean_kj(std::vector<double> const& per_face, std::size_t L, std::size_t k,
                              std::size_t j, std::int32_t jn) {
    return jn < 0 ? per_face[k * L + j]
                  : 0.5 * (per_face[k * L + j] +
                           per_face[bcc::mirror_face[k] * L + static_cast<std::size_t>(jn)]);
  }

  // Volumetric sponge pass (BoundaryCondition::vol_rate / vol_damp): u relaxes toward the target
  // per leaf; each face copy is scaled by the factor shared with its partner copy.
  // fixed_u / fixed_phi bits are honoured.
  static void _step_sponge_pass(AdaptiveCellCluster& cl, std::size_t begin, std::size_t end,
                                double dt) {
    EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::_step_sponge_pass");
    auto const& bc = *cl.bc;
    auto const g_max = cl.g_max;
    for (std::size_t j = begin; j < end; ++j) {
      auto const g = cl.leaf_indices[j];
      auto cell    = cl[g];
      if (!cell.fixed_u) cell.u = bc.target + (cell.u - bc.target) * bc.sponge_leaf(j, dt);
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        if (cell.fixed_phi & (1u << k)) continue;
        auto const nbg = cell.nb[k];
        auto const jn  = (nbg >= 0 && nbg <= g_max) ? cl.g_to_j[static_cast<std::size_t>(nbg)]
                                                    : std::int32_t{-1};
        cell.phi[k] *= bc.sponge_face(j, jn, dt);
      }
    }
  }


  // A face with nb[k] < 0 is invisible to the wave step (_step_pass1/2/3 all
  // gate on nb[k] >= 0) — either a genuine domain edge, or an AMR-inherited
  // "dead" direction: wire_split_children sets the one child whose octant
  // faces the parent's own interior to nb[k] = -1, and EVERY further split
  // generation re-inherits that -1 unchanged into all non-adjacent
  // descendants (wire_split_children step 3/4 — "all other children also
  // inherit par.nb[k]").  A cell refined N generations deep can therefore
  // carry up to N such inert directions, one contributed by each ancestor —
  // not just its own immediate one.  step_cross must neither fire a
  // crossing INTO these faces (their phi is stale prolongation residue: the
  // wave step never updates it) nor redistribute a boost INTO them (real
  // flux parked where nothing ever reads it again) — matches this function
  // treating nb[k] < 0 identically to how the wave step already does.
  //
  // Periodic ghosts are the one case where nb[k] < 0 does NOT mean inert
  // (a ghost ring or box wrap resolves it) — _step_cross_mirror already
  // handles that separately via find_face_ghost, so this local pass treats
  // every face as potentially live whenever ghosts exist, deferring the
  // real liveness call to the mirror pass exactly as before.
  // gate/m0: see CrossGate — CrossGate::Barrier adds a per-cell pooled-drive
  // early-out (drive = Σ_k W_k·max(-phi[k],0) over live faces, same `live()`
  // definition used below so periodic-ghost faces still count); everything
  // else in this function is gate-agnostic.
  static void _step_cross_local(AdaptiveCellCluster& cl, double dt, double scale,
                                std::size_t begin, std::size_t end,
                                CrossGate gate = CrossGate::V1, double m0 = 0.0) {
    auto const L          = cl.leaf_indices.size();
    auto const has_ghosts = cl.bc && cl.bc->has_face_ghosts();
    for (std::size_t j = begin; j < end; ++j) {
      auto const g    = cl.leaf_indices[j];
      auto const cell = cl[g];
      auto const si2  = cell.sigma() * cell.sigma();
      auto const live = [&](std::size_t face) { return cell.nb[face] >= 0 || has_ghosts; };
      if (gate == CrossGate::Barrier) {
        double drive = 0.0;
        for (std::size_t k = 0; k < CellSlab::NF; ++k) {
          if (!live(k)) continue;
          auto const inward_k = cell.phi[k] < 0.0 ? -cell.phi[k] : 0.0;  // = max(-phi[k],0)
          drive += bcc::face_weight[k] * inward_k;
        }
        if (drive <= m0) continue;  // gate closed -- cell contributes nothing
      }
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        if (!live(k)) continue;
        auto const inward = cell.phi[k] < 0.0 ? cell.phi[k] : 0.0;
        if (inward == 0.0) continue;
        // dt is calibrated for L=1 (L0) cells (see TimeStepper::cfl_fn /
        // by_level_fn: dt halves each time L halves) — step_cross previously
        // applied the flat L0 dt to every cell regardless of level, so a
        // level-2 cell (L=1/4) got 4x the CFL-stable "cost" per crossing.
        // Scaling by cell.L reproduces the same per-level fraction. See
        // `step_cross_level_scale_power`'s doc comment for when this factor
        // is redundant (caller already passes a per-mesh-matched dt).
        auto const l_scale = step_cross_level_scale_power == 1.0
                                 ? cell.L
                                 : (step_cross_level_scale_power == 0.0
                                        ? 1.0
                                        : std::pow(cell.L, step_cross_level_scale_power));
        auto const delta = -dt * l_scale * scale * si2 * inward;  // >= 0
        int n_adj = 0;
        for (int m = 0; m < 6 && bcc::face_adj[k][m] >= 0; ++m)
          if (live(static_cast<std::size_t>(bcc::face_adj[k][m]))) ++n_adj;
        if (!(cell.fixed_phi & (1u << k))) {
          // Own-face term only (not the adj-boost below): face k's nb[] can
          // be one of several of this leaf's OWN faces the Odd27 coarse-proxy
          // approximation routes to the SAME external target (e.g. a fine
          // leaf's faces {3,8,11,13} all aliasing one coarser neighbour), or
          // one of several SIBLING leaves under one proxy independently
          // claiming that same external neighbour — exactly what
          // face_multiplicity/ancestor_group_multiplicity already divide out
          // for the wave step (_step_pass2). step_cross had no equivalent
          // correction: an aliased face's own crossing fired a full,
          // undivided mirror straight at the shared external target, once
          // per alias, over-representing a single physical connection by
          // however many faces/siblings route to it. Reusing the same
          // divisors here removes that over-count the same way pass2 does —
          // see [[project_step_cross]] investigation (traced via
          // wire_split_children: every aliasing case is the documented,
          // validated Odd27 coarse-proxy pattern, not a wiring bug).
          auto const own = n_adj * delta / cl.face_multiplicity.get(k, L, j)
                                  / cl.ancestor_group_multiplicity.get(k, L, j);
          cl.temp_phi[k * L + j] += own;
        }
        for (int m = 0; m < 6; ++m) {
          auto const adj = bcc::face_adj[k][m];
          if (adj < 0) break;
          if (!live(static_cast<std::size_t>(adj))) continue;
          if (!(cell.fixed_phi & (1u << adj))) {
            // Same reasoning as the own-face term above: this boost lands in
            // face `adj`'s own slot, which gets mirrored out via nb[adj] --
            // if THAT face is one of the aliased/duplicated ones, it needs
            // the same divisor, regardless of which firing face produced the
            // contribution landing there.
            auto const adjk = static_cast<std::size_t>(adj);
            auto const idx  = adjk * L + j;
            cl.temp_phi[idx] +=
                delta / cl.face_multiplicity.get(adjk, L, j) / cl.ancestor_group_multiplicity.get(adjk, L, j);
          }
        }
      }
    }
  }

  // Atomic -= into a temp_phi slot.  Mirror workers write into OTHER cells'
  // slots: single-writer per slot on a uniform lattice, but coarse receivers
  // at AMR junctions take concurrent writes from several fine siblings.
  // relaxed suffices — pool_->wait() orders the pass against _step_cross_apply.
  static void _temp_phi_sub(double& slot, double v) {
    std::atomic_ref<double>(slot).fetch_sub(v, std::memory_order_relaxed);
  }

  // Step 1b — mirror the FROZEN post-local temp_phi (the full aggregate
  // _step_cross_local already computed for every face of every cell, summed
  // once in a single fixed per-cell order) into each face's neighbour.
  //
  // This reads cl.temp_phi[k*L+j] directly as the amount to mirror, rather
  // than re-deriving delta/n_adj per contributing firing face the way the
  // old implementation did. That distinction is what makes a same-level
  // pair's two sides exact negatives of each other: writing amount=temp_phi
  // makes slot(k,A) end up as fl(V_A - V_B) and slot(mirror(k),B) as
  // fl(V_B - V_A), and IEEE-754 subtraction guarantees fl(a-b) == -fl(b-a)
  // exactly, for any a, b. The old per-term scheme instead subtracted V_B's
  // several constituent deltas one at a time onto an already-rounded V_A
  // (and vice-versa) — a compound expression that is NOT guaranteed to
  // equal a clean single subtraction of the two aggregates, and empirically
  // didn't: confirmed via [[project_step_cross]] investigation, the gap
  // only appears once a face's local value is built from more than one
  // contributing term (e.g. a rim leaf with several proxy-adjacent faces),
  // and grows roughly quadratically over repeated calls because the drift
  // itself feeds back into the next call's delta. A post-hoc "symmetrize
  // each pair to zero" fix was tried and rejected — it broke Σphi
  // conservation, because two faces sharing a contributing firing face
  // (e.g. both in that face's face_adj list) are not independent 2-body
  // systems, so forcing each pair to zero independently double-adjusts the
  // shared term. Sourcing every mirror write from the SAME already-computed
  // local aggregate avoids that: each raw delta still contributes to
  // exactly the same set of slots it always did, just combined once instead
  // of being re-split across two independently-ordered passes.
  //
  // Writes into temp_phi_mirror (NOT temp_phi) so reading a face's frozen
  // local aggregate can never race against another thread's mirror write
  // landing on that exact slot; step_cross merges the two buffers with one
  // elementwise add (_step_cross_merge_mirror) before _step_cross_apply.
  //
  // Gate-agnostic: a cell the Barrier gate skipped already has all-zero
  // local values (see _step_cross_local), so nothing here needs to re-check
  // m0 — dt/scale/gate/m0 are unused, kept only so call sites don't change.
  // face_ghost (periodic) neighbours are resolved via bc when nb[k] == -1.
  static void _step_cross_mirror(AdaptiveCellCluster& cl,
                                 std::vector<std::int32_t> const& g_to_j,
                                 std::int32_t g_max,
                                 double /*dt*/, double /*scale*/, std::size_t L,
                                 CrossGate /*gate*/, double /*m0*/,
                                 std::size_t begin, std::size_t end) {
    auto const has_ghosts = cl.bc && cl.bc->has_face_ghosts();
    for (std::size_t j = begin; j < end; ++j) {
      auto const g    = cl.leaf_indices[j];
      auto const cell = cl[g];
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        auto const amount = cl.temp_phi[k * L + j];  // frozen local aggregate for this face
        if (amount == 0.0) continue;

        auto nb_k = cell.nb[k];
        if (nb_k < 0 && has_ghosts) nb_k = cl.bc->find_face_ghost(g, k, cell.level);
        if (nb_k < 0) continue;

        auto const mk  = bcc::mirror_face[k];
        auto const jnb = nb_k <= g_max ? g_to_j[static_cast<std::size_t>(nb_k)]
                                       : std::int32_t{-1};
        if (jnb >= 0) {
          // Leaf neighbour (regular or periodic ghost) — full correction 1:1.
          // Includes a COARSER leaf across a fine-side junction face, where
          // 1:1 is exact under the sum convention (each fine sibling
          // mirrors its own share).
          auto const nbc = cl[nb_k];
          if (!(nbc.fixed_phi & (1u << mk)))
            _temp_phi_sub(cl.temp_phi_mirror[mk * L + static_cast<std::size_t>(jnb)], amount);
        } else if (cl[nb_k].n_children > 0) {
          // Non-leaf neighbour: a refinement proxy. Distribute across its
          // leaf descendants facing this cell instead of dropping the write.
          _step_cross_mirror_proxy(cl, g_to_j, g_max, nb_k, mk,
                                   static_cast<std::int32_t>(g), amount, L);
        }
      }
    }
  }

  // Centre position of cell g (slab storage).
  static void _cell_pos(AdaptiveCellCluster& cl, std::int32_t g, double out[3]) {
    auto const s    = AdaptiveCellCluster::slab_of(g);
    auto const slot = AdaptiveCellCluster::slot_of(g);
    auto& slab      = cl.slab(s);
    out[0] = slab.pos[0 * CellSlab::CAPACITY + slot];
    out[1] = slab.pos[1 * CellSlab::CAPACITY + slot];
    out[2] = slab.pos[2 * CellSlab::CAPACITY + slot];
  }

  // Cheap, 1-hop (no BFS) least-squares gradient fit for cell g's own value,
  // using only its live SAME-LEVEL neighbours -- the local counterpart to
  // _prolong_u_gradient_wide's 5-hop BFS fit (bcc_geometry.hpp), sized for
  // being called on the hot per-substep cross-level flux path rather than
  // once at split time. Same Cramer's-rule normal-equation solve, same
  // degenerate-stencil fallback to grad=(0,0,0) (flat/unchanged) as that
  // function. Restricted to same-level neighbours only so it never has to
  // recurse into ANOTHER coarse-fine junction to build its own stencil.
  static void _local_gradient_same_level(AdaptiveCellCluster& cl, std::int32_t g,
                                         double grad_out[3]) {
    constexpr double DET_EPS = 1e-12;
    auto const cell = cl[g];
    double p0[3];
    _cell_pos(cl, g, p0);
    double const u0 = cell.u;
    double const L0 = cell.L;

    double Mxx = 0, Mxy = 0, Mxz = 0, Myy = 0, Myz = 0, Mzz = 0, bx = 0, by = 0, bz = 0;
    for (std::size_t k = 0; k < CellSlab::NF; ++k) {
      auto const nbg = cell.nb[k];
      if (nbg < 0) continue;
      auto const nbc = cl[nbg];
      if (nbc.L != L0) continue;  // same-level only
      double p1[3];
      _cell_pos(cl, nbg, p1);
      double const x  = p1[0] - p0[0], y = p1[1] - p0[1], z = p1[2] - p0[2];
      double const du = nbc.u - u0;
      Mxx += x * x; Mxy += x * y; Mxz += x * z;
      Myy += y * y; Myz += y * z; Mzz += z * z;
      bx += x * du; by += y * du; bz += z * du;
    }
    double const det = Mxx * (Myy * Mzz - Myz * Myz) - Mxy * (Mxy * Mzz - Myz * Mxz) +
                       Mxz * (Mxy * Myz - Myy * Mxz);
    grad_out[0] = grad_out[1] = grad_out[2] = 0.0;
    if (std::abs(det) > DET_EPS) {
      double const detx = bx * (Myy * Mzz - Myz * Myz) - Mxy * (by * Mzz - Myz * bz) +
                          Mxz * (by * Myz - Myy * bz);
      double const dety = Mxx * (by * Mzz - bz * Myz) - bx * (Mxy * Mzz - Myz * Mxz) +
                          Mxz * (Mxy * bz - by * Mxz);
      double const detz = Mxx * (Myy * bz - Myz * by) - Mxy * (Mxy * bz - by * Mxz) +
                          bx * (Mxy * Myz - Myy * Mxz);
      grad_out[0] = detx / det;
      grad_out[1] = dety / det;
      grad_out[2] = detz / det;
    }
  }

  // Extrapolate cell g's own value to target position face[3] via
  // _local_gradient_same_level -- the shared correction applied to BOTH
  // sides of a cross-level face's (nb.u - u_src) difference when
  // enable_cross_level_gradient_correction is set.
  static double _gradient_corrected_value(AdaptiveCellCluster& cl, std::int32_t g,
                                          double const face[3]) {
    double grad[3];
    _local_gradient_same_level(cl, g, grad);
    double p0[3];
    _cell_pos(cl, g, p0);
    return cl[g].u + grad[0] * (face[0] - p0[0]) + grad[1] * (face[1] - p0[1]) +
          grad[2] * (face[2] - p0[2]);
  }

  // Centre of the (hypothetical) face-k neighbour of cell g at g's own level.
  // hex k: per-axis offset h = L/sqrt(3); sq k: 2h along one axis.
  static void _neighbour_pos(AdaptiveCellCluster& cl, std::int32_t g, std::size_t k,
                             double out[3]) {
    _cell_pos(cl, g, out);
    auto const h = cl[g].L / std::sqrt(3.0);
    if (k < 8) {
      for (int i = 0; i < 3; ++i) out[i] += bcc::hex_dir[k][i] * h;
    } else {
      for (int i = 0; i < 3; ++i) out[i] += bcc::sq_dir[k - 8][i] * 2.0 * h;
    }
  }

  // Descend from proxy_g toward the cell centred on tp, stopping at `level`
  // or at a leaf, whichever comes first.  Children sit at parent ± h_child
  // per axis (child_octant), so the nearest child to tp is unambiguous at
  // every level (exact match vs >= one child spacing).  An early leaf is
  // returned as-is — it is a coarser counterpart (partial refinement) and
  // the caller mirrors into it 1:1.
  static std::int32_t _descend_to(AdaptiveCellCluster& cl, std::int32_t proxy_g,
                                  std::uint8_t level, double const tp[3]) {
    auto cur = proxy_g;
    while (cl[cur].level < level && cl[cur].n_children > 0) {
      auto const P  = cl[cur];
      auto const fc = P.first_child;
      std::int32_t best = -1;
      double best_d2 = std::numeric_limits<double>::infinity();
      for (std::uint8_t c = 0; c < P.n_children; ++c) {
        double cp[3];
        _cell_pos(cl, fc + c, cp);
        auto const dx = cp[0] - tp[0], dy = cp[1] - tp[1], dz = cp[2] - tp[2];
        auto const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < best_d2) {
          best_d2 = d2;
          best    = fc + c;
        }
      }
      if (best < 0) return -1;
      cur = best;
    }
    return cur;
  }

  // Mirror a step_cross correction into a refinement-proxy neighbour: write
  // -amount, summed, onto the leaf faces that physically tile cell
  // target_g's face mirror(mk) — the sum convention phi_coarse = -Σ phi_fine
  // that _step_pass2_amr_correct maintains for the wave step.
  //
  // Two wirings occur (children inherit the parent's nb gid verbatim):
  //   • coarse-fine: proxy is at target's level; its facing leaves are wired
  //     to target_g directly — collect and distribute.
  //   • cross-group: proxy is target's neighbour GROUP at a coarser level;
  //     its leaves are wired to an ANCESTOR of target_g, so matching by
  //     target_g finds nothing.  Resolve the same-level geometric partner by
  //     descending the proxy subtree toward the neighbour position of
  //     target's face: a leaf partner (same level or coarser under partial
  //     refinement) takes the full correction 1:1; a refined partner has its
  //     facing leaves collected via the nb gid they inherited from it.
  //
  // Distribution over multiple faces is proportional to |phi| — each fine
  // face gives up its share of the aggregate crossing — with an equal split
  // when the faces carry no phi.  Writing per-pair like this (rather than
  // balancing junction sums after the fact) keeps every face pair
  // consistent, which the single-processing rule (inward side fires, mirror
  // side receives) depends on.  fixed_phi faces take no share; if all
  // candidate faces are fixed the correction is dropped, matching the
  // leaf-neighbour path.
  static void _step_cross_mirror_proxy(AdaptiveCellCluster& cl,
                                       std::vector<std::int32_t> const& g_to_j,
                                       std::int32_t g_max, std::int32_t proxy_g,
                                       std::size_t mk, std::int32_t target_g,
                                       double amount, std::size_t L) {
    static thread_local std::vector<std::int32_t> desc;
    desc.clear();
    _collect_leaves_facing(cl, proxy_g, mk, target_g, g_max, g_to_j, desc);
    if (desc.empty()) {
      // Cross-group junction: the proxy's leaves are wired to an ancestor of
      // target_g (children inherit the parent's nb gid), so matching by
      // target_g found nothing.
      if (mk < 8) {
        // Hex: fine hex faces read the neighbour GROUP as a coarse proxy
        // (wire_split_children step 3 — no 1:1 fine hex partner exists in
        // this split geometry).  Distribute across the group's facing
        // leaves — the same junction granularity the wave step's
        // _step_pass2_amr_correct pairs use.
        auto const anc = cl[proxy_g].nb[mk];
        if (anc < 0) return;
        _collect_leaves_facing(cl, proxy_g, mk, anc, g_max, g_to_j, desc);
        if (desc.empty()) return;
      } else {
        // Sq: the same-level geometric partner exists (the neighbour group's
        // child at exactly the sq offset) — resolve it by position descent.
        double tp[3];
        _neighbour_pos(cl, target_g, bcc::mirror_face[mk], tp);
        auto const part = _descend_to(cl, proxy_g, cl[target_g].level, tp);
        if (part < 0) return;
        if (cl[part].n_children == 0) {
          // same-level (or coarser) leaf partner — plain 1:1 mirror
          if (part <= g_max && g_to_j[static_cast<std::size_t>(part)] >= 0) {
            auto const pc = cl[part];
            if (!(pc.fixed_phi & (1u << mk)))
              _temp_phi_sub(
                  cl.temp_phi_mirror[mk * L +
                                     static_cast<std::size_t>(g_to_j[static_cast<std::size_t>(part)])],
                  amount);
          }
          return;
        }
        // partner refined below target — its facing leaves carry the nb gid
        // they inherited from it
        _collect_leaves_facing(cl, part, mk, cl[part].nb[mk], g_max, g_to_j, desc);
        if (desc.empty()) return;
      }
    }
    double wsum = 0.0;
    int n_free  = 0;
    for (auto const dg : desc) {
      auto const dc = cl[dg];
      if (dc.fixed_phi & (1u << mk)) continue;
      wsum += std::abs(dc.phi[mk]);
      ++n_free;
    }
    if (n_free == 0) return;
    for (auto const dg : desc) {
      auto const dc = cl[dg];
      if (dc.fixed_phi & (1u << mk)) continue;
      auto const w = wsum > 0.0 ? std::abs(dc.phi[mk]) / wsum : 1.0 / n_free;
      _temp_phi_sub(cl.temp_phi_mirror[mk * L + static_cast<std::size_t>(g_to_j[dg])], amount * w);
    }
  }

  // Merge step_cross's mirror corrections (temp_phi_mirror, written by
  // _step_cross_mirror) into temp_phi before _step_cross_apply. Pure
  // elementwise add over disjoint indices — safe to parallelise.
  static void _step_cross_merge_mirror(AdaptiveCellCluster& cl, std::size_t begin, std::size_t end) {
    auto const L = cl.leaf_indices.size();
    for (std::size_t j = begin; j < end; ++j)
      for (std::size_t k = 0; k < CellSlab::NF; ++k)
        cl.temp_phi[k * L + j] += cl.temp_phi_mirror[k * L + j];
  }

  // Step 2 — parallel: apply accumulated temp_phi to phi.
  static void _step_cross_apply(AdaptiveCellCluster& cl, std::size_t begin, std::size_t end) {
    auto const L = cl.leaf_indices.size();
    for (std::size_t j = begin; j < end; ++j) {
      auto const g = cl.leaf_indices[j];
      auto cell    = cl[g];
      for (std::size_t k = 0; k < CellSlab::NF; ++k)
        if (!(cell.fixed_phi & (1u << k))) cell.phi[k] += cl.temp_phi[k * L + j];
    }
  }

  // Pass 1: du = -dt·sigma·Σ W[k]·phi[k]; u_new written directly to slab.u.
  // Reads phi[] (old), writes slab.u (u_new) and temp_du.
  // Safe to run in parallel — each thread writes only its own leaf range in slab.u.
  // Does boundary face k of leaf (g, cell) carry a ghost-driven phi that u must see? fn (other than
  // absorbing, whose ghost = u_i leaves phi at 0) and remote_flux faces, on unrefined clusters only
  // (AMR dead faces also have nb < 0). Face ghosts stay excluded (step_cross bookkeeping; including
  // them was unstable, project_bc_enum). A Dirichlet ghost is a fixed-u neighbour.
  static bool _bc_face_to_u(AdaptiveCellCluster const& cl, std::int32_t g, std::uint8_t level,
                            std::size_t k) {
    auto const* bc = cl.bc.get();
    if (!bc || cl.n_non_leaf_cells != 0) return false;
    if (bc->remote_flux == 0.0 && (!bc->fn || bc->fn == BoundaryCondition::absorbing)) return false;
    return !(bc->has_face_ghosts() && bc->find_face_ghost(g, k, level) >= 0);
  }

  static void _step_pass1(AdaptiveCellCluster& cl, TimeStepper const& ts,
                          std::size_t begin, std::size_t end) {
    EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::_step_pass1");

    auto const L = cl.leaf_indices.size();
    // Periodic BC faces are wired directly via nb[k] = partner_g by wire_periodic_bcc,
    // so they appear here as regular nb[k] >= 0 faces — no special casing needed.
    // has_outflow: see bc.hpp's outflow_active doc comment — the ONE case
    // where a boundary face (nb[k]<0) legitimately contributes to u here;
    // every other BC path (fn/face_ghost/remote_flux) stays excluded.
    auto const has_outflow = cl.bc && !cl.bc->outflow_active.empty();
    auto const has_taper   = cl.bc && !cl.bc->outflow_taper_weight.empty();
    for (std::size_t j = begin; j < end; ++j) {
      EQ_MCA_BEGIN("discrete_pass1");
      auto const g     = cl.leaf_indices[j];
      auto const cell  = cl[g];
      auto const dt    = ts(cell.L, cell.level);
      double acc = 0.0;
      // Uses the SAME per-face coefficient (stale-proxy resolution via
      // `_resolve_face_neighbour`, ancestor-substitution L_src via
      // `_face_flux_coefficient`) that `_step_pass2` used to build this
      // phi[k] value in the first place — a mismatched coefficient here
      // breaks both H=u²+phi² conservation (a first-order-in-dt leak,
      // derived + confirmed 2026-08-14) and Σu conservation at cross-level
      // faces specifically (TestFluxConservation's coarse-fine/common-
      // denominator regressions, 2026-08-14).
      //
      // Multiplicity is deliberately NOT reapplied here — see
      // `_step_pass1_lev`'s identical comment: pass2 already divides dphi by
      // it when constructing phi[k], so phi[k] is already this face's
      // correctly-scaled private share; dividing again double-counts it.
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        auto const nbg = cell.nb[k];
        if (nbg >= 0) {
          auto const nbg_r = _resolve_face_neighbour(cl, g, cell, k, nbg);
          auto const L_src = _ancestor_L_src(cl, cell, nbg_r, /*respect_disable_gate=*/false);
          auto const coef  = _face_flux_coefficient(cl, cell, k, nbg_r, L_src,
                                                     /*apply_multiplicity=*/false, L, j);
          acc += coef * cell.phi[k];
        } else if (_bc_face_to_u(cl, g, cell.level, k)) {
          acc += cell.sigma() * (bcc::face_weight[k] / cell.L) * cell.phi[k];
        } else if (has_outflow && cl.bc->outflow_active[k * L + j]) {
          // No real neighbour; matches the coefficient _step_pass2_bc's
          // outflow branch already uses for this same face (ghost sigma/L
          // assumed equal to the owning cell's own).
          acc += cell.sigma() * (bcc::face_weight[k] / cell.L) * cell.phi[k];
        }
      }
      auto const du = -dt * acc;
      cl.temp_du[j] = du;
      if (!cell.fixed_u) cell.u += du;  // write u_new into slab

      // Taper band sink (see bc.hpp's outflow_taper_weight doc comment) —
      // additive only, applied after the real-neighbour update above.
      if (has_taper) {
        auto const w = cl.bc->outflow_taper_weight[j];
        if (w > 0.0 && !cell.fixed_u) {
          auto const c_i = cl.bc->outflow_c.empty() ? cell.sigma() * cl.bc->outflow_c_lattice
                                                     : cl.bc->outflow_c[j];
          auto const L_i        = cell.L;
          auto const r          = (L_i - c_i * dt) / (L_i + c_i * dt);
          auto const u_i_new    = cell.u;
          auto const u_i_old    = u_i_new - du;
          auto const ghost_prev = cl.bc->outflow_taper_ghost[j];
          auto const target     = u_i_old + r * (ghost_prev - u_i_new);
          cl.bc->outflow_taper_ghost[j] = target;
          cell.u -= w * (u_i_new - target) * (c_i * dt / L_i);
        }
      }
      EQ_MCA_END("discrete_pass1");
    }
  }

  // Damped variant of _step_pass1 — see bc.hpp's `gamma` doc comment.
  // Identical to _step_pass1 except the OLD u is decayed (exp(-gamma*dt))
  // before adding the same undamped flux increment `du`, folding the sponge
  // into this leapfrog stage instead of a separate post-step multiply.
  static void _step_pass1_damped(AdaptiveCellCluster& cl, TimeStepper const& ts,
                                  std::size_t begin, std::size_t end) {
    EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::_step_pass1_damped");

    auto const L         = cl.leaf_indices.size();
    bool const has_gamma   = cl.bc && !cl.bc->gamma.empty();
    auto const has_outflow = cl.bc && !cl.bc->outflow_active.empty();
    for (std::size_t j = begin; j < end; ++j) {
      auto const g     = cl.leaf_indices[j];
      auto const cell  = cl[g];
      auto const dt    = ts(cell.L, cell.level);
      double acc = 0.0;
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        auto const nbg = cell.nb[k];
        if (nbg >= 0) {
          auto const nbg_r = _resolve_face_neighbour(cl, g, cell, k, nbg);
          auto const L_src = _ancestor_L_src(cl, cell, nbg_r, /*respect_disable_gate=*/false);
          auto const coef  = _face_flux_coefficient(cl, cell, k, nbg_r, L_src,
                                                     /*apply_multiplicity=*/false, L, j);
          acc += coef * cell.phi[k];
        } else if (_bc_face_to_u(cl, g, cell.level, k)) {
          acc += cell.sigma() * (bcc::face_weight[k] / cell.L) * cell.phi[k];
        } else if (has_outflow && cl.bc->outflow_active[k * L + j]) {
          // Same outflow-at-the-true-edge branch as plain _step_pass1 (see
          // bc.hpp's outflow_active doc comment) -- lets outflow combine with
          // this leaf's own gamma-based sponge decay below.
          acc += cell.sigma() * (bcc::face_weight[k] / cell.L) * cell.phi[k];
        }
      }
      auto const du = -dt * acc;
      cl.temp_du[j] = du;
      if (!cell.fixed_u) {
        auto const decay = has_gamma ? std::exp(-cl.bc->gamma[j] * dt) : 1.0;
        cell.u = decay * cell.u + du;
      }
    }
  }

  // Split-field PML pass1 — see bc.hpp's proj_r/u_aux doc comment. Updates
  // cell.u to the SAME total the plain _step_pass1 would produce, minus the
  // auxiliary u_aux's decay loss for this step; u_aux itself only tracks
  // the radially-projected (proj_r-weighted) fraction of the flux and
  // decays at rate gamma[j]. Reduces to plain _step_pass1 exactly wherever
  // gamma==0 (outside the sponge) since decay_loss==0 there.
  static void _step_pass1_pml(AdaptiveCellCluster& cl, TimeStepper const& ts,
                               std::size_t begin, std::size_t end) {
    EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::_step_pass1_pml");

    auto const L      = cl.leaf_indices.size();
    auto* bc          = cl.bc.get();
    bool const has_pml     = bc && !bc->u_aux.empty();
    bool const has_outflow = bc && !bc->outflow_active.empty();
    for (std::size_t j = begin; j < end; ++j) {
      auto const g    = cl.leaf_indices[j];
      auto const cell = cl[g];
      auto const dt   = ts(cell.L, cell.level);
      double acc = 0.0, acc_r = 0.0;
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        auto const nbg = cell.nb[k];
        if (nbg >= 0) {
          auto const nbg_r = _resolve_face_neighbour(cl, g, cell, k, nbg);
          auto const L_src = _ancestor_L_src(cl, cell, nbg_r, /*respect_disable_gate=*/false);
          auto const coef  = _face_flux_coefficient(cl, cell, k, nbg_r, L_src,
                                                     /*apply_multiplicity=*/false, L, j);
          auto const contrib = coef * cell.phi[k];
          acc += contrib;
          if (has_pml) acc_r += contrib * bc->proj_r[k * L + j];
        } else if (_bc_face_to_u(cl, g, cell.level, k)) {
          acc += cell.sigma() * (bcc::face_weight[k] / cell.L) * cell.phi[k];
        } else if (has_outflow && bc->outflow_active[k * L + j]) {
          // Same outflow-at-the-true-edge branch as plain _step_pass1 (see
          // bc.hpp's outflow_active doc comment). Deliberately NOT added to
          // acc_r/u_aux -- there's no real neighbour to project radially,
          // this is genuine loss at the edge itself, not sponge-tracked flux.
          acc += cell.sigma() * (bcc::face_weight[k] / cell.L) * cell.phi[k];
        }
      }
      auto const du = -dt * acc;
      cl.temp_du[j] = du;
      if (!cell.fixed_u) {
        if (has_pml) {
          auto const gamma_r     = bc->gamma[j];
          auto const decay       = std::exp(-gamma_r * dt);
          auto& u_r              = bc->u_aux[j];
          auto const decay_loss  = u_r * (1.0 - decay);
          u_r    = u_r * decay - dt * acc_r;
          cell.u = cell.u + du - decay_loss;
        } else {
          cell.u += du;
        }
      }
    }
  }

  // Pass 2 — uniform path: all cells at the same level with no coarse-fine interfaces.
  // Drops the ancestor walk, face_dt min, and L_src/level_src tracking that _step_pass2
  // carries for AMR grids.  dt and inv_L are hoisted per cell (same across all cells
  // in practice, but kept per-cell for cfl-mode correctness).
  // Only valid when cluster.n_non_leaf_cells == 0.
  static void _step_pass2_uniform(AdaptiveCellCluster& cl, TimeStepper const& ts,
                                  std::size_t begin, std::size_t end) {
    EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::_step_pass2_uniform");

    auto const L = cl.leaf_indices.size();
    for (std::size_t j = begin; j < end; ++j) {
      EQ_MCA_BEGIN("discrete_pass2_uniform");
      auto const g    = cl.leaf_indices[j];
      auto const cell = cl[g];
      auto const dt    = ts(cell.L, cell.level);
      auto const u_i   = cell.u;
      auto const sig_i = cell.sigma();
      auto const inv_L = 1.0 / cell.L;
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        double dphi = 0.0;
        auto const nbg = cell.nb[k];
        if (nbg >= 0 && !(cell.fixed_phi & (1u << k))) {
          auto const nb    = cl[nbg];
          auto const sig_f = 0.5 * (sig_i + nb.sigma());
          auto const w     = bcc::face_weight[k] * inv_L;
          dphi = -dt * sig_f * w * (nb.u - u_i);
        }
        cl.temp_phi[k * L + j] = dphi;
      }
      EQ_MCA_END("discrete_pass2_uniform");
    }
  }

  // Pass 2: dphi[k] = -dt·sigma_face·W[k]·2/(L_i+L_nb)·(u_new_nb - u_new_i).
  // Reads slab.u (u_new from pass 1), writes temp_phi.
  // Must run after pass 1 barrier so all u_new values are visible.
  // dt is min(dt_i, dt_nb) per face to preserve phi antisymmetry at all interfaces.
  static void _step_pass2(AdaptiveCellCluster& cl, TimeStepper const& ts,
                          std::size_t begin, std::size_t end) {
    EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::_step_pass2");

    auto const L = cl.leaf_indices.size();
    for (std::size_t j = begin; j < end; ++j) {
      auto const g    = cl.leaf_indices[j];
      auto const cell = cl[g];
      auto const u_i   = cell.u;  // u_new (written by pass 1)
      auto const L_i   = cell.L;
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        double dphi = 0.0;
        auto const nbg = cell.nb[k];
        if (nbg >= 0 && !(cell.fixed_phi & (1u << k))) {
          // Stale-proxy resolution -- see `_resolve_face_neighbour`'s comment
          // for the full rationale (hex faces only, sq faces have genuine
          // multi-child fan-in so there's no unambiguous partner there).
          auto const nbg_r = _resolve_face_neighbour(cl, g, cell, k, nbg);
          auto nb = cl[nbg_r];
          double u_src = u_i;
          double L_src = L_i;
          std::uint8_t level_src = cell.level;
          std::int32_t anc = -1;
          if (nb.L > L_i) {
            anc = cell.parent;
            while (anc >= 0 && cl[anc].L < nb.L) anc = cl[anc].parent;
            if (anc >= 0) {
              u_src     = enable_ancestor_debias_correction ? _debiased_proxy_u(cl, anc) : cl[anc].u;
              L_src     = cl[anc].L;
              level_src = cl[anc].level;
            }
          }
          auto nb_u_for_flux = (enable_ancestor_debias_correction && nb.n_children > 0)
                                   ? _debiased_proxy_u(cl, nbg_r)
                                   : nb.u;
          if (enable_cross_level_gradient_correction && nb.L != L_i) {
            double p0[3], p1[3];
            _cell_pos(cl, g, p0);
            _cell_pos(cl, nbg_r, p1);
            auto const fdx = p1[0] - p0[0], fdy = p1[1] - p0[1], fdz = p1[2] - p0[2];
            auto const flen = std::sqrt(fdx * fdx + fdy * fdy + fdz * fdz);
            double const face[3] = {p0[0] + 0.5 * L_i * fdx / flen,
                                    p0[1] + 0.5 * L_i * fdy / flen,
                                    p0[2] + 0.5 * L_i * fdz / flen};
            nb_u_for_flux = _gradient_corrected_value(cl, nbg_r, face);
            if (anc >= 0) u_src = _gradient_corrected_value(cl, anc, face);
          }
          auto const dt   = ts.face_dt(L_src, level_src, nb.L, nb.level);
          // See _step_pass2_lev's identical comment: multiplicity divisors
          // are keyed to the STORED (possibly stale) topology and must not
          // apply once resolution has found a genuine same-level 1:1 partner.
          auto const coef = _face_flux_coefficient(cl, cell, k, nbg_r, L_src, nbg_r == nbg, L, j);
          dphi = -dt * coef * (nb_u_for_flux - u_src);
        }
        cl.temp_phi[k * L + j] = dphi;
      }
    }
  }

  // Pass 3: phi[k] += dphi[k]; then update criterion_min/max from fresh phi.
  // Refine if criterion_max > threshold; coarsen if criterion_max < threshold.
  // Reads temp_phi, writes slab.phi only for its own leaf range.
  static void _step_pass3(AdaptiveCellCluster& cl, std::size_t begin, std::size_t end) {
    EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::_step_pass3");

    auto const L = cl.leaf_indices.size();
    for (std::size_t j = begin; j < end; ++j) {
      EQ_MCA_BEGIN("discrete_pass3");
      auto const g = cl.leaf_indices[j];
      auto cell    = cl[g];
      for (std::size_t k = 0; k < CellSlab::NF; ++k)
        if (!(cell.fixed_phi & (1u << k))) cell.phi[k] += cl.temp_phi[k * L + j];

      // Criterion: min/max |phi[k]| over faces that have a live neighbour.
      // NaN if isolated (no active faces) — same as the pre-step default.
      constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
      double cmin = NaN, cmax = NaN;
      bool any = false;
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        if (cell.nb[k] < 0) continue;
        auto const v = std::abs(cell.phi[k]);
        if (!any || v < cmin) cmin = v;
        if (!any || v > cmax) cmax = v;
        any = true;
      }
      cl.criterion_min[j] = cmin;
      cl.criterion_max[j] = cmax;
      EQ_MCA_END("discrete_pass3");
    }
  }

  // Damped variant of _step_pass3 — see bc.hpp's `gamma` doc comment.
  // Identical to _step_pass3 except the OLD phi is decayed (exp(-gamma*dt))
  // before adding the same undamped flux increment from temp_phi, folding
  // the sponge into this leapfrog stage instead of a separate post-step
  // multiply. Uses the cell's own gamma[j] for all of its own phi[k]
  // components, matching vol_damp's per-leaf (not per-face) semantics.
  static void _step_pass3_damped(AdaptiveCellCluster& cl, TimeStepper const& ts,
                                  std::size_t begin, std::size_t end) {
    EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::_step_pass3_damped");

    auto const L         = cl.leaf_indices.size();
    bool const has_gamma = cl.bc && !cl.bc->gamma.empty();
    for (std::size_t j = begin; j < end; ++j) {
      auto const g = cl.leaf_indices[j];
      auto cell    = cl[g];
      auto const dt = ts(cell.L, cell.level);
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        if (cell.fixed_phi & (1u << k)) continue;
        auto const decay =
            has_gamma ? std::exp(-_face_mean(cl.bc->gamma, j, _leaf_nb(cl, cell, k)) * dt) : 1.0;
        cell.phi[k] = decay * cell.phi[k] + cl.temp_phi[k * L + j];
      }

      // Criterion: min/max |phi[k]| over faces that have a live neighbour.
      // NaN if isolated (no active faces) — same as the pre-step default.
      constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
      double cmin = NaN, cmax = NaN;
      bool any = false;
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        if (cell.nb[k] < 0) continue;
        auto const v = std::abs(cell.phi[k]);
        if (!any || v < cmin) cmin = v;
        if (!any || v > cmax) cmax = v;
        any = true;
      }
      cl.criterion_min[j] = cmin;
      cl.criterion_max[j] = cmax;
    }
  }

  // Directional (PML-lite) variant of _step_pass3 — see bc.hpp's
  // `gamma_face` doc comment. u is untouched (pass1 is the plain, undamped
  // _step_pass1); only phi[k] decays, per-face, at a rate that vanishes for
  // faces tangential to the sponge boundary and peaks for radially-outward
  // faces — unlike `gamma`'s per-leaf (all-faces-equal) rate.
  static void _step_pass3_dir(AdaptiveCellCluster& cl, TimeStepper const& ts,
                               std::size_t begin, std::size_t end) {
    EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::_step_pass3_dir");

    auto const L         = cl.leaf_indices.size();
    bool const has_gamma = cl.bc && !cl.bc->gamma_face.empty();
    for (std::size_t j = begin; j < end; ++j) {
      auto const g = cl.leaf_indices[j];
      auto cell    = cl[g];
      auto const dt = ts(cell.L, cell.level);
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        if (cell.fixed_phi & (1u << k)) continue;
        auto const decay =
            has_gamma ? std::exp(-_face_mean_kj(cl.bc->gamma_face, L, k, j, _leaf_nb(cl, cell, k)) * dt)
                      : 1.0;
        cell.phi[k] = decay * cell.phi[k] + cl.temp_phi[k * L + j];
      }

      constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
      double cmin = NaN, cmax = NaN;
      bool any = false;
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        if (cell.nb[k] < 0) continue;
        auto const v = std::abs(cell.phi[k]);
        if (!any || v < cmin) cmin = v;
        if (!any || v > cmax) cmax = v;
        any = true;
      }
      cl.criterion_min[j] = cmin;
      cl.criterion_max[j] = cmax;
    }
  }

  // Split-field PML pass3 — see bc.hpp's proj_r/phi_aux doc comment. Same
  // structure as _step_pass1_pml: cell.phi[k] gets the SAME total the plain
  // _step_pass3 would produce, minus phi_aux[k]'s decay loss; phi_aux[k]
  // only tracks the proj_r-weighted fraction of this step's ordinary flux
  // increment (temp_phi) and decays at rate gamma[j] (same per-leaf rate as
  // pass1, not face-dependent). Reduces to plain _step_pass3 exactly
  // wherever gamma==0.
  static void _step_pass3_pml(AdaptiveCellCluster& cl, TimeStepper const& ts,
                               std::size_t begin, std::size_t end) {
    EQ_SCOPE_TIMER("equilibrium::dst::discrete::AdaptiveLattice::_step_pass3_pml");

    auto const L       = cl.leaf_indices.size();
    auto* bc           = cl.bc.get();
    bool const has_pml = bc && !bc->phi_aux.empty();
    for (std::size_t j = begin; j < end; ++j) {
      auto const g = cl.leaf_indices[j];
      auto cell    = cl[g];
      auto const dt = ts(cell.L, cell.level);
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        if (cell.fixed_phi & (1u << k)) continue;
        auto const dphi_k = cl.temp_phi[k * L + j];
        if (has_pml) {
          auto const jn         = _leaf_nb(cl, cell, k);
          auto const decay      = std::exp(-_face_mean(bc->gamma, j, jn) * dt);
          auto const proj       = _face_mean_kj(bc->proj_r, L, k, j, jn);
          auto& phi_r           = bc->phi_aux[k * L + j];
          auto const decay_loss = phi_r * (1.0 - decay);
          phi_r       = phi_r * decay + proj * dphi_k;
          cell.phi[k] = cell.phi[k] + dphi_k - decay_loss;
        } else {
          cell.phi[k] += dphi_k;
        }
      }

      constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
      double cmin = NaN, cmax = NaN;
      bool any = false;
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        if (cell.nb[k] < 0) continue;
        auto const v = std::abs(cell.phi[k]);
        if (!any || v < cmin) cmin = v;
        if (!any || v > cmax) cmax = v;
        any = true;
      }
      cl.criterion_min[j] = cmin;
      cl.criterion_max[j] = cmax;
    }
  }

  std::unique_ptr<BS::thread_pool<>> pool_;
};

}  // namespace equilibrium::dst::discrete

#endif  // _EQUILIBRIUM_CORE_DATA_LATTICE_ADAPTIVE_LATTICE_HPP_
