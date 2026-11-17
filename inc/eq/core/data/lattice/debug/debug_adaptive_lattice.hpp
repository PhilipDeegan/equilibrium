#ifndef EQUILIBRIUM_CORE_DATA_LATTICE_ADAPTIVE_LATTICE_DEBUG_HPP
#define EQUILIBRIUM_CORE_DATA_LATTICE_ADAPTIVE_LATTICE_DEBUG_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace equilibrium::dst::discrete {

struct AdaptiveLatticeException : public std::runtime_error {
  explicit AdaptiveLatticeException(std::string const& msg) : std::runtime_error{msg} {}
};

// ---------------------------------------------------------------------------
// accumulate_to_parent checks (called from accumulate_to_parent)
// ---------------------------------------------------------------------------

// Before: children are live, correctly linked, and have finite u.
void check_state_priot_accumulate_to_parent_start(auto& cluster, auto parent_g) {
  auto p = cluster[parent_g];
  std::uint8_t nc = p.n_children;
  std::int32_t fc = p.first_child;
  std::uint8_t expected_child_level = static_cast<std::uint8_t>(p.level + 1);

  for (std::uint8_t c = 0; c < nc; ++c) {
    std::int32_t cg = fc + c;
    auto ch = cluster[cg];

    if (ch.parent < 0) {
      std::ostringstream oss;
      oss << "accumulate_to_parent[" << parent_g << "]: child[" << (int)c << "] (g=" << cg
          << ") freed or invalid (parent=" << ch.parent << ")";
      throw AdaptiveLatticeException(oss.str());
    }
    if (ch.parent != parent_g) {
      std::ostringstream oss;
      oss << "accumulate_to_parent[" << parent_g << "]: child[" << (int)c << "] (g=" << cg
          << ") parent=" << ch.parent << " expected " << parent_g;
      throw AdaptiveLatticeException(oss.str());
    }
    if (ch.level != expected_child_level) {
      std::ostringstream oss;
      oss << "accumulate_to_parent[" << parent_g << "]: child[" << (int)c << "] (g=" << cg
          << ") level=" << (int)ch.level << " expected " << (int)expected_child_level;
      throw AdaptiveLatticeException(oss.str());
    }
    if (!std::isfinite(ch.u)) {
      std::ostringstream oss;
      oss << "accumulate_to_parent[" << parent_g << "]: child[" << (int)c << "] (g=" << cg
          << ") u not finite: " << ch.u;
      throw AdaptiveLatticeException(oss.str());
    }
  }
}

// After: parent.u must equal mean(children.u) — u conservation invariant.
// Odd27 has 27 children (well-conditioned single-call summation, ~27*eps),
// but this runs every subcycle substep -- at deep levels a single parent can
// be re-accumulated thousands of times over a run, and p.u + (new_u - p.u)
// is not always bit-identical to new_u in IEEE-754, so a few ULPs/call
// compounds. REL_TOL must survive that repetition, not just one call.
void check_state_priot_accumulate_to_parent_end(auto& cluster, auto parent_g) {
  constexpr double REL_TOL = 1e-9;
  auto p = cluster[parent_g];
  std::uint8_t nc = p.n_children;
  std::int32_t fc = p.first_child;

  if (!std::isfinite(p.u)) {
    std::ostringstream oss;
    oss << "accumulate_to_parent[" << parent_g << "]: parent.u not finite: " << p.u;
    throw AdaptiveLatticeException(oss.str());
  }

  double u_sum = 0.0;
  for (std::uint8_t c = 0; c < nc; ++c) u_sum += cluster[fc + c].u;
  double u_mean = u_sum / nc;
  double u_err = std::abs(p.u - u_mean);
  double u_scale = std::max(std::abs(u_mean), 1e-300);
  if (u_err > REL_TOL * u_scale) {
    std::ostringstream oss;
    oss << "accumulate_to_parent[" << parent_g << "]: u conservation violation: "
        << "parent.u=" << p.u << " mean(children.u)=" << u_mean
        << " rel_err=" << (u_err / u_scale);
    throw AdaptiveLatticeException(oss.str());
  }
}

// ---------------------------------------------------------------------------
// split() end check
// ---------------------------------------------------------------------------

// After split: children inherit parent u (constant prolongation), L shrinks
// by `ratio` (2.0 for the legacy hex splits, 3.0 for Odd27), level is
// parent.level+1, and parent has become a non-leaf.
void check_state_after_split(auto& cluster, auto parent_g, auto first_child, auto nc,
                             double ratio = 2.0) {
  auto p = cluster[parent_g];

  if (p.n_children == 0) {
    std::ostringstream oss;
    oss << "split[" << parent_g << "]: parent still a leaf after split";
    throw AdaptiveLatticeException(oss.str());
  }

  double expected_child_L = p.L / ratio;
  std::uint8_t expected_child_level = static_cast<std::uint8_t>(p.level + 1);

  for (std::uint8_t c = 0; c < nc; ++c) {
    std::int32_t cg = first_child + c;
    auto ch = cluster[cg];

    if (ch.parent != parent_g) {
      std::ostringstream oss;
      oss << "split[" << parent_g << "]: child[" << (int)c << "] (g=" << cg
          << ") parent=" << ch.parent << " expected " << parent_g;
      throw AdaptiveLatticeException(oss.str());
    }
    if (ch.level != expected_child_level) {
      std::ostringstream oss;
      oss << "split[" << parent_g << "]: child[" << (int)c << "] (g=" << cg
          << ") level=" << (int)ch.level << " expected " << (int)expected_child_level;
      throw AdaptiveLatticeException(oss.str());
    }
    double u_err = std::abs(ch.u - p.u);
    double u_scale = std::max(std::abs(p.u), 1e-300);
    if (u_err > 1e-12 * u_scale) {
      std::ostringstream oss;
      oss << "split[" << parent_g << "]: child[" << (int)c << "] (g=" << cg
          << "] u=" << ch.u << " != parent.u=" << p.u << " (constant prolongation violated)";
      throw AdaptiveLatticeException(oss.str());
    }
    double L_err = std::abs(ch.L - expected_child_L);
    if (L_err > 1e-12 * expected_child_L) {
      std::ostringstream oss;
      oss << "split[" << parent_g << "]: child[" << (int)c << "] (g=" << cg
          << ") L=" << ch.L << " expected " << expected_child_L;
      throw AdaptiveLatticeException(oss.str());
    }
  }
}

// ---------------------------------------------------------------------------
// coarsen() end check
// ---------------------------------------------------------------------------

// After coarsen: parent must be a leaf (n_children==0) with finite u.
void check_state_after_coarsen(auto& cluster, auto parent_g) {
  auto p = cluster[parent_g];

  if (p.n_children != 0) {
    std::ostringstream oss;
    oss << "coarsen[" << parent_g << "]: parent still non-leaf after coarsen (n_children="
        << (int)p.n_children << ")";
    throw AdaptiveLatticeException(oss.str());
  }
  if (p.first_child != -1) {
    std::ostringstream oss;
    oss << "coarsen[" << parent_g << "]: first_child=" << p.first_child
        << " not cleared after coarsen";
    throw AdaptiveLatticeException(oss.str());
  }
  if (!std::isfinite(p.u)) {
    std::ostringstream oss;
    oss << "coarsen[" << parent_g << "]: parent.u not finite after coarsen: " << p.u;
    throw AdaptiveLatticeException(oss.str());
  }
}

// ---------------------------------------------------------------------------
// Step check: total dphi balance
//
// Called after _step_pass2_amr_correct.  Checks the face_weight-WEIGHTED sum
// of dphi over every active leaf face, not the raw sum:
//   Σ_{leaf i, face k where nb[k]>=0}  face_weight[k] * temp_phi[k*L + j_i]  == 0
// Pass1's own du is -dt*sigma*Σ_k face_weight[k]*phi[k], so a face's pull on
// Σu scales with face_weight[k], not with a flat 1 — the weighted sum is the
// invariant that actually corresponds to Σu conservation. _step_pass2_amr_correct
// (and face_multiplicity/ancestor_group_multiplicity) already target exactly
// this weighted quantity (see their own doc comments). Checking the RAW sum
// instead is a stricter, no-longer-correct invariant: Odd27's coarse-proxy
// routing routinely mixes hex (weight 3) and sq (weight 1) faces within one
// correction group, so zeroing the weighted sum does not zero the raw sum —
// confirmed directly (see [[project_odd27_amr_junction_leak]] Part 14): the
// raw sum can be off by ~10% while Σu itself stays conserved to machine
// precision. face_weight is passed in by the caller (rather than this
// header referencing bcc::face_weight directly) because this function is a
// template whose non-dependent names are looked up at definition point, and
// this header is included before bcc::face_weight is declared in
// adaptive_lattice.hpp.
// ---------------------------------------------------------------------------
// 2026-08-04: briefly changed this to bucket BY EACH LEAF'S OWN LEVEL,
// hypothesizing that a single mesh-wide total/scale pair could mask a real,
// deep-level-only defect swamped by unrelated shallow-level scale. DISPROVEN
// by direct measurement (tst/.../test_boundary_same_level_conservation,
// single Odd27 split + one flat step): level 0's total and level 1's total
// came out as EXACT negatives of each other (+8.4517e-06 / -8.4517e-06, to
// fp precision) — legitimate cross-level flux exchange, conserved in
// aggregate exactly as the reflux-correction machinery is supposed to do,
// NOT a defect. Per-level bucketing was the wrong invariant: two levels
// that genuinely, correctly exchange flux will ALWAYS make each side's own
// total nonzero — that's not a bug, it's the mechanism working. Reverted to
// the original global aggregate (mathematically the sum of the per-level
// totals/scales below, so bit-identical to the pre-2026-08-04 behaviour).
// The per-level breakdown is kept as an opt-in diagnostic print
// (EQ_DEBUG_DPHI_LEVELS) since it's a genuinely useful view of how flux
// exchange is distributed across levels — it just isn't the right thing to
// assert on. See doc/dst_algo.md (2026-08-04) for the full account.
void check_total_dphi_balanced(auto& cluster, std::size_t nfaces, auto const& face_weight) {
  constexpr double REL_TOL = 1e-10;
  std::size_t L = cluster.leaf_indices.size();

  std::vector<double> total_by_level(cluster.n_levels(), 0.0);
  std::vector<double> scale_by_level(cluster.n_levels(), 0.0);
  double total = 0.0;
  double scale = 0.0;

  for (std::size_t j = 0; j < L; ++j) {
    std::int32_t g = cluster.leaf_indices[j];
    auto cell = cluster[g];
    std::size_t const lvl = cell.level;
    for (std::size_t k = 0; k < nfaces; ++k) {
      if (cell.nb[k] < 0) continue;
      double v = face_weight[k] * cluster.temp_phi[k * L + j];
      total_by_level[lvl] += v;
      scale_by_level[lvl] += std::abs(v);
      total += v;
      scale += std::abs(v);
    }
  }

  EQUILIBRIUM_DEBUG_DO({
    if (std::getenv("EQ_DEBUG_DPHI_LEVELS")) {
      for (std::size_t lvl = 0; lvl < total_by_level.size(); ++lvl) {
        std::fprintf(stderr, "DPHI_LEVEL lvl=%zu total=%.10g scale=%.10g rel=%.10g\n", lvl,
                     total_by_level[lvl], scale_by_level[lvl],
                     scale_by_level[lvl] > 0.0 ? std::abs(total_by_level[lvl]) / scale_by_level[lvl]
                                               : 0.0);
      }
    }
  })

  if (scale == 0.0) return;
  if (std::abs(total) > REL_TOL * scale) {
    std::ostringstream oss;
    oss << "check_total_dphi_balanced: total=" << total << " scale=" << scale
        << " rel=" << (std::abs(total) / scale)
        << " — unbalanced weighted dphi (AMR correction incomplete or inward-face leak)";
    throw AdaptiveLatticeException(oss.str());
  }
}

// ---------------------------------------------------------------------------
// step_subcycle() reflux magnitude — called once per outer dt_base step,
// after _accumulate_amr_registers has summed every substep's contribution
// but BEFORE _apply_amr_reflux consumes reg_a/reg_b.
//
// Unlike check_total_dphi_balanced (valid only for step()'s single-pass,
// same-cadence junctions), step_subcycle() DEFERS the junction correction
// across a whole outer step by design (see _accumulate_amr_registers), so a
// nonzero reg_a[pi]+reg_b[pi] is expected, not a bug on its own -- reflux
// exists to correct exactly this. What's diagnostic is the RELATIVE size of
// that correction: printed (not thrown, since some correction is normal)
// so callers can compare rel across --max-level runs of the SAME synthetic
// IC/cell-count to see whether it grows with depth (pointing at a
// double-count feeding reg_a/reg_b) or stays flat (genuine, bounded
// coarse/fine physical mismatch).
// ---------------------------------------------------------------------------
inline void print_amr_reflux_magnitude(auto& cluster, std::vector<double> const& reg_a,
                                       std::vector<double> const& reg_b, char const* tag) {
  if (cluster.amr_correction_pairs.empty()) return;
  double total_abs_error = 0.0;
  double total_abs_scale = 0.0;

  // Bucket by (level_A, level_B) junction type -- level_A <= level_B by
  // convention below -- to see whether one specific junction depth (rather
  // than "more levels" in general) dominates the correction.
  struct SubBucket {
    double abs_error = 0.0;
    double abs_scale = 0.0;
    std::size_t n = 0;
  };
  struct Bucket {
    SubBucket matched, mismatched;
    std::size_t count_a_total = 0;
    std::size_t count_b_total = 0;
  };
  std::map<std::pair<int, int>, Bucket> by_level;
  // hex (k<8) vs sq (k>=8) split -- resolve_stale_external_proxies only
  // re-wires hex faces to a genuine same-level sibling; sq faces always go
  // through ancestor-substitution (proxy's own averaged .u), so a growing
  // same-level reflux residual is expected to concentrate in sq if that's
  // the mechanism, not a leftover hex staleness bug.
  SubBucket hex_bucket, sq_bucket;

  auto const& pairs = cluster.amr_correction_pairs;
  auto const* js    = cluster.amr_js.data();
  auto const* jf    = cluster.amr_js_face.data();
  for (std::size_t pi = 0; pi < reg_a.size(); ++pi) {
    auto const err   = std::abs(reg_a[pi] + reg_b[pi]);
    auto const scale = std::abs(reg_a[pi]) + std::abs(reg_b[pi]);
    total_abs_error += err;
    total_abs_scale += scale;

    auto const& p  = pairs[pi];
    auto const g_a = cluster.leaf_indices[static_cast<std::size_t>(js[p.begin])];
    auto const g_b = cluster.leaf_indices[static_cast<std::size_t>(js[p.mid])];
    int lvl_a      = cluster[g_a].level;
    int lvl_b      = cluster[g_b].level;
    auto key       = lvl_a <= lvl_b ? std::make_pair(lvl_a, lvl_b) : std::make_pair(lvl_b, lvl_a);
    auto& b        = by_level[key];
    auto const count_a = static_cast<std::size_t>(p.mid - p.begin);
    auto const count_b = static_cast<std::size_t>(p.end - p.mid);
    b.count_a_total += count_a;
    b.count_b_total += count_b;
    auto& sb = (count_a != count_b) ? b.mismatched : b.matched;
    sb.abs_error += err;
    sb.abs_scale += scale;
    sb.n += 1;

    auto& hs = (jf[p.begin] < 8) ? hex_bucket : sq_bucket;
    hs.abs_error += err;
    hs.abs_scale += scale;
    hs.n += 1;
  }

  double const rel = total_abs_scale > 0.0 ? total_abs_error / total_abs_scale : 0.0;
  std::fprintf(stderr,
               "[reflux-probe] tag=%s n_pairs=%zu total_abs_error=%.6e "
               "total_abs_scale=%.6e rel=%.6e\n",
               tag, reg_a.size(), total_abs_error, total_abs_scale, rel);
  for (auto const& [key, b] : by_level) {
    double const m_rel = b.matched.abs_scale > 0.0 ? b.matched.abs_error / b.matched.abs_scale : 0.0;
    double const x_rel =
        b.mismatched.abs_scale > 0.0 ? b.mismatched.abs_error / b.mismatched.abs_scale : 0.0;
    std::fprintf(stderr,
                 "  [reflux-probe]   junction L%d/L%d: count_a_total=%zu count_b_total=%zu\n"
                 "      matched:    n=%zu abs_error=%.6e abs_scale=%.6e rel=%.6e\n"
                 "      mismatched: n=%zu abs_error=%.6e abs_scale=%.6e rel=%.6e\n",
                 key.first, key.second, b.count_a_total, b.count_b_total, b.matched.n,
                 b.matched.abs_error, b.matched.abs_scale, m_rel, b.mismatched.n,
                 b.mismatched.abs_error, b.mismatched.abs_scale, x_rel);
  }
  double const hex_rel = hex_bucket.abs_scale > 0.0 ? hex_bucket.abs_error / hex_bucket.abs_scale : 0.0;
  double const sq_rel  = sq_bucket.abs_scale > 0.0 ? sq_bucket.abs_error / sq_bucket.abs_scale : 0.0;
  std::fprintf(stderr,
               "  [reflux-probe]   hex(k<8): n=%zu abs_error=%.6e abs_scale=%.6e rel=%.6e\n"
               "  [reflux-probe]   sq(k>=8): n=%zu abs_error=%.6e abs_scale=%.6e rel=%.6e\n",
               hex_bucket.n, hex_bucket.abs_error, hex_bucket.abs_scale, hex_rel,
               sq_bucket.n, sq_bucket.abs_error, sq_bucket.abs_scale, sq_rel);
}

}  // namespace equilibrium::dst::discrete

#endif /* EQUILIBRIUM_CORE_DATA_LATTICE_ADAPTIVE_LATTICE_DEBUG_HPP */
