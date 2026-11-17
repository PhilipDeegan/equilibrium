/*
Standalone diagnostic: verify that AMR reflux correction entries
(AdaptiveCellCluster::amr_correction_pairs / amr_js / amr_js_face) never
reference the same (leaf, own-face) slot twice, on a REAL saved lattice
(not a synthetic test mesh). Answers directly: at a spheromak's level
boundary a single cell can sit at several coarse-fine/proxy-proxy junctions
on different faces -- this checks that step_subcycle's per-pair
accumulation (_accumulate_amr_registers / _apply_amr_reflux) never sums or
applies the SAME physical face twice across those junctions.

Usage: check_amr_junctions <path.h5>
*/

#include "eq/core/data/lattice/adaptive_lattice.hpp"
#include "eq/core/data/lattice/bcc_geometry.hpp"
#include "eq/core/data/lattice/cluster_io.hpp"

#include "highfive/H5File.hpp"

#include <cstdio>
#include <map>
#include <utility>

using namespace equilibrium::dst;
using namespace equilibrium::dst::discrete;

int main(int argc, char** argv) {
  std::string const path = argc > 1 ? argv[1] : "/bk/l1_wider_margin_ic220.h5";

  std::printf("loading %s ...\n", path.c_str());
  HighFive::File file(path, HighFive::File::ReadOnly);
  std::size_t nc = 0;
  file.getAttribute("n_clusters").read(nc);

  AdaptiveLattice lat;
  lat.clusters.clear();
  for (std::size_t i = 0; i < nc; ++i) {
    auto grp = file.getGroup("cluster_" + std::to_string(i));
    lat.clusters.push_back(load_cluster(grp));
    lat.rebuild_leaf_indices(lat.clusters.back());
  }
  std::printf("loaded %zu cluster(s)\n", nc);

  bool any_dup = false;
  for (std::size_t ci = 0; ci < lat.clusters.size(); ++ci) {
    auto& cl = lat.clusters[ci];

    std::map<std::uint8_t, std::size_t> level_counts;
    for (std::int32_t g : cl.leaf_indices) level_counts[cl[g].level]++;
    std::printf("cluster %zu: %zu leaves, %zu non-leaf (proxy) cells, %zu correction pairs\n", ci,
                cl.leaf_indices.size(), cl.n_non_leaf_cells, cl.amr_correction_pairs.size());
    for (auto const& [lvl, n] : level_counts)
      std::printf("  level %u: %zu leaves\n", static_cast<unsigned>(lvl), n);

    // The invariant: every (leaf global id, own-face index) slot referenced
    // by amr_js/amr_js_face belongs to exactly one pair-side entry. If a
    // single cell touches N distinct junctions (N different faces), each
    // face is a SEPARATE slot -- fine. A count > 1 on the SAME slot means
    // two different pairs (or the same pair twice) are about to sum/apply
    // that one physical temp_phi/phi entry more than once.
    std::map<std::pair<std::int32_t, std::uint8_t>, int> seen;
    // Also track, per leaf, how many DISTINCT junction faces it participates
    // in -- this is the real-world "multiple junctions per cell" count the
    // spheromak boundary is expected to produce, reported for context.
    std::map<std::int32_t, int> junctions_per_leaf;
    for (auto const& p : cl.amr_correction_pairs) {
      for (std::int32_t i = p.begin; i < p.end; ++i) {
        auto const leaf_g = cl.leaf_indices[cl.amr_js[i]];
        auto const face_k = cl.amr_js_face[i];
        seen[{leaf_g, face_k}]++;
        junctions_per_leaf[leaf_g]++;
      }
    }

    int max_count = 0;
    std::size_t n_dup_slots = 0;
    std::int32_t dup_g = -1;
    std::uint8_t dup_k = 0;
    for (auto const& [key, count] : seen) {
      if (count > 1) {
        ++n_dup_slots;
        if (count > max_count) {
          max_count = count;
          dup_g     = key.first;
          dup_k     = key.second;
        }
      }
    }

    int max_junctions_per_leaf = 0;
    std::int32_t busiest_leaf = -1;
    std::size_t n_multi_junction_leaves = 0;
    for (auto const& [g, n] : junctions_per_leaf) {
      if (n > 1) ++n_multi_junction_leaves;
      if (n > max_junctions_per_leaf) {
        max_junctions_per_leaf = n;
        busiest_leaf           = g;
      }
    }

    std::printf("  %zu distinct (leaf,face) slots referenced across all pairs\n", seen.size());
    std::printf("  %zu leaves touch >1 junction face (max=%d, at leaf g=%d)\n",
                n_multi_junction_leaves, max_junctions_per_leaf, busiest_leaf);

    if (n_dup_slots > 0) {
      any_dup = true;
      std::fprintf(stderr,
                    "  DUP FOUND: %zu (leaf,face) slots referenced by >1 pair-entry "
                    "(worst: leaf g=%d face k=%u referenced %d times)\n",
                    n_dup_slots, dup_g, dup_k, max_count);
    } else {
      std::printf("  OK: no (leaf,face) slot is referenced by more than one pair-entry\n");
    }
  }

  std::printf(any_dup ? "RESULT: DOUBLE-COUNT DETECTED\n" : "RESULT: NO DOUBLE-COUNT\n");
  return any_dup ? 1 : 0;
}
