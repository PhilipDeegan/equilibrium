/*
Diagnostic: measure the absolute error the coarse-fine junction introduces
on a REAL checkpoint's actual current field.

Loads the same checkpoint twice and steps each once with the identical dt:
  - "real":  AdaptiveLattice::enable_cross_level_gradient_correction = false
             (production default -- exactly what every existing run used)
  - "ghost": ...= true -- at every cross-level face, both sides of
             (nb_u - u_src) are replaced by a fictitious same-resolution
             value reconstructed via a local gradient fit at the fine
             cell's own geometric face position (_gradient_corrected_value,
             adaptive_lattice.hpp). Nothing persists -- the ghost value is
             recomputed fresh from the real field every call, it is not an
             independently-evolved cell -- it exists only to give the flux
             formula a "what if this were same-resolution" reference.

The per-leaf |u_ghost - u_real| / |phi_ghost - phi_real| after one step is
therefore a direct, real-field measurement of the level border's own
absolute error, decoupled from how far the source's tail has decayed by
the time it reaches the junction (unlike the margin=10 vs margin=30
comparison, which conflates the two).

Usage: measure_junction_ghost_error <checkpoint.h5> [dt]
*/

#include "eq/core/data/lattice/adaptive_evolver.hpp"
#include "eq/core/data/lattice/adaptive_lattice.hpp"
#include "eq/core/data/lattice/bcc_geometry.hpp"
#include "eq/core/data/lattice/cluster_io.hpp"

#include "highfive/H5File.hpp"

#include <cmath>
#include <cstdio>

using namespace equilibrium::dst;
using namespace equilibrium::dst::discrete;

static AdaptiveLattice load_one(std::string const& path) {
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
  return lat;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: measure_junction_ghost_error <checkpoint.h5> [dt]\n");
    return 2;
  }
  std::string const path = argv[1];
  double const dt         = argc > 2 ? std::atof(argv[2]) : 5.128205e-03;

  std::printf("loading two copies of %s (dt=%.6e) ...\n", path.c_str(), dt);
  auto lat_real  = load_one(path);
  auto lat_ghost = load_one(path);
  std::printf("loaded. %zu cluster(s)\n", lat_real.clusters.size());

  auto ts = TimeStepper::make_by_level(dt);

  AdaptiveLattice::enable_cross_level_gradient_correction = false;
  { AdaptiveLatticeEvolver ev(lat_real); ev.step(ts); }

  AdaptiveLattice::enable_cross_level_gradient_correction = true;
  { AdaptiveLatticeEvolver ev(lat_ghost); ev.step(ts); }
  AdaptiveLattice::enable_cross_level_gradient_correction = false;

  for (std::size_t ci = 0; ci < lat_real.clusters.size(); ++ci) {
    auto& cr = lat_real.clusters[ci];
    auto& cg = lat_ghost.clusters[ci];
    if (cr.leaf_indices.empty()) continue;
    if (cr.leaf_indices != cg.leaf_indices) {
      std::fprintf(stderr, "cluster %zu: leaf_indices mismatch between real/ghost -- abort\n", ci);
      continue;
    }

    double max_du_junction = 0.0, sum_du_junction = 0.0;
    double max_du_interior = 0.0, sum_du_interior = 0.0;
    std::int32_t worst_g   = -1;
    std::size_t n_junction = 0, n_interior = 0;

    for (std::size_t j = 0; j < cr.leaf_indices.size(); ++j) {
      auto const g   = cr.leaf_indices[j];
      auto const c_r = cr[g];
      auto const c_g = cg[g];
      bool at_junction = false;
      for (std::size_t k = 0; k < CellSlab::NF; ++k) {
        auto const nbg = c_r.nb[k];
        if (nbg >= 0 && cr[nbg].L != c_r.L) { at_junction = true; break; }
      }
      double const du = std::fabs(c_g.u - c_r.u);
      if (at_junction) {
        ++n_junction;
        sum_du_junction += du;
        if (du > max_du_junction) { max_du_junction = du; worst_g = g; }
      } else {
        ++n_interior;
        sum_du_interior += du;
        max_du_interior = std::max(max_du_interior, du);
      }
    }

    std::printf(
        "cluster %zu: %zu leaves, %zu at a coarse-fine junction, %zu interior\n"
        "  junction:  mean|du|=%.6e  max|du|=%.6e  (worst leaf g=%d)\n"
        "  interior:  mean|du|=%.6e  max|du|=%.6e\n",
        ci, cr.leaf_indices.size(), n_junction, n_interior,
        n_junction ? sum_du_junction / static_cast<double>(n_junction) : 0.0, max_du_junction,
        worst_g, n_interior ? sum_du_interior / static_cast<double>(n_interior) : 0.0,
        max_du_interior);
  }
  return 0;
}
