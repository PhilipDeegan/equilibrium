#ifndef _EQUILIBRIUM_CORE_DATA_LATTICE_CONTINUOUS_DAMN_CLUSTERS_MPI_HPP_
#define _EQUILIBRIUM_CORE_DATA_LATTICE_CONTINUOUS_DAMN_CLUSTERS_MPI_HPP_

// continuous/damn_clusters.hpp over MPI ranks: each rank holds its patches' clusters, ghosts are
// filled from any rank by damn::mpi::copy (collective over MPI_COMM_WORLD)

#include "eq/core/data/lattice/continuous/damn_clusters.hpp"

#include "damn/mpi/copy.hpp"
#include "damn/mpi/mpi_utils.hpp"
#include "damn/mpi/topology.hpp"

#include <array>
#include <string>
#include <vector>

namespace equilibrium::dst::continuous::clusters::mpi {

inline void exchange(Clusters& c, std::vector<std::string> const& names) {
  damn::mpi::copy(damn::amr::fill_ghosts, c.neighbours, names, c.data);
}

// clusters::build, build_periodic, build_box over every rank, this one's patches
[[nodiscard]] inline Clusters build(AdaptiveLattice& lat, Keys const& keys,
                                    std::array<int, 3> const& patch_cells, double const d1 = 1.0,
                                    double const u0 = 0.0, double const lam0 = 0.0) {
  return clusters::build(lat, keys, patch_cells, d1, u0, lam0, damn::mpi::rank(),
                         damn::mpi::topology());
}

[[nodiscard]] inline Clusters build_periodic(AdaptiveLattice& lat, std::array<int, 3> const& P,
                                             std::array<int, 3> const& patch_cells,
                                             double const d1 = 1.0, double const u0 = 0.0,
                                             double const lam0 = 0.0) {
  return clusters::build_periodic(lat, P, patch_cells, d1, u0, lam0, damn::mpi::rank(),
                                  damn::mpi::topology());
}

[[nodiscard]] inline Clusters build_box(AdaptiveLattice& lat, std::array<int, 3> const& N,
                                        std::array<int, 3> const& patch_cells,
                                        double const d1 = 1.0, double const u0 = 0.0,
                                        double const lam0 = 0.0) {
  return clusters::build_box(lat, N, patch_cells, d1, u0, lam0, damn::mpi::rank(),
                             damn::mpi::topology());
}

inline void step_all(AdaptiveLattice& lat, Clusters& c, discrete::TimeStepper const& ts) {
  clusters::step_all(lat, c, ts, [](Clusters& cs) { mpi::exchange(cs, u_names); });
}

}  // namespace equilibrium::dst::continuous::clusters::mpi

#endif  // _EQUILIBRIUM_CORE_DATA_LATTICE_CONTINUOUS_DAMN_CLUSTERS_MPI_HPP_
