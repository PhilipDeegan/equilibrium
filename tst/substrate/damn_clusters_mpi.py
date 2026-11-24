"""
test_damn_clusters.py's comparison over MPI ranks (any number): each rank's owned sites against one
cluster of the same sites, the ranks' owned sites together every site once
  mpirun -n 3 python3 -u tst/substrate/damn_clusters_mpi.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from test_damn_clusters import mismatches, reference, run_clusters

from equilibrium import lattice as _lat


def main():
    _lat.mpi_init()
    rank, size = _lat.mpi_rank(), _lat.mpi_size()
    exec_model = _lat.ExecutionModel(usable_threads=1, mpi=True)
    assert exec_model.mpi_ranks == size and exec_model.mpi_only()

    bad = 0
    for P, N in (([16, 12, 8], None), (None, [7, 6, 5])):
        ref = reference(P, N)
        for patch_cells in ([4, 3, 2], [1, 1, 1]):
            c, keys = run_clusters(exec_model, patch_cells, P, N)
            bad += mismatches(ref, keys, c.owned_u, c.owned_phi)
            bad += int(_lat.mpi_sum(c.n_owned) != len(ref))
    bad = int(_lat.mpi_sum(bad))
    if rank == 0:
        print("PASS" if bad == 0 else f"{bad} mismatches over {size} ranks", flush=True)
    _lat.mpi_finalize()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
