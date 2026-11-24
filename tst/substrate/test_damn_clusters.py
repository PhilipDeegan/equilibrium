"""
lattice.continuous.Clusters: a box as several clusters (damn patches, ghosts exchanged by damn),
built and stepped from python, against one cluster of the same sites (bit for bit, u and phi):
periodic and bounded, on the main thread and with leased threads; and over MPI ranks
(test_damn_clusters_mpi.py under mpirun -n 2, oversubscribed)
"""

import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from equilibrium import lattice as _lat

ct = _lat.continuous
H = 1.0 / np.sqrt(3.0)


def u_init(keys):
    k = np.asarray(keys)
    return 0.01 * ((7 * k[:, 0] + 13 * k[:, 1] + 29 * k[:, 2]) % 17) - 0.05


def lam_init(keys):
    k = np.asarray(keys)
    return 0.1 * ((k[:, 0] + 2 * k[:, 1] + 3 * k[:, 2]) & 3)


def wrap(keys, P):
    if P is None:
        return keys
    P = np.asarray(P)
    return (keys + P // 2) % P - P // 2


def reference(P=None, N=None, steps=20, dt=0.05):
    """{key: (u, phi)} of one cluster after steps"""
    lat = ct.AdaptiveLattice()
    if P is not None:
        ct.build_bcc_box_periodic(lat[0], *P)
    else:
        ct.build_bcc_box(lat[0], *(2 * n * H for n in N))
    lat.rebuild_all()
    cl = lat[0]
    keys = np.rint(cl.leaf_pos / H).astype(np.int64)
    cl.set_leaf_u(u_init(keys))
    cl.set_leaf_lam(lam_init(keys))
    for _ in range(steps):
        lat.step(0, dt)
    return {
        tuple(k): (u, np.ascontiguousarray(phi))
        for k, u, phi in zip(keys, cl.leaf_u, cl.leaf_phi.T)
    }


def run_clusters(exec_model, patch_cells, P=None, N=None, steps=20, dt=0.05):
    lat = ct.AdaptiveLattice()
    if P is not None:
        c = ct.Clusters.periodic(lat, P, patch_cells, exec_model)
    else:
        c = ct.Clusters.box(lat, N, patch_cells, exec_model)
    keys = wrap(c.owned_keys.astype(np.int64), P)
    c.set_owned_lam(lam_init(keys))
    c.set_owned_u(u_init(keys))
    c.step(dt, steps)
    return c, keys


def mismatches(ref, keys, u, phi):
    bad = 0
    for k, uk, pk in zip(keys, u, phi):
        r = ref[tuple(k)]
        bad += int(r[0].tobytes() != uk.tobytes()) + int(r[1].tobytes() != pk.tobytes())
    return bad


class TestDamnClusters(unittest.TestCase):
    def check(self, threads, P=None, N=None):
        ref = reference(P, N)
        exec_model = _lat.ExecutionModel(usable_threads=threads)
        for patch_cells in ([8, 8, 8], [3, 2, 4], [1, 1, 1]):
            c, keys = run_clusters(exec_model, patch_cells, P, N)
            self.assertEqual(c.n_owned, len(ref))
            self.assertEqual(len({tuple(k) for k in keys}), len(ref))
            self.assertEqual(
                mismatches(ref, keys, c.owned_u, c.owned_phi), 0, (patch_cells, threads)
            )
        self.assertEqual(exec_model.threads_used, 1)

    def test_periodic(self):
        for threads in (1, 4):
            self.check(threads, P=[16, 12, 8])

    def test_bounded(self):
        for threads in (1, 4):
            self.check(threads, N=[7, 6, 5])

    def test_execution_model(self):
        e = _lat.ExecutionModel()
        self.assertEqual(
            (e.mpi_ranks, e.usable_threads, e.threads_used, e.schedulable), (0, 1, 1, 1)
        )
        self.assertFalse(e.mpi() or e.threaded() or e.mpi_only())
        self.assertEqual(_lat.ExecutionModel(usable_threads=8).schedulable, 9)
        self.assertEqual(
            _lat.ExecutionModel(usable_threads=8, max_scheduled=2.0).schedulable, 16
        )
        with self.assertRaises(RuntimeError):
            _lat.ExecutionModel(usable_threads=0)
        with self.assertRaises(RuntimeError):
            _lat.ExecutionModel(mpi=True)  # before mpi_init

    @unittest.skipUnless(shutil.which("mpirun"), "no mpirun")
    def test_mpi(self):
        script = Path(__file__).with_name("damn_clusters_mpi.py")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        out = subprocess.run(
            ["mpirun", "--oversubscribe", "-n", "2", sys.executable, "-u", str(script)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("PASS", out.stdout)


if __name__ == "__main__":
    unittest.main()
