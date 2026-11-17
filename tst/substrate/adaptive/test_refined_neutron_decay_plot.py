"""
Sanity checks for refined_neutron_decay.plot_summary against synthetic
sweep results -- no lattice/sim required.

Covers the bug this was pulled out to fix: a sweep run for hours, then
plt.savefig() throwing FileNotFoundError at the very end because the
--out directory didn't exist yet (see equilibrium.utils.ensure_dir /
equilibrium.plot.savefig).
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from run.substrate.adaptive.refined_neutron_decay import R_SCAN, plot_summary


def _fake_results(cross_scales, n=10):
    return {
        cs: dict(
            t=np.linspace(0, 1, n),
            H=np.linspace(1.0, 0.5, n),
            circ=np.random.default_rng(0).random((n, len(R_SCAN))),
            lam_mean=np.linspace(0.1, 0.2, n),
        )
        for cs in cross_scales
    }


class TestPlotSummary(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_writes_file(self):
        out = self.tmp / "summary.png"
        plot_summary(_fake_results([1e-3, 2e-3]), [1e-3, 2e-3], str(out))
        self.assertTrue(out.is_file())

    def test_missing_nested_output_dir_is_created(self):
        out = self.tmp / "does" / "not" / "exist" / "summary.png"
        self.assertFalse(out.parent.is_dir())
        plot_summary(_fake_results([1e-3]), [1e-3], str(out))
        self.assertTrue(out.is_file())

    def test_single_cross_scale(self):
        out = self.tmp / "one_cs.png"
        plot_summary(_fake_results([5e-4]), [5e-4], str(out))
        self.assertTrue(out.is_file())


if __name__ == "__main__":
    unittest.main()
