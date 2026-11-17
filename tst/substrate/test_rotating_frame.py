"""
Verify equilibrium.substrate.rotating_frame's vector reconstruction/
projection primitives against a known analytic gradient, before trusting
them in any rotating-frame correction-term experiment.
"""

import unittest

import numpy as np

from equilibrium.substrate.lattice import SubstrateLattice
from equilibrium.substrate.rotating_frame import (
    face_design,
    project_to_faces,
    reconstruct_vector,
)
from equilibrium.substrate.traveling_wave import face_offsets

R = 12.0
HALF_LEN = 12.0


class TestRotatingFrameReconstruction(unittest.TestCase):
    def test_reconstructs_known_linear_gradient(self):
        lat = SubstrateLattice.cylinder(R, HALF_LEN)
        offsets, _ = face_offsets(lat)
        hop_len, pinv = face_design(offsets)

        a = np.array([0.3, -1.7, 2.1])
        u_lin = lat.pos @ a  # exact linear field, true gradient == a everywhere

        valid = lat._valid
        nb = lat.nb
        u_nb = np.where(valid, u_lin[np.where(valid, nb, 0)], u_lin[:, None])
        du_face = u_nb - u_lin[:, None]

        grad = reconstruct_vector(du_face, hop_len, pinv, valid=valid)

        interior = valid.all(axis=1)
        self.assertGreater(interior.sum(), 50)
        np.testing.assert_allclose(grad[interior], np.broadcast_to(a, grad[interior].shape), atol=1e-10)

    def test_project_is_inverse_of_reconstruct_on_curl_free_phi(self):
        lat = SubstrateLattice.cylinder(R, HALF_LEN)
        offsets, _ = face_offsets(lat)
        hop_len, pinv = face_design(offsets)

        a = np.array([1.0, 0.5, -0.4])
        u_lin = lat.pos @ a
        valid = lat._valid
        nb = lat.nb
        u_nb = np.where(valid, u_lin[np.where(valid, nb, 0)], u_lin[:, None])
        phi_lin = np.where(valid, u_nb - u_lin[:, None], 0.0)  # exact curl-free phi

        vec = reconstruct_vector(phi_lin, hop_len, pinv, valid=valid)
        phi_roundtrip = project_to_faces(vec, offsets)

        interior = valid.all(axis=1)
        np.testing.assert_allclose(phi_roundtrip[interior], phi_lin[interior], atol=1e-8)


if __name__ == "__main__":
    unittest.main()
