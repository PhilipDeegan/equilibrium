"""
Reciprocal boundary condition: whatever u leaves the domain is injected back at randomly chosen
boundary cells (exponential weights, renormalised), so total u is held fixed without reflecting
at the face it left through and without draining. Pure global u bookkeeping: no local wave
absorption (alone it reflects; compose with a sponge). Acts on u only, so phi antisymmetry is
untouched.

`thermalize` adds zero-mean noise at the boundary: ambient equilibration only, never alongside a
decay probe (doc/mems/feedback_saha_noise_scope.md). A thermal ambient state is an initial
condition (equilibrium/substrate/ambient.py), not something noise maintains.

Two front ends over the same rule: ReciprocalBC for SubstrateLattice-like objects (u, nb,
fixed_u arrays), ReciprocalBCCluster for C++ AdaptiveCellCluster (leaf_u / set_leaf_u). The
cluster's nb < 0 also marks AMR dead faces, so on refined clusters "boundary" includes them.
"""

import numpy as np


class _Reciprocal:
    def __init__(self, u, nb, fixed_u, u_target=None, rng=None, seed=0):
        boundary = np.any(nb < 0, axis=1) & ~fixed_u
        self._bc_idx = np.where(boundary)[0]
        self._u_target = float(u_target if u_target is not None else u.sum())
        self._rng = rng or np.random.default_rng(seed)

    @property
    def n_boundary(self) -> int:
        return len(self._bc_idx)

    def _compensated(self, u):
        deficit = self._u_target - float(u.sum())
        if self.n_boundary == 0 or abs(deficit) < 1e-15:
            return None
        weights = self._rng.exponential(size=self.n_boundary)
        weights /= weights.sum()
        u = u.copy()
        u[self._bc_idx] += deficit * weights
        return u

    def _noised(self, u, amplitude):
        if self.n_boundary == 0 or amplitude <= 0:
            return None
        noise = amplitude * self._rng.standard_normal(size=self.n_boundary)
        noise -= noise.mean()
        u = u.copy()
        u[self._bc_idx] += noise
        return u


class ReciprocalBC(_Reciprocal):
    def __init__(self, lat, u_target: float = None, rng=None):
        super().__init__(lat.u, lat.nb, lat.fixed_u, u_target, rng)

    def compensate(self, lat):
        """Inject back whatever u was lost since construction / the last set_target."""
        u = self._compensated(lat.u)
        if u is not None:
            lat.u[:] = u

    def thermalize(self, lat, amplitude: float):
        """Zero-mean Gaussian noise at boundary cells (total u unchanged)."""
        u = self._noised(lat.u, amplitude)
        if u is not None:
            lat.u[:] = u

    def set_target(self, lat):
        self._u_target = float(lat.u.sum())


class ReciprocalBCCluster(_Reciprocal):
    def __init__(self, cl, u_target: float = None, rng=None):
        super().__init__(
            np.array(cl.leaf_u), np.array(cl.leaf_nb), np.array(cl.leaf_fixed_u).astype(bool), u_target, rng
        )

    def compensate(self, cl):
        u = self._compensated(np.array(cl.leaf_u))
        if u is not None:
            cl.set_leaf_u(u)

    def thermalize(self, cl, amplitude: float):
        u = self._noised(np.array(cl.leaf_u), amplitude)
        if u is not None:
            cl.set_leaf_u(u)

    def set_target(self, cl):
        self._u_target = float(np.array(cl.leaf_u).sum())
