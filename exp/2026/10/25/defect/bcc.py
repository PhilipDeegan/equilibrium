"""Periodic BCC box (two cubic sublattices A at integer sites, B at +1/2), discrete step() with face sigma.

Unique faces: 8 hex A(i) -> B(i + o - 1), o in {0,1}^3; 3 forward square per sublattice A(i) -> A(i + e_d).
phi stored once per face (outgoing from A for hex, forward for square); H = 1/2 sum u^2 + 1/2 sum_faces phi^2.
Per-cell sigma = s / S (integer size count s, background S); face sigma = mean of the pair.
"""

import itertools

import numpy as np

W_HEX, W_SQ = 3.0**0.5, 1.0
OFF = list(itertools.product((0, 1), repeat=3))
AX = (0, 1, 2)


def _sh(a, o):
    return np.roll(a, shift=tuple(1 - x for x in o), axis=AX)


def _shb(a, o):
    return np.roll(a, shift=tuple(x - 1 for x in o), axis=AX)


class Box:
    def __init__(self, n, sA=None, sB=None, S=1):
        self.n = n
        self.uA, self.uB = np.zeros(n), np.zeros(n)
        self.ph = np.zeros((8,) + tuple(n))
        self.sqA, self.sqB = np.zeros((3,) + tuple(n)), np.zeros((3,) + tuple(n))
        self.set_size(np.full(n, S) if sA is None else sA, np.full(n, S) if sB is None else sB, S)

    def set_size(self, sA, sB, S):
        gA, gB = np.asarray(sA, float) / S, np.asarray(sB, float) / S
        self.fh = np.stack([W_HEX * 0.5 * (gA + _sh(gB, o)) for o in OFF])
        self.fsA = np.stack([W_SQ * 0.5 * (gA + np.roll(gA, -1, axis=d)) for d in AX])
        self.fsB = np.stack([W_SQ * 0.5 * (gB + np.roll(gB, -1, axis=d)) for d in AX])

    def step(self, dt):
        cA = (self.fh * self.ph).sum(0)
        cB = -sum(_shb(self.fh[j] * self.ph[j], o) for j, o in enumerate(OFF))
        for d in AX:
            qa, qb = self.fsA[d] * self.sqA[d], self.fsB[d] * self.sqB[d]
            cA = cA + qa - np.roll(qa, 1, axis=d)
            cB = cB + qb - np.roll(qb, 1, axis=d)
        self.uA -= dt * cA
        self.uB -= dt * cB
        for j, o in enumerate(OFF):
            self.ph[j] -= dt * self.fh[j] * (_sh(self.uB, o) - self.uA)
        for d in AX:
            self.sqA[d] -= dt * self.fsA[d] * (np.roll(self.uA, -1, axis=d) - self.uA)
            self.sqB[d] -= dt * self.fsB[d] * (np.roll(self.uB, -1, axis=d) - self.uB)

    def energy(self):
        return 0.5 * (np.sum(self.uA**2) + np.sum(self.uB**2) + np.sum(self.ph**2) + np.sum(self.sqA**2) + np.sum(self.sqB**2))

    def fields(self):
        return [self.uA, self.uB, self.ph, self.sqA, self.sqB]


def diff_energy(a, b):
    return 0.5 * sum(np.sum((x - y) ** 2) for x, y in zip(a.fields(), b.fields()))
