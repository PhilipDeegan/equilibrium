import unittest

import numpy as np


class LazyTest(unittest.TestCase):
    def test_lazy(self):
        N = int(1e8)
        l0 = np.zeros(N) + 1
        l1 = np.zeros(N) + 2
        l2 = np.zeros(N) + 3
        l3 = np.zeros(N) + 4
        l4 = np.zeros(N) + 5

        a4 = l0 * l1 + l2 * l3 + l4 * l1 + l2 * l3 + l4 * l1
        print(a4[0], a4[-1])

        a4 = l0 * l1 + l2 + l3 + l4 * l1 + l2 * l3 + l4 * l1
        print(a4[0], a4[-1])
