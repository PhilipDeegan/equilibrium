# very simple, comparison via perf stat

import unittest

import equilibrium as eq

try:
    from equilibrium import cpp as cpp  # pylint: disable = no-name-in-module

    class LazyTest(unittest.TestCase):
        def test_lazy(self):
            N = int(1e8)
            l0 = eq.lz(N, 1)
            l1 = eq.lz(N, 2)
            l2 = eq.lz(N, 3)
            l3 = eq.lz(N, 4)
            l4 = eq.lz(N, 5)

            lz = l0 * l1 + l2 * l3 + l4 * l1 + l2 * l3 + l4 * l1
            a4 = eq.lz(lz)
            # ev = cpp.LzEval_d(lz.lz)
            # ev.compile()
            # ev.write_compilable("src/cppa.cpp")
            # ev = None
            print(a4[0], a4[-1])

            lz = l0 * l1 + l2 + l3 + l4 * l1 + l2 * l3 + l4 * l1
            # ev = cpp.LzEval_d(lz.lz)
            # ev.compile()
            # ev.write_compilable("src/cppb.cpp")
            # ev = None
            a4 = eq.lz(lz)
            print(a4[0], a4[-1])

except ImportError as e:
    raise ValueError(str(e))
finally:
    ...
