#
#
# import equilibrium as eq
# from equilibrium import cppa as cppa
# from equilibrium import cppb as cppb

# N = int(1e8)
# l0 = eq.lz(N, 1)
# l1 = eq.lz(N, 2)
# l2 = eq.lz(N, 3)
# l3 = eq.lz(N, 4)
# l4 = eq.lz(N, 5)

# lz = l0 * l1 + l2 * l3 + l4 * l1 + l2 * l3 + l4 * l1
# a4 = cppa.exec_array_copy(lz)
# print(a4[0], a4[-1], lz.size())

# a4 = cppb.exec_array_copy(l0 * l1 + l2 + l3 + l4 * l1 + l2 * l3 + l4 * l1)
# print(a4[0], a4[-1], lz.size())
