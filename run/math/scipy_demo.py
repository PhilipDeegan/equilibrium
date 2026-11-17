import numpy as np
import scipy.constants as sp

import equilibrium as eq
from eq.physical.electrical import electric as el

np.set_printoptions(precision=22)

PI = np.longdouble("3.141592653589793238462643383279502884")

log = eq.log.getLogger(__file__)


def format(*args):
    def f(a):
        return f"{a:<25}"
        # return f"{a:0>12.12f}"

    if len(args) == 1:
        return f(args[0])
    return [f(a) for a in args]


def info(*args):
    return " ".join(str(format(a)) if type(a) is float else str(a) for a in args)


def main():
    us = (-sp.e * sp.hbar) / (2 * sp.m_e)
    log.info(info(us))

    # m = el.bohr_electron_velocity(1, 1)
    v = el.bohr_electron_velocity(1, 1)
    r = el.bohr_shell_radius(1, 1)
    L = sp.m_e * v * r

    ul = (-sp.e * L) / (2 * sp.m_e)

    log.info(info(ul))
    log.info(info(us + ul))

    μ0 = PI * 4 * 1e-7  # sp.mu_0
    log.info(info(sp.mu_0, μ0))

    ε0 = 1 / (μ0 * sp.c**2)
    log.info(info(sp.epsilon_0, ε0))

    c2 = 1 / (μ0 * ε0)
    log.info(info(sp.c**2, c2))

    ε0 = 1 / (μ0 * (1 / (μ0 * ε0)))

    log.info(info(sp.epsilon_0, ε0))

    c = 1 / np.sqrt(μ0 * ε0)
    log.info(info(sp.c, c))

    log.info(info(1 / np.sqrt((μ0) * (ε0))))  # smaller
    # log.info(info(1 / np.sqrt((μ0 + 1e-8) * (ε0 + 1e-7))))  # smaller
    # log.info(info(1 / np.sqrt((μ0 + 1e-8) * (ε0))))  # smaller
    # log.info(info(1 / np.sqrt((μ0) * (ε0 + 1e-7))))  # smaller

    log.info(type(μ0))
    log.info(info((μ0 - 1e-7) * (ε0 - 1e-7)))
    # log.info(info(1 / np.sqrt((μ0 - 1e-7) * (ε0 - 1e-7))))
    # log.info(info(1 / np.sqrt((μ0 - 1e-7) * (ε0 - 1e-7))))
    # log.info(info(1 / np.sqrt((μ0 - 1e-7) * (ε0 - 1e-7))))
    # log.info(info(1 / np.sqrt((μ0 - 1e-7) * (ε0 - 1e-7))))

    log.info(info(1 / np.sqrt((μ0 - 1e-4) * (ε0 - 1e-7))))
    log.info(info(1 / np.sqrt((μ0) * (ε0 - 1e-12))))
    log.info(info(1 / np.sqrt((μ0 - 1e-7) * (ε0))))
    # log.info(info(1 / np.sqrt((μ0) * (ε0 - 1e-7))))

    ℏ = sp.hbar
    e = 0.30282212088 * np.sqrt(ε0 * ℏ * c)  # sp.e
    log.info(info(sp.e, e))

    α = e**2 / (PI * 4 * ε0 * ℏ * c)
    log.info(info(sp.alpha, α))

    μ0 = PI * 4 * ((α * ℏ) / (e**2 * c))
    log.info(info(sp.mu_0, μ0))

    ε02 = 1 / (μ0 * (1 / μ0))
    log.info(info(ε0**2, ε02))


if __name__ == "__main__":
    main()
