import scipy as sp

from equilibrium.log import getLogger
from equilibrium.math import constants as consts

logger = getLogger(__name__)


def rydberg_joules():
    # https://en.wikipedia.org/wiki/Rydberg_constant
    e = sp.constants.e
    h = sp.constants.h
    ε0 = sp.constants.epsilon_0
    m_e = sp.constants.m_e
    return e**4 * m_e / (8 * ε0**2 * h**2)


_rydberg_joules = rydberg_joules()


class Electron:
    def __init__(self):
        logger.info("Alloc " + str(type(self)))


def bohr_shell_radius_full(n, m, Z):
    """
    n : int   = shell number
    m : float = mass
    z : int   = atomic number (proton count)
    """
    return (n**2) * (sp.h**2) / (4 * (consts.PI**2) * m * (sp.e**2))


def bohr_shell_radius(n: int, Z: int):
    """
    n : int   = shell number
    z : int   = atomic number (proton count)
    """
    return 0.53 * ((n**2) / Z) * 1e-10


def bohr_electron_velocity(n: int, Z: int):
    """
    n : int   = shell number
    z : int   = atomic number (proton count)
    """
    return 2.186 * 1e6 * (Z / n)


def bohr_electron_energy(n: int, Z: int):
    """
    n : int   = shell number
    z : int   = atomic number (proton count)
    """
    return _rydberg_joules * ((Z**2) / (n**2))


class ElectronShell:
    def __init__(self, n, Z):
        self.Zn = n
        self.Z = Z
        self.v = bohr_electron_velocity(1, Z)  # meters per second
        self.e = bohr_electron_energy(1, Z)  # electron volts


class ElectronShellK:
    def __init__(self, Z):
        super().__init__(1, Z)


class ElectronShellL:
    def __init__(self, Z):
        super().__init__(2, Z)


class ElectronShellM:
    def __init__(self, Z):
        super().__init__(3, Z)


class ElectronShellN:
    def __init__(self, Z):
        super().__init__(4, Z)


class Electrons:
    def __init__(self):
        logger.info("Alloc " + str(type(self)))


def electron_shell_jump_photon_wave_length(dE: float):
    """
    dE : float   = energy delta
    """
    return sp.constants.h * sp.constants.c / dE
