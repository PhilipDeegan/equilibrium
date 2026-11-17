#
#
#
# https://docs.scipy.org/doc/scipy/reference/constants.html

from dataclasses import dataclass, field

import numpy as np
import scipy.constants as sp

PI = np.longdouble("3.141592653589793238462643383279502884")


def phys_const(s):
    return sp.physical_constants[s][0]


@dataclass
class Planck:
    length: float = field(default_factory=lambda: phys_const("Planck length"))
    time: float = field(default_factory=lambda: phys_const("Planck time"))
    mass: float = field(default_factory=lambda: phys_const("Planck mass"))

    def h(self):
        return (PI * 2 * self.mass * (self.length**2)) / self.time

    def hbar(self):
        return (self.mass * (self.length**2)) / self.time
