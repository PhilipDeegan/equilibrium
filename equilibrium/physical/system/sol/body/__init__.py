from equilibrium.physical.body import GasGiant

from .earth import Earth
from .jupiter import Jupiter
from .mars import Mars
from .mercury import Mercury
from .venus import Venus


# move/fix following
class Saturn(GasGiant):
    radius: float = 1  # update


class Neptune(GasGiant):
    radius: float = 1  # update


class Uranus(GasGiant):
    radius: float = 1  # update


__all__ = [
    "Earth",
    "Jupiter",
    "Mars",
    "Mercury",
    "Neptune",
    "Saturn",
    "Uranus",
    "Venus",
]
