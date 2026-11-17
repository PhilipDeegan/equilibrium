import pydantic

import equilibrium.utils as eq_utils

# import equilibrium.physical.terms as terms
# from equilibrium.utils import T


class Body(pydantic.BaseModel):
    def __str__(self, s="", indent=0):
        return eq_utils.to_string(self)


class Sun(Body):
    radius: float
    mass: float


class Planet(Body): ...


class GasGiant(Body): ...


class Moon(Planet):
    name: str
    radius: float
    avg_dist: float
    mass: float
    orbital_radius: float
