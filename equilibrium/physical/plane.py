# from typing import Any

import pydantic

# import equilibrium as eq
# from equilibrium.physical import body, metrics


class Galaxy(pydantic.BaseModel): ...


class SolarSystem(pydantic.BaseModel): ...


# class SolSystem(SolarSystem):
#     c: Any = metrics.m_per_sec(299792458)
#     bodies: Any = body.system()

#     def __iter__(self):
#         return iter(self.bodies)


# class MilkyWay(Galaxy):
#     # center = {}
#     solar_systems: list = []


# class AlphaCentauri(Galaxy):
#     # center = {}
#     solar_systems: list = []
