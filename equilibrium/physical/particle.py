from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, RootModel

from equilibrium.log import getLogger
from equilibrium.math import constants as mc
from equilibrium.math import numpy as nq

from . import constants as pc

# from .electrical import Electrons
from .electrical import electric as el

logger = getLogger(__name__)


def as_particle(obj, **kwargs):
    return ParticleWrapper.make(obj, **kwargs)


class Particle(BaseModel):
    velocity: nq.array_t = Field(default_factory=lambda: np.zeros(1))
    position: nq.array_t = Field(default_factory=lambda: np.zeros(1))
    mass: float = Field(default_factory=lambda: 1)
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ParticleWrapper(RootModel[dict[str, Any]]):
    root: dict[str, Any]

    @classmethod
    def make(cls, data, **kwargs) -> "ParticleWrapper":
        if isinstance(data, BaseModel):
            keys = type(data).__fields__.keys()
            # logger.info("MASS DOUBLING")
            if all("mass" in L for L in [kwargs, keys]):
                logger.info("MASS DOUBLING")
            if "mass" in keys:
                logger.info("MASS DOUBLING")
                kwargs["mass"] = data.mass
        # if "mass" in kwargs:
        p = Particle(**kwargs)
        return cls(
            {
                "real": data,
                "base": p,
                "velocity": p.velocity,
                "position": p.position,
                "mass": p.mass,
            },
        )

    # def __getitem__(self):
    #     logger.info("what")
    #     return 1

    def __getattr__(self, key):
        if key in self.root:
            return self.root[key]
        raise ValueError(f"No key found: {key}")

    def keys(self):
        ignore = ["real", "base"]
        return tuple([k for k in self.root.keys() if k not in ignore])

    def __iter__(self):
        return self.root["real"]


def as_electron_volts(m):
    return 1.78266192 * 1e-36 * m


def kg_to_electron_volts(m=1.0):
    return 5.6095883571872 * 1e35 * m


def daltons_to_ev(da):
    return as_electron_volts(da * 931.4941 * 1e6)


class Proton:
    e = as_electron_volts(0.983 * 1e9)
    m = 1.67262192 * 1e-27
    r = 0.841 * 1e-15
    v = 4 / 3 * mc.PI * r**3
    md = m / v
    charge = pc.e

    def __repr__(self):
        return f"""Proton:
        energy: {self.e} ev
        mass: {self.m} kg
        radius: {self.r} m
        volume: {self.v} m³
        mass density: {self.md}  kg/m³
        charge: {self.charge} coloumbs
        """


class Electron:
    e = as_electron_volts(0.511 * 1e6)
    m = 9.1093837 * 1e-31
    charge = -pc.e


class Neutron:
    m = 1.67492749804 * 1e-27  # kg


class Atom(Particle):
    ...
    # protons: list[Proton]
    # neutrons: list[Neutron]
    # electrons: Electrons


class Hydrogen(Atom):
    r: float = el.bohr_shell_radius(1, 1)
    ie1: float = 13.5984  # first ionization energy, eV (protium, ground state)


class Protium(Hydrogen):
    m: float = 1.007825031898  # daltons
    ie1: float = 13.5984


class Deuterium(Hydrogen):
    m: float = 2.014101777844
    ie1: float = 13.6021


class Tritium(Hydrogen):
    m: float = 3.016049281320
    ie1: float = 13.6034


class Helium(Atom):
    ie1: float = 24.5874  # first ionization energy, eV (He-4)


class Helium3(Helium):
    m: float = 3.016029321967
    ie1: float = 24.5874  # isotope shift ~µeV, negligible at this precision


class Helium4(Helium):
    m: float = 4.002603254130
    ie1: float = 24.5874


class Neon(Atom):
    # No stable neutral Ne compound has ever been synthesized (as of 2026);
    # high-pressure DFT analogs to Na2He have been proposed but not observed.
    ie1: float = 21.5645  # eV


class Argon(Atom):
    # HArF exists only as a matrix-isolated species below ~27 K (Khriachtchev 2000).
    ie1: float = 15.7596  # eV


# PtF6 (Bartlett 1962, first noble-gas compound XePtF6) is a transition-metal
# hexafluoride oxidizer (EA ~6.8-7.0 eV) — modeling it needs multi-atom orbital
# chemistry this Particle/Atom model doesn't represent; not attempted here.


class Carbon(Atom): ...


class Carbon12(Carbon): ...


class Uranium(Atom): ...


class Uranium235(Uranium): ...


class Uranium238(Uranium): ...
