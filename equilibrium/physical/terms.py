import pydantic

from equilibrium.base import BaseModel
from equilibrium.utils import T


class Base(BaseModel):
    def _set_(self, **kwargs):
        for k in self.__class__.__fields__.keys():
            if v := kwargs.get(k):
                getattr(self, k)(v)
        return self

    @pydantic.root_validator(pre=True)
    def validate(cls, values):
        return {
            k: (T(n=k, v=v) if not isinstance(v, T) else v) for k, v in values.items()
        }


mass = T(n="mass")  # ?

seconds = T(n="seconds")
s = seconds

meters = T(n="meters")
m = meters  # ?

acceleration = T(n="acceleration")
a = acceleration

velocity = T(n="velocity")
v = velocity


class Velocity(Base):
    meters: T = T(n="meters")
    seconds: T = T(n="seconds", v=1)

    def __call__(self, **kwargs):
        if not kwargs:
            return (self.meters * self.seconds)()
        return self._set_(**kwargs)


class Acceleration(Base):
    meters: T = T(n="meters")
    seconds: T = T(n="seconds", v=1)

    def __call__(self, **kwargs):
        if not kwargs:
            return (self.meters * self.seconds * self.seconds)()
        return self._set_(**kwargs)
