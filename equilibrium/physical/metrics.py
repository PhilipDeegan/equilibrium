from datetime import timedelta

import pydantic


class DistanceDelta(pydantic.BaseModel):
    meters: int | None = None
    km: int | None = None

    def __func__(self):
        x = 0
        if self.meters:
            x += self.meters
        if self.km:
            x += self.km * 1e3
        return x


class DistancePerTime(pydantic.BaseModel):
    x: DistanceDelta = None
    t: timedelta = None


class WeightDelta(pydantic.BaseModel):
    g: int | None = None
    k: int | None = None
    t: int | None = None
    kt: int | None = None
    mt: int | None = None
    gt: int | None = None

    def __func__(self):
        w = self.g if self.meters else 0
        w += self.k * 1e3 if self.km else 0
        return w


def m_per_sec(x):
    return DistancePerTime(x=DistanceDelta(meters=x), t=timedelta(seconds=1))


def km_per_sec(x):
    return DistancePerTime(x=DistanceDelta(km=x), t=timedelta(seconds=1))


class Force(pydantic.BaseModel):
    x: DistanceDelta = DistanceDelta(meters=1)
    t: timedelta = timedelta(seconds=1)
    w: WeightDelta = WeightDelta(k=1)


class Newton(Force): ...


mole = 6.02214076 * 1e23
