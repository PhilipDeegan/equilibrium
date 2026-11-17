# from typing import TypeVar
from typing import Annotated

import numpy as np

# from pydantic import ConfigDict
# from pydantic import BaseModel
from pydantic import BeforeValidator, PlainSerializer


def is_nd_array(a):
    return isinstance(a, np.ndarray)


def to_array(a):
    if is_nd_array(a):
        return a
    return np.array(a)


def is_scalar(x):
    return np.isscalar(x)


def as_arrays(*args):
    return to_array(args[0]) if len(args) == 1 else (to_array(a) for a in args)


class Point:
    @staticmethod
    def copy(a: np.ndarray, *args, **kwargs):
        if args and kwargs:
            raise RuntimeError

        c = a.copy()
        print("c.shape", c.shape)
        if args:
            ...

        if kwargs:
            ...

        return c


def nd_array_custom_before_validator(x):  # custom before validation logic
    return x


def nd_array_custom_serializer(x):  # custom serialization logic
    return str(x)


array_t = Annotated[
    np.ndarray,
    BeforeValidator(nd_array_custom_before_validator),
    PlainSerializer(nd_array_custom_serializer, return_type=str),
]


def array(shape):
    if not isinstance(shape, (list, tuple)):
        shape = tuple(shape)
    return np.array(shape)
