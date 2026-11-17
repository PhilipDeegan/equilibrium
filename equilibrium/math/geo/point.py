import numpy as np

_dimdices = {"x": 0, "y": 1, "z": 2}


def copy(a: np.ndarray, *args, **kwargs):
    if args and kwargs:
        raise RuntimeError
    c = a.copy()
    if args:
        ...
    if kwargs:
        for k, v in kwargs.items():
            if k not in _dimdices:
                raise RuntimeError("Unknown Point member, valid are ['x','y','z']")
            c[_dimdices[k]] = v
    return c
