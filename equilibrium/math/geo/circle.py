import numpy as np

PI = np.longdouble("3.141592653589793238462643383279502884")


def points(p, r, axis=1, points=6):
    a = np.array([p.copy() for _ in range(points)])
    a[0][0] += r
    θ = (PI * 2) / points
    for i in range(1, points):
        θi = θ * i
        a[i][0] += np.cos(θi) * r
        a[i][axis] += np.sin(θi) * r
        np.testing.assert_array_less(a[i], p + r + 1e-5)

    return a
