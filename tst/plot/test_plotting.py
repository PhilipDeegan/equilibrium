import unittest

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import equilibrium as eq

matplotlib.use("Agg")
origin = np.array([0, 0, 0]).astype("double")


class EqPlotTest(unittest.TestCase):
    def test_sphere_points(self):
        sp = eq.math.geo.sphere.Sphere(origin, 1, 5, 3)
        fig = eq.plot.Fig3D(cols=4, rows=2, scale=2)

        p = sp.points()
        print(p.flatten())

        def sub_plot(elev=30, azim=50, roll=0):
            fig.sub_plot_3d(p, elev, azim, roll)

        sub_plot(90, 0)
        sub_plot(00, 40, 0)
        sub_plot(00)
        sub_plot(20, 0)
        plt.show()

    def test_sphere_vertices(self):
        sp = eq.math.geo.sphere.Sphere(origin, 1, 5, 3)
        fig = eq.plot.Fig3D(cols=4, rows=3, scale=5)

        def sub_plot(elev=30, azim=50, roll=0):
            fig.sub_plot_3d(sp.v, elev, azim, roll)

        sub_plot(90, 0)
        sub_plot(00, 40, 0)
        sub_plot(00)
        sub_plot(20, 0)
        plt.show()


if __name__ == "__main__":
    unittest.main()
