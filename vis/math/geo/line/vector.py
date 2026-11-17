import matplotlib.pyplot as plt
import numpy as np

import equilibrium as eq


def main():
    origin = np.array([0, 0, 0]).astype("double")
    fig = eq.plot.Fig3D(cols=2, rows=2, scale=3)
    V = eq.math.geo.line.Vector

    x = origin + [0.5, 0, 0]
    y = origin + [0, 0.5, 0]
    z = origin + [0, 0, 0.5]

    vecs = [V(x), V(y), V(z)]

    vecs.append(eq.math.geo.vector.tail_to_tip(vecs))

    fig.sub_plot_3d(vecs)

    plt.show()


if __name__ == "__main__":
    main()
