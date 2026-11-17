import matplotlib.pyplot as plt

import equilibrium as eq


def main():
    fig = eq.plot.Fig(cols=2, rows=2, scale=3)
    L = eq.math.geo.line.Line

    lines = [L([0, 0], [0.2, 0.1]), L([0, 0], [0.1, 0.2])]
    fig.sub_plot(lines)

    plt.show()


if __name__ == "__main__":
    main()
