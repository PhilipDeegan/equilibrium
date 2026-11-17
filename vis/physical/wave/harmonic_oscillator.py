import matplotlib.pyplot as plt
import numpy as np


def main():
    PI = np.longdouble("3.141592653589793238462643383279502884")
    t = np.linspace(0, 10, 1000)
    A = 1  # amplitude
    ϕ = 0  # phase constant
    T = 1  # period/interval
    ω = PI * 2 / T  # angular frequency
    y = A * np.cos(ω * t + ϕ)
    plt.subplots()[1].plot(t, y, ":", label="wave", color="black")
    plt.show()


if __name__ == "__main__":
    main()
