import math


def e(x=1, n=10):
    e = 1 + x
    for i in range(2, n):
        e += (x**i) / float(math.factorial(i))
    return e


if __name__ == "__main__":
    print(e())
