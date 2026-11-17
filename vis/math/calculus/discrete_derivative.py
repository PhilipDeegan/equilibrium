import argparse


def central(fn, x, h=1e-5):
    return (fn(x + h) - fn(x - h)) / (2 * h)


def forward_unit(fn, x):
    return fn(x + 1) - fn(x)


def table(x=10, powers=(2, 3, 4, 5)):
    rows = []
    for n in powers:
        fn = lambda v, n=n: v**n
        d, c = forward_unit(fn, x), central(fn, x)
        rows.append((n, d, c, d - c))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="unit-step difference vs continuum derivative of x^n"
    )
    ap.add_argument("--x", type=float, default=10)
    a = ap.parse_args(argv)
    print(f"{'n':>2} {'f(x+1)-f(x)':>14} {'df/dx':>14} {'difference':>14}")
    for n, d, c, diff in table(a.x):
        print(f"{n:>2} {d:14.4f} {c:14.4f} {diff:14.4f}")


if __name__ == "__main__":
    main()
