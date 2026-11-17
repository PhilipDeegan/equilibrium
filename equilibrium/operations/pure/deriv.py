# forward diff?
def derive(fn, v=2, diff=1e-5):
    return (fn(v + diff) - fn(v - diff)) / (2 * diff)
