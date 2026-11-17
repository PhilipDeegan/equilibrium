def tail_to_tip(vs):
    if not len(vs):
        raise ValueError("Input must be non-empty list of geo.Vector")

    from equilibrium.math.geo import line

    return line.Line([0] * vs[0].c.shape[0], sum(vs[i].c for i in range(len(vs))))
