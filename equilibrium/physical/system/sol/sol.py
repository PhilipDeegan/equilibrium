from equilibrium.physical.body import Sun


class Sol(Sun):
    radius: float = 696340
    mass: float = 1.9884e30


def system():
    from equilibrium.physical.system.sol import body as planets

    return [
        Sol,
        planets.Mercury,
        planets.Venus,
        planets.Earth,
        planets.Mars,
        planets.Jupiter,
        planets.Saturn,
        planets.Neptune,
        planets.Uranus,
    ]


def distances():
    from equilibrium.physical.system.sol import body as planets

    return {
        Sol: {
            planets.Mercury: (46000000, 70000000),  # check
            planets.Earth: (147100000, 152100000),
            planets.Mars: (227900000,),
            planets.Jupiter: (778500000,),
            planets.Venus: (108200000,),
        }
    }


def com(cls):
    planet = cls()
    r2 = distances()[Sol][cls][1]
    m1 = Sol().mass
    m2 = planet.mass
    return (m2 * r2) / (m1 + m2)


if __name__ == "__main__":
    from equilibrium.physical.system.sol import body as planets

    print(com(planets.Earth))
