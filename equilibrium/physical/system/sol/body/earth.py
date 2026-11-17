from equilibrium.physical import terms
from equilibrium.physical.body import Moon, Planet
from equilibrium.utils import T


class Earth(Planet):
    g: T = terms.Acceleration(meters=9.807)
    # G: T = terms.Pressure()
    radius: float = 6371
    tropos: float = 15
    strato: float = 50
    meso: float = 90
    mass: float = 5.972168 * 1e24
    moons: list = [
        Moon(
            name="Luna",
            radius=1738.1,
            avg_dist=384400,
            mass=7.342 * 1e22,
            orbital_radius=348000,
            orbital_speed=1.022,
        )
    ]
