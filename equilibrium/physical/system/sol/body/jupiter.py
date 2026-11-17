from equilibrium.physical.body import GasGiant, Moon

# from equilibrium.utils import T
# import equilibrium.physical.terms as terms


class Jupiter(GasGiant):
    radius: float = 69911
    moons: list = [
        Moon(
            name="Io",
            radius=1822,
            avg_dist=384400000,
            mass=8.9 * 1e22,
            orbital_radius=421700,
        ),
        Moon(
            name="Europa",
            radius=1561,
            avg_dist=384400000,
            mass=4.8 * 1e22,
            orbital_radius=671034,
        ),
        Moon(
            name="Ganymede",
            radius=2631,
            avg_dist=384400000,
            mass=14.8 * 1e22,
            orbital_radius=1070412,
        ),
        Moon(
            name="Callisto",
            radius=2411,
            avg_dist=384400000,
            mass=10.8 * 1e22,
            orbital_radius=1882709,
        ),
    ]
