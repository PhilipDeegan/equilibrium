# pip install astropy jplephem


def test():
    from astropy.coordinates import (  # pylint: disable=import-error
        get_body_barycentric,
        solar_system_ephemeris,
    )
    from astropy.time import Time  # pylint: disable=import-error

    t = Time("2025-04-21 20:49:05")  # Replace with your desired time
    with solar_system_ephemeris.set("de432s"):
        mars_pos = get_body_barycentric("mars", t)
    print(mars_pos)


def main():
    try:
        test()
    except ModuleNotFoundError:
        print("astropy not installed, run: pip install astropy jplephem")


if __name__ == "__main__":
    main()
