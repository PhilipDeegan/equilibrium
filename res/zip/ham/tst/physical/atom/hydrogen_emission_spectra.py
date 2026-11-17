from equilibrium.physical.electrical import electric as el

# https://webbtelescope.org/contents/articles/spectroscopy-101--how-absorption-and-emission-spectra-work


def main():
    hydrogen_S_2_energy = el.bohr_electron_energy(2, 1)
    wave_lengths = [0.0] * 4
    for i in range(4):
        wave_lengths[i] = el.electron_shell_jump_photon_wave_length(
            hydrogen_S_2_energy - el.bohr_electron_energy(i + 3, 1)
        )

    ref_wave_lengths = [
        6.561122764245458e-07,
        4.860090936478118e-07,
        4.3393669075697486e-07,
        4.1007017276534123e-07,
    ]

    assert wave_lengths == ref_wave_lengths


if __name__ == "__main__":
    main()
