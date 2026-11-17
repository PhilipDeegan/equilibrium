import argparse
import dataclasses

import numpy as np

EV = 1.602e-19
D_BAND_MIN = -7.0 * EV
CONDUCTION_BAND_MIN = 0.0 * EV
NEIGHBOURS = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
RULES = ("share", "half", "decay", "sequential")


@dataclasses.dataclass
class Lattice:
    shape: tuple = (3, 3, 3)
    capacitance: float = 5.0 * EV
    activation: float = 0.5 * EV
    capped: bool = True

    def __post_init__(self):
        self.electron = np.full(self.shape, D_BAND_MIN)
        self.phonon = np.zeros(self.shape)
        self.electrical_capacitance = np.full(self.shape, self.capacitance)
        self.thermal_capacitance = np.full(self.shape, self.capacitance)

    def neighbours(self, x, y, z):
        for dx, dy, dz in NEIGHBOURS:
            n = (x + dx, y + dy, z + dz)
            if all(0 <= n[i] < self.shape[i] for i in range(3)):
                yield n

    def absorb(self, pos, energy):
        if self.capped and energy > self.electrical_capacitance[pos]:
            return "Energy rejected (exceeds capacitance)"
        target = self.electron[pos] + energy
        if target >= CONDUCTION_BAND_MIN:
            self.electron[pos] = CONDUCTION_BAND_MIN
            return "Excited to conduction"
        self.electron[pos] = target
        return "Excited within d-band"

    def relax(self, pos):
        lost = self.electron[pos] - D_BAND_MIN
        if lost <= 0:
            return None
        self.electron[pos] = D_BAND_MIN
        if self.capped and self.phonon[pos] + lost > self.thermal_capacitance[pos]:
            return "Energy rejected (thermal capacitance exceeded)"
        self.phonon[pos] += lost
        return None

    def _transfer(self, src, new, pos, rule, decay):
        x, y, z = pos
        if rule == "share":
            ns = list(self.neighbours(x, y, z))
            share = src[pos] / (len(ns) + 1)
            new[pos] -= share * len(ns)
            for n in ns:
                new[n] += share
            return
        for n in self.neighbours(x, y, z):
            diff = src[pos] - src[n]
            if diff <= self.activation:
                continue
            transfer = diff / 2 if rule == "half" else decay * diff
            limit = diff if rule == "half" else transfer
            if src[n] + limit <= self.thermal_capacitance[n]:
                new[pos] -= transfer
                new[n] += transfer

    def propagate_atom(self, pos, rule="decay", decay=0.1):
        new = self.phonon.copy()
        self._transfer(self.phonon, new, pos, rule, decay)
        self.phonon = new

    def propagate(self, rule="decay", decay=0.1):
        if rule == "sequential":
            for pos in np.ndindex(self.shape):
                self.propagate_atom(pos, "decay", decay)
            return
        new = self.phonon.copy()
        for pos in np.ndindex(self.shape):
            self._transfer(self.phonon, new, pos, rule, decay)
        self.phonon = new


def run(
    rule="decay",
    decay=0.1,
    steps=1,
    photon_ev=3.0,
    capped=True,
    relax=True,
    seed_atom=False,
):
    lat = Lattice(capped=capped)
    pos = tuple(s // 2 for s in lat.shape)
    result = lat.absorb(pos, photon_ev * EV)
    if relax:
        lat.relax(pos)
    if seed_atom:
        lat.propagate_atom(pos, rule, decay)
    for _ in range(steps):
        lat.propagate(rule, decay)
    return lat, result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", choices=RULES, default="decay")
    ap.add_argument("--decay", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--photon-ev", type=float, default=3.0)
    ap.add_argument("--uncapped", action="store_true")
    ap.add_argument("--no-relax", action="store_true")
    ap.add_argument("--seed-atom", action="store_true")
    a = ap.parse_args(argv)
    lat, result = run(
        a.rule,
        a.decay,
        a.steps,
        a.photon_ev,
        not a.uncapped,
        not a.no_relax,
        a.seed_atom,
    )
    print("Photon absorption result:", result)
    print("Updated electron states:")
    print(lat.electron)
    print("Updated phonon energy distribution:")
    print(lat.phonon)


if __name__ == "__main__":
    main()
