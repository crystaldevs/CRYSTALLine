"""Local vibrational modes: CRYSTAL's LOCMODES internal coordinates, and finding them again.

A LOCMODES run decomposes the normal modes into local modes, one per internal
coordinate — a bond length, an angle or a dihedral — and gives each its local
(adiabatic) force constant, mass and frequency. What the window offers is the
other way round: measure a bond, an angle or a dihedral on screen, and if it is
one of the internal coordinates LOCMODES worked out, its force constant is
there beside the value.

Recognising one takes two things the measurement has and the table has too:
which atoms (CRYSTAL's numbering is the order of the primitive cell, which is
the source structure's), and where they are relative to each other. An
internal coordinate names its atoms by number and by the cell each one is in,
``( 1 0 0)`` being the next cell along +a; the first atom is always in the
reference cell. Those cell indices are relative to CRYSTAL's *Cartesian*
coordinates of the primitive cell, which are folded into a cell differently
from its fractional ones — in an oblique cell, an atom can sit a whole lattice
vector away from where the fractional coordinates put it — so the positions
are CRYSTAL's own, read alongside the table (see
:func:`crystalline.crystalio.read_local_modes`).

A measurement is a match when it names the same atoms, forwards or backwards,
and its atoms are placed the same way relative to each other: the whole
coordinate may sit one lattice translation away — a bond measured in the next
cell is the same bond — but not one atom on its own.

Qt-free and file-free: the reading is :mod:`crystalline.crystalio`'s.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from crystalline.core.measure import ANGLE, DIHEDRAL, DISTANCE

# How far (Å) a measured atom may sit from where the internal coordinate puts
# it and still be that atom. CRYSTAL prints thirteen significant figures, and
# the structure on screen is read from the same run, so the two agree to far
# better than this; an atom of another coordinate is never this close.
MATCH_TOLERANCE = 1e-3

# The three kinds of internal coordinate, as CRYSTAL marks them — (L), (A) and
# (D) — and what each is when measured.
KINDS = {"L": DISTANCE, "A": ANGLE, "D": DIHEDRAL}

# How many atoms each kind takes.
ATOM_COUNT = {DISTANCE: 2, ANGLE: 3, DIHEDRAL: 4}


@dataclass(frozen=True)
class LocalCoordinate:
    """One internal coordinate, with its local mode.

    ``atoms`` are 0-based indices into CRYSTAL's primitive cell (one less than
    the numbers it prints) and ``cells`` the lattice translation of each, the
    first being ``(0, 0, 0)``. For an angle the second atom is the vertex, as in
    :func:`crystalline.core.measure.angle`; a dihedral runs 1–2–3–4.

    ``value`` is CRYSTAL's: Å for a bond, degrees for an angle or a dihedral.
    It is not always the plain angle of the atoms — a near-linear angle comes
    as two bends over the same three atoms, and some dihedrals are defined
    otherwise — which is why a measurement is recognised by its atoms, not by
    its value.
    """

    number: int                                  # CRYSTAL's internal coordinate number
    kind: str                                    # measure.DISTANCE, ANGLE or DIHEDRAL
    atoms: Tuple[int, ...]
    cells: Tuple[Tuple[int, int, int], ...]
    elements: Tuple[str, ...]
    value: float
    force_constant: float                        # kᵃ, mdyn/Å
    angular_force_constant: Optional[float]      # kᵃ(d), mdyn·Å/rad²; None for a bond
    mass: float                                  # mᵃ, amu (bonds) or amu·Å (angles)
    frequency: float                             # ωᵃ, cm⁻¹

    def __post_init__(self) -> None:
        if len(self.atoms) != ATOM_COUNT.get(self.kind, -1) or len(self.cells) != len(self.atoms):
            raise ValueError(f"internal coordinate {self.number}: {len(self.atoms)} atoms "
                             f"and {len(self.cells)} cells for a {self.kind}")

    def atom_label(self) -> str:
        """``"N73–O13 (1 -1 1)"``: the atoms as CRYSTAL numbers them.

        From 1, as the output prints them, each followed by its cell when it is
        not the reference one — a lattice translation of the primitive cell.
        """
        return "–".join(
            f"{element}{atom + 1}"
            + ("" if cell == (0, 0, 0) else f" ({cell[0]} {cell[1]} {cell[2]})")
            for element, atom, cell in zip(self.elements, self.atoms, self.cells)
        )


@dataclass(frozen=True)
class LocalModes:
    """Every internal coordinate a LOCMODES run computed a force constant for.

    ``positions`` are CRYSTAL's Cartesian coordinates of the primitive cell's
    atoms and ``lattice`` its lattice vectors as rows, both in Å — the frame
    the cell indices are counted in.
    """

    positions: np.ndarray
    lattice: np.ndarray
    coordinates: Tuple[LocalCoordinate, ...]
    # (kind, atoms) → the coordinates over exactly those atoms, in that order.
    _by_atoms: Dict[Tuple[str, Tuple[int, ...]], List[LocalCoordinate]] = field(
        default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "positions", np.asarray(self.positions, dtype=float))
        object.__setattr__(self, "lattice", np.asarray(self.lattice, dtype=float))
        index: Dict[Tuple[str, Tuple[int, ...]], List[LocalCoordinate]] = {}
        for coordinate in self.coordinates:
            index.setdefault((coordinate.kind, coordinate.atoms), []).append(coordinate)
        self._by_atoms.clear()
        self._by_atoms.update(index)

    def __len__(self) -> int:
        return len(self.coordinates)

    def points(self, coordinate: LocalCoordinate) -> np.ndarray:
        """Where ``coordinate``'s atoms are, in Å, in CRYSTAL's frame."""
        atoms = np.asarray(coordinate.atoms, dtype=int)
        cells = np.asarray(coordinate.cells, dtype=float)
        return self.positions[atoms] + cells @ self.lattice

    def find(self, kind: str, atoms: Sequence[int], points,
             tolerance: float = MATCH_TOLERANCE) -> Tuple[LocalCoordinate, ...]:
        """The internal coordinates a measurement of ``kind`` on these atoms is.

        ``atoms`` are the measured atoms' indices in the primitive cell (the
        source structure), in the order measured, and ``points`` where they
        were measured. Usually one coordinate or none; two for a near-linear
        angle, which LOCMODES takes as two bends over the same atoms, or for
        two dihedral-like coordinates it defines over the same four atoms in
        the same order (DAFADNP's 681 and 726). Sorted by CRYSTAL's number.
        """
        atoms = tuple(int(a) for a in atoms)
        points = np.asarray(points, dtype=float)
        if len(atoms) != ATOM_COUNT.get(kind, -1) or points.shape != (len(atoms), 3):
            return ()
        found = []
        # The same coordinate, measured from either end: the vertex of an angle
        # and the central bond of a dihedral stay where they are.
        for order in (atoms, atoms[::-1]):
            measured = points if order is atoms else points[::-1]
            for coordinate in self._by_atoms.get((kind, order), ()):
                if coordinate not in found and _same_shape(self.points(coordinate), measured,
                                                           tolerance):
                    found.append(coordinate)
        return tuple(sorted(found, key=lambda c: c.number))


def _same_shape(expected: np.ndarray, measured: np.ndarray, tolerance: float) -> bool:
    """Whether ``measured`` is ``expected`` moved as a whole, atom for atom."""
    return bool(np.all(np.linalg.norm((measured - measured[0]) - (expected - expected[0]),
                                      axis=1) <= tolerance))


__all__ = [
    "ATOM_COUNT",
    "KINDS",
    "LocalCoordinate",
    "LocalModes",
    "MATCH_TOLERANCE",
]
