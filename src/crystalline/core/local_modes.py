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

import dataclasses
import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from crystalline.core.cells import periodic_distances
from crystalline.core.measure import ANGLE, DIHEDRAL, DISTANCE
from crystalline.core.symmetry import cartesian_operation

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
    # Set when a measurement is this coordinate only by symmetry: the operation
    # of the crystal that carries the measurement onto it, written as the
    # International Tables write one but with the coordinates named after the
    # cell axes ("-a+1/2, b+1/2, -c+1/2"). The numbers are this coordinate's —
    # a symmetry of the crystal leaves a local mode as it is.
    symmetry: Optional[str] = None

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
    the cell indices are counted in. ``types`` tells the atoms apart for the
    crystal's symmetry: CRYSTAL's own atomic numbers, in which an atom given
    another basis set or a pseudopotential is a kind of its own. Without them
    a measurement is found only among the coordinates listed.
    ``operation_count`` is how many symmetry operators CRYSTAL itself used, the
    identity among them, when the output says.
    """

    positions: np.ndarray
    lattice: np.ndarray
    coordinates: Tuple[LocalCoordinate, ...]
    types: Optional[Sequence[int]] = None
    operation_count: Optional[int] = None
    # (kind, atoms) → the coordinates over exactly those atoms, in that order.
    _by_atoms: Dict[Tuple[str, Tuple[int, ...]], List[LocalCoordinate]] = field(
        default_factory=dict, repr=False, compare=False)
    # The crystal's symmetry operations that the local modes bear out, worked
    # out the first time a measurement is not among the coordinates listed.
    _operations: List["_Operation"] = field(default_factory=list, repr=False, compare=False)
    _operations_found: List[bool] = field(default_factory=lambda: [False], repr=False,
                                          compare=False)

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
        the same order. Sorted by CRYSTAL's number.
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

    # ── by symmetry ─────────────────────────────────────────────────────
    def find_by_symmetry(self, kind: str, atoms: Sequence[int], points,
                         tolerance: float = MATCH_TOLERANCE) -> Tuple[LocalCoordinate, ...]:
        """The listed coordinates a measurement is the symmetry image of.

        LOCMODES does not list every coordinate of a set the crystal's symmetry
        makes equivalent — in one run, 211 images of listed coordinates are not
        themselves listed — and an equivalent coordinate has the same local
        mode. So a measurement not found as it is may be found once carried by
        an operation of the crystal onto a coordinate that is. What comes back
        are those coordinates, each marked with the operation that does it
        (:attr:`LocalCoordinate.symmetry`); the first operation that finds any
        is the one used.
        """
        atoms = tuple(int(a) for a in atoms)
        points = np.asarray(points, dtype=float)
        if len(atoms) != ATOM_COUNT.get(kind, -1) or points.shape != (len(atoms), 3):
            return ()
        for operation in self.operations():
            moved = operation.apply(points)
            found = self.find(kind, [int(operation.permutation[a]) for a in atoms], moved,
                              tolerance)
            if found:
                return tuple(dataclasses.replace(c, symmetry=operation.symbol) for c in found)
        return ()

    def operations(self) -> List["_Operation"]:
        """The crystal's symmetry operations, other than the identity, that the
        local modes bear out.

        Found by spglib on CRYSTAL's own positions, and then put to the data.
        An operation is dropped if it carries one listed coordinate onto
        another with a different ωᵃ, kᵃ(d) or mᵃ, beyond the digits CRYSTAL
        prints. Of the rest, all are kept when spglib finds as many operators
        as CRYSTAL says it used — the two groups are then the same one — and
        otherwise only those the data has borne out at least once: the
        calculation was made in a subgroup of what the geometry has, and an
        operation outside it is not a symmetry of the local modes. kᵃ is left
        out of the test: for an angle or a dihedral crossing a cell edge,
        CRYSTAL's kᵃ can differ between equivalent coordinates (see the notes
        on LOCMODES).
        """
        if not self._operations_found[0]:
            self._operations_found[0] = True
            self._operations.extend(self._verified_operations())
        return list(self._operations)

    def _verified_operations(self) -> List["_Operation"]:
        if self.types is None or not len(self.coordinates):
            return []
        try:
            import spglib
        except ImportError:  # pragma: no cover - spglib is a dependency
            return []
        inverse = np.linalg.inv(self.lattice)
        fractional = self.positions @ inverse
        cell = (self.lattice, fractional, [int(t) for t in self.types])
        found = spglib.get_symmetry(cell, symprec=_SYMMETRY_PRECISION)
        if not found:
            return []
        rotations, translations = found["rotations"], found["translations"]
        # The same group as CRYSTAL's: as many operators, and CRYSTAL's are
        # among the geometry's.
        same_group = self.operation_count is not None and len(rotations) == self.operation_count
        kept = []
        for rotation, translation in zip(rotations, translations):
            rotation = np.asarray(rotation, dtype=int)
            translation = np.asarray(translation, dtype=float)
            if np.array_equal(rotation, np.eye(3, dtype=int)) and np.allclose(
                    translation - np.round(translation), 0.0, atol=1e-6):
                continue  # the identity: found as listed, or not at all
            operation = _Operation.build(rotation, translation, self.lattice, fractional)
            if operation is None:
                continue
            verdict = self._put_to_the_data(operation)
            if verdict == _BORNE_OUT or (verdict == _UNTESTED and same_group):
                kept.append(operation)
        return kept

    def _put_to_the_data(self, operation: "_Operation") -> str:
        """Whether the local modes are those of a crystal with this symmetry:
        borne out, contradicted, or never put to the test."""
        tested = 0
        for coordinate in self.coordinates:
            moved = operation.apply(self.points(coordinate))
            images = self.find(coordinate.kind,
                               [int(operation.permutation[a]) for a in coordinate.atoms], moved)
            if len(images) != 1 or len(self._by_atoms.get((coordinate.kind, coordinate.atoms),
                                                          ())) != 1:
                continue  # a near-linear angle's two bends cannot be told apart this way
            image = images[0]
            if not (_same_number(image.frequency, coordinate.frequency)
                    and _same_number(image.mass, coordinate.mass)
                    and _same_number(image.angular_force_constant or 0.0,
                                     coordinate.angular_force_constant or 0.0)):
                return _CONTRADICTED
            tested += 1
        return _BORNE_OUT if tested else _UNTESTED


# What putting a symmetry operation to the local modes can find.
_BORNE_OUT, _CONTRADICTED, _UNTESTED = "borne out", "contradicted", "untested"

# How precisely (Å) the crystal's atoms are to repeat under one of its symmetry
# operations. CRYSTAL's positions are symmetric to its thirteen figures.
_SYMMETRY_PRECISION = 1e-3

# Two numbers CRYSTAL prints to three decimals are the same number if they
# differ by no more than their rounding: half the last place, on each.
_PRINTED_EQUAL = 1e-3 + 1e-9


def _same_number(a: float, b: float) -> bool:
    return abs(a - b) <= _PRINTED_EQUAL


@dataclass(frozen=True)
class _Operation:
    """One symmetry operation of the crystal, in Cartesian coordinates.

    ``rotation`` and ``translation`` act as ``x' = rotation @ x + translation``
    (Å); ``permutation[i]`` is the atom that atom ``i`` is carried onto, in
    whichever cell. ``symbol`` is the operation in the primitive cell's
    fractional coordinates, as a crystallographer writes it.
    """

    rotation: np.ndarray
    translation: np.ndarray
    permutation: np.ndarray
    symbol: str

    @staticmethod
    def build(rotation, translation, lattice, fractional) -> Optional["_Operation"]:
        """From spglib's fractional form, or ``None`` if it does not map atoms onto atoms."""
        moved = fractional @ rotation.T + translation
        permutation = np.empty(len(fractional), dtype=int)
        for i, position in enumerate(moved):
            distances = periodic_distances(fractional, position, lattice)
            j = int(np.argmin(distances))
            if distances[j] > _SYMMETRY_PRECISION * 10:
                return None
            permutation[i] = j
        cartesian, shift = cartesian_operation(rotation, translation, lattice)
        return _Operation(cartesian, shift, permutation, _symbol(rotation, translation))

    def apply(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=float) @ self.rotation.T + self.translation


def _symbol(rotation: np.ndarray, translation: np.ndarray) -> str:
    """``"-a+1/2, b+1/2, -c+1/2"``: the operation as the International Tables
    write it, with the fractional coordinates named after the cell axes.

    a, b, c rather than the Tables' x, y, z (Filippo, 2026-10-21): beside the
    measurements, and in their export, x, y, z are Cartesian positions in Å,
    and these are fractions of the primitive cell's a, b and c.
    """
    parts = []
    for row, shift in zip(rotation, translation):
        text = ""
        for coefficient, axis in zip(row, "abc"):
            if coefficient == 0:
                continue
            sign = "-" if coefficient < 0 else ("+" if text else "")
            magnitude = "" if abs(coefficient) == 1 else str(abs(int(coefficient)))
            text += f"{sign}{magnitude}{axis}"
        shift = float(shift) - math.floor(float(shift) + 1e-9)
        if shift > 1e-6:
            fraction = Fraction(shift).limit_denominator(12)
            text += f"+{fraction.numerator}/{fraction.denominator}"
        parts.append(text or "0")
    return ", ".join(parts)


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
