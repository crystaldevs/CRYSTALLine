"""A new structure from nothing: its periodicity, its lattice and its first atoms.

What CRYSTAL's own geometry input asks for, in the same order: how many
directions repeat (a molecule, a polymer, a slab or a crystal), the lattice
parameters those directions need, and the atoms. The result is a structure like
any read from a file, laid out the way the readers lay one out, so everything
downstream — the editor, the symmetry finder, the deck writer — takes it as it
takes them:

* a **slab** repeats along *a* and *b*, in the *xy* plane, and a **polymer**
  along *a*, on *x*; each aperiodic direction is given CRYSTAL's formal 500 Å;
* a **molecule** has no cell at all;
* coordinates are **fractional along the directions that repeat and in Å along
  the others**, as a CRYSTAL deck writes them (all Å for a molecule).

A crystal can be given a space group, in which case the atoms listed are its
asymmetric unit and the group fills in the rest. Slabs and polymers are built
in P1 with every atom listed; their symmetry is found when a deck is written.

Qt-free, like the rest of ``core``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from crystalline.core.structure import Structure

MOLECULE, POLYMER, SLAB, CRYSTAL = 0, 1, 2, 3
PERIODICITIES = ((CRYSTAL, "Crystal (3D)"), (SLAB, "Slab (2D)"),
                 (POLYMER, "Polymer (1D)"), (MOLECULE, "Molecule (0D)"))

#: CRYSTAL's formal length for a direction that does not repeat, in Å.
VACUUM = 500.0

PARAMETERS = ("a", "b", "c", "alpha", "beta", "gamma")


@dataclass(frozen=True)
class LatticeSystem:
    """A shape of lattice: which parameters are free, and what the rest are.

    ``free`` names the parameters a user sets. ``tied`` sets each of the others
    either to a free one (``"b": "a"``) or to a fixed angle (``"gamma": 120``).
    """

    key: str
    label: str
    free: Tuple[str, ...]
    tied: Dict[str, object] = field(default_factory=dict)

    def complete(self, given: Dict[str, float]) -> Dict[str, float]:
        """All six parameters, from the free ones."""
        values = {name: float(given[name]) for name in self.free}
        for name, rule in self.tied.items():
            values[name] = values[rule] if isinstance(rule, str) else float(rule)
        return values


_RIGHT = {"alpha": 90.0, "beta": 90.0, "gamma": 90.0}

#: The lattice systems each periodicity offers, most symmetric first.
SYSTEMS: Dict[int, Tuple[LatticeSystem, ...]] = {
    CRYSTAL: (
        LatticeSystem("cubic", "Cubic", ("a",), {"b": "a", "c": "a", **_RIGHT}),
        LatticeSystem("tetragonal", "Tetragonal", ("a", "c"), {"b": "a", **_RIGHT}),
        LatticeSystem("orthorhombic", "Orthorhombic", ("a", "b", "c"), dict(_RIGHT)),
        LatticeSystem("hexagonal", "Hexagonal", ("a", "c"),
                      {"b": "a", "alpha": 90.0, "beta": 90.0, "gamma": 120.0}),
        LatticeSystem("rhombohedral", "Rhombohedral (rhombohedral axes)", ("a", "alpha"),
                      {"b": "a", "c": "a", "beta": "alpha", "gamma": "alpha"}),
        LatticeSystem("monoclinic", "Monoclinic (unique axis b)", ("a", "b", "c", "beta"),
                      {"alpha": 90.0, "gamma": 90.0}),
        LatticeSystem("triclinic", "Triclinic", PARAMETERS),
    ),
    SLAB: (
        LatticeSystem("square", "Square", ("a",), {"b": "a", "gamma": 90.0}),
        LatticeSystem("rectangular", "Rectangular", ("a", "b"), {"gamma": 90.0}),
        LatticeSystem("hexagonal", "Hexagonal", ("a",), {"b": "a", "gamma": 120.0}),
        LatticeSystem("oblique", "Oblique", ("a", "b", "gamma")),
    ),
    POLYMER: (LatticeSystem("chain", "Chain", ("a",)),),
    MOLECULE: (LatticeSystem("none", "No lattice", ()),),
}


def system(periodicity: int, key: str) -> LatticeSystem:
    for candidate in SYSTEMS[periodicity]:
        if candidate.key == key:
            return candidate
    raise ValueError(f"no {key!r} lattice for periodicity {periodicity}")


def system_of_space_group(number: int) -> str:
    """The lattice system a space group's standard setting is written in.

    Trigonal groups are offered on hexagonal axes, as CRYSTAL reads them by
    default; the rhombohedral ones can be set on rhombohedral axes by hand.
    """
    for limit, key in ((2, "triclinic"), (15, "monoclinic"), (74, "orthorhombic"),
                       (142, "tetragonal"), (194, "hexagonal"), (230, "cubic")):
        if number <= limit:
            return key
    raise ValueError(f"no space group {number}")


@dataclass(frozen=True)
class BuildSpec:
    """Everything a new structure is built from."""

    periodicity: int = CRYSTAL
    lattice: str = "cubic"
    parameters: Dict[str, float] = field(default_factory=lambda: {"a": 5.0})
    space_group: int = 1
    #: ``(symbol, (x, y, z))``: fractional along the periodic directions, Å along
    #: the others.
    atoms: Tuple[Tuple[str, Tuple[float, float, float]], ...] = ()


class BuildError(ValueError):
    """The spec does not describe a structure; the message says why."""


def build(spec: BuildSpec) -> Structure:
    """The structure ``spec`` describes."""
    from ase import Atoms
    from ase.data import atomic_numbers

    if not spec.atoms:
        raise BuildError("A structure needs at least one atom.")
    symbols: List[str] = []
    for symbol, _position in spec.atoms:
        name = str(symbol).strip().capitalize()
        if name not in atomic_numbers or name == "X":
            raise BuildError(f"{symbol!r} is not an element.")
        symbols.append(name)
    coordinates = np.array([position for _symbol, position in spec.atoms], dtype=float)

    if spec.periodicity == MOLECULE:
        return Structure.from_ase(Atoms(symbols, positions=coordinates, pbc=False))

    values = system(spec.periodicity, spec.lattice).complete(spec.parameters)
    cell = lattice_vectors(spec.periodicity, values)
    pbc = tuple(axis < spec.periodicity for axis in range(3))

    if spec.periodicity == CRYSTAL and spec.space_group != 1:
        return _from_space_group(spec.space_group, cell, symbols, coordinates)

    # Fractional along the periodic directions, Å along the others: the Å are
    # turned into fractions of the 500 Å vectors, which are orthogonal to the
    # periodic ones and to each other.
    fractional = coordinates.copy()
    for axis in range(spec.periodicity, 3):
        fractional[:, axis] = coordinates[:, axis] / VACUUM
    atoms = Atoms(symbols, scaled_positions=fractional, cell=cell, pbc=pbc)
    return Structure.from_ase(atoms)


def lattice_vectors(periodicity: int, values: Dict[str, float]) -> np.ndarray:
    """The cell, rows *a*, *b*, *c*, with *a* along *x* and *b* in the *xy* plane.

    Positive lengths and angles that close a cell, or :class:`BuildError`.
    """
    for name in PARAMETERS[:periodicity]:
        if not values.get(name, 1.0) > 0:
            raise BuildError(f"{name} has to be longer than zero.")
    if periodicity == CRYSTAL:
        from ase.geometry import cellpar_to_cell

        parameters = [values[name] for name in PARAMETERS]
        alpha, beta, gamma = (math.radians(v) for v in parameters[3:])
        # The cell closes when the volume factor is positive.
        volume = (1 - math.cos(alpha) ** 2 - math.cos(beta) ** 2 - math.cos(gamma) ** 2
                  + 2 * math.cos(alpha) * math.cos(beta) * math.cos(gamma))
        if volume <= 1e-8:
            raise BuildError("These angles do not make a cell.")
        return np.asarray(cellpar_to_cell(parameters), dtype=float)
    cell = np.diag([VACUUM, VACUUM, VACUUM])
    cell[0] = [values["a"], 0.0, 0.0]
    if periodicity == SLAB:
        gamma = math.radians(values["gamma"])
        if not 0 < gamma < math.pi:
            raise BuildError("γ has to lie between 0° and 180°.")
        cell[1] = [values["b"] * math.cos(gamma), values["b"] * math.sin(gamma), 0.0]
    return cell


def _from_space_group(number: int, cell, symbols, fractional) -> Structure:
    """The crystal ``number`` makes of the asymmetric unit, in the cell given."""
    from pymatgen.core import Lattice
    from pymatgen.core import Structure as PmgStructure
    from pymatgen.io.ase import AseAtomsAdaptor

    try:
        crystal = PmgStructure.from_spacegroup(
            int(number), Lattice(cell), list(symbols), [list(p) for p in fractional],
            tol=1e-3)
    except Exception as exc:  # noqa: BLE001 - said to the user as it is
        raise BuildError(f"Space group {number}: {exc}") from exc
    crystal.merge_sites(tol=1e-3, mode="delete")
    return Structure.from_ase(AseAtomsAdaptor().get_atoms(crystal))


def coordinate_units(periodicity: int) -> Tuple[str, str, str]:
    """How each coordinate is given: ``"frac"`` or ``"Å"``."""
    return tuple("frac" if axis < periodicity else "Å" for axis in range(3))


__all__ = [
    "BuildError",
    "BuildSpec",
    "CRYSTAL",
    "LatticeSystem",
    "MOLECULE",
    "PARAMETERS",
    "PERIODICITIES",
    "POLYMER",
    "SLAB",
    "SYSTEMS",
    "VACUUM",
    "build",
    "coordinate_units",
    "lattice_vectors",
    "system",
    "system_of_space_group",
]
