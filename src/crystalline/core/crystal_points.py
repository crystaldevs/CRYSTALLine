"""CRYSTAL's own names for the special points of each Bravais lattice.

Tables 14.1 and 14.2 of the CRYSTAL23 manual: the labels the ``BAND`` keyword
recognises, and where each one sits, in fractional coordinates of the primitive
cell's reciprocal lattice. They are the letters CRYSTAL prints, reads with
``ISS=0``, and that its tutorials use — so they are the letters this app has to
speak, on a plot's axis as much as in a deck.

They are not the only convention. The standard tables everyone else follows
(Setyawan and Curtarolo, which ASE and pymatgen implement) agree with CRYSTAL
for the lattices most people use — cubic in all three centrings, hexagonal,
primitive tetragonal, primitive and face-centred orthorhombic — and disagree
elsewhere in two ways:

* the same letter for a *different* point. CRYSTAL's body-centred tetragonal P
  is (½,½,½) where the standard tables put it at (¼,¼,¼); its primitive
  monoclinic Y and Z are the other way round from theirs.
* the same point, written about different axes — rhombohedral F and L,
  base-centred orthorhombic T and Y, primitive monoclinic A and E.

The second kind is harmless and the first is not, which is why the whole table
is carried here rather than a patch for the differences.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

Point = Tuple[float, float, float]

_THIRD = 1.0 / 3.0

# Table 14.1
CUBIC_P: Dict[str, Point] = {"M": (.5, .5, 0), "R": (.5, .5, .5), "X": (0, .5, 0)}
CUBIC_F: Dict[str, Point] = {"X": (.5, 0, .5), "L": (.5, .5, .5), "W": (.5, .25, .75)}
CUBIC_I: Dict[str, Point] = {"H": (.5, -.5, .5), "P": (.25, .25, .25), "N": (0, 0, .5)}
HEXAGONAL: Dict[str, Point] = {"M": (.5, 0, 0), "K": (_THIRD, _THIRD, 0),
                               "A": (0, 0, .5), "L": (.5, 0, .5),
                               "H": (_THIRD, _THIRD, .5)}
RHOMBOHEDRAL: Dict[str, Point] = {"T": (.5, .5, -.5), "F": (0, .5, .5), "L": (0, 0, .5)}
MONOCLINIC_P: Dict[str, Point] = {"A": (.5, -.5, 0), "B": (.5, 0, 0), "C": (0, .5, .5),
                                  "D": (.5, 0, .5), "E": (.5, -.5, .5), "Y": (0, .5, 0),
                                  "Z": (0, 0, .5)}
MONOCLINIC_C: Dict[str, Point] = {"A": (.5, 0, 0), "Y": (0, .5, .5), "M": (.5, .5, .5)}

# Table 14.2
ORTHORHOMBIC_P: Dict[str, Point] = {"S": (.5, .5, 0), "T": (0, .5, .5), "U": (.5, 0, .5),
                                    "R": (.5, .5, .5), "X": (.5, 0, 0), "Y": (0, .5, 0),
                                    "Z": (0, 0, .5)}
ORTHORHOMBIC_F: Dict[str, Point] = {"Z": (.5, .5, 0), "Y": (.5, 0, .5), "T": (1, .5, .5)}
ORTHORHOMBIC_C: Dict[str, Point] = {"S": (0, .5, 0), "T": (.5, .5, .5), "R": (0, .5, .5),
                                    "Y": (.5, .5, 0), "Z": (0, 0, .5)}
ORTHORHOMBIC_I: Dict[str, Point] = {"S": (.5, 0, 0), "T": (0, 0, .5), "R": (0, .5, 0),
                                    "X": (.5, -.5, .5), "W": (.25, .25, .25)}
TETRAGONAL_P: Dict[str, Point] = {"M": (.5, .5, 0), "R": (0, .5, .5), "A": (.5, .5, .5),
                                  "X": (0, .5, 0), "Z": (0, 0, .5)}
TETRAGONAL_I: Dict[str, Point] = {"M": (.5, .5, -.5), "P": (.5, .5, .5), "X": (0, 0, .5)}

# Keyed by the Bravais lattice of the cell, in ASE's naming. Triclinic (TRI)
# is absent: CRYSTAL names no point of it but Γ.
TABLES: Dict[str, Dict[str, Point]] = {
    "CUB": CUBIC_P, "FCC": CUBIC_F, "BCC": CUBIC_I,
    "HEX": HEXAGONAL, "RHL": RHOMBOHEDRAL,
    "TET": TETRAGONAL_P, "BCT": TETRAGONAL_I,
    "ORC": ORTHORHOMBIC_P, "ORCF": ORTHORHOMBIC_F,
    "ORCI": ORTHORHOMBIC_I, "ORCC": ORTHORHOMBIC_C,
    "MCL": MONOCLINIC_P, "MCLC": MONOCLINIC_C,
}


def bravais_key(structure) -> Optional[str]:
    """``"FCC"`` and so on — the row of the tables *this cell* belongs to.

    The cell's own lattice, not the crystal's space group. Both settings of a
    face-centred crystal are offered in the zone picker, and its conventional
    cell is a simple cubic lattice whose zone is a cube with M, R and X on it —
    asking the space group instead answers FCC and puts X, L and W on a zone
    that has no such points.
    """
    try:
        from ase import Atoms

        cell = Atoms(cell=np.asarray(structure.cell, dtype=float), pbc=True).cell
        name = cell.get_bravais_lattice(eps=1e-4).name
    except Exception:  # noqa: BLE001 - an unclassified lattice simply has no table
        return None
    return name if name in TABLES else None


def points_for(structure) -> Dict[str, Point]:
    """CRYSTAL's named points for this lattice — empty if it names none.

    Γ is in every one of them: the tables leave it out because, as the manual
    says, the letter G means the zone centre whatever the Bravais lattice is.
    """
    key = bravais_key(structure)
    if key is None:
        return {}
    return {"G": (0.0, 0.0, 0.0), **TABLES[key]}


def reciprocal_rotations(structure):
    """The lattice's own rotations, in reciprocal fractional coordinates.

    The *lattice's* (one atom at the origin), not the crystal's, because that is
    what special points are defined from — and it keeps apart labels a lower
    symmetry would confuse: nothing in an orthorhombic lattice's point group
    carries X into Y.
    """
    try:
        import spglib

        cell = (np.asarray(structure.cell, dtype=float), [[0.0, 0.0, 0.0]], [1])
        rotations = np.asarray(spglib.get_symmetry(cell, symprec=1e-5)["rotations"],
                               dtype=float)
        return [np.linalg.inv(rotation).T for rotation in rotations] or [np.eye(3)]
    except Exception:  # noqa: BLE001 - exact comparison alone still names most points
        return [np.eye(3)]


def same_point(one, other, rotations) -> bool:
    """One point of the zone, however its axes were chosen — or a copy of it in
    a neighbouring cell of the reciprocal lattice."""
    first, second = np.asarray(one, dtype=float), np.asarray(other, dtype=float)
    for rotation in rotations:
        delta = rotation @ second - first
        if np.allclose(delta, np.round(delta), atol=1e-6):
            return True
    return False


def align(points: Dict[str, Point], structure) -> Dict[str, Point]:
    """``points`` renamed and re-placed to CRYSTAL's convention.

    CRYSTAL's own points come first, exactly as its tables give them. A point
    from the standard tables is kept after that only if CRYSTAL has nothing to
    say about it: dropped if it is one of CRYSTAL's points under another name or
    seen from another axis, and dropped if CRYSTAL gives its letter to a
    different point — keeping it would put a label on the axis of a plot, or in
    a deck, that means something else to the program the file came from.
    """
    table = points_for(structure)
    if not table:
        return dict(points)
    rotations = reciprocal_rotations(structure)
    aligned: Dict[str, Point] = dict(table)
    for label, point in points.items():
        if label in aligned:
            continue
        if any(same_point(point, known, rotations) for known in table.values()):
            continue
        aligned[label] = point
    return aligned


__all__ = ["TABLES", "align", "bravais_key", "points_for", "reciprocal_rotations",
           "same_point"]
