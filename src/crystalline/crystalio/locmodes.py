"""The LOCMODES section of a CRYSTAL output, read through CRYSTALClear.

``Crystal_output.get_locmodes()`` (CRYSTALClear 0.2.17 on) turns the section
into three tables, one each for bonds, angles and dihedrals, with every
internal coordinate's atoms, the cell each atom is in, its value and its local
force constant, mass and frequency. (The decomposition of the normal modes
into the local ones is drawn by :mod:`crystalline.crystalio.cnm`, not read
here.) This module turns those tables into
:class:`crystalline.core.local_modes.LocalModes`, with the one thing they leave
out: where CRYSTAL has the atoms.

That has to come from the output too, and from a particular block of it. The
cell indices are counted from CRYSTAL's Cartesian coordinates of the primitive
cell, which fold the atoms into a cell differently from the fractional ones the
structure is read from: in DAFADNP's monoclinic cell 18 of 104 atoms sit a
whole lattice vector apart in the two, and with the fractional positions a
fifth of the bonds came out the wrong length. With the Cartesian block every
one of them comes out as CRYSTAL prints it.

The tables are taken as CRYSTALClear gives them, row by row, and a row that
does not hold together — atoms that are not numbers, a cell index that is not
three of them, a value that is not a number — is left out rather than guessed
at. CRYSTALClear 0.2.17 drops a cell index printed with no spaces at all,
``(-1-1-1)``, and shifts the rest of that row one column left: such a row is
one of these. 0.2.18 reads those rows, and gives a number that
overflowed its field (``*****``) as NaN, which is not a number either.

That release can also group the coordinates by the crystal's symmetry, a check
of its own on the geometry that fails on a file it cannot check. The Measure
section finds equivalent coordinates itself (see
:meth:`~crystalline.core.local_modes.LocalModes.find_by_symmetry`), so the
tables are read without it.

Kept Qt-free, like the sibling modules.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import math
import re
from typing import List, Optional, Sequence, Tuple

import numpy as np

from crystalline.core.local_modes import KINDS, LocalCoordinate, LocalModes

# The line that opens the list of internal coordinates LOCMODES kept, which is
# the list the force constants are given for — the earlier, unscreened list in
# the same output is not.
_SCREENED = "N OF REMAINING INTERNAL COORDINATES AFTER SCREENING"
_CARTESIAN = "CARTESIAN COORDINATES - PRIMITIVE CELL"
_LATTICE = "DIRECT LATTICE VECTORS CARTESIAN COMPONENTS (ANGSTROM)"

_SYMMOPS = re.compile(r"^\s*\*+\s+(\d+)\s+SYMMOPS - TRANSLATORS IN FRACTIONAL UNITS")
_ELEMENT = re.compile(r"^[A-Z][a-z]?$")
_INTEGER = re.compile(r"-?\d+")

# CRYSTALClear's tables, as (marker, attribute, value column, angular constant?).
_TABLES = (("L", "bonds", "Bond Length", False),
           ("A", "angle", "Angle", True),
           ("D", "dih", "Angle", True))


class LocalModesError(RuntimeError):
    """The output has a LOCMODES section, and it could not be read."""


def has_local_modes(path: Optional[str]) -> bool:
    """Whether ``path`` has a LOCMODES section, without reading it.

    A text scan for the line that opens the screened list of internal
    coordinates: present only once LOCMODES has run, and asked of every output
    opened.
    """
    if not path:
        return False
    try:
        with open(path, "r", errors="ignore") as handle:
            return any(_SCREENED in line for line in handle)
    except OSError:
        return False


def read_local_modes(path: Optional[str]) -> Optional[LocalModes]:
    """The local modes in the output at ``path``; ``None`` if it has none.

    Raises :class:`LocalModesError` when there is a LOCMODES section and it
    cannot be read — the CRYSTALClear installed is too old to know about it,
    or the section is not what the reader expects — so the window can say so
    instead of showing a run's measurements without their force constants as
    if it had none.
    """
    if not has_local_modes(path):
        return None
    try:
        from CRYSTALClear.crystal_io import Crystal_output
    except ImportError as exc:  # pragma: no cover - CRYSTALClear is a dependency
        raise LocalModesError("CRYSTALClear is not installed") from exc
    if not hasattr(Crystal_output, "get_locmodes"):
        raise LocalModesError("local modes need CRYSTALClear 0.2.17 or later")
    try:
        # CRYSTALClear prints as it parses; that belongs in no terminal.
        with contextlib.redirect_stdout(io.StringIO()):
            out = Crystal_output(path)
            out.get_locmodes(**_tables_only())
        lines = list(out.data)
        start = _first(lines, _SCREENED)
        positions, types = _cartesian_positions(lines, start)
        lattice = _lattice(lines, start)
        coordinates = _coordinates(out, len(positions))
        operation_count = _operation_count(lines, start)
    except LocalModesError:
        raise
    except Exception as exc:  # noqa: BLE001 - whatever the parse trips on, it is said
        raise LocalModesError(f"the LOCMODES section could not be read ({exc})") from exc
    if not coordinates:
        raise LocalModesError("the LOCMODES section lists no internal coordinates")
    return LocalModes(positions, lattice, tuple(coordinates), types, operation_count)


def reads_by_symmetry() -> bool:
    """Whether ``get_locmodes`` takes ``symmetry``: CRYSTALClear 0.2.18 on.

    That release reads the rows 0.2.17 drops, and can group the coordinates
    by the crystal's symmetry. It is known by what it takes, not by its
    version: a copy of CRYSTALClear's ``main`` installed before the release
    carried the same version number as the one before.
    """
    try:
        from CRYSTALClear.crystal_io import Crystal_output
    except Exception:  # noqa: BLE001 - CRYSTALClear missing/broken
        return False
    method = getattr(Crystal_output, "get_locmodes", None)
    if method is None:
        return False
    try:
        return "symmetry" in inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False


def _tables_only() -> dict:
    """``get_locmodes``'s arguments for the tables alone.

    ``symmetry=False`` where it takes one (CRYSTALClear 0.2.18 on), which
    otherwise groups the coordinates by symmetry by default: nothing here uses
    the groups, and the check behind them raises on an output whose geometry
    it cannot read.
    """
    return {"symmetry": False} if reads_by_symmetry() else {}


# ── the blocks around the tables ──────────────────────────────────────────
def _first(lines: Sequence[str], text: str) -> int:
    for index, line in enumerate(lines):
        if text in line:
            return index
    raise LocalModesError(f"no {text!r} line")


def _last_before(lines: Sequence[str], text: str, end: int) -> int:
    """The last line holding ``text`` before line ``end``: the geometry LOCMODES ran on."""
    for index in range(end - 1, -1, -1):
        if text in lines[index]:
            return index
    raise LocalModesError(f"no {text!r} block before the LOCMODES section")


def _cartesian_positions(lines: Sequence[str], end: int) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """CRYSTAL's Cartesian coordinates of the primitive cell's atoms, in Å, and their kinds.

    Rows of ``number, Z, element, x, y, z`` under four lines of header, to the
    first line that is not one. ``Z`` is CRYSTAL's conventional atomic number,
    which tells apart two atoms of one element given different basis sets or
    pseudopotentials: what the crystal's symmetry has to keep apart too.
    """
    start = _last_before(lines, _CARTESIAN, end)
    positions, types = [], []
    for line in lines[start + 4:]:
        parts = line.split()
        if len(parts) != 6 or not parts[0].isdigit():
            break
        if int(parts[0]) != len(positions) + 1:
            raise LocalModesError(f"atom {parts[0]} out of order in the Cartesian block")
        positions.append([float(value) for value in parts[3:6]])
        types.append(int(parts[1]))
    if not positions:
        raise LocalModesError("an empty Cartesian block")
    return np.asarray(positions, dtype=float), tuple(types)


def _lattice(lines: Sequence[str], end: int) -> np.ndarray:
    """The direct lattice vectors, as rows in Å: the steps a cell index counts."""
    start = _last_before(lines, _LATTICE, end)
    rows = [[float(value) for value in lines[start + 2 + i].split()] for i in range(3)]
    lattice = np.asarray(rows, dtype=float)
    if lattice.shape != (3, 3) or abs(np.linalg.det(lattice)) < 1e-8:
        raise LocalModesError("the lattice vectors span no volume")
    return lattice


def _operation_count(lines: Sequence[str], end: int) -> Optional[int]:
    """How many symmetry operators CRYSTAL used, the identity among them.

    From the head of its table, ``****    4 SYMMOPS - TRANSLATORS IN FRACTIONAL
    UNITS``, the last before the LOCMODES section. Only the count: the
    operators themselves are printed in the conventional cell's frame, and the
    positions are the primitive cell's. The count is what tells whether the
    symmetry found from the geometry is the calculation's.

    Read here rather than through ``Crystal_output.get_symmops``, which in
    CRYSTALClear 0.2.17 looks for exactly three spaces before the number and
    so misses every table of fewer than ten operators.
    """
    for index in range(end - 1, -1, -1):
        match = _SYMMOPS.match(lines[index])
        if match:
            return int(match.group(1))
    return None


# ── the tables ────────────────────────────────────────────────────────────
def _coordinates(out, natoms: int) -> List[LocalCoordinate]:
    """Every row of the three tables that holds together, in CRYSTAL's order."""
    found = []
    for marker, attribute, value_column, angular in _TABLES:
        table = getattr(out, attribute, None)
        if table is None:
            continue
        count = {"L": 2, "A": 3, "D": 4}[marker]
        for row in table.to_dict("records"):
            coordinate = _coordinate(row, KINDS[marker], count, value_column, angular, natoms)
            if coordinate is not None:
                found.append(coordinate)
    found.sort(key=lambda c: c.number)
    return found


def _coordinate(row: dict, kind: str, count: int, value_column: str, angular: bool,
                natoms: int) -> Optional[LocalCoordinate]:
    """One row as a :class:`LocalCoordinate`, or ``None`` if it does not hold together.

    CRYSTALClear's columns are named for the old tables: ``Atom N label`` holds
    the atom's *number* and ``Atom N`` its element. The first atom has no cell
    index: it is in the reference cell.
    """
    try:
        number = _whole(row["ic index"])
        atoms = tuple(_whole(row[f"Atom {n} label"]) - 1 for n in range(1, count + 1))
        elements = tuple(str(row[f"Atom {n}"]) for n in range(1, count + 1))
        cells = ((0, 0, 0),) + tuple(_cell(row[f"Atom {n} cell index"])
                                     for n in range(2, count + 1))
        value = _number(row[value_column])
        force_constant = _number(row["k^a"])
        angular_constant = _number(row["k^a(d)"]) if angular else None
        mass = _number(row["m^a"])
        frequency = _number(row["w^a"])
    except (KeyError, TypeError, ValueError):
        return None
    if any(not 0 <= atom < natoms for atom in atoms):
        return None
    if any(not _ELEMENT.match(element) for element in elements):
        return None
    return LocalCoordinate(number, kind, atoms, cells, elements, value, force_constant,
                           angular_constant, mass, frequency)


def _whole(value) -> int:
    if isinstance(value, bool) or not float(value).is_integer():
        raise ValueError(f"not a whole number: {value!r}")
    return int(value)


def _number(value) -> float:
    if isinstance(value, str):
        raise ValueError(f"not a number: {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"not a finite number: {value!r}")
    return number


def _cell(text) -> Tuple[int, int, int]:
    """``"( 0 0-1)"`` → ``(0, 0, -1)``: three signed integers, however spaced.

    Indices of the same sign run together when negative — CRYSTAL prints each
    in two columns — so the text is read for its integers, not split on spaces.
    """
    if not isinstance(text, str) or not text.strip().startswith("("):
        raise ValueError(f"not a cell index: {text!r}")
    values = [int(v) for v in _INTEGER.findall(text)]
    if len(values) != 3:
        raise ValueError(f"not three cell indices: {text!r}")
    return values[0], values[1], values[2]


__all__ = ["LocalModesError", "has_local_modes", "read_local_modes", "reads_by_symmetry"]
