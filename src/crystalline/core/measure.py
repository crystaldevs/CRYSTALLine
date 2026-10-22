"""Geometry measurements on a structure: positions, distances, angles, dihedrals.

Measurements are taken on the atoms **as drawn**. The displayed cell is usually
boundary-completed, so an atom that straddles the boundary appears at every cell
position it touches; measuring the drawn coordinates is what the user means when
they click two atoms on screen (VESTA behaves the same way). No minimum-image
convention is applied — pick the image you can see.

Qt-free and free of the renderer, so the maths is unit-tested on its own; the
panel turns a selection into a :class:`Measurement` and the renderer draws it.

A plane through the selection is not a measurement here: it is fitted with the
lattice planes (:func:`crystalline.core.lattice_planes.fit_plane`), listed and
drawn with them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

# What a selection of N atoms measures.
DISTANCE, ANGLE, DIHEDRAL, POINT = "distance", "angle", "dihedral", "point"

# How thick (Å, the diameter) the line of a distance, angle or dihedral is drawn
# unless told otherwise: thin enough to follow a bond without hiding it.
DEFAULT_THICKNESS = 0.07

_DEGREES = "°"
_ANGSTROM = "Å"


@dataclass(frozen=True)
class Measurement:
    """One measured quantity, ready to list in the panel and draw in 3D.

    ``points`` are the cartesian positions the measurement was taken on, kept so
    the renderer can draw it without re-reading the structure (and so it stays
    meaningful if the selection changes). ``value`` is in Å for a distance and
    in degrees for an angle or a dihedral.
    """

    kind: str
    indices: Tuple[int, ...]
    value: float
    label: str
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    # Optional per-item colour ("#rrggbb"); ``None`` uses the type's default from
    # RenderSettings (measure_point/line_color).
    color: Optional[str] = None
    # How thick (Å) its line is drawn. A point is a dot, and has no line.
    thickness: float = DEFAULT_THICKNESS

    @property
    def unit(self) -> str:
        if self.kind == DISTANCE:
            return _ANGSTROM
        if self.kind in (ANGLE, DIHEDRAL):
            return _DEGREES
        return ""

    def summary(self) -> str:
        """``"Si(1)–O(2)   1.612 Å"``-style text for the measurement list."""
        if self.kind == POINT:
            return self.label
        return f"{self.label}   {self.value:.3f} {self.unit}".rstrip()


def distance(positions: np.ndarray, i: int, j: int) -> float:
    """Interatomic distance ``i``–``j`` in Å."""
    p = np.asarray(positions, dtype=float)
    return float(np.linalg.norm(p[i] - p[j]))


def angle_between(u, v) -> float:
    """Angle between two vectors, in degrees — ``nan`` if either has no length.

    ``nan`` rather than a number, because there is no angle to report: it
    compares false against any threshold, which is what every caller here
    wants of a degenerate pair.
    """
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0.0 or nv == 0.0:
        return float("nan")
    cosine = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def angle(positions: np.ndarray, i: int, j: int, k: int) -> float:
    """Angle i–j–k in degrees, with ``j`` at the vertex."""
    p = np.asarray(positions, dtype=float)
    return angle_between(p[i] - p[j], p[k] - p[j])


def dihedral(positions: np.ndarray, i: int, j: int, k: int, m: int) -> float:
    """Torsion angle i–j–k–m in degrees, signed, in (-180, 180].

    The standard convention: the angle between the plane through i–j–k and the
    plane through j–k–m, looking along j→k.
    """
    p = np.asarray(positions, dtype=float)
    b0, b1, b2 = p[j] - p[i], p[k] - p[j], p[m] - p[k]
    n1 = np.cross(b0, b1)
    n2 = np.cross(b1, b2)
    b1n = np.linalg.norm(b1)
    if b1n == 0.0 or np.linalg.norm(n1) == 0.0 or np.linalg.norm(n2) == 0.0:
        return float("nan")
    # atan2 form: numerically stable and gives the sign for free
    x = float(np.dot(n1, n2))
    y = float(np.dot(np.cross(n1, n2), b1 / b1n))
    return float(np.degrees(np.arctan2(y, x)))


def measure(
    positions: np.ndarray, symbols: Sequence[str], indices: Sequence[int]
) -> Optional[Measurement]:
    """Measure whatever a selection of atoms defines, or ``None`` if it defines nothing.

    1 atom → its position, 2 → a distance, 3 → an angle, 4 → a dihedral. No
    atoms, or more than four, measure nothing.
    """
    indices = [int(i) for i in indices]
    if len(indices) == 1:
        return measure_point(positions, symbols, indices[0])
    if len(indices) == 2:
        value = distance(positions, *indices)
        return Measurement(DISTANCE, tuple(indices), value, _label(symbols, indices, "–"),
                           points=np.asarray(positions, dtype=float)[indices])
    if len(indices) == 3:
        value = angle(positions, *indices)
        return Measurement(ANGLE, tuple(indices), value, _label(symbols, indices, "–"),
                           points=np.asarray(positions, dtype=float)[indices])
    if len(indices) == 4:
        value = dihedral(positions, *indices)
        return Measurement(DIHEDRAL, tuple(indices), value, _label(symbols, indices, "–"),
                           points=np.asarray(positions, dtype=float)[indices])
    return None


def measure_point(positions: np.ndarray, symbols: Sequence[str], index: int) -> Measurement:
    """A single atom's position, as a labelled point."""
    p = np.asarray(positions, dtype=float)[int(index)]
    label = f"{symbols[int(index)]}({index})  ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})"
    return Measurement(POINT, (int(index),), float("nan"), label, points=p.reshape(1, 3))


# Beyond this, an atom list is abbreviated so the row stays readable.
_LABEL_MAX_ATOMS = 4


def _label(symbols: Sequence[str], indices: Sequence[int], joiner: str) -> str:
    """``"Si(1)–O(2)"``; long lists are truncated with a "+N more" tail."""
    shown = list(indices)[:_LABEL_MAX_ATOMS]
    text = joiner.join(f"{symbols[i]}({i})" for i in shown)
    extra = len(indices) - len(shown)
    return f"{text} +{extra} more" if extra > 0 else text


# The table's columns, written out: what was measured.
EXPORT_COLUMNS = ("Kind", "Atoms", "Value", "Unit")
EXPORT_POSITION_COLUMNS = ("x (Å)", "y (Å)", "z (Å)")


def export_rows(measurements: Sequence[Measurement]) -> Tuple[list, list]:
    """``(header, rows)`` for writing ``measurements`` out as a table.

    One row per measurement, in the order given — the order on screen — with
    the measured value in full rather than to the three places the table
    shows. A point's position goes in three columns of its own, there only
    when the table has a point.
    """
    points = any(m.kind == POINT for m in measurements)
    header = list(EXPORT_COLUMNS)
    if points:
        header += EXPORT_POSITION_COLUMNS
    rows = []
    for m in measurements:
        if m.kind == POINT:
            measured = [m.kind, m.label.partition("  ")[0], "", ""]
        else:
            measured = [m.kind, m.label, f"{m.value:.6f}", m.unit]
        if points:
            measured += ([f"{v:.6f}" for v in np.asarray(m.points, dtype=float)[0]]
                         if m.kind == POINT else ["", "", ""])
        rows.append(measured)
    return header, rows


def selection_hint(count: int) -> str:
    """What the current number of selected atoms will measure."""
    return {
        0: "Select atoms in the 3D view to measure them.",
        1: "1 atom: position. Select 2 for a distance.",
        2: "2 atoms: distance. Select 3 for an angle.",
        3: "3 atoms: angle.",
        4: "4 atoms: dihedral.",
    }.get(count, f"{count} atoms: select 4 or fewer to measure, or fit a plane "
                 f"to them under Lattice planes.")


__all__ = [
    "DEFAULT_THICKNESS",
    "Measurement",
    "DISTANCE",
    "ANGLE",
    "DIHEDRAL",
    "POINT",
    "distance",
    "angle",
    "angle_between",
    "dihedral",
    "measure",
    "measure_point",
    "selection_hint",
    "EXPORT_COLUMNS",
    "EXPORT_POSITION_COLUMNS",
    "export_rows",
]
