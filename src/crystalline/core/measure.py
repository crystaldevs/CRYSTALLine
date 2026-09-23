"""Geometry measurements on a structure: distances, angles, dihedrals, planes.

Measurements are taken on the atoms **as drawn**. The displayed cell is usually
boundary-completed, so an atom that straddles the boundary appears at every cell
position it touches; measuring the drawn coordinates is what the user means when
they click two atoms on screen (VESTA behaves the same way). No minimum-image
convention is applied — pick the image you can see.

Qt-free and free of the renderer, so the maths is unit-tested on its own; the
panel turns a selection into a :class:`Measurement` and the renderer draws it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

# What a selection of N atoms measures.
DISTANCE, ANGLE, DIHEDRAL, PLANE, POINT = "distance", "angle", "dihedral", "plane", "point"

_DEGREES = "°"
_ANGSTROM = "Å"


@dataclass(frozen=True)
class Measurement:
    """One measured quantity, ready to list in the panel and draw in 3D.

    ``points`` are the cartesian positions the measurement was taken on, kept so
    the renderer can draw it without re-reading the structure (and so it stays
    meaningful if the selection changes). ``value`` is in Å for a distance,
    degrees for an angle/dihedral, and the RMS deviation (Å) for a plane fit.
    """

    kind: str
    indices: Tuple[int, ...]
    value: float
    label: str
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    # Planes only: a point on the plane and its unit normal.
    origin: Optional[np.ndarray] = None
    normal: Optional[np.ndarray] = None
    # Optional per-item colour ("#rrggbb"); ``None`` uses the type's default from
    # RenderSettings (measure_point/line/plane_color).
    color: Optional[str] = None

    @property
    def unit(self) -> str:
        if self.kind == DISTANCE:
            return _ANGSTROM
        if self.kind in (ANGLE, DIHEDRAL):
            return _DEGREES
        if self.kind == PLANE:
            return _ANGSTROM  # RMS deviation of the fitted atoms
        return ""

    def summary(self) -> str:
        """``"Si(1)–O(2)   1.612 Å"``-style text for the measurement list."""
        if self.kind == POINT:
            return self.label
        if self.kind == PLANE:  # the value is an out-of-plane deviation, so say so
            return f"{self.label}   rms {self.value:.3f} {self.unit}"
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


def plane(positions: np.ndarray, indices: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, float]:
    """Least-squares plane through ≥3 atoms: ``(centroid, unit normal, rms)``.

    The normal is the singular vector of least variance (SVD of the centred
    coordinates), and ``rms`` is the root-mean-square out-of-plane deviation —
    0 for exactly coplanar atoms, and a measure of planarity otherwise. Its sign
    is fixed so the normal's largest component is positive, keeping the result
    stable from call to call.
    """
    p = np.asarray(positions, dtype=float)[list(indices)]
    centroid = p.mean(axis=0)
    _u, _s, vh = np.linalg.svd(p - centroid)
    normal = np.asarray(vh[2], dtype=float)
    if normal[int(np.argmax(np.abs(normal)))] < 0:
        normal = -normal
    rms = float(np.sqrt(np.mean(((p - centroid) @ normal) ** 2)))
    return centroid, normal, rms


def measure(
    positions: np.ndarray, symbols: Sequence[str], indices: Sequence[int]
) -> Optional[Measurement]:
    """Measure whatever a selection of atoms defines, or ``None`` if it defines nothing.

    1 atom → its position, 2 → a distance, 3 → an angle, 4 → a dihedral. Five or
    more can only sensibly be a plane fit, so that is what they give; use
    :func:`measure_plane` to force a plane for 3 or 4 atoms instead.
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
    if len(indices) > 4:
        return measure_plane(positions, symbols, indices)
    return None


def measure_point(positions: np.ndarray, symbols: Sequence[str], index: int) -> Measurement:
    """A single atom's position, as a labelled point."""
    p = np.asarray(positions, dtype=float)[int(index)]
    label = f"{symbols[int(index)]}({index})  ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})"
    return Measurement(POINT, (int(index),), float("nan"), label, points=p.reshape(1, 3))


def measure_plane(
    positions: np.ndarray, symbols: Sequence[str], indices: Sequence[int]
) -> Optional[Measurement]:
    """Fit a plane through ≥3 atoms (``None`` for fewer)."""
    indices = [int(i) for i in indices]
    if len(indices) < 3:
        return None
    centroid, normal, rms = plane(positions, indices)
    label = (
        f"Plane {_label(symbols, indices, ', ')}"
        f"  n=({normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f})"
    )
    return Measurement(
        PLANE, tuple(indices), rms, label,
        points=np.asarray(positions, dtype=float)[indices], origin=centroid, normal=normal,
    )


# Beyond this, a plane's atom list is abbreviated so the row stays readable.
_LABEL_MAX_ATOMS = 4


def _label(symbols: Sequence[str], indices: Sequence[int], joiner: str) -> str:
    """``"Si(1)–O(2)"``; long lists are truncated with a "+N more" tail."""
    shown = list(indices)[:_LABEL_MAX_ATOMS]
    text = joiner.join(f"{symbols[i]}({i})" for i in shown)
    extra = len(indices) - len(shown)
    return f"{text} +{extra} more" if extra > 0 else text


def selection_hint(count: int) -> str:
    """What the current number of selected atoms will measure."""
    return {
        0: "Select atoms in the 3D view to measure them.",
        1: "1 atom: position. Select 2 for a distance.",
        2: "2 atoms: distance. Select 3 for an angle.",
        3: "3 atoms: angle (or a plane).",
        4: "4 atoms: dihedral (or a plane).",
    }.get(count, f"{count} atoms: plane fit.")


__all__ = [
    "Measurement",
    "DISTANCE",
    "ANGLE",
    "DIHEDRAL",
    "PLANE",
    "POINT",
    "distance",
    "angle",
    "angle_between",
    "dihedral",
    "plane",
    "measure",
    "measure_point",
    "measure_plane",
    "selection_hint",
]
