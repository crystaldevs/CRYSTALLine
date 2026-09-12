"""Layer-group symmetry, for slabs.

A slab is not a crystal with a very long third axis. Its symmetry is a *layer
group* — one of eighty, the 2D counterpart of the 230 space groups — and every
tool that reaches for a space group instead is answering a question about a
lattice the slab does not have. The vacuum is not a lattice vector, and the
operations that would use it are not symmetries.

The app already reports the layer group in its Info panel. What it did not do
was *use* it: a slab deck was written in layer group 1 with every atom listed,
throwing away symmetry that had already been found and making CRYSTAL work
through orbits it could have been told about. This module computes what that
needs — the operations, the orbits, and the standard setting they are named in.

**The setting is the risk, as always.** A layer group number means a particular
arrangement of axes, and spglib's standardisation need not be the cell the slab
was loaded in. So an asymmetric unit is only offered once it has been checked
the one way that cannot be fooled: apply the group's own operations to the
chosen representatives and see whether the slab comes back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from crystalline.core.structure import Structure

# Two fractional positions closer than this (in the periodic plane) are one site.
_SITE_TOL = 1e-4
# ...and this, in Ångström, across the slab, where the coordinate is a length.
_HEIGHT_TOL = 1e-3

# Which cell parameters CRYSTAL wants for a slab, by layer group. The eighty
# layer groups run oblique, rectangular, square, hexagonal in that order, and
# the record carries only the parameters the group does not already fix — the
# same economy as the 3D cell record.
_SYSTEMS = ((2, "oblique"), (48, "rectangular"), (64, "square"), (80, "hexagonal"))

# In the standard setting of a layer group the aperiodic direction is c,
# whichever axis carried it in the cell the slab was loaded in.
_STD_NORMAL = 2


@dataclass(frozen=True)
class LayerSymmetry:
    """A slab's layer group, in the setting its number refers to.

    ``lattice`` and ``positions`` are the standardised cell spglib puts the
    group in; ``normal_axis`` is the row of ``lattice`` across the slab, which
    carries no periodicity and no symmetry of its own.
    """

    number: int
    symbol: str
    lattice: np.ndarray                 # rows are cell vectors, standardised
    positions: np.ndarray               # fractional in that cell
    numbers: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray
    normal_axis: int = 2

    @property
    def system(self) -> str:
        """``oblique`` / ``rectangular`` / ``square`` / ``hexagonal``."""
        for limit, name in _SYSTEMS:
            if self.number <= limit:
                return name
        return "oblique"

    @property
    def plane_axes(self) -> Tuple[int, int]:
        return tuple(axis for axis in (0, 1, 2) if axis != self.normal_axis)

    def summary(self) -> str:
        return f"{self.symbol} (layer group No. {self.number})"


def analyse(structure: Structure, symprec: float = 1e-2) -> Optional[LayerSymmetry]:
    """The slab's layer group and the cell it is named in.

    ``None`` for anything that is not a slab, and for a search that fails — a
    caller then writes the trivial group and every atom, which is always true.
    """
    if not is_slab(structure):
        return None
    try:
        import spglib

        atoms = structure.to_ase()
        lattice = np.asarray(atoms.get_cell(), dtype=float)
        positions = np.asarray(atoms.get_scaled_positions(), dtype=float)
        numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
        normal = normal_axis(structure)

        first = spglib.get_layergroup(
            (lattice, positions, numbers), aperiodic_dir=normal, symprec=symprec)
        if first is None:
            return None

        # The operations come back in the basis of whatever cell was handed
        # over, so the standardised cell is asked about a second time: what is
        # written has to be described in the setting it is written in.
        # The standard setting of a layer group puts the aperiodic direction
        # along c, wherever it was in the cell that was loaded. Carrying the
        # input's axis index over here is how a slab stacked along a came back
        # as p1: the second pass was told to ignore a periodic direction and
        # to use the vacuum as one.
        std_lattice = np.asarray(first.std_lattice, dtype=float)
        std_positions = _centred_across(
            np.asarray(first.std_positions, dtype=float) % 1.0, _STD_NORMAL)
        std_numbers = np.asarray(first.std_types, dtype=int)
        second = spglib.get_layergroup(
            (std_lattice, std_positions, std_numbers),
            aperiodic_dir=_STD_NORMAL, symprec=symprec)
        if second is None or int(second.number) != int(first.number):
            return None
    except Exception:  # noqa: BLE001 - no layer group is a usable answer
        return None

    return LayerSymmetry(
        number=int(second.number),
        symbol=str(second.international),
        lattice=std_lattice,
        positions=std_positions,
        numbers=std_numbers,
        rotations=np.asarray(second.rotations, dtype=int),
        translations=np.asarray(second.translations, dtype=float),
        normal_axis=_STD_NORMAL,
    )


def _centred_across(positions: np.ndarray, normal: int) -> np.ndarray:
    """Move the layer into the middle of the cell, across the aperiodic axis.

    That axis has no natural zero — it is vacuum — and spglib's standardisation
    can leave the slab straddling the cell boundary, with atoms at z = 0.02 and
    z = 0.98 that are in fact neighbours. Asked about a cell like that the
    detector misses the mirror through the layer's centre: a Cu(100) slab comes
    back p4mm instead of p4/mmm, losing half its symmetry. Sitting the layer
    whole in the middle of the cell puts that right.

    The middle is found through the widest *gap* rather than by averaging: the
    gap is the vacuum, so whatever is not the gap is the layer, wrapped or not.
    """
    across = np.sort(positions[:, normal] % 1.0)
    if len(across) < 2:
        centre = float(across[0]) if len(across) else 0.0
    else:
        gaps = np.diff(np.append(across, across[0] + 1.0))
        widest = int(np.argmax(gaps))
        # The layer runs from just after the gap round to just before it.
        start = across[(widest + 1) % len(across)]
        span = 1.0 - gaps[widest]
        centre = float(start + span / 2.0)
    shifted = positions.copy()
    shifted[:, normal] = (shifted[:, normal] - centre + 0.5) % 1.0
    return shifted


def is_slab(structure: Structure) -> bool:
    """Exactly two periodic directions."""
    try:
        return bool(structure.is_periodic) and int(sum(bool(p) for p in structure.pbc)) == 2
    except Exception:  # noqa: BLE001
        return False


def normal_axis(structure: Structure) -> int:
    """The axis across the slab — the one that is not periodic."""
    return next(index for index, periodic in enumerate(structure.pbc) if not periodic)


def asymmetric_unit(symmetry: LayerSymmetry) -> Tuple[np.ndarray, np.ndarray]:
    """``(numbers, positions)``: one representative of every orbit.

    Orbits under the layer group's own operations, in its own setting — which
    is what CRYSTAL will apply when it reads the group number back.
    """
    positions, numbers = symmetry.positions, symmetry.numbers
    seen = np.zeros(len(positions), dtype=bool)
    keep: List[int] = []
    for index in range(len(positions)):
        if seen[index]:
            continue
        keep.append(index)
        for rotation, translation in zip(symmetry.rotations, symmetry.translations):
            image = rotation @ positions[index] + translation
            hit = _matching(symmetry, image, numbers[index])
            seen[hit] = True
    return numbers[keep], positions[keep]


def regenerates(symmetry: LayerSymmetry) -> bool:
    """Does the group, applied to the asymmetric unit, give the slab back?

    The group theory is exact; the setting is the risk. A layer group number
    means a particular arrangement of axes, and a deck written in the wrong one
    is a plausible file describing a different surface — so it is checked by
    doing what CRYSTAL will do: expand the representatives and compare.
    """
    numbers, coords = asymmetric_unit(symmetry)
    built: List[Tuple[int, np.ndarray]] = []
    for z, position in zip(numbers, coords):
        for rotation, translation in zip(symmetry.rotations, symmetry.translations):
            image = rotation @ position + translation
            if not any(other == int(z) and _same_site(symmetry, image, seen)
                       for other, seen in built):
                built.append((int(z), image))
    if len(built) != len(symmetry.positions):
        return False
    for z, position in zip(symmetry.numbers, symmetry.positions):
        if not any(other == int(z) and _same_site(symmetry, position, made)
                   for other, made in built):
            return False
    return True


def cell_record(symmetry: LayerSymmetry) -> Tuple[float, ...]:
    """The lattice parameters CRYSTAL expects for this layer group.

    Only what the group does not already fix: a square cell needs one number,
    an oblique one needs three.
    """
    first, second = symmetry.plane_axes
    a = float(np.linalg.norm(symmetry.lattice[first]))
    b = float(np.linalg.norm(symmetry.lattice[second]))
    gamma = float(np.degrees(np.arccos(np.clip(
        np.dot(symmetry.lattice[first], symmetry.lattice[second]) / (a * b), -1.0, 1.0))))
    system = symmetry.system
    if system in ("square", "hexagonal"):
        return (a,)
    if system == "rectangular":
        return (a, b)
    return (a, b, gamma)


def deck_coordinates(symmetry: LayerSymmetry, positions) -> np.ndarray:
    """Fractional in the plane, Ångström across it — CRYSTAL's 2D convention.

    The height is measured from the *layer group's own origin*, which sits at
    the middle of the layer — that is where a horizontal mirror or a centre of
    inversion lies, and the group is written about it. Measuring from the cell
    edge instead puts a flat monolayer at z = -8 Å, and p6/mmm then mirrors it
    into a second sheet sixteen Ångström away: a different material.

    :func:`analyse` has already sat the layer at the middle of the standardised
    cell, so the origin is half a cell up from the edge.
    """
    first, second = symmetry.plane_axes
    height = float(np.linalg.norm(symmetry.lattice[symmetry.normal_axis]))
    out = []
    for position in np.atleast_2d(np.asarray(positions, dtype=float)):
        across = (position[symmetry.normal_axis] % 1.0) - 0.5
        out.append((position[first], position[second], across * height))
    return np.asarray(out, dtype=float)


# ── plumbing ──────────────────────────────────────────────────────────────
def _matching(symmetry: LayerSymmetry, image: np.ndarray, number: int) -> np.ndarray:
    """Indices of atoms of this element sitting at ``image``."""
    same = symmetry.numbers == number
    delta = np.abs((symmetry.positions - image + 0.5) % 1.0 - 0.5)
    close = _within(symmetry, delta)
    return np.flatnonzero(same & close)


def _same_site(symmetry: LayerSymmetry, left, right) -> bool:
    delta = np.abs((np.asarray(left) - np.asarray(right) + 0.5) % 1.0 - 0.5)
    return bool(_within(symmetry, delta.reshape(1, 3))[0])


def _within(symmetry: LayerSymmetry, delta: np.ndarray) -> np.ndarray:
    """Close in the plane, and close *in Ångström* across the slab.

    The two are not the same test. The normal axis is a vacuum vector tens of
    Ångström long, so a fractional tolerance that is right in the plane is
    hundreds of times too loose across it — two layers a whole Ångström apart
    would count as one atom.
    """
    first, second = symmetry.plane_axes
    height = float(np.linalg.norm(symmetry.lattice[symmetry.normal_axis])) or 1.0
    in_plane = (delta[:, first] < _SITE_TOL) & (delta[:, second] < _SITE_TOL)
    across = delta[:, symmetry.normal_axis] * height < _HEIGHT_TOL
    return in_plane & across


__all__ = [
    "LayerSymmetry",
    "analyse",
    "asymmetric_unit",
    "cell_record",
    "deck_coordinates",
    "is_slab",
    "normal_axis",
    "regenerates",
]
