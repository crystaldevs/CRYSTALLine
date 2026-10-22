"""Rigid motions of a selection: turning atoms about an axis, images and all.

A selection in a periodic structure is never only the atoms clicked. Each of
them may have periodic images on screen — the copies boundary completion draws
on the faces of the cell, or the copies of a supercell — and an image that is
left where it was is no longer an image: the crystal on screen stops being a
crystal. A drag already carries them along (see
:meth:`~crystalline.viz.renderer.StructureRenderer.active_move_group`); a
rotation has to as well, and for a rotation that is less simple than it sounds.

A translation moves every atom by the same vector, images included. A rotation
does not: an atom and its image sit at different distances from the axis, and
turning both would move them by different amounts and break the translation
that made one the image of the other. So the selection is first sorted into
classes of atoms that are images of one another; one atom of each class — the
one nearest the centre of the rotation, which is the one the rotation is
meant for — is turned, and every other member of its class is moved by the
same displacement. A chain selected right across a cell face (its first carbon
and that carbon's image at the far face both selected) therefore turns as one
chain, and stays periodic.

Qt-free, like the rest of ``core``.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np


def rotation_matrix(axis: Iterable[float], angle_degrees: float) -> np.ndarray:
    """The matrix turning a vector by ``angle_degrees`` about ``axis`` (right-handed).

    Positive angles are anticlockwise seen looking down the axis towards its
    origin — the right-hand rule, as for every rotation in crystallography.

    Raises:
        ValueError: For an axis of no length, about which there is no turning.
    """
    axis = np.asarray(list(axis), dtype=float)
    length = float(np.linalg.norm(axis))
    if length < 1e-12:
        raise ValueError("the rotation axis has no length")
    x, y, z = axis / length
    theta = np.radians(float(angle_degrees))
    c, s = np.cos(theta), np.sin(theta)
    t = 1.0 - c
    return np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ])


def rotate_points(points, axis: Iterable[float], angle_degrees: float,
                  centre: Iterable[float]) -> np.ndarray:
    """``points`` turned about the line through ``centre`` along ``axis``."""
    points = np.asarray(points, dtype=float)
    centre = np.asarray(list(centre), dtype=float)
    return (points - centre) @ rotation_matrix(axis, angle_degrees).T + centre


def image_classes(selected: Sequence[int],
                  images_of: Callable[[int], Iterable[int]]) -> List[List[int]]:
    """The selection and its images, sorted into classes of mutual images.

    ``images_of(i)`` gives the atoms that are periodic images of atom ``i``.
    Each class holds every atom of one class that is selected or is an image of
    a selected atom, selected ones first, in the order they were selected.
    """
    seen: Dict[int, int] = {}
    classes: List[List[int]] = []
    for atom in selected:
        atom = int(atom)
        if atom in seen:
            continue
        members = [atom] + [int(j) for j in images_of(atom) if int(j) != atom]
        number = len(classes)
        classes.append([])
        for member in members:
            if member not in seen:
                seen[member] = number
                classes[number].append(member)
    selected_set = {int(i) for i in selected}
    return [[i for i in members if i in selected_set] + [i for i in members if i not in selected_set]
            for members in classes]


def rotated_selection(positions, selected: Sequence[int],
                      images_of: Callable[[int], Iterable[int]],
                      axis: Iterable[float], angle_degrees: float,
                      centre: Iterable[float]) -> Dict[int, np.ndarray]:
    """Where the selection and its images go when the selection is turned.

    Returns ``{atom: new cartesian position}`` for every atom that moves. In
    each class of mutual images (:func:`image_classes`) the selected atom
    nearest ``centre`` is turned, and the rest of its class moves by the same
    displacement, so images stay images.
    """
    positions = np.asarray(positions, dtype=float)
    centre = np.asarray(list(centre), dtype=float)
    selected_set = {int(i) for i in selected}
    turn = rotation_matrix(axis, angle_degrees)
    moved: Dict[int, np.ndarray] = {}
    for members in image_classes(selected, images_of):
        candidates = [i for i in members if i in selected_set] or members
        representative = min(candidates,
                             key=lambda i: float(np.linalg.norm(positions[i] - centre)))
        displacement = (positions[representative] - centre) @ turn.T + centre \
            - positions[representative]
        for member in members:
            moved[member] = positions[member] + displacement
    return moved


__all__ = ["image_classes", "rotate_points", "rotated_selection", "rotation_matrix"]
