"""Helpers the panel tests share: reading a form, picking atoms, finding them.

Each of these was written two or more times over in the test files before it
was here. They need no window — a panel, or plain positions — so they serve the
panel tests and the real-window ones (``_window.py``) alike.
"""

from __future__ import annotations

import numpy as np


def form_rows(form):
    """A ``QFormLayout``'s rows as ``(label, value)``, labels as shown ("Code:")."""
    from PySide6.QtWidgets import QFormLayout

    rows = []
    for row in range(form.rowCount()):
        label = form.itemAt(row, QFormLayout.LabelRole)
        value = form.itemAt(row, QFormLayout.FieldRole)
        if value is None:
            continue
        rows.append((label.widget().text() if label is not None else "", value.widget().text()))
    return rows


def output_rows(info_panel):
    """The Info panel's CRYSTAL-output rows; ``[]`` while the group is hidden."""
    if info_panel._output_group.isHidden():
        return []
    return form_rows(info_panel._output_form)


def pick_and_measure(panel, atoms):
    """Pick ``atoms`` on a Geometry panel one at a time, as in the 3D view; measure.

    The selection model hands the panel each pick's selection sorted by index,
    so that is what is given here: the order is the panel's to keep (an angle is
    measured at the second atom picked). Returns the measurement made.
    """
    panel.set_selection([])
    for picked in range(1, len(atoms) + 1):
        panel.set_selection(sorted(atoms[:picked]))
    panel._measure_selection()
    return panel.measurements()[-1]


def atoms_at(positions, source, points, atoms, anchor=None, tolerance=1e-3):
    """The atoms among ``positions`` at ``points``, all moved by one translation.

    ``source[i]`` is the atom of the primitive cell that ``positions[i]`` is an
    image of, and ``atoms`` are the source atoms the points are positions of —
    an internal coordinate's, say: each atom picked is an image of its own. The
    first is looked for among the indices in ``anchor`` (all of them by
    default), the rest wherever they are. ``None`` if they are not all there.
    """
    positions = np.asarray(positions, dtype=float)
    source = np.asarray(source)
    points = np.asarray(points, dtype=float)
    candidates = np.arange(len(positions)) if anchor is None else np.asarray(anchor)
    for first in candidates[source[candidates] == atoms[0]]:
        shift = positions[first] - points[0]
        picked = [int(first)]
        for atom, point in zip(atoms[1:], points[1:]):
            near = np.flatnonzero((source == atom)
                                  & (np.linalg.norm(positions - (point + shift), axis=1)
                                     < tolerance))
            if not len(near):
                break
            picked.append(int(near[0]))
        else:
            return picked
    return None


def closest_pair(positions):
    """The two closest atoms of ``positions``: a bond to measure on anything."""
    positions = np.asarray(positions, dtype=float)
    distances = np.linalg.norm(positions[:, None] - positions[None], axis=2)
    np.fill_diagonal(distances, np.inf)
    i, j = np.unravel_index(np.argmin(distances), distances.shape)
    return [int(i), int(j)]
