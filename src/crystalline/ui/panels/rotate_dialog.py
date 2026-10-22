"""Turn the selected atoms about an axis: a chain about its own length, a group about a bond.

The rotation is shown in the 3D view as it is dialled in, and only reaches the
structure when the dialog is accepted — as one edit, one undo step. The dialog
itself knows nothing of the view or the structure: it describes the rotation
(an axis, an angle, a point the axis passes through) and hands it to whoever
asked, to preview and, on OK, to apply.

The axis is given the way a structure is thought about: along a lattice vector,
along a cartesian direction, through two atoms (a bond, or a chain's own
length), or as typed. The point it passes through is the selection's centroid
unless said otherwise — the rotation people mean by "turn this chain" — or one
of the selected atoms, the origin, or a typed point. An axis through two atoms
already passes through them, so the point is theirs and the choice is set
aside.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# What the axis can be. Lattice vectors only when there is a lattice.
LATTICE_AXES = ("a", "b", "c")
CARTESIAN_AXES = ("x", "y", "z")
THROUGH_ATOMS = "atoms"
CUSTOM = "custom"

# Where the axis passes through.
CENTROID = "centroid"
ATOM = "atom"
ORIGIN = "origin"

# How far the slider turns, either way; the box beside it goes round twice.
_SLIDER_RANGE = 180


def atom_label(symbols: Sequence[str], index: int) -> str:
    """``C(3)``: how the app names an atom everywhere else (see ``core.measure``)."""
    return f"{symbols[index]}({index})"


class RotateDialog(QDialog):
    """Describe a rotation of the selection; read it back with :meth:`rotation`.

    ``preview(axis, angle, centre)`` is called whenever the rotation changes,
    and ``preview(None, 0, None)`` when the dialog closes without it — the
    caller shows the atoms turned, or puts them back.
    """

    def __init__(
        self,
        selected: Sequence[int],
        positions,
        symbols: Sequence[str],
        cell=None,
        preview: Optional[Callable] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Rotate selected")
        self._selected = [int(i) for i in selected]
        self._positions = np.asarray(positions, dtype=float)
        self._symbols = list(symbols)
        self._cell = None
        if cell is not None:
            cell = np.asarray(cell, dtype=float)
            if cell.shape == (3, 3) and abs(np.linalg.det(cell)) > 1e-8:
                self._cell = cell
        self._preview = preview

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        # ── angle ───────────────────────────────────────────────────────
        self.angle = QDoubleSpinBox(self)
        self.angle.setRange(-360.0, 360.0)
        self.angle.setDecimals(2)
        self.angle.setSingleStep(5.0)
        self.angle.setSuffix(" °")
        self.angle.setToolTip("Anticlockwise looking down the axis, as the right-hand rule has it.")
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(-_SLIDER_RANGE, _SLIDER_RANGE)
        self.slider.setPageStep(15)
        row = QHBoxLayout()
        row.addWidget(self.slider, 1)
        row.addWidget(self.angle)
        form.addRow("Angle", row)
        self.slider.valueChanged.connect(self._slid)
        self.angle.valueChanged.connect(self._typed_angle)

        # ── axis ────────────────────────────────────────────────────────
        self.axis = QComboBox(self)
        if self._cell is not None:
            for name in LATTICE_AXES:
                self.axis.addItem(f"Along {name}", name)
        for name in CARTESIAN_AXES:
            self.axis.addItem(f"Along {name}", name)
        if len(self._selected) >= 2:
            self.axis.addItem("Through two atoms", THROUGH_ATOMS)
        self.axis.addItem("Custom direction", CUSTOM)
        form.addRow("Axis", self.axis)

        self.axis_from, self.axis_to = self._atom_box(), self._atom_box()
        if len(self._selected) >= 2:
            self.axis_to.setCurrentIndex(1)
        self._axis_atoms = self._pair(self.axis_from, "→", self.axis_to)
        form.addRow("", self._axis_atoms)
        self.direction = self._vector_boxes((0.0, 0.0, 1.0), "Direction, cartesian")
        form.addRow("", self.direction)

        # ── the point the axis passes through ──────────────────────────
        self.centre = QComboBox(self)
        self.centre.addItem("Centroid of the selection", CENTROID)
        self.centre.addItem("A selected atom", ATOM)
        self.centre.addItem("Origin", ORIGIN)
        self.centre.addItem("Custom point", CUSTOM)
        form.addRow("Through", self.centre)
        self.centre_atom = self._atom_box()
        form.addRow("", self.centre_atom)
        self.point = self._vector_boxes(tuple(self._centroid()), "Point, cartesian (Å)")
        form.addRow("", self.point)
        self._through_note = QLabel("The axis passes through the two atoms.", self)
        self._through_note.setEnabled(False)
        form.addRow("", self._through_note)
        self._form = form

        for box in (self.axis, self.centre, self.axis_from, self.axis_to, self.centre_atom):
            box.currentIndexChanged.connect(self._changed)
        for vector in (self.direction, self.point):
            for spin in vector.findChildren(QDoubleSpinBox):
                spin.valueChanged.connect(self._changed)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._sync()

    # ── building blocks ─────────────────────────────────────────────────
    def _atom_box(self) -> QComboBox:
        box = QComboBox(self)
        for index in self._selected:
            box.addItem(atom_label(self._symbols, index), index)
        return box

    def _pair(self, first: QWidget, joiner: str, second: QWidget) -> QWidget:
        holder = QWidget(self)
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(first, 1)
        row.addWidget(QLabel(joiner, holder))
        row.addWidget(second, 1)
        return holder

    def _vector_boxes(self, values, tooltip: str) -> QWidget:
        holder = QWidget(self)
        holder.setToolTip(tooltip)
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        for name, value in zip(CARTESIAN_AXES, values):
            spin = QDoubleSpinBox(holder)
            spin.setRange(-1000.0, 1000.0)
            spin.setDecimals(3)
            spin.setSingleStep(0.1)
            spin.setPrefix(f"{name} ")
            spin.setValue(float(value))
            row.addWidget(spin, 1)
        return holder

    @staticmethod
    def _vector(holder: QWidget) -> np.ndarray:
        return np.array([spin.value() for spin in holder.findChildren(QDoubleSpinBox)])

    def _centroid(self) -> np.ndarray:
        return self._positions[self._selected].mean(axis=0)

    # ── the rotation described ──────────────────────────────────────────
    def axis_vector(self) -> np.ndarray:
        """The direction of the axis, cartesian (not normalised)."""
        kind = self.axis.currentData()
        if kind in LATTICE_AXES:
            return self._cell[LATTICE_AXES.index(kind)].copy()
        if kind in CARTESIAN_AXES:
            return np.eye(3)[CARTESIAN_AXES.index(kind)]
        if kind == THROUGH_ATOMS:
            start, end = self.axis_from.currentData(), self.axis_to.currentData()
            return self._positions[end] - self._positions[start]
        return self._vector(self.direction)

    def centre_point(self) -> np.ndarray:
        """A point the axis passes through, cartesian (Å)."""
        if self.axis.currentData() == THROUGH_ATOMS:
            return self._positions[self.axis_from.currentData()].copy()
        kind = self.centre.currentData()
        if kind == CENTROID:
            return self._centroid()
        if kind == ATOM:
            return self._positions[self.centre_atom.currentData()].copy()
        if kind == ORIGIN:
            return np.zeros(3)
        return self._vector(self.point)

    def is_valid(self) -> bool:
        """Whether the axis has a direction: two different atoms, a non-zero vector."""
        return float(np.linalg.norm(self.axis_vector())) > 1e-8

    def rotation(self):
        """``(axis, angle in degrees, centre)``, or ``None`` if there is no axis."""
        if not self.is_valid():
            return None
        return self.axis_vector(), self.angle.value(), self.centre_point()

    # ── keeping it in step ──────────────────────────────────────────────
    def _slid(self, value: int) -> None:
        if int(round(self.angle.value())) != value:
            self.angle.setValue(float(value))

    def _typed_angle(self, value: float) -> None:
        wrapped = (value + 180.0) % 360.0 - 180.0   # the slider shows where it points
        blocked = self.slider.blockSignals(True)
        self.slider.setValue(int(round(wrapped)))
        self.slider.blockSignals(blocked)
        self._changed()

    def _changed(self, *_args) -> None:
        self._sync()
        if self._preview is None:
            return
        rotation = self.rotation()
        if rotation is None:
            self._preview(None, 0.0, None)
        else:
            self._preview(*rotation)

    def _sync(self) -> None:
        """Show the boxes the current choices use, and nothing else."""
        through = self.axis.currentData() == THROUGH_ATOMS
        shown = {
            self._axis_atoms: through,
            self.direction: self.axis.currentData() == CUSTOM,
            self.centre: not through,
            self.centre_atom: not through and self.centre.currentData() == ATOM,
            self.point: not through and self.centre.currentData() == CUSTOM,
            self._through_note: through,
        }
        for widget, visible in shown.items():
            widget.setVisible(visible)
            label = self._form.labelForField(widget)
            if label is not None:
                label.setVisible(visible)
        ok = self.findChild(QDialogButtonBox)
        if ok is not None:
            ok.button(QDialogButtonBox.Ok).setEnabled(self.is_valid())

    def done(self, result: int) -> None:
        if result != QDialog.Accepted and self._preview is not None:
            self._preview(None, 0.0, None)   # put the atoms back
        super().done(result)


__all__ = ["RotateDialog", "atom_label"]
