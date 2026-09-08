"""Geometry panel: measure the selection, and edit atoms with the mouse.

Docked beside Info and Display. Two halves:

* **Measure** — turn the current 3D selection into a distance (2 atoms), angle
  (3), dihedral (4) or least-squares plane (3+), keep a list of them, and draw
  the ones that are ticked over the structure as points/lines/planes.
* **Atoms** — add, delete, duplicate, re-element, translate the selection and
  place a single atom at typed cartesian coordinates, without going to the Edit
  menu. These mirror the Edit-menu actions and are
  gated the same way: they need **Editing mode**, which is exposed here as a
  checkbox so it is obvious (and one click away) why a button is greyed out.

The panel owns no editing logic — it emits intent and :class:`MainWindow` runs
the same slots the menu uses, so undo/redo and the selection model behave
identically whichever route the user takes.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Sequence

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from crystalline.core import measure as measure_mod
from crystalline.core.structure import Structure

# Same starter palette the (hidden) structure panel used; free text is allowed.
_ELEMENTS = ["H", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Na", "Mg", "Al", "Ca", "Ti", "Fe"]
# Range of the position boxes. Fractional coordinates sit in [0, 1) for a plain
# cell and a little outside it for the images boundary completion adds, but the
# aperiodic axis of a slab is shown in Å against CRYSTAL's formal 500 Å vector,
# so the range has to cover that too.
_COORD_RANGE = 1000.0
# Spinner step: a hundredth of a cell edge, or a tenth of an Ångström.
_FRACTIONAL_STEP = 0.01
_ANGSTROM_STEP = 0.1
# Below this (Å) a typed coordinate counts as unchanged, so tabbing out of a box
# without editing it does not record an undo step for a move of nothing.
_COORD_EPS = 1e-9
# How far a coordinate box may be squeezed (px). Their *preferred* width is
# ignored entirely — see the size policy in _build_atoms_group.
_COORD_BOX_MIN_WIDTH = 56


class GeometryPanel(QWidget):
    """Measurements and atom-editing tools for the current selection."""

    # editing intent — MainWindow runs the matching Edit-menu slot
    editing_toggled = Signal(bool)
    add_atom_requested = Signal(str)      # element symbol
    delete_requested = Signal()
    duplicate_requested = Signal()
    translate_requested = Signal()
    set_element_requested = Signal(str)   # element symbol
    set_position_requested = Signal(object)  # cartesian (x, y, z) in Å for the one selected atom
    # the measurements that should be drawn in 3D (possibly empty)
    annotations_changed = Signal(list)

    def __init__(self, structure: Structure, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._structure = structure
        self._selection: List[int] = []
        self._measurements: List[measure_mod.Measurement] = []
        self._editing = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(self._build_measure_group(), 1)
        layout.addWidget(self._build_atoms_group())
        layout.addStretch(0)
        self._sync_buttons()

    # ── construction ────────────────────────────────────────────────────
    def _build_measure_group(self) -> QWidget:
        group = QGroupBox("Measure")
        box = QVBoxLayout(group)

        self._hint = QLabel(measure_mod.selection_hint(0))
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet("color: palette(mid);")
        box.addWidget(self._hint)

        row = QHBoxLayout()
        self._measure_btn = QPushButton("Measure selection")
        self._measure_btn.setToolTip(
            "2 atoms → distance, 3 → angle, 4 → dihedral, 5+ → plane fit"
        )
        self._measure_btn.clicked.connect(self._measure_selection)
        row.addWidget(self._measure_btn, 1)
        self._plane_btn = QPushButton("Fit plane")
        self._plane_btn.setToolTip("Least-squares plane through 3 or more selected atoms")
        self._plane_btn.clicked.connect(self._measure_plane)
        row.addWidget(self._plane_btn)
        box.addLayout(row)

        # Ticked measurements are drawn in the 3D view.
        self._list = QListWidget()
        self._list.setToolTip("Tick a measurement to show it in the 3D view")
        self._list.setSelectionMode(QListWidget.ExtendedSelection)
        self._list.itemChanged.connect(lambda _item: self._emit_annotations())
        box.addWidget(self._list, 1)

        row = QHBoxLayout()
        self._color_btn = QPushButton("Colour…")
        self._color_btn.setToolTip("Set the colour of the selected measurement(s)")
        self._color_btn.clicked.connect(self._set_measurement_colour)
        row.addWidget(self._color_btn)
        self._remove_btn = QPushButton("Remove")
        self._remove_btn.clicked.connect(self._remove_selected_measurements)
        row.addWidget(self._remove_btn)
        self._clear_btn = QPushButton("Clear all")
        self._clear_btn.clicked.connect(self.clear_measurements)
        row.addWidget(self._clear_btn)
        row.addStretch(1)
        box.addLayout(row)
        # The colour button follows the list selection, not just "any measurement".
        self._list.itemSelectionChanged.connect(self._sync_buttons)
        return group

    def _build_atoms_group(self) -> QWidget:
        group = QGroupBox("Atoms")
        box = QVBoxLayout(group)

        # The gate the Edit menu applies invisibly — shown here so a greyed-out
        # button explains itself.
        self._edit_check = QCheckBox("Editing mode")
        self._edit_check.setToolTip(
            "Atom tools (and dragging atoms in 3D) need editing mode — same as Edit ▸ Editing mode (Ctrl+E)"
        )
        self._edit_check.toggled.connect(self._on_editing_toggled)
        box.addWidget(self._edit_check)

        row = QHBoxLayout()
        row.addWidget(QLabel("Element"))
        self._element = QComboBox()
        self._element.addItems(_ELEMENTS)
        self._element.setCurrentText("C")
        self._element.setEditable(True)  # any symbol, e.g. "Zr"
        row.addWidget(self._element, 1)
        # A drawn glyph rather than the "⊞" box-drawing character, whose shape and
        # weight are whatever font the platform happens to resolve it in.
        self._table_btn = QPushButton()
        self._table_btn.setIcon(_table_icon())
        self._table_btn.setToolTip("Pick an element from the periodic table")
        self._table_btn.setFixedWidth(34)
        self._table_btn.clicked.connect(self._pick_from_periodic_table)
        row.addWidget(self._table_btn)
        self._add_btn = QPushButton("Add")
        self._add_btn.setToolTip("Add an atom of this element at the centre of the cell")
        self._add_btn.clicked.connect(
            lambda: self.add_atom_requested.emit(self._element.currentText().strip())
        )
        row.addWidget(self._add_btn)
        self._set_element_btn = QPushButton("Set")
        self._set_element_btn.setToolTip("Change the selected atoms to this element")
        self._set_element_btn.clicked.connect(
            lambda: self.set_element_requested.emit(self._element.currentText().strip())
        )
        row.addWidget(self._set_element_btn)
        box.addLayout(row)

        row = QHBoxLayout()
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setToolTip("Delete the selected atoms (Del)")
        self._delete_btn.clicked.connect(self.delete_requested)
        row.addWidget(self._delete_btn)
        self._duplicate_btn = QPushButton("Duplicate")
        self._duplicate_btn.clicked.connect(self.duplicate_requested)
        row.addWidget(self._duplicate_btn)
        self._translate_btn = QPushButton("Translate…")
        self._translate_btn.clicked.connect(self.translate_requested)
        row.addWidget(self._translate_btn)
        box.addLayout(row)

        # ── exact position of a single atom ─────────────────────────────
        # Dragging is quick but never exact; a structure is usually specified by
        # coordinates, so one atom at a time can be placed by typing them.
        # ── exact position of a single atom ─────────────────────────────
        # Dragging is quick but never exact; a structure is usually specified by
        # coordinates, so one atom at a time can be placed by typing them.
        self._coord_label = QLabel()
        box.addWidget(self._coord_label)
        row = QHBoxLayout()
        self._coord_boxes = []
        for axis in ("x", "y", "z"):
            spin = QDoubleSpinBox()
            spin.setRange(-_COORD_RANGE, _COORD_RANGE)
            spin.setDecimals(4)
            spin.setPrefix(f"{axis} ")
            # A box wide enough for "-1000.0000" would add ~110px each to the
            # panel's preferred width, and a dock takes its width from that — three
            # of them pushed this dock wider than the 3D view beside it. Ignoring
            # the preferred width lets them simply share the row and shrink with
            # the dock, down to _COORD_BOX_MIN_WIDTH.
            spin.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            spin.setMinimumWidth(_COORD_BOX_MIN_WIDTH)
            # Applied as soon as the value is committed — Enter, tabbing out, or a
            # click on the spinner. Keyboard tracking stays off so that *typing*
            # "0.25" commits once, rather than moving the atom to 0, then 0.2,
            # then 0.25: three edits, three undo steps, three redraws.
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(self._emit_position)
            row.addWidget(spin, 1)
            self._coord_boxes.append(spin)
        box.addLayout(row)
        return group

    # ── coordinate frame ────────────────────────────────────────────────
    def _lattice(self):
        """``(cell, periodic)`` for the shown structure, or ``None`` without a cell.

        ``None`` means a molecule: there is no lattice for a fraction to be of,
        so the boxes fall back to plain Ångström.
        """
        cell = np.asarray(self._structure.cell, dtype=float)
        if cell.shape != (3, 3) or abs(np.linalg.det(cell)) < 1e-8:
            return None
        return cell, np.asarray(self._structure.pbc, dtype=bool)

    def _to_display(self, position):
        """Cartesian position → the three numbers the boxes show.

        Fractional along each *periodic* lattice vector, Ångström along an
        aperiodic one — the convention CRYSTAL itself uses for slabs and polymers
        (and that this app's deck writer already follows), because a fraction of
        the formal 500 Å vacuum vector is not a coordinate anyone can read.
        """
        frame = self._lattice()
        if frame is None:
            return [float(v) for v in position]
        cell, periodic = frame
        fractional = np.asarray(position, dtype=float) @ np.linalg.inv(cell)
        return [
            float(fractional[k]) if periodic[k]
            else float(fractional[k] * np.linalg.norm(cell[k]))
            for k in range(3)
        ]

    def _from_display(self, values):
        """The inverse of :meth:`_to_display` — box values → a cartesian position."""
        frame = self._lattice()
        if frame is None:
            return np.asarray(values, dtype=float)
        cell, periodic = frame
        fractional = np.asarray(
            [
                values[k] if periodic[k]
                else values[k] / float(np.linalg.norm(cell[k]))
                for k in range(3)
            ],
            dtype=float,
        )
        return fractional @ cell

    def _current_position(self):
        """The selected atom's position, or ``None`` unless exactly one is selected."""
        if len(self._selection) != 1:
            return None
        index = self._selection[0]
        if not 0 <= index < len(self._structure):
            return None
        return self._structure.positions[index]

    def sync_position(self) -> None:
        """Show the selected atom's coordinates, in the frame that suits the cell.

        Called on every selection change *and* on every structure change, so the
        boxes still read true after the atom is dragged, nudged or undone —
        otherwise they would keep offering a stale position to move back to.

        Signals are blocked while filling: these boxes apply on ``valueChanged``,
        and writing the atom's own position back into them must not be mistaken
        for the user asking to move it there again.
        """
        frame = self._lattice()
        if frame is None:
            self._coord_label.setText("Position of the selected atom (Å)")
        elif bool(np.all(frame[1])):
            self._coord_label.setText("Position of the selected atom (fractional)")
        else:
            # A slab or polymer: fractional along the periodic axes, Å along the
            # vacuum one. Naming both keeps the mixed row honest.
            self._coord_label.setText("Position of the selected atom (fractional / Å)")

        position = self._current_position()
        shown = [0.0, 0.0, 0.0] if position is None else self._to_display(position)
        periodic = (True, True, True) if frame is None else frame[1]
        for k, spin in enumerate(self._coord_boxes):
            blocked = spin.blockSignals(True)  # programmatic fill is not an edit
            step = _FRACTIONAL_STEP if (frame is not None and periodic[k]) else _ANGSTROM_STEP
            spin.setSingleStep(step)
            spin.setValue(shown[k])
            spin.blockSignals(blocked)
        self._sync_buttons()

    def _emit_position(self, _value: float = 0.0) -> None:
        """Ask for the selected atom to be placed at the coordinates now shown.

        Silent when nothing would change: a box also commits on focus loss, so
        merely clicking away from an untouched one must not record an undo step
        for a move of nothing.
        """
        position = self._current_position()
        if position is None or not self._editing:
            return
        typed = [spin.value() for spin in self._coord_boxes]
        if all(
            abs(typed[k] - shown) < _COORD_EPS
            for k, shown in enumerate(self._to_display(position))
        ):
            return
        self.set_position_requested.emit(
            [float(v) for v in self._from_display(typed)]
        )

    def _pick_from_periodic_table(self) -> None:
        """Open the visual periodic table and load the chosen element into the box."""
        from crystalline.ui.periodic_table import PeriodicTableDialog

        symbol = PeriodicTableDialog.pick(self, current=self._element.currentText().strip())
        if symbol:
            self._element.setCurrentText(symbol)

    # ── external hooks ──────────────────────────────────────────────────
    def refresh_theme_icons(self) -> None:
        """Redraw the periodic-table glyph after a theme change."""
        self._table_btn.setIcon(_table_icon())

    def set_structure(self, structure: Structure) -> None:
        """Rebind after a load or a view change; measurements no longer apply."""
        self._structure = structure
        self._selection = []
        self.clear_measurements()
        self.sync_position()  # also calls _sync_buttons

    def set_selection(self, indices: Sequence[int]) -> None:
        """Track the shared selection (driven by 3D picking)."""
        self._selection = [int(i) for i in indices]
        self._hint.setText(measure_mod.selection_hint(len(self._selection)))
        self.sync_position()  # also calls _sync_buttons

    def set_editing_enabled(self, enabled: bool) -> None:
        """Reflect the Edit menu's editing mode without re-emitting it."""
        self._editing = bool(enabled)
        blocked = self._edit_check.blockSignals(True)
        self._edit_check.setChecked(self._editing)
        self._edit_check.blockSignals(blocked)
        self._sync_buttons()

    def measurements(self) -> List[measure_mod.Measurement]:
        """Every measurement in the list, shown or not (handy in tests)."""
        return list(self._measurements)

    def shown_annotations(self) -> List[measure_mod.Measurement]:
        """The measurements currently ticked for display in 3D."""
        return [
            m
            for m, item in zip(self._measurements, self._items())
            if item.checkState() == Qt.Checked
        ]

    def clear_measurements(self) -> None:
        self._measurements = []
        self._list.clear()
        self._emit_annotations()

    # ── measuring ───────────────────────────────────────────────────────
    def _measure_selection(self) -> None:
        self._add_measurement(
            measure_mod.measure(
                self._structure.positions, self._structure.symbols, self._selection
            )
        )

    def _measure_plane(self) -> None:
        self._add_measurement(
            measure_mod.measure_plane(
                self._structure.positions, self._structure.symbols, self._selection
            )
        )

    def _add_measurement(self, result) -> None:
        if result is None:
            return
        self._measurements.append(result)
        item = QListWidgetItem(result.summary())
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)  # new measurements are shown straight away
        self._list.addItem(item)
        self._emit_annotations()

    def _set_measurement_colour(self) -> None:
        """Recolour the selected measurement(s) — a per-item override of the
        type's default colour. Each item can carry its own colour."""
        rows = sorted(self._list.row(i) for i in self._list.selectedItems())
        if not rows:
            return
        current = self._measurements[rows[0]].color
        seed = QColor(current) if current else QColor("#ff7f0e")
        chosen = QColorDialog.getColor(seed, self, "Measurement colour")
        if not chosen.isValid():
            return
        hex_color = chosen.name()
        for row in rows:
            self._measurements[row] = dataclasses.replace(
                self._measurements[row], color=hex_color
            )
            self._list.item(row).setIcon(_colour_swatch(hex_color))
        self._emit_annotations()

    def _remove_selected_measurements(self) -> None:
        for row in sorted((self._list.row(i) for i in self._list.selectedItems()), reverse=True):
            self._list.takeItem(row)
            del self._measurements[row]
        self._emit_annotations()

    def _items(self) -> List[QListWidgetItem]:
        return [self._list.item(row) for row in range(self._list.count())]

    def _emit_annotations(self) -> None:
        self.annotations_changed.emit(self.shown_annotations())

    # ── editing ─────────────────────────────────────────────────────────
    def _on_editing_toggled(self, enabled: bool) -> None:
        self._editing = bool(enabled)
        self._sync_buttons()
        self.editing_toggled.emit(self._editing)

    def _sync_buttons(self) -> None:
        """Measuring needs a selection; editing needs editing mode as well."""
        count = len(self._selection)
        self._measure_btn.setEnabled(count >= 1)
        self._plane_btn.setEnabled(count >= 3)
        self._remove_btn.setEnabled(bool(self._measurements))
        self._clear_btn.setEnabled(bool(self._measurements))
        self._color_btn.setEnabled(bool(self._list.selectedItems()))

        self._add_btn.setEnabled(self._editing)
        on_selection = self._editing and count >= 1
        for button in (self._delete_btn, self._duplicate_btn,
                       self._translate_btn, self._set_element_btn):
            button.setEnabled(on_selection)

        # A typed position names one atom, so it needs exactly one selected —
        # with several, there is no single thing the coordinates could mean.
        one_atom = self._editing and count == 1
        for spin in self._coord_boxes:
            spin.setEnabled(one_atom)


def _table_icon() -> QIcon:
    """The periodic-table glyph, in the current theme's text colour."""
    from PySide6.QtWidgets import QApplication

    from crystalline.ui import theme

    palette = theme.active_palette(QApplication.instance())
    return theme.monochrome_icon("table.svg", palette.text)


def _colour_swatch(color: str, size: int = 12) -> QIcon:
    """A small solid-colour square, shown beside a recoloured measurement."""
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(color))
    return QIcon(pixmap)


__all__ = ["GeometryPanel"]
