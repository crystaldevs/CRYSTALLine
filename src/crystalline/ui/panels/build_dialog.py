"""Build a new structure: its periodicity, its lattice and its first atoms.

The questions CRYSTAL's geometry input asks, in its order (see
:mod:`crystalline.core.builder`): a molecule, a polymer, a slab or a crystal;
the lattice those directions need, as a lattice system so that a hexagonal cell
cannot be given a γ of 119.9°; for a crystal, the space group whose asymmetric
unit the atoms are; and the atoms. The rest — more atoms, moved ones — is done
in the editor, where the structure opens.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from crystalline.core import builder as b
from crystalline.ui.panels.controls import Section
from crystalline.ui.safety import guard
from crystalline.ui.widgets.wrapped_label import WrappedLabel

_LABELS = {"a": "a (Å)", "b": "b (Å)", "c": "c (Å)",
           "alpha": "α (°)", "beta": "β (°)", "gamma": "γ (°)"}
# What each parameter starts at: a cell that closes in every system, and γ at
# 90° for the square, rectangular and oblique nets.
_START = {"a": 5.0, "b": 5.0, "c": 5.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0}


class BuildDialog(QDialog):
    """Choose a periodicity, a lattice and atoms; :meth:`structure` is the result."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New structure")
        self.resize(560, 640)
        self._built = None
        outer = QVBoxLayout(self)

        kind = Section(outer, "Structure")
        self.periodicity = QComboBox(self)
        for key, label in b.PERIODICITIES:
            self.periodicity.addItem(label, key)
        self.periodicity.currentIndexChanged.connect(lambda _i: self._on_periodicity())
        kind.add("Periodicity", self.periodicity)

        lattice = Section(outer, "Lattice")
        self.space_group = QSpinBox(self)
        self.space_group.setRange(1, 230)
        self.space_group.setToolTip(
            "The atoms listed are the asymmetric unit; the group fills in the rest. "
            "Choosing a group sets the lattice system it is written in.")
        self.space_group.valueChanged.connect(lambda _v: self._on_space_group())
        self._group_symbol = QLabel(self)
        group_row = QWidget(self)
        line = QHBoxLayout(group_row)
        line.setContentsMargins(0, 0, 0, 0)
        line.addWidget(self.space_group)
        line.addWidget(self._group_symbol, 1)
        lattice.add("Space group", group_row)
        self._group_row = lattice.row_widgets("Space group")
        self.lattice = QComboBox(self)
        self.lattice.currentIndexChanged.connect(lambda _i: self._sync_parameters())
        lattice.add("System", self.lattice)
        self._parameters = {}
        self._parameter_rows = {}
        for name in b.PARAMETERS:
            box = QDoubleSpinBox(self)
            angle = name in ("alpha", "beta", "gamma")
            if angle:
                box.setRange(1.0, 179.0)
            else:
                box.setRange(0.1, 200.0)
            box.setDecimals(3 if angle else 4)
            box.setSingleStep(1.0 if angle else 0.1)
            box.setValue(_START[name])
            box.valueChanged.connect(lambda _v: self._sync_parameters())
            lattice.add(_LABELS[name], box)
            self._parameters[name] = box
            self._parameter_rows[name] = lattice.row_widgets(_LABELS[name])
        self._lattice_section = lattice

        atoms = Section(outer, "Atoms")
        # The table takes whatever height the dialog has to spare.
        outer.setStretchFactor(atoms.body, 1)
        self._table = QTableWidget(0, 4, self)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setMinimumHeight(180)
        atoms.append(self._table)
        add = QPushButton("Add atom", self)
        add.clicked.connect(lambda _checked=False: self._add_row("C", (0.0, 0.0, 0.0)))
        remove = QPushButton("Remove", self)
        remove.clicked.connect(lambda _checked=False: self._remove_rows())
        buttons_row = QWidget(self)
        line = QHBoxLayout(buttons_row)
        line.setContentsMargins(0, 0, 0, 0)
        line.addWidget(add)
        line.addWidget(remove)
        line.addStretch(1)
        atoms.append(buttons_row)
        self._add_row("C", (0.0, 0.0, 0.0))

        self._error = WrappedLabel()
        self._error.setStyleSheet("color: #c2414b;")
        outer.addWidget(self._error)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.button(QDialogButtonBox.Ok).setText("Build")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        self._on_periodicity()
        self._on_space_group()

    # ── keeping the controls honest ─────────────────────────────────────
    def _current_periodicity(self) -> int:
        return int(self.periodicity.currentData())

    def _on_periodicity(self) -> None:
        """Offer the lattice systems, parameters and coordinates this periodicity has."""
        periodicity = self._current_periodicity()
        self.lattice.blockSignals(True)
        self.lattice.clear()
        for candidate in b.SYSTEMS[periodicity]:
            self.lattice.addItem(candidate.label, candidate.key)
        self.lattice.blockSignals(False)
        self._lattice_section.set_visible(periodicity != b.MOLECULE)
        for widget in self._group_row:
            widget.setVisible(periodicity == b.CRYSTAL)
        if periodicity == b.CRYSTAL:
            self._on_space_group()
        units = b.coordinate_units(periodicity)
        self._table.setHorizontalHeaderLabels(
            ["Element"] + [f"{axis} ({unit})" for axis, unit in zip("xyz", units)])
        self._sync_parameters()

    def _on_space_group(self) -> None:
        number = self.space_group.value()
        try:
            from pymatgen.symmetry.groups import SpaceGroup

            self._group_symbol.setText(SpaceGroup.from_int_number(number).symbol)
        except Exception:  # noqa: BLE001 - the number is what matters
            self._group_symbol.setText("")
        if self._current_periodicity() != b.CRYSTAL or number == 1:
            return
        wanted = b.system_of_space_group(number)
        index = self.lattice.findData(wanted)
        rhombohedral = self.lattice.currentData() == "rhombohedral" and wanted == "hexagonal"
        if index >= 0 and not rhombohedral:
            self.lattice.setCurrentIndex(index)

    def _sync_parameters(self) -> None:
        """Show the parameters the periodicity has; let the system's free ones be set."""
        periodicity = self._current_periodicity()
        key = self.lattice.currentData()
        if key is None:
            return
        chosen = b.system(periodicity, key)
        relevant = {b.CRYSTAL: b.PARAMETERS, b.SLAB: ("a", "b", "gamma"),
                    b.POLYMER: ("a",), b.MOLECULE: ()}[periodicity]
        values = chosen.complete({name: self._parameters[name].value()
                                  for name in chosen.free})
        for name, box in self._parameters.items():
            shown = name in relevant
            for widget in self._parameter_rows[name]:
                widget.setVisible(shown)
            box.setEnabled(name in chosen.free)
            if shown and name not in chosen.free and name in values:
                blocked = box.blockSignals(True)
                box.setValue(values[name])
                box.blockSignals(blocked)

    # ── atoms ───────────────────────────────────────────────────────────
    def _add_row(self, symbol: str, position) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setItem(row, 0, QTableWidgetItem(symbol))
        for column, value in enumerate(position, start=1):
            self._table.setItem(row, column, QTableWidgetItem(f"{value:g}"))

    def _remove_rows(self) -> None:
        rows = sorted({index.row() for index in self._table.selectedIndexes()}, reverse=True)
        for row in rows or [self._table.rowCount() - 1]:
            if row >= 0:
                self._table.removeRow(row)

    def _atoms(self):
        atoms = []
        for row in range(self._table.rowCount()):
            cells = [self._table.item(row, column) for column in range(4)]
            texts = [cell.text().strip() if cell is not None else "" for cell in cells]
            if not any(texts):
                continue
            try:
                position = tuple(float(text or 0.0) for text in texts[1:])
            except ValueError as exc:
                raise b.BuildError(f"Row {row + 1}: a coordinate is not a number.") from exc
            atoms.append((texts[0], position))
        return tuple(atoms)

    # ── the answer ──────────────────────────────────────────────────────
    def spec(self) -> b.BuildSpec:
        periodicity = self._current_periodicity()
        return b.BuildSpec(
            periodicity=periodicity,
            lattice=self.lattice.currentData(),
            parameters={name: box.value() for name, box in self._parameters.items()},
            space_group=self.space_group.value() if periodicity == b.CRYSTAL else 1,
            atoms=self._atoms(),
        )

    @guard()
    def accept(self) -> None:
        """Build, and close only if what was asked for is a structure."""
        try:
            self._built = b.build(self.spec())
        except b.BuildError as exc:
            self._error.setText(str(exc))
            return
        super().accept()

    def structure(self):
        """The structure built when the dialog was accepted, else ``None``."""
        return self._built


__all__ = ["BuildDialog"]
