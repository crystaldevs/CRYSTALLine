"""Choose which cut through the anharmonic PES to draw.

An ANHAPES run computes a quartic surface over every mode it was given, which
is far too many dimensions to look at: the figure is always a cut, and picking
it is the whole job of this dialog. One mode gives a curve, two give a map.

The pair list is the part that has to work: a 36-mode run couples 630 pairs and
almost all of them are worth nothing, so they are ordered by how strongly the
two modes actually couple and can be narrowed to one mode at a time. Scrolling
630 alphabetical rows to find the four that matter is not a picker.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from crystalline.ui.panels.controls import Section, VALUE_WIDTH, left, slider_row
from crystalline.crystalio.pes import (
    DEFAULT_NSTATES,
    DEFAULT_RANGE,
    DIMENSIONS,
    QUANTITIES,
    PESRun,
    representations,
)


class PESDialog(QDialog):
    """Pick a cut and its options; read them back with :meth:`options`."""

    def __init__(self, run: PESRun, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Anharmonic PES")
        self._run = run

        layout = QVBoxLayout(self)
        summary = QLabel(run.summary, self)
        summary.setWordWrap(True)
        layout.addWidget(summary)

        cut = Section(layout, "Cut")
        self.dimension = QComboBox(self)
        for label, key in DIMENSIONS:
            self.dimension.addItem(label, key)
        # Nothing to couple means nothing to map; a run can scan a single mode.
        if not run.pairs:
            self.dimension.model().item(1).setEnabled(False)
        self.dimension.currentIndexChanged.connect(self._sync)
        cut.add("Dimension", self.dimension)

        self.span = slider_row(
            cut, "Range ±ξ", DEFAULT_RANGE, 0.5, 10.0, 0.5, decimals=1,
        )
        self.span.setToolTip(
            "Half-width of the window, in classical ground state amplitudes.\n"
            "CRYSTAL fits the constants over less than one of them, so a wide "
            "window extrapolates a quartic."
        )

        # ── one mode ──
        self._mode_box = QGroupBox("Mode", self)
        mode_layout = QVBoxLayout(self._mode_box)
        self.modes = QListWidget(self)
        for entry in run.modes:
            item = QListWidgetItem(f"{entry.label}    ({entry.detail})")
            item.setData(Qt.UserRole, entry.mode)
            self.modes.addItem(item)
        if run.modes:
            self.modes.setCurrentRow(0)
        mode_layout.addWidget(self.modes)

        mode_options = Section(mode_layout, "Overlays")
        self.harmonic = QCheckBox("Harmonic potential", self)
        self.harmonic.setChecked(True)
        self.harmonic.setToolTip("The parabola of the mode, for comparison.")
        mode_options.add_wide(self.harmonic)
        self.levels = QCheckBox("Vibrational states", self)
        self.levels.setToolTip(
            "Solve this one-mode potential and draw its levels and "
            "wavefunctions.\nThey are the states of this cut alone — a VSCF or "
            "VCI step couples the modes and lands elsewhere."
        )
        self.levels.toggled.connect(self._sync)
        mode_options.add_wide(self.levels)
        self.nstates = QSpinBox(self)
        self.nstates.setRange(1, 30)
        self.nstates.setValue(DEFAULT_NSTATES)
        self.nstates.setFixedWidth(VALUE_WIDTH)
        mode_options.add("States", left(self.nstates))
        layout.addWidget(self._mode_box)

        # ── two modes ──
        self._pair_box = QGroupBox("Pair", self)
        pair_layout = QVBoxLayout(self._pair_box)

        pair_form = QFormLayout()
        self.filter = QComboBox(self)
        self.filter.addItem("Any mode", None)
        for entry in run.modes:
            self.filter.addItem(f"Mode {entry.mode}", entry.mode)
        self.filter.currentIndexChanged.connect(self._fill_pairs)
        pair_form.addRow("Involving", self.filter)

        self.quantity = QComboBox(self)
        for label, key in QUANTITIES:
            self.quantity.addItem(label, key)
        self.quantity.setToolTip(
            "The harmonic bowl is one to two orders of magnitude deeper than "
            "anything the\nanharmonic constants add, so contours of the total "
            "surface are plain ellipses."
        )
        pair_form.addRow("Show", self.quantity)

        self.representation = QComboBox(self)
        for label, key in representations():
            self.representation.addItem(label, key)
        self.representation.setToolTip(
            "A map can be read off; a surface shows the shape.\n"
            "Once it is in the dock a surface can be turned with the mouse."
        )
        pair_form.addRow("Draw as", self.representation)
        pair_layout.addLayout(pair_form)

        self.pairs = QListWidget(self)
        self.pairs.setToolTip("Ordered by how strongly the two modes couple.")
        pair_layout.addWidget(self.pairs)
        layout.addWidget(self._pair_box)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._fill_pairs()
        self._sync()

    # ── state ───────────────────────────────────────────────────────────
    def _fill_pairs(self) -> None:
        """List the pairs the filter leaves, strongest coupling first."""
        self.pairs.clear()
        for pair in self._run.pairs_with(self.filter.currentData()):
            item = QListWidgetItem(f"{pair.label}    ({pair.detail})")
            item.setData(Qt.UserRole, (pair.modei, pair.modej))
            self.pairs.addItem(item)
        if self.pairs.count():
            self.pairs.setCurrentRow(0)
        self._sync()

    def _sync(self) -> None:
        """Show the picker the chosen cut uses, and refuse an empty one."""
        key = self.dimension.currentData()
        self._mode_box.setVisible(key == "1D")
        self._pair_box.setVisible(key == "2D")
        self.nstates.setEnabled(self.levels.isChecked())
        ready = (self.modes.currentItem() is not None if key == "1D"
                 else self.pairs.currentItem() is not None)
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(bool(ready))
        self.adjustSize()

    # ── results ─────────────────────────────────────────────────────────
    def _chosen_mode(self) -> Optional[int]:
        item = self.modes.currentItem()
        return None if item is None else item.data(Qt.UserRole)

    def _chosen_pair(self):
        item = self.pairs.currentItem()
        return (None, None) if item is None else item.data(Qt.UserRole)

    def options(self) -> dict:
        """The chosen settings, as :func:`plot_pes` keyword arguments."""
        modei, modej = self._chosen_pair()
        return {
            "dimension": self.dimension.currentData(),
            "representation": self.representation.currentData() or "map",
            "mode": self._chosen_mode(),
            "modei": modei,
            "modej": modej,
            "span": self.span.value(),
            "harmonic": self.harmonic.isChecked(),
            "levels": self.levels.isChecked(),
            "nstates": self.nstates.value(),
            "quantity": self.quantity.currentData(),
        }

    def title(self) -> str:
        """A tab title naming the cut that was picked."""
        if self.dimension.currentData() == "1D":
            return f"PES mode {self._chosen_mode()}"
        modei, modej = self._chosen_pair()
        surface = "" if self.representation.currentData() == "map" else " 3D"
        return f"PES {modei} × {modej}{surface}"


__all__ = ["PESDialog"]
