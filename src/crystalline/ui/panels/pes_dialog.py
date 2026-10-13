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
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from crystalline.ui.panels.controls import Section, chooser_size
from crystalline.ui.panels.pes_options import PESOptions
from crystalline.crystalio.pes import DIMENSIONS, PESRun
from crystalline.ui.widgets.wrapped_label import WrappedLabel


# A floor for the lists: the controls above them have a minimum height of
# their own, and the modes are the thing being chosen from.
_LIST_MIN_HEIGHT = 180


class PESDialog(QDialog):
    """Pick a cut and its options; read them back with :meth:`options`."""

    def __init__(self, run: PESRun, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Anharmonic PES")
        self._run = run

        layout = QVBoxLayout(self)
        summary = WrappedLabel(run.summary, self)
        layout.addWidget(summary)

        which = Section(layout, "Cut")
        self.dimension = QComboBox(self)
        for label, key in DIMENSIONS:
            self.dimension.addItem(label, key)
        # Nothing to couple means nothing to map; a run can scan a single mode.
        if not run.pairs:
            self.dimension.model().item(1).setEnabled(False)
        self.dimension.currentIndexChanged.connect(self._sync)
        which.add("Dimension", self.dimension)

        # How the cut is drawn, in the same widget the plot itself carries, so
        # that tuning it here and tuning it there cannot drift apart. What is
        # *not* here is which mode or which pair: that decides what the plot
        # is, and it stays with the lists below.
        self.options_widget = PESOptions(run.scale_wf, self)
        layout.addWidget(self.options_widget)
        # Named here too: dialog_state remembers a dialog by walking the
        # widgets it holds, and these names are what the tests reach for.
        self.span = self.options_widget.span
        self.harmonic = self.options_widget.harmonic
        self.levels = self.options_widget.levels
        self.nstates = self.options_widget.nstates
        self.quantity = self.options_widget.quantity
        self.representation = self.options_widget.representation
        self.levels.toggled.connect(self._sync)

        # ── one mode ──
        self._mode_box = QGroupBox("Mode", self)
        mode_layout = QVBoxLayout(self._mode_box)
        self.modes = QListWidget(self)
        self.modes.setMinimumHeight(_LIST_MIN_HEIGHT)
        for entry in run.modes:
            item = QListWidgetItem(f"{entry.label}    ({entry.detail})")
            item.setData(Qt.UserRole, entry.mode)
            self.modes.addItem(item)
        if run.modes:
            self.modes.setCurrentRow(0)
        mode_layout.addWidget(self.modes)
        # The list is what wants the room: it is the thing being chosen from.
        layout.addWidget(self._mode_box, 1)

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

        pair_layout.addLayout(pair_form)

        self.pairs = QListWidget(self)
        self.pairs.setMinimumHeight(_LIST_MIN_HEIGHT)
        self.pairs.setToolTip("Ordered by how strongly the two modes couple.")
        pair_layout.addWidget(self.pairs)
        layout.addWidget(self._pair_box, 1)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._fill_pairs()
        self._sync()
        self.resize(*chooser_size(self, (self.modes, self.pairs)))

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
        self.options_widget.set_dimension(key)
        ready = (self.modes.currentItem() is not None if key == "1D"
                 else self.pairs.currentItem() is not None)
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(bool(ready))
        # Deliberately not adjustSize(): this runs on every change of cut, and
        # a window that resizes itself under the hand is both startling and a
        # way to lose whatever size the user had chosen. The layout gives the
        # freed room to the list that is now showing.

    # ── results ─────────────────────────────────────────────────────────
    def _chosen_mode(self) -> Optional[int]:
        item = self.modes.currentItem()
        return None if item is None else item.data(Qt.UserRole)

    def _chosen_pair(self):
        item = self.pairs.currentItem()
        return (None, None) if item is None else item.data(Qt.UserRole)

    def selection_state(self) -> dict:
        """The cut that was chosen, named — what :mod:`dialog_state` keeps.

        A list's *row* means nothing to the next run, which scanned other
        modes; the mode numbers do, so those are what travel. The same rule as
        a combo, whose choice is kept by its data and never by its index.
        """
        modei, modej = self._chosen_pair()
        return {"mode": self._chosen_mode(), "pair": [modei, modej],
                "filter": self.filter.currentData()}

    def set_selection_state(self, chosen) -> None:
        """Pick those modes again, as far as this run has them."""
        if not isinstance(chosen, dict):
            return
        index = self.filter.findData(chosen.get("filter"))
        if index >= 0:
            self.filter.setCurrentIndex(index)   # refills the pairs
        _select(self.modes, chosen.get("mode"))
        pair = chosen.get("pair") or [None, None]
        _select(self.pairs, tuple(pair) if None not in pair else None)
        self._sync()

    def options(self) -> dict:
        """The chosen settings, as :func:`plot_pes` keyword arguments."""
        modei, modej = self._chosen_pair()
        return {
            "dimension": self.dimension.currentData(),
            "mode": self._chosen_mode(),
            "modei": modei,
            "modej": modej,
            **self.options_widget.options(),
        }

    def title(self) -> str:
        """A tab title naming the cut that was picked."""
        if self.dimension.currentData() == "1D":
            return f"PES mode {self._chosen_mode()}"
        modei, modej = self._chosen_pair()
        surface = "" if self.representation.currentData() == "map" else " 3D"
        return f"PES {modei} × {modej}{surface}"


def _select(listing, wanted) -> None:
    """Make the row carrying ``wanted`` the current one, if this run has it."""
    if wanted is None:
        return
    for row in range(listing.count()):
        if listing.item(row).data(Qt.UserRole) == wanted:
            listing.setCurrentRow(row)
            return


__all__ = ["PESDialog"]
