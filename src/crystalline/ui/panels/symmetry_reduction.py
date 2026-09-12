"""Choosing a lower symmetry for the crystal, by giving up point operators.

The gesture is "remove this operator"; the arithmetic underneath is "descend to
a subgroup that lacks it", and the two are not the same thing. Removing the
inversion from m3̄m takes twenty-three other operators with it and leaves *two*
different crystals to choose between — so unticking a box cannot simply do
something. It asks, and this dialog is where it asks.

Nothing here moves an atom. What changes is how many of them are independent:
the count beside each candidate is the size of the asymmetric unit, which is
exactly how many sites a geometry optimisation would then be free to move.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from crystalline.core import symmetry_reduction as reduction
from crystalline.core.structure import Structure
from crystalline.ui.safety import guard


class SymmetryReductionDialog(QDialog):
    """Untick operators; pick which of the surviving groups you meant."""

    def __init__(self, structure: Structure, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Reduce symmetry")
        self.resize(760, 520)
        self._structure = structure
        self._symmetry = reduction.analyse(structure)
        self._current: Optional[reduction.Subgroup] = None
        self._options: List[reduction.Subgroup] = []

        outer = QVBoxLayout(self)
        note = QLabel(
            "Untick an operator to give it up. What is left has to be a group, "
            "so others go with it — often many others, and sometimes there is "
            "more than one way to do it. The choices appear on the right; the "
            "count is how many atoms become free to move independently."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: palette(mid);")
        outer.addWidget(note)

        self._header = QLabel()
        font = self._header.font()
        font.setBold(True)
        self._header.setFont(font)
        self._header.setWordWrap(True)
        outer.addWidget(self._header)

        body = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(QLabel("Point operators"))
        self._operators = QListWidget()
        self._operators.setToolTip(
            "Ticked operators are the symmetry the crystal is being given. "
            "Untick one to ask what is left without it."
        )
        self._operators.itemChanged.connect(self._on_operator_toggled)
        left.addWidget(self._operators, 1)
        body.addLayout(left, 3)

        right = QVBoxLayout()
        right.addWidget(QLabel("What that leaves"))
        self._choices = QListWidget()
        self._choices.setToolTip(
            "Each is a real subgroup. A greyed one cannot be written: its "
            "standard setting is not the cell this crystal is held in."
        )
        self._choices.itemDoubleClicked.connect(lambda _item: self._take_choice())
        right.addWidget(self._choices, 1)
        take = QPushButton("Use this symmetry")
        take.clicked.connect(lambda: self._take_choice())
        right.addWidget(take)
        restore = QPushButton("Restore full symmetry")
        restore.clicked.connect(lambda: self._restore())
        right.addWidget(restore)
        body.addLayout(right, 4)
        outer.addLayout(body)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        self._start()

    # ── state ───────────────────────────────────────────────────────────
    def _start(self) -> None:
        if self._symmetry is None:
            self._header.setText(_nothing_to_reduce(self._structure))
            self._operators.setEnabled(False)
            self._choices.setEnabled(False)
            return
        kept = self._structure.reduced_symmetry
        self._current = (reduction.descend(self._symmetry, [np.asarray(r, dtype=int)
                                                            for r in kept])
                         if kept else self._symmetry.full())
        self._fill_operators()
        self._refresh_header()

    def _refresh_header(self) -> None:
        current, full = self._current, self._symmetry.full()
        if current.order == full.order:
            self._header.setText(
                f"{full.symbol} (No. {full.number}) — the crystal's own symmetry: "
                f"{full.order} point operators, {full.sites} independent sites")
            return
        self._header.setText(
            f"{current.symbol} (No. {current.number}) — reduced from {full.symbol}: "
            f"{current.order} of {full.order} point operators, "
            f"{current.sites} independent sites (was {full.sites})")

    def _fill_operators(self) -> None:
        """Every operator of the full group; ticked means the reduction keeps it."""
        self._operators.blockSignals(True)
        self._operators.clear()
        kept = {reduction._key(rotation) for rotation in self._current.rotations}
        for operator in self._symmetry.operators:
            item = QListWidgetItem(operator.label)
            item.setData(Qt.UserRole, operator.rotation)
            present = operator.key() in kept
            item.setCheckState(Qt.Checked if present else Qt.Unchecked)
            if operator.is_identity:
                # Every group contains it; a tick box that cannot be unticked
                # is a lie, so it is shown as fixed instead.
                item.setFlags(item.flags() & ~Qt.ItemIsUserCheckable)
                item.setForeground(QColor("gray"))
            elif not present:
                item.setForeground(QColor("gray"))
            self._operators.addItem(item)
        self._operators.blockSignals(False)

    # ── the descent ─────────────────────────────────────────────────────
    @guard()
    def _on_operator_toggled(self, item: QListWidgetItem) -> None:
        """Unticking asks a question; it does not answer it.

        Re-ticking is not the inverse of unticking — putting an operator back
        would mean climbing to some supergroup, which is a different and much
        less well-posed question — so the tick is restored and the user is
        pointed at the restore button instead.
        """
        rotation = np.asarray(item.data(Qt.UserRole), dtype=int)
        if item.checkState() == Qt.Checked:
            self._fill_operators()          # putting one back: show the truth again
            self._choices.clear()
            self._choices.addItem("To go back up, use “Restore full symmetry”.")
            return
        self._offer(rotation, item.text())

    def _offer(self, rotation, label: str) -> None:
        self._options = reduction.subgroups_without(
            self._symmetry, [rotation], within=self._current.rotations)
        self._choices.clear()
        if not self._options:
            self._choices.addItem(f"Nothing can be left without {label}.")
            self._fill_operators()
            return
        for option in self._options:
            lost = self._current.order - option.order
            text = (f"{option.symbol}  (No. {option.number})\n"
                    f"    {option.order} operators — {lost} given up, "
                    f"{option.sites} independent sites")
            item = QListWidgetItem(text)
            if not option.writable:
                item.setForeground(QColor("gray"))
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
                item.setToolTip(
                    "A real subgroup, but its standard setting is not this "
                    "crystal's cell — a deck written from it would describe a "
                    "different crystal, so it cannot be used from here."
                )
            self._choices.addItem(item)
        writable = [index for index, option in enumerate(self._options)
                    if option.writable]
        if writable:
            self._choices.setCurrentRow(writable[0])

    @guard()
    def _take_choice(self) -> None:
        row = self._choices.currentRow()
        if not (0 <= row < len(self._options)):
            return
        option = self._options[row]
        if not option.writable:
            return
        self._current = option
        self._options = []
        self._choices.clear()
        self._fill_operators()
        self._refresh_header()

    @guard()
    def _restore(self) -> None:
        self._current = self._symmetry.full()
        self._options = []
        self._choices.clear()
        self._fill_operators()
        self._refresh_header()

    # ── the answer ──────────────────────────────────────────────────────
    def chosen(self):
        """The operators to keep, or ``()`` for the crystal's own symmetry."""
        if self._symmetry is None or self._current is None:
            return ()
        if self._current.order == self._symmetry.full().order:
            return ()
        return tuple(tuple(map(tuple, rotation)) for rotation in self._current.rotations)

    @classmethod
    def run(cls, structure: Structure, parent=None) -> bool:
        """Show the dialog and apply what it chose. ``True`` if anything changed."""
        dialog = cls(structure, parent)
        if dialog.exec() != QDialog.Accepted:
            return False
        chosen = dialog.chosen()
        if tuple(structure.reduced_symmetry) == chosen:
            return False
        structure.set_reduced_symmetry(chosen)
        return True


def _nothing_to_reduce(structure) -> str:
    """Say *why* there is nothing to do, which differs by dimensionality."""
    try:
        periodic = int(sum(bool(p) for p in structure.pbc))
    except Exception:  # noqa: BLE001
        periodic = 0
    if periodic == 2:
        return ("A slab's symmetry is a layer group, not a space group, and "
                "reducing one is not supported yet — its deck is written in "
                "the layer group found for it.")
    if periodic == 1:
        return "A polymer's symmetry is a rod group; reducing one is not supported yet."
    if periodic == 0:
        return "A molecule has no crystal symmetry to reduce."
    return "No symmetry could be found for this structure."


__all__ = ["SymmetryReductionDialog"]
