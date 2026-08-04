"""Symmetry panel: show the structure's point-symmetry elements over it.

Docked beside Info, Display and Geometry. The analysis
(:mod:`crystalline.core.symmetry`) turns the point group's operators into the
axes, planes and centre they act about; this panel lists them under **Axes /
Planes / Centre** — so a whole type is one click, the heading's tick following
its children — and draws the ticked ones in 3D, with the same three shapes the
Geometry panel draws its measurements with, in their own colours.

The analysis only runs when it is going to be seen: most sessions never open
this panel. Editing the structure marks the result stale rather than
recomputing it — a half-finished edit has no symmetry worth reporting.
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from crystalline.core import symmetry as symmetry_mod
from crystalline.core.structure import Structure

# The three groups, in the order the analysis sorts its elements.
_GROUPS = (
    # Crystals are labelled in Hermann–Mauguin and molecules in Schoenflies, as
    # their groups are, so the tooltips name both.
    (symmetry_mod.AXIS, "Symmetry axes",
     "Rotation axes (2, 3, 4, 6 — C₂, C₃, C₄, C₆) and the rotoinversion\n"
     "axes that share them (3̄, 4̄, 6̄ — the rotoreflections S₆, S₄, S₃)"),
    (symmetry_mod.PLANE, "Mirror planes", "Reflection planes (m — σ)"),
    (symmetry_mod.POINT, "Centre of inversion",
     "The point every atom reflects through (1̄ — i)"),
)


class SymmetryPanel(QWidget):
    """List the structure's symmetry elements and draw the ticked ones."""

    # the elements that should be drawn in 3D, and whether to label them
    elements_changed = Signal(list, bool)

    def __init__(self, structure: Structure, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._structure = structure
        self._elements: List[symmetry_mod.SymmetryElement] = []
        self._analysed = False  # whether _elements reflects the current structure

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._show = QCheckBox("Show point-symmetry elements")
        self._show.setToolTip(
            "Find the point symmetry of the structure — the rotation axes, mirror\n"
            "planes and centre of inversion that hold one point of it fixed — and\n"
            "draw the ticked ones over it."
        )
        self._show.toggled.connect(self._on_show_toggled)
        layout.addWidget(self._show)

        row = QHBoxLayout()
        row.addWidget(QLabel("Tolerance"))
        self._symprec = QDoubleSpinBox()
        self._symprec.setDecimals(4)
        self._symprec.setRange(1e-4, 1.0)
        self._symprec.setSingleStep(0.005)
        self._symprec.setValue(0.01)
        self._symprec.setSuffix(" Å")
        self._symprec.setToolTip(
            "How far an atom may sit from its symmetric position and still count.\n"
            "Loosen it to recognise a nearly-symmetric geometry (a relaxed cell),\n"
            "tighten it to insist on an exact one."
        )
        self._symprec.editingFinished.connect(self._reanalyse)
        row.addWidget(self._symprec)
        row.addStretch(1)
        layout.addLayout(row)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: palette(mid);")
        layout.addWidget(self._status)

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._tree.setToolTip("Tick an element to draw it in the 3D view")
        self._tree.itemChanged.connect(self._on_item_changed)
        self._tree.itemSelectionChanged.connect(self._sync_buttons)
        layout.addWidget(self._tree, 1)

        row = QHBoxLayout()
        self._all_btn = QPushButton("All")
        self._all_btn.setToolTip("Draw every element")
        self._all_btn.clicked.connect(lambda: self._set_all(True))
        row.addWidget(self._all_btn)
        self._none_btn = QPushButton("None")
        self._none_btn.clicked.connect(lambda: self._set_all(False))
        row.addWidget(self._none_btn)
        row.addStretch(1)
        layout.addLayout(row)

        self._labels = QCheckBox("Labels")
        self._labels.setToolTip(
            "Write each element's Hermann–Mauguin symbol (4, m, 1̄) beside it in the 3D view"
        )
        self._labels.toggled.connect(lambda _on: self._emit())
        layout.addWidget(self._labels)

        self._sync_buttons()

    # ── external hooks ──────────────────────────────────────────────────
    def show_analysis(self) -> None:
        """Run the analysis and draw what it finds, unless that has been done.

        What "Point symmetry analysis" in the Cell menu means: the panel opens
        with the answer already in it rather than with a box to tick. Asking for
        it again after switching the elements off leaves that choice alone — the
        panel is already showing its analysis, which is what was asked for.
        """
        if self._analysed:
            return
        if self._show.isChecked():
            self._analyse()  # ticked, but with nothing found yet to draw
        else:
            self._show.setChecked(True)  # -> analyse, and draw what it finds

    def set_structure(self, structure: Structure) -> None:
        """Rebind after a load or a view change; the old elements no longer apply."""
        self._structure = structure
        self._clear(analysed=False)
        if self._show.isChecked():
            self._analyse()
        else:
            self._emit()

    def invalidate(self, structure: Structure) -> None:
        """The geometry was edited: rebind to it, and drop what is drawn.

        Re-running the search on every nudge of an atom would be both slow and
        misleading — a half-finished edit has no symmetry — so the panel waits to
        be asked instead.
        """
        self._structure = structure
        if not self._analysed:
            return
        self._clear(analysed=False)
        if self._show.isChecked():
            self._status.setText(
                "Structure edited — re-tick 'Show point-symmetry elements' to update."
            )
            blocked = self._show.blockSignals(True)
            self._show.setChecked(False)
            self._show.blockSignals(blocked)
        self._emit()

    def elements(self) -> List[symmetry_mod.SymmetryElement]:
        """Every element found, shown or not (handy in tests)."""
        return list(self._elements)

    def shown_elements(self) -> List[symmetry_mod.SymmetryElement]:
        """The elements currently ticked for display in 3D."""
        return [
            element
            for element, item in self._rows()
            if item.checkState(0) == Qt.Checked
        ]

    # ── analysis ────────────────────────────────────────────────────────
    def _on_show_toggled(self, enabled: bool) -> None:
        if enabled and not self._analysed:
            self._analyse()
        else:
            self._emit()

    def _reanalyse(self) -> None:
        """Re-run the search after the tolerance changed (if it is being shown)."""
        if self._show.isChecked():
            self._analyse()

    def _analyse(self) -> None:
        analysis = symmetry_mod.analyse(self._structure, symprec=float(self._symprec.value()))
        self._elements = analysis.elements
        self._analysed = True
        self._fill(analysis)
        self._emit()

    def _fill(self, analysis) -> None:
        """Rebuild the tree with everything ticked — a point group is small enough."""
        elements = analysis.elements
        blocked = self._tree.blockSignals(True)
        self._tree.clear()
        for kind, title, tip in _GROUPS:
            members = [e for e in elements if e.kind == kind]
            if not members:
                continue
            parent = QTreeWidgetItem(self._tree, [f"{title} ({len(members)})"])
            parent.setToolTip(0, tip)
            parent.setFlags(parent.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
            parent.setExpanded(True)
            for element in members:
                item = QTreeWidgetItem(parent, [element.summary()])
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(0, Qt.Checked)
                item.setData(0, Qt.UserRole, element)
        self._tree.blockSignals(blocked)
        # Where the elements meet is said once, here, rather than on every row:
        # point symmetry puts all of them through the same centre.
        self._status.setText(analysis.summary())
        self._status.setToolTip(
            "The symmetry operations that hold this point of the structure fixed.\n"
            "A crystal's screw axes and glide planes are not among them: they move\n"
            "the structure along as they turn or reflect it, so they hold no point."
        )
        self._sync_buttons()

    def _clear(self, analysed: bool) -> None:
        self._elements = []
        self._analysed = analysed
        blocked = self._tree.blockSignals(True)
        self._tree.clear()
        self._tree.blockSignals(blocked)
        self._status.setText("")
        self._sync_buttons()

    # ── list interaction ────────────────────────────────────────────────
    def _rows(self):
        """``(element, item)`` for every element row (not the group headers)."""
        for index in range(self._tree.topLevelItemCount()):
            parent = self._tree.topLevelItem(index)
            for child_index in range(parent.childCount()):
                item = parent.child(child_index)
                yield item.data(0, Qt.UserRole), item

    def _on_item_changed(self, _item, _column) -> None:
        self._emit()

    def _set_all(self, shown: bool) -> None:
        state = Qt.Checked if shown else Qt.Unchecked
        blocked = self._tree.blockSignals(True)
        for _element, item in self._rows():
            item.setCheckState(0, state)
        self._tree.blockSignals(blocked)
        # Parents are auto-tristate and follow their children, but only when the
        # children's changes are signalled — which they were not, so set them too.
        for index in range(self._tree.topLevelItemCount()):
            self._tree.topLevelItem(index).setCheckState(0, state)

    def _sync_buttons(self) -> None:
        has_elements = self._tree.topLevelItemCount() > 0
        self._all_btn.setEnabled(has_elements)
        self._none_btn.setEnabled(has_elements)
        self._labels.setEnabled(has_elements)

    def _emit(self) -> None:
        shown = self.shown_elements() if self._show.isChecked() else []
        self.elements_changed.emit(shown, self._labels.isChecked())



__all__ = ["SymmetryPanel"]
