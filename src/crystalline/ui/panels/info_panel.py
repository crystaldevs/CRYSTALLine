"""Info panel: crystallographic summary of the loaded system.

Shows the space group, lattice parameters, density, formula, etc. (derived from
the structure via :func:`crystalline.core.crystallography.analyze`), plus what
the CRYSTAL ``.out`` file says about itself — how the run was set up (code,
task, functional, k-point mesh, basis size, SCF thresholds) and what it
computed (energy, band gap, Fermi energy).

A crystal can be described in its own cell, the one its file holds, or in
pymatgen's standard conventional cell — two sets of lattice parameters for one crystal
whenever the calculation was run in a non-standard setting (P2₁/n rather than
P2₁/c, say). The **Cell** box at the top chooses which one the rows describe.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from crystalline.core.cell_setting import CELL_CHOICES, DEFAULT_CHOICE
from crystalline.core.crystallography import analyze
from crystalline.core.structure import Structure

_CELL_TOOLTIP = (
    "Which cell the lattice parameters describe.\n\n"
    "As in the file: the cell as the file holds it — for a CRYSTAL output, the "
    "cell the calculation ran in — with the space group named in that cell's "
    "setting, so P2₁/n stays P2₁/n.\n"
    "Standard setting: the conventional standard cell, which can be a "
    "different setting of the same group, with other lattice parameters."
)
_CELL_TOOLTIP_NA = "Only a 3D crystal has a choice of cell to describe it in."


class InfoPanel(QWidget):
    """Read-only crystallographic + CRYSTAL-output summary."""

    #: The Cell box was changed by the user: ``"computed"`` or ``"standard"``.
    cell_choice_changed = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._structure: Optional[Structure] = None
        self._props: dict = {}
        # A CRYSTAL run reports a couple of dozen rows between the two groups —
        # more than a dock is tall — so the whole summary scrolls rather than
        # forcing the dock wider or clipping the last rows.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.NoFrame)
        outer.addWidget(self._scroll)

        content = QWidget()
        self._scroll.setWidget(content)
        layout = QVBoxLayout(content)

        self._cell = QComboBox()
        for key, label in CELL_CHOICES:
            self._cell.addItem(label, key)
        self._cell.setCurrentIndex(self._cell.findData(DEFAULT_CHOICE))
        self._cell.setToolTip(_CELL_TOOLTIP)
        self._cell.currentIndexChanged.connect(self._on_cell_changed)
        # As wide as its longest entry and no wider. A form row stretched it
        # across the whole dock, which made a two-item choice look like the most
        # important thing in the panel.
        self._cell.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._cell.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)

        # Inside the group it governs, at the top of it: the choice belongs to
        # the Crystallography rows — it is which cell they describe — so it sits
        # within that heading rather than floating above it as a section of its
        # own. The rows are refilled on every edit, so the choice is kept out of
        # that form and in a box around it.
        self._crystal_group = QGroupBox("Crystallography")
        crystal_box = QVBoxLayout(self._crystal_group)
        choice = QHBoxLayout()
        choice.setContentsMargins(0, 0, 0, 0)
        choice.addWidget(QLabel("Cell:"))
        choice.addWidget(self._cell)
        choice.addStretch(1)
        crystal_box.addLayout(choice)
        self._crystal_form = _form(None)
        crystal_box.addLayout(self._crystal_form)
        layout.addWidget(self._crystal_group)

        self._output_group = QGroupBox("CRYSTAL output")
        self._output_form = _form(self._output_group)
        layout.addWidget(self._output_group)
        self._output_group.setVisible(False)

        layout.addStretch(1)
        self.clear()

    # ── public API ──────────────────────────────────────────────────────
    def show_structure(self, structure: Structure, output_props: Optional[dict] = None) -> None:
        """Analyse ``structure`` and display it, with optional CRYSTAL-output rows.

        ``structure`` should be one clean cell — the cell on screen folded back
        from any supercell or boundary images — since "as in the file" describes
        exactly the cell it is handed.
        """
        if structure is None or len(structure) == 0:
            self.clear()
            return
        self._structure = structure
        self._props = dict(output_props or {})
        self._render()

    def set_output_props(self, output_props: Optional[dict]) -> None:
        """Show the CRYSTAL output's own rows, leaving the crystallography alone.

        They arrive after the structure (an output's account of its run costs
        more to read than its geometry), so they are put in by themselves
        rather than by analysing the cell again.
        """
        self._props = dict(output_props or {})
        if self._structure is None:
            return
        _fill(self._output_form, list(self._props.items()))
        self._output_group.setVisible(bool(self._props))

    def cell_choice(self) -> str:
        """``"computed"`` or ``"standard"`` — what the Cell box is set to."""
        return str(self._cell.currentData())

    def set_cell_choice(self, choice: str) -> None:
        """Set the Cell box without announcing it (it is not the user's doing)."""
        index = self._cell.findData(choice)
        if index < 0 or index == self._cell.currentIndex():
            return
        blocked = self._cell.blockSignals(True)
        self._cell.setCurrentIndex(index)
        self._cell.blockSignals(blocked)
        self._render()

    def clear(self) -> None:
        self._structure = None
        self._props = {}
        _fill(self._crystal_form, [("", "No structure loaded")])
        self._output_group.setVisible(False)
        self._sync_cell_box()

    # ── internals ───────────────────────────────────────────────────────
    def _on_cell_changed(self, _index: int) -> None:
        self._render()
        self.cell_choice_changed.emit(self.cell_choice())

    def _render(self) -> None:
        self._sync_cell_box()
        structure = self._structure
        if structure is None:
            return
        info = analyze(structure, cell=self.cell_choice())
        _fill(self._crystal_form, _with_reduction(info.rows(), structure))
        _fill(self._output_form, list(self._props.items()))
        self._output_group.setVisible(bool(self._props))

    def _sync_cell_box(self) -> None:
        """Offer the choice only where there is one: a 3D crystal."""
        structure = self._structure
        crystal = (structure is not None and len(structure) > 0
                   and bool(structure.is_periodic) and bool(all(structure.pbc)))
        self._cell.setEnabled(crystal)
        self._cell.setToolTip(_CELL_TOOLTIP if crystal else _CELL_TOOLTIP_NA)


def _with_reduction(rows: List[Tuple[str, str]],
                    structure: Structure) -> List[Tuple[str, str]]:
    """Splice the reduction in directly under the space group.

    Beside it, not at the bottom of the panel: the two lines answer the same
    question — which group is this crystal in — and the second only makes sense
    while the first is still in the eye.
    """
    extra = _reduction_rows(structure)
    if not extra:
        return list(rows)
    for index, (label, _value) in enumerate(rows):
        if label.lower().startswith("space group"):
            return list(rows[:index + 1]) + extra + list(rows[index + 1:])
    return list(rows) + extra


def _reduction_rows(structure: Structure) -> List[Tuple[str, str]]:
    """A row saying the crystal is being *treated* as less symmetric than it is.

    Both facts belong here, and they are different facts: the space group above
    is the symmetry these atoms have, and this is the one they are being given.
    No atom moves in a reduction, so without a line saying so it is an
    invisible setting deciding what every deck contains.
    """
    kept = getattr(structure, "reduced_symmetry", ())
    if not kept:
        return []
    try:
        import numpy as np

        from crystalline.core import symmetry_reduction as reduction

        symmetry = reduction.analyse(structure)
        if symmetry is None:
            return [("Treated as", f"{len(kept)} point operators")]
        current = reduction.descend(symmetry, [np.asarray(r, dtype=int) for r in kept])
        full = symmetry.full()
        return [
            ("Treated as", f"{current.symbol} (No. {current.number})"),
            ("Independent sites", f"{current.sites} (was {full.sites})"),
        ]
    except Exception:  # noqa: BLE001 - a row we cannot fill is not worth a crash
        return [("Treated as", "a reduced symmetry")]


def _form(parent: Optional[QWidget]) -> QFormLayout:
    """A form whose rows survive a narrow dock.

    A dock narrower than label + value would clip the value at the right edge
    (``8 × 8 × 8 (125 in the IBZ)`` showing as ``125 in the``).
    ``WrapLongRows`` drops the value onto its own line instead, and the fields
    are allowed to shrink rather than forcing the dock wider.
    """
    form = QFormLayout(parent) if parent is not None else QFormLayout()
    form.setRowWrapPolicy(QFormLayout.WrapLongRows)
    form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    return form


def _fill(form: QFormLayout, rows: List[Tuple[str, str]]) -> None:
    """Replace the form's contents with ``(label, value)`` rows."""
    while form.rowCount():
        form.removeRow(0)
    for label, value in rows:
        value_label = QLabel(str(value))
        value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)  # copyable
        # Long values (functional names above all) wrap instead of forcing the
        # dock wider than the user sized it.
        value_label.setWordWrap(True)
        form.addRow(f"{label}:" if label else "", value_label)


__all__ = ["InfoPanel"]
