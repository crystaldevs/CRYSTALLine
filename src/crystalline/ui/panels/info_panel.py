"""Info panel: crystallographic summary of the loaded system.

Shows the space group, lattice parameters, density, formula, etc. (derived from
the structure via :func:`crystalline.core.crystallography.analyze`), plus what
the CRYSTAL ``.out`` file says about itself — how the run was set up (code,
task, functional, k-point mesh, basis size, SCF thresholds) and what it
computed (energy, band gap, Fermi energy).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from crystalline.core.crystallography import analyze
from crystalline.core.structure import Structure


class InfoPanel(QWidget):
    """Read-only crystallographic + CRYSTAL-output summary."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
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

        self._crystal_group = QGroupBox("Crystallography")
        self._crystal_form = _form(self._crystal_group)
        layout.addWidget(self._crystal_group)

        self._output_group = QGroupBox("CRYSTAL output")
        self._output_form = _form(self._output_group)
        layout.addWidget(self._output_group)
        self._output_group.setVisible(False)

        layout.addStretch(1)
        self.clear()

    # ── public API ──────────────────────────────────────────────────────
    def show_structure(self, structure: Structure, output_props: Optional[dict] = None) -> None:
        """Analyse ``structure`` and display it, with optional CRYSTAL-output rows."""
        if structure is None or len(structure) == 0:
            self.clear()
            return
        _fill(self._crystal_form,
              _with_reduction(analyze(structure).rows(), structure))
        props = output_props or {}
        _fill(self._output_form, list(props.items()))
        self._output_group.setVisible(bool(props))

    def clear(self) -> None:
        _fill(self._crystal_form, [("", "No structure loaded")])
        self._output_group.setVisible(False)


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


def _form(parent: QWidget) -> QFormLayout:
    """A form whose rows survive a narrow dock.

    Values used to be clipped at the right edge when the dock was narrower than
    label + value (``8 × 8 × 8 (125 in the IBZ)`` showing as ``125 in the``).
    ``WrapLongRows`` drops the value onto its own line instead, and the fields
    are allowed to shrink rather than forcing the dock wider.
    """
    form = QFormLayout(parent)
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
