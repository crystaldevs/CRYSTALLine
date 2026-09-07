"""Choose the font every subsequent plot is drawn with.

Figures leave CRYSTALLine to go into papers and talks, where they sit next to
body text set in something other than matplotlib's default sans — Computer
Modern, if the surrounding document is LaTeX. So the family and the base size are
worth a control rather than being hard-coded.

matplotlib reads its rcParams when a figure is *created*, so the choice governs
plots opened afterwards and leaves the ones already in the dock alone. The dialog
says so instead of pretending otherwise, since silently doing nothing to the
visible figure would read as a bug.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from crystalline.crystalio.plotting import (
    DEFAULT_FONT_FAMILY,
    DEFAULT_FONT_SIZE,
    FONT_FAMILIES,
    installed_font_names,
)

# Marks the entry that swaps the family list for a free choice of installed font.
_OTHER = "\x00other"


class PlotFontDialog(QDialog):
    """Pick a font family and size; read them back with :meth:`options`."""

    def __init__(
        self,
        family: str = DEFAULT_FONT_FAMILY,
        size: float = DEFAULT_FONT_SIZE,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Plot font")

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.family = QComboBox(self)
        for label, key in FONT_FAMILIES:
            self.family.addItem(label, key)
        self.family.addItem("Other installed font…", _OTHER)
        self.family.currentIndexChanged.connect(self._sync)
        form.addRow("Family", self.family)

        # Populated once: enumerating the font cache is not instant, and the list
        # cannot change while the dialog is open.
        self.installed = QComboBox(self)
        self.installed.addItems(installed_font_names())
        self.installed.setEditable(True)
        form.addRow("Font", self.installed)

        self.size = QDoubleSpinBox(self)
        self.size.setRange(4.0, 48.0)
        self.size.setDecimals(1)
        self.size.setSingleStep(0.5)
        self.size.setValue(float(size))
        self.size.setToolTip(
            "Base size in points. Titles and tick labels scale with it, except "
            "where a plot sets its own."
        )
        form.addRow("Size (pt)", self.size)
        layout.addLayout(form)

        note = QLabel(
            "Applies to plots opened from now on; figures already in the "
            "Plots dock keep the font they were drawn with.", self
        )
        note.setWordWrap(True)
        note.setEnabled(False)
        layout.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._select(family)
        self._sync()

    # ── state ───────────────────────────────────────────────────────────
    def _select(self, family: str) -> None:
        """Preselect ``family``, whether it is a named recipe or a font name."""
        keys = [self.family.itemData(i) for i in range(self.family.count())]
        if family in keys:
            self.family.setCurrentIndex(keys.index(family))
            return
        self.family.setCurrentIndex(keys.index(_OTHER))
        self.installed.setCurrentText(family)

    def _sync(self) -> None:
        """The installed-font box matters only for the "other" choice."""
        other = self.family.currentData() == _OTHER
        self.installed.setVisible(other)
        label = self.layout().itemAt(0).layout().labelForField(self.installed)
        if label is not None:
            label.setVisible(other)
        self.adjustSize()

    def options(self) -> dict:
        """The chosen font, as :func:`apply_font` keyword arguments."""
        family = self.family.currentData()
        if family == _OTHER:
            family = self.installed.currentText().strip() or DEFAULT_FONT_FAMILY
        return {"family": family, "size": self.size.value()}


__all__ = ["PlotFontDialog"]
