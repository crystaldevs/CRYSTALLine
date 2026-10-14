"""What an elastic plot shows, as controls beside it.

The same split as the other plot controls: every elastic plot of a run is a
view of one tensor, read once, so which property, which view and how it is
coloured can all be changed where the plot is, and redrawn from memory.

What does not apply is hidden rather than greyed, as elsewhere: the average,
minimum or maximum belongs only to the two properties with a second direction
to take it over (shear and Poisson's ratio), the planes only to sections, the
colour map only to a surface.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QSizePolicy,
                               QVBoxLayout, QWidget)

from crystalline.crystalio import elastic as el
from crystalline.ui.panels.controls import Section


def _combo(choices) -> QComboBox:
    box = QComboBox()
    for key, label in choices:
        box.addItem(label, key)
    return box


class ElasticControls(QWidget):
    """Property, view and look of an elastic plot; ``changed`` when any moves."""

    changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        announce = lambda *_args: self._announce()  # noqa: E731

        what = Section(layout, "Property")
        self.quantity = _combo(el.QUANTITIES)
        self.quantity.currentIndexChanged.connect(announce)
        what.add("Property", self.quantity)
        self.statistic = _combo(el.STATISTICS)
        self.statistic.setToolTip("Over the directions perpendicular to each one.")
        self.statistic.currentIndexChanged.connect(announce)
        what.add("Over the others", self.statistic)
        self._statistic_row = what.row_widgets("Over the others")

        how = Section(layout, "View")
        self.view = _combo(el.VIEWS)
        self.view.currentIndexChanged.connect(announce)
        how.add("Show as", self.view)
        planes = QWidget()
        line = QHBoxLayout(planes)
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(12)
        self.planes = {}
        for plane in el.PLANES:
            tick = QCheckBox(plane)
            tick.setChecked(True)
            tick.toggled.connect(lambda _on, p=plane: self._on_plane(p))
            line.addWidget(tick)
            self.planes[plane] = tick
        line.addStretch(1)
        how.add("Planes", planes)
        self._planes_row = how.row_widgets("Planes")

        look = Section(layout, "Appearance", collapsed=True)
        self.cmap = QComboBox()
        self.cmap.addItems(el.COLOUR_MAPS)
        self.cmap.setToolTip("viridis and cividis are even in lightness, and read in "
                             "greyscale and for colour-blind readers.")
        self.cmap.currentIndexChanged.connect(announce)
        look.add("Colour map", self.cmap)
        self._cmap_row = look.row_widgets("Colour map")
        self.resolution = QComboBox()
        for label, steps in el.RESOLUTIONS:
            self.resolution.addItem(label, steps)
        self.resolution.setToolTip("How many directions the property is evaluated in. "
                                   "Fine takes noticeably longer.")
        self.resolution.currentIndexChanged.connect(announce)
        look.add("Resolution", self.resolution)

        self._quiet = False
        self.set_options(dict(el.DEFAULTS))

    # ── state ───────────────────────────────────────────────────────────
    def options(self) -> dict:
        return {
            "quantity": self.quantity.currentData(),
            "statistic": self.statistic.currentData(),
            "view": self.view.currentData(),
            "planes": tuple(p for p in el.PLANES if self.planes[p].isChecked()),
            "cmap": self.cmap.currentText(),
            "resolution": self.resolution.currentData(),
        }

    def set_options(self, options: Optional[dict]) -> None:
        """Show a plot's options without announcing them: it already has them."""
        if not options:
            return
        self._quiet = True
        try:
            for box, key in ((self.quantity, "quantity"), (self.statistic, "statistic"),
                             (self.view, "view"), (self.resolution, "resolution")):
                index = box.findData(options.get(key))
                if index >= 0:
                    box.setCurrentIndex(index)
            if options.get("cmap"):
                index = self.cmap.findText(options["cmap"])
                if index >= 0:
                    self.cmap.setCurrentIndex(index)
            if options.get("planes") is not None:
                wanted = set(options["planes"])
                for plane, tick in self.planes.items():
                    tick.setChecked(plane in wanted)
        finally:
            self._quiet = False
        self._show_what_applies()

    # ── changes ─────────────────────────────────────────────────────────
    def _announce(self) -> None:
        self._show_what_applies()
        if not self._quiet:
            self.changed.emit()

    def _on_plane(self, plane: str) -> None:
        """At least one plane: unticking the last would leave an empty plot."""
        if not any(tick.isChecked() for tick in self.planes.values()):
            self.planes[plane].blockSignals(True)
            self.planes[plane].setChecked(True)
            self.planes[plane].blockSignals(False)
            return
        self._announce()

    def _show_what_applies(self) -> None:
        for widget in self._statistic_row:
            widget.setVisible(el.needs_statistic(self.quantity.currentData()))
        sections = self.view.currentData() == "sections"
        for widget in self._planes_row:
            widget.setVisible(sections)
        for widget in self._cmap_row:
            widget.setVisible(not sections)


__all__ = ["ElasticControls"]
