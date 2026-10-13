"""How a PES cut is drawn, as a widget rather than a dialog.

The same split as the spectra controls: *which* cut — this mode, that pair —
decides what the plot is and stays with the lists in the dialog, while how it
is drawn belongs beside the plot, where a window that extrapolates a quartic
can be seen doing it and pulled back in.

Both dimensions' knobs live here together, and the ones the current cut has no
use for are greyed rather than hidden: a one-mode cut has overlays and no
surface, a two-mode map has a surface and no levels, and a strip whose rows
appear and vanish as the cut changes is harder to read than one that stays put.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QSizePolicy, QSpinBox,
                               QVBoxLayout, QWidget)

from crystalline.crystalio.pes import (
    DEFAULT_NSTATES,
    DEFAULT_WF_SCALE,
    DEFAULT_RANGE,
    QUANTITIES,
    representations,
)
from crystalline.ui.panels.controls import (Section, left, scale_bounds,
                                            set_value_width, slider_row)


class PESOptions(QWidget):
    """Window, overlays and how a surface is drawn, with a signal when they move."""

    #: Any control was touched; the reader asks :meth:`options` for all of it.
    changed = Signal()

    def __init__(self, scale_wf: float = 0.0, parent: Optional[QWidget] = None) -> None:
        """``scale_wf`` is the run's own suggested wavefunction height, in cm⁻¹
        (see :attr:`~crystalline.crystalio.pes.PESRun.scale_wf`); the slider
        opens there and spans a decade either side of it."""
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # No stretch of its own: whoever hosts this decides what the leftover
        # height is for. Beside a plot it goes under the rows, to hold them at
        # the top; in the dialog it belongs to the list of modes, which was
        # squeezed into a few lines when this widget took it.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        # "Window", not "Cut": which cut it is — one mode or a pair — is the
        # dialog's own heading, and two sections called the same thing read as
        # one section drawn twice.
        window = Section(layout, "Window")
        self.span = slider_row(
            window, "Range ±ξ", DEFAULT_RANGE, 0.5, 10.0, 0.5, decimals=1,
            on_change=lambda _value: self.changed.emit(),
        )
        self.span.setToolTip(
            "Half-width of the window, in classical ground state amplitudes.\n"
            "CRYSTAL fits the constants over less than one of them, so a wide "
            "window extrapolates a quartic."
        )

        overlays = Section(layout, "Overlays")
        self.harmonic = QCheckBox("Harmonic potential")
        self.harmonic.setChecked(True)
        self.harmonic.setToolTip("The parabola of the mode, for comparison.")
        self.harmonic.toggled.connect(lambda _on: self.changed.emit())
        overlays.add_wide(self.harmonic)

        self.levels = QCheckBox("Vibrational states")
        self.levels.setToolTip(
            "Solve this one-mode potential and draw its levels and "
            "wavefunctions.\nThey are the states of this cut alone — a VSCF or "
            "VCI step couples the modes and lands elsewhere."
        )
        self.levels.toggled.connect(self._on_levels)
        overlays.add_wide(self.levels)

        self.nstates = QSpinBox()
        self.nstates.setRange(1, 30)
        self.nstates.setValue(DEFAULT_NSTATES)
        set_value_width(self.nstates)
        self.nstates.valueChanged.connect(lambda _n: self.changed.emit())
        overlays.add("States", left(self.nstates))

        # The wavefunctions are drawn on the potential's own axis, in cm-1, so
        # how tall to draw them is set by the data and not by the program: the
        # run's level spacing is the height that puts one between its
        # neighbours rather than flat along a level or over the top of the
        # well. The slider opens there and spans a decade either side,
        # logarithmic, as the anharmonic scan's does — a height is judged by
        # ratio, and the useful part of the travel belongs under the hand.
        suggested = float(scale_wf) or DEFAULT_WF_SCALE
        self.scale_wf = slider_row(
            overlays, "ψ scale", suggested, *scale_bounds(suggested),
            decimals=1, logarithmic=True,
            on_change=lambda _value: self.changed.emit(),
        )
        self.scale_wf.setToolTip(
            "How tall the wavefunctions are drawn, in cm⁻¹.\n"
            "Opens at this run's level spacing, which is what they are read "
            "against."
        )
        # As on a spectrum: named by default, off when the box gets in the
        # way. Here among the overlays, because the legend names *them* — the
        # harmonic parabola and the states drawn over the potential — and a
        # contour map has a colourbar rather than a legend, so a pair cut has
        # no use for it either.
        self.legend = QCheckBox("Legend")
        self.legend.setChecked(True)
        self.legend.setToolTip(
            "Name the curves drawn over the potential.\n"
            "A one-mode cut only; a map is read against its colourbar."
        )
        self.legend.toggled.connect(lambda _on: self.changed.emit())
        overlays.add_wide(self.legend)
        self._overlays = overlays

        surface = Section(layout, "Surface")
        self.quantity = QComboBox()
        for label, key in QUANTITIES:
            self.quantity.addItem(label, key)
        self.quantity.setToolTip(
            "The harmonic bowl is one to two orders of magnitude deeper than "
            "anything the\nanharmonic constants add, so contours of the total "
            "surface are plain ellipses."
        )
        self.quantity.currentIndexChanged.connect(lambda _i: self.changed.emit())
        surface.add("Show", self.quantity)

        self.representation = QComboBox()
        for label, key in representations():
            self.representation.addItem(label, key)
        self.representation.setToolTip(
            "A map can be read off; a surface shows the shape.\n"
            "Once it is in the dock a surface can be turned with the mouse."
        )
        self.representation.currentIndexChanged.connect(lambda _i: self.changed.emit())
        surface.add("Draw as", self.representation)
        self._surface = surface

        self.set_dimension("1D")

    # ── state ───────────────────────────────────────────────────────────
    def options(self) -> dict:
        """What this widget decides, as :func:`plot_pes` keyword arguments."""
        return {
            "span": self.span.value(),
            "harmonic": self.harmonic.isChecked(),
            "levels": self.levels.isChecked(),
            "nstates": self.nstates.value(),
            "scale_wf": self.scale_wf.value() or None,
            "legend": self.legend.isChecked(),
            "quantity": self.quantity.currentData(),
            "representation": self.representation.currentData() or "map",
        }

    def set_options(self, options: Optional[dict]) -> None:
        """Show ``options`` without announcing them: the caller already has them."""
        if not options:
            return
        blocked = self.blockSignals(True)
        try:
            if options.get("span") is not None:
                self.span.setValue(float(options["span"]))
            if options.get("nstates") is not None:
                self.nstates.setValue(int(options["nstates"]))
            if options.get("scale_wf"):
                self.scale_wf.setValue(float(options["scale_wf"]))
            for name in ("harmonic", "levels", "legend"):
                if options.get(name) is not None:
                    getattr(self, name).setChecked(bool(options[name]))
            for name in ("quantity", "representation"):
                index = getattr(self, name).findData(options.get(name))
                if index >= 0:
                    getattr(self, name).setCurrentIndex(index)
        finally:
            self.blockSignals(blocked)
        self.set_dimension(options.get("dimension", "1D"))

    def set_dimension(self, dimension: str) -> None:
        """Grey what this cut has no use for: a map has no levels to draw."""
        one_mode = dimension != "2D"
        # Hidden, not greyed: a one-mode cut has no surface at all, and a
        # greyed section still asks to be read before it can be dismissed.
        self._overlays.set_visible(one_mode)
        self._surface.set_visible(not one_mode)
        self._set_wavefunction_rows(one_mode and self.levels.isChecked())

    def _on_levels(self, _on: bool) -> None:
        self._set_wavefunction_rows(self.levels.isVisible() and self.levels.isChecked())
        self.changed.emit()

    def _set_wavefunction_rows(self, solving: bool) -> None:
        """How many states, and how tall: both are about states being solved."""
        self.nstates.setEnabled(solving)
        self.scale_wf.setEnabled(solving)


__all__ = ["PESOptions"]
