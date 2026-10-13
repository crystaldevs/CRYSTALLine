"""The broadening and range of a spectrum, as a widget rather than a dialog.

A spectrum is tuned by eye: a width is judged against the peaks it separates or
merges, and the only way to know is to look. Behind a modal dialog that means
guessing a number, pressing OK, looking, and opening the dialog again — and
since the plot arrives in a tab of its own each time, comparing two widths means
comparing two tabs of the same spectrum.

So the controls live here, in a widget, and the dialog is one of two places that
hosts it: the other is the plot itself, where turning a knob redraws what is on
screen (see :class:`~crystalline.ui.panels.plot_view.PlotRecipe`). Both show the
same controls, because they are the same controls.

What is *not* here is which curves to draw. That is a tree of everything the
output holds, it is as tall as the dialog, and it decides what the plot is
rather than how it looks; it stays where there is room for it.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QSizePolicy, QVBoxLayout,
                               QWidget)

from crystalline.crystalio.spectra import (
    BROADENING_PARAMETERS,
    DEFAULT_ETA,
    DEFAULT_HWHM,
    DEFAULT_LINEWIDTH,
    DEFAULT_STDEV,
    LINESHAPES,
)
from crystalline.ui.panels.controls import Section, range_row, slider_row

# Label, and the plot keyword each row is read back into.
_WIDTHS = (
    ("hwhm", "Lorentzian HWHM (cm⁻¹)", DEFAULT_HWHM, 0.1, 200.0, 1.0, 1, True,
     "Half-width at half-maximum of the Lorentzian profile."),
    ("stdev", "Gaussian std. dev. (cm⁻¹)", DEFAULT_STDEV, 0.1, 200.0, 1.0, 1, True,
     "Standard deviation of the Gaussian profile."),
    ("eta", "Mixing η (Lorentzian fraction)", DEFAULT_ETA, 0.0, 1.0, 0.05, 2, False,
     "Lorentzian fraction of the pseudo-Voigt mixture:\n"
     "1.00 is pure Lorentzian, 0.00 is pure Gaussian."),
)


class SpectraOptions(QWidget):
    """Lineshape, widths and frequency range, and a signal when any of them moves."""

    #: Any control was touched. Carries nothing: the reader asks :meth:`options`,
    #: which is the whole answer and cannot go stale between the two.
    changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # No stretch of its own: whoever hosts this decides what the leftover
        # height is for. Beside a plot it goes under the rows, to hold them at
        # the top; in the dialog it belongs to the list of modes, which was
        # squeezed into a few lines when this widget took it.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        broadening = Section(layout, "Broadening")
        self.lineshape = QComboBox(self)
        for label, name in LINESHAPES:
            self.lineshape.addItem(label, name)
        self.lineshape.currentIndexChanged.connect(self._on_lineshape)
        broadening.add("Lineshape", self.lineshape)

        # One control per parameter rather than a single relabelled "width":
        # a pseudo-Voigt is eta*Lorentzian(HWHM) + (1-eta)*Gaussian(sigma), so
        # its two widths are independent and both have to be reachable.
        #
        # Widths are logarithmic: a broadening is judged by ratio, and on a
        # linear 0.1-200 scale every value anyone uses is squeezed into the
        # first few pixels. The mixing fraction is a plain fraction, so it
        # stays linear.
        self._rows = {}
        for name, caption, value, low, high, step, decimals, log, tip in _WIDTHS:
            box = slider_row(
                broadening, caption, value, low, high, step,
                decimals=decimals, logarithmic=log,
                on_change=lambda _value: self.changed.emit(),
            )
            box.setToolTip(tip)
            setattr(self, name, box)
            self._rows[name] = broadening.row_widgets(caption)

        # Stacking, for the case overlaying is bad at: six Raman components on
        # one baseline is six curves nobody can tell apart. A fraction of the
        # tallest peak drawn rather than an intensity, so the slider's travel is
        # the height of the spectra on screen, whatever their units and
        # whatever the broadening — from one baseline at 0 to each curve lifted
        # clear of the one below at 1.
        curves = Section(layout, "Curves")
        self.offset = slider_row(
            curves, "Offset", 0.0, 0.0, 1.0, 0.05, decimals=2,
            on_change=lambda _value: self.changed.emit(),
        )
        self.offset.setToolTip(
            "Raise each curve above the one before, as a fraction of the\n"
            "tallest peak drawn. 0 lays them all on the same baseline;\n"
            "1 lifts each by that whole height, so no two overlap."
        )
        self._curves = curves

        # How the lines look, as on a band structure: the same section, with
        # the same two things in it.
        look = Section(layout, "Appearance")
        self.linewidth = slider_row(
            look, "Line width", DEFAULT_LINEWIDTH, 0.3, 4.0, 0.1, decimals=1,
            on_change=lambda _value: self.changed.emit(),
        )
        self.legend = QCheckBox("Legend")
        self.legend.setChecked(True)
        self.legend.setToolTip(
            "Name the curves on the figure. A legend sits over the plot, and\n"
            "on a crowded spectrum it can hide the peaks it names."
        )
        self.legend.toggled.connect(lambda _on: self.changed.emit())
        look.add_wide(self.legend)

        # One track with two handles, rather than a "from" and a "to" that know
        # nothing about each other: the span is the thing being chosen, and this
        # shows how much of the spectrum it covers.
        span = Section(layout, "Frequency range")
        _slider, self.fmin, self.fmax = range_row(
            span, "Range (cm⁻¹)", 0.0, 4000.0, 0.0, 5000.0, decimals=0, step=100.0,
            on_change=lambda _low, _high: self.changed.emit(),
        )
        # Shown until the host says how many curves there are, which it does
        # before anything is on screen: the dialog once its size is taken, the
        # plot through set_options. Hiding it here instead put it among what
        # the dialog measures as hidden-and-shown-again, which Qt sizes for
        # real while it leaves every untouched section at a few pixels, and the
        # dialog came out taller than the other choosers.
        self._sync_broadening()

    # ── state ───────────────────────────────────────────────────────────
    def options(self) -> dict:
        """Broadening and range, as :func:`plot_spectra` keyword arguments."""
        low, high = sorted((self.fmin.value(), self.fmax.value()))
        return {
            "lineshape": self.lineshape.currentData(),
            "hwhm": self.hwhm.value(),
            "stdev": self.stdev.value(),
            "eta": self.eta.value(),
            "offset": self.offset.value(),
            "legend": self.legend.isChecked(),
            "linewidth": self.linewidth.value(),
            # An inverted or zero-width range means "don't clip".
            "frequency_range": None if low == high else (low, high),
        }

    def set_options(self, options: Optional[dict]) -> None:
        """Show ``options`` without announcing them: the caller already has them."""
        if not options:
            return
        blocked = self.blockSignals(True)
        try:
            index = self.lineshape.findData(options.get("lineshape"))
            if index >= 0:
                self.lineshape.setCurrentIndex(index)
            for name in ("hwhm", "stdev", "eta", "offset", "linewidth"):
                if options.get(name) is not None:
                    getattr(self, name).setValue(float(options[name]))
            if options.get("legend") is not None:
                self.legend.setChecked(bool(options["legend"]))
            span = options.get("frequency_range")
            if span:
                self.fmin.setValue(float(span[0]))
                self.fmax.setValue(float(span[1]))
        finally:
            self.blockSignals(blocked)
        self._sync_broadening()
        # Beside a plot, the options it was drawn with name its curves.
        if options.get("curves") is not None:
            self.set_curve_count(len(options["curves"]))

    def set_curve_count(self, count: int) -> None:
        """Offer stacking and a legend only when there is more than one curve.

        Hidden rather than greyed, as a PES cut's Surface section is: one curve
        has nothing to be raised above and nothing to be told apart from, so
        neither applies at all rather than being unavailable. The line width
        stays: one curve still has a line.
        """
        self._curves.set_visible(count > 1)
        self.legend.setVisible(count > 1)

    def _on_lineshape(self, _index: int) -> None:
        self._sync_broadening()
        self.changed.emit()

    def _sync_broadening(self) -> None:
        """Show only the parameters the chosen lineshape actually uses.

        A Lorentzian ignores the Gaussian width, a stick spectrum ignores all of
        them, and only a pseudo-Voigt needs the mixing fraction. Greying the
        others out beats leaving controls that quietly do nothing.
        """
        used = BROADENING_PARAMETERS.get(self.lineshape.currentData(), ())
        for name, (label, widget) in self._rows.items():
            enabled = name in used
            label.setEnabled(enabled)
            widget.setEnabled(enabled)


__all__ = ["SpectraOptions"]
