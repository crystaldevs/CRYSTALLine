"""How the VCI states are drawn, as a widget rather than a dialog.

The same split as the spectra and PES controls. Which states are drawn — the
wavenumber window, the symmetry block — is found by looking: a band is known by
where it sits, and the window is right when the columns that matter are in it
and the rest are not. So the window goes beside the plot with everything else,
where it can be pulled until they are, and the dialog keeps only what it is
for: saying what the run holds before there is a figure to look at.

A window can catch nothing, or more states than a figure can show. Beside a
plot that is said in the count line and the plot is left as it was, rather than
redrawn into an error at every step the slider takes on its way back.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QLabel, QSizePolicy,
                               QVBoxLayout, QWidget)

from crystalline.crystalio.vci import (
    DEFAULT_THRESHOLD,
    MAX_STATES,
    REPRESENTATIONS,
    VCIRun,
)
from crystalline.ui.panels.controls import Section, range_row, slider_row


def title(run: VCIRun, options: dict) -> str:
    """A tab title naming the flavour and the representation drawn."""
    label = dict((key, text) for text, key in REPRESENTATIONS).get(
        options.get("representation"), "")
    return f"{run.label} {label.lower()}".strip()


class VCIOptions(QWidget):
    """Representation, window and styling, with a signal when they move."""

    #: Any control was touched and the window can be drawn; the reader asks
    #: :meth:`options` for all of it.
    changed = Signal()
    #: The window was counted again: whether it can be drawn. The dialog
    #: greys its OK button on it; beside a plot the count line says why.
    window_checked = Signal(bool)

    def __init__(self, run: VCIRun, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._run = run
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # No stretch of its own, as with the other plot controls.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        states = Section(layout, "States")
        self.representation = QComboBox()
        for label, key in REPRESENTATIONS:
            self.representation.addItem(label, key)
        self.representation.currentIndexChanged.connect(self._on_representation)
        states.add("Representation", self.representation)

        # A wavenumber window rather than a count: a band one wants to look at is
        # known by where it sits in the spectrum, not by its rank.
        top = run.energies[-1] if run.energies else 4000.0
        default_min, default_max = run.default_window()
        ceiling = max(top, default_max) + 1000.0
        _slider, self.fmin, self.fmax = range_row(
            states, "Window (cm⁻¹)", default_min, default_max, 0.0, ceiling,
            decimals=0, step=50.0, on_change=lambda _a, _b: self._on_window(),
        )
        self.fmin.setToolTip("Lower edge of the window, as ENE - ZPE.")
        self.fmax.setToolTip("Upper edge of the window, as ENE - ZPE.")

        # A blocked VCI matrix numbers its states within each irrep, and states
        # of different irreps do not mix, so one block at a time is the reading
        # that makes sense — but the low-lying states across all of them is the
        # more usual first look, so that stays the default.
        self.irrep = QComboBox()
        self.irrep.addItem("All", None)
        for value in run.irreps:
            self.irrep.addItem(f"Irrep {value}", value)
        self.irrep.setEnabled(bool(run.irreps))
        self.irrep.currentIndexChanged.connect(lambda _i: self._on_window())
        states.add("Symmetry block", self.irrep)

        # How many states the window catches, live: a run spans thousands, so
        # widening it by a few hundred cm⁻¹ can quietly ask for an unreadable
        # figure. Saying so up front beats failing after the parse.
        self.count = QLabel()
        states.add_wide(self.count)

        self.threshold = slider_row(
            states, "Threshold", DEFAULT_THRESHOLD, 0.0, 1.0, 0.005, decimals=3,
            on_change=lambda _value: self._emit(),
        )
        self.threshold.setToolTip(
            "Drop contributions smaller than this.\n"
            "Raise it for a sparser figure, lower it to see weak mixing."
        )

        # One section per representation, hidden while the other is drawn: a
        # Sankey has no cells to print values in, a map no ribbons to size.
        self.map_section = Section(layout, "Coefficient map")
        self.signed = QCheckBox("Show the sign of the coefficients")
        self.signed.setToolTip(
            "Map A(n,s) on a diverging scale instead of its magnitude; the "
            "relative sign of two contributions is physically meaningful."
        )
        self.signed.toggled.connect(lambda _on: self._emit())
        self.map_section.add_wide(self.signed)
        self.annotate = QCheckBox("Print values in the cells")
        self.annotate.setToolTip("Readable only for a small number of states.")
        self.annotate.toggled.connect(lambda _on: self._emit())
        self.map_section.add_wide(self.annotate)

        self.sankey_section = Section(layout, "Sankey")
        self.weight = QComboBox()
        # |A|^2 is the population, which is what a flow width ought to mean; |A|
        # makes the weak mixing visible, which is usually the interesting part.
        self.weight.addItem("Weight |A|²", "square")
        self.weight.addItem("Amplitude |A|", "abs")
        self.weight.currentIndexChanged.connect(lambda _i: self._emit())
        self.sankey_section.add("Ribbon width", self.weight)

        self._sync_representation()
        self._check_window()

    # ── state ───────────────────────────────────────────────────────────
    def window(self) -> tuple:
        """The chosen window as ``(min, max)``, in ascending order."""
        return tuple(sorted((self.fmin.value(), self.fmax.value())))

    def acceptable(self) -> bool:
        """Whether the window catches something, and not too much, to draw."""
        low, high = self.window()
        return 0 < self._run.count_in(low, high, self.irrep.currentData()) <= MAX_STATES

    def options(self) -> dict:
        """What this widget decides, as :func:`plot_vci` keyword arguments."""
        return {
            "representation": self.representation.currentData(),
            "frange": self.window(),
            "irrep": self.irrep.currentData(),
            "threshold": self.threshold.value(),
            "signed": self.signed.isChecked(),
            "annotate": self.annotate.isChecked(),
            "weight": self.weight.currentData(),
            "modes": self._run.modes,
        }

    def set_options(self, options: Optional[dict]) -> None:
        """Show ``options`` without announcing them: the caller already has them."""
        if not options:
            return
        blocked = self.blockSignals(True)
        try:
            for name in ("representation", "irrep", "weight"):
                if name in options:
                    index = getattr(self, name).findData(options[name])
                    if index >= 0:
                        getattr(self, name).setCurrentIndex(index)
            span = options.get("frange")
            if span:
                self.fmin.setValue(float(span[0]))
                self.fmax.setValue(float(span[1]))
            if options.get("threshold") is not None:
                self.threshold.setValue(float(options["threshold"]))
            for name in ("signed", "annotate"):
                if options.get(name) is not None:
                    getattr(self, name).setChecked(bool(options[name]))
        finally:
            self.blockSignals(blocked)
        self._sync_representation()
        self._check_window()

    def _emit(self) -> None:
        """Announce a change only when what it asks for can be drawn."""
        if self.acceptable():
            self.changed.emit()

    def _on_representation(self, _index: int) -> None:
        self._sync_representation()
        self._emit()

    def _on_window(self) -> None:
        self._check_window()
        self._emit()

    def _sync_representation(self) -> None:
        """Show only the section the chosen representation uses."""
        key = self.representation.currentData()
        self.map_section.set_visible(key == "map")
        self.sankey_section.set_visible(key == "sankey")

    def _check_window(self) -> None:
        """Report what the window catches, and whether that can be drawn.

        An empty window has nothing to draw; a very wide one would build a
        figure with hundreds of columns, which is not a plot of anything.
        """
        low, high = self.window()
        count = self._run.count_in(low, high, self.irrep.currentData())
        if count == 0:
            self.count.setText("no states in this window")
        elif count > MAX_STATES:
            self.count.setText(f"{count} states — too many, narrow the window")
        else:
            self.count.setText(f"{count} state{'s' if count != 1 else ''} in this window")
        self.window_checked.emit(0 < count <= MAX_STATES)


__all__ = ["VCIOptions", "title"]
