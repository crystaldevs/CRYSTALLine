"""How the CNM of a LOCMODES run is drawn, as widgets rather than dialogs.

The same split as the VCI and spectra controls: what is judged by looking goes
beside the plot, where it can be moved until the figure says what it should.
For one run that is which modes — a window in cm⁻¹, as for the VCI states,
since a band is known by where it sits and not by its rank — and how small a
contribution is lumped into the residues. For a series it is which mode is
followed, and from which overlap on the match is doubted.

A window can catch no mode, or more bars than a figure can hold. Beside a plot
that is said in the count line and the plot is left as it was, rather than
redrawn into an error at every step the slider takes on its way back.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QLabel, QSizePolicy, QVBoxLayout, QWidget

from crystalline.crystalio.cnm import (
    DEFAULT_MIN_OVERLAP,
    DEFAULT_THRESHOLD,
    MAX_MODES,
    READABLE_GROUPS,
    TICKS,
    CNMRun,
)
from crystalline.ui.panels.controls import Section, range_row, slider_row

_THRESHOLD_TIP = (
    "Contributions smaller than this, in per cent of the mode, are summed into\n"
    "the grey residues at the top of each bar. Raise it for fewer groups."
)
_LEGEND_TIP = "Name the groups of internal coordinates beside the bars."


def title(run: CNMRun, options: dict) -> str:
    """A tab title naming the window drawn: ``"CNM 1644–3364 cm⁻¹"``."""
    low, high = sorted(options.get("frange") or run.default_window())
    return f"CNM {low:.0f}–{high:.0f} cm⁻¹"


def series_title(options: dict) -> str:
    """A tab title naming the mode followed: ``"CNM of mode 100 along 6 runs"``."""
    runs = len(options.get("paths") or ())
    return f"CNM of mode {options.get('mode')} along {runs} runs"


def _threshold(section) -> object:
    box = slider_row(section, "Threshold (%)", DEFAULT_THRESHOLD, 0.0, 20.0, 0.5,
                     decimals=1)
    box.setToolTip(_THRESHOLD_TIP)
    return box


class CNMOptions(QWidget):
    """The window of modes and how the bars are drawn, with a signal when they move."""

    #: Any control was touched and the window can be drawn; the reader asks
    #: :meth:`options` for all of it.
    changed = Signal()
    #: The window was counted again: whether it can be drawn. The dialog greys
    #: its OK button on it; beside a plot the count line says why.
    window_checked = Signal(bool)

    def __init__(self, run: CNMRun, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._run = run
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # No stretch of its own, as with the other plot controls.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        modes = Section(layout, "Normal modes")
        top = max(run.frequencies, default=4000.0)
        low, high = run.default_window()
        _slider, self.fmin, self.fmax = range_row(
            modes, "Window (cm⁻¹)", low, high, 0.0, float(int(top) + 100),
            decimals=0, step=10.0, on_change=lambda _a, _b: self._on_window(),
        )
        self.fmin.setToolTip("Lower edge of the window: the lowest mode drawn.")
        self.fmax.setToolTip("Upper edge of the window: the highest mode drawn.")
        # How many bars the window catches, live: a run has hundreds of modes,
        # and widening the window by a few hundred cm⁻¹ can ask for far more
        # bars than a figure can hold.
        self.count = QLabel()
        modes.add_wide(self.count)

        self.ticks = QComboBox()
        for label, key in TICKS:
            self.ticks.addItem(label, key)
        self.ticks.setToolTip("What each bar is labelled with along the axis.")
        self.ticks.currentIndexChanged.connect(lambda _i: self._emit())
        modes.add("Label bars by", self.ticks)

        groups = Section(layout, "Internal coordinates")
        self.threshold = _threshold(groups)
        self.threshold.valueChanged.connect(lambda _v: self._on_window())
        # How many groups the bars show above the threshold: in the low modes of
        # a molecular crystal a 1 % threshold leaves a hundred, and the legend
        # becomes a wall of names. Said where the threshold is set.
        self.groups = QLabel()
        groups.add_wide(self.groups)
        self.legend = QCheckBox("Legend")
        self.legend.setChecked(True)
        self.legend.setToolTip(_LEGEND_TIP)
        self.legend.toggled.connect(lambda _on: self._emit())
        groups.add_wide(self.legend)

        self._check_window()

    # ── state ───────────────────────────────────────────────────────────
    def window(self) -> tuple:
        """The chosen window as ``(min, max)``, in ascending order."""
        return tuple(sorted((self.fmin.value(), self.fmax.value())))

    def set_window(self, low: float, high: float) -> None:
        """Move the window without announcing it."""
        blocked = self.blockSignals(True)
        try:
            self.fmin.setValue(float(low))
            self.fmax.setValue(float(high))
        finally:
            self.blockSignals(blocked)
        self._check_window()

    def acceptable(self) -> bool:
        """Whether the window catches something, and not too much, to draw."""
        return 0 < self._run.count_in(*self.window()) <= MAX_MODES

    def options(self) -> dict:
        """What this widget decides, as :func:`plot_cnm` keyword arguments."""
        return {
            "frange": self.window(),
            "threshold": self.threshold.value(),
            "ticks": self.ticks.currentData(),
            "legend": self.legend.isChecked(),
        }

    def set_options(self, options: Optional[dict]) -> None:
        """Show ``options`` without announcing them: the caller already has them."""
        if not options:
            return
        blocked = self.blockSignals(True)
        try:
            span = options.get("frange")
            if span:
                self.fmin.setValue(float(span[0]))
                self.fmax.setValue(float(span[1]))
            if options.get("threshold") is not None:
                self.threshold.setValue(float(options["threshold"]))
            index = self.ticks.findData(options.get("ticks"))
            if index >= 0:
                self.ticks.setCurrentIndex(index)
            if options.get("legend") is not None:
                self.legend.setChecked(bool(options["legend"]))
        finally:
            self.blockSignals(blocked)
        self._check_window()

    def _emit(self) -> None:
        """Announce a change only when what it asks for can be drawn."""
        if self.acceptable():
            self.changed.emit()

    def _on_window(self) -> None:
        self._check_window()
        self._emit()

    def _check_window(self) -> None:
        count = self._run.count_in(*self.window())
        if count == 0:
            self.count.setText("no modes in this window")
        elif count > MAX_MODES:
            self.count.setText(f"{count} modes, too many: narrow the window")
        else:
            self.count.setText(f"{count} mode{'s' if count != 1 else ''} in this window")
        groups = self._run.groups_in(*self.window(), self.threshold.value())
        text = f"{groups} group{'s' if groups != 1 else ''} above the threshold"
        if groups > READABLE_GROUPS:
            text += ": raise it to read the legend"
        self.groups.setText(text)
        self.window_checked.emit(0 < count <= MAX_MODES)


class CNMSeriesOptions(QWidget):
    """Which mode is followed along a series, and how its bars are drawn."""

    changed = Signal()

    def __init__(self, first: CNMRun, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        mode = Section(layout, "Normal mode")
        # Numbered as in the first run: the series is followed from there, and
        # in the others the same mode can have another number.
        self.mode = QComboBox()
        for number in first.modes:
            self.mode.addItem(first.label(number), number)
        self.mode.setToolTip(
            f"The mode of the first run, {first.name}, to follow along the series."
        )
        self.mode.currentIndexChanged.connect(lambda _i: self.changed.emit())
        mode.add("Mode", self.mode)
        self.min_overlap = slider_row(mode, "Least overlap", DEFAULT_MIN_OVERLAP,
                                      0.0, 1.0, 0.05, decimals=2)
        self.min_overlap.setToolTip(
            "The mode is followed from each run to the next by the overlap of its\n"
            "eigenvector. Once a step falls below this, near a crossing, the match\n"
            "is no longer sure: that bar and the ones after it are hatched, with\n"
            "the lowest overlap so far over them."
        )
        self.min_overlap.valueChanged.connect(lambda _v: self.changed.emit())

        groups = Section(layout, "Internal coordinates")
        self.threshold = _threshold(groups)
        self.threshold.valueChanged.connect(lambda _v: self.changed.emit())
        self.legend = QCheckBox("Legend")
        self.legend.setChecked(True)
        self.legend.setToolTip(_LEGEND_TIP)
        self.legend.toggled.connect(lambda _on: self.changed.emit())
        groups.add_wide(self.legend)

    def options(self) -> dict:
        """What this widget decides, as :func:`plot_cnm_series` keyword arguments."""
        return {
            "mode": self.mode.currentData(),
            "threshold": self.threshold.value(),
            "min_overlap": self.min_overlap.value(),
            "legend": self.legend.isChecked(),
        }

    def set_options(self, options: Optional[dict]) -> None:
        if not options:
            return
        blocked = self.blockSignals(True)
        try:
            index = self.mode.findData(options.get("mode"))
            if index >= 0:
                self.mode.setCurrentIndex(index)
            for name in ("threshold", "min_overlap"):
                if options.get(name) is not None:
                    getattr(self, name).setValue(float(options[name]))
            if options.get("legend") is not None:
                self.legend.setChecked(bool(options["legend"]))
        finally:
            self.blockSignals(blocked)


__all__ = ["CNMOptions", "CNMSeriesOptions", "series_title", "title"]
