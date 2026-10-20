"""How the CNM of a LOCMODES run is drawn, as widgets rather than dialogs.

The same split as the VCI and spectra controls: what is judged by looking goes
beside the plot, where it can be changed until the figure says what it should.
For one run that is which modes, and how small a contribution is lumped into
the residues. For a series it is which mode is followed, and from which
overlap on the match is doubted.

The modes are ticked in a list of all the run has, by CRYSTAL's number with
the frequency beside it (Filippo, 2026-10-20: the modes asked about are known
by number, and are not always next to each other). A window of frequencies is
kept for ticking a span at once, as a band is known by where it sits.

A choice can hold no mode, or more bars than a figure can hold. Beside a plot
that is said in the count line and the plot is left as it was, rather than
redrawn into an error.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

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
# Rows of the mode list shown before it scrolls.
_LIST_ROWS = 10
# A title longer than this names the modes by count and span instead.
_TITLE_CHARACTERS = 24


def spans(modes: Sequence[int]) -> str:
    """``[95, 96, 97, 150]`` → ``"95–97, 150"``: runs of numbers as ranges."""
    ordered = sorted(set(int(m) for m in modes))
    parts: List[str] = []
    start = previous = None
    for mode in ordered + [None]:
        if start is not None and mode == previous + 1:
            previous = mode
            continue
        if start is not None:
            parts.append(str(start) if start == previous else f"{start}–{previous}")
        start = previous = mode
    return ", ".join(parts)


def title(run: CNMRun, options: dict) -> str:
    """A tab title naming the modes drawn: ``"CNM, modes 95–106, 150"``."""
    modes = sorted(options.get("modes") or run.default_modes())
    named = spans(modes)
    if len(named) > _TITLE_CHARACTERS:
        return f"CNM, {len(modes)} modes, {modes[0]}–{modes[-1]}"
    return f"CNM, mode{'s' if len(modes) != 1 else ''} {named}"


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
    """The modes ticked and how the bars are drawn, with a signal when they change."""

    #: Any control was touched and what it asks for can be drawn; the reader
    #: asks :meth:`options` for all of it.
    changed = Signal()
    #: The ticked modes were counted again: whether they can be drawn. The
    #: dialog greys its OK button on it; beside a plot the count line says why.
    selection_checked = Signal(bool)

    def __init__(self, run: CNMRun, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._run = run
        # Set while the window's handles are moved to follow the ticks, so that
        # moving them does not tick the window's span over what was ticked.
        self._following = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # No stretch of its own, as with the other plot controls.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        section = Section(layout, "Normal modes")
        self.modes = QListWidget()
        self.modes.setSelectionMode(QAbstractItemView.NoSelection)
        for mode in run.modes:
            item = QListWidgetItem(run.label(mode))
            item.setData(Qt.UserRole, mode)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.modes.addItem(item)
        rows = self.modes.sizeHintForRow(0) if self.modes.count() else 18
        self.modes.setMinimumHeight(max(rows, 18) * _LIST_ROWS + 4)
        self.modes.setToolTip("Tick the modes to draw, a bar each: at most "
                              f"{MAX_MODES}.")
        self.modes.itemChanged.connect(self._on_ticked)
        section.add_wide(self.modes)

        top = max(run.frequencies, default=4000.0)
        low, high = run.default_window()
        _slider, self.fmin, self.fmax = range_row(
            section, "Tick span (cm⁻¹)", low, high, 0.0, float(int(top) + 100),
            decimals=0, step=10.0, on_change=lambda _a, _b: self._on_window(),
        )
        tip = ("Ticks every mode between the two frequencies, and only those;\n"
               "single modes can then be ticked or unticked in the list.")
        self.fmin.setToolTip(tip)
        self.fmax.setToolTip(tip)
        # How many are ticked, live: a run has hundreds of modes, and a wide
        # window can tick far more bars than a figure can hold.
        self.count = QLabel()
        section.add_wide(self.count)

        self.ticks = QComboBox()
        for label, key in TICKS:
            self.ticks.addItem(label, key)
        self.ticks.setToolTip("What each bar is labelled with along the axis.")
        self.ticks.currentIndexChanged.connect(lambda _i: self._emit())
        section.add("Label bars by", self.ticks)

        groups = Section(layout, "Internal coordinates")
        self.threshold = _threshold(groups)
        self.threshold.valueChanged.connect(lambda _v: self._on_threshold())
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

        self.set_ticked(run.default_modes())

    # ── state ───────────────────────────────────────────────────────────
    def ticked(self) -> List[int]:
        """The modes ticked, by CRYSTAL's number, ascending."""
        return [self.modes.item(i).data(Qt.UserRole) for i in range(self.modes.count())
                if self.modes.item(i).checkState() == Qt.Checked]

    def set_ticked(self, modes: Sequence[int]) -> None:
        """Tick ``modes`` and only those, without announcing it.

        A number the run does not have is passed over. The window's handles
        follow, to the span of what is ticked, without ticking anything.
        """
        wanted = {int(m) for m in modes}
        blocked = self.modes.blockSignals(True)
        try:
            for i in range(self.modes.count()):
                item = self.modes.item(i)
                item.setCheckState(Qt.Checked if item.data(Qt.UserRole) in wanted
                                   else Qt.Unchecked)
        finally:
            self.modes.blockSignals(blocked)
        self._follow()
        self._check()
        first = next((i for i in range(self.modes.count())
                      if self.modes.item(i).checkState() == Qt.Checked), None)
        if first is not None:
            self.modes.scrollToItem(self.modes.item(first), QAbstractItemView.PositionAtTop)

    def acceptable(self) -> bool:
        """Whether something, and not too much, is ticked to draw."""
        return 0 < len(self.ticked()) <= MAX_MODES

    def options(self) -> dict:
        """What this widget decides, as :func:`plot_cnm` keyword arguments."""
        return {
            "modes": tuple(self.ticked()),
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
            if options.get("modes"):
                self.set_ticked(options["modes"])
            if options.get("threshold") is not None:
                self.threshold.setValue(float(options["threshold"]))
            index = self.ticks.findData(options.get("ticks"))
            if index >= 0:
                self.ticks.setCurrentIndex(index)
            if options.get("legend") is not None:
                self.legend.setChecked(bool(options["legend"]))
        finally:
            self.blockSignals(blocked)
        self._check()

    # ── changes ─────────────────────────────────────────────────────────
    def _emit(self) -> None:
        """Announce a change only when what it asks for can be drawn."""
        if self.acceptable():
            self.changed.emit()

    def _on_ticked(self, _item) -> None:
        self._follow()
        self._check()
        self._emit()

    def _on_window(self) -> None:
        """The window moved by hand: tick exactly the modes inside it."""
        if self._following:
            return
        low, high = sorted((self.fmin.value(), self.fmax.value()))
        wanted = set(self._run.modes_in(low, high))
        blocked = self.modes.blockSignals(True)
        try:
            for i in range(self.modes.count()):
                item = self.modes.item(i)
                item.setCheckState(Qt.Checked if item.data(Qt.UserRole) in wanted
                                   else Qt.Unchecked)
        finally:
            self.modes.blockSignals(blocked)
        self._check()
        self._emit()

    def _on_threshold(self) -> None:
        self._check()
        self._emit()

    def _follow(self) -> None:
        """Put the window's handles round what is ticked, ticking nothing."""
        ticked = self.ticked()
        if not ticked:
            return
        frequencies = [self._run.frequency(m) for m in ticked]
        low, high = float(int(min(frequencies) - 0.5)), float(int(max(frequencies) + 0.5) + 1)
        self._following = True
        try:
            # The end that keeps the two in order first.
            if low > self.fmax.value():
                self.fmax.setValue(high)
                self.fmin.setValue(low)
            else:
                self.fmin.setValue(low)
                self.fmax.setValue(high)
        finally:
            self._following = False

    def _check(self) -> None:
        ticked = self.ticked()
        count = len(ticked)
        if count == 0:
            self.count.setText("no modes ticked")
        elif count > MAX_MODES:
            self.count.setText(f"{count} modes ticked, too many: at most {MAX_MODES}")
        else:
            self.count.setText(f"{count} mode{'s' if count != 1 else ''} ticked: "
                               f"{spans(ticked)}")
        groups = self._run.groups_in(ticked, self.threshold.value())
        text = f"{groups} group{'s' if groups != 1 else ''} above the threshold"
        if groups > READABLE_GROUPS:
            text += ": raise it to read the legend"
        self.groups.setText(text)
        self.selection_checked.emit(0 < count <= MAX_MODES)


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


__all__ = ["CNMOptions", "CNMSeriesOptions", "series_title", "spans", "title"]
