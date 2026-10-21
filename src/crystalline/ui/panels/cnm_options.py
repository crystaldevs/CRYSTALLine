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

Every group the figure shows can be given a colour of its own, from a
dropdown beside its name (Filippo, 2026-10-20): the house colours by name, or
any other. The list is the figure's, read from it as each draw lands, since a
series' groups are known only once its runs have been matched.

A choice can hold no mode, or more bars than a figure can hold. Beside a plot
that is said in the count line and the plot is left as it was, rather than
redrawn into an error.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFormLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from crystalline.crystalio.cnm import (
    DEFAULT_MIN_OVERLAP,
    DEFAULT_THRESHOLD,
    MAX_MODES,
    READABLE_GROUPS,
    RESIDUES,
    RESIDUES_COLOUR,
    TICKS,
    CNMRun,
    drawn_groups,
    named_palette,
    series_labels,
)
from crystalline.ui.panels.controls import LABEL_COLUMN, Section, range_row, slider_row

_THRESHOLD_TIP = (
    "Contributions smaller than this, in per cent of the mode, are summed into\n"
    "the grey residues at the top of each bar. Raise it for fewer groups."
)
_LEGEND_TIP = "Name the groups of internal coordinates beside the bars."
# Rows of the mode list shown before it scrolls.
_LIST_ROWS = 10
# A title longer than this names the modes by count and span instead.
_TITLE_CHARACTERS = 24
# Rows of the colour list shown before it scrolls, and the side of a swatch.
_COLOUR_ROWS = 8
_SWATCH = 14
# The last entry of a colour dropdown, which opens the colour picker.
_OTHER = "Other…"


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


def _swatch(colour: str) -> QIcon:
    """A small square of ``colour``, edged so a pale one still shows."""
    pixmap = QPixmap(_SWATCH, _SWATCH)
    pixmap.fill(QColor(colour))
    painter = QPainter(pixmap)
    painter.setPen(QColor(0, 0, 0, 90))
    painter.drawRect(0, 0, _SWATCH - 1, _SWATCH - 1)
    painter.end()
    return QIcon(pixmap)


class _ColourBox(QComboBox):
    """A dropdown of colours: the house ones by name, the residues' grey, the
    one shown if it is none of these, and Other… for any colour at all."""

    #: A colour was chosen, as ``#rrggbb``.
    chosen = Signal(str)

    def __init__(self, colour: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._colour = QColor(colour).name()
        for name, value in named_palette() + (("Pale grey", RESIDUES_COLOUR),):
            self.addItem(_swatch(value), name, QColor(value).name())
        self.addItem(_OTHER, None)
        self._show(self._colour)
        self.activated.connect(self._on_activated)

    def colour(self) -> str:
        return self._colour

    def _show(self, colour: str) -> None:
        """Show ``colour`` as the current entry, listing it first if it is new."""
        index = self.findData(colour)
        if index < 0:
            self.insertItem(0, _swatch(colour), colour, colour)
            index = 0
        blocked = self.blockSignals(True)
        self.setCurrentIndex(index)
        self.blockSignals(blocked)

    def _on_activated(self, index: int) -> None:
        value = self.itemData(index)
        if value is None:                                   # Other…
            picked = QColorDialog.getColor(QColor(self._colour), self, "Choose colour")
            if not picked.isValid():
                self._show(self._colour)
                return
            value = picked.name()
        self._colour = value
        self._show(value)
        self.chosen.emit(value)


class GroupColours(QWidget):
    """Every group a CNM figure shows, each with a dropdown of its colour.

    Read from the figure as it lands (:meth:`show_groups`), so the list is
    what is drawn, in the legend's order; :meth:`chosen` is what the user set,
    by group, which the next draw is asked to keep.
    """

    changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._chosen: Dict[str, str] = {}
        self._shown: List[Tuple[str, str]] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.groups = QTreeWidget(self)
        self.groups.setColumnCount(2)
        self.groups.setHeaderHidden(True)
        self.groups.setRootIsDecorated(False)
        self.groups.setSelectionMode(QAbstractItemView.NoSelection)
        self.groups.setFocusPolicy(Qt.NoFocus)
        header = self.groups.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setStretchLastSection(True)
        self.groups.setToolTip("The colour of each group of internal coordinates, "
                               "kept as the plot is drawn again.")
        layout.addWidget(self.groups)

    def chosen(self) -> Dict[str, str]:
        """The colours the user chose, by group."""
        return dict(self._chosen)

    def set_chosen(self, colours: Optional[Dict[str, str]]) -> None:
        """Take ``colours`` as the user's choice, without announcing it."""
        self._chosen = dict(colours or {})
        shown, self._shown = self._shown, []
        self.show_groups([(g, self._chosen.get(g, c)) for g, c in shown])

    def show_groups(self, groups: Sequence[Tuple[str, str]]) -> None:
        """List ``(group, colour)`` as drawn; nothing is rebuilt if it is the same."""
        groups = [(g, QColor(c).name()) for g, c in groups]
        if groups == self._shown:
            return
        self._shown = groups
        self.groups.clear()
        for group, colour in groups:
            item = QTreeWidgetItem([group])
            item.setToolTip(0, "Contributions under the threshold, summed"
                            if group == RESIDUES else f"The group of internal "
                            f"coordinate {group.split()[0]} and those equivalent to it")
            self.groups.addTopLevelItem(item)
            box = _ColourBox(colour)
            box.chosen.connect(lambda value, g=group: self._on_chosen(g, value))
            self.groups.setItemWidget(item, 1, box)
        rows = max(self.groups.sizeHintForRow(0), 24) if groups else 24
        self.groups.setFixedHeight(rows * min(max(len(groups), 1), _COLOUR_ROWS) + 4)

    def box(self, group: str) -> Optional[_ColourBox]:
        """The dropdown of ``group``, or ``None`` if it is not shown."""
        for i in range(self.groups.topLevelItemCount()):
            item = self.groups.topLevelItem(i)
            if item.text(0) == group:
                return self.groups.itemWidget(item, 1)
        return None

    def _on_chosen(self, group: str, colour: str) -> None:
        self._chosen[group] = colour
        self._shown = [(g, colour if g == group else c) for g, c in self._shown]
        self.changed.emit()


def _colour_section(layout) -> Tuple[object, GroupColours]:
    """A Colours section holding a :class:`GroupColours`, hidden until a figure
    has been drawn: in a dialog, before there is one, it has nothing to list."""
    section = Section(layout, "Colours")
    colours = GroupColours()
    section.add_wide(colours)
    section.set_visible(False)
    return section, colours


class _ColouredGroups:
    """What both CNM controls do with each figure: list its groups, in their
    colours (a :class:`GroupColours` in ``self.colours``, inside
    ``self._colour_section``)."""

    def figure_drawn(self, figure) -> None:
        """The figure now beside these controls: list its groups, in their colours."""
        groups = drawn_groups(figure)
        self._colour_section.set_visible(bool(groups))
        self.colours.show_groups(groups)


def _threshold(section) -> object:
    box = slider_row(section, "Threshold (%)", DEFAULT_THRESHOLD, 0.0, 20.0, 0.5,
                     decimals=1)
    box.setToolTip(_THRESHOLD_TIP)
    return box


class CNMOptions(_ColouredGroups, QWidget):
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

        self._colour_section, self.colours = _colour_section(layout)
        self.colours.changed.connect(self._emit)

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
        self._tick_only(wanted)
        self._follow()
        self._check()
        first = next((i for i in range(self.modes.count())
                      if self.modes.item(i).checkState() == Qt.Checked), None)
        if first is not None:
            self.modes.scrollToItem(self.modes.item(first), QAbstractItemView.PositionAtTop)

    def _tick_only(self, wanted) -> None:
        """Tick the modes in ``wanted`` and untick the rest, without a signal per item."""
        blocked = self.modes.blockSignals(True)
        try:
            for i in range(self.modes.count()):
                item = self.modes.item(i)
                item.setCheckState(Qt.Checked if item.data(Qt.UserRole) in wanted
                                   else Qt.Unchecked)
        finally:
            self.modes.blockSignals(blocked)

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
            "group_colours": self.colours.chosen(),
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
            if options.get("group_colours") is not None:
                self.colours.set_chosen(options["group_colours"])
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
        self._tick_only(wanted)
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


class CNMSeriesOptions(_ColouredGroups, QWidget):
    """Which mode is followed along a series, and how its bars are drawn and labelled.

    The label under each bar is the run's, written in the set-up dialog and
    here too (Filippo, 2026-10-20: "the labels for each of the column"), each
    run on a row of its own; a label is taken when it is left, not at every
    key, and only while every run has one of its own.
    """

    changed = Signal()

    def __init__(self, first: CNMRun, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._paths: tuple = ()
        self._labels: tuple = ()
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

        runs = Section(layout, "Bar labels")
        # One field per run, in the order of the bars, rebuilt when the series
        # is (set_options): an edit through the dialog can change its runs.
        self._label_rows = QWidget()
        self._label_form = QFormLayout(self._label_rows)
        self._label_form.setContentsMargins(0, 0, 0, 0)
        # On the shared grid's columns, so the fields line up with every other
        # value in the column (controls.Section: 112 px of label, 10 between).
        self._label_form.setHorizontalSpacing(10)
        self._label_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.label_edits: List[QLineEdit] = []
        runs.add_wide(self._label_rows)
        self.label_problem = QLabel()
        self.label_problem.setEnabled(False)
        self.label_problem.hide()
        runs.add_wide(self.label_problem)
        self.frequencies = QCheckBox("Frequencies under the labels")
        self.frequencies.setChecked(True)
        self.frequencies.setToolTip(
            "The followed mode's frequency in each run, printed under its label.")
        self.frequencies.toggled.connect(lambda _on: self.changed.emit())
        runs.add_wide(self.frequencies)

        groups = Section(layout, "Internal coordinates")
        self.threshold = _threshold(groups)
        self.threshold.valueChanged.connect(lambda _v: self.changed.emit())
        self.legend = QCheckBox("Legend")
        self.legend.setChecked(True)
        self.legend.setToolTip(_LEGEND_TIP)
        self.legend.toggled.connect(lambda _on: self.changed.emit())
        groups.add_wide(self.legend)

        self._colour_section, self.colours = _colour_section(layout)
        self.colours.changed.connect(self.changed.emit)

    def options(self) -> dict:
        """What this widget decides, as :func:`plot_cnm_series` keyword arguments."""
        options = {
            "mode": self.mode.currentData(),
            "threshold": self.threshold.value(),
            "min_overlap": self.min_overlap.value(),
            "legend": self.legend.isChecked(),
            "frequencies": self.frequencies.isChecked(),
            "group_colours": self.colours.chosen(),
        }
        if self._labels:
            options["labels"] = self._labels
        return options

    def _show_runs(self, paths: Sequence[str], labels: Sequence[str]) -> None:
        """A label field for each run, named by what tells its file apart."""
        paths, labels = tuple(paths), tuple(str(l) for l in labels)
        if paths == self._paths and labels == self._labels:
            return
        self._paths, self._labels = paths, labels
        while self._label_form.rowCount():
            self._label_form.removeRow(0)
        self.label_edits = []
        for name, path, label in zip(series_labels(paths), paths, labels):
            edit = QLineEdit(label)
            edit.setToolTip(f"The label under the bar of {os.path.basename(path)}.")
            edit.editingFinished.connect(self._on_label)
            title = QLabel(name)
            title.setToolTip(path)
            title.setMinimumWidth(LABEL_COLUMN)
            self._label_form.addRow(title, edit)
            self.label_edits.append(edit)
        self.label_problem.hide()

    def _on_label(self) -> None:
        """A label was written: taken if every run still has one of its own."""
        labels = tuple(edit.text().strip() for edit in self.label_edits)
        if labels == self._labels:
            return
        problem = ""
        if any(not label for label in labels):
            problem = "Give every run a label."
        elif len(set(labels)) != len(labels):
            problem = "Give each run a label of its own."
        self.label_problem.setText(problem)
        self.label_problem.setVisible(bool(problem))
        if problem:
            return
        self._labels = labels
        self.changed.emit()


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
            if options.get("frequencies") is not None:
                self.frequencies.setChecked(bool(options["frequencies"]))
            if options.get("group_colours") is not None:
                self.colours.set_chosen(options["group_colours"])
            if options.get("paths") and options.get("labels"):
                self._show_runs(options["paths"], options["labels"])
        finally:
            self.blockSignals(blocked)


__all__ = ["CNMOptions", "CNMSeriesOptions", "GroupColours", "series_title", "spans", "title"]
