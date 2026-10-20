"""Choose how to draw the CNM of a LOCMODES run, or of a series of them.

The characterisation of normal modes says, for every normal mode, how much of
it is each internal coordinate's local mode. A run has hundreds of modes and a
bar each, so there is no single figure to show: the window of modes is the
dialog's first question, as the window of states is for the VCI. The same
controls go beside the plot (:mod:`crystalline.ui.panels.cnm_options`).

A series is a different question — what becomes of one mode as the pressure
rises, say — and its dialog asks what only the user knows: which runs make up
the series, in which order, and what each stands for. CRYSTAL does not print
the pressure a geometry was optimised at, so the labels start as the part of
the file names that differs and are there to be written as numbers.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from crystalline.crystalio.cnm import DEFAULT_SERIES_AXIS, CNMRun, series_labels
from crystalline.ui.panels import cnm_options
from crystalline.ui.panels.cnm_options import CNMOptions
from crystalline.ui.widgets.wrapped_label import WrappedLabel

_OUTPUTS = "CRYSTAL output (*.out *.outp);;All files (*)"


class CNMDialog(QDialog):
    """Pick the window of modes and how they are drawn; read it back with :meth:`options`."""

    def __init__(self, run: CNMRun, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Normal modes by local modes")
        self._run = run

        layout = QVBoxLayout(self)
        layout.addWidget(WrappedLabel(f"{run.name}: {run.summary}.", self))

        # The same controls the plot carries beside it (see CNMOptions), so
        # what is set here and what is pulled there cannot drift apart.
        self.options_widget = CNMOptions(run, self)
        layout.addWidget(self.options_widget)
        # Named here as well: dialog_state remembers a dialog by walking the
        # widgets it holds.
        for name in ("fmin", "fmax", "ticks", "threshold", "legend"):
            setattr(self, name, getattr(self.options_widget, name))

        self._buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)
        # An empty or overfull window cannot be accepted.
        ok = self._buttons.button(QDialogButtonBox.Ok)
        self.options_widget.window_checked.connect(ok.setEnabled)
        ok.setEnabled(self.options_widget.acceptable())

    def fit_window(self, around: Optional[int] = None) -> None:
        """Open on the run's own window when the remembered one does not suit it.

        A window remembered from another run can catch none of this one's
        modes, or too many; and one that misses the mode selected in the
        Phonons panel misses what was most likely being asked about.
        """
        widget = self.options_widget
        low, high = widget.window()
        if widget.acceptable() and (around is None or around in self._run.modes_in(low, high)):
            return
        widget.set_window(*self._run.default_window(around))

    def options(self) -> dict:
        """The chosen settings, as :func:`plot_cnm` keyword arguments."""
        return self.options_widget.options()

    def title(self) -> str:
        return cnm_options.title(self._run, self.options())


class CNMSeriesDialog(QDialog):
    """Pick the runs of a series, their order and their labels."""

    def __init__(self, paths: Sequence[str], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("A normal mode along a series")
        layout = QVBoxLayout(self)
        layout.addWidget(WrappedLabel(
            "Follow a normal mode of the first run along the runs below, from "
            "each to the next by the overlap of its eigenvector, and show how "
            "much of it is each internal coordinate at each. Every run needs a "
            "LOCMODES section and its eigenvectors. Label each run with what it "
            "stands for, the pressure say, and put them in the order of the "
            "series.", self))

        self.runs = QTreeWidget(self)
        self.runs.setColumnCount(2)
        self.runs.setHeaderLabels(["Run", "Label"])
        self.runs.setRootIsDecorated(False)
        self.runs.setSelectionMode(QAbstractItemView.SingleSelection)
        # Only the label is written in; the run is its file. A double click
        # anywhere on a row edits its label.
        self.runs.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.runs.itemDoubleClicked.connect(lambda item, _column: self.runs.editItem(item, 1))
        header = self.runs.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setStretchLastSection(False)
        self.runs.itemChanged.connect(lambda _item, _column: self._check())
        layout.addWidget(self.runs, 1)

        buttons = QHBoxLayout()
        self._up = QPushButton("Up")
        self._down = QPushButton("Down")
        self._add = QPushButton("Add…")
        self._remove = QPushButton("Remove")
        self._up.setToolTip("Move the selected run earlier in the series.")
        self._down.setToolTip("Move the selected run later in the series.")
        self._add.setToolTip("Add outputs that are not open.")
        self._remove.setToolTip("Take the selected run out of the list.")
        self._up.clicked.connect(lambda: self._move(-1))
        self._down.clicked.connect(lambda: self._move(1))
        self._add.clicked.connect(self._choose_files)
        self._remove.clicked.connect(self._take_selected)
        for button in (self._up, self._down, self._add, self._remove):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        axis = QHBoxLayout()
        axis.addWidget(QLabel("Axis:"))
        self.axis = QLineEdit(DEFAULT_SERIES_AXIS)
        self.axis.setToolTip("What the labels are, named under the bars.")
        axis.addWidget(self.axis, 1)
        layout.addLayout(axis)

        self._problem = QLabel()
        self._problem.setEnabled(False)
        layout.addWidget(self._problem)

        self._buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._add_paths(paths)
        self.resize(560, 420)

    # ── the list ────────────────────────────────────────────────────────
    def _add_paths(self, paths: Sequence[str]) -> None:
        known = {item.data(0, Qt.UserRole) for item in self._items()}
        fresh = [p for p in paths if p not in known]
        if not fresh:
            return
        everything = [item.data(0, Qt.UserRole) for item in self._items()] + fresh
        labels = dict(zip(everything, series_labels(everything)))
        blocked = self.runs.blockSignals(True)
        try:
            for path in fresh:
                item = QTreeWidgetItem([os.path.basename(path), labels[path]])
                item.setData(0, Qt.UserRole, path)
                item.setToolTip(0, path)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
                item.setCheckState(0, Qt.Checked)
                self.runs.addTopLevelItem(item)
        finally:
            self.runs.blockSignals(blocked)
        self._check()

    def _items(self) -> List[QTreeWidgetItem]:
        return [self.runs.topLevelItem(i) for i in range(self.runs.topLevelItemCount())]

    def _move(self, step: int) -> None:
        item = self.runs.currentItem()
        if item is None:
            return
        row = self.runs.indexOfTopLevelItem(item)
        target = row + step
        if not 0 <= target < self.runs.topLevelItemCount():
            return
        blocked = self.runs.blockSignals(True)
        try:
            self.runs.takeTopLevelItem(row)
            self.runs.insertTopLevelItem(target, item)
            self.runs.setCurrentItem(item)
        finally:
            self.runs.blockSignals(blocked)
        self._check()

    def _take_selected(self) -> None:
        item = self.runs.currentItem()
        if item is not None:
            self.runs.takeTopLevelItem(self.runs.indexOfTopLevelItem(item))
            self._check()

    def _choose_files(self) -> None:
        start = ""
        items = self._items()
        if items:
            start = os.path.dirname(items[-1].data(0, Qt.UserRole))
        paths, _ = QFileDialog.getOpenFileNames(self, "Add LOCMODES outputs", start, _OUTPUTS)
        self._add_paths(paths)

    # ── state ───────────────────────────────────────────────────────────
    def chosen(self) -> List[Tuple[str, str]]:
        """``(path, label)`` of every ticked run, in the order of the series."""
        return [(item.data(0, Qt.UserRole), item.text(1).strip())
                for item in self._items() if item.checkState(0) == Qt.Checked]

    def options(self) -> dict:
        """What the series is, as :func:`plot_cnm_series` arguments (the runs by path)."""
        chosen = self.chosen()
        return {"paths": tuple(p for p, _ in chosen), "labels": tuple(l for _, l in chosen),
                "axis": self.axis.text().strip() or DEFAULT_SERIES_AXIS}

    def set_series(self, options: dict) -> None:
        """Show a series drawn before: its runs ticked, in its order, with its labels.

        A run whose file has gone since is left out.
        """
        pairs = [(p, l) for p, l in zip(options.get("paths") or (), options.get("labels") or ())
                 if os.path.isfile(p)]
        paths, labels = [p for p, _ in pairs], [l for _, l in pairs]
        self._add_paths(paths)
        blocked = self.runs.blockSignals(True)
        try:
            items = {item.data(0, Qt.UserRole): item for item in self._items()}
            rest = [item for path, item in items.items() if path not in paths]
            for item in list(items.values()):
                self.runs.takeTopLevelItem(self.runs.indexOfTopLevelItem(item))
            for path, label in zip(paths, labels):
                item = items[path]
                item.setText(1, label)
                item.setCheckState(0, Qt.Checked)
                self.runs.addTopLevelItem(item)
            for item in rest:
                item.setCheckState(0, Qt.Unchecked)
                self.runs.addTopLevelItem(item)
        finally:
            self.runs.blockSignals(blocked)
        if options.get("axis"):
            self.axis.setText(options["axis"])
        self._check()

    # dialog_state keeps what a dialog has selected through these two: here,
    # the series last drawn, so that reopening the dialog starts from it.
    def selection_state(self) -> dict:
        return self.options()

    def set_selection_state(self, state: dict) -> None:
        self.set_series(state or {})

    def problem(self) -> str:
        """Why the series cannot be drawn as it stands, or ``""``."""
        chosen = self.chosen()
        if len(chosen) < 2:
            return "Tick two runs or more."
        labels = [label for _, label in chosen]
        if any(not label for label in labels):
            return "Give every ticked run a label."
        if len(set(labels)) != len(labels):
            return "Give each ticked run a label of its own."
        return ""

    def _check(self) -> None:
        problem = self.problem()
        self._problem.setText(problem)
        self._problem.setVisible(bool(problem))
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(not problem)

    def title(self) -> str:
        return cnm_options.series_title({**self.options(), "mode": "…"})


__all__ = ["CNMDialog", "CNMSeriesDialog"]
