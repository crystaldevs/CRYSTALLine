"""What the electronic and the phonon band dialogs have in common.

Both plot a band structure, a density of states, or the two side by side, from
files a run left beside its output, and both are laid out the same way: what to
plot on the left — the files, the window, the path's labels — and how it looks
on the right — the projections of the DOS and the appearance. Two columns,
because one column of five sections came out twice as tall as it was wide.

The files are listed by the subclass, which knows how to tell its own from the
others and which of them belong to the open output, and are read for what the
dialog needs: the path's corners, the projections, the span of the data. The
window is not remembered from last time: it belongs to the data, and is fitted
to the files each time — the track spans everything in them, the handles start
on the suggested window — while the units and every colour are remembered.

A subclass builds its own window section and appearance rows, reads its own
files, and answers :meth:`request` with its own options.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from crystalline.crystalio import electronic as el
from crystalline.ui.panels.controls import (ColourButton, Section, range_row, set_span,
                                            slider_row)
from crystalline.ui.safety import guard
from crystalline.ui.widgets.wrapped_label import WrappedLabel


class BandsDialog(QDialog):
    """Pick a band file and a DOS file, the window, the labels and the look."""

    TITLE = ""
    #: ``(kind, caption, placeholder)`` for the band and the DOS file rows.
    KINDS = (("band", "Band file", ""), ("dos", "DOS file", ""))
    #: File-dialog captions and filters, by kind. By extension only: names
    #: are rewritten, and a file is known by its contents.
    BROWSE = {"band": ("", ""), "dos": ("", "")}
    K_PLACEHOLDER = ""
    K_TOOLTIP = ""
    DOS_DECIMALS, DOS_STEP = 2, 0.1

    # Where files were last found in this session — the fallback when the
    # loaded output's folder has none. One per subclass.
    last_folder = ""

    def __init__(self, structure=None, folder: str = "",
                 parent: Optional[QWidget] = None) -> None:
        """Set up the state; the subclass sets its own, then calls :meth:`_build`."""
        super().__init__(parent)
        self.setWindowTitle(self.TITLE)
        self.resize(960, 600)
        self._structure = structure
        self._folder = folder
        self._bands_info = None
        self._dos_info = None
        self._auto_names: List[str] = []
        self._errors: dict = {}
        self._loaded: dict = {}
        self._projection_colours: List[ColourButton] = []
        # A plot's own window and DOS range, when the dialog is opened to edit
        # one: put on the tracks once the first show has fitted them to the
        # files, which would otherwise move the handles back to where they
        # start for a new plot (see set_request).
        self._held = None

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        columns = QHBoxLayout()
        left, right = QVBoxLayout(), QVBoxLayout()
        columns.addLayout(left, 1)
        columns.addSpacing(24)
        columns.addLayout(right, 1)
        outer.addLayout(columns, 1)

        self._build_data(left)
        self._build_window(left)
        self._build_bands(left)
        left.addStretch(1)
        self._build_dos(right)
        self._build_appearance(right)
        right.addStretch(1)

        self._buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        outer.addWidget(self._buttons)

        last = type(self).last_folder
        if not self._populate(self._folder) and last not in ("", self._folder):
            self._populate(last)
        self._sync()

    # ── what a subclass provides ────────────────────────────────────────
    def _build_window(self, column) -> None:
        """The section with the unit and the window (``self._window``)."""
        raise NotImplementedError

    def _build_lines(self, look) -> None:
        """The appearance rows after the band colour's."""
        raise NotImplementedError

    def _find_files(self, folder: str):
        """``(band paths, DOS paths, {kind: chosen path})``, best first."""
        raise NotImplementedError

    def _describe_file(self, kind: str, path: str):
        raise NotImplementedError

    def _auto_labels(self, info) -> List[str]:
        raise NotImplementedError

    def _fit_window(self) -> None:
        raise NotImplementedError

    def _fit_dos(self) -> None:
        raise NotImplementedError

    def _describe(self, need_bands: bool, need_dos: bool) -> str:
        raise NotImplementedError

    def _polarised(self):
        """``(bands, DOS)`` spin-polarised, for whether a legend names anything."""
        return False, False

    def _sync_lines(self, need_bands: bool, need_dos: bool) -> None:
        """Keep the subclass's own rows honest."""

    # ── building ────────────────────────────────────────────────────────
    def _build_data(self, column) -> None:
        data = Section(column, "Data")
        self.mode = QComboBox(self)
        for key, label in el.MODES:
            self.mode.addItem(label, key)
        self.mode.currentIndexChanged.connect(self._on_mode_changed)
        data.add("Show", self.mode)

        self._files: dict = {}
        self._file_rows: dict = {}
        for kind, caption, hint in self.KINDS:
            combo = QComboBox(self)
            combo.setEditable(True)
            combo.setInsertPolicy(QComboBox.NoInsert)
            combo.setMinimumContentsLength(16)
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.lineEdit().setPlaceholderText(hint)
            combo.setToolTip("Every file of this kind in the output's folder, best first. "
                             "Type a path, or Browse for one elsewhere.")
            combo.currentIndexChanged.connect(lambda _i, k=kind: self._load(k))
            combo.lineEdit().editingFinished.connect(lambda k=kind: self._load(k))
            browse = QPushButton("Browse…", self)
            browse.clicked.connect(lambda _checked=False, k=kind: self._browse(k))
            row = QWidget(self)
            line = QHBoxLayout(row)
            line.setContentsMargins(0, 0, 0, 0)
            line.addWidget(combo, 1)
            line.addWidget(browse)
            self._files[kind] = combo
            self._file_rows[kind] = row
            data.add(caption, row)

        self._summary = WrappedLabel()
        self._summary.setStyleSheet("color: palette(mid);")
        data.add_wide(self._summary)

    def _build_bands(self, column) -> None:
        bands = Section(column, "Band structure")
        self._k_edit = QLineEdit(self)
        self._k_edit.setPlaceholderText(self.K_PLACEHOLDER)
        self._k_edit.setToolTip(self.K_TOOLTIP)
        self._k_edit.textChanged.connect(lambda _text: self._sync())
        bands.add("Path labels", self._k_edit)
        self._bands_section = bands

    def _build_dos(self, column) -> None:
        dos = Section(column, "Density of states")
        self._table = QTableWidget(0, 2, self)
        self._table.setHorizontalHeaderLabels(["Projection", "Colour"])
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._table.setMinimumHeight(150)
        # Ticking a projection can be what makes a legend worth having.
        self._table.itemChanged.connect(lambda _item: self._sync())
        self._table.setToolTip("Tick the projections to draw. Double-click a name "
                               "to change what the legend calls it.")
        dos.add("Projections", self._table)
        self.overlay = QCheckBox("Draw the projections on one axis", self)
        self.overlay.setChecked(True)
        self.overlay.toggled.connect(lambda _on: self._sync())
        self.overlay.setToolTip(
            "On one axis with a legend, or one panel each. The band+DOS view "
            "always draws them together.")
        dos.add_wide(self.overlay)
        self._build_dos_extras(dos)
        self.dos_auto = QCheckBox("Fit the DOS axis to what is drawn", self)
        self.dos_auto.setChecked(True)
        self.dos_auto.toggled.connect(lambda _on: self._sync())
        dos.add_wide(self.dos_auto)
        self._dos_span = range_row(dos, "DOS range", 0.0, 1.0, 0.0, 10.0,
                                   decimals=self.DOS_DECIMALS, step=self.DOS_STEP)
        self._dos_range_row = dos.row_widgets("DOS range")
        self._dos_section = dos

    def _build_dos_extras(self, dos) -> None:
        """Rows of the DOS section between the overlay and the axis range."""

    def _build_appearance(self, column) -> None:
        # Folded, as beside the plot: set once, if at all.
        look = Section(column, "Appearance", collapsed=True)
        self.band_colour = ColourButton(el.BAND_COLOUR, "Band colour", self)
        look.add("Bands", row(self.band_colour))
        self._build_lines(look)
        self.linewidth = slider_row(look, "Line width", 1.0, 0.3, 4.0, 0.1, decimals=1)
        self._title = QLineEdit(self)
        self._title.setPlaceholderText("No title")
        look.add("Title", self._title)
        self.legend = QCheckBox("Legend", self)
        self.legend.setChecked(True)
        look.add_wide(self.legend)

    def _reference_line(self, look, label: str, colour: str, width: float):
        """A row for a line drawn across the plot at a reference value — the
        Fermi level, zero frequency: ``(draw, colour, style, width)``."""
        show = QCheckBox("Draw", self)
        show.setChecked(True)
        show.toggled.connect(lambda _on: self._sync())
        swatch = ColourButton(colour, f"{label} colour", self)
        style = style_box(self, "-")
        spin = QDoubleSpinBox(self)
        spin.setRange(0.3, 5.0)
        spin.setSingleStep(0.1)
        spin.setDecimals(1)
        spin.setValue(width)
        spin.setToolTip(f"Line width of the {label.lower()} line")
        look.add(label, row(show, swatch, style, spin))
        return show, swatch, style, spin

    # ── files ───────────────────────────────────────────────────────────
    def _populate(self, folder: str) -> bool:
        """List the band and DOS files in ``folder``, and load the chosen ones."""
        if not folder:
            return False
        bands, doss, chosen = self._find_files(folder)
        for kind, paths in (("band", bands), ("dos", doss)):
            combo = self._files[kind]
            combo.blockSignals(True)
            combo.clear()
            for path in paths:
                combo.addItem(Path(path).name, path)
                combo.setItemData(combo.count() - 1, path, Qt.ToolTipRole)
            combo.setCurrentIndex(paths.index(chosen[kind]) if chosen.get(kind) in paths
                                  else (0 if paths else -1))
            combo.blockSignals(False)
            if paths:
                self._load(kind)
        return bool(bands or doss)

    def _path(self, kind: str) -> str:
        combo = self._files[kind]
        text = combo.currentText().strip()
        if not text:
            return ""
        index = combo.findText(text)
        if index >= 0 and combo.itemData(index):
            return str(combo.itemData(index))
        return os.path.expanduser(text)

    @guard()
    def _browse(self, kind: str) -> None:
        current = self._path(kind)
        start = ((str(Path(current).parent) if current else "") or self._folder
                 or type(self).last_folder)
        caption, file_filter = self.BROWSE[kind]
        path, _filter = QFileDialog.getOpenFileName(self, caption, start, file_filter)
        if path:
            self._choose(kind, path)

    def _choose(self, kind: str, path: str) -> None:
        """Make ``path`` the chosen file of ``kind``, listed if it was not."""
        combo = self._files[kind]
        index = combo.findData(path)
        if index < 0:
            combo.addItem(Path(path).name, path)
            index = combo.count() - 1
            combo.setItemData(index, path, Qt.ToolTipRole)
        combo.setCurrentIndex(index)
        self._load(kind)

    @guard()
    def _load(self, kind: str) -> None:
        """Read a chosen file for what the dialog needs: ticks, projections, spans."""
        # A long name reads from its start, not cut off at the front.
        self._files[kind].lineEdit().setCursorPosition(0)
        path = self._path(kind)
        if self._loaded.get(kind) == path:
            return
        self._loaded[kind] = path
        info, error = None, ""
        if path:
            try:
                info = self._describe_file(kind, path)
            except Exception as exc:  # noqa: BLE001 - said in the dialog, not raised
                error = f"Could not read {Path(path).name}: {exc}"
            else:
                type(self).last_folder = str(Path(path).parent)
        self._errors[kind] = error
        if kind == "band":
            self._bands_info = info
            self._auto_names = self._auto_labels(info) if info else []
            self._k_edit.setText(" ".join(self._auto_names))
        else:
            self._dos_info = info
            self._fill_projections(info)
            self._fit_dos()
        self._fit_window()
        self._sync()

    def _fill_projections(self, info) -> None:
        self._table.setRowCount(0)
        self._projection_colours = []
        count = info.n_projections if info else 0
        self._table.setRowCount(count)
        for index in range(count):
            item = QTableWidgetItem(f"Projection {index + 1}")
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable
                          | Qt.ItemIsEditable | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self._table.setItem(index, 0, item)
            swatch = ColourButton(el.PROJECTION_COLOURS[index % len(el.PROJECTION_COLOURS)],
                                  f"Colour for projection {index + 1}", self)
            self._table.setCellWidget(index, 1, centred(swatch))
            self._projection_colours.append(swatch)

    # ── the window ──────────────────────────────────────────────────────
    def _on_mode_changed(self, _index: int = 0) -> None:
        self._fit_window()
        self._sync()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt's name
        # Remembered settings were put back after construction; fit the window
        # in whatever units they left.
        if not getattr(self, "_shown_once", False):
            self._shown_once = True
            self._fit_window()
            self._fit_dos()
            self._apply_held()
        super().showEvent(event)

    def _apply_held(self) -> None:
        """Put a plot's own window and DOS range on the freshly fitted tracks.

        Widened to take them if need be: a window dragged past where the
        files' span was rounded to is still the window the plot is drawn in.
        """
        if self._held is None:
            return
        window, dos = self._held
        self._held = None
        for triple, values in ((self._window, window), (self._dos_span, dos)):
            if values is None:
                continue
            low, high = sorted(values)
            bottom, top = triple[0].bounds()
            set_span(triple, min(bottom, low), max(top, high), low, high)

    def window_values(self):
        _slider, low, high = self._window
        return tuple(sorted((low.value(), high.value())))

    def window_bounds(self):
        return tuple(self._window[0].bounds())

    # ── keeping the controls honest ─────────────────────────────────────
    def _needs(self):
        mode = self.mode.currentData()
        return mode in (el.BANDS, el.BANDS_AND_DOS), mode in (el.DOS, el.BANDS_AND_DOS)

    def _ticked(self) -> List[int]:
        return [index for index in range(self._table.rowCount())
                if self._table.item(index, 0) is not None
                and self._table.item(index, 0).checkState() == Qt.Checked]

    def _sync(self) -> None:
        """Offer only what the chosen plot uses, and allow OK only when it can work.

        A control that would quietly do nothing is greyed rather than left
        live: the band settings for a DOS plot, the overlay for the combined
        view (which always overlays). The legend is hidden rather than greyed,
        as beside the plot: with nothing to tell apart there is none to draw.
        """
        need_bands, need_dos = self._needs()
        self._file_rows["band"].setEnabled(need_bands)
        self._file_rows["dos"].setEnabled(need_dos)
        self._bands_section.set_enabled(need_bands)
        self._dos_section.set_enabled(need_dos)
        self.overlay.setEnabled(self.mode.currentData() == el.DOS)
        for widget in self._dos_range_row:
            widget.setEnabled(need_dos and not self.dos_auto.isChecked())
        self.band_colour.setEnabled(need_bands)
        self._sync_lines(need_bands, need_dos)
        bands_polarised, dos_polarised = self._polarised()
        self.legend.setVisible(el.legend_names_anything(
            self.mode.currentData(), len(self._ticked()),
            need_bands and bands_polarised, need_dos and dos_polarised,
            self.overlay.isChecked()))

        ready = ((not need_bands or self._bands_info is not None)
                 and (not need_dos or self._dos_info is not None))
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(ready)
        self._summary.setText(self._describe(need_bands, need_dos))

    def _band_lines(self, describe_info: str) -> List[str]:
        """The summary's band line, the labels' warning, or why there is none."""
        info = self._bands_info
        if info is None:
            return [self._errors.get("band") or "No band file found here — choose one."]
        lines = [describe_info]
        typed = self._typed_labels()
        if typed and len(typed) != len(info.ticks):
            lines.append(f"Path labels: {len(info.ticks)} corners, {len(typed)} "
                         "labels typed — the automatic ones will be used.")
        return lines

    def _dos_lines(self, describe_info: str) -> List[str]:
        if self._dos_info is None:
            return [self._errors.get("dos") or "No DOS file found here — choose one."]
        return [describe_info]

    # ── the answer ──────────────────────────────────────────────────────
    def _typed_labels(self) -> List[str]:
        return [token for token in re.split(r"[\s,]+", self._k_edit.text().strip()) if token]

    def k_labels(self) -> Optional[List[str]]:
        """One name per corner, typed or automatic — never the wrong count.

        CRYSTALClear marks the corners with their k-distances when given no
        names, and refuses a list of the wrong length. So a typed list that
        does not fit falls back to the automatic names, which always do.
        """
        info = self._bands_info
        if info is None:
            return None
        typed = self._typed_labels()
        if typed and len(typed) == len(info.ticks):
            return [greek(token) for token in typed]
        return list(self._auto_names) or None

    def _projection_choice(self) -> dict:
        """The ticked projections, their names and colours, as option fields."""
        rows = self._ticked()
        return {
            "projections": [index + 1 for index in rows] or None,
            "projection_labels": [self._table.item(index, 0).text().strip()
                                  or f"Projection {index + 1}" for index in rows] or None,
            "projection_colours": [self._projection_colours[index].colour()
                                   for index in rows] or None,
        }

    def _dos_range(self):
        _slider, low, high = self._dos_span
        low, high = sorted((low.value(), high.value()))
        return None if self.dos_auto.isChecked() or low == high else (low, high)

    def _extent(self) -> dict:
        """The DOS track, carried with a plot's options for the controls."""
        _slider, dlow, dhigh = self._dos_span
        return {
            "dos_bounds": tuple(self._dos_span[0].bounds()),
            "dos_handles": tuple(sorted((dlow.value(), dhigh.value()))),
            "dos_projections": self._dos_info.n_projections if self._dos_info else 0,
        }

    def _restore(self, bands_path: Optional[str], dos_path: Optional[str], options,
                 window, boxes=()) -> None:
        """Show a plot's own choices, so that editing it starts from them.

        Everything :meth:`request` hands out comes back: the files, which may
        have been browsed for anywhere; the projections ticked, with the names
        and colours they were given; the path labels and the title, typed by
        hand. The files go first, since reading them is what lays out the
        projection table and the path labels. The window and the DOS range are
        held until the dialog is shown (see :meth:`_apply_held`). ``boxes`` are
        the subclass's combo boxes, as ``(box, value)``: set, not chosen, so
        that ``activated`` does not carry the held window across a change of
        units it is already in.
        """
        index = self.mode.findData(options.mode)
        if index >= 0:
            self.mode.setCurrentIndex(index)
        for kind, path in (("band", bands_path), ("dos", dos_path)):
            if path:
                self._choose(kind, path)
        for box, value in boxes:
            found = box.findData(value)
            if found >= 0:
                box.setCurrentIndex(found)
        if options.k_labels:
            self._k_edit.setText(" ".join(options.k_labels))
        if options.projections is not None:
            chosen = list(options.projections)
            names = list(options.projection_labels or [])
            colours = list(options.projection_colours or [])
            for index in range(self._table.rowCount()):
                item = self._table.item(index, 0)
                ticked = index + 1 in chosen
                item.setCheckState(Qt.Checked if ticked else Qt.Unchecked)
                if ticked:
                    place = chosen.index(index + 1)
                    if place < len(names):
                        item.setText(names[place])
                    if place < len(colours):
                        self._projection_colours[index].setColour(colours[place])
        self.overlay.setChecked(options.overlay_projections)
        self.dos_auto.setChecked(options.dos_range is None)
        self.band_colour.setColour(options.band_colour)
        self.linewidth.setValue(options.linewidth)
        self.legend.setChecked(options.legend)
        self._title.setText(options.title or "")
        self._held = (window, options.dos_range)


# ── small pieces ──────────────────────────────────────────────────────────
def nice(value: float) -> float:
    """Round up to two significant figures: 1.63 -> 1.7, 73.8 -> 74."""
    import math

    if value <= 0:
        return 1.0
    step = 10.0 ** (math.floor(math.log10(value)) - 1)
    return math.ceil(value / step - 1e-9) * step


def style_box(parent, default: str) -> QComboBox:
    box = QComboBox(parent)
    for key, label in el.LINE_STYLES:
        box.addItem(label, key)
    box.setCurrentIndex(max(0, box.findData(default)))
    return box


def row(*widgets) -> QWidget:
    """Controls side by side at the left of a row, at their own sizes."""
    holder = QWidget()
    line = QHBoxLayout(holder)
    line.setContentsMargins(0, 0, 0, 0)
    line.setSpacing(8)
    for widget in widgets:
        line.addWidget(widget, 0)
    line.addStretch(1)
    return holder


def centred(widget) -> QWidget:
    holder = QWidget()
    line = QHBoxLayout(holder)
    line.setContentsMargins(4, 0, 4, 0)
    line.addStretch(1)
    line.addWidget(widget)
    line.addStretch(1)
    return holder


def greek(token: str) -> str:
    """``G`` / ``Gamma`` -> ``Γ``: CRYSTAL writes Γ as G, and so do people."""
    return "Γ" if token.lower() in ("g", "gamma") else token


__all__ = ["BandsDialog", "centred", "greek", "nice", "row", "style_box"]
