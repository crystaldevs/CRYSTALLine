"""Choose how to plot an electronic band structure, a DOS, or both.

The Plot menu used to carry the two as one-click entries drawn at CRYSTALClear's
defaults: energies always relative, the path's corners marked with k-distances,
and no way to put the bands beside the DOS. This dialog is the same move the
vibrational spectra made — one entry, every option in view — for the one-
electron properties.

Two columns, what to plot on the left and how it looks on the right, because
one column of five sections came out twice as tall as it was wide.

The band and DOS files come from a PROPERTIES (``.d3``) run, which is written
beside the SCF output. The dialog lists every band and DOS file it finds there
— recognised by content, see :func:`~crystalline.crystalio.electronic.find_files`
— with the likeliest one already chosen.

The energy window is not remembered from last time: it belongs to the data, and
a window that suited MgO's valence band shows nothing useful of urea's. It is
fitted to the files each time — the track spans everything in them, the handles
start on the suggested window — while the unit, the reference and every colour
are remembered.
"""

from __future__ import annotations

import math
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
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from crystalline.crystalio import electronic as el
from crystalline.ui.panels.controls import ColourButton, Section, range_row, slider_row
from crystalline.ui.safety import guard

# By extension only: BAND.DAT and fort.25 are names, and names are rewritten.
_BAND_FILTER = "Band structures (*.BAND *.DAT *.f25 *.25);;All files (*)"
_DOS_FILTER = "Densities of states (*.DOSS *.DAT *.f25 *.25);;All files (*)"
_HARTREE_EV = 27.211386245988
_KINDS = (("band", "Band file", "BAND.DAT, .BAND or fort.25"),
          ("dos", "DOS file", "DOSS.DAT, .DOSS or fort.25"))


class ElectronicDialog(QDialog):
    """Pick the files and the options; :meth:`request` hands them to the plotter."""

    # Where band or DOS files were last found in this session — the fallback
    # when the loaded output's folder has none.
    last_folder = ""

    def __init__(self, structure=None, folder: str = "", stem: str = "",
                 parent: Optional[QWidget] = None, efermi: Optional[float] = None) -> None:
        """``efermi`` is the open output's Fermi level (eV): files recording the
        same one are that calculation's, and are offered first."""
        super().__init__(parent)
        self._output_efermi = efermi
        self.setWindowTitle("Electronic bands & DOS")
        self.resize(960, 600)
        self._structure = structure
        self._folder = folder
        self._stem = stem
        self._bands_info: Optional[el.BandsInfo] = None
        self._dos_info: Optional[el.DosInfo] = None
        self._auto_names: List[str] = []
        self._errors: dict = {}
        self._loaded: dict = {}
        self._projection_colours: List[ColourButton] = []
        # The unit and reference the range boxes are written in, so a change
        # of either can carry the numbers across instead of reinterpreting them.
        self._frame_shown = ("eV", el.RELATIVE)

        outer = QVBoxLayout(self)
        intro = QLabel(
            "Band and DOS files from a PROPERTIES run. CRYSTAL writes their "
            "energies relative to the Fermi level; choose absolute energies to "
            "move everything, the Fermi line included, up by E_F.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: palette(mid);")
        outer.addWidget(intro)

        columns = QHBoxLayout()
        left, right = QVBoxLayout(), QVBoxLayout()
        columns.addLayout(left, 1)
        columns.addSpacing(24)
        columns.addLayout(right, 1)
        outer.addLayout(columns, 1)

        self._build_data(left)
        self._build_energy(left)
        self._build_bands(left)
        left.addStretch(1)
        self._build_dos(right)
        self._build_appearance(right)
        right.addStretch(1)

        self._buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        outer.addWidget(self._buttons)

        if not self._populate(folder) and ElectronicDialog.last_folder not in ("", folder):
            self._populate(ElectronicDialog.last_folder)
        self._sync()

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
        for kind, caption, hint in _KINDS:
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

        self._summary = QLabel()
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet("color: palette(mid);")
        data.add_wide(self._summary)

    def _build_energy(self, column) -> None:
        # Unit and reference are restored before the dialog is shown, and the
        # window is fitted when it is — in whichever frame they left it in.
        # They carry the window across only when a *person* picks a new one
        # (``activated``), never when a remembered value is put back.
        energy = Section(column, "Energy")
        self.unit = QComboBox(self)
        for key, label in el.UNITS:
            self.unit.addItem(label, key)
        self.unit.activated.connect(self._on_frame_chosen)
        energy.add("Unit", self.unit)
        self.reference = QComboBox(self)
        for key, label in el.REFERENCES:
            self.reference.addItem(label, key)
        self.reference.setToolTip(
            "CRYSTAL writes band and DOS energies relative to the Fermi level. "
            "Absolute energies add E_F back, and move the Fermi line with them.")
        self.reference.activated.connect(self._on_frame_chosen)
        energy.add("Reference", self.reference)
        # Held in a tuple rather than as attributes of their own, so the
        # dialog's memory passes them by: the window belongs to the data.
        self._energy = range_row(energy, "Window", -10.0, 10.0, -30.0, 30.0,
                                 decimals=2, step=0.5)
        fit = QPushButton("Suggested window", self)
        fit.setToolTip("Down to the first wide gap below the Fermi level — where "
                       "the core levels start — and as far again above the "
                       "bottom of the conduction band.")
        fit.clicked.connect(lambda _checked=False: self._fit_energy())
        energy.append(_row(fit))

    def _build_bands(self, column) -> None:
        bands = Section(column, "Band structure")
        self._k_edit = QLineEdit(self)
        self._k_edit.setPlaceholderText("Γ X W L Γ")
        self._k_edit.setToolTip(
            "One name per corner of the path, separated by spaces. Filled in "
            "from the loaded structure where the corners can be recognised; "
            "“G” is read as Γ.")
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
        self._table.setToolTip("Tick the projections to draw. Double-click a name "
                               "to change what the legend calls it.")
        dos.add("Projections", self._table)
        self.overlay = QCheckBox("Draw the projections on one axis", self)
        self.overlay.setChecked(True)
        self.overlay.setToolTip(
            "On one axis with a legend, or one panel each. The band+DOS view "
            "always draws them together.")
        dos.add_wide(self.overlay)
        self.spin_down = QComboBox(self)
        self.spin_down.addItem("Mirrored below the axis", el.SPIN_MIRRORED)
        self.spin_down.addItem("On the same side as spin up", el.SPIN_ALONGSIDE)
        self.spin_down.setToolTip("Only a spin-polarised DOS has a spin-down part.")
        self.spin_down.currentIndexChanged.connect(lambda _i: self._fit_dos())
        dos.add("Spin down", self.spin_down)
        self.dos_auto = QCheckBox("Fit the DOS axis to what is drawn", self)
        self.dos_auto.setChecked(True)
        self.dos_auto.toggled.connect(lambda _on: self._sync())
        dos.add_wide(self.dos_auto)
        self._dos_span = range_row(dos, "DOS range", 0.0, 1.0, 0.0, 10.0,
                                   decimals=2, step=0.1)
        self._dos_range_row = dos.row_widgets("DOS range")
        self._dos_section = dos

    def _build_appearance(self, column) -> None:
        look = Section(column, "Appearance")
        self.band_colour = ColourButton(el.BAND_COLOUR, "Band colour", self)
        look.add("Bands", _row(self.band_colour))

        self.beta_colour = ColourButton(el.BETA_COLOUR, "Spin-down band colour", self)
        self.beta_style = _style_box(self, "--")
        look.add("Spin-down bands", _row(self.beta_colour, self.beta_style))
        self._beta_row = look.row_widgets("Spin-down bands")

        self.show_fermi = QCheckBox("Draw", self)
        self.show_fermi.setChecked(True)
        self.show_fermi.toggled.connect(lambda _on: self._sync())
        self.fermi_colour = ColourButton(el.FERMI_COLOUR, "Fermi level colour", self)
        self.fermi_style = _style_box(self, "-")
        self.fermi_width = QDoubleSpinBox(self)
        self.fermi_width.setRange(0.3, 5.0)
        self.fermi_width.setSingleStep(0.1)
        self.fermi_width.setDecimals(1)
        self.fermi_width.setValue(1.5)
        self.fermi_width.setToolTip("Line width of the Fermi level")
        look.add("Fermi level", _row(self.show_fermi, self.fermi_colour,
                                     self.fermi_style, self.fermi_width))

        self.linewidth = slider_row(look, "Line width", 1.0, 0.3, 4.0, 0.1, decimals=1)
        self._title = QLineEdit(self)
        self._title.setPlaceholderText("No title")
        look.add("Title", self._title)
        self.legend = QCheckBox("Legend", self)
        self.legend.setChecked(True)
        look.add_wide(self.legend)
        hint = QLabel("The figure font is shared by every plot: Plot ▸ Plot font…")
        hint.setStyleSheet("color: palette(mid);")
        look.append(hint)

    # ── files ───────────────────────────────────────────────────────────
    def _populate(self, folder: str) -> bool:
        """List the band and DOS files in ``folder``, and load the best of each."""
        if not folder:
            return False
        bands, doss = el.find_files(folder, efermi=self._output_efermi)
        # A band structure and a DOS from the same calculation, not merely the
        # best-ranked of each — see pair_files.
        chosen = dict(zip(("band", "dos"),
                          el.pair_files(bands, doss, own_run=self._output_efermi is not None)))
        for kind, paths in (("band", bands), ("dos", doss)):
            combo = self._files[kind]
            combo.blockSignals(True)
            combo.clear()
            for path in paths:
                combo.addItem(Path(path).name, path)
                combo.setItemData(combo.count() - 1, path, Qt.ToolTipRole)
            combo.setCurrentIndex(paths.index(chosen[kind]) if chosen[kind] in paths
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
                 or ElectronicDialog.last_folder)
        path, _filter = QFileDialog.getOpenFileName(
            self, "Open a band structure" if kind == "band" else "Open a density of states",
            start, _BAND_FILTER if kind == "band" else _DOS_FILTER)
        if not path:
            return
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
        path = self._path(kind)
        if self._loaded.get(kind) == path:
            return
        self._loaded[kind] = path
        info, error = None, ""
        if path:
            try:
                info = el.describe_bands(path) if kind == "band" else el.describe_dos(path)
            except Exception as exc:  # noqa: BLE001 - said in the dialog, not raised
                error = f"Could not read {Path(path).name}: {exc}"
            else:
                ElectronicDialog.last_folder = str(Path(path).parent)
        self._errors[kind] = error
        if kind == "band":
            self._bands_info = info
            self._auto_names = el.path_labels(info, self._structure) if info else []
            self._k_edit.setText(" ".join(self._auto_names))
        else:
            self._dos_info = info
            self._fill_projections(info)
            self._fit_dos()
        self._fit_energy()
        self._sync()

    def _fill_projections(self, info: Optional[el.DosInfo]) -> None:
        self._table.setRowCount(0)
        self._projection_colours = []
        count = info.n_projections if info else 0
        self._table.setRowCount(count)
        for row in range(count):
            item = QTableWidgetItem(f"Projection {row + 1}")
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable
                          | Qt.ItemIsEditable | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self._table.setItem(row, 0, item)
            swatch = ColourButton(el.PROJECTION_COLOURS[row % len(el.PROJECTION_COLOURS)],
                                  f"Colour for projection {row + 1}", self)
            self._table.setCellWidget(row, 1, _centred(swatch))
            self._projection_colours.append(swatch)

    # ── the energy window ───────────────────────────────────────────────
    def _frame(self):
        return (self.unit.currentData(), self.reference.currentData())

    def _efermi(self) -> float:
        """E_F in eV — the band file's when there is one, as the plotter uses."""
        info = self._bands_info or self._dos_info
        return info.efermi if info is not None else 0.0

    def _shown(self, value: float, frame) -> float:
        """eV relative to E_F -> the number the boxes show in ``frame``."""
        unit, reference = frame
        value += self._efermi() if reference == el.ABSOLUTE else 0.0
        return value / _HARTREE_EV if unit == "Hartree" else value

    def _canonical(self, value: float, frame) -> float:
        unit, reference = frame
        value = value * _HARTREE_EV if unit == "Hartree" else value
        return value - (self._efermi() if reference == el.ABSOLUTE else 0.0)

    def _extent(self):
        """``(window, span)`` in eV relative to E_F for what is to be drawn."""
        need_bands, need_dos = self._needs()
        bands = self._bands_info if need_bands else None
        doss = self._dos_info if need_dos else None
        spans = [info.energy_span for info in (bands, doss) if info is not None]
        if not spans:
            return None, None
        # The bands decide the window when there are any: a DOS covers the
        # window its .d3 asked for, which is already a choice somebody made.
        window = (bands.window if bands is not None
                  else (float(math.floor(doss.energy_span[0])),
                        float(math.ceil(doss.energy_span[1]))))
        span = (math.floor(min([s[0] for s in spans] + [window[0]])),
                math.ceil(max([s[1] for s in spans] + [window[1]])))
        return window, span

    def _format_energy(self, unit: str) -> None:
        _slider, low, high = self._energy
        for box in (low, high):
            box.setDecimals(4 if unit == "Hartree" else 2)
            box.setSingleStep(0.02 if unit == "Hartree" else 0.5)

    def _fit_energy(self) -> None:
        """Span the track over the data, and set the handles on the suggestion."""
        frame = self._frame()
        self._format_energy(frame[0])
        self._frame_shown = frame
        window, span = self._extent()
        if window is None:
            return
        bounds = sorted(self._shown(v, frame) for v in span)
        values = sorted(self._shown(v, frame) for v in window)
        _set_range(self._energy, *bounds, *values)

    @guard()
    def _on_frame_chosen(self, _index: int = 0) -> None:
        new, old = self._frame(), self._frame_shown
        if new == old:
            return
        slider, low, high = self._energy
        values = [self._shown(self._canonical(box.value(), old), new) for box in (low, high)]
        bounds = sorted(self._shown(self._canonical(v, old), new) for v in slider.bounds())
        self._format_energy(new[0])
        _set_range(self._energy, *bounds, *sorted(values))
        if new[0] != old[0]:
            # states/eV <-> states/Hartree: CRYSTALClear rescales the DOS too.
            factor = _HARTREE_EV if new[0] == "Hartree" else 1.0 / _HARTREE_EV
            dslider, dlow, dhigh = self._dos_span
            dbounds = [v * factor for v in dslider.bounds()]
            _set_range(self._dos_span, *dbounds, dlow.value() * factor, dhigh.value() * factor)
        self._frame_shown = new

    def _fit_dos(self) -> None:
        info = self._dos_info
        if info is None or info.dos_max <= 0:
            return
        top = info.dos_max * (_HARTREE_EV if self.unit.currentData() == "Hartree" else 1.0)
        mirrored = info.spin == 2 and self.spin_down.currentData() == el.SPIN_MIRRORED
        bound, value = _nice(top * 1.25), _nice(top * 1.05)
        _set_range(self._dos_span, -bound if mirrored else 0.0, bound,
                   -value if mirrored else 0.0, value)

    def _on_mode_changed(self, _index: int = 0) -> None:
        self._fit_energy()
        self._sync()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt's name
        # Remembered settings were put back after construction; fit the window
        # in whatever unit and reference they left.
        if not getattr(self, "_shown_once", False):
            self._shown_once = True
            self._fit_energy()
            self._fit_dos()
        super().showEvent(event)

    # ── keeping the controls honest ─────────────────────────────────────
    def _needs(self):
        mode = self.mode.currentData()
        return mode in (el.BANDS, el.BANDS_AND_DOS), mode in (el.DOS, el.BANDS_AND_DOS)

    def _sync(self) -> None:
        """Offer only what the chosen plot uses, and allow OK only when it can work.

        A control that would quietly do nothing is greyed rather than left
        live: the band settings for a DOS plot, the spin-down choices for a
        spin-restricted run, the overlay for the combined view (which always
        overlays).
        """
        need_bands, need_dos = self._needs()
        self._file_rows["band"].setEnabled(need_bands)
        self._file_rows["dos"].setEnabled(need_dos)
        self._bands_section.set_enabled(need_bands)
        self._dos_section.set_enabled(need_dos)
        polarised_dos = need_dos and self._dos_info is not None and self._dos_info.spin == 2
        self.spin_down.setEnabled(polarised_dos)
        self.overlay.setEnabled(self.mode.currentData() == el.DOS)
        for widget in self._dos_range_row:
            widget.setEnabled(need_dos and not self.dos_auto.isChecked())

        polarised_bands = (need_bands and self._bands_info is not None
                           and self._bands_info.spin == 2)
        for widget in self._beta_row:
            widget.setEnabled(polarised_bands)
        self.band_colour.setEnabled(need_bands)
        for widget in (self.fermi_colour, self.fermi_style, self.fermi_width):
            widget.setEnabled(self.show_fermi.isChecked())

        ready = ((not need_bands or self._bands_info is not None)
                 and (not need_dos or self._dos_info is not None))
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(ready)
        self._summary.setText(self._describe(need_bands, need_dos))

    def _describe(self, need_bands: bool, need_dos: bool) -> str:
        lines = []
        if need_bands:
            info = self._bands_info
            if info is not None:
                spin = "spin-polarised" if info.spin == 2 else "spin-restricted"
                lines.append(f"Bands: {info.n_bands} bands, {spin}, "
                             f"E_F = {info.efermi:.3f} eV")
                typed = self._typed_labels()
                if typed and len(typed) != len(info.ticks):
                    lines.append(f"Path labels: {len(info.ticks)} corners, {len(typed)} "
                                 "labels typed — the automatic ones will be used.")
            elif self._errors.get("band"):
                lines.append(self._errors["band"])
            else:
                lines.append("No band file found here — choose one.")
        if need_dos:
            info = self._dos_info
            if info is not None:
                spin = "spin-polarised" if info.spin == 2 else "spin-restricted"
                lines.append(f"DOS: {info.n_projections} projection"
                             f"{'s' if info.n_projections != 1 else ''}, {spin}, "
                             f"E_F = {info.efermi:.3f} eV")
            elif self._errors.get("dos"):
                lines.append(self._errors["dos"])
            else:
                lines.append("No DOS file found here — choose one.")
        if need_bands and need_dos and self._bands_info and self._dos_info:
            gap = abs(self._bands_info.efermi - self._dos_info.efermi)
            if gap > 0.01:
                # Each file is relative to its own E_F, so side by side the DOS
                # is shifted against the bands by exactly this much.
                lines.append(f"⚠ The two Fermi levels differ by {gap:.3f} eV — the DOS "
                             "will sit that far off the bands. Same calculation?")
        return "\n".join(lines)

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
            return [_greek(token) for token in typed]
        return list(self._auto_names) or None

    def energy_window(self):
        _slider, low, high = self._energy
        return tuple(sorted((low.value(), high.value())))

    def energy_bounds(self):
        return tuple(self._energy[0].bounds())

    def request(self):
        """``(band_path, dos_path, ElectronicOptions)`` for :func:`plot_electronic`."""
        need_bands, need_dos = self._needs()
        low, high = self.energy_window()
        _slider, dlow, dhigh = self._dos_span
        dos_low, dos_high = sorted((dlow.value(), dhigh.value()))
        rows = [row for row in range(self._table.rowCount())
                if self._table.item(row, 0).checkState() == Qt.Checked]
        options = el.ElectronicOptions(
            mode=self.mode.currentData(),
            unit=self.unit.currentData(),
            reference=self.reference.currentData(),
            # A zero-width range means "don't clip", as in the spectra dialog.
            energy_range=None if low == high else (low, high),
            dos_range=(None if self.dos_auto.isChecked() or dos_low == dos_high
                       else (dos_low, dos_high)),
            k_labels=self.k_labels() if need_bands else None,
            projections=[row + 1 for row in rows] or None,
            projection_labels=[self._table.item(row, 0).text().strip()
                               or f"Projection {row + 1}" for row in rows] or None,
            projection_colours=[self._projection_colours[row].colour() for row in rows] or None,
            overlay_projections=self.overlay.isChecked(),
            spin_down=self.spin_down.currentData(),
            band_colour=self.band_colour.colour(),
            beta_colour=self.beta_colour.colour(),
            beta_style=self.beta_style.currentData(),
            linewidth=self.linewidth.value(),
            show_fermi=self.show_fermi.isChecked(),
            fermi_colour=self.fermi_colour.colour(),
            fermi_style=self.fermi_style.currentData(),
            fermi_width=self.fermi_width.value(),
            legend=self.legend.isChecked(),
            title=self._title.text().strip() or None,
        )
        return (self._path("band") if need_bands else None,
                self._path("dos") if need_dos else None,
                options)

    def plot_title(self) -> str:
        return {el.BANDS: "Band structure", el.DOS: "Density of states",
                el.BANDS_AND_DOS: "Bands + DOS"}[self.mode.currentData()]


# ── small pieces ──────────────────────────────────────────────────────────
def _set_range(triple, low_bound: float, high_bound: float, low: float, high: float) -> None:
    """Move a :func:`range_row`'s track and handles together."""
    slider, low_box, high_box = triple
    if high_bound <= low_bound:
        high_bound = low_bound + 1.0
    slider.setBounds(low_bound, high_bound)
    for box in (low_box, high_box):
        box.blockSignals(True)
        box.setRange(low_bound, high_bound)
        box.blockSignals(False)
    low_box.setValue(low)
    high_box.setValue(high)
    slider.setValues(low_box.value(), high_box.value())


def _nice(value: float) -> float:
    """Round up to two significant figures: 1.63 -> 1.7, 73.8 -> 74."""
    if value <= 0:
        return 1.0
    step = 10.0 ** (math.floor(math.log10(value)) - 1)
    return math.ceil(value / step - 1e-9) * step


def _style_box(parent, default: str) -> QComboBox:
    box = QComboBox(parent)
    for key, label in el.LINE_STYLES:
        box.addItem(label, key)
    box.setCurrentIndex(max(0, box.findData(default)))
    return box


def _row(*widgets) -> QWidget:
    """Controls side by side at the left of a row, at their own sizes."""
    holder = QWidget()
    line = QHBoxLayout(holder)
    line.setContentsMargins(0, 0, 0, 0)
    line.setSpacing(8)
    for widget in widgets:
        line.addWidget(widget, 0)
    line.addStretch(1)
    return holder


def _centred(widget) -> QWidget:
    holder = QWidget()
    line = QHBoxLayout(holder)
    line.setContentsMargins(4, 0, 4, 0)
    line.addStretch(1)
    line.addWidget(widget)
    line.addStretch(1)
    return holder


def _greek(token: str) -> str:
    """``G`` / ``Gamma`` -> ``Γ``: CRYSTAL writes Γ as G, and so do people."""
    return "Γ" if token.lower() in ("g", "gamma") else token


__all__ = ["ElectronicDialog"]
