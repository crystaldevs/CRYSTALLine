"""Choose which scalar field to draw over the structure, and how.

An ECH3 or POT3 run leaves its grids beside the SCF output — ``DENS_CUBE.DAT``,
``SPIN_CUBE.DAT``, ``POT_CUBE.DAT``, ``fort.31`` — so they are found rather
than asked for, the way the orbitals and the band files already are. They are
recognised by what they hold, not by their names: ``DENS_CUBE.DAT`` shares its
extension with ``BAND.DAT``.

Three things here are decisions rather than defaults, and each has a reason:

* **The isovalue slider is logarithmic.** A charge density runs from ~10⁻⁴
  e/bohr³ in the middle of a void to hundreds at a nucleus. On a linear slider
  everything worth looking at is in the first millimetre of travel.
* **A second field is a first-class choice**, because the two pictures people
  actually want out of a pair of grids are the potential painted onto the
  density and one density minus another. Both need two files and neither is
  reachable by drawing them one after the other.
* **The window is fitted to the data.** A default isovalue that suits a bulk
  oxide shows nothing of a molecule in a box, so the starting value comes from
  the field: the slider spans its own range and opens at a value that has a
  surface to show.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from crystalline.crystalio import density as den
from crystalline.ui.panels.controls import ColourButton, Section, slider_row
from crystalline.ui.safety import guard

# By extension only: a cube's name is rewritten freely, its extension is not.
_FILTER = "Grid files (*.cube *.cub *.dat *.31);;All files (*)"

# What the second file is for. "Nothing" is the common case and comes first.
_ALONE = "alone"
_PAINT = "paint"
_SUBTRACT = "subtract"
_COMPANIONS = (
    (_ALONE, "Draw it on its own"),
    (_PAINT, "Colour the surface by a second field"),
    (_SUBTRACT, "Subtract a second field"),
)

# How much of the cell the opening isosurface wraps. A fifth of it clears the
# drawn atoms in every case tried — a metal, an oxide, a semiconductor and a
# molecular crystal — without the surfaces merging into one mass. Much less and
# the surface shrinks inside the atom spheres and nothing is seen at all.
DENSE_FRACTION = 0.2

_COLOURMAPS = ("coolwarm", "RdBu_r", "seismic", "viridis", "plasma", "inferno", "turbo")


class DensityDialog(QDialog):
    """Pick the field and the options; :meth:`request` hands them to the renderer."""

    # Where fields were last found in this session — the fallback when the
    # loaded output's folder has none.
    last_folder = ""

    def __init__(self, folder: str = "", stem: str = "",
                 parent: Optional[QWidget] = None, miller_cell=None, cell=None) -> None:
        """``miller_cell`` is the conventional cell, which Miller indices refer to;
        ``cell`` is the open structure's own, which a grid of it shares."""
        super().__init__(parent)
        self._cell = cell
        self.setWindowTitle("Electron density & potential")
        self.resize(840, 480)
        self._miller_cell = miller_cell
        self._folder = folder
        self._stem = stem
        self._field: Optional[den.ScalarField] = None
        self._second: Optional[den.ScalarField] = None
        self._error = ""

        outer = QVBoxLayout(self)
        intro = QLabel(
            "A charge density, a spin density or an electrostatic potential "
            "from an ECH3 or POT3 run. Values are in the atomic units CRYSTAL "
            "writes them in.")
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
        self._build_view(left)
        left.addStretch(1)
        self._build_appearance(right)
        right.addStretch(1)

        self._buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        outer.addWidget(self._buttons)

        if not self._populate(folder) and DensityDialog.last_folder not in ("", folder):
            self._populate(DensityDialog.last_folder)
        self._sync()

    # ── building ────────────────────────────────────────────────────────
    def _build_data(self, column) -> None:
        section = Section(column, "Data")
        self.file = QComboBox(self)
        self.file.setMinimumWidth(260)
        self.file.currentIndexChanged.connect(lambda _i: self._load())
        browse = QPushButton("Browse…")
        browse.clicked.connect(lambda: self._browse("first"))
        section.add("Field", _with_button(self.file, browse))

        self.companion = QComboBox(self)
        for key, label in _COMPANIONS:
            self.companion.addItem(label, key)
        self.companion.currentIndexChanged.connect(lambda _i: self._sync())
        section.add("Second field", self.companion)

        self.second = QComboBox(self)
        self.second.currentIndexChanged.connect(lambda _i: self._load_second())
        second_browse = QPushButton("Browse…")
        second_browse.clicked.connect(lambda: self._browse("second"))
        section.add("and", _with_button(self.second, second_browse))
        self._second_row = section.row_widgets("and")

        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color: palette(mid);")
        section.append(self.info)

    def _build_view(self, column) -> None:
        section = Section(column, "View")
        self.view = QComboBox(self)
        for key, label in den.VIEWS:
            self.view.addItem(label, key)
        self.view.currentIndexChanged.connect(lambda _i: self._sync())
        section.add("Show as", self.view)

        # Logarithmic: a density spans four orders of magnitude between a void
        # and a bond, and every value worth seeing is in the bottom decade.
        self.isovalue = slider_row(
            section, "Isovalue", den.DEFAULT_ISOVALUE, 1e-4, 10.0, 0.01,
            decimals=4, logarithmic=True,
            on_change=lambda _v: self._describe(),
        )
        self._isovalue_row = section.row_widgets("Isovalue")
        # Wider than the shared value column: an isovalue is 0,0020, not 0,85,
        # and the standard width cuts it to "0,00".
        self.isovalue.setFixedWidth(112)
        # Miller indices, in the conventional cell. A plane named by a lattice
        # vector of the primitive cell was not a plane anyone asks for: MgO's
        # "across c" is a {111} layer holding one kind of atom.
        indices = QHBoxLayout()
        indices.setContentsMargins(0, 0, 0, 0)
        self.miller: List[QSpinBox] = []
        for index, value in zip("hkl", (0, 0, 1)):
            box = QSpinBox(self)
            box.setRange(-9, 9)
            box.setValue(value)
            box.valueChanged.connect(lambda _v: self._sync())
            indices.addWidget(box)
            self.miller.append(box)
            # Each box its own attribute as well: the remembered settings are
            # read off the dialog's attributes, and a list of boxes is not one.
            setattr(self, f"miller_{index}", box)
        indices.addStretch(1)
        holder = QWidget()
        holder.setLayout(indices)
        holder.setToolTip("Miller indices of the plane, in the conventional cell.")
        section.add("Plane (hkl)", holder)
        self._miller_row = section.row_widgets("Plane (hkl)")
        # 0 and 1 are the same plane of the family, one spacing apart.
        self.offset = slider_row(section, "Position", 0.0, 0.0, 1.0, 0.01, decimals=2,
                                 on_change=lambda _v: self._describe())
        self.offset.setToolTip(
            "Where the plane sits along its normal, as a fraction of the "
            "interplanar spacing d(hkl): 0 passes through the cell origin.")
        self._offset_row = section.row_widgets("Position")
        self.cutaway = QCheckBox("Cut away the crystal in front of the plane")
        self.cutaway.setChecked(True)
        self.cutaway.setToolTip(
            "Atoms, bonds and cell edges on the near side of the plane are not "
            "drawn, so the plane is not hidden inside the crystal. Atoms lying "
            "in the plane are kept, and show as domes on it. Swap the sign of "
            "the indices to cut away the other side.")
        section.add_wide(self.cutaway)
        self.logarithmic = QCheckBox("Colour on a logarithmic scale")
        self.logarithmic.setChecked(True)
        self.logarithmic.setToolTip(
            "A linear scale puts everything but the core region at one end of "
            "the colour map.")
        section.add_wide(self.logarithmic)

    def _build_appearance(self, column) -> None:
        section = Section(column, "Appearance")
        self.positive = ColourButton(den.POSITIVE_COLOUR, "Surface colour")
        section.add("Surface", self.positive)
        self._positive_row = section.row_widgets("Surface")
        self.negative = ColourButton(den.NEGATIVE_COLOUR, "Negative-lobe colour")
        section.add("Negative lobe", self.negative)
        self._negative_row = section.row_widgets("Negative lobe")
        self.cmap = QComboBox(self)
        for name in _COLOURMAPS:
            self.cmap.addItem(name, name)
        section.add("Colour map", self.cmap)
        self._cmap_row = section.row_widgets("Colour map")
        self.opacity = slider_row(section, "Opacity", 0.85, 0.05, 1.0, 0.05, decimals=2)
        self.colour_bar = QCheckBox("Show a colour bar")
        self.colour_bar.setToolTip(
            "Key the colour map with a bar at the right of the view. Offered for "
            "a plane and for a surface coloured by a second field; a plain "
            "surface has one colour and nothing to key.")
        section.add_wide(self.colour_bar)

        self.clip = QCheckBox("Clip to the cell")
        self.clip.setChecked(False)
        self.clip.setToolTip(
            "Cut the surfaces at the faces of the outlined cell. Off, they are "
            "drawn whole around every atom on screen.")
        section.add_wide(self.clip)

    # ── files ───────────────────────────────────────────────────────────
    def _populate(self, folder: str) -> bool:
        """Fill both combos from ``folder``. False if it held nothing."""
        found = den.find_fields(folder, cell=self._cell) if folder else []
        if not found:
            return False
        for combo in (self.file, self.second):
            blocked = combo.blockSignals(True)
            combo.clear()
            for path in found:
                combo.addItem(Path(path).name, path)
            combo.blockSignals(blocked)
        # The second combo opens on a different file when there is one, since
        # painting or subtracting a field onto itself shows nothing.
        if self.second.count() > 1:
            self.second.setCurrentIndex(1)
        DensityDialog.last_folder = folder
        self._folder = folder
        self._load()
        self._load_second()
        return True

    @guard()
    def _browse(self, which: str) -> None:
        start = self._folder or DensityDialog.last_folder or os.getcwd()
        path, _filter = QFileDialog.getOpenFileName(self, "Open a scalar field", start, _FILTER)
        if not path:
            return
        combo = self.file if which == "first" else self.second
        combo.insertItem(0, Path(path).name, path)
        combo.setCurrentIndex(0)
        DensityDialog.last_folder = str(Path(path).parent)

    def _read(self, path: str):
        try:
            return den.read_field(path), ""
        except (den.DensityError, OSError) as error:
            return None, str(error)

    @guard()
    def _load(self) -> None:
        path = self.file.currentData()
        self._field, self._error = (None, "") if not path else self._read(path)
        self._fit()
        self._sync()

    @guard()
    def _load_second(self) -> None:
        path = self.second.currentData()
        self._second, _error = (None, "") if not path else self._read(path)
        self._sync()

    def _fit(self) -> None:
        """Open the isovalue where this field has something to show.

        No fixed default serves a metal, an oxide and a molecular crystal at
        once: 0.1 e/bohr³ separates beryllium's atoms nicely and buries MgO's
        under overlapping spheres. Nor do the obvious data-driven guesses.

        * A fraction of the **peak** tracks the cores, which are orders of
          magnitude denser than anything else, so the surface is a dot on each
          nucleus.
        * The level enclosing most of the **charge** tracks the valence tail: a
          grid has vastly more points in the sparse regions than in the dense
          ones, so nine tenths of the charge is reached at 0.035 e/bohr³ in
          beryllium and the surface fills the cell.

        What behaves is a fraction of the **volume**. The level wrapping the
        densest fifth of the cell clears the drawn atoms whatever they are: on
        the tutorial's own four cases it gives 0.043 (Be), 0.101 (MgO), 0.061
        (Si) and 0.058 (urea) e/bohr³, which draw respectively the hexagonal
        shells of a metal, the oxide ion dwarfing its cation, silicon's bonds,
        and a molecular envelope around each urea.

        A tighter surface is not wrong, it is invisible: at three per cent the
        surface of MgO's oxygen has a radius of 0.4 Å and sits entirely inside
        the sphere drawn for the atom.
        """
        if self._field is None:
            return
        import numpy as np

        values = np.abs(np.asarray(self._field.values, dtype=float)).ravel()
        positive = values[values > 0]
        if positive.size == 0:
            return
        low = max(float(positive.min()), 1e-4)
        high = max(float(values.max()), low * 10)
        self.isovalue.setRange(low, high)
        # The unit goes on the label, not in the box: a suffix would leave
        # "0,1377 e/boh" in a box wide enough for the number itself.
        self._isovalue_row[0].setText(f"Isovalue ({self._field.unit})")
        # A diverging map says "above or below zero", which is the question for
        # a spin density or a difference and a meaningless one for a charge
        # density, whose slice then comes out as one half of the map.
        self.cmap.setCurrentIndex(self.cmap.findData(
            "coolwarm" if self._field.signed else "viridis"))
        if self._field.signed:
            # A spin density or a difference has no dense core to speak of, and
            # is near zero over most of the cell, so a volume fraction of it is
            # noise. A tenth of the peak is the convention for an orbital.
            guess = 0.1 * float(values.max())
        else:
            guess = _dense_fraction_level(values, DENSE_FRACTION)
        self.isovalue.setValue(min(max(guess, low), high))

    # ── state ───────────────────────────────────────────────────────────
    @guard()
    def _sync(self) -> None:
        surface = self.view.currentData() == den.ISOSURFACE
        companion = self.companion.currentData()
        painting = companion == _PAINT
        _show(self._second_row, companion != _ALONE)
        _show(self._isovalue_row, surface)
        _show(self._miller_row, not surface)
        _show(self._offset_row, not surface)
        self.logarithmic.setVisible(not surface)
        self.cutaway.setVisible(not surface)
        # A slice takes its colours from the map, not from the two swatches.
        _show(self._positive_row, surface)
        _show(self._negative_row, surface and self._signed())
        _show(self._cmap_row, not surface or painting)
        self.colour_bar.setVisible(not surface or painting)
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(
            self._field is not None and (surface or self._indices() != (0, 0, 0)))
        self._describe()

    def _indices(self):
        return tuple(box.value() for box in self.miller)

    def _plane_note(self) -> str:
        """d(hkl) and where the chosen plane sits, for the info line."""
        hkl = self._indices()
        if hkl == (0, 0, 0):
            return "(000) is not a plane: give at least one non-zero index."
        if self._miller_cell is None:
            return ""
        try:
            _point, _normal, spacing = den.miller_plane(self._miller_cell, hkl)
        except (den.DensityError, ValueError, ArithmeticError):
            return ""
        name = "(" + " ".join(str(v) for v in hkl) + ")"
        return (f"{name}: d = {spacing:.3f} Å, plane at "
                f"{self.offset.value() * spacing:.3f} Å from the origin")

    def _signed(self) -> bool:
        if self.companion.currentData() == _SUBTRACT:
            return True
        return self._field is not None and self._field.signed

    @guard()
    def _describe(self) -> None:
        if self._error:
            self.info.setText(self._error)
            return
        if self._field is None:
            self.info.setText(
                "No ECH3 or POT3 grid was found beside the output. Browse for a "
                "CUBE file or a fort.31.")
            return
        field = self._field
        na, nb, nc = field.shape
        lengths = [float(v) for v in (field.span() ** 2).sum(axis=1) ** 0.5]
        text = (f"{den.KIND_NAMES.get(field.kind, 'Field')} · {na}×{nb}×{nc} points "
                f"over {lengths[0]:.2f} × {lengths[1]:.2f} × {lengths[2]:.2f} Å · "
                f"peak {field.peak:.4g} {field.unit}")
        if self.view.currentData() == den.SLICE:
            note = self._plane_note()
            if note:
                text += "\n" + note
        if self.companion.currentData() != _ALONE and self._second is not None:
            agrees = den.same_grid(field, self._second)
            text += ("\nThe second field is on the same grid."
                     if agrees else
                     "\nThe second field is on a different grid — it must come "
                     "from a run with the same cell, points and extents.")
        self.info.setText(text)

    # ── the answer ──────────────────────────────────────────────────────
    def request(self):
        """``(field, options)`` for the renderer, or ``(None, None)``.

        A subtraction is done here rather than in the renderer: the difference
        *is* the field being drawn, and everything downstream — its sign, its
        peak, the isovalue read against it — follows from that.
        """
        if self._field is None:
            return None, None
        field = self._field
        companion = self.companion.currentData()
        colour_by = None
        if companion == _SUBTRACT and self._second is not None:
            field = den.difference(field, self._second)
        elif companion == _PAINT and self._second is not None:
            colour_by = self._second
        options = den.DensityOptions(
            view=self.view.currentData(),
            isovalue=self.isovalue.value(),
            opacity=self.opacity.value(),
            positive=self.positive.colour(),
            negative=self.negative.colour(),
            colour_by=colour_by,
            cmap=self.cmap.currentData(),
            miller=self._indices(),
            offset=self.offset.value(),
            cutaway=self.cutaway.isChecked(),
            colour_bar=self.colour_bar.isChecked() and (
                self.view.currentData() == den.SLICE or colour_by is not None),
            logarithmic=self.logarithmic.isChecked(),
            clip_to_cell=self.clip.isChecked(),
        )
        return field, options


def _dense_fraction_level(values, fraction: float) -> float:
    """The level above which ``fraction`` of the cell's volume lies.

    Every grid point stands for the same small volume, so this is just a
    percentile of the values.
    """
    import numpy as np

    return float(np.percentile(np.asarray(values, dtype=float).ravel(),
                               100.0 * (1.0 - fraction)))


def _show(row, visible: bool) -> None:
    """Show or hide both halves of a labelled row.

    A row that does not apply — a slice's position while a surface is being
    drawn — should go away rather than sit there greyed, and a row is a label
    and a control, not one widget.
    """
    for widget in row:
        widget.setVisible(visible)


def _with_button(widget: QWidget, button: QWidget) -> QWidget:
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(widget, 1)
    row.addWidget(button, 0)
    holder = QWidget()
    holder.setLayout(row)
    return holder
