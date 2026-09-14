"""Build a CRYSTAL ``PROPERTIES`` deck (``.d3``) for the loaded structure.

Small on purpose. A ``.d3`` carries no geometry and no basis — only a list of
properties — so the choices are few, and the ones offered here are the ones
whose output CRYSTALLine can open again: band structures, densities of states,
and the crystalline orbitals the orbital viewer draws.

The band path is the part worth automating: it is a dozen lines of integers
over a shrinking factor, and getting one wrong gives a band structure of a path
nobody asked for that still looks like a band structure. It comes from the
lattice instead, and the preview shows exactly what will be written.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from crystalline.ui.panels.band_path_editor import BandPathEditor
from crystalline.core.crystal_input import suggest_shrink
from crystalline.core.properties_input import (
    BandOptions,
    CoopOptions,
    DossOptions,
    EmdOptions,
    Grid3DOptions,
    NewkOptions,
    OrbitalsOptions,
    PropertiesInputError,
    PropertiesSpec,
    XrdOptions,
    build_properties_input,
)
from crystalline.core.structure import Structure


_CONVENTIONAL_NOTE = "the conventional path for this lattice"


class PropertiesBuilderDialog(QDialog):
    """Choose what a PROPERTIES run should compute, and save the deck."""

    def __init__(self, structure: Structure, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Build CRYSTAL properties input (.d3)")
        self.resize(880, 560)  # the same shape as the .d12 builder
        self._structure = structure

        outer = QVBoxLayout(self)
        note = QLabel(
            "A .d3 runs after the SCF and reads its wave function, so it carries "
            "no geometry or basis — run it in the same folder as the completed "
            ".out, with the same file name."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: palette(mid);")
        outer.addWidget(note)

        body = QHBoxLayout()
        body.addWidget(self._controls(), 0)
        body.addWidget(self._preview_panel(), 1)
        outer.addLayout(body)

        buttons = QDialogButtonBox(QDialogButtonBox.Close, self)
        save = QPushButton("Save .d3…")
        save.setDefault(True)
        save.clicked.connect(self._save)
        buttons.addButton(save, QDialogButtonBox.AcceptRole)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        self._refresh()

    # ── the form ────────────────────────────────────────────────────────
    def _controls(self) -> QWidget:
        """One tab per family of properties.

        A .d3 can ask for a couple of dozen different things, and a single
        column of group boxes made the dialog a scroll with no shape to it.
        Tabs give each family its own page and keep the preview beside all of
        them, so what the deck will contain is visible whichever page is open.
        """
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(self._tab_bands(), "Bands && DOS")
        tabs.addTab(self._tab_density(), "Density && potential")
        tabs.addTab(self._tab_orbitals(), "Orbitals")
        tabs.addTab(self._tab_analysis(), "Analysis")
        tabs.setMinimumWidth(330)
        self._connect_refresh()
        return tabs

    def _tab_bands(self) -> QWidget:
        layout, shrink = _column(), self._shrink_guess()

        newk = QGroupBox("NEWK — eigenvectors on a finer mesh")
        newk_form = QFormLayout(newk)
        self._newk_shrink = _spin(shrink, 1, 96)
        self._newk_shrink2 = _spin(shrink * 2, 1, 192)
        newk_form.addRow("Shrinking factor", self._newk_shrink)
        newk_form.addRow("Gilat factor", self._newk_shrink2)
        newk.setToolTip(
            "Always written. It computes the eigenvectors the rest of the run "
            "reads, and is the first step of essentially every properties deck; "
            "without it a run silently uses whatever the SCF left behind."
        )
        layout.addWidget(newk)

        band = _checkable("BAND — band structure", checked=True)
        form = QFormLayout(band)
        self._band_title = QLineEdit("Band structure")
        self._band_points = _spin(200, 10, 5000)
        self._band_first = _spin(1, 1, 9999)
        self._band_last = _spin(0, 0, 9999)
        self._band_last.setSpecialValueText("auto")
        form.addRow("Title", self._band_title)
        form.addRow("Points along the path", self._band_points)
        form.addRow("First band", self._band_first)
        form.addRow("Last band", self._band_last)
        layout.addWidget(band)
        self._band = band

        path = QGroupBox("Path through the Brillouin zone")
        path_layout = QVBoxLayout(path)
        self._path_editor = BandPathEditor(self._structure)
        self._path_editor.changed.connect(self._refresh)
        path_layout.addWidget(self._path_editor)
        layout.addWidget(path)
        self._path = path

        doss = _checkable("DOSS — density of states")
        form = QFormLayout(doss)
        self._doss_points = _spin(300, 10, 5000)
        self._doss_window = QCheckBox("Give an energy window instead of a band range")
        self._doss_low, self._doss_high = _float_spin(-0.7), _float_spin(0.8)
        self._doss_projections = QLineEdit()
        self._doss_projections.setPlaceholderText("e.g.  1, 2   — one projection per atom")
        self._doss_projections.setToolTip(
            "Atoms to project the DOS onto, by CRYSTAL's numbering. Separate "
            "projections with commas; put several atoms in one projection with "
            "spaces, e.g. '1 2, 3'."
        )
        form.addRow("Energy points", self._doss_points)
        form.addRow(self._doss_window)
        form.addRow("", _range_row(self._doss_low, self._doss_high))
        form.addRow("Project onto atoms", self._doss_projections)
        layout.addWidget(doss)
        self._doss = doss

        coop = _checkable("COOP / COHP — bonding analysis")
        form = QFormLayout(coop)
        self._coop_hamiltonian = QCheckBox("Hamiltonian-weighted (COHP)")
        self._coop_points = _spin(300, 10, 5000)
        self._coop_pairs = QLineEdit()
        self._coop_pairs.setPlaceholderText("e.g.  1 : 2,  1 : 3 4")
        self._coop_pairs.setToolTip(
            "Interactions to look at, as two groups of atoms either side of a "
            "colon. Separate interactions with commas: '1 : 2, 1 : 3 4'."
        )
        form.addRow(self._coop_hamiltonian)
        form.addRow("Energy points", self._coop_points)
        form.addRow("Interactions", self._coop_pairs)
        layout.addWidget(coop)
        self._coop = coop

        layout.addStretch(1)
        return _scrollable(layout)

    # ── the grid's extent along directions the lattice does not bound ────
    def _extent_box(self) -> QWidget:
        """How far ECH3 and POT3 sample along a slab's or a molecule's open axes.

        One group for both, because the two keywords define the same grid: a
        deck asking for the charge density and the potential over two different
        boxes could not paint one onto the other. Hidden for a bulk, where the
        cell bounds every direction and the record must not be written at all.
        """
        box = QGroupBox("Grid extent along the open directions")
        form = QFormLayout(box)
        self._extent_mode = QComboBox()
        self._extent_mode.addItem("Scale the atoms' own extent", "scale")
        self._extent_mode.addItem("Explicit range", "range")
        self._extent_scale = _float_spin(3.0, decimals=2)
        self._extent_scale.setMinimum(0.1)
        self._extent_low = _float_spin(-4.0, decimals=2)
        self._extent_high = _float_spin(12.0, decimals=2)
        form.addRow("Extent", self._extent_mode)
        form.addRow("Scale", self._extent_scale)
        form.addRow("Range (bohr)", _bohr_range_row(self._extent_low, self._extent_high))
        note = _muted()
        note.setText(
            "A slab, a polymer or a molecule has directions the cell does not "
            "bound, and ECH3 and POT3 need to be told how far to sample along "
            "each of them."
        )
        form.addRow(note)

        def _sync() -> None:
            explicit = self._extent_mode.currentData() == "range"
            form.labelForField(self._extent_scale).setVisible(not explicit)
            self._extent_scale.setVisible(not explicit)
            holder = self._extent_low.parentWidget()
            form.labelForField(holder).setVisible(explicit)
            holder.setVisible(explicit)

        self._extent_mode.currentIndexChanged.connect(lambda _i: _sync())
        _sync()
        self._extents_box = box
        return box

    def _extents(self) -> dict:
        """The extent arguments for a :class:`Grid3DOptions`."""
        if self._extent_mode.currentData() == "range":
            return {"bounds": (self._extent_low.value(), self._extent_high.value())}
        return {"scale": self._extent_scale.value()}

    def _tab_density(self) -> QWidget:
        layout = _column()

        density = _checkable("ECH3 — charge density on a 3D grid")
        form = QFormLayout(density)
        self._ech3_points = _spin(100, 2, 2000)
        form.addRow("Points along a", self._ech3_points)
        layout.addWidget(density)
        self._density = density

        potential = _checkable("POT3 — electrostatic potential on a 3D grid")
        form = QFormLayout(potential)
        self._pot3_points = _spin(100, 2, 2000)
        self._pot3_tol = _spin(5, 1, 20)
        self._pot3_tol.setToolTip("ITOL, the penetration tolerance. 5 is the manual's suggestion.")
        form.addRow("Points along a", self._pot3_points)
        form.addRow("Tolerance (ITOL)", self._pot3_tol)
        layout.addWidget(potential)
        self._potential = potential

        extents = self._extent_box()
        layout.addWidget(extents)
        # After it is in the layout: a widget shown while it is still parentless
        # is shown as a window, and reparenting it then hides it again.
        extents.setVisible(any(not periodic for periodic in self._structure.pbc))

        emd = _checkable("EMDL — electron momentum density")
        form = QFormLayout(emd)
        self._emd_directions = QLineEdit("1 0 0")
        self._emd_directions.setPlaceholderText("e.g.  1 0 0, 1 1 0")
        self._emd_directions.setToolTip(
            "Directions in oblique coordinates, at most ten, separated by commas."
        )
        self._emd_pmax = _float_spin(3.0)
        self._emd_step = _float_spin(0.1)
        form.addRow("Directions", self._emd_directions)
        form.addRow("Max momentum (a.u.)", self._emd_pmax)
        form.addRow("Step", self._emd_step)
        layout.addWidget(emd)
        self._emd = emd

        layout.addStretch(1)
        return _scrollable(layout)

    def _tab_orbitals(self) -> QWidget:
        layout = _column()

        orbitals = _checkable("ORBITALS — crystalline orbitals (Molden)")
        orbitals.setToolTip(
            "Writes one Molden file per sampled k-point. These are the files "
            "CRYSTALLine's own orbital viewer reads."
        )
        form = QFormLayout(orbitals)
        self._orb_name = QLineEdit("orbitals")
        self._orb_wannier = QCheckBox("Wannier functions (LOCALI, then ILOC=1)")
        self._orb_wannier.setToolTip(
            "Localised orbitals rather than the canonical Bloch ones. LOCALI is "
            "written before ORBITALS automatically: ILOC=1 needs it to have run, "
            "and a deck without it does not fail — it quietly returns canonical "
            "orbitals instead."
        )
        form.addRow("File name", self._orb_name)
        form.addRow(self._orb_wannier)
        layout.addWidget(orbitals)
        self._orbitals = orbitals

        layout.addStretch(1)
        return _scrollable(layout)

    def _tab_analysis(self) -> QWidget:
        layout = _column()

        xrd = _checkable("XRDSPEC — X-ray diffraction spectrum")
        form = QFormLayout(xrd)
        self._xrd_index = _spin(6, 1, 40)
        # four decimals: Cu Kalpha is 1.5406 A, and three rounded it to 1.541
        self._xrd_lambda = _float_spin(1.5406, decimals=4)
        self._xrd_lambda.setToolTip("Wavelength in Angstrom. 1.5406 is Cu Kα.")
        self._xrd_b = _float_spin(1.0)
        self._xrd_b.setToolTip("Isotropic Debye-Waller B, typically 0.5 to 1.5.")
        form.addRow("Max Miller index", self._xrd_index)
        form.addRow("Wavelength (Å)", self._xrd_lambda)
        form.addRow("Debye–Waller B", self._xrd_b)
        layout.addWidget(xrd)
        self._xrd = xrd

        self._ppan = QCheckBox("PPAN — Mulliken population analysis")
        layout.addWidget(self._ppan)

        self._pato = QCheckBox("PATO — density of non-interacting atoms")
        self._pato.setToolTip(
            "Replaces the density matrix with a superposition of atomic "
            "densities. With ECH3 or POT3 it is written before them, so the grids "
            "hold the density of non-interacting atoms — the reference for a "
            "deformation density — and PSCF restores the SCF density for "
            "whatever follows."
        )
        layout.addWidget(self._pato)

        self._extra = QPlainTextEdit()
        self._extra.setPlaceholderText("Extra PROPERTIES keywords, one per line (optional)")
        self._extra.setFixedHeight(70)
        layout.addWidget(self._extra)
        layout.addStretch(1)
        return _scrollable(layout)

    def _shrink_guess(self) -> int:
        return suggest_shrink(self._structure) if self._structure.is_periodic else 1

    def _connect_refresh(self) -> None:
        """Every control re-renders the preview; the checkable ones also re-run
        the enable rules."""
        for widget in (self._band, self._doss, self._coop, self._density,
                       self._potential, self._emd, self._orbitals, self._xrd,
                       self._ppan, self._pato,
                       self._doss_window, self._orb_wannier, self._coop_hamiltonian):
            widget.toggled.connect(self._refresh)
        for widget in (self._band_title, self._orb_name, self._doss_projections,
                       self._coop_pairs, self._emd_directions):
            widget.textChanged.connect(self._refresh)
        for widget in (self._newk_shrink, self._newk_shrink2, self._band_points,
                       self._band_first, self._band_last, self._doss_points,
                       self._coop_points, self._ech3_points, self._pot3_points,
                       self._pot3_tol, self._xrd_index,
                       self._doss_low, self._doss_high, self._emd_pmax,
                       self._emd_step, self._xrd_lambda, self._xrd_b):
            widget.valueChanged.connect(self._refresh)
        self._extra.textChanged.connect(self._refresh)

    def _preview_panel(self) -> QWidget:
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Preview"))
        self._preview = QPlainTextEdit()
        self._preview.setReadOnly(True)
        self._preview.setFont(QFont("Menlo", 11))
        self._preview.setLineWrapMode(QPlainTextEdit.NoWrap)
        layout.addWidget(self._preview, 1)
        return holder

    # ── reading the form ────────────────────────────────────────────────
    def spec(self) -> PropertiesSpec:
        last = self._band_last.value()
        return PropertiesSpec(
            newk=NewkOptions(shrink=self._newk_shrink.value(),
                             shrink2=self._newk_shrink2.value()),
            band=BandOptions(
                enabled=self._band.isChecked(),
                title=self._band_title.text().strip() or "Band structure",
                # Empty while the conventional path is in force: the core
                # derives that one from the lattice itself, so passing the
                # list's copy would freeze the path against a structure that
                # may since have changed.
                conventional=self._path_editor.is_conventional(),
                segments=() if self._path_editor.is_conventional()
                else tuple(self._path_editor.segments()),
                labels=() if self._path_editor.is_conventional()
                else tuple(self._path_editor.labels()),
                shrink=self._path_editor.shrink(),
                points=self._band_points.value(),
                first_band=self._band_first.value(),
                last_band=last or None,
            ),
            doss=DossOptions(
                enabled=self._doss.isChecked(),
                points=self._doss_points.value(),
                window=((self._doss_low.value(), self._doss_high.value())
                        if self._doss_window.isChecked() else None),
                projections=_parse_projections(self._doss_projections.text()),
            ),
            orbitals=OrbitalsOptions(
                enabled=self._orbitals.isChecked(),
                name=self._orb_name.text().strip(),
                fractional=self._structure.is_periodic,
                wannier=self._orb_wannier.isChecked(),
            ),
            charge_density=Grid3DOptions(
                enabled=self._density.isChecked(), points=self._ech3_points.value(),
                **self._extents()),
            potential=Grid3DOptions(
                enabled=self._potential.isChecked(), points=self._pot3_points.value(),
                tolerance=self._pot3_tol.value(), **self._extents()),
            coop=CoopOptions(
                enabled=self._coop.isChecked(),
                hamiltonian=self._coop_hamiltonian.isChecked(),
                points=self._coop_points.value(),
                interactions=_parse_interactions(self._coop_pairs.text())),
            emd=EmdOptions(
                enabled=self._emd.isChecked(),
                directions=_parse_directions(self._emd_directions.text()),
                pmax=self._emd_pmax.value(), step=self._emd_step.value()),
            xrd=XrdOptions(
                enabled=self._xrd.isChecked(), max_index=self._xrd_index.value(),
                wavelength=self._xrd_lambda.value(), debye_waller=self._xrd_b.value()),
            ppan=self._ppan.isChecked(),
            pato=self._pato.isChecked(),
            extra_keywords=self._extra.toPlainText(),
        )

    def _refresh(self) -> None:
        for widget in (self._doss_low, self._doss_high):
            widget.setEnabled(self._doss_window.isChecked())
        try:
            text = build_properties_input(self._structure, self.spec())
        except (PropertiesInputError, ValueError) as exc:
            self._preview.setPlainText(f"# {exc}")
            return
        self._preview.setPlainText(text)

    def _save(self) -> None:
        try:
            text = build_properties_input(self._structure, self.spec())
        except (PropertiesInputError, ValueError) as exc:
            QMessageBox.warning(self, "Cannot build the input", str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CRYSTAL properties input", "properties.d3",
            "CRYSTAL properties input (*.d3);;All files (*)",
        )
        if not path:
            return
        try:
            with open(path, "w") as handle:
                handle.write(text)
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.accept()


# ── helpers ───────────────────────────────────────────────────────────────
def _column() -> QVBoxLayout:
    layout = QVBoxLayout()
    layout.setContentsMargins(4, 4, 4, 4)
    return layout


def _page_of(layout) -> QWidget:
    """Wrap a layout so it can sit in another layout or a form row."""
    holder = QWidget()
    holder.setLayout(layout)
    return holder


def _scrollable(layout) -> QWidget:
    """A tab page that can be made smaller than its own contents.

    Without this the dialog's minimum height is the tallest tab's — 910 pixels
    once the Bands tab grew a path editor — so it opened that tall and could not
    be shrunk at all, whatever it was asked to resize to. A tab whose contents
    do not fit should scroll, not hold the whole window open.
    """
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.NoFrame)
    area.setWidget(_page_of(layout))
    return area


def _checkable(title: str, checked: bool = False) -> QGroupBox:
    box = QGroupBox(title)
    box.setCheckable(True)
    box.setChecked(checked)
    return box


def _muted() -> QLabel:
    label = QLabel()
    label.setWordWrap(True)
    label.setStyleSheet("color: palette(mid);")
    return label


def _bohr_range_row(low: QDoubleSpinBox, high: QDoubleSpinBox) -> QWidget:
    """``low`` to ``high``, in bohr — the unit ECH3 and POT3 read extents in."""
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(low)
    row.addWidget(QLabel("to"))
    row.addWidget(high)
    row.addWidget(QLabel("bohr"))
    row.addStretch(1)
    holder = QWidget()
    holder.setLayout(row)
    return holder


def _range_row(low: QDoubleSpinBox, high: QDoubleSpinBox) -> QWidget:
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(low)
    row.addWidget(QLabel("to"))
    row.addWidget(high)
    row.addWidget(QLabel("hartree"))
    row.addStretch(1)
    holder = QWidget()
    holder.setLayout(row)
    return holder


def _parse_interactions(text: str) -> tuple:
    """``"1 : 2, 1 : 3 4"`` -> ``(((1,), (2,)), ((1,), (3, 4)))``.

    A COOP interaction is between two *groups*, so the colon separates the
    sides and spaces group atoms within a side. An entry missing its colon is
    skipped rather than guessed at — silently pairing an atom with itself would
    be a plot of nothing.
    """
    interactions = []
    for chunk in text.split(","):
        if ":" not in chunk:
            continue
        left, right = chunk.split(":", 1)
        a = tuple(int(t) for t in left.split() if t.strip())
        b = tuple(int(t) for t in right.split() if t.strip())
        if a and b:
            interactions.append((a, b))
    return tuple(interactions)


def _parse_directions(text: str) -> tuple:
    """``"1 0 0, 1 1 0"`` -> ``((1, 0, 0), (1, 1, 0))``; anything not a triple
    is dropped rather than padded into a direction nobody asked for."""
    directions = []
    for chunk in text.split(","):
        values = [t for t in chunk.split() if t.strip()]
        if len(values) == 3:
            try:
                directions.append(tuple(int(v) for v in values))
            except ValueError:
                continue
    return tuple(directions)


def _spin(value: int, minimum: int, maximum: int) -> QSpinBox:
    box = QSpinBox()
    box.setRange(minimum, maximum)
    box.setValue(value)
    return box


def _float_spin(value: float, decimals: int = 3) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(-100.0, 100.0)
    box.setDecimals(decimals)
    box.setSingleStep(0.1)
    box.setValue(value)
    return box


def _parse_projections(text: str) -> tuple:
    """``"1 2, 3"`` -> ``((1, 2), (3,))``.

    Commas separate projections, spaces group atoms within one — so a DOS "on
    the oxygens" is one entry and "on each of them" is several. Silently drops
    an empty entry so a trailing comma is not an error.
    """
    projections = []
    for chunk in text.split(","):
        atoms = tuple(int(token) for token in chunk.split() if token.strip())
        if atoms:
            projections.append(atoms)
    return tuple(projections)


__all__ = ["PropertiesBuilderDialog"]
