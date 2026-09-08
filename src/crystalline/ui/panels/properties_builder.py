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
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from crystalline.core.crystal_input import suggest_shrink
from crystalline.core.properties_input import (
    BandOptions,
    DossOptions,
    NewkOptions,
    OrbitalsOptions,
    PropertiesInputError,
    PropertiesSpec,
    build_properties_input,
)
from crystalline.core.structure import Structure


class PropertiesBuilderDialog(QDialog):
    """Choose what a PROPERTIES run should compute, and save the deck."""

    def __init__(self, structure: Structure, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Build CRYSTAL properties input (.d3)")
        self.resize(720, 640)
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
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        shrink = suggest_shrink(self._structure) if self._structure.is_periodic else 1

        newk = QGroupBox("NEWK — eigenvectors on a finer mesh")
        newk_form = QFormLayout(newk)
        self._newk_shrink = _spin(shrink, 1, 96)
        self._newk_shrink2 = _spin(shrink * 2, 1, 192)
        newk_form.addRow("Shrinking factor", self._newk_shrink)
        newk_form.addRow("Gilat factor", self._newk_shrink2)
        newk.setToolTip(
            "Written automatically whenever something needs it — DOSS and "
            "ORBITALS both read its eigenvectors."
        )
        layout.addWidget(newk)

        band = QGroupBox("BAND — band structure")
        band.setCheckable(True)
        band.setChecked(True)
        band_form = QFormLayout(band)
        self._band_title = QLineEdit("Band structure")
        self._band_points = _spin(200, 10, 5000)
        self._band_first = _spin(1, 1, 9999)
        self._band_last = _spin(0, 0, 9999)
        self._band_last.setSpecialValueText("auto")
        self._band_last.setToolTip("0 lets the builder pick a range covering the gap.")
        band_form.addRow("Title", self._band_title)
        band_form.addRow("Points along the path", self._band_points)
        band_form.addRow("First band", self._band_first)
        band_form.addRow("Last band", self._band_last)
        self._band_path_label = QLabel()
        self._band_path_label.setWordWrap(True)
        self._band_path_label.setStyleSheet("color: palette(mid);")
        band_form.addRow("Path", self._band_path_label)
        layout.addWidget(band)
        self._band = band

        doss = QGroupBox("DOSS — density of states")
        doss.setCheckable(True)
        doss.setChecked(False)
        doss_form = QFormLayout(doss)
        self._doss_points = _spin(300, 10, 5000)
        self._doss_window = QCheckBox("Give an energy window instead of a band range")
        self._doss_low = _float_spin(-0.7)
        self._doss_high = _float_spin(0.8)
        self._doss_projections = QLineEdit()
        self._doss_projections.setPlaceholderText("e.g.  1, 2   — one projection per atom")
        self._doss_projections.setToolTip(
            "Atoms to project the DOS onto, by CRYSTAL's numbering. Separate "
            "projections with commas; put several atoms in one projection with "
            "spaces, e.g. '1 2, 3'."
        )
        doss_form.addRow("Energy points", self._doss_points)
        doss_form.addRow(self._doss_window)
        window_row = QHBoxLayout()
        window_row.addWidget(self._doss_low)
        window_row.addWidget(QLabel("to"))
        window_row.addWidget(self._doss_high)
        window_row.addWidget(QLabel("hartree"))
        window_row.addStretch(1)
        holder_row = QWidget()
        holder_row.setLayout(window_row)
        window_row.setContentsMargins(0, 0, 0, 0)
        doss_form.addRow("", holder_row)
        doss_form.addRow("Project onto atoms", self._doss_projections)
        layout.addWidget(doss)
        self._doss = doss

        orbitals = QGroupBox("ORBITALS — crystalline orbitals (Molden)")
        orbitals.setCheckable(True)
        orbitals.setChecked(False)
        orbitals.setToolTip(
            "Writes one Molden file per sampled k-point. These are the files "
            "CRYSTALLine's own orbital viewer reads."
        )
        orb_form = QFormLayout(orbitals)
        self._orb_name = QLineEdit("orbitals")
        self._orb_wannier = QCheckBox("Wannier functions (needs LOCALI first)")
        orb_form.addRow("File name", self._orb_name)
        orb_form.addRow(self._orb_wannier)
        layout.addWidget(orbitals)
        self._orbitals = orbitals

        self._ppan = QCheckBox("PPAN — Mulliken population analysis")
        layout.addWidget(self._ppan)

        self._extra = QPlainTextEdit()
        self._extra.setPlaceholderText("Extra PROPERTIES keywords, one per line (optional)")
        self._extra.setFixedHeight(56)
        layout.addWidget(self._extra)
        layout.addStretch(1)

        for widget in (band, doss, orbitals):
            widget.toggled.connect(self._refresh)
        for widget in (self._ppan, self._doss_window, self._orb_wannier):
            widget.toggled.connect(self._refresh)
        for widget in (self._band_title, self._orb_name, self._doss_projections):
            widget.textChanged.connect(self._refresh)
        for widget in (self._newk_shrink, self._newk_shrink2, self._band_points,
                       self._band_first, self._band_last, self._doss_points):
            widget.valueChanged.connect(self._refresh)
        for widget in (self._doss_low, self._doss_high):
            widget.valueChanged.connect(self._refresh)
        self._extra.textChanged.connect(self._refresh)
        return holder

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
            ppan=self._ppan.isChecked(),
            extra_keywords=self._extra.toPlainText(),
        )

    def _refresh(self) -> None:
        for widget in (self._doss_low, self._doss_high):
            widget.setEnabled(self._doss_window.isChecked())
        try:
            text = build_properties_input(self._structure, self.spec())
        except (PropertiesInputError, ValueError) as exc:
            self._preview.setPlainText(f"# {exc}")
            self._band_path_label.setText("")
            return
        self._preview.setPlainText(text)
        for line in text.splitlines()[:2]:
            if "(" in line and line.endswith(")"):
                self._band_path_label.setText(line[line.index("(") + 1:-1])
                break
        else:
            self._band_path_label.setText("")

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
def _spin(value: int, minimum: int, maximum: int) -> QSpinBox:
    box = QSpinBox()
    box.setRange(minimum, maximum)
    box.setValue(value)
    return box


def _float_spin(value: float) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(-100.0, 100.0)
    box.setDecimals(3)
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
