"""Choose which crystalline orbital to draw, and how.

An ORBITALS run writes one Molden file per sampled k-point, each holding every
orbital of that k — 180 of them for a small oxide — so the choice is a k-point,
then an orbital within it, then how far down the amplitude to cut the surface.

The orbital list is the informative part: energy and occupation per row, with
the HOMO and LUMO marked, since "the orbital just below the gap" is what someone
is usually looking for and counting to it by hand is the tedious way.

Away from Γ the same thing is true of a crystalline orbital as of a phonon away
from the zone centre: it is a wave over the lattice, and a single cell is a
snapshot of it with nothing to compare against. So this dialog offers what the
phonon panel offers — the tiling in which one whole period of k fits.

What it draws by default is the signed amplitude Re(ψ), the picture an orbital
is usually wanted for; its lobes do carry the phase from cell to cell, as a
change of shape. The phase view under **Show** makes that relation explicit
instead, for when it is the thing being looked at.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from crystalline.core import orbitals
from crystalline.crystalio import molden
from crystalline.viz.renderer import DEFAULT_ORBITAL_ISOVALUE

# Hartree to electronvolt, so the list reads in the unit a band structure is
# quoted in; the file itself stores Hartree.
_HARTREE_TO_EV = 27.211386245988

# Grid density along each axis. The cost is cubic, so the range is deliberately
# short: 60 is a couple of seconds and enough to see the lobes, 120 is for a
# figure and takes appreciably longer.
_SAMPLES = (40, 60, 80, 100, 120)
_DEFAULT_SAMPLES = 60


class OrbitalDialog(QDialog):
    """Pick a k-point, an orbital and an isosurface level."""

    def __init__(self, paths: Sequence[str], parent: Optional[QWidget] = None,
                 view_cell=None) -> None:
        """``view_cell`` is the unit cell the view will show the orbital on.

        Which supercell one period of k spans depends on it, not on k alone —
        the crystallographic cell of a rhombohedral crystal already holds three
        primitive ones — so without it the dialog could only offer an unnamed
        "tile", and could not say when there is nothing to tile to.
        """
        super().__init__(parent)
        self.setWindowTitle("Crystalline orbitals")
        self.resize(460, 600)
        self._paths = list(paths)
        self._data: Optional[molden.MoldenOrbitals] = None
        self._view_cell = None if view_cell is None else np.asarray(view_cell, dtype=float)
        # What the user asked for, as against what this k-point can offer. Γ has
        # no phase to draw and no period to tile to, so both controls are forced
        # there — and moving off Γ has to restore the choice rather than leave
        # the forced value standing, which is what made picking a k-point look
        # like it had turned the phase view off.
        self._draw_wanted = orbitals.AMPLITUDE
        self._tile_wanted = True
        self._syncing = False

        layout = QVBoxLayout(self)

        # ── which k-point ───────────────────────────────────────────────
        form = QFormLayout()
        self.kpoint = QComboBox()
        self._kpoints: dict = {}
        for path in self._real_files():
            vector = molden.k_vector(path)
            self._kpoints[path] = vector
            complex_too = molden.companion_part(path) is not None
            self.kpoint.addItem(
                _kpoint_text(path, vector) + ("" if not complex_too else "   (complex)"),
                path,
            )
        self.kpoint.setToolTip(
            "Which sampled k-point's orbitals to list.\n\n"
            "k is read from the CRYSTAL output's k-point table — the file names\n"
            "carry only the integer coordinates, not the shrinking factor they\n"
            "are in units of."
        )
        form.addRow("k-point", self.kpoint)
        layout.addLayout(form)

        self._note = QLabel()
        self._note.setWordWrap(True)
        self._note.setStyleSheet("color: palette(mid);")
        layout.addWidget(self._note)

        # ── which orbital ───────────────────────────────────────────────
        layout.addWidget(QLabel("Orbital — energy and occupation:"))
        self.orbitals = QListWidget()
        layout.addWidget(self.orbitals, 1)

        # ── how to draw it ──────────────────────────────────────────────
        group = QGroupBox("Isosurface")
        options = QFormLayout(group)
        self.isovalue = QDoubleSpinBox()
        self.isovalue.setRange(0.01, 0.95)
        self.isovalue.setSingleStep(0.05)
        self.isovalue.setDecimals(2)
        self.isovalue.setValue(DEFAULT_ORBITAL_ISOVALUE)
        self.isovalue.setToolTip(
            "How much of the orbital the surface encloses.\n\n"
            "Low draws a large surface out in the orbital's faint tails; high draws a\n"
            "tight one around only its strongest regions, which is what separates the\n"
            "individual lobes.\n\n"
            "Measured against the orbital's own strongest point rather than as an\n"
            "absolute amplitude, because a crystalline orbital is spread over a whole\n"
            "cell — a fixed value would swamp one orbital and show nothing of the next."
        )
        options.addRow("Surface level", self.isovalue)

        self.samples = QComboBox()
        for value in _SAMPLES:
            self.samples.addItem(f"{value} × {value} × {value} per cell", value)
        self.samples.setCurrentIndex(_SAMPLES.index(_DEFAULT_SAMPLES))
        self.samples.setToolTip(
            "Grid density, counted across one cell. The cost grows as the cube of\n"
            "it, and a tiled view is sampled as finely per cell as a single one —\n"
            "so tiling costs what the extra cells are worth, rather than quietly\n"
            "spreading the same points thinner."
        )
        options.addRow("Grid", self.samples)

        # Only meaningful away from Γ, where the orbital is complex; enabled by
        # _on_kpoint_changed when the chosen k has a companion file.
        self.draw = QComboBox()
        self.draw.addItem("Amplitude — Re(ψ), ± lobes", orbitals.AMPLITUDE)
        self.draw.addItem("Phase — |ψ|, coloured per cell", orbitals.PHASE)
        self.draw.addItem("Modulus — |ψ|, one colour", orbitals.MODULUS)
        self.draw.setToolTip(
            "Away from Γ a crystalline orbital is complex, and there is no one\n"
            "picture of it.\n\n"
            "Amplitude draws a signed section Re(ψ) — the familiar ± lobes, and the\n"
            "picture an orbital is usually wanted for. The phase relation is in there,\n"
            "as a different shape from one cell to the next, but it takes reading.\n\n"
            "Phase draws |ψ|, which is identical in every cell, and colours each\n"
            "cell by its Bloch phase e^(2πi k·T) — so the colour is the only thing\n"
            "that changes from cell to cell, and it changes by exactly that. It is\n"
            "the same device the phonon panel uses to colour a mode's arrows away\n"
            "from Γ, and the one to reach for when the relation *is* the question.\n\n"
            "Modulus draws |ψ| with no colouring: every cell is identical and the\n"
            "phase is gone. The right choice when the density, not the phase, is the\n"
            "question."
        )
        options.addRow("Show", self.draw)

        self.phase = QDoubleSpinBox()
        self.phase.setRange(0.0, 360.0)
        self.phase.setSingleStep(15.0)
        self.phase.setDecimals(0)
        self.phase.setSuffix("°")
        self.phase.setWrapping(True)
        self.phase.setToolTip(
            "Which section of the complex orbital to draw: Re(ψ·e^(iθ)).\n\n"
            "The global phase of a crystalline orbital is a convention — CRYSTAL's\n"
            "choice, not physics — so no one section is *the* real part. Turning this\n"
            "steps through the family, as a phonon animation steps through its cycle.\n"
            "The relation *between* cells is fixed by k and does not move with it."
        )
        options.addRow("Phase", self.phase)
        layout.addWidget(group)

        # ── tiling ──────────────────────────────────────────────────────
        # The same offer the phonon panel makes for a mode away from Γ, and for
        # the same reason: a phase relation needs two cells to be a relation.
        tiling = QHBoxLayout()
        self.tile = QCheckBox("Tile to one whole period of this k")
        self.tile.setChecked(True)   # a relation needs two cells to be a relation
        self.tile.setToolTip(
            "Show the smallest supercell in which one whole period of this k fits,\n"
            "the way the phonon panel's Tile does for a mode at the same q.\n\n"
            "On by default: one cell draws the orbital correctly but gives nothing to\n"
            "compare it against, so nothing about the phase can be seen in it.\n\n"
            "How many cells that is depends on the cell being shown, not only on k —\n"
            "the crystallographic cell of a rhombohedral crystal already holds three\n"
            "primitive ones, and can already span a whole period on its own."
        )
        tiling.addWidget(self.tile)
        self._tile_note = QLabel()
        self._tile_note.setStyleSheet("color: palette(mid);")
        tiling.addWidget(self._tile_note, 1)
        layout.addLayout(tiling)

        self._buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self.kpoint.currentIndexChanged.connect(self._on_kpoint_changed)
        self.orbitals.itemSelectionChanged.connect(self._sync_ok)
        # These record the *user's* intent. They fire for a restored setting too
        # — which is the point, a remembered choice is still a choice — but not
        # for this dialog's own forcing, which brackets itself with _syncing.
        self.draw.currentIndexChanged.connect(self._remember_draw)
        self.tile.toggled.connect(self._remember_tile)
        self._on_kpoint_changed()

    # ── contents ────────────────────────────────────────────────────────
    def _real_files(self) -> List[str]:
        """One entry per k-point: the ``_real`` file, which every k has.

        The ``_complex`` files are the imaginary halves of the same orbitals, not
        k-points of their own, so they are not offered as separate choices.
        """
        return [p for p in self._paths if molden.companion_part(p) is not None
                or "_complex" not in os.path.basename(p).lower()]

    def _on_kpoint_changed(self) -> None:
        """Load the chosen file and list its orbitals."""
        path = self.kpoint.currentData()
        self.orbitals.clear()
        self._data = None
        if not path:
            self._sync_ok()
            return
        try:
            self._data = molden.load(path)
        except Exception as exc:  # noqa: BLE001 - an unreadable file must not kill the dialog
            self._note.setText(f"Could not read this file: {exc}")
            self._sync_ok()
            return

        data = self._data
        has_imaginary = molden.companion_part(path) is not None
        for widget in (self.draw, self.phase):
            widget.setEnabled(has_imaginary)
        wanted = self._draw_wanted if has_imaginary else orbitals.AMPLITUDE
        self._force(self.draw.setCurrentIndex, self.draw.findData(wanted))
        if not has_imaginary:
            self._force(self.phase.setValue, 0.0)
        vector = self._kpoints.get(path)
        if has_imaginary and vector is None:
            # Without k there is no phase to apply, and summing the images in
            # phase anyway would draw the Γ orbital of these coefficients — a
            # different orbital that looks perfectly plausible. Say so.
            detail = "  ·  k unknown (no CRYSTAL output beside the files): drawn at Γ"
        elif has_imaginary:
            detail = "  ·  complex: the ± lobes carry the phase from cell to cell"
        else:
            detail = "  ·  real at Γ, drawn with ± lobes"
        self._note.setText(
            f"{len(data.orbitals)} orbitals on {len(data.numbers)} atoms" + detail
        )
        self._update_tiling(vector)

        homo = data.homo_index
        for i, orbital in enumerate(data.orbitals):
            mark = ""
            if homo is not None and i == homo:
                mark = "   ← HOMO"
            elif homo is not None and i == homo + 1:
                mark = "   ← LUMO"
            item = QListWidgetItem(
                f"{i + 1:4d}   {orbital.energy * _HARTREE_TO_EV:10.3f} eV"
                f"   occ {orbital.occupation:.2f}{mark}"
            )
            item.setData(Qt.UserRole, i)
            self.orbitals.addItem(item)
        if homo is not None:
            self.orbitals.setCurrentRow(homo)  # the interesting one, by default
        self._sync_ok()

    def _remember_draw(self, _index: int) -> None:
        if not self._syncing:
            self._draw_wanted = self.draw.currentData()

    def _remember_tile(self, checked: bool) -> None:
        if self._syncing:
            return
        self._tile_wanted = bool(checked)
        if checked and not self.tile.isEnabled():
            # Remembered from a k-point that had a period, ticked onto one that
            # has none. Keep the intent, but don't show a tick that does nothing.
            self._force(self.tile.setChecked, False)

    def _force(self, setter, value) -> None:
        """Set a control from this dialog rather than from the user."""
        self._syncing = True
        try:
            setter(value)
        finally:
            self._syncing = False

    def _update_tiling(self, vector) -> None:
        """Offer the tiling, and name it, only where there is a period to show.

        Three ways there can be nothing to offer, and each is worth saying out
        loud rather than leaving as a box that does nothing when ticked:

        * Γ, where every cell carries the same orbital;
        * an unknown k, where the phase is not known to be applied at all;
        * a k whose period the *displayed* cell already spans — k = (1/3, 1/3,
          1/3) of a rhombohedral cell is a zone-centre point of the hexagonal
          one, so the crystallographic view of it repeats every single cell and
          no amount of tiling will show a relation.
        """
        reps = self.repeat(vector)
        available = reps != (1, 1, 1)
        self.tile.setEnabled(available)
        if available:
            self._force(self.tile.setChecked, self._tile_wanted)
            self._tile_note.setText("{}×{}×{} cells".format(*reps))
            return
        self._force(self.tile.setChecked, False)
        if vector is None:
            note = "k unknown — nothing to tile to"
        elif not np.any(vector):
            note = "one cell shows the whole orbital at Γ"
        else:
            note = "the cell on screen already spans a whole period of this k"
        self._tile_note.setText(note)

    def repeat(self, vector) -> tuple:
        """The tiling of the displayed cell that one period of ``vector`` needs."""
        if vector is None or self._data is None:
            return (1, 1, 1)
        return orbitals.commensurate_repeats_in(vector, self._data.cell, self._view_cell)

    def _sync_ok(self) -> None:
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(
            self._data is not None and self.orbitals.currentItem() is not None
        )

    # ── result ──────────────────────────────────────────────────────────
    def selection(self) -> dict:
        """The chosen file, orbital and drawing options.

        ``imaginary_path`` is the companion file holding the imaginary part, or
        ``None`` at Γ where the orbital is real. It is needed for *either* view
        of a complex orbital — the modulus and the signed amplitude are both
        built from the two halves — so, unlike before, it is not withheld when
        the amplitude is asked for.

        ``tile`` asks for the supercell holding one whole period of ``kpoint``;
        the caller applies it, having the live view to apply it to.
        """
        path = self.kpoint.currentData()
        item = self.orbitals.currentItem()
        vector = self._kpoints.get(path)
        return {
            "path": path,
            "index": int(item.data(Qt.UserRole)) if item is not None else 0,
            "isovalue": float(self.isovalue.value()),
            "samples": int(self.samples.currentData()),
            "imaginary_path": molden.companion_part(path) if path else None,
            "kpoint": vector,
            "draw": self.draw.currentData(),
            "phase": np.radians(float(self.phase.value())),
            "tile": bool(self.tile.isChecked() and self.tile.isEnabled()),
        }


def _kpoint_text(path: str, vector) -> str:
    """How a k-point reads in the combo: ``Γ``, ``k = (2/3, 1/3, 0)``, or the digits.

    The vector is the useful form — it is what the phase and the tiling are
    computed from — so it is shown when the output gave one, and the file's own
    label only when it did not.
    """
    label = molden.k_label(path) or os.path.basename(path)
    if vector is not None and not np.any(vector):
        return "Γ"
    if vector is None:
        return f"k[{label}]"
    return "k = (" + ", ".join(_fraction(component) for component in vector) + ")"


def _fraction(value: float, limit: int = 24) -> str:
    """``"2/3"`` rather than ``"0.667"``: a k component is a simple fraction."""
    from fractions import Fraction

    ratio = Fraction(float(value)).limit_denominator(limit)
    return str(ratio.numerator) if ratio.denominator == 1 else f"{ratio.numerator}/{ratio.denominator}"


__all__ = ["OrbitalDialog"]
