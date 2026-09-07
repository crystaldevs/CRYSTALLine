"""Choose how to draw the VCI wavefunctions of the loaded output.

A VCI run holds one state per configuration — tens of thousands of them for a
crystal — so unlike the elastic surfaces there is no single figure to show: which
states, and how much mixing to keep, decide whether the plot says anything. Hence
a dialog rather than two one-click menu entries, the same reasoning as for
:mod:`crystalline.ui.panels.spectra_dialog`.

The two representations answer different questions, so picking one switches the
options: a coefficient map takes many states and can show the sign of
:math:`A_{n,s}`, a Sankey takes a handful and reads as weights.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from crystalline.ui.panels.controls import Section, range_row, slider_row
from crystalline.crystalio.vci import (
    DEFAULT_THRESHOLD,
    MAX_STATES,
    REPRESENTATIONS,
    VCIRun,
)


class VCIDialog(QDialog):
    """Pick a representation and its options; read them back with :meth:`options`."""

    def __init__(self, run: VCIRun, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("VCI states")
        self._run = run

        layout = QVBoxLayout(self)
        summary = QLabel(run.summary, self)
        summary.setWordWrap(True)
        layout.addWidget(summary)
        if not run.modes:
            # Without the PES scan's mode indices the labels fall back on a dense
            # 1..nmodes numbering, which is not what the Phonons dock shows.
            note = QLabel(
                "The VCI-active modes could not be matched to the phonon "
                "numbering, so configurations are labelled from 1.", self
            )
            note.setWordWrap(True)
            note.setEnabled(False)
            layout.addWidget(note)

        options = Section(layout, "States")
        self.representation = QComboBox(self)
        for label, key in REPRESENTATIONS:
            self.representation.addItem(label, key)
        self.representation.currentIndexChanged.connect(self._sync)
        options.add("Representation", self.representation)

        # A wavenumber window rather than a count: a band one wants to look at is
        # known by where it sits in the spectrum, not by its rank.
        top = run.energies[-1] if run.energies else 4000.0
        default_min, default_max = run.default_window()
        ceiling = max(top, default_max) + 1000.0
        _slider, self.fmin, self.fmax = range_row(
            options, "Window (cm⁻¹)", default_min, default_max, 0.0, ceiling,
            decimals=0, step=50.0, on_change=lambda _a, _b: self._sync_window(),
        )
        self.fmin.setToolTip("Lower edge of the window, as ENE - ZPE.")
        self.fmax.setToolTip("Upper edge of the window, as ENE - ZPE.")

        # A blocked VCI matrix numbers its states within each irrep, and states
        # of different irreps do not mix, so one block at a time is the reading
        # that makes sense — but the low-lying states across all of them is the
        # more usual first look, so that stays the default.
        self.irrep = QComboBox(self)
        self.irrep.addItem("All", None)
        for value in run.irreps:
            self.irrep.addItem(f"Irrep {value}", value)
        self.irrep.setEnabled(bool(run.irreps))
        self.irrep.currentIndexChanged.connect(self._sync_window)
        options.add("Symmetry block", self.irrep)

        # How many states the window catches, live: a run spans thousands, so
        # widening it by a few hundred cm⁻¹ can quietly ask for an unreadable
        # figure. Saying so up front beats failing after the parse.
        self._count = QLabel(self)
        options.add_wide(self._count)

        self.threshold = slider_row(
            options, "Threshold", DEFAULT_THRESHOLD, 0.0, 1.0, 0.005, decimals=3,
        )
        self.threshold.setToolTip(
            "Drop contributions smaller than this.\n"
            "Raise it for a sparser figure, lower it to see weak mixing."
        )

        self._map_box = QGroupBox("Coefficient map", self)
        map_form = QFormLayout(self._map_box)
        self.signed = QCheckBox("Show the sign of the coefficients", self)
        self.signed.setToolTip(
            "Map A(n,s) on a diverging scale instead of its magnitude; the "
            "relative sign of two contributions is physically meaningful."
        )
        map_form.addRow(self.signed)
        self.annotate = QCheckBox("Print values in the cells", self)
        self.annotate.setToolTip("Readable only for a small number of states.")
        map_form.addRow(self.annotate)
        layout.addWidget(self._map_box)

        self._sankey_box = QGroupBox("Sankey", self)
        sankey_form = QFormLayout(self._sankey_box)
        self.weight = QComboBox(self)
        # |A|^2 is the population, which is what a flow width ought to mean; |A|
        # makes the weak mixing visible, which is usually the interesting part.
        self.weight.addItem("Weight |A|²", "square")
        self.weight.addItem("Amplitude |A|", "abs")
        sankey_form.addRow("Ribbon width", self.weight)
        layout.addWidget(self._sankey_box)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._sync()
        self._sync_window()

    # ── state ───────────────────────────────────────────────────────────
    def _sync(self) -> None:
        """Show only the options the chosen representation uses."""
        key = self.representation.currentData()
        self._map_box.setVisible(key == "map")
        self._sankey_box.setVisible(key == "sankey")
        self.adjustSize()

    def _sync_window(self) -> None:
        """Report what the window catches, and refuse the extremes.

        An empty window has nothing to draw; a very wide one would build a
        figure with hundreds of columns, which is not a plot of anything.
        """
        low, high = self.window()
        count = self._run.count_in(low, high, self.irrep.currentData())
        if count == 0:
            self._count.setText("no states in this window")
        elif count > MAX_STATES:
            self._count.setText(f"{count} states — too many, narrow the window")
        else:
            self._count.setText(f"{count} state{'s' if count != 1 else ''} in this window")
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(0 < count <= MAX_STATES)

    def window(self) -> tuple:
        """The chosen window as ``(min, max)``, in ascending order."""
        return tuple(sorted((self.fmin.value(), self.fmax.value())))

    def options(self) -> dict:
        """The chosen settings, as :func:`plot_vci` keyword arguments."""
        return {
            "representation": self.representation.currentData(),
            "frange": self.window(),
            "irrep": self.irrep.currentData(),
            "threshold": self.threshold.value(),
            "signed": self.signed.isChecked(),
            "annotate": self.annotate.isChecked(),
            "weight": self.weight.currentData(),
            "modes": self._run.modes,
        }

    def title(self) -> str:
        """A tab title naming the flavour and the representation."""
        return f"{self._run.label} {self.representation.currentText().lower()}"


__all__ = ["VCIDialog"]
