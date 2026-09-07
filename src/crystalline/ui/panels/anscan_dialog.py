"""Choose what to draw on top of an anharmonic scan.

The scan itself — the points, the fitted potential, the levels — is the figure
whatever one asks for; what is optional is everything drawn *on* it. The
wavefunctions and their densities are the reason to open the plot at all, but
they are normalised curves being drawn on an energy axis, so what they are
multiplied by is a setting rather than a fact. It opens on the mean level
spacing of the run, which puts a state at roughly three quarters of the gap
above it — the doublet of a double well being the case that still overlaps.

A dialog rather than a menu entry for the same reason as
:mod:`crystalline.ui.panels.vci_dialog`: the useful figure is not the same for a
stiff single well and a shallow double one.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from crystalline.ui.panels.controls import Section, VALUE_WIDTH, left, slider_row
from crystalline.crystalio.anscan import AnscanRun


def _scale_bounds(value: float) -> tuple:
    """A decade either side of ``value``, so the run's own suggestion is central.

    The scale that suits one mode is nothing like the one that suits another —
    it is set from the level spacing — so a fixed range would put most runs hard
    against a stop. Anchoring the range on the suggestion keeps the useful part
    of the travel where the hand is.
    """
    value = float(value) if value and value > 0 else 1.0
    return (value / 10.0, value * 10.0, value / 10.0)


class AnscanDialog(QDialog):
    """Pick what goes on the scan; read it back with :meth:`options`."""

    def __init__(self, run: AnscanRun, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Anharmonic scan")
        self._run = run

        layout = QVBoxLayout(self)
        summary = QLabel(run.summary, self)
        summary.setWordWrap(True)
        layout.addWidget(summary)

        overlays = Section(layout, "Overlays")

        # A scale factor is judged by ratio — twice as tall, half as tall — so the
        # slider is logarithmic and the run's own suggested value sits mid-track.
        self.wavefunctions = QCheckBox("Wavefunctions", self)
        self.wavefunctions.setChecked(True)
        self.wavefunctions.toggled.connect(self._sync)
        overlays.add_wide(self.wavefunctions)
        self.scale_wf = slider_row(
            overlays, "ψ scale", run.scale_wf,
            *_scale_bounds(run.scale_wf), decimals=1, logarithmic=True,
        )
        self.scale_wf.setToolTip(
            "ψ is multiplied by this before being drawn on its level.\n"
            "The states are normalised, so the factor is what sets their "
            "height in cm⁻¹; it opens on the mean level spacing of the run."
        )

        self.densities = QCheckBox("Probability densities", self)
        self.densities.toggled.connect(self._sync)
        overlays.add_wide(self.densities)
        self.scale_prob = slider_row(
            overlays, "|ψ|² scale", run.scale_prob,
            *_scale_bounds(run.scale_prob), decimals=1, logarithmic=True,
        )
        self.scale_prob.setToolTip("|ψ|² is multiplied by this before being drawn.")

        # Only the states CRYSTAL wrote coefficients for can carry a curve; the
        # levels above them are still drawn, they just have nothing on them.
        self.nstates = QSpinBox(self)
        self.nstates.setRange(1, max(run.nwf, 1))
        self.nstates.setValue(max(run.nwf, 1))
        self.nstates.setFixedWidth(VALUE_WIDTH)
        self.nstates.setToolTip(
            f"Counting up from the ground state. CRYSTAL wrote {run.nwf} "
            "wavefunctions for this run."
        )
        overlays.add("States", left(self.nstates))

        self.harmonic = QCheckBox("Harmonic potential", self)
        self.harmonic.setToolTip(
            "The parabola of the harmonic frequency, for comparison."
            + (" It opens downwards for this mode, which is imaginary."
               if run.imaginary else "")
        )
        overlays.add_wide(self.harmonic)

        self.points = QCheckBox("Scanned points", self)
        self.points.setChecked(True)
        self.points.setToolTip(
            "The energies ANSCAN actually computed, which the potential is a "
            "fit to."
        )
        overlays.add_wide(self.points)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._sync()

    # ── state ───────────────────────────────────────────────────────────
    @staticmethod
    def _setup_scale(spin: QDoubleSpinBox, value: float) -> None:
        """A height box spanning two decades either side of ``value``."""
        spin.setRange(0.0, max(value, 1.0) * 100.0)
        spin.setDecimals(1)
        spin.setSingleStep(max(value, 1.0) / 10.0)
        spin.setValue(value)

    def _sync(self) -> None:
        """A height is only worth setting for a curve that is being drawn."""
        self.scale_wf.setEnabled(self.wavefunctions.isChecked())
        self.scale_prob.setEnabled(self.densities.isChecked())

    def options(self) -> dict:
        """The chosen settings, as :func:`plot_anscan` keyword arguments."""
        return {
            "scale_wf": self.scale_wf.value() if self.wavefunctions.isChecked() else None,
            "scale_prob": self.scale_prob.value() if self.densities.isChecked() else None,
            "harmpot": self.harmonic.isChecked(),
            "scanpot": self.points.isChecked(),
            "nstates": self.nstates.value(),
        }

    def title(self) -> str:
        """A tab title naming the mode that was scanned."""
        return f"ANSCAN mode {self._run.mode}" if self._run.mode else "ANSCAN"


__all__ = ["AnscanDialog"]
