"""What is drawn on an anharmonic scan, as a widget rather than a dialog.

The same split as the spectra and PES controls, except that here nothing is
left to choose in the dialog: the scan is one mode of one run, so the figure
*is* the run, and everything optional is drawn on top of it. All of it is
judged by looking — a wavefunction scale is right when the states sit between
their neighbours, not at any number — so all of it goes beside the plot, where
the scale can be pulled until they do.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QSizePolicy, QSpinBox, QVBoxLayout, QWidget

from crystalline.crystalio.anscan import AnscanRun
from crystalline.crystalio.pes import DEFAULT_LINEWIDTH
from crystalline.ui.panels.controls import (Section, left, scale_bounds,
                                            set_value_width, slider_row)


class AnscanOptions(QWidget):
    """The overlays of a scan, and a signal when any of them moves."""

    #: Any control was touched; the reader asks :meth:`options` for all of it.
    changed = Signal()

    def __init__(self, run: AnscanRun, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # No stretch of its own, as with the other plot controls: beside a plot
        # the tab puts one under the rows, and in the dialog there is nothing
        # else to give the height to.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        overlays = Section(layout, "Overlays")

        # A scale factor is judged by ratio — twice as tall, half as tall — so the
        # slider is logarithmic and the run's own suggested value sits mid-track.
        self.wavefunctions = QCheckBox("Wavefunctions")
        self.wavefunctions.setChecked(True)
        self.wavefunctions.toggled.connect(self._on_curve)
        overlays.add_wide(self.wavefunctions)
        self.scale_wf = slider_row(
            overlays, "ψ scale", run.scale_wf,
            *scale_bounds(run.scale_wf), decimals=1, logarithmic=True,
            on_change=lambda _value: self.changed.emit(),
        )
        self.scale_wf.setToolTip(
            "ψ is multiplied by this before being drawn on its level.\n"
            "The states are normalised, so the factor is what sets their "
            "height in cm⁻¹; it opens on the mean level spacing of the run."
        )

        self.densities = QCheckBox("Probability densities")
        self.densities.toggled.connect(self._on_curve)
        overlays.add_wide(self.densities)
        self.scale_prob = slider_row(
            overlays, "|ψ|² scale", run.scale_prob,
            *scale_bounds(run.scale_prob), decimals=1, logarithmic=True,
            on_change=lambda _value: self.changed.emit(),
        )
        self.scale_prob.setToolTip("|ψ|² is multiplied by this before being drawn.")

        # Only the states CRYSTAL wrote coefficients for can carry a curve; the
        # levels above them are still drawn, they just have nothing on them.
        self.nstates = QSpinBox()
        self.nstates.setRange(1, max(run.nwf, 1))
        self.nstates.setValue(max(run.nwf, 1))
        set_value_width(self.nstates)
        self.nstates.setToolTip(
            f"Counting up from the ground state. CRYSTAL wrote {run.nwf} "
            "wavefunctions for this run."
        )
        self.nstates.valueChanged.connect(lambda _n: self.changed.emit())
        overlays.add("States", left(self.nstates))

        self.harmonic = QCheckBox("Harmonic potential")
        self.harmonic.setToolTip(
            "The parabola of the harmonic frequency, for comparison."
            + (" It opens downwards for this mode, which is imaginary."
               if run.imaginary else "")
        )
        self.harmonic.toggled.connect(lambda _on: self.changed.emit())
        overlays.add_wide(self.harmonic)

        self.points = QCheckBox("Scanned points")
        self.points.setChecked(True)
        self.points.setToolTip(
            "The energies ANSCAN actually computed, which the potential is a "
            "fit to."
        )
        self.points.toggled.connect(lambda _on: self.changed.emit())
        overlays.add_wide(self.points)

        # How it is drawn, folded as on every other plot.
        look = Section(layout, "Appearance", collapsed=True)
        self.linewidth = slider_row(
            look, "Line width", DEFAULT_LINEWIDTH, 0.3, 4.0, 0.1, decimals=1,
            on_change=lambda _value: self.changed.emit(),
        )
        self.linewidth.setToolTip(
            "The potential's width, in points; the states are drawn finer, "
            "in proportion."
        )
        # As on a spectrum and on a PES cut: named by default, off when the
        # box gets in the way of what it names.
        self.legend = QCheckBox("Legend")
        self.legend.setChecked(True)
        self.legend.setToolTip(
            "Name the curves drawn on the scan: the potential, the harmonic\n"
            "parabola, and the states if they are shown."
        )
        self.legend.toggled.connect(lambda _on: self.changed.emit())
        look.add_wide(self.legend)

        self._sync()

    # ── state ───────────────────────────────────────────────────────────
    def options(self) -> dict:
        """What this widget decides, as :func:`plot_anscan` keyword arguments."""
        return {
            "legend": self.legend.isChecked(),
            "scale_wf": self.scale_wf.value() if self.wavefunctions.isChecked() else None,
            "scale_prob": self.scale_prob.value() if self.densities.isChecked() else None,
            "harmpot": self.harmonic.isChecked(),
            "scanpot": self.points.isChecked(),
            "nstates": self.nstates.value(),
            "linewidth": self.linewidth.value(),
        }

    def set_options(self, options: Optional[dict]) -> None:
        """Show ``options`` without announcing them: the caller already has them.

        A curve is drawn when its scale is given and left off when it is
        ``None``, so the scale says both whether and how tall; a missing scale
        leaves the slider where it was, ready for the curve to come back.
        """
        if not options:
            return
        blocked = self.blockSignals(True)
        try:
            for curve, box, key in ((self.wavefunctions, self.scale_wf, "scale_wf"),
                                    (self.densities, self.scale_prob, "scale_prob")):
                if key in options:
                    curve.setChecked(options[key] is not None)
                    if options[key] is not None:
                        box.setValue(float(options[key]))
            if options.get("nstates") is not None:
                self.nstates.setValue(int(options["nstates"]))
            if options.get("linewidth") is not None:
                self.linewidth.setValue(float(options["linewidth"]))
            for name, key in (("harmonic", "harmpot"), ("points", "scanpot"),
                              ("legend", "legend")):
                if options.get(key) is not None:
                    getattr(self, name).setChecked(bool(options[key]))
        finally:
            self.blockSignals(blocked)
        self._sync()

    def _on_curve(self, _on: bool) -> None:
        self._sync()
        self.changed.emit()

    def _sync(self) -> None:
        """A scale is only worth setting for a curve that is being drawn."""
        self.scale_wf.setEnabled(self.wavefunctions.isChecked())
        self.scale_prob.setEnabled(self.densities.isChecked())


__all__ = ["AnscanOptions"]
