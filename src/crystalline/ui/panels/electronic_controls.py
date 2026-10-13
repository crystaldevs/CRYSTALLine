"""How a band structure or a DOS is drawn, beside the plot.

The same split as the spectra controls. What the plot *is* — which files, bands
or DOS or both, which projections and what they are called, the names of the
path's corners — is decided in the dialog, which reads the files to offer it.
What is judged by looking goes here, where it can be judged: the energy window
first of all, pulled until the bands that matter fill the axis, then the DOS
axis, how a spin-down DOS is laid out, the line widths, the Fermi line and the
legend.

The unit and the reference stay in the dialog with the colours. They are the
frame the window's numbers are written in, chosen once before reading the plot,
and carrying a window across a change of frame needs the files the dialog has
read.

Alongside the plot's own options these controls need the tracks the dialog
offered — how far the window could go, how tall the DOS axis — which travel in
the same dictionary under :data:`EXTENT_KEYS` and are left out of the plot.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QSizePolicy,
                               QVBoxLayout, QWidget)

from crystalline.crystalio import electronic as el
from crystalline.ui.panels.controls import (ColourButton, Section, range_row, set_span,
                                            slider_row)

#: What the dialog's tracks were, carried with a plot's options for these
#: controls and not handed to the plotter.
EXTENT_KEYS = ("energy_bounds", "dos_bounds", "dos_handles", "dos_polarised",
               "bands_polarised", "dos_projections")

#: The plot's own files, likewise carried with its options.
PATH_KEYS = ("bands_path", "dos_path")


def plot_arguments(options: dict) -> tuple:
    """``(bands_path, dos_path, ElectronicOptions)`` from a plot's options."""
    fields = {key: value for key, value in options.items()
              if key not in EXTENT_KEYS + PATH_KEYS}
    return options.get("bands_path"), options.get("dos_path"), el.ElectronicOptions(**fields)


# Measured, not chosen: drawn at the size the Plots window opens at, the corners
# of MgO's and silicon's band paths ran into each other along the bottom at
# 4:3, 1.6 and 16:9, and came clear at 2:1. The DOS panel takes its share of the
# width out of the bands', which is what a single band structure never pays.
BANDS_AND_DOS_ASPECT = 2.0


def aspect(options: dict) -> Optional[float]:
    """The figure's shape, width over height: wide for bands beside a DOS.

    ``None`` for the rest, which take the window's own.
    """
    return BANDS_AND_DOS_ASPECT if options.get("mode") == el.BANDS_AND_DOS else None


def title(options: dict) -> str:
    """A tab title naming what is drawn: bands, a DOS, or both."""
    return el.plot_title(options.get("mode", el.BANDS))


class ElectronicControls(QWidget):
    """Window, DOS axis and line styling, with a signal when any of them moves."""

    #: Any control was touched; the reader asks :meth:`options` for all of it.
    changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # No stretch of its own, as with the other plot controls.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        announce = lambda *_args: self.changed.emit()  # noqa: E731

        energy = Section(layout, "Energy")
        # The track is the dialog's, put here by set_options: it spans the
        # files, which these controls never read.
        self._energy = range_row(energy, "Window", -10.0, 10.0, -30.0, 30.0,
                                 decimals=2, step=0.5, on_change=announce)
        self._energy[1].setToolTip("Bottom of the energy window, in the plot's unit "
                                   "and reference.")
        self._energy[2].setToolTip("Top of the energy window.")

        dos = Section(layout, "Density of states")
        self.dos_auto = QCheckBox("Fit the DOS axis to what is drawn")
        self.dos_auto.setChecked(True)
        self.dos_auto.toggled.connect(self._on_auto)
        dos.add_wide(self.dos_auto)
        self._dos_span = range_row(dos, "DOS range", 0.0, 1.0, 0.0, 10.0,
                                   decimals=2, step=0.1, on_change=announce)
        self._dos_range_row = dos.row_widgets("DOS range")
        self.spin_down = QComboBox()
        self.spin_down.addItem("Mirrored below the axis", el.SPIN_MIRRORED)
        self.spin_down.addItem("On the same side as spin up", el.SPIN_ALONGSIDE)
        self.spin_down.currentIndexChanged.connect(announce)
        dos.add("Spin down", self.spin_down)
        self._spin_row = dos.row_widgets("Spin down")
        self.overlay = QCheckBox("Draw the projections on one axis")
        self.overlay.setChecked(True)
        self.overlay.setToolTip("On one axis with a legend, or one panel each.")
        self.overlay.toggled.connect(announce)
        # Split into panels, a spin-polarised DOS gains an α/β legend.
        self.overlay.toggled.connect(lambda _on: self._sync_legend())
        dos.add_wide(self.overlay)
        self._dos_section = dos

        # Folded until wanted: what is looked for beside a plot is the window
        # and the DOS axis, and the look is settled once. The colours are the
        # dialog's, in the same rows, so a plot can be recoloured without
        # being made again from it.
        look = Section(layout, "Appearance", collapsed=True)
        self.band_colour = ColourButton(el.BAND_COLOUR, "Band colour", self)
        self.band_colour.colourChanged.connect(announce)
        look.add("Bands", self.band_colour)
        self._band_row = look.row_widgets("Bands")
        self.beta_colour = ColourButton(el.BETA_COLOUR, "Spin-down band colour", self)
        self.beta_colour.colourChanged.connect(announce)
        look.add("Spin-down bands", self.beta_colour)
        self._beta_row = look.row_widgets("Spin-down bands")
        self.show_fermi = QCheckBox("Draw")
        self.show_fermi.setChecked(True)
        self.show_fermi.toggled.connect(announce)
        self.fermi_colour = ColourButton(el.FERMI_COLOUR, "Fermi level colour", self)
        self.fermi_colour.colourChanged.connect(announce)
        look.add("Fermi level", _row(self.show_fermi, self.fermi_colour))
        self.linewidth = slider_row(look, "Line width", 1.0, 0.3, 4.0, 0.1,
                                    decimals=1, on_change=announce)
        # As on every other plot: named by default, off when the box gets in
        # the way of what it names.
        self.legend = QCheckBox("Legend")
        self.legend.setChecked(True)
        self.legend.toggled.connect(announce)
        look.add_wide(self.legend)

        self._sync_dos_range()
        self._plot: dict = {}

    # ── state ───────────────────────────────────────────────────────────
    def options(self) -> dict:
        """What these controls decide, as keys of the plot's options."""
        _slider, low, high = self._energy
        low, high = sorted((low.value(), high.value()))
        _slider, dlow, dhigh = self._dos_span
        dlow, dhigh = sorted((dlow.value(), dhigh.value()))
        return {
            # A zero-width window means "don't clip", as in the dialog.
            "energy_range": None if low == high else (low, high),
            "dos_range": None if self.dos_auto.isChecked() or dlow == dhigh else (dlow, dhigh),
            "dos_handles": (dlow, dhigh),
            "spin_down": self.spin_down.currentData(),
            "overlay_projections": self.overlay.isChecked(),
            "linewidth": self.linewidth.value(),
            "show_fermi": self.show_fermi.isChecked(),
            "band_colour": self.band_colour.colour(),
            "beta_colour": self.beta_colour.colour(),
            "fermi_colour": self.fermi_colour.colour(),
            "legend": self.legend.isChecked(),
        }

    def set_options(self, options: Optional[dict]) -> None:
        """Show a plot's options without announcing them: it already has them."""
        if not options:
            return
        blocked = self.blockSignals(True)
        try:
            bounds = options.get("energy_bounds")
            if bounds:
                window = options.get("energy_range") or bounds
                set_span(self._energy, min(bounds[0], *window), max(bounds[1], *window),
                         *sorted(window))
            bounds = options.get("dos_bounds")
            if bounds:
                handles = options.get("dos_range") or options.get("dos_handles") or bounds
                set_span(self._dos_span, min(bounds[0], *handles), max(bounds[1], *handles),
                         *sorted(handles))
            self.dos_auto.setChecked(options.get("dos_range") is None)
            index = self.spin_down.findData(options.get("spin_down"))
            if index >= 0:
                self.spin_down.setCurrentIndex(index)
            for name, key in (("overlay", "overlay_projections"),
                              ("show_fermi", "show_fermi"), ("legend", "legend")):
                if options.get(key) is not None:
                    getattr(self, name).setChecked(bool(options[key]))
            if options.get("linewidth") is not None:
                self.linewidth.setValue(float(options["linewidth"]))
            for name in ("band_colour", "beta_colour", "fermi_colour"):
                if options.get(name):
                    getattr(self, name).setColour(options[name])
        finally:
            self.blockSignals(blocked)
        self._show_what_applies(options)
        self._sync_dos_range()

    def _show_what_applies(self, options: dict) -> None:
        """Hide what this plot has no use for, as the other controls do.

        A band structure has no DOS axis; a spin-restricted DOS has no spin-down
        part; and the band+DOS view always draws its projections together, so
        only a DOS on its own can split them into panels.
        """
        mode = options.get("mode", el.BANDS)
        self._dos_section.set_visible(mode in (el.DOS, el.BANDS_AND_DOS))
        bands = mode in (el.BANDS, el.BANDS_AND_DOS)
        for widget in self._band_row:
            widget.setVisible(bands)
        for widget in self._beta_row:
            widget.setVisible(bands and bool(options.get("bands_polarised")))
        for widget in self._spin_row:
            widget.setVisible(bool(options.get("dos_polarised")))
        self.overlay.setVisible(mode == el.DOS)
        self._plot = dict(options)
        self._sync_legend()

    def _sync_legend(self) -> None:
        """Offer a legend only where one tells something apart.

        Not for a single band structure, nor for one projection: the plotter
        draws none there whatever is asked (see legend_names_anything).
        """
        plot = self._plot
        chosen = plot.get("projections")
        self.legend.setVisible(el.legend_names_anything(
            plot.get("mode", el.BANDS),
            len(chosen) if chosen else int(plot.get("dos_projections") or 0),
            bool(plot.get("bands_polarised")), bool(plot.get("dos_polarised")),
            self.overlay.isChecked()))

    def _on_auto(self, _on: bool) -> None:
        self._sync_dos_range()
        self.changed.emit()

    def _sync_dos_range(self) -> None:
        """A DOS range is only worth setting when it is not being fitted."""
        for widget in self._dos_range_row:
            widget.setEnabled(not self.dos_auto.isChecked())


def _row(*widgets) -> QWidget:
    """Controls side by side at the left of a row, as the dialog sets them."""
    holder = QWidget()
    line = QHBoxLayout(holder)
    line.setContentsMargins(0, 0, 0, 0)
    line.setSpacing(8)
    for widget in widgets:
        line.addWidget(widget, 0)
    line.addStretch(1)
    return holder


__all__ = ["BANDS_AND_DOS_ASPECT", "EXTENT_KEYS", "ElectronicControls", "PATH_KEYS",
           "aspect", "plot_arguments", "title"]
