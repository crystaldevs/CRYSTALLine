"""How a band structure or a DOS is drawn, beside the plot.

The controls every band plot has (:mod:`~crystalline.ui.panels.bands_controls`),
for energies: the window in the plot's unit and reference, the Fermi level as
the reference line, and the spin — how a spin-down DOS is laid out, the colour
of the spin-down bands.

The unit and the reference stay in the dialog with the colours. They are the
frame the window's numbers are written in, chosen once before reading the plot,
and carrying a window across a change of frame needs the files the dialog has
read.
"""

from __future__ import annotations

from PySide6.QtWidgets import QComboBox

from crystalline.crystalio import electronic as el
from crystalline.ui.panels.bands_controls import (BANDS_AND_DOS_ASPECT, PATH_KEYS,
                                                  BandsControls, aspect, split_options)
from crystalline.ui.panels.controls import ColourButton

#: What the dialog's tracks were, carried with a plot's options for these
#: controls and not handed to the plotter.
EXTENT_KEYS = ("energy_bounds", "dos_bounds", "dos_handles", "dos_polarised",
               "bands_polarised", "dos_projections")


def plot_arguments(options: dict) -> tuple:
    """``(bands_path, dos_path, ElectronicOptions)`` from a plot's options."""
    return split_options(options, EXTENT_KEYS, el.ElectronicOptions)


def title(options: dict) -> str:
    """A tab title naming what is drawn: bands, a DOS, or both."""
    return el.plot_title(options.get("mode", el.BANDS))


class ElectronicControls(BandsControls):
    """Energy window, DOS axis, spin and line styling."""

    WINDOW_SECTION = "Energy"
    WINDOW_KEY, BOUNDS_KEY = "energy_range", "energy_bounds"
    WINDOW_TRACK = (-30.0, 30.0, -10.0, 10.0, 2, 0.5)
    LINE_KEYS = ("show_fermi", "fermi_colour")
    LINE_LABEL = "Fermi level"
    LINE_COLOUR = el.FERMI_COLOUR

    def _build_dos_extras(self, dos) -> None:
        self.spin_down = QComboBox()
        self.spin_down.addItem("Mirrored below the axis", el.SPIN_MIRRORED)
        self.spin_down.addItem("On the same side as spin up", el.SPIN_ALONGSIDE)
        self.spin_down.currentIndexChanged.connect(self._announce)
        dos.add("Spin down", self.spin_down)
        self._spin_row = dos.row_widgets("Spin down")

    def _build_lines(self, look) -> None:
        self.beta_colour = ColourButton(el.BETA_COLOUR, "Spin-down band colour", self)
        self.beta_colour.colourChanged.connect(self._announce)
        look.add("Spin-down bands", self.beta_colour)
        self._beta_row = look.row_widgets("Spin-down bands")

    def options(self) -> dict:
        return {**super().options(), "spin_down": self.spin_down.currentData(),
                "beta_colour": self.beta_colour.colour()}

    def _set_extras(self, options: dict) -> None:
        index = self.spin_down.findData(options.get("spin_down"))
        if index >= 0:
            self.spin_down.setCurrentIndex(index)
        if options.get("beta_colour"):
            self.beta_colour.setColour(options["beta_colour"])

    def _show_what_applies(self, options: dict) -> None:
        """A spin-restricted run has no spin-down part to lay out or colour."""
        bands = options.get("mode", el.BANDS) in (el.BANDS, el.BANDS_AND_DOS)
        for widget in self._beta_row:
            widget.setVisible(bands and bool(options.get("bands_polarised")))
        for widget in self._spin_row:
            widget.setVisible(bool(options.get("dos_polarised")))
        super()._show_what_applies(options)


__all__ = ["BANDS_AND_DOS_ASPECT", "EXTENT_KEYS", "ElectronicControls", "PATH_KEYS",
           "aspect", "plot_arguments", "title"]
