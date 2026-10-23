"""What the controls beside an electronic and a phonon band plot have in common.

The same split as the spectra controls. What the plot *is* — which files, bands
or DOS or both, which projections and what they are called, the names of the
path's corners, the units — is decided in the dialog, which reads the files to
offer it. What is judged by looking goes here, where it can be judged: the
window first of all, pulled until the bands that matter fill the axis, then the
DOS axis, the line widths, the reference line and the legend.

Alongside the plot's own options these controls need the tracks the dialog
offered — how far the window could go, how tall the DOS axis — which travel in
the same dictionary under the subclass's ``EXTENT_KEYS`` and are left out of the
plot.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QSizePolicy, QVBoxLayout, QWidget

from crystalline.crystalio import electronic as el
from crystalline.ui.panels.bands_dialog import row
from crystalline.ui.panels.controls import (ColourButton, Section, range_row, set_span,
                                            slider_row)

#: The plot's own files, carried with its options and not handed to the plotter.
PATH_KEYS = ("bands_path", "dos_path")

# Measured, not chosen: drawn at the size the Plots window opens at, the corners
# of MgO's and silicon's band paths ran into each other along the bottom at
# 4:3, 1.6 and 16:9, and came clear at 2:1. The DOS panel takes its share of the
# width out of the bands', which is what a single band structure never pays.
BANDS_AND_DOS_ASPECT = 2.0


def split_options(options: dict, extent_keys, make) -> tuple:
    """``(bands_path, dos_path, make(**the plot's own options))``."""
    fields = {key: value for key, value in options.items()
              if key not in tuple(extent_keys) + PATH_KEYS}
    return options.get("bands_path"), options.get("dos_path"), make(**fields)


def aspect(options: dict) -> Optional[float]:
    """The figure's shape, width over height: wide for bands beside a DOS.

    ``None`` for the rest, which take the window's own.
    """
    return BANDS_AND_DOS_ASPECT if options.get("mode") == el.BANDS_AND_DOS else None


class BandsControls(QWidget):
    """Window, DOS axis and line styling, with a signal when any of them moves."""

    #: Any control was touched; the reader asks :meth:`options` for all of it.
    changed = Signal()

    WINDOW_SECTION = ""
    #: The option the window is, and the one its track travels under.
    WINDOW_KEY = BOUNDS_KEY = ""
    #: ``(bound low, bound high, low, high, decimals, step)`` of the track.
    WINDOW_TRACK = (0.0, 1.0, 0.0, 1.0, 2, 0.1)
    DOS_DECIMALS, DOS_STEP = 2, 0.1
    #: The reference line's two options: whether it is drawn, and its colour.
    LINE_KEYS = ("", "")
    LINE_LABEL = ""
    LINE_COLOUR = ""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # No stretch of its own, as with the other plot controls.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self._announce = lambda *_args: self.changed.emit()  # noqa: E731

        window = Section(layout, self.WINDOW_SECTION)
        # The track is the dialog's, put here by set_options: it spans the
        # files, which these controls never read.
        low, high, start, end, decimals, step = self.WINDOW_TRACK
        self._window = range_row(window, "Window", start, end, low, high,
                                 decimals=decimals, step=step, on_change=self._announce)
        self._window[1].setToolTip(f"Bottom of the {self.WINDOW_SECTION.lower()} window, "
                                   "in the plot's units.")
        self._window[2].setToolTip(f"Top of the {self.WINDOW_SECTION.lower()} window.")

        dos = Section(layout, "Density of states")
        self.dos_auto = QCheckBox("Fit the DOS axis to what is drawn")
        self.dos_auto.setChecked(True)
        self.dos_auto.toggled.connect(self._on_auto)
        dos.add_wide(self.dos_auto)
        self._dos_span = range_row(dos, "DOS range", 0.0, 1.0, 0.0, 10.0,
                                   decimals=self.DOS_DECIMALS, step=self.DOS_STEP,
                                   on_change=self._announce)
        self._dos_range_row = dos.row_widgets("DOS range")
        self._build_dos_extras(dos)
        self.overlay = QCheckBox("Draw the projections on one axis")
        self.overlay.setChecked(True)
        self.overlay.setToolTip("On one axis with a legend, or one panel each.")
        self.overlay.toggled.connect(self._announce)
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
        self.band_colour.colourChanged.connect(self._announce)
        look.add("Bands", self.band_colour)
        self._band_row = look.row_widgets("Bands")
        self._build_lines(look)
        self.show_line = QCheckBox("Draw")
        self.show_line.setChecked(True)
        self.show_line.toggled.connect(self._announce)
        self.line_colour = ColourButton(self.LINE_COLOUR, f"{self.LINE_LABEL} colour", self)
        self.line_colour.colourChanged.connect(self._announce)
        look.add(self.LINE_LABEL, row(self.show_line, self.line_colour))
        self.linewidth = slider_row(look, "Line width", 1.0, 0.3, 4.0, 0.1,
                                    decimals=1, on_change=self._announce)
        # As on every other plot: named by default, off when the box gets in
        # the way of what it names.
        self.legend = QCheckBox("Legend")
        self.legend.setChecked(True)
        self.legend.toggled.connect(self._announce)
        look.add_wide(self.legend)

        self._sync_dos_range()
        self._plot: dict = {}

    def _build_dos_extras(self, dos) -> None:
        """Rows of the DOS section after the axis range."""

    def _build_lines(self, look) -> None:
        """Appearance rows between the band colour and the reference line."""

    # ── state ───────────────────────────────────────────────────────────
    def options(self) -> dict:
        """What these controls decide, as keys of the plot's options."""
        _slider, low, high = self._window
        low, high = sorted((low.value(), high.value()))
        _slider, dlow, dhigh = self._dos_span
        dlow, dhigh = sorted((dlow.value(), dhigh.value()))
        show, colour = self.LINE_KEYS
        return {
            # A zero-width window means "don't clip", as in the dialog.
            self.WINDOW_KEY: None if low == high else (low, high),
            "dos_range": None if self.dos_auto.isChecked() or dlow == dhigh else (dlow, dhigh),
            "dos_handles": (dlow, dhigh),
            "overlay_projections": self.overlay.isChecked(),
            "linewidth": self.linewidth.value(),
            show: self.show_line.isChecked(),
            "band_colour": self.band_colour.colour(),
            colour: self.line_colour.colour(),
            "legend": self.legend.isChecked(),
        }

    def set_options(self, options: Optional[dict]) -> None:
        """Show a plot's options without announcing them: it already has them."""
        if not options:
            return
        blocked = self.blockSignals(True)
        try:
            bounds = options.get(self.BOUNDS_KEY)
            if bounds:
                window = options.get(self.WINDOW_KEY) or bounds
                set_span(self._window, min(bounds[0], *window), max(bounds[1], *window),
                         *sorted(window))
            bounds = options.get("dos_bounds")
            if bounds:
                handles = options.get("dos_range") or options.get("dos_handles") or bounds
                set_span(self._dos_span, min(bounds[0], *handles), max(bounds[1], *handles),
                         *sorted(handles))
            self.dos_auto.setChecked(options.get("dos_range") is None)
            show, colour = self.LINE_KEYS
            for widget, key in ((self.overlay, "overlay_projections"),
                                (self.show_line, show), (self.legend, "legend")):
                if options.get(key) is not None:
                    widget.setChecked(bool(options[key]))
            if options.get("linewidth") is not None:
                self.linewidth.setValue(float(options["linewidth"]))
            for widget, key in ((self.band_colour, "band_colour"), (self.line_colour, colour)):
                if options.get(key):
                    widget.setColour(options[key])
            self._set_extras(options)
        finally:
            self.blockSignals(blocked)
        self._show_what_applies(options)
        self._sync_dos_range()

    def _set_extras(self, options: dict) -> None:
        """Put back the subclass's own options."""

    def _show_what_applies(self, options: dict) -> None:
        """Hide what this plot has no use for, as the other controls do.

        A band structure has no DOS axis, a DOS no band colour, and the
        band+DOS view always draws its projections together, so only a DOS on
        its own can split them into panels.
        """
        mode = options.get("mode", el.BANDS)
        self._dos_section.set_visible(mode in (el.DOS, el.BANDS_AND_DOS))
        for widget in self._band_row:
            widget.setVisible(mode in (el.BANDS, el.BANDS_AND_DOS))
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


__all__ = ["BANDS_AND_DOS_ASPECT", "PATH_KEYS", "BandsControls", "aspect", "split_options"]
