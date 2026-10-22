"""Choose how to plot phonon bands, a phonon DOS, or both.

The counterpart of the electronic bands dialog for a dispersion run, built on the
same base (:mod:`~crystalline.ui.panels.bands_dialog`). The band and DOS files
are those the run wrote beside its output — ``PHONBANDS.DAT``, ``PHONDOS.DAT``
and their ``fort.25`` — recognised by content (see
:func:`~crystalline.crystalio.phonon_bands.find_files`), with the open output's
own band file chosen first. What is particular here is the unit, cm⁻¹ or THz,
and the zero-frequency line.
"""

from __future__ import annotations

import math
from typing import List, Optional

from CRYSTALClear.units import cm_to_thz
from PySide6.QtWidgets import QComboBox, QPushButton, QWidget

from crystalline.crystalio import phonon_bands as pb
from crystalline.ui.panels.bands_dialog import BandsDialog, nice, row
from crystalline.ui.panels.controls import Section, range_row, set_span
from crystalline.ui.safety import guard


class PhononDialog(BandsDialog):
    """Pick the files and the options; :meth:`request` hands them to the plotter."""

    TITLE = "Phonon bands & DOS"
    KINDS = (("band", "Band file", "PHONBANDS.DAT or fort.25"),
             ("dos", "DOS file", "PHONDOS.DAT or fort.25"))
    BROWSE = {"band": ("Open phonon bands",
                       "Phonon bands (*.DAT *.f25 *.25 *.BAND);;All files (*)"),
              "dos": ("Open a phonon DOS",
                      "Phonon densities of states (*.DAT *.f25 *.25 *.DOSS);;All files (*)")}
    K_PLACEHOLDER = "Γ K M Γ A"
    K_TOOLTIP = ("One name per corner of the path, separated by spaces. Filled in "
                 "from the path the output or the input deck states, where the "
                 "corners can be recognised; “G” is read as Γ.")
    DOS_DECIMALS, DOS_STEP = 4, 0.001
    last_folder = ""

    def __init__(self, structure=None, folder: str = "", parent: Optional[QWidget] = None,
                 output: Optional[str] = None) -> None:
        """``output`` is the open output: its band file is offered first, and
        the path it states names the corners."""
        super().__init__(structure, folder, parent)
        self._output = output
        # The unit the range boxes are written in, so that a change of unit can
        # carry the numbers across instead of reinterpreting them.
        self._unit_shown = pb.CM
        self._build()

    # ── building ────────────────────────────────────────────────────────
    def _build_window(self, column) -> None:
        frequency = Section(column, "Frequency")
        self.unit = QComboBox(self)
        for key, label in pb.UNITS:
            self.unit.addItem(label, key)
        # Carries the window across only when a person picks a unit; a
        # remembered one is put back before the window is fitted.
        self.unit.activated.connect(self._on_unit_chosen)
        frequency.add("Unit", self.unit)
        # Held in a tuple rather than as attributes of their own, so the
        # dialog's memory passes them by: the window belongs to the data.
        self._window = range_row(frequency, "Window", 0.0, 1000.0, -200.0, 1200.0,
                                 decimals=1, step=10.0)
        fit = QPushButton("Suggested window", self)
        fit.setToolTip("From zero, or from the lowest imaginary branch where there "
                       "is one, to just above the highest branch.")
        fit.clicked.connect(lambda _checked=False: self._fit_window())
        frequency.append(row(fit))

    def _build_lines(self, look) -> None:
        (self.show_zero, self.zero_colour, self.zero_style,
         self.zero_width) = self._reference_line(look, "Zero frequency", pb.ZERO_COLOUR, 0.8)

    # ── files ───────────────────────────────────────────────────────────
    def _find_files(self, folder: str):
        bands, doss = pb.find_files(folder, output=self._output)
        if doss and not bands and self.mode.currentData() != pb.DOS:
            # A run that wrote only a DOS: show what there is.
            self.mode.setCurrentIndex(self.mode.findData(pb.DOS))
        return bands, doss, {}

    def _describe_file(self, kind: str, path: str):
        return pb.describe_bands(path) if kind == "band" else pb.describe_dos(path)

    def _auto_labels(self, info) -> List[str]:
        return pb.path_labels(info, self._structure, output=self._output)

    # ── the frequency window ────────────────────────────────────────────
    @staticmethod
    def _factor(unit: str) -> float:
        """cm⁻¹ -> ``unit``."""
        return 1.0 if unit == pb.CM else cm_to_thz(1.0)

    def _spans(self):
        """``(window, span)`` in cm⁻¹ for what is to be drawn."""
        need_bands, need_dos = self._needs()
        bands = self._bands_info if need_bands else None
        doss = self._dos_info if need_dos else None
        spans = [info.span for info in (bands, doss) if info is not None]
        if not spans:
            return None, None
        window = bands.window if bands is not None else pb.suggest_window(doss.span)
        low = min([s[0] for s in spans] + [window[0]])
        high = max([s[1] for s in spans] + [window[1]])
        step = pb._step(high - low)
        return window, (math.floor(low / step) * step - step,
                        math.ceil(high / step) * step + step)

    def _format_frequency(self, unit: str) -> None:
        _slider, low, high = self._window
        for box in (low, high):
            box.setDecimals(1 if unit == pb.CM else 3)
            box.setSingleStep(10.0 if unit == pb.CM else 0.25)

    def _fit_window(self) -> None:
        """Span the track over the data, and set the handles on the suggestion."""
        unit = self.unit.currentData()
        self._format_frequency(unit)
        self._unit_shown = unit
        window, span = self._spans()
        if window is None:
            return
        factor = self._factor(unit)
        set_span(self._window, span[0] * factor, span[1] * factor,
                 window[0] * factor, window[1] * factor)

    @guard()
    def _on_unit_chosen(self, _index: int = 0) -> None:
        new, old = self.unit.currentData(), self._unit_shown
        if new == old:
            return
        ratio = self._factor(new) / self._factor(old)
        slider, low, high = self._window
        bounds = sorted(v * ratio for v in slider.bounds())
        values = sorted((low.value() * ratio, high.value() * ratio))
        self._format_frequency(new)
        set_span(self._window, *bounds, *values)
        # States per cm⁻¹ and per THz go the other way.
        dslider, dlow, dhigh = self._dos_span
        set_span(self._dos_span, *(v / ratio for v in dslider.bounds()),
                 dlow.value() / ratio, dhigh.value() / ratio)
        self._unit_shown = new

    def _fit_dos(self) -> None:
        info = self._dos_info
        if info is None or info.dos_max <= 0:
            return
        top = info.dos_max / self._factor(self.unit.currentData())
        set_span(self._dos_span, 0.0, nice(top * 1.25), 0.0, nice(top * 1.05))

    # ── keeping the controls honest ─────────────────────────────────────
    def _sync_lines(self, need_bands: bool, need_dos: bool) -> None:
        for widget in (self.zero_colour, self.zero_style, self.zero_width):
            widget.setEnabled(self.show_zero.isChecked())

    def _describe(self, need_bands: bool, need_dos: bool) -> str:
        lines = []
        if need_bands:
            info = self._bands_info
            lines += self._band_lines(info and (
                f"Bands: {info.n_bands} branches, {info.n_kpoints} points, "
                f"{info.span[0]:.1f} to {info.span[1]:.1f} cm⁻¹"))
            if info is None and not self._errors.get("band"):
                lines.append("A dispersion run (DISPERSI with BANDS) writes them.")
            elif info is not None and info.span[0] < -pb.NOISE:
                lines.append(f"⚠ Imaginary frequencies down to {info.span[0]:.1f} cm⁻¹.")
        if need_dos:
            info = self._dos_info
            lines += self._dos_lines(info and (
                f"DOS: {info.n_projections} projection"
                f"{'s' if info.n_projections != 1 else ''}"))
            if info is None and not self._errors.get("dos"):
                lines.append("A dispersion run with PDOS writes them.")
        return "\n".join(lines)

    # ── the answer ──────────────────────────────────────────────────────
    def request(self):
        """``(band_path, dos_path, PhononOptions)`` for :func:`plot_phonons`."""
        need_bands, need_dos = self._needs()
        low, high = self.window_values()
        options = pb.PhononOptions(
            mode=self.mode.currentData(),
            unit=self.unit.currentData(),
            # A zero-width range means "don't clip", as in the other dialogs.
            frequency_range=None if low == high else (low, high),
            dos_range=self._dos_range(),
            k_labels=self.k_labels() if need_bands else None,
            **self._projection_choice(),
            overlay_projections=self.overlay.isChecked(),
            band_colour=self.band_colour.colour(),
            linewidth=self.linewidth.value(),
            show_zero=self.show_zero.isChecked(),
            zero_colour=self.zero_colour.colour(),
            zero_style=self.zero_style.currentData(),
            zero_width=self.zero_width.value(),
            legend=self.legend.isChecked(),
            title=self._title.text().strip() or None,
        )
        return (self._path("band") if need_bands else None,
                self._path("dos") if need_dos else None,
                options)

    def plot_title(self) -> str:
        return pb.plot_title(self.mode.currentData())

    def extent(self) -> dict:
        """The tracks the window and the DOS range were chosen on."""
        return {"frequency_bounds": self.window_bounds(), **self._extent()}

    def set_request(self, bands_path: Optional[str], dos_path: Optional[str],
                    options: "pb.PhononOptions") -> None:
        """Show a plot's own choices, so that editing it starts from them."""
        self._restore(bands_path, dos_path, options, options.frequency_range, boxes=(
            (self.unit, options.unit), (self.zero_style, options.zero_style)))
        self.show_zero.setChecked(options.show_zero)
        self.zero_colour.setColour(options.zero_colour)
        self.zero_width.setValue(options.zero_width)
        self._sync()


__all__ = ["PhononDialog"]
