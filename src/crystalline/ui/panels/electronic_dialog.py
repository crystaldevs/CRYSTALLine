"""Choose how to plot an electronic band structure, a DOS, or both.

One entry with every option in view, for the one-electron properties: energies
relative to the Fermi level or absolute, the path's corners named, and the bands
beside the DOS. The layout and the file handling are those of every band dialog
(:mod:`~crystalline.ui.panels.bands_dialog`); what is particular here is the
energy frame and the spin.

The band and DOS files come from a PROPERTIES (``.d3``) run, which is written
beside the SCF output. The dialog lists every band and DOS file it finds there
— recognised by content, see :func:`~crystalline.crystalio.electronic.find_files`
— with the likeliest one already chosen, and a band file and a DOS of the same
calculation paired by their Fermi level.
"""

from __future__ import annotations

import math
from typing import List, Optional

from CRYSTALClear.units import H_to_eV, eV_to_H
from PySide6.QtWidgets import QComboBox, QPushButton, QWidget

from crystalline.crystalio import electronic as el
from crystalline.ui.panels.bands_dialog import BandsDialog, nice, row, style_box
from crystalline.ui.panels.controls import ColourButton, Section, range_row, set_span
from crystalline.ui.safety import guard


class ElectronicDialog(BandsDialog):
    """Pick the files and the options; :meth:`request` hands them to the plotter."""

    TITLE = "Electronic bands & DOS"
    KINDS = (("band", "Band file", "BAND.DAT, .BAND or fort.25"),
             ("dos", "DOS file", "DOSS.DAT, .DOSS or fort.25"))
    BROWSE = {"band": ("Open a band structure",
                       "Band structures (*.BAND *.DAT *.f25 *.25);;All files (*)"),
              "dos": ("Open a density of states",
                      "Densities of states (*.DOSS *.DAT *.f25 *.25);;All files (*)")}
    K_PLACEHOLDER = "Γ X W L Γ"
    K_TOOLTIP = ("One name per corner of the path, separated by spaces. Filled in "
                 "from the loaded structure where the corners can be recognised; "
                 "“G” is read as Γ.")
    last_folder = ""

    def __init__(self, structure=None, folder: str = "",
                 parent: Optional[QWidget] = None, efermi: Optional[float] = None) -> None:
        """``efermi`` is the open output's Fermi level (eV): files recording the
        same one are that calculation's, and are offered first."""
        super().__init__(structure, folder, parent)
        self._output_efermi = efermi
        # The unit and reference the range boxes are written in, so a change
        # of either can carry the numbers across instead of reinterpreting them.
        self._frame_shown = ("eV", el.RELATIVE)
        self._build()

    # ── building ────────────────────────────────────────────────────────
    def _build_window(self, column) -> None:
        # Unit and reference are restored before the dialog is shown, and the
        # window is fitted when it is — in whichever frame they left it in.
        # They carry the window across only when a *person* picks a new one
        # (``activated``), never when a remembered value is put back.
        energy = Section(column, "Energy")
        self.unit = QComboBox(self)
        for key, label in el.UNITS:
            self.unit.addItem(label, key)
        self.unit.activated.connect(self._on_frame_chosen)
        energy.add("Unit", self.unit)
        self.reference = QComboBox(self)
        for key, label in el.REFERENCES:
            self.reference.addItem(label, key)
        self.reference.setToolTip(
            "CRYSTAL writes band and DOS energies relative to the Fermi level. "
            "Absolute energies add E_F back, and move the Fermi line with them.")
        self.reference.activated.connect(self._on_frame_chosen)
        energy.add("Reference", self.reference)
        # Held in a tuple rather than as attributes of their own, so the
        # dialog's memory passes them by: the window belongs to the data.
        self._window = range_row(energy, "Window", -10.0, 10.0, -30.0, 30.0,
                                 decimals=2, step=0.5)
        fit = QPushButton("Suggested window", self)
        fit.setToolTip("Down to the first wide gap below the Fermi level — where "
                       "the core levels start — and as far again above the "
                       "bottom of the conduction band.")
        fit.clicked.connect(lambda _checked=False: self._fit_window())
        energy.append(row(fit))

    def _build_dos_extras(self, dos) -> None:
        self.spin_down = QComboBox(self)
        self.spin_down.addItem("Mirrored below the axis", el.SPIN_MIRRORED)
        self.spin_down.addItem("On the same side as spin up", el.SPIN_ALONGSIDE)
        self.spin_down.setToolTip("Only a spin-polarised DOS has a spin-down part.")
        self.spin_down.currentIndexChanged.connect(lambda _i: self._fit_dos())
        dos.add("Spin down", self.spin_down)

    def _build_lines(self, look) -> None:
        self.beta_colour = ColourButton(el.BETA_COLOUR, "Spin-down band colour", self)
        self.beta_style = style_box(self, "--")
        look.add("Spin-down bands", row(self.beta_colour, self.beta_style))
        self._beta_row = look.row_widgets("Spin-down bands")
        (self.show_fermi, self.fermi_colour, self.fermi_style,
         self.fermi_width) = self._reference_line(look, "Fermi level", el.FERMI_COLOUR, 1.5)

    # ── files ───────────────────────────────────────────────────────────
    def _find_files(self, folder: str):
        bands, doss = el.find_files(folder, efermi=self._output_efermi)
        # A band structure and a DOS from the same calculation, not merely the
        # best-ranked of each — see pair_files.
        chosen = dict(zip(("band", "dos"),
                          el.pair_files(bands, doss, own_run=self._output_efermi is not None)))
        return bands, doss, chosen

    def _describe_file(self, kind: str, path: str):
        return el.describe_bands(path) if kind == "band" else el.describe_dos(path)

    def _auto_labels(self, info) -> List[str]:
        return el.path_labels(info, self._structure)

    # ── the energy window ───────────────────────────────────────────────
    def _frame(self):
        return (self.unit.currentData(), self.reference.currentData())

    def _efermi(self) -> float:
        """E_F in eV — the band file's when there is one, as the plotter uses."""
        info = self._bands_info or self._dos_info
        return info.efermi if info is not None else 0.0

    def _shown(self, value: float, frame) -> float:
        """eV relative to E_F -> the number the boxes show in ``frame``."""
        unit, reference = frame
        value += self._efermi() if reference == el.ABSOLUTE else 0.0
        return eV_to_H(value) if unit == "Hartree" else value

    def _canonical(self, value: float, frame) -> float:
        unit, reference = frame
        value = H_to_eV(value) if unit == "Hartree" else value
        return value - (self._efermi() if reference == el.ABSOLUTE else 0.0)

    def _spans(self):
        """``(window, span)`` in eV relative to E_F for what is to be drawn."""
        need_bands, need_dos = self._needs()
        bands = self._bands_info if need_bands else None
        doss = self._dos_info if need_dos else None
        spans = [info.energy_span for info in (bands, doss) if info is not None]
        if not spans:
            return None, None
        # The bands decide the window when there are any: a DOS covers the
        # window its .d3 asked for, which is already a choice somebody made.
        window = (bands.window if bands is not None
                  else (float(math.floor(doss.energy_span[0])),
                        float(math.ceil(doss.energy_span[1]))))
        span = (math.floor(min([s[0] for s in spans] + [window[0]])),
                math.ceil(max([s[1] for s in spans] + [window[1]])))
        return window, span

    def _format_energy(self, unit: str) -> None:
        _slider, low, high = self._window
        for box in (low, high):
            box.setDecimals(4 if unit == "Hartree" else 2)
            box.setSingleStep(0.02 if unit == "Hartree" else 0.5)

    def _fit_window(self) -> None:
        """Span the track over the data, and set the handles on the suggestion."""
        frame = self._frame()
        self._format_energy(frame[0])
        self._frame_shown = frame
        window, span = self._spans()
        if window is None:
            return
        bounds = sorted(self._shown(v, frame) for v in span)
        values = sorted(self._shown(v, frame) for v in window)
        set_span(self._window, *bounds, *values)

    @guard()
    def _on_frame_chosen(self, _index: int = 0) -> None:
        new, old = self._frame(), self._frame_shown
        if new == old:
            return
        slider, low, high = self._window
        values = [self._shown(self._canonical(box.value(), old), new) for box in (low, high)]
        bounds = sorted(self._shown(self._canonical(v, old), new) for v in slider.bounds())
        self._format_energy(new[0])
        set_span(self._window, *bounds, *sorted(values))
        if new[0] != old[0]:
            # states/eV <-> states/Hartree: CRYSTALClear rescales the DOS too.
            factor = H_to_eV(1.0) if new[0] == "Hartree" else eV_to_H(1.0)
            dslider, dlow, dhigh = self._dos_span
            dbounds = [v * factor for v in dslider.bounds()]
            set_span(self._dos_span, *dbounds, dlow.value() * factor, dhigh.value() * factor)
        self._frame_shown = new

    def _fit_dos(self) -> None:
        info = self._dos_info
        if info is None or info.dos_max <= 0:
            return
        top = info.dos_max * (H_to_eV(1.0) if self.unit.currentData() == "Hartree" else 1.0)
        mirrored = info.spin == 2 and self.spin_down.currentData() == el.SPIN_MIRRORED
        bound, value = nice(top * 1.25), nice(top * 1.05)
        set_span(self._dos_span, -bound if mirrored else 0.0, bound,
                 -value if mirrored else 0.0, value)

    # ── keeping the controls honest ─────────────────────────────────────
    def _polarised(self):
        return (self._bands_info is not None and self._bands_info.spin == 2,
                self._dos_info is not None and self._dos_info.spin == 2)

    def _sync_lines(self, need_bands: bool, need_dos: bool) -> None:
        """The spin-down choices only for a spin-polarised run."""
        bands_polarised, dos_polarised = self._polarised()
        self.spin_down.setEnabled(need_dos and dos_polarised)
        for widget in self._beta_row:
            widget.setEnabled(need_bands and bands_polarised)
        for widget in (self.fermi_colour, self.fermi_style, self.fermi_width):
            widget.setEnabled(self.show_fermi.isChecked())

    def _describe(self, need_bands: bool, need_dos: bool) -> str:
        lines = []
        if need_bands:
            info = self._bands_info
            lines += self._band_lines(info and (
                f"Bands: {info.n_bands} bands, "
                f"{'spin-polarised' if info.spin == 2 else 'spin-restricted'}, "
                f"E_F = {info.efermi:.3f} eV"))
        if need_dos:
            info = self._dos_info
            lines += self._dos_lines(info and (
                f"DOS: {info.n_projections} projection"
                f"{'s' if info.n_projections != 1 else ''}, "
                f"{'spin-polarised' if info.spin == 2 else 'spin-restricted'}, "
                f"E_F = {info.efermi:.3f} eV"))
        if need_bands and need_dos and self._bands_info and self._dos_info:
            gap = abs(self._bands_info.efermi - self._dos_info.efermi)
            if gap > 0.01:
                # Each file is relative to its own E_F, so side by side the DOS
                # is shifted against the bands by exactly this much.
                lines.append(f"⚠ The two Fermi levels differ by {gap:.3f} eV — the DOS "
                             "will sit that far off the bands. Same calculation?")
        return "\n".join(lines)

    # ── the answer ──────────────────────────────────────────────────────
    def request(self):
        """``(band_path, dos_path, ElectronicOptions)`` for :func:`plot_electronic`."""
        need_bands, need_dos = self._needs()
        low, high = self.window_values()
        options = el.ElectronicOptions(
            mode=self.mode.currentData(),
            unit=self.unit.currentData(),
            reference=self.reference.currentData(),
            # A zero-width range means "don't clip", as in the spectra dialog.
            energy_range=None if low == high else (low, high),
            dos_range=self._dos_range(),
            k_labels=self.k_labels() if need_bands else None,
            **self._projection_choice(),
            overlay_projections=self.overlay.isChecked(),
            spin_down=self.spin_down.currentData(),
            band_colour=self.band_colour.colour(),
            beta_colour=self.beta_colour.colour(),
            beta_style=self.beta_style.currentData(),
            linewidth=self.linewidth.value(),
            show_fermi=self.show_fermi.isChecked(),
            fermi_colour=self.fermi_colour.colour(),
            fermi_style=self.fermi_style.currentData(),
            fermi_width=self.fermi_width.value(),
            legend=self.legend.isChecked(),
            title=self._title.text().strip() or None,
        )
        return (self._path("band") if need_bands else None,
                self._path("dos") if need_dos else None,
                options)

    def plot_title(self) -> str:
        return el.plot_title(self.mode.currentData())

    def extent(self) -> dict:
        """The tracks the window and the DOS range were chosen on.

        Not plot options: a plot does not need to know how far its window
        could have gone. The controls beside it do, to offer the same track.
        """
        bands_polarised, dos_polarised = self._polarised()
        return {"energy_bounds": self.window_bounds(), **self._extent(),
                # What the files hold, for deciding whether a legend names anything.
                "dos_polarised": dos_polarised, "bands_polarised": bands_polarised}

    def set_request(self, bands_path: Optional[str], dos_path: Optional[str],
                    options: "el.ElectronicOptions") -> None:
        """Show a plot's own choices, so that editing it starts from them."""
        self._restore(bands_path, dos_path, options, options.energy_range, boxes=(
            (self.unit, options.unit), (self.reference, options.reference),
            (self.spin_down, options.spin_down), (self.beta_style, options.beta_style),
            (self.fermi_style, options.fermi_style)))
        self.beta_colour.setColour(options.beta_colour)
        self.show_fermi.setChecked(options.show_fermi)
        self.fermi_colour.setColour(options.fermi_colour)
        self.fermi_width.setValue(options.fermi_width)
        self._sync()


__all__ = ["ElectronicDialog"]
