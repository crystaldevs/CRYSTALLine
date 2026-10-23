"""Phonon band structures and densities of states, with their options.

The counterpart of :mod:`crystalline.crystalio.electronic` for a dispersion
run (``FREQCALC`` with ``DISPERSI``), and drawn through the same code: the
frequencies stand where the energies would, and the zero-frequency line where
the Fermi level is. What differs is what the files say and how a reader finds
its way round them:

* **Finding the files.** A dispersion run writes ``PHONBANDS.DAT`` and a
  ``fort.25`` for the bands, and ``PHONDOS.DAT`` and a ``fort.25`` for the
  density of states, in the formats a PROPERTIES run uses for electrons. They
  are told from electronic ones by their contents (see
  :func:`~crystalline.crystalio.electronic.data_kind`), and a band file is
  matched to the open output by the frequencies at the first point of its
  path, which the output prints too.
* **The path's corners.** A ``fort.25`` writes them as integers without the
  shrinking factor they are over. The output states each segment with its
  denominator (``FROM K ( 0 0 0 ) TO K ( 4 4 0 ) WITH DENOMINATOR 12``), and
  so does the input deck in its ``DISPERSI`` / ``BANDS`` block, so the corners
  are named from those against the structure's special points.
* **The window.** Down to zero, or to the lowest imaginary branch where there
  is one deeper than the numerical noise at Γ, and up to just above the
  highest branch.
* **Units.** cm⁻¹ or THz, for the frequencies and, inversely, for the DOS.

Qt-free, like the rest of ``crystalio``: the dialog chooses, this draws.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from CRYSTALClear.units import cm_to_thz, thz_to_cm

from crystalline.crystalio import electronic as el
from crystalline.crystalio.electronic import BANDS, BANDS_AND_DOS, DOS, MODES  # noqa: F401

CM = "cm-1"
THZ = "THz"
UNITS = ((CM, "cm⁻¹"), (THZ, "THz"))

BAND_COLOUR = el.BAND_COLOUR
# The zero line is a reference, not a quantity: in the house slate, so that an
# imaginary branch dipping under it stands out in the band colour.
ZERO_COLOUR = "#5b6878"

# Imaginary frequencies this small (cm⁻¹) are the acoustic branches missing
# zero at Γ by the accuracy of the force constants, not an instability, and do
# not move the bottom of the suggested window below zero. A real soft mode is
# tens of cm⁻¹ deep.
NOISE = 5.0


# ── what a file holds ─────────────────────────────────────────────────────
@dataclass(frozen=True)
class BandsInfo:
    """Enough about a phonon band file to set the dialog up, in cm⁻¹."""

    path: str
    n_bands: int
    n_kpoints: int
    ticks: Tuple[str, ...] = ()
    span: Tuple[float, float] = (0.0, 0.0)
    window: Tuple[float, float] = (0.0, 1000.0)
    first_point: Tuple[float, ...] = ()       # the frequencies at the path's start


@dataclass(frozen=True)
class DosInfo:
    """Enough about a phonon DOS file to set the dialog up, in cm⁻¹."""

    path: str
    n_projections: int
    span: Tuple[float, float] = (0.0, 0.0)
    dos_max: float = 0.0                      # states/cm⁻¹, the tallest projection


def describe_bands(path: str) -> BandsInfo:
    """Read a phonon band file for the dialog, remembered by its version."""
    from crystalline.crystalio import remembered

    return remembered.read("phonon-bands-info", (path,), lambda: _describe_bands(path),
                           light=True)


def _describe_bands(path: str) -> BandsInfo:
    bands = _read_bands(path)
    frequencies = thz_to_cm(np.asarray(bands.bands, dtype=float)[:, :, 0])
    return BandsInfo(path=path, n_bands=int(bands.n_bands), n_kpoints=int(bands.n_kpoints),
                     ticks=tuple(el.clean_tick(t) for t in bands.tick_label),
                     span=(float(frequencies.min()), float(frequencies.max())),
                     window=suggest_window(frequencies),
                     first_point=tuple(float(f) for f in np.sort(frequencies[:, 0])))


def describe_dos(path: str) -> DosInfo:
    """Read a phonon DOS file for the dialog, remembered by its version."""
    from crystalline.crystalio import remembered

    return remembered.read("phonon-dos-info", (path,), lambda: _describe_dos(path),
                           light=True)


def _describe_dos(path: str) -> DosInfo:
    doss = _read_dos(path)
    data = np.asarray(doss.doss, dtype=float)
    frequencies = thz_to_cm(data[:, 0, ...])
    values = cm_to_thz(data[:, 1:, ...])          # states/THz -> states/cm⁻¹
    return DosInfo(path=path, n_projections=int(doss.n_proj),
                   span=(float(frequencies.min()), float(frequencies.max())),
                   dos_max=float(np.abs(values).max()) if values.size else 0.0)


def suggest_window(frequencies) -> Tuple[float, float]:
    """A frequency window (cm⁻¹) that shows every branch, rounded out.

    From zero, or from the lowest branch when it is imaginary by more than
    :data:`NOISE`, to a little above the highest; rounded to a step suited to
    the span, so a ZnO run up to 565 cm⁻¹ gets 0–600.
    """
    f = np.asarray(frequencies, dtype=float)
    if f.size == 0:
        return (0.0, 1000.0)
    low, high = float(f.min()), float(f.max())
    lower = low if low < -NOISE else 0.0
    upper = max(high, lower + 1.0)
    step = _step(upper - lower)
    return (float(math.floor(lower * 1.05 / step) * step),
            float(math.ceil(upper * 1.03 / step) * step))


def _step(span: float) -> float:
    """A round step for a span: 50 for hundreds of cm⁻¹, 5 for tens."""
    if span <= 0:
        return 1.0
    return 10.0 ** math.floor(math.log10(span)) / 2.0


# ── the output's own path ─────────────────────────────────────────────────
@dataclass(frozen=True)
class OutputPath:
    """What a dispersion output says about the path it was computed along."""

    corners: Tuple[Tuple[int, int, int], ...]
    shrink: int
    first_point: Tuple[float, ...] = ()       # cm⁻¹, sorted


_SEGMENT = re.compile(
    r"FROM K\s*\(\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*\)\s*TO K\s*\(\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)"
    r"\s*\)\s*WITH DENOMINATOR\s+(\d+)")
_FREQUENCY_ROW = re.compile(r"^\s*(\d+)-\s*(\d+)\s+\S+\s+(-?\d+\.\d+)\s+-?\d+\.\d+")


def output_path(path: Optional[str]) -> Optional[OutputPath]:
    """The path of a dispersion run's ``PHONON BANDS``, or ``None``.

    Read from the output, where CRYSTAL states every segment with its
    denominator, and with the frequencies at the path's first point — what a
    band file from the same run starts with. Remembered by the version of the
    file, as the other reads are.
    """
    if not path:
        return None
    from crystalline.crystalio import remembered

    return remembered.read("phonon-output-path", (path,), lambda: _output_path(path),
                           light=True)


def _output_path(path: str) -> Optional[OutputPath]:
    segments: List[Tuple[Tuple[int, ...], Tuple[int, ...], int]] = []
    first: List[float] = []
    reading = done = False
    try:
        with open(path, "r", errors="ignore") as handle:
            for line in handle:
                if "WITH DENOMINATOR" in line:
                    match = _SEGMENT.search(line)
                    if match:
                        numbers = [int(g) for g in match.groups()]
                        segments.append((tuple(numbers[:3]), tuple(numbers[3:6]), numbers[6]))
                    continue
                if done or not segments:
                    continue
                if not reading:
                    reading = "DISPERSION K POINT NUMBER" in line
                    continue
                row = _FREQUENCY_ROW.match(line)
                if row:
                    count = int(row.group(2)) - int(row.group(1)) + 1
                    first.extend([float(row.group(3))] * count)
                elif first:
                    done = True
    except OSError:
        return None
    if not segments:
        return None
    corners = [segments[0][0]] + [segment[1] for segment in segments]
    return OutputPath(corners=tuple(corners), shrink=segments[0][2],
                      first_point=tuple(sorted(first)))


# ── the path an input deck asked for ──────────────────────────────────────
_DECK_EXTENSIONS = (".d12", ".inp")
_DECK_LIMIT = 1 << 20


@dataclass(frozen=True)
class DispersionDeck:
    """The ``BANDS`` block of a deck's ``DISPERSI``: the path asked for."""

    path: str
    shrink: int
    corners: Tuple[Tuple[int, int, int], ...]
    labels: Tuple[str, ...] = ()


def read_dispersion_decks(folder) -> List[DispersionDeck]:
    """Every ``DISPERSI`` / ``BANDS`` path in the input decks of ``folder``.

    A deck is recognised by what it holds — a ``DISPERSI`` keyword line with a
    ``BANDS`` block after it — not by its name.
    """
    decks: List[DispersionDeck] = []
    try:
        entries = sorted(Path(folder).iterdir())
    except OSError:
        return decks
    for path in entries:
        if not path.is_file() or path.suffix.lower() not in _DECK_EXTENSIONS:
            continue
        try:
            if path.stat().st_size > _DECK_LIMIT:
                continue
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        decks.extend(_dispersion_blocks(text, str(path)))
    return decks


def _dispersion_blocks(text: str, path: str) -> List[DispersionDeck]:
    """``DISPERSI`` … ``BANDS`` / ``ISS NK`` / ``NLINE`` / one line per segment,
    the segments written as a PROPERTIES ``BAND`` block writes them."""
    blocks: List[DispersionDeck] = []
    lines = [line.strip() for line in text.splitlines()]
    in_dispersion = False
    for index, line in enumerate(lines):
        word = line.upper()
        if word == "DISPERSI":
            in_dispersion = True
            continue
        if not in_dispersion or word != "BANDS" or index + 2 >= len(lines):
            continue
        try:
            shrink = int(lines[index + 1].split()[0])
            n_lines = int(lines[index + 2].split()[0])
        except (ValueError, IndexError):
            continue
        path_read = el.read_segments(lines[index + 3: index + 3 + n_lines], n_lines)
        if path_read is not None:
            corners, labels = path_read
            blocks.append(DispersionDeck(path=path, shrink=shrink, corners=corners,
                                         labels=labels))
    return blocks


# ── naming the path's corners ─────────────────────────────────────────────
def _agrees(ticks: Sequence[str], corners: Sequence[Sequence[int]], shrink: int) -> bool:
    """Whether a file's corners are these, written over ``shrink``.

    A fort.25 writes the integers alone, which are compared as they are; a
    text file writes its own denominator, and the fractions are compared.
    """
    if len(ticks) != len(corners) or not shrink:
        return False
    for tick, corner in zip(ticks, corners):
        match = el._TICK.search(el.clean_tick(tick))
        if match is None:
            return False
        numbers = np.array([int(match.group(i)) for i in (1, 2, 3)], dtype=float)
        given = np.asarray(corner, dtype=float)
        if match.group(4) is None:
            if not np.array_equal(numbers, given):
                return False
        elif not np.allclose(numbers / int(match.group(4)), given / shrink, atol=1e-6):
            return False
    return True


def path_labels(info: BandsInfo, structure=None, folder=None,
                output: Optional[str] = None) -> List[str]:
    """Names for a band file's corners, from the best source there is.

    The run's own statement of its path first — the output, then an input
    deck beside the file — whose denominators turn the file's integers into
    points that the structure's special points can name; letters the deck
    gives win over those. Anything nothing can name keeps the coordinates it
    came with.
    """
    from crystalline.core.brillouin import display_label

    ticks = list(info.ticks)
    stated = output_path(output)
    if stated is not None and _agrees(ticks, stated.corners, stated.shrink):
        return el.name_ticks(_written(stated.corners, stated.shrink), structure)
    for deck in read_dispersion_decks(folder or Path(info.path).parent):
        if deck.corners and _agrees(ticks, deck.corners, deck.shrink):
            named = el.name_ticks(_written(deck.corners, deck.shrink), structure)
            return [display_label(given) if given else name
                    for given, name in zip(deck.labels, named)]
    return el.name_ticks(ticks, structure)


def _written(corners, shrink: int) -> List[str]:
    """Corners as a text band file writes them: ``(4,4,0)/12``."""
    return ["({},{},{})/{}".format(*corner, shrink) for corner in corners]


# ── finding the files ─────────────────────────────────────────────────────
# Two first points this close (cm⁻¹) are one run's: the output prints four
# decimals, a fort.25 six significant figures.
_SAME_POINT = 0.05


def find_files(folder, output: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """Phonon band and DOS files in ``folder``, best first.

    Picked out by extension and identified by their contents, never by the
    rest of the name. The open output's own band file comes first, known by
    the frequencies at the start of its path; then a text file over a
    fort.25, whose corners lack their denominator; then the newest.
    """
    found = {el.BAND_FILE: [], el.DOS_FILE: []}
    try:
        entries = list(Path(folder).iterdir())
    except OSError:
        return [], []
    for path in entries:
        if path.is_file() and path.suffix.lower() in el._DATA_EXTENSIONS:
            kind = el.data_kind(path)
            if kind is not None and kind[0] == el.PHONON:
                found[kind[1]].append(path)

    stated = output_path(output)

    def own_run(path: Path) -> int:
        if stated is None or not stated.first_point:
            return 0
        try:
            first = describe_bands(str(path)).first_point
        except Exception:  # noqa: BLE001 - an unreadable file is nobody's
            return 1
        same = (len(first) == len(stated.first_point)
                and np.allclose(first, stated.first_point, atol=_SAME_POINT))
        return 0 if same else 1

    def rank(path: Path, match=None):
        try:
            age = -path.stat().st_mtime
        except OSError:
            age = 0.0
        return ((match(path) if match else 0), el._is_fort25(path), age, str(path))

    return ([str(p) for p in sorted(found[el.BAND_FILE], key=lambda p: rank(p, own_run))],
            [str(p) for p in sorted(found[el.DOS_FILE], key=rank)])


# ── the picture ───────────────────────────────────────────────────────────
@dataclass
class PhononOptions:
    """Everything the dialog decides, as one object the plotter takes."""

    mode: str = BANDS
    unit: str = CM
    frequency_range: Optional[Tuple[float, float]] = None   # in the chosen unit
    dos_range: Optional[Tuple[float, float]] = None         # None = fit what is drawn
    k_labels: Optional[List[str]] = None
    projections: Optional[List[int]] = None                 # 1-based; None = all
    projection_labels: Optional[List[str]] = None
    projection_colours: Optional[List[str]] = None
    overlay_projections: bool = True
    band_colour: str = BAND_COLOUR
    linewidth: float = 1.0
    show_zero: bool = True
    zero_colour: str = ZERO_COLOUR
    zero_style: str = "-"
    zero_width: float = 0.8
    legend: bool = True
    title: Optional[str] = None


def plot_title(mode: str) -> str:
    """What a tab drawn in ``mode`` is called."""
    return {BANDS: "Phonon bands", DOS: "Phonon DOS",
            BANDS_AND_DOS: "Phonon bands + DOS"}[mode]


def plot_phonons(bands_path: Optional[str] = None, dos_path: Optional[str] = None,
                 options: Optional[PhononOptions] = None):
    """Draw phonon bands, a phonon DOS, or the two side by side. Returns a Figure."""
    from crystalline.crystalio.plotting import _plot_module

    options = options or PhononOptions()
    _plot_module()
    need_bands = options.mode in (BANDS, BANDS_AND_DOS)
    need_dos = options.mode in (DOS, BANDS_AND_DOS)
    if need_bands and not bands_path:
        raise ValueError("Phonon bands need a band file (PHONBANDS.DAT or fort.25).")
    if need_dos and not dos_path:
        raise ValueError("A phonon DOS needs a DOS file (PHONDOS.DAT or fort.25).")

    # CRYSTALClear reads both in THz. They are put in the unit asked for here,
    # and then said to be in the unit the drawing is asked for, so that
    # CRYSTALClear's electronic plotters, which do the drawing, convert nothing.
    factor = thz_to_cm(1.0) if options.unit == CM else 1.0
    bands = _read_bands(bands_path) if need_bands else None
    doss = _read_dos(dos_path) if need_dos else None
    if bands is not None:
        bands.bands = np.asarray(bands.bands, dtype=float) * factor
        bands.unit = "eV"
    if doss is not None:
        data = np.array(doss.doss, dtype=float)
        data[:, 0, :] *= factor
        data[:, 1:, :] /= factor
        doss.doss = data
        doss.unit = "eV"

    drawing = el.ElectronicOptions(
        mode=options.mode, unit="eV", reference=el.RELATIVE,
        energy_range=options.frequency_range, dos_range=options.dos_range,
        k_labels=options.k_labels, projections=options.projections,
        projection_labels=options.projection_labels,
        projection_colours=options.projection_colours,
        overlay_projections=options.overlay_projections, spin_down=el.SPIN_ALONGSIDE,
        band_colour=options.band_colour, linewidth=options.linewidth,
        show_fermi=options.show_zero, fermi_colour=options.zero_colour,
        fermi_style=options.zero_style, fermi_width=options.zero_width,
        legend=options.legend, title=options.title)
    figure, axis = el.draw(bands, doss, drawing, 0.0)
    _label_axes(figure, axis, options)
    return figure


def _label_axes(figure, axis: str, options: PhononOptions) -> None:
    """Frequency on one axis, and the DOS in states per unit on the other."""
    unit = r"cm$^{-1}$" if options.unit == CM else "THz"
    frequency = f"Frequency ({unit})"
    if axis == "y":
        figure.supylabel(frequency)
    else:
        figure.supxlabel(frequency)
        figure.supylabel(f"DOS (states/{unit})")


# ── plumbing ──────────────────────────────────────────────────────────────
def _read_bands(path: str):
    from CRYSTALClear.crystal_io import External_unit

    return External_unit().read_phonon_band(path)


def _read_dos(path: str):
    from CRYSTALClear.crystal_io import External_unit

    return External_unit().read_phonon_dos(path)


__all__ = [
    "BANDS",
    "BANDS_AND_DOS",
    "BAND_COLOUR",
    "CM",
    "DOS",
    "MODES",
    "NOISE",
    "THZ",
    "UNITS",
    "ZERO_COLOUR",
    "BandsInfo",
    "DispersionDeck",
    "DosInfo",
    "OutputPath",
    "PhononOptions",
    "describe_bands",
    "describe_dos",
    "find_files",
    "output_path",
    "path_labels",
    "plot_phonons",
    "plot_title",
    "read_dispersion_decks",
    "suggest_window",
]
