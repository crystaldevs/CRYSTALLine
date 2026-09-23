"""Electronic band structures and densities of states, with their options.

The Plot menu used to offer a band structure and a DOS as two one-click
entries, each drawn with every CRYSTALClear option at its default and no way to
put the two side by side. This is the model behind a dialog instead — the same
move the vibrational spectra made — and it deals with what the defaults got
wrong:

* **The energy reference.** CRYSTAL writes band and DOS energies already
  relative to the Fermi level, and CRYSTALClear draws its Fermi line at zero
  and keeps the absolute ``efermi`` on the object. Showing absolute energies
  means adding ``efermi`` back *and* moving that line, which is hard-coded at
  zero in all three of CRYSTALClear's plotters.
* **The window.** A band file can run from a 1s level at −1265 eV to empty
  bands at +180; drawn whole, the valence bands are a flat smear. The
  suggested window stops at the first wide gap below the Fermi level — where
  the core levels start — and runs as far above it.
* **The path's corners.** Every band file labels them with raw coordinates —
  ``"(4,0,4)/8"`` — and without explicit labels CRYSTALClear marks them with
  their k-distance instead (``0.613``). The corners are named here from the
  loaded structure's special points, the same ones the path builder uses, and
  matched up to the lattice's own symmetry: CRYSTAL's tutorial writes
  beryllium's M as (0,½,0) where the standard set has (½,0,0), which is the
  same point seen from another axis.
* **Spin-polarised bands on their own.** CRYSTALClear's standalone band plot
  draws both spin channels in one colour and one style, so α and β cannot be
  told apart. Here β is laid over α in a colour and style of its own.
* **Finding the files.** A PROPERTIES run leaves ``BAND.DAT``, ``*.BAND``,
  ``fort.25`` … beside the SCF output, often several of each. They are told
  apart by what their first line says, not by their names.

Qt-free, like the rest of ``crystalio``: the dialog chooses, this draws.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

BANDS = "bands"
DOS = "dos"
BANDS_AND_DOS = "banddos"
MODES = ((BANDS, "Band structure"), (DOS, "Density of states"),
         (BANDS_AND_DOS, "Band structure and DOS"))

RELATIVE = "relative"
ABSOLUTE = "absolute"
REFERENCES = ((RELATIVE, "Relative to the Fermi level  (E − E_F)"),
              (ABSOLUTE, "Absolute energy"))

UNITS = (("eV", "eV"), ("Hartree", "Hartree"))

LINE_STYLES = (("-", "Solid"), ("--", "Dashed"), (":", "Dotted"), ("-.", "Dash-dot"))

# How a spin-down DOS is drawn: below the axis, mirrored, or alongside α.
SPIN_MIRRORED = "down"
SPIN_ALONGSIDE = "up"

# CRYSTALClear's own Fermi-line colour, used as a *marker*: every plotter is
# handed it, the lines drawn in it are found again afterwards, and only then
# given the colour, style and width that were asked for. Found first, styled
# second — so a Fermi line asked for in black cannot be confused with the black
# zero lines beside it.
FERMI_COLOUR = "forestgreen"

BAND_COLOUR = "#1f3a93"
BETA_COLOUR = "#c0392b"

# A qualitative palette for the DOS projections, with no green in it — green
# is the Fermi level's.
PROJECTION_COLOURS = ("#1f77b4", "#d62728", "#9467bd", "#8c564b", "#e377c2",
                      "#7f7f7f", "#bcbd22", "#17becf", "#ff7f0e", "#393b79")

# A gap wider than this below the Fermi level is taken as the top of the core.
# Valence manifolds are separated by less (urea's 2s-2p gap is 8.7 eV, LiF's
# F-centre sits 8.8 eV above its valence band); core levels by far more (Be 1s
# is 104 eV below the valence band, MgO's O 2s 12 eV).
CORE_GAP = 10.0

_HARTREE_EV = 27.211386245988
_TICK = re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*(?:/\s*(\d+))?")
_OCCUPIED = 0.05          # eV: a band starting below this is (partly) occupied


# ── what a file holds ─────────────────────────────────────────────────────
@dataclass(frozen=True)
class BandsInfo:
    """Enough about a band file to set the dialog up, read without drawing."""

    path: str
    spin: int
    efermi: float                      # eV, absolute
    n_bands: int
    ticks: Tuple[str, ...] = ()        # as the file writes them, cleaned
    energy_span: Tuple[float, float] = (0.0, 0.0)   # eV, relative to E_F
    window: Tuple[float, float] = (-10.0, 10.0)     # suggested, likewise


@dataclass(frozen=True)
class DosInfo:
    path: str
    spin: int
    efermi: float                      # eV, absolute
    n_projections: int
    energy_span: Tuple[float, float] = (0.0, 0.0)   # eV, relative to E_F
    dos_max: float = 0.0               # states/eV, the tallest projection


def describe_bands(path: str) -> BandsInfo:
    """Read a band file (``BAND.DAT``, ``*.BAND``, ``fort.25``) for the dialog."""
    bands = _read_bands(path)
    energies = np.asarray(bands.bands, dtype=float)
    return BandsInfo(path=path, spin=int(bands.spin), efermi=float(bands.efermi),
                     n_bands=int(bands.n_bands),
                     ticks=tuple(clean_tick(t) for t in bands.tick_label),
                     energy_span=(float(energies.min()), float(energies.max())),
                     window=suggest_window(energies))


def describe_dos(path: str) -> DosInfo:
    """Read a DOS file (``DOSS.DAT``, ``*.DOSS``, ``fort.25``) for the dialog."""
    doss = _read_dos(path)
    data = np.asarray(doss.doss, dtype=float)
    energies, values = data[:, 0, ...], data[:, 1:, ...]
    return DosInfo(path=path, spin=int(doss.spin), efermi=float(doss.efermi),
                   n_projections=int(doss.n_proj),
                   energy_span=(float(energies.min()), float(energies.max())),
                   dos_max=float(np.abs(values).max()) if values.size else 0.0)


def suggest_window(energies) -> Tuple[float, float]:
    """An energy window (eV, relative to E_F) that shows the bands that matter.

    Down from the Fermi level through the valence manifolds, stopping at the
    first gap wider than :data:`CORE_GAP` — below it are core levels, flat and
    far away, and including them squashes everything else into a line. Up
    from the bottom of the conduction band by as much again as the valence
    region spans (at least 5 eV), so the gap and the lowest empty bands show.
    Rounded out to whole eV.
    """
    e = np.asarray(energies, dtype=float)
    if e.size == 0:
        return (-10.0, 10.0)
    per_band = e.reshape(e.shape[0], -1)
    lows, highs = per_band.min(axis=1), per_band.max(axis=1)
    manifolds: List[List[float]] = []
    for low, high in sorted(zip(lows, highs)):
        if manifolds and low <= manifolds[-1][1]:
            manifolds[-1][1] = max(manifolds[-1][1], high)
        else:
            manifolds.append([float(low), float(high)])

    occupied = [m for m in manifolds if m[0] <= _OCCUPIED]
    if occupied:
        index = len(occupied) - 1
        while index > 0 and occupied[index][0] - occupied[index - 1][1] <= CORE_GAP:
            index -= 1
        lower = occupied[index][0]
    else:
        lower = manifolds[0][0]
    empty = [m for m in manifolds if m[0] > _OCCUPIED]
    start = empty[0][0] if empty else 0.0
    upper = min(float(e.max()), start + max(-lower, 5.0))
    lower, upper = math.floor(lower - 0.5), math.ceil(upper + 0.5)
    return (float(lower), float(max(upper, lower + 1)))


# ── finding the files ─────────────────────────────────────────────────────
BAND_FILE = "band"
DOS_FILE = "dos"
# The extensions band and DOS data come with: CRYSTAL's BAND.DAT, DOSS.DAT and
# fort.25, and the .BAND, .DOSS and .f25 they are saved as.
_DATA_EXTENSIONS = (".dat", ".band", ".doss", ".f25", ".25")


def _head(path, lines: int = 6) -> List[str]:
    """The first few lines of a file, or nothing if it cannot be read."""
    try:
        with open(path, "r", errors="ignore") as handle:
            return [handle.readline(400) for _ in range(lines)]
    except OSError:
        return []


def file_kind(path) -> Optional[str]:
    """``"band"``, ``"dos"`` or ``None``, from what the file says it is.

    Never from its name. ``BAND.DAT``, ``mgo_band.BAND``, ``si_band.f25`` and
    ``fort.25`` are all band structures, a ``fort.25`` could as well be a
    charge-density map, and any of them can be renamed to anything. The first
    line says which: the text formats open with ``# NKPT`` (bands) or
    ``# NEPTS`` (DOS), a fort.25 with ``-%-0BAND`` or ``-%-0DOSS``.

    COOP and COHP files open with the same ``# NEPTS`` as a DOS. What tells
    them apart is the y-axis label a few lines down — "DENSITY OF STATES" in a
    DOSS.DAT (manual, Appendix D) — so a ``# NEPTS`` file whose label says
    anything else is not taken for a density of states.
    """
    head = _head(path)
    if not head:
        return None
    first = head[0].strip()
    if first.startswith("-%-"):
        tag = first[4:8].upper()
        return BAND_FILE if tag == "BAND" else DOS_FILE if tag == "DOSS" else None
    if re.match(r"# NKPT\s+\d+", first):
        return BAND_FILE
    if re.match(r"# NEPTS\s+\d+", first):
        for line in head[1:]:
            text = line.strip().upper()
            if text.startswith("@ YAXIS LABEL"):
                return DOS_FILE if "DENSITY OF STATES" in text else None
        return DOS_FILE
    return None


def _is_fort25(path) -> bool:
    head = _head(path, 1)
    return bool(head) and head[0].lstrip().startswith("-%-")


def find_files(folder, efermi: Optional[float] = None
               ) -> Tuple[List[str], List[str]]:
    """Band and DOS files in ``folder``, best first.

    Files are picked out by extension and identified by their first line —
    never by the rest of the name. Best is a text file over a fort.25 — told
    apart by that first line: a fort.25 drops the shrinking factor under its tick labels, so its
    corners cannot be named — then the newest, since ``BAND.DAT`` is
    overwritten by every run and the one just made is the one wanted.

    First of all, though, the files of the calculation that is open: a folder
    shared by several systems would otherwise offer whichever ran last. Which
    run a file came from is read from the file, not from its name — the Fermi
    level it records — and files are ordered by how close that is to
    ``efermi`` (eV, from the output). By closeness rather than by equality: an
    insulator's files carry the SCF's level to the last digit, but a metal's
    DOS is computed on a finer NEWK mesh that moves it — beryllium's by a tenth
    of an eV — and would otherwise match nothing.

    """
    found = {BAND_FILE: [], DOS_FILE: []}
    try:
        entries = list(Path(folder).iterdir())
    except OSError:
        return [], []
    for path in entries:
        if path.is_file() and path.suffix.lower() in _DATA_EXTENSIONS:
            kind = file_kind(path)
            if kind is not None:
                found[kind].append(path)

    def ranker(reader):
        def rank(path: Path):
            try:
                age = -path.stat().st_mtime
            except OSError:
                age = 0.0
            distance = 0.0
            if efermi is not None:
                level = _efermi_of(reader, str(path))
                distance = float("inf") if level is None else abs(level - float(efermi))
                # Levels this close are one SCF's; within that, the usual order.
                distance = 0.0 if distance <= _SAME_FERMI else distance
            return (distance, _is_fort25(path), age, str(path))
        return rank

    return ([str(p) for p in sorted(found[BAND_FILE], key=ranker(_read_bands))],
            [str(p) for p in sorted(found[DOS_FILE], key=ranker(_read_dos))])


# Two Fermi levels this close (eV) are the same SCF's: the output prints four
# decimals, the data files five in hartree.
_SAME_FERMI = 2e-3



def pair_files(bands: Sequence[str], doss: Sequence[str],
               tolerance: float = 1e-3, own_run: bool = False
               ) -> Tuple[Optional[str], Optional[str]]:
    """The best band file, and a DOS from the same calculation.

    Ranked separately, the newest band structure and the newest DOS can come
    from different runs — a B3LYP band structure beside a PBE DOS — and drawn
    side by side the DOS sits shifted against the bands by the difference in
    their Fermi levels. The Fermi level each file carries identifies the SCF it
    was computed from, so the first pair that agrees wins, in the band files'
    order. With no pair agreeing, the best of each.

    ``own_run`` keeps the search with the best band file's own calculation —
    band files recording its Fermi level, and no others — for when the lists
    were ranked by the open output. Otherwise a metal, whose DOS run moves the
    Fermi level with its finer mesh and so pairs with none of its band files,
    fell through to the first pair of another system that happened to agree.
    """
    if not bands or not doss:
        return (bands[0] if bands else None, doss[0] if doss else None)
    levels = [(path, _efermi_of(_read_dos, path)) for path in doss]
    first = _efermi_of(_read_bands, bands[0])
    for band in bands:
        level = _efermi_of(_read_bands, band)
        if level is None:
            continue
        if own_run and (first is None or abs(level - first) > tolerance):
            continue
        for path, other in levels:
            if other is not None and abs(level - other) <= tolerance:
                return band, path
    return bands[0], doss[0]


def _efermi_of(reader, path: str) -> Optional[float]:
    try:
        return float(reader(path).efermi)
    except Exception:  # noqa: BLE001 - an unreadable file simply pairs with nothing
        return None


# ── naming the path's corners ─────────────────────────────────────────────
def clean_tick(label) -> str:
    """A tick as the file writes it, without the Fortran quoting."""
    return str(label).strip().strip('"').strip("'").strip()


def tick_coordinates(label) -> Optional[Tuple[float, float, float]]:
    """``"(4,0,4)/8"`` -> ``(0.5, 0.0, 0.5)``, or ``None`` if it cannot be read.

    ``None`` for anything without its denominator — a ``fort.25`` writes the
    integers but not the shrinking factor they are over, and guessing it would
    name a point after one the path never visited — and for a Fortran overflow
    such as ``(*,4,3)``, where the integer did not fit its field.
    """
    match = _TICK.search(clean_tick(label))
    if match is None or match.group(4) is None:
        return None
    shrink = int(match.group(4))
    if shrink == 0:
        return None
    return tuple(float(Fraction(int(match.group(i)), shrink)) for i in (1, 2, 3))


def name_ticks(ticks: Sequence[str], structure=None) -> List[str]:
    """Names for the path's corners: Γ, X, W… where they can be worked out.

    Matched against the loaded structure's special points in its primitive
    reciprocal basis — the basis CRYSTAL writes a band path in — so the names
    are the ones the path builder shows. A corner that matches nothing, or a
    file with no structure to match against, keeps the coordinates it came
    with: a wrong name is worse than an honest coordinate.
    """
    cleaned = [clean_tick(t) for t in ticks]
    if structure is None:
        return cleaned
    try:
        from crystalline.core.brillouin import display_label, special_points, zone_lattice

        from crystalline.core.crystal_points import reciprocal_rotations, same_point

        lattice = zone_lattice(structure)
        points = special_points(lattice)
        rotations = reciprocal_rotations(lattice)
    except Exception:  # noqa: BLE001 - no names is a usable answer
        return cleaned
    named = []
    for raw in cleaned:
        coords = tick_coordinates(raw)
        name = None
        if coords is not None:
            for label, point in points.items():
                if same_point(np.asarray(coords, dtype=float), point, rotations):
                    name = display_label(label)
                    break
        named.append(name if name is not None else raw)
    return named


# ── the path a deck asked for ─────────────────────────────────────────────
@dataclass(frozen=True)
class BandDeck:
    """The BAND block of a ``.d3``: the path someone asked CRYSTAL to compute."""

    path: str
    shrink: int
    corners: Tuple[Tuple[int, int, int], ...]
    labels: Tuple[str, ...] = ()       # one per corner, "" where the deck is silent


def read_band_decks(folder) -> List[BandDeck]:
    """Every BAND block in the ``.d3`` files of ``folder``."""
    decks: List[BandDeck] = []
    try:
        entries = sorted(Path(folder).iterdir())
    except OSError:
        return decks
    for path in entries:
        if (path.is_file() and path.suffix.lower() in _DECK_EXTENSIONS
                and _is_properties_deck(path)):
            try:
                decks.extend(_band_blocks(path.read_text(errors="ignore"), str(path)))
            except OSError:
                continue
    return decks


_DECK_EXTENSIONS = (".d3", ".inp")

# A properties deck is a few dozen lines; an output, a wavefunction or a density
# matrix is megabytes. Nothing larger is opened to be looked at.
_DECK_LIMIT = 1 << 20


def _is_properties_deck(path) -> bool:
    """Whether ``path`` holds a PROPERTIES input, judged by its content.

    Not by a ``.d3`` suffix: a deck is plain text that opens on a keyword line —
    ``NEWK``, ``BAND``, ``ECH3`` — and closes with ``END``. A CRYSTAL output
    opens with its banner instead, and never passes for one.
    """
    try:
        if Path(path).stat().st_size > _DECK_LIMIT:
            return False
        text = Path(path).read_text(errors="ignore")
    except OSError:
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or not re.fullmatch(r"[A-Z][A-Z0-9_]{2,}", lines[0]):
        return False
    return any(line.upper() == "END" for line in lines)


def _band_blocks(text: str, path: str) -> List[BandDeck]:
    """Parse ``BAND`` / title / ``NLINE ISS …`` / segment lines out of a deck.

    A segment line is six integers — the two ends, over the shrinking factor —
    optionally followed by the two letters naming them. CRYSTAL also takes the
    path by label alone (``G X``, with the shrinking factor written 0), and
    such a deck carries nothing *but* the names.
    """
    blocks: List[BandDeck] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip().upper() != "BAND" or index + 2 >= len(lines):
            continue
        header = lines[index + 2].split()          # the line after the title
        try:
            n_lines, shrink = int(header[0]), int(header[1])
        except (ValueError, IndexError):
            continue
        corners: List[Tuple[int, int, int]] = []
        labels: List[str] = []
        by_name = True
        for raw in lines[index + 3: index + 3 + n_lines]:
            tokens = raw.split()
            names = [token for token in tokens if token.isalpha()]
            numbers: List[int] = []
            for token in tokens[:6]:
                try:
                    numbers.append(int(token))
                except ValueError:
                    break
            if len(numbers) == 6:
                by_name = False
                start, end = tuple(numbers[:3]), tuple(numbers[3:6])
                if not corners:
                    corners.append(start)
                elif start != corners[-1]:
                    break                           # a path with a jump in it
                corners.append(end)
            elif not (by_name and len(names) == 2):
                break                               # neither form: not a path line
            first, second = (names + ["", ""])[:2]
            if not labels:
                labels.append(first)
            labels.append(second)
        if len(labels) == n_lines + 1 and (by_name or len(corners) == n_lines + 1):
            blocks.append(BandDeck(path=path, shrink=shrink,
                                   corners=tuple(corners), labels=tuple(labels)))
    return blocks


def deck_labels(ticks: Sequence[str], folder) -> Optional[List[str]]:
    """The letters a ``.d3`` beside the file gave these corners, if one did.

    The deck is what was actually asked for, so it beats guessing — and it is
    the only way back for a corner CRYSTAL could not write down: beryllium's
    ``(-2,4,3)`` overflows its field in ``BAND.DAT`` and comes out ``(*,4,3)``,
    coordinates gone. A deck is accepted only when its shrinking factor, its
    number of corners and every coordinate still readable agree with the file,
    so a leftover deck for another path cannot rename anything.
    """
    cleaned = [clean_tick(t) for t in ticks]
    wanted = [tick_coordinates(t) for t in cleaned]
    fitting = [deck for deck in read_band_decks(folder)
               if len(deck.labels) == len(cleaned)]
    for deck in fitting:
        if deck.corners and any(deck.labels) and all(
                coords is None or np.allclose(coords, np.asarray(corner) / deck.shrink,
                                              atol=1e-6)
                for coords, corner in zip(wanted, deck.corners)):
            return list(deck.labels)
    # A path written by label alone carries no coordinates to check against, so
    # it is taken only when no other deck here could describe this path either.
    by_name = [deck for deck in fitting if not deck.corners and any(deck.labels)]
    return list(by_name[0].labels) if len(by_name) == 1 == len(fitting) else None


def path_labels(info: "BandsInfo", structure=None, folder=None) -> List[str]:
    """Names for a band file's corners, from the best source there is.

    The deck that asked for the path first, then the loaded structure's special
    points, and the raw coordinates for anything neither can name.
    """
    from crystalline.core.brillouin import display_label

    names = name_ticks(info.ticks, structure)
    given = deck_labels(info.ticks, folder or Path(info.path).parent)
    if given is None:
        return names
    return [display_label(deck) if deck else name for deck, name in zip(given, names)]


# ── the picture ───────────────────────────────────────────────────────────
@dataclass
class ElectronicOptions:
    """Everything the dialog decides, as one object the plotter takes."""

    mode: str = BANDS
    unit: str = "eV"
    reference: str = RELATIVE
    energy_range: Optional[Tuple[float, float]] = None   # in the chosen reference
    dos_range: Optional[Tuple[float, float]] = None      # None = fit what is drawn
    k_labels: Optional[List[str]] = None
    projections: Optional[List[int]] = None              # 1-based; None = all
    projection_labels: Optional[List[str]] = None        # one per chosen projection
    projection_colours: Optional[List[str]] = None       # likewise
    overlay_projections: bool = True
    spin_down: str = SPIN_MIRRORED
    band_colour: str = BAND_COLOUR
    beta_colour: str = BETA_COLOUR
    beta_style: str = "--"
    linewidth: float = 1.0
    show_fermi: bool = True
    fermi_colour: str = FERMI_COLOUR
    fermi_style: str = "-"
    fermi_width: float = 1.5
    legend: bool = True
    title: Optional[str] = None
    _extra: dict = field(default_factory=dict)


def plot_electronic(bands_path: Optional[str] = None, dos_path: Optional[str] = None,
                    options: Optional[ElectronicOptions] = None):
    """Draw a band structure, a DOS, or the two side by side. Returns a Figure."""
    from crystalline.crystalio.plotting import _plot_module, _to_figure

    options = options or ElectronicOptions()
    plot = _plot_module()
    need_bands = options.mode in (BANDS, BANDS_AND_DOS)
    need_dos = options.mode in (DOS, BANDS_AND_DOS)
    if need_bands and not bands_path:
        raise ValueError("A band structure needs a band file (BAND.DAT, .BAND or fort.25).")
    if need_dos and not dos_path:
        raise ValueError("A density of states needs a DOS file (DOSS.DAT, .DOSS or fort.25).")

    bands = _read_bands(bands_path) if need_bands else None
    doss = _read_dos(dos_path) if need_dos else None

    # Everything is read in eV, relative to E_F. An absolute plot moves the data
    # up by E_F *before* CRYSTALClear converts units, so the shift and the unit
    # conversion compose the way they should.
    efermi_ev = float((bands if bands is not None else doss).efermi)
    shift = efermi_ev if options.reference == ABSOLUTE else 0.0
    if shift:
        if bands is not None:
            bands.bands[...] += shift
        if doss is not None:
            doss.doss[:, 0, :] += shift

    fermi_at = shift if options.unit == "eV" else shift / _HARTREE_EV
    common = dict(unit=options.unit, energy_range=_span(options.energy_range),
                  fermi=FERMI_COLOUR, title=options.title or None)

    if options.mode == BANDS:
        beta = None
        if bands.spin == 2:
            # CRYSTALClear draws both channels alike here; α goes through it,
            # β is laid over afterwards where it can be told apart.
            beta = bands.bands[:, :, 1].copy()
            bands.bands = bands.bands[:, :, 0:1].copy()
            bands.spin = 1
        figure = _to_figure(plot.plot_electron_band(
            bands, k_labels=options.k_labels, color=options.band_colour,
            linewidth=options.linewidth, **common))
        if beta is not None:
            if options.unit != "eV":
                beta = beta / _HARTREE_EV
            _overlay_beta(figure, beta, bands.n_kpoints, options)
        energy_axis = "y"

    elif options.mode == DOS:
        chosen = _projections(options.projections, doss.n_proj)
        colours, names = _colours(options, chosen), _labels(options, chosen)
        overlay = options.overlay_projections
        # One panel per projection takes a single colour and no labels —
        # CRYSTALClear swaps a list for 'blue' and drops the names — so each
        # panel is coloured and named here instead.
        figure = _to_figure(plot.plot_electron_dos(
            doss, beta=options.spin_down, overlap=overlay,
            prj=chosen, dos_range=_span(options.dos_range),
            color=colours if overlay else colours[0],
            labels=names if overlay else None,
            linewidth=options.linewidth, **common))
        if not overlay:
            for ax, colour, name in zip(figure.axes, colours, names):
                for line in ax.get_lines():
                    if len(line.get_xdata()) > 2:
                        line.set_color(colour)
                if options.legend:
                    ax.text(0.99, 0.92, name, transform=ax.transAxes,
                            ha="right", va="top", fontsize="small")
        elif not options.legend:
            for ax in figure.axes:
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
        energy_axis = "x"

    else:
        chosen = _projections(options.projections, doss.n_proj)
        figure = _to_figure(plot.plot_electron_banddos(
            bands, doss, k_labels=options.k_labels, dos_beta=options.spin_down,
            dos_prj=chosen, dos_range=_span(options.dos_range),
            color_band=options.band_colour, color_dos=_colours(options, chosen),
            labels=_labels(options, chosen), linewidth=options.linewidth,
            legend=options.legend and len(chosen) > 1, **common))
        if bands.spin == 2:
            _restyle_beta(figure.axes[0], bands.n_kpoints, options)
        _legend_beside(figure, figure.axes[-1])
        energy_axis = "y"

    _style_fermi(figure, energy_axis, fermi_at, options)
    _label_energy(figure, energy_axis, options)
    return figure


# ── finishing touches CRYSTALClear does not offer ─────────────────────────
def _style_fermi(figure, axis: str, at: float, options: ElectronicOptions) -> None:
    """Move CRYSTALClear's Fermi lines to ``at`` and style them — or hide them.

    They are drawn at zero, as two-point lines in :data:`FERMI_COLOUR`,
    whatever the energy scale — found here by exactly that, so the black
    zero-lines beside them (the start of the k axis, the zero of the DOS) are
    left alone.
    """
    from matplotlib.colors import to_rgba

    target = to_rgba(FERMI_COLOUR)
    for ax in figure.axes:
        for line in ax.get_lines():
            data = line.get_ydata() if axis == "y" else line.get_xdata()
            if len(data) != 2 or not np.allclose(to_rgba(line.get_color()), target):
                continue
            if not np.allclose(np.asarray(data, dtype=float), 0.0):
                continue
            if axis == "y":
                line.set_ydata([at, at])
            else:
                line.set_xdata([at, at])
            line.set_color(options.fermi_colour)
            line.set_linestyle(options.fermi_style)
            line.set_linewidth(options.fermi_width)
            line.set_visible(options.show_fermi)


def _label_energy(figure, axis: str, options: ElectronicOptions) -> None:
    """Say which energy the axis carries — CRYSTALClear's labels can't know.

    Its combined plot calls the axis plain "Energy" although it is E − E_F,
    and its standalone band label sets a fixed size that ignores the figure
    font the app lets people choose.
    """
    unit = options.unit
    text = (rf"$E - E_\mathrm{{F}}$ ({unit})" if options.reference == RELATIVE
            else rf"$E$ ({unit})")
    if axis == "y":
        figure.supylabel(text)
    else:
        figure.supxlabel(text)


def _spin_legend(ax, options: ElectronicOptions) -> None:
    from matplotlib.lines import Line2D

    ax.legend(handles=[
        Line2D([], [], color=options.band_colour, linestyle="-", label="α (spin up)"),
        Line2D([], [], color=options.beta_colour, linestyle=options.beta_style,
               label="β (spin down)"),
    ], loc="best", fontsize="small")


def _overlay_beta(figure, beta: np.ndarray, n_kpoints: int,
                  options: ElectronicOptions) -> None:
    """Draw the spin-down bands over the α ones, on the same k axis."""
    ax = figure.axes[0]
    x = next((np.asarray(line.get_xdata(), dtype=float) for line in ax.get_lines()
              if len(line.get_xdata()) == n_kpoints), None)
    if x is None:
        return
    for band in beta:
        ax.plot(x, band, color=options.beta_colour, linestyle=options.beta_style,
                linewidth=options.linewidth)
    if options.legend:
        _spin_legend(ax, options)


def _legend_beside(figure, ax) -> None:
    """Move the DOS legend out beside its panel.

    The combined view's DOS panel is a third of the figure wide, and
    CRYSTALClear puts the legend inside it: with nine projections the legend
    was bigger than the curves it named, and covered them.
    """
    legend = ax.get_legend()
    if legend is None:
        return
    handles, labels = legend.legend_handles, [t.get_text() for t in legend.get_texts()]
    legend.remove()
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.03, 1.0),
              borderaxespad=0.0, fontsize="small", frameon=False)
    figure.subplots_adjust(right=0.78)


def _restyle_beta(ax, n_kpoints: int, options: ElectronicOptions) -> None:
    """Give the combined plot's β bands the colour and style asked for.

    CRYSTALClear draws them dashed, in the α colour; that dash is how they are
    recognised here.
    """
    for line in ax.get_lines():
        if len(line.get_xdata()) == n_kpoints and line.get_linestyle() == "--":
            line.set_color(options.beta_colour)
            line.set_linestyle(options.beta_style)
    if options.legend:
        _spin_legend(ax, options)
    elif ax.get_legend() is not None:
        ax.get_legend().remove()


# ── plumbing ──────────────────────────────────────────────────────────────
def _read_bands(path: str):
    from CRYSTALClear.crystal_io import Properties_output

    return Properties_output().read_electron_band(path)


def _read_dos(path: str):
    from CRYSTALClear.crystal_io import Properties_output

    return Properties_output().read_electron_dos(path)


def _span(value) -> Optional[List[float]]:
    """A ``(low, high)`` as CRYSTALClear wants it; a zero width means "don't clip"."""
    if value is None:
        return None
    low, high = sorted(float(v) for v in value)
    return None if low == high else [low, high]


def _projections(chosen: Optional[Sequence[int]], available: int) -> List[int]:
    if not chosen:
        return list(range(1, available + 1))
    return [int(p) for p in chosen if 1 <= int(p) <= available] or [1]


def _colours(options: ElectronicOptions, chosen: Sequence[int]) -> List[str]:
    given = options.projection_colours
    if given and len(given) == len(chosen):
        return list(given)
    return [PROJECTION_COLOURS[(p - 1) % len(PROJECTION_COLOURS)] for p in chosen]


def _labels(options: ElectronicOptions, chosen: Sequence[int]) -> List[str]:
    given = options.projection_labels
    if given and len(given) == len(chosen):
        return [str(name) for name in given]
    return [f"Projection {p}" for p in chosen]


__all__ = [
    "ABSOLUTE",
    "BANDS",
    "BANDS_AND_DOS",
    "BAND_COLOUR",
    "BETA_COLOUR",
    "DOS",
    "FERMI_COLOUR",
    "LINE_STYLES",
    "MODES",
    "PROJECTION_COLOURS",
    "REFERENCES",
    "RELATIVE",
    "SPIN_ALONGSIDE",
    "SPIN_MIRRORED",
    "UNITS",
    "BandsInfo",
    "DosInfo",
    "ElectronicOptions",
    "clean_tick",
    "describe_bands",
    "describe_dos",
    "file_kind",
    "find_files",
    "BandDeck",
    "deck_labels",
    "name_ticks",
    "pair_files",
    "path_labels",
    "read_band_decks",
    "plot_electronic",
    "suggest_window",
    "tick_coordinates",
]
