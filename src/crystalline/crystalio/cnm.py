"""The characterisation of normal modes (CNM) of a LOCMODES run, and its figures.

LOCMODES decomposes each normal mode into the local modes of the run's internal
coordinates: how much of mode 100 is the N–O stretch, how much the C–N–O bend.
CRYSTALClear (from the release after 0.2.17) reads that decomposition with
``Crystal_output.get_locmodes`` and, asked to, sums it over the coordinates the
crystal's symmetry makes equivalent — the eight N–O bonds of a P2₁/c cell with
two molecules are one column rather than eight. It draws it two ways, and this
module is the single point of contact with both:

* :func:`plot_cnm` — one run, one stacked bar per normal mode
  (``plot_cry_cnm``).
* :func:`plot_cnm_series` — one normal mode followed along a series of runs,
  at increasing pressure say, one bar per run (``plot_cry_cnm_pressure``). The
  mode is followed by the overlap of its eigenvector from one run to the next,
  so a series needs every run's eigenvectors.

A run is read once (:func:`cnm_run`) and drawn as often as its controls are
moved; one run's modes are chosen by number (a window of frequencies only
ticks a span of them at once). CRYSTALClear colours a group it is not given a colour for at random —
differently at every draw — so the groups are given the house colours here,
and each plot keeps the colour it gave a group for as long as it is open.

Kept Qt-free, like the sibling modules.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from crystalline.crystalio.locmodes import reads_by_symmetry

# Contributions smaller than this (%) are summed into the grey "Residues" at the
# top of each bar: plot_cry_cnm's own default.
DEFAULT_THRESHOLD = 1.0
# Below this eigenvector overlap a mode followed along a series is no longer
# surely the same mode: plot_cry_cnm_pressure's own default.
DEFAULT_MIN_OVERLAP = 0.5
# How many bars a figure can hold and still be read, and how many the window
# opens on: a run has hundreds of modes, at half an inch a bar.
MAX_MODES = 40
DEFAULT_WINDOW_MODES = 12
# What the bars of one run are labelled with, as (label, plot_cry_cnm's name).
TICKS = (("Frequency", "freq"), ("Mode number", "mode"))
# What the series' axis is named unless told otherwise.
DEFAULT_SERIES_AXIS = "Pressure (GPa)"
# The legend's rows a column, and its type size: what fits in the height the
# Plots window opens at, about five inches, where CRYSTALClear's forty rows at
# 9 pt suit its own figure, ten inches tall, and ran off the top and the
# bottom (thirty still lost one at each end).
LEGEND_ROWS = 25
LEGEND_SIZE = 8
# Past this many groups a legend is a wall of names: the count line says so.
READABLE_GROUPS = 60

RESIDUES = "Residues"


@dataclass(frozen=True)
class CNMRun:
    """A LOCMODES run read for its CNM: enough to drive the controls.

    ``output`` is CRYSTALClear's ``Crystal_output`` the figures are drawn from,
    with ``get_locmodes`` and ``get_phonon`` already called on it. ``note`` says
    why the coordinates are not grouped by symmetry, when they are not:
    CRYSTALClear checks the bonds it prints against the geometry it reads, and
    a run it cannot check is drawn one coordinate per column instead.
    """

    path: str
    output: object = field(repr=False, compare=False)
    modes: Tuple[int, ...]              # the normal modes characterised, ascending
    frequencies: Tuple[float, ...]      # cm⁻¹, one per mode in ``modes``
    coordinates: int                    # internal coordinates LOCMODES kept
    groups: int                         # columns of the CNM: sets, or coordinates
    spacegroup: Optional[str] = None
    note: Optional[str] = None
    eigenvectors: bool = False          # whether a series can follow its modes

    @property
    def grouped(self) -> bool:
        """Whether equivalent coordinates are summed into one column."""
        return self.note is None

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def summary(self) -> str:
        """One line describing the run, for the dialog's header."""
        modes = f"{len(self.modes)} normal modes"
        if self.grouped:
            group = f" in {_spacegroup(self.spacegroup)}" if self.spacegroup else ""
            return (f"{modes}, {self.coordinates} internal coordinates in "
                    f"{self.groups} sets equivalent by symmetry{group}")
        return (f"{modes}, {self.coordinates} internal coordinates, each on its "
                f"own: not grouped by symmetry ({self.note})")

    def frequency(self, mode: int) -> float:
        return self.frequencies[self.modes.index(mode)]

    def label(self, mode: int) -> str:
        """``"Mode 100, 469.3 cm⁻¹"``: how a mode is offered to be chosen."""
        return f"Mode {mode}, {self.frequency(mode):.1f} cm⁻¹"

    def modes_in(self, fmin: float, fmax: float) -> List[int]:
        """The modes whose frequency is in ``[fmin, fmax]``, both edges included."""
        low, high = sorted((fmin, fmax))
        return [mode for mode, frequency in zip(self.modes, self.frequencies)
                if low <= frequency <= high]

    def count_in(self, fmin: float, fmax: float) -> int:
        return len(self.modes_in(fmin, fmax))

    def groups_in(self, modes: Sequence[int], threshold: float = DEFAULT_THRESHOLD) -> int:
        """How many groups the bars of ``modes`` show, each in a colour and named
        in the legend: those that reach ``threshold`` (%) in one of them, as
        ``plot_cry_cnm`` keeps them."""
        modes = [m for m in modes if m in self.modes]
        table = getattr(self.output, "cnm_sym", None)
        if not modes or table is None:
            return 0
        return int((table.loc[modes] >= float(threshold)).any(axis=0).sum())

    def default_modes(self, around: Optional[int] = None) -> List[int]:
        """The modes ticked to begin with: those of :meth:`default_window`."""
        return self.modes_in(*self.default_window(around))

    def default_window(self, around: Optional[int] = None) -> Tuple[float, float]:
        """A window of :data:`DEFAULT_WINDOW_MODES` modes, in cm⁻¹, edges rounded out.

        Around mode ``around`` when it is one of the run's — the mode selected in
        the Phonons panel, which is usually the question — else from the lowest.
        """
        if not self.modes:
            return 0.0, 100.0
        start = 0
        if around in self.modes:
            start = max(0, self.modes.index(around) - DEFAULT_WINDOW_MODES // 2)
        start = min(start, max(0, len(self.modes) - DEFAULT_WINDOW_MODES))
        chosen = self.frequencies[start:start + DEFAULT_WINDOW_MODES]
        # Half a wavenumber out from the first and last, so that rounding the
        # boxes' display never drops a mode on the edge.
        return float(int(chosen[0] - 0.5)), float(int(chosen[-1] + 0.5) + 1)


def _spacegroup(text: str) -> str:
    """spglib's ``"P2_1/c (14)"`` as it is written: P2₁/c (No. 14)."""
    symbol, _, number = str(text).partition(" ")
    symbol = symbol.replace("_1", "₁").replace("_2", "₂").replace("_3", "₃") \
                   .replace("_4", "₄").replace("_5", "₅").replace("_6", "₆")
    number = number.strip("()")
    return f"{symbol} (No. {number})" if number else symbol


# ── what the installed CRYSTALClear can do ─────────────────────────────────
def _crystalclear():
    """``(Crystal_output, plot module)``, or ``(None, None)`` if they will not import."""
    try:
        from CRYSTALClear import plot as CCplt
        from CRYSTALClear.crystal_io import Crystal_output
    except Exception:  # noqa: BLE001 - CRYSTALClear missing/broken
        return None, None
    return Crystal_output, CCplt


def plottable() -> bool:
    """Whether the installed CRYSTALClear can read a run's CNM and draw it.

    Both arrived together, after 0.2.17: ``get_locmodes`` learnt to read the
    decomposition (and to group it by symmetry, its ``symmetry`` argument) in
    the same change that added ``plot_cry_cnm``. 0.2.17 reads the local modes
    the Measure section shows, and offers no CNM.
    """
    Crystal_output, CCplt = _crystalclear()
    if Crystal_output is None or not hasattr(CCplt, "plot_cry_cnm"):
        return False
    return reads_by_symmetry()


def series_plottable() -> bool:
    """Whether it can also follow a mode along a series of runs."""
    _Crystal_output, CCplt = _crystalclear()
    return plottable() and hasattr(CCplt, "plot_cry_cnm_pressure")


# ── reading ─────────────────────────────────────────────────────────────────
def cnm_run(path: str) -> CNMRun:
    """Read the CNM of the LOCMODES run at ``path``, grouped by symmetry if it can be.

    Raises when the run has no CNM to draw — a LOCMODES section without the
    decomposition, or a CRYSTALClear too old to read it — with a message that
    says which. The eigenvectors are read too when the output prints them, so
    that the run can be followed along a series without being read again.
    """
    from CRYSTALClear.crystal_io import Crystal_output
    from CRYSTALClear.units import thz_to_cm

    if not plottable():
        raise RuntimeError("the CNM needs a CRYSTALClear newer than 0.2.17")
    with contextlib.redirect_stdout(io.StringIO()):  # it prints as it parses
        output = Crystal_output(path)
        note = None
        try:
            output.get_locmodes(symmetry=True)
        except Exception as exc:  # noqa: BLE001 - drawn ungrouped, and said why
            note = _reason(exc)
            output.get_locmodes(symmetry=False)
        eigenvectors = True
        try:
            output.get_phonon(read_eigvt=True)
        except Exception:  # noqa: BLE001 - no eigenvectors printed: one run only
            eigenvectors = False
            output.get_phonon()
    modes = tuple(int(mode) for mode in output.cnm_modes)
    if not modes:
        raise RuntimeError("the LOCMODES section has no characterisation of normal modes")
    every = [float(f) for f in thz_to_cm(output.frequency[0])]
    frequencies = tuple(every[mode - 1] for mode in modes)
    found = getattr(output, "eigenvector", None)
    eigenvectors = eigenvectors and found is not None and len(found) > 0
    return CNMRun(
        path=path, output=output, modes=modes, frequencies=frequencies,
        coordinates=int(len(output.cnm)), groups=int(output.cnm_sym.shape[1]),
        spacegroup=getattr(output, "ic_spacegroup", None), note=note,
        eigenvectors=eigenvectors,
    )


def _reason(exc: BaseException) -> str:
    """The first sentence of an exception, which is what a line can hold."""
    text = str(exc).strip() or type(exc).__name__
    first = text.split(": ", 1)[0] if len(text) > 120 else text
    return first.rstrip(".")


def series_labels(paths: Sequence[str]) -> List[str]:
    """What tells each run of a series apart, from its file name.

    The names are taken in their parts (between ``_``, ``-``, ``.`` and
    spaces), and the parts every run shares are dropped; of what is left, the
    parts with a number in them, when there are any, since that is usually
    what the series runs over. ``DAFADNP_FREQ_LOCMODES_02GPa_bugfix.out``
    beside ``…_04GPa_bugfix.out`` and ``DAFADNP_FREQ_tight_LOCMODES.out`` is
    ``02GPa``, the last ``tight``. CRYSTAL does not print the pressure a
    geometry was optimised at, so this is a starting point for the label, to
    be written as the number it stands for. A single run, or names that come
    out alike, keep their whole stem.
    """
    stems = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    if len(stems) < 2:
        return stems
    parts = [[t for t in re.split(r"[_\-. ]+", stem) if t] for stem in stems]
    shared = set.intersection(*(set(p) for p in parts))
    labels = []
    for stem, mine in zip(stems, parts):
        rest = [t for t in mine if t not in shared]
        numbered = [t for t in rest if any(c.isdigit() for c in t)]
        labels.append("_".join(numbered or rest) or stem)
    return labels if len(set(labels)) == len(labels) else stems


def legend_columns(groups: int) -> int:
    """How many columns a legend of ``groups`` (and the residues) takes."""
    return max(1, math.ceil((groups + 1) / LEGEND_ROWS))


def figure_aspect(run: CNMRun, options: dict) -> float:
    """Width over height a figure of these options wants: the Plots window's.

    Half an inch a bar on a figure five inches tall, and an inch and a third
    for each column of the legend beside them; never narrower than 4:3.
    """
    modes = options.get("modes") or run.default_modes()
    bars = min(len(modes), MAX_MODES)
    legend = 0.0
    if options.get("legend", True):
        threshold = options.get("threshold", DEFAULT_THRESHOLD)
        legend = 1.35 * legend_columns(run.groups_in(modes, threshold))
    return max(4 / 3, (0.45 * bars + 0.9 + legend) / 5.0)


def series_aspect(runs: int, figure=None) -> float:
    """Width over height a series of ``runs`` wants, with the legend ``figure`` drew.

    Three quarters of an inch a run, as CRYSTALClear gives it, and the legend's
    columns beside: which groups it names is known only once the runs have
    been matched and drawn, so it is read off the figure drawn last.
    """
    columns = getattr(figure, LEGEND_COLUMNS, 0) if figure is not None else 0
    return max(4 / 3, (0.75 * runs + 0.9 + 1.35 * columns) / 5.0)


# Set on a CNM figure: how many columns its legend has (0 without one).
LEGEND_COLUMNS = "_crystalline_legend_columns"


# ── colour ──────────────────────────────────────────────────────────────────────
def palette() -> Tuple[str, ...]:
    """The colours groups are given, in order: the house curve colours, then
    lighter and darker versions of each — a run's figure can hold a couple of
    dozen groups, and eight would repeat."""
    from matplotlib.colors import to_hex, to_rgb

    from crystalline.crystalio.figure_style import CYCLE, INK

    def mix(colour, other, amount):
        a, b = to_rgb(colour), to_rgb(other)
        return to_hex(tuple(x + (y - x) * amount for x, y in zip(a, b)))

    return (tuple(CYCLE) + tuple(mix(c, "#ffffff", 0.45) for c in CYCLE)
            + tuple(mix(c, INK, 0.35) for c in CYCLE))


# What the house colours are called where one is chosen: the curve colours in
# figure_style.CYCLE's order, each then lighter and darker, as palette() has them.
_COLOUR_NAMES = ("cyan", "orange", "green", "purple", "red", "slate", "gold", "navy")


def named_palette() -> Tuple[Tuple[str, str], ...]:
    """``(name, colour)`` for every colour of :func:`palette`, in its order:
    "Cyan", …, then "Light cyan", …, then "Dark cyan", …"""
    names = ([n.capitalize() for n in _COLOUR_NAMES]
             + [f"Light {n}" for n in _COLOUR_NAMES]
             + [f"Dark {n}" for n in _COLOUR_NAMES])
    return tuple(zip(names, palette()))


# Everything under the threshold: quiet, as the brand guide asks of whatever is
# not the point (its border grey), with ink on it.
RESIDUES_COLOUR = "#d8dbe2"


def drawn_groups(figure) -> List[Tuple[str, str]]:
    """``(group, colour)`` for every group a CNM figure shows, in the legend's
    order, the residues last: what can be given a colour of its own."""
    from matplotlib.colors import to_hex
    from matplotlib.container import BarContainer

    if figure is None or not figure.axes:
        return []
    found = [(c.get_label(), to_hex(c.patches[0].get_facecolor()))
             for c in figure.axes[0].containers
             if isinstance(c, BarContainer) and c.patches]
    return ([g for g in found if g[0] != RESIDUES]
            + [g for g in found if g[0] == RESIDUES])


def assign_colours(groups: Sequence[str], memory: Dict[str, str]) -> Dict[str, str]:
    """A colour for each of ``groups``, kept in ``memory`` from one draw to the next.

    A group drawn before keeps its colour, so that moving the window or the
    threshold does not repaint every bar; a new one takes the first colour of
    the palette that no other group in this figure has — or, once every colour
    is on one, the palette again from the start. ``memory`` is updated.
    """
    colours = palette()
    taken = {memory[g] for g in groups if g in memory}
    again = 0
    for group in groups:
        if group in memory:
            continue
        free = [c for c in colours if c not in taken]
        if free:
            chosen = free[0]
        else:
            chosen, again = colours[again % len(colours)], again + 1
        memory[group] = chosen
        taken.add(chosen)
    return {group: memory[group] for group in groups}


def _text_colour(face) -> str:
    """Ink or white, whichever reads on ``face`` (CRYSTALClear's own rule)."""
    from matplotlib.colors import to_rgb

    from crystalline.crystalio.figure_style import INK

    r, g, b = to_rgb(face)
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b <= 0.5 else INK


def _restyle(figure, memory: Dict[str, str], legend: bool,
             chosen: Optional[Dict[str, str]] = None) -> None:
    """The house colours and lines on a CNM figure CRYSTALClear has drawn.

    The bars are recoloured group by group (a ``BarContainer`` per group,
    labelled with it), the numbers printed on them take the colour that reads
    on their new bar, and the legend is drawn again from the bars — it holds
    copies of their colours, not the bars — or taken away. ``chosen`` are the
    colours the user gave groups (the residues among them): they go into the
    plot's memory, so a group keeps one once chosen, whatever else changes.
    """
    from matplotlib.container import BarContainer

    from crystalline.crystalio.figure_style import INK, STYLE

    ax = figure.axes[0]
    bars = [c for c in ax.containers if isinstance(c, BarContainer)]
    groups = [c.get_label() for c in bars if c.get_label() != RESIDUES]
    memory.update(chosen or {})
    colours = assign_colours(groups, memory)
    for container in bars:
        face = memory.get(RESIDUES, RESIDUES_COLOUR) if container.get_label() == RESIDUES \
            else colours[container.get_label()]
        for patch in container.patches:
            patch.set_facecolor(face)
            # Thin, in ink: CRYSTALClear's 1.5 pt black edges outweigh the
            # thinnest segments they separate.
            patch.set_edgecolor(INK)
            patch.set_linewidth(0.5)
    for text in ax.texts:
        # A bar's number is centred on it; the lowest overlap printed over a
        # hatched bar sits on its top edge and keeps the ink it was given.
        if text.get_verticalalignment() != "center":
            continue
        x, y = text.get_position()
        for container in bars:
            hit = next((p for p in container.patches
                        if p.get_x() <= x <= p.get_x() + p.get_width()
                        and p.get_y() <= y <= p.get_y() + p.get_height()), None)
            if hit is not None:
                text.set_color(_text_colour(hit.get_facecolor()))
                break
    for spine in ax.spines.values():
        spine.set_linewidth(STYLE["axes.linewidth"])
    # One bar per mode or per run: there is nothing between two to tick.
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.tick_params(axis="x", which="major", top=False)
    setattr(figure, LEGEND_COLUMNS, 0)
    old = ax.get_legend()
    if old is None:
        return
    old.remove()
    if legend:
        columns = legend_columns(len(groups))
        ax.legend(ncol=columns, loc="center left", bbox_to_anchor=(1, 0.5),
                  prop={"size": LEGEND_SIZE})
        setattr(figure, LEGEND_COLUMNS, columns)


# ── drawing ─────────────────────────────────────────────────────────────────
def plot_cnm(run: CNMRun, *, modes: Sequence[int], threshold: float = DEFAULT_THRESHOLD,
             ticks: str = "freq", legend: bool = True,
             group_colours: Optional[Dict[str, str]] = None,
             colours: Optional[Dict[str, str]] = None):
    """The CNM of ``modes`` of ``run`` (CRYSTAL's numbers), one bar each, as a figure.

    The bars stand in the order of the modes, whatever order they are given
    in. ``group_colours`` are colours chosen for groups by name (``"1 NO"``,
    or :data:`RESIDUES`). ``colours`` is the plot's memory of which group it
    drew in which colour (see :func:`assign_colours`); a plot drawn again
    passes the same dict.
    """
    from crystalline.crystalio.plotting import _plot_module

    CCplt = _plot_module()
    modes = sorted(set(int(m) for m in modes))
    unknown = [m for m in modes if m not in run.modes]
    if unknown:
        raise ValueError("not characterised in this run: mode "
                         + ", ".join(str(m) for m in unknown))
    if not modes:
        raise ValueError("no normal modes chosen")
    if len(modes) > MAX_MODES:
        raise ValueError(f"{len(modes)} modes chosen, more than {MAX_MODES}")
    figure, _ax, _used = CCplt.plot_cry_cnm(
        run.output, modes=modes, threshold=float(threshold), xticks=ticks)
    _restyle(figure, colours if colours is not None else {}, legend, group_colours)
    return figure


def plot_cnm_series(runs: Sequence[CNMRun], *, labels: Sequence[str], mode: int,
                    threshold: float = DEFAULT_THRESHOLD,
                    min_overlap: float = DEFAULT_MIN_OVERLAP,
                    axis: str = DEFAULT_SERIES_AXIS, legend: bool = True,
                    frequencies: bool = True,
                    group_colours: Optional[Dict[str, str]] = None,
                    colours: Optional[Dict[str, str]] = None):
    """Mode ``mode`` of the first run followed along ``runs``, one bar per run.

    ``labels`` name the runs in the order given, which is the order of the
    series, each under its bar; ``axis`` names what they are. With
    ``frequencies`` CRYSTALClear prints the mode's frequency in each run under
    its label, and adds "(frequency [cm⁻¹])" to the axis; without, each bar
    carries its label alone. A bar reached through an overlap below
    ``min_overlap``, and every bar after it, is hatched: from there the mode
    followed may be another.
    ``group_colours`` and ``colours`` as for :func:`plot_cnm`.
    """
    from crystalline.crystalio.plotting import _plot_module

    CCplt = _plot_module()
    if len(runs) < 2:
        raise ValueError("a series needs two runs or more")
    if any(not str(label).strip() for label in labels):
        raise ValueError("every run of the series needs a label")
    if len(set(labels)) != len(labels):
        raise ValueError("each run of the series needs a label of its own")
    missing = [run.name for run in runs if not run.eigenvectors]
    if missing:
        raise ValueError("no eigenvectors printed in " + ", ".join(missing)
                         + ": a mode cannot be followed without them")
    with contextlib.redirect_stdout(io.StringIO()):
        figure, _ax, _used = CCplt.plot_cry_cnm_pressure(
            [run.output for run in runs], int(mode), labels=[str(l) for l in labels],
            threshold=float(threshold), min_overlap=float(min_overlap), xlabel=axis)
    if not frequencies:
        ax = figure.axes[0]
        ax.set_xticklabels([str(label) for label in labels])
        ax.set_xlabel(axis, fontsize=ax.xaxis.label.get_fontsize())
    _restyle(figure, colours if colours is not None else {}, legend, group_colours)
    return figure


__all__ = [
    "CNMRun",
    "DEFAULT_MIN_OVERLAP",
    "DEFAULT_SERIES_AXIS",
    "DEFAULT_THRESHOLD",
    "DEFAULT_WINDOW_MODES",
    "LEGEND_ROWS",
    "MAX_MODES",
    "RESIDUES",
    "TICKS",
    "assign_colours",
    "cnm_run",
    "drawn_groups",
    "figure_aspect",
    "legend_columns",
    "named_palette",
    "palette",
    "plot_cnm",
    "plot_cnm_series",
    "plottable",
    "reads_by_symmetry",
    "series_aspect",
    "series_labels",
    "series_plottable",
]
