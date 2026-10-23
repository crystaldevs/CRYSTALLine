"""Adapter over ``CRYSTALClear.plot``: turn CRYSTAL results into matplotlib
figures the UI can embed.

Two kinds of source, distinguished by :attr:`PlotKind.source`:

* ``"output"`` — read straight from the CRYSTAL ``.out`` via ``Crystal_output``
  (elastic surfaces, equation of state). These need no extra files, so the UI
  reuses the already-loaded output. Vibrational spectra are *not* among them:
  they run to dozens of curves per file and have their own dialog, in
  :mod:`crystalline.crystalio.spectra`.
* ``"data"`` — need a PROPERTIES data file: the XRD pattern. Band
  structures and densities of states, electronic and phonon, have dialogs of
  their own.

Kept Qt-free (returns a bare ``matplotlib.figure.Figure``) so the mapping from
"kind of plot" → "reader + plotter" lives in one testable place, mirroring how
:mod:`crystalline.crystalio.loader` is the single point of contact for the
geometry/phonon side of CRYSTALClear.

Each :class:`PlotKind` pairs a menu label (and, for data plots, a file filter)
with a ``build(path) -> Figure`` callable. Adding a plot is one entry in
:func:`available_plots`; the UI builds its menu straight from that list.

CRYSTALClear (and matplotlib) are imported lazily inside the builders so the
rest of the package stays importable — and unit-testable — without them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional


class PlotUnavailable(RuntimeError):
    """CRYSTALClear (or a plotting dependency) isn't installed."""


@dataclass(frozen=True)
class PlotKind:
    """One entry in the Plot menu: how to turn a file into a figure.

    Attributes
    ----------
    key:      stable identifier (also handy in tests).
    label:    menu text.
    source:   ``"output"`` (a CRYSTAL ``.out``) or ``"data"`` (a PROPERTIES
              data file). Output plots reuse the loaded output; data plots prompt.
    caption:  the file dialog's title (used when a file must be chosen).
    file_filter: Qt ``getOpenFileName`` filter for the expected file.
    build:    ``path -> matplotlib.figure.Figure``.
    group:    optional submenu name (e.g. "Elastic properties").
    """

    key: str
    label: str
    source: str
    caption: str
    file_filter: str
    build: Callable[[str], "object"]
    group: Optional[str] = None
    # For ``source="output"``: ``(getter, attribute)`` used to test whether the
    # loaded output actually contains this plot's data (see output_availability).
    probe: Optional[tuple] = None


# ── figure extraction ──────────────────────────────────────────────────────
def _to_figure(result):
    """Normalise a CRYSTALClear plot function's return into a single Figure.

    The plot functions are inconsistent: most return ``(fig, ax)``, some return
    a bare ``fig``, a few draw onto the pyplot state and return ``None``, and
    ``plot_cry_ela`` returns ``(fig_list, ax_list, plt_list)``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    if isinstance(result, Figure):
        return result
    if isinstance(result, tuple) and result:
        first = result[0]
        if isinstance(first, Figure):
            return first
        if isinstance(first, (list, tuple)) and first and isinstance(first[0], Figure):
            return first[0]
    return plt.gcf()  # plt-based plotters (e.g. plot_cry_xrd) draw on the current fig


def _require_crystalclear():
    try:
        from CRYSTALClear import crystal_io, plot  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - surface a clear, actionable message
        raise PlotUnavailable(
            "CRYSTALClear is required for plotting but could not be imported "
            f"({exc}). Install it with `pip install CRYSTALClear`."
        ) from exc


def _plot_module():
    """Import CRYSTALClear.plot with a non-interactive matplotlib backend.

    Agg keeps CRYSTALClear's internal ``plt.show()`` (e.g. in ``plot_cry_xrd``)
    from popping a stray window; the figure is embedded in Qt afterwards, which
    is backend-independent.
    """
    _require_crystalclear()
    import matplotlib

    matplotlib.use("Agg")
    from CRYSTALClear import plot as CCplt

    # Every plot comes through here, CRYSTALClear's and the app's own, so this
    # is where the house style is put in place: cheap, and idempotent.
    from crystalline.crystalio import figure_style

    figure_style.apply()
    if not _font_applied:
        apply_font()
    return CCplt


# ── builders: from a CRYSTAL .out via Crystal_output ────────────────────────
def _elastic_builder(menu_key: str) -> Callable[[str], object]:
    """The elastic view a menu entry opens on, drawn as its tab draws it."""

    def build(path: str):
        from crystalline.crystalio import elastic

        return elastic.plot_elastic(elastic.read_tensor(path), **elastic.options_for(menu_key))

    return build


def _eos_builder() -> Callable[[str], object]:
    """Equation-of-state E(V) fit from an EOS run."""

    def build(path: str):
        CCplt = _plot_module()
        from CRYSTALClear.crystal_io import Crystal_output

        co = Crystal_output(path)
        co.get_EOS()
        return _to_figure(CCplt.plot_cry_EOS(co))

    return build


# ── builders: from a PROPERTIES/dispersion data file ────────────────────────
def _prop_builder(read_method: str, plot_func: str) -> Callable[[str], object]:
    """Read with ``Properties_output`` and plot with ``plot_func``."""

    def build(path: str):
        CCplt = _plot_module()
        from CRYSTALClear.crystal_io import Properties_output

        reader = Properties_output()
        data = getattr(reader, read_method)(path)
        obj = data if data is not None else reader  # readers may return self/None
        return _to_figure(getattr(CCplt, plot_func)(obj))

    return build


# ── registry ────────────────────────────────────────────────────────────────
_OUT = "CRYSTAL output (*.out *.outp);;All files (*)"
_DAT = "Data files (*.DAT *.dat *.f25 fort.25);;All files (*)"


def available_plots() -> List[PlotKind]:
    """The plot kinds CRYSTALLine offers, in menu order.

    Output-file plots (elastic/EOS) read the loaded ``.out`` directly; the XRD
    pattern prompts for its PROPERTIES file. IR and Raman, and band structures
    and densities of states, are handled by dialogs of their own, not from here.
    """
    return [
        # ── from the CRYSTAL output file ──
        # IR and Raman are not here: a run's spectra span Raman polarisations
        # and anharmonic levels, far more than two menu entries could offer, so
        # they live behind the "Vibrational spectra…" dialog instead
        # (see crystalline.crystalio.spectra).
        *_elastic_plots(),
        PlotKind("eos", "Equation of state", "output", "Open CRYSTAL output (.out)", _OUT,
                 _eos_builder(), probe=("get_EOS", "VvsE")),
        # ── from a PROPERTIES / dispersion data file ──
        # Electronic and phonon bands and DOS are not here either: they have a
        # dozen options worth setting, a combined band+DOS view, and files to
        # find and tell apart, so they live behind the "Electronic bands &
        # DOS…" and "Phonon bands & DOS…" dialogs (see
        # crystalline.crystalio.electronic and .phonon_bands).
        PlotKind("xrd", "Simulated XRD pattern…", "data",
                 "Open XRD spectrum (XRDATO.DAT)", _DAT,
                 _prop_builder("read_cry_xrd_spec", "plot_cry_xrd")),
    ]


# The properties of the elastic entries, as (key, label): each as a surface and
# as sections through the principal planes, in a submenu each.
_ELASTIC_ENTRIES = (("young", "Young's modulus"), ("comp", "Linear compressibility"),
                    ("shear", "Shear modulus (avg)"), ("poisson", "Poisson's ratio (avg)"))
_ELASTIC_GROUPS = (("ela", "Elastic properties"), ("ela2d", "Elastic properties (2D)"))


def _elastic_plots() -> List[PlotKind]:
    """The elastic surfaces, then the sections, read from the loaded output."""
    probe = ("get_elatensor", "elatensor")
    return [PlotKind(f"{prefix}_{key}", label, "output", "Open CRYSTAL output (.out)", _OUT,
                     _elastic_builder(f"{prefix}_{key}"), group=group, probe=probe)
            for prefix, group in _ELASTIC_GROUPS for key, label in _ELASTIC_ENTRIES]


def crystalclear_available() -> bool:
    """Whether CRYSTALClear can be imported (so the UI can hint if not)."""
    try:
        _require_crystalclear()
        return True
    except PlotUnavailable:
        return False


# ── typography ──────────────────────────────────────────────────────────────
# The named recipes are CRYSTALClear's (``plot.set_font``), so a figure drawn
# here and one drawn in a notebook with CRYSTALClear look alike. STIX is added
# as the house face: a Times-like face with its own matching maths, shipped with
# matplotlib itself, so every machine draws the same figure with nothing to
# install — and with the Greek, Å and the true minus sign the figures use (it
# lacks only the superscript ⁻, which no label here writes as a character).
_STIX = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
}
DEFAULT_FONT_FAMILY = "stix"
DEFAULT_FONT_SIZE = 10.0


def font_families() -> tuple:
    """The named font choices offered for every figure, as (label, key)."""
    from CRYSTALClear.plot import FONT_FAMILIES

    return (("STIX", "stix"),) + tuple(FONT_FAMILIES)


def apply_font(family: str = DEFAULT_FONT_FAMILY, size: float = DEFAULT_FONT_SIZE) -> dict:
    """Set the font of every figure built afterwards, and report what was set.

    matplotlib reads its rcParams when a figure is *created*, so this governs
    plots opened from now on and leaves the ones already in the dock alone —
    which is why the dialog says so.

    ``family`` is a key of :func:`font_families`, or the name of any installed
    font ("Times New Roman"), which CRYSTALClear passes to matplotlib as it is;
    matplotlib falls back with its own warning if it has no such font.
    """
    import matplotlib
    from CRYSTALClear.plot import set_font

    if family == "stix":
        # Put on top of CRYSTALClear's reset, so nothing of another recipe stays.
        applied = set_font(family="default", size=float(size))
        applied.update(_STIX)
        matplotlib.rcParams.update(_STIX)
    else:
        applied = set_font(family=family, size=float(size))
    global _font_applied
    _font_applied = True
    return applied


# Whether a font has been put in place yet. The first plot drawn applies the
# default, so a figure is in the house face whether or not Plot ▸ Plot font was
# ever opened; once anything is applied — that default, or a choice made in the
# dialog — it is left alone.
_font_applied = False


def installed_font_names() -> list:
    """Font families matplotlib can actually use, sorted, for the "other" box.

    Empty if matplotlib is unavailable — the named recipes still work, since
    they are only applied once a figure is being built.
    """
    try:
        from matplotlib import font_manager
    except Exception:  # noqa: BLE001 - matplotlib missing/broken
        return []
    return sorted({font.name for font in font_manager.fontManager.ttflist})


# A line CRYSTAL only prints when the block a getter parses is actually there.
# Checked before the getter runs, because the getters cost hundreds of times
# more than a substring scan and this runs on every file open: get_EOS spends
# ~126 ms on a 3 MB output establishing that it holds no equation of state,
# which is the answer for nearly every file anyone opens.
_PROBE_MARKERS = {
    "get_elatensor": " SYMMETRIZED ELASTIC",
    "get_EOS": "SORTING VOLUMES/ENERGIES",
}


def _has_marker(path: str, marker: str) -> bool:
    """Whether ``marker`` appears anywhere in ``path``.

    Line by line rather than one ``read()``: an output can run to tens of MB and
    the marker is usually absent, so there is nothing to gain from holding the
    whole file to answer no.
    """
    try:
        with open(path, "r", errors="ignore") as handle:
            return any(marker in line for line in handle)
    except OSError:
        return False


def _probe_output(path: str, getter: str, attribute: str) -> bool:
    """Whether ``Crystal_output(path).getter()`` yields non-empty ``attribute``.

    Any parse/extraction failure (the data simply isn't in this run) counts as
    "not available". CRYSTALClear's chatter is swallowed so probing is silent.

    A getter with a known banner is only run once that banner has been found:
    the scan and the parse agree on absence, and the scan is the cheaper way to
    establish it by two orders of magnitude.
    """
    import contextlib
    import io

    from CRYSTALClear.crystal_io import Crystal_output

    marker = _PROBE_MARKERS.get(getter)
    if marker is not None and not _has_marker(path, marker):
        return False

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            co = Crystal_output(path)
            getattr(co, getter)()
        value = getattr(co, attribute, None)
        return value is not None and (not hasattr(value, "__len__") or len(value) > 0)
    except Exception:  # noqa: BLE001 - absent data raises in various ways
        return False


def output_availability(path: Optional[str]) -> set:
    """Keys of the output-file plots whose data is present in ``path``.

    Returns an empty set when ``path`` is falsy (no output loaded) or CRYSTALClear
    is unavailable. Each distinct ``(getter, attribute)`` probe is run once even
    though several elastic plots share it, and a probe whose block is not in the
    file at all is answered by a text scan rather than a parse — this runs on
    every file open, so what it costs is what opening a file costs.
    """
    if not path:
        return set()
    try:
        _require_crystalclear()
    except PlotUnavailable:
        return set()

    cache: dict = {}
    available = set()
    for kind in available_plots():
        if kind.source != "output" or not kind.probe:
            continue
        if kind.probe not in cache:
            cache[kind.probe] = _probe_output(path, *kind.probe)
        if cache[kind.probe]:
            available.add(kind.key)
    return available


__all__ = [
    "DEFAULT_FONT_FAMILY",
    "DEFAULT_FONT_SIZE",
    "PlotKind",
    "PlotUnavailable",
    "apply_font",
    "font_families",
    "available_plots",
    "crystalclear_available",
    "installed_font_names",
    "output_availability",
]
