"""The anharmonic potential-energy surface an ANHAPES run computes, and the
figures built from it.

CRYSTAL numerically differentiates the energy along its normal modes and prints
what it finds as cubic and quartic terms: one set per mode, one set per coupled
pair, and — on a large run — one per coupled triplet. Together they are a
quartic Taylor expansion of the PES in the dimensionless normal coordinates,

.. math::
    V = \\tfrac{1}{2}\\omega_I\\xi_I^2 + \\tfrac{1}{6}\\eta_{III}\\xi_I^3
        + \\tfrac{1}{24}\\eta_{IIII}\\xi_I^4
        + \\tfrac{1}{2}\\eta_{IIJ}\\xi_I^2\\xi_J + \\ldots

and what one wants to look at is a cut through it: one mode on its own, or the
surface of a pair. This module is the single point of contact with the two
CRYSTALClear functions that draw those.

Three things shape the interface:

* **the harmonic bowl swamps everything.** The anharmonic terms are worth a few
  percent of it over the range the constants were fitted on, so contours of the
  total surface are ellipses and show nothing. The default is to take the
  harmonic part out and map what is left.
* **there can be hundreds of pairs.** A 36-mode run couples 630 of them, which
  is not a combo box; :attr:`PESRun.pairs` comes back sorted by how strongly the
  two modes actually couple, so the interesting ones are at the top.
* **the constants are a local fit.** CRYSTAL differentiates over a fraction of
  a classical amplitude either side of equilibrium, so a wide window is an
  extrapolation of a quartic. Hence a modest default range.

Kept Qt-free, like the sibling modules, so the mapping from "what the user
picked" to "which CRYSTALClear call" stays testable on its own.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import Optional, Tuple

# The banner ANHAPES prints, and one printed derivative: both have to be there,
# since a run can reach the step and be cut short before it differentiates.
_BANNER = "CALCULATION OF CUBIC AND QUARTIC TERMS"
_ETA = "ETA("

# What a cut can be, as (menu label, key).
DIMENSIONS = (
    ("One mode", "1D"),
    ("Two modes", "2D"),
)

# What a two-mode map can show, as (menu label, key). The harmonic part is one
# to two orders of magnitude deeper than the rest, so it is out by default.
QUANTITIES = (
    ("Anharmonic (V − V_harm)", "anharmonic"),
    ("Coupling terms only", "coupling"),
    ("Total surface", "total"),
)

# How a two-mode cut is drawn, as (menu label, key). The map is the default
# because it is the one that can be read off quantitatively; the surface shows
# the shape better and can be turned with the mouse once it is in the dock.
REPRESENTATIONS = (
    ("Contour map", "map"),
    ("3D surface", "surface"),
)

# key -> the CRYSTALClear.plot function that draws it.
PLOT_FUNCTIONS = {"map": "plot_cry_pes_2D", "surface": "plot_cry_pes_3D"}

# Half-width of the window the dialog opens on, in classical amplitudes. The
# constants come from a scan of ±0.9 of one, so this is already an extrapolation
# and a wider default would invite reading a quartic well past its fit.
DEFAULT_RANGE = 2.0

DEFAULT_NSTATES = 5
# How tall to draw the wavefunctions, in cm-1, when the run itself suggests
# nothing — a mid-range vibrational wavenumber. A real run says better: see
# PESRun.scale_wf, which reads it off the modes that were scanned.
DEFAULT_WF_SCALE = 1000.0


@dataclass(frozen=True)
class PESMode:
    """One mode's own terms: the quadratic, cubic and quartic of its 1D cut."""

    mode: int
    frequency: float        # cm^-1, signed
    eta3: float
    eta4: float

    @property
    def label(self) -> str:
        """``"Mode 12 — 3026.5 cm⁻¹"``, for the picker."""
        return f"Mode {self.mode} — {self.frequency:.1f} cm⁻¹"

    @property
    def detail(self) -> str:
        """The two constants, for the line under the label."""
        return f"η₃ {self.eta3:+.1f}, η₄ {self.eta4:+.1f} cm⁻¹"


@dataclass(frozen=True)
class PESPair:
    """Two modes and the five constants that couple them."""

    modei: int
    modej: int
    frequencyi: float
    frequencyj: float
    iij: float
    ijj: float
    iiij: float
    ijjj: float
    iijj: float

    @property
    def strength(self) -> float:
        """What the coupling is worth, in cm⁻¹, at unit displacement of both.

        The sum of the terms' magnitudes rather than of the terms, so that two
        large contributions of opposite sign read as a strong coupling — which
        they are — instead of cancelling into an apparently uncoupled pair.
        """
        return (abs(self.iij)/2 + abs(self.ijj)/2 + abs(self.iiij)/6
                + abs(self.ijjj)/6 + abs(self.iijj)/4)

    @property
    def label(self) -> str:
        """``"12 × 14 — 3026.5, 3128.4 cm⁻¹"``, for the picker."""
        return (f"{self.modei} × {self.modej} — "
                f"{self.frequencyi:.1f}, {self.frequencyj:.1f} cm⁻¹")

    @property
    def detail(self) -> str:
        """The coupling strength, which the list is ordered by."""
        return f"coupling {self.strength:.1f} cm⁻¹"

    def involves(self, mode: int) -> bool:
        """Whether ``mode`` is one of the two."""
        return mode in (self.modei, self.modej)


@dataclass(frozen=True)
class PESRun:
    """What the ANHAPES block holds, enough to drive the dialog."""

    modes: Tuple[PESMode, ...]
    pairs: Tuple[PESPair, ...]      # strongest coupling first
    ntriplets: int

    @property
    def scale_wf(self) -> float:
        """How tall to draw a wavefunction on this run's potentials, in cm⁻¹.

        The level spacing is what a wavefunction is read against — it is drawn
        on the potential's own axis, and the levels it belongs to are about
        ħω apart — so a harmonic frequency of this run is the height that puts
        one between its neighbours instead of flat along a level or over the
        top of the well. The middle mode rather than any particular one: the
        suggestion anchors a slider spanning a decade either side, and it has
        to suit whichever cut is chosen next.
        """
        frequencies = sorted(mode.frequency for mode in self.modes if mode.frequency > 0)
        if not frequencies:
            return 0.0
        return float(frequencies[len(frequencies) // 2])

    @property
    def summary(self) -> str:
        """One line describing the run, for the dialog's header."""
        parts = [f"{len(self.modes)} mode{'s' if len(self.modes) != 1 else ''} scanned"]
        if self.pairs:
            parts.append(f"{len(self.pairs)} coupled pairs")
        if self.ntriplets:
            # Not drawable — a triplet cut is a 4D object — but worth saying,
            # since a run that computed them took a long time doing it.
            parts.append(f"{self.ntriplets} triplets (not plotted)")
        window = self.frequency_span
        if window:
            parts.append(f"{window[0]:.0f}–{window[1]:.0f} cm⁻¹")
        return ", ".join(parts)

    @property
    def frequency_span(self) -> Optional[Tuple[float, float]]:
        """Lowest and highest harmonic frequency among the scanned modes."""
        if not self.modes:
            return None
        frequencies = [entry.frequency for entry in self.modes]
        return min(frequencies), max(frequencies)

    def mode(self, index: int) -> Optional[PESMode]:
        """The entry for CRYSTAL mode ``index``, or ``None``."""
        for entry in self.modes:
            if entry.mode == index:
                return entry
        return None

    def pairs_with(self, mode: Optional[int] = None) -> Tuple[PESPair, ...]:
        """The pairs involving ``mode``, or all of them when it is ``None``."""
        if mode is None:
            return self.pairs
        return tuple(pair for pair in self.pairs if pair.involves(mode))


def has_pes(path: Optional[str]) -> bool:
    """Whether ``path`` carries computed PES constants, without parsing them.

    A text scan, since this is asked every time a file is loaded. Both the
    banner and at least one printed derivative are required: a run can reach the
    ANHAPES step and be cut short with nothing to plot.
    """
    if not path:
        return False
    try:
        with open(path, "r", errors="ignore") as handle:
            text = handle.read()
    except OSError:
        return False
    return _BANNER in text and _ETA in text


def load_pes(path: str):
    """The output with its PES constants and frequencies parsed, or ``None``.

    Both parses are needed: the cubic and quartic terms come from the ANHAPES
    block, the quadratic one from the harmonic frequencies. CRYSTALClear
    chatters to stdout while parsing, which is swallowed here.
    """
    try:
        from CRYSTALClear.crystal_io import Crystal_output
    except Exception:  # noqa: BLE001 - CRYSTALClear missing/broken
        return None

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            out = Crystal_output(path)
            out.get_anh_const()
            out.get_phonon(rm_imaginary=False)
    except Exception:  # noqa: BLE001 - no ANHAPES block, or an unreadable file
        return None
    if getattr(out, "PES_single", None) is None or not len(out.PES_single):
        return None
    if getattr(out, "frequency", None) is None or not len(out.frequency):
        return None
    return out


def pes_run(path: Optional[str]):
    """``(PESRun, Crystal_output)`` for ``path``, or ``(None, None)``.

    The parsed output comes back with the description because it is what the
    figures are drawn from, and the two parses together are not instant on a
    large run.
    """
    if not path:
        return None, None
    out = load_pes(path)
    if out is None:
        return None, None

    from CRYSTALClear import units

    nmodes = int(out.frequency.shape[1])

    def frequency(index: int) -> Optional[float]:
        # A restart can carry PES constants for modes the frequency block of
        # this file does not cover; such a mode has no quadratic term and so
        # cannot be drawn at all.
        if not (1 <= index <= nmodes):
            return None
        return float(units.thz_to_cm(out.frequency[0, index - 1]))

    modes = []
    for row in out.PES_single:
        index = int(row[0])
        value = frequency(index)
        if value is not None:
            modes.append(PESMode(mode=index, frequency=value,
                                 eta3=float(row[1]), eta4=float(row[2])))
    if not modes:
        return None, None
    drawable = {entry.mode for entry in modes}

    pairs = []
    couple = getattr(out, "PES_couple", None)
    for row in (couple if couple is not None and len(couple) else ()):
        modei, modej = int(row[0]), int(row[1])
        if modei not in drawable or modej not in drawable:
            continue
        pairs.append(PESPair(
            modei=modei, modej=modej,
            frequencyi=frequency(modei), frequencyj=frequency(modej),
            iij=float(row[2]), ijj=float(row[3]), iiij=float(row[4]),
            ijjj=float(row[5]), iijj=float(row[6]),
        ))
    # Strongest first: a run couples every pair it was given, and most of them
    # are worth nothing. Which ones matter is the question the picker answers.
    pairs.sort(key=lambda pair: -pair.strength)

    triplet = getattr(out, "PES_triplet", None)
    run = PESRun(
        modes=tuple(modes),
        pairs=tuple(pairs),
        ntriplets=int(len(triplet)) if triplet is not None and len(triplet) else 0,
    )
    return run, out


def plottable() -> bool:
    """Whether the installed CRYSTALClear can draw either cut."""
    try:
        from CRYSTALClear import plot as CCplt
    except Exception:  # noqa: BLE001 - CRYSTALClear missing/broken
        return False
    return hasattr(CCplt, "plot_cry_pes_1D") and any(
        hasattr(CCplt, name) for name in PLOT_FUNCTIONS.values())


def representations() -> Tuple[Tuple[str, str], ...]:
    """The two-mode representations the installed CRYSTALClear can draw.

    The 3D surface arrived after the map, so a build carrying only the latter
    offers one choice rather than an entry that fails once it is picked.
    """
    try:
        from CRYSTALClear import plot as CCplt
    except Exception:  # noqa: BLE001 - CRYSTALClear missing/broken
        return ()
    return tuple((label, key) for label, key in REPRESENTATIONS
                 if hasattr(CCplt, PLOT_FUNCTIONS[key]))


def plot_pes(
    out,
    *,
    dimension: str = "1D",
    representation: str = "map",
    mode: Optional[int] = None,
    modei: Optional[int] = None,
    modej: Optional[int] = None,
    span: float = DEFAULT_RANGE,
    harmonic: bool = True,
    levels: bool = False,
    nstates: int = DEFAULT_NSTATES,
    scale_wf: Optional[float] = None,
    legend: bool = True,
    quantity: str = "anharmonic",
):
    """Draw a cut through ``out``'s PES and return the figure.

    ``span`` is the half-width of the window, in classical amplitudes, applied
    to both axes. On a one-mode cut with ``levels`` it is a floor rather than a
    bound: states that reach further widen it, since one drawn cut off at the
    edge of the window says less than the extrapolation needed to contain it.

    ``representation`` chooses between the contour map and the 3D surface, and
    is ignored by a one-mode cut, which is a curve either way.
    """
    from crystalline.crystalio.plotting import PlotUnavailable, _plot_module, _to_figure

    if dimension not in {key for _label, key in DIMENSIONS}:
        raise ValueError(f"unknown PES cut {dimension!r}")
    if representation not in PLOT_FUNCTIONS:
        raise ValueError(f"unknown PES representation {representation!r}")

    CCplt = _plot_module()  # imports CRYSTALClear with a non-interactive backend
    name = ("plot_cry_pes_1D" if dimension == "1D"
            else PLOT_FUNCTIONS[representation])
    function = getattr(CCplt, name, None)
    if function is None:
        raise PlotUnavailable(
            f"The installed CRYSTALClear cannot draw a PES (it has no plot.{name})."
        )

    if dimension == "1D":
        if mode is None:
            raise ValueError("a one-mode cut needs a mode")
        figure = _to_figure(function(
            out, mode, xlim=[-span, span],
            harmonic=harmonic, levels=levels, nstates=nstates,
            # None leaves the height to the plotter, which reads it off the
            # level spacing; a number overrides it.
            scale_wf=scale_wf or None,
            legend=legend,
        ))
        if levels:
            _levels_inside_the_well(figure.axes[0])
        return figure

    if modei is None or modej is None:
        raise ValueError("a two-mode cut needs two modes")
    extra = {"contours": True} if representation == "surface" else {}
    return _to_figure(function(
        out, modei, modej, quantity=quantity, xlim=[-span, span], **extra,
    ))


def _levels_inside_the_well(ax) -> None:
    """Draw each level only where the state can classically be.

    CRYSTALClear rules every level across the whole window, so the lines run
    straight through the walls of the potential and on to the edge — which is
    not how a level diagram is read, and buries the curve under them. Each now
    spans the stretch of the anharmonic potential below it around the minimum.
    The wavefunctions are left whole: their tails past the walls are physics.
    """
    import numpy as np
    from matplotlib.collections import LineCollection

    curve = next((line for line in ax.get_lines()
                  if line.get_label() == "cubic + quartic"), None)
    if curve is None:
        return
    x = np.asarray(curve.get_xdata(), dtype=float)
    v = np.asarray(curve.get_ydata(), dtype=float)
    bottom = int(np.argmin(v))
    for collection in ax.collections:
        if not isinstance(collection, LineCollection):
            continue
        segments = collection.get_segments()
        if len(segments) != 1 or len(segments[0]) != 2:
            continue
        (x0, e0), (x1, e1) = segments[0]
        if not np.isclose(e0, e1):
            continue                                   # not a level
        below = v <= e0
        if not below[bottom]:
            continue
        left = bottom
        while left > 0 and below[left - 1]:
            left -= 1
        right = bottom
        while right < len(x) - 1 and below[right + 1]:
            right += 1
        collection.set_segments([[(x[left], e0), (x[right], e0)]])


__all__ = [
    "DEFAULT_NSTATES",
    "DEFAULT_WF_SCALE",
    "DEFAULT_RANGE",
    "DIMENSIONS",
    "PLOT_FUNCTIONS",
    "QUANTITIES",
    "REPRESENTATIONS",
    "representations",
    "PESMode",
    "PESPair",
    "PESRun",
    "has_pes",
    "load_pes",
    "pes_run",
    "plot_pes",
    "plottable",
]
