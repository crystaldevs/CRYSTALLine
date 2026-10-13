"""Vibrational spectra held in a CRYSTAL output, and figures built from them.

A FREQCALC output can carry a lot more than one IR curve. Raman is resolved
into polarisations — the powder averages (total, parallel, perpendicular) and
the six single-crystal components — and an ANHARM run adds VSCF, VPT2 and VCI
levels on top of the harmonic ones, each at 0 K and at the run's temperature.
That is up to sixty distinct curves, which is far too many for a menu, so this
module *enumerates what a given file actually holds* and the UI offers only
those.

Every spectrum is the same thing underneath: an ``(N, 2)`` array of transition
frequency (cm^-1) and intensity, which ``CRYSTALClear.plot.plot_cry_spec``
broadens into a lineshape. Overlaying several on one axes is what makes the
components useful — xx/yy/zz together show the anisotropy, harmonic against VCI
shows the anharmonic shift.

Discovery is by attribute *name* rather than a hard-coded list, so a level or
component that a future CRYSTALClear adds is picked up without a change here.

CRYSTALClear and matplotlib are imported lazily, as in the sibling modules.
"""

from __future__ import annotations

import contextlib
import io
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

# Crystal_output attribute names: ``IR_<level>_<temperature>`` and
# ``Ram_<level>_<temperature>_<polarisation>``.
_IR_NAME = re.compile(r"^IR_(?P<level>[A-Z0-9]+)_(?P<temperature>0K|T)$")
_RAMAN_NAME = re.compile(
    r"^Ram_(?P<level>[A-Z0-9]+)_(?P<temperature>0K|T)_(?P<polarisation>tot|par|per|comp_[xyz]{2})$"
)

# CRYSTAL's abbreviations, and the order a menu should list them in.
_LEVELS = {"HO": "harmonic", "VSCF": "VSCF", "VPT2": "VPT2", "VCI": "VCI"}
_LEVEL_ORDER = ["harmonic", "VSCF", "VPT2", "VCI"]
_POLARISATIONS = {"tot": "total", "par": "parallel", "per": "perpendicular"}
_POLARISATION_ORDER = [
    "total", "parallel", "perpendicular", "xx", "xy", "xz", "yy", "yz", "zz",
]
# The getters that populate them. Each is tried independently: a harmonic run
# has no anharmonic block and vice versa, and neither is an error.
_GETTERS = ("get_IR", "get_Raman", "get_anh_spectra")

# Lineshapes plot_cry_spec can draw, as (label, its own name for it).
LINESHAPES = (
    ("Lorentzian", "lorentz"),
    ("Gaussian", "gauss"),
    ("Pseudo-Voigt", "pvoigt"),
    ("Sticks (no broadening)", "bars"),
)

# Which broadening parameters each lineshape actually uses. A pseudo-Voigt is a
# weighted sum of the other two, so it needs both widths *and* the mixing
# fraction; offering one shared "width" would silently tie them together.
BROADENING_PARAMETERS = {
    "lorentz": ("hwhm",),
    "gauss": ("stdev",),
    "pvoigt": ("hwhm", "stdev", "eta"),
    "bars": (),
}

# plot_cry_spec's own defaults, so an untouched dialog reproduces its output.
DEFAULT_HWHM = 5.0    # Lorentzian half-width at half-maximum, cm^-1
DEFAULT_STDEV = 3.0   # Gaussian standard deviation, cm^-1
DEFAULT_ETA = 0.5     # Lorentzian fraction of a pseudo-Voigt, 0 to 1
DEFAULT_LINEWIDTH = 2.0   # plot_cry_spec_multi's own, in points


@dataclass(frozen=True)
class SpectrumKind:
    """One curve a CRYSTAL output can supply.

    ``attribute`` is the name it lives under on ``Crystal_output``; the rest is
    what that name means, parsed out so the UI can group and label it.
    """

    attribute: str
    kind: str            # "IR" or "Raman"
    level: str           # "harmonic", "VSCF", "VPT2", "VCI"
    temperature: str     # "0 K" or "T"
    polarisation: str    # "" for IR, else "total"/"parallel"/"xx"/…

    @property
    def label(self) -> str:
        """``"Raman xx (VCI, T)"`` — short enough for a legend."""
        head = self.kind if not self.polarisation else f"{self.kind} {self.polarisation}"
        return f"{head} ({self.level}, {self.temperature})"

    @property
    def group(self) -> str:
        """The heading this curve sits under.

        Every IR curve goes in one section: there is at most one per level and
        temperature, so splitting them would give a column of single-entry
        branches. Raman splits by level and temperature because each of those
        carries nine polarisations, which is a section's worth on its own.
        """
        if self.kind == "IR":
            return "IR"
        return f"Raman ({self.level}, {self.temperature})"

    @property
    def leaf_label(self) -> str:
        """What distinguishes this curve *within* its section."""
        if self.kind == "IR":
            return f"{self.level}, {self.temperature}"
        return self.polarisation

    @property
    def sort_key(self) -> tuple:
        return (
            0 if self.kind == "IR" else 1,
            _LEVEL_ORDER.index(self.level) if self.level in _LEVEL_ORDER else len(_LEVEL_ORDER),
            0 if self.temperature == "0 K" else 1,
            _POLARISATION_ORDER.index(self.polarisation)
            if self.polarisation in _POLARISATION_ORDER
            else len(_POLARISATION_ORDER),
        )


def _parse(attribute: str) -> Optional[SpectrumKind]:
    """The meaning of a ``Crystal_output`` spectrum attribute, or ``None``."""
    match = _IR_NAME.match(attribute)
    if match:
        polarisation = ""
    else:
        match = _RAMAN_NAME.match(attribute)
        if not match:
            return None
        raw = match.group("polarisation")
        polarisation = _POLARISATIONS.get(raw, raw.replace("comp_", ""))
    level = _LEVELS.get(match.group("level"))
    if level is None:
        return None  # an abbreviation we don't know how to label
    return SpectrumKind(
        attribute=attribute,
        kind="IR" if attribute.startswith("IR_") else "Raman",
        level=level,
        temperature="0 K" if match.group("temperature") == "0K" else "T",
        polarisation=polarisation,
    )


def load_spectra(path: str) -> Dict[SpectrumKind, np.ndarray]:
    """Every vibrational spectrum ``path`` holds, keyed by what it is.

    Each getter is tried on its own and its failure ignored: a plain FREQCALC
    has no Raman block, an ANHARM-less run has no VSCF/VCI ones, and a file
    with none of them yields an empty dict rather than an error. CRYSTALClear
    prints progress to stdout while parsing, which is swallowed here.

    Returns arrays of shape ``(N, 2)`` — frequency in cm^-1, intensity in
    whatever units that spectrum is reported in.
    """
    try:
        from CRYSTALClear.crystal_io import Crystal_output
    except Exception:  # noqa: BLE001 - CRYSTALClear missing/broken
        return {}

    try:
        out = Crystal_output(path)
    except Exception:  # noqa: BLE001 - unreadable file
        return {}

    for getter in _GETTERS:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                getattr(out, getter)()
        except Exception:  # noqa: BLE001 - that block simply isn't in this run
            continue

    found: Dict[SpectrumKind, np.ndarray] = {}
    for attribute in dir(out):
        if not attribute.startswith(("IR_", "Ram_")):
            continue
        kind = _parse(attribute)
        if kind is None:
            continue
        data = np.asarray(getattr(out, attribute, None), dtype=float)
        if data.ndim == 2 and data.shape[1] == 2 and len(data):
            found[kind] = data
    return found


def available_spectra(path: Optional[str]) -> List[SpectrumKind]:
    """The spectra in ``path``, in menu order. Empty for ``None`` or no data."""
    if not path:
        return []
    return sorted(load_spectra(path), key=lambda k: k.sort_key)


def plot_spectra(
    datasets: Sequence[tuple],
    *,
    lineshape: str = "lorentz",
    hwhm: float = DEFAULT_HWHM,
    stdev: float = DEFAULT_STDEV,
    eta: float = DEFAULT_ETA,
    frequency_range: Optional[tuple] = None,
    offset: float = 0.0,
    legend: bool = True,
    linewidth: float = DEFAULT_LINEWIDTH,
    title: Optional[str] = None,
):
    """Overlay ``datasets`` — ``(label, (N, 2) array)`` pairs — on one figure.

    Overlaying is the point: the six single-crystal Raman components on shared
    axes show the anisotropy at a glance, and a harmonic curve under its VCI
    counterpart shows the anharmonic shift. ``plot_cry_spec`` takes an existing
    axes, so each curve is drawn onto the same one and labelled for the legend.

    The three broadening parameters are independent, because a pseudo-Voigt
    profile needs all of them at once — it is the sum
    ``eta * lorentzian(hwhm) + (1 - eta) * gaussian(stdev)``, so ``eta = 1`` is
    pure Lorentzian and ``eta = 0`` pure Gaussian. A plain Lorentzian uses only
    ``hwhm``, a plain Gaussian only ``stdev``, and a stick spectrum none of them;
    the unused ones are passed anyway and ignored downstream.

    ``legend`` names the curves on the figure. Off, they keep their colours and
    the panel's own list says which is which; a legend sits over the plot, and
    on a crowded spectrum it hides the peaks it is there to name.

    ``offset`` stacks the curves instead of laying them on top of one another,
    each one raised above the one before: six Raman components overlaid exactly
    are six curves sharing every baseline, and which is which stops being
    readable long before the sixth. It is given here as a *fraction of the
    tallest peak drawn*, not as an intensity, because what the intensities mean
    differs between an IR spectrum and a Raman one and no absolute number could
    be right for both; ``plot_cry_spec_multi``, which does the stacking, is
    handed the fraction converted to the units of the data in hand (see
    :func:`_tallest_peak`).
    """
    from crystalline.crystalio.plotting import _plot_module

    CCplt = _plot_module()  # imports CRYSTALClear with a non-interactive backend

    if not datasets:
        raise ValueError("no spectra selected")

    fmin, fmax = (None, None) if frequency_range is None else frequency_range
    curves = [np.asarray(data, dtype=float) for _label, data in datasets]
    if lineshape == "bars" and frequency_range is not None:
        # plot_cry_spec draws every stick it is handed, whatever fmin and fmax
        # say — the window only bounds the zero baseline they stand on — so a
        # stick spectrum asked for 200-1200 cm-1 ran out to the O-H stretch.
        # A broadened one is left whole: lines outside the window still have
        # tails inside it.
        low, high = sorted(float(bound) for bound in frequency_range)
        curves = [curve[(curve[:, 0] >= low) & (curve[:, 0] <= high)] for curve in curves]
    labels = [label for label, _data in datasets]
    figure, axes = CCplt.plot_cry_spec_multi(
        curves,
        typeS=lineshape,
        bwidth=hwhm,
        stdev=stdev,
        eta=eta,
        fmin=fmin,
        fmax=fmax,
        # A legend of one entry names the thing its own title already names,
        # so one curve never has one; past that it is the figure's owner who
        # decides, since a legend over a crowded spectrum hides the peaks it
        # is there to name.
        label=labels if (legend and len(curves) > 1) else None,
        offset=float(offset) * _tallest_peak(
            curves, lineshape=lineshape, hwhm=hwhm, stdev=stdev, eta=eta,
            frequency_range=frequency_range,
        ) if offset else 0.0,
        linewidth=linewidth,
        figsize=(9, 5),
    )
    if title:
        axes.set_title(title)
    return figure


# How many points the tallest peak is looked for at in one go: a (points x
# transitions) block of this many rows stays a few MB even for the two thousand
# transitions of an anharmonic Raman curve at temperature.
_PEAK_ROWS = 256


def _tallest_peak(curves, *, lineshape, hwhm, stdev, eta, frequency_range) -> float:
    """The height of the tallest peak among ``curves`` as they are drawn.

    Not the tallest transition: broadening spreads a transition's intensity
    over its width, so a Lorentzian of HWHM 5 cm⁻¹ peaks at a sixteenth of it,
    and an offset measured against the transitions lifted each curve by many
    times the height of anything on screen. Measured against what is drawn,
    the slider's travel is the height of the spectra, and the stack keeps its
    proportions as the widths change, since the curves and the gaps between
    them scale together.

    Only what falls inside ``frequency_range`` counts, as only that is drawn,
    though every transition's tail adds to it.

    The broadened curve is evaluated at the transitions themselves rather than
    over the drawing's own grid: a peak sits on a transition, or on several
    that coincide, which a degenerate mode does. Two close lines merging into
    one bump between them are the exception, and peak a few per cent above
    either line — immaterial for a control set by eye, and the price of the
    answer costing nothing next to drawing the curves.
    """
    tallest = 0.0
    for curve in curves:
        transitions = np.asarray(curve, dtype=float)
        if not len(transitions):
            continue
        centres, intensities = transitions[:, 0], transitions[:, 1]
        drawn = np.ones(len(centres), dtype=bool)
        if frequency_range is not None:
            low, high = sorted(float(bound) for bound in frequency_range)
            drawn = (centres >= low) & (centres <= high)
        if not drawn.any():
            continue
        if lineshape == "bars":
            # Sticks are drawn one by one: coincident ones overlap, not add.
            tallest = max(tallest, float(np.max(intensities[drawn])))
            continue
        points = centres[drawn]
        for start in range(0, len(points), _PEAK_ROWS):
            squared = (points[start:start + _PEAK_ROWS, None] - centres[None, :]) ** 2
            heights = _profile(squared, lineshape, hwhm, stdev, eta) @ intensities
            tallest = max(tallest, float(np.max(heights)))
    return tallest


def _profile(squared, lineshape, hwhm, stdev, eta):
    """``plot_cry_spec``'s unit-area profiles, from squared distances."""
    if lineshape == "gauss":
        return _gaussian(squared, stdev)
    lorentzian = (hwhm / math.pi) / (squared + hwhm * hwhm)
    if lineshape == "lorentz":
        return lorentzian
    return eta * lorentzian + (1.0 - eta) * _gaussian(squared, stdev)


def _gaussian(squared, stdev):
    return np.exp(squared / (-2.0 * stdev * stdev)) / (stdev * math.sqrt(2.0 * math.pi))


__all__ = [
    "BROADENING_PARAMETERS",
    "DEFAULT_ETA",
    "DEFAULT_HWHM",
    "DEFAULT_LINEWIDTH",
    "DEFAULT_STDEV",
    "LINESHAPES",
    "SpectrumKind",
    "available_spectra",
    "load_spectra",
    "plot_spectra",
]
