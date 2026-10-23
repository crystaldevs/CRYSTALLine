"""VCI wavefunctions held in a CRYSTAL output, and the figures built from them.

A VCI state is a linear combination of zeroth-order configurations
:math:`\\Psi_s = \\sum_n A_{n,s} \\Phi^\\mathbf{n}`, and what one wants to see is
which configurations mix into which state — that is where the anharmonicity
shows up, as a resonance between a fundamental and an overtone. CRYSTALClear
draws that two ways, and this module is the single point of contact with them:

* ``"map"`` — heatmap of :math:`A_{n,s}`, one row per configuration and one
  column per state, labelled by its energy relative to the ZPE.
* ``"sankey"`` — the same numbers as ribbons whose width is the weight of a
  configuration in a state, which reads better for a handful of states.

The zeroth-order basis is not the same in the two flavours of the calculation:
a VCI@HO run expands on Hartree products of harmonic eigenfunctions, a VCI@VSCF
run on products of VSCF modals. CRYSTALClear detects which from the output and
labels the quanta accordingly, so nothing here has to choose.

Parsing a VCI block is not free — a run with tens of thousands of states takes a
second or two — so :func:`has_vci` answers the cheap question "is there one at
all?" for menu enabling, and the full parse happens only once the user asks for
a plot.

Kept Qt-free, like the sibling modules, so the mapping from "what the user
picked" to "which CRYSTALClear call" stays testable on its own.
"""

from __future__ import annotations

import contextlib
import io
import math
import re
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

# The representations CRYSTALClear can draw, as (menu label, key).
REPRESENTATIONS = (
    ("Coefficient map", "map"),
    ("State composition (Sankey)", "sankey"),
)

# key -> the CRYSTALClear.plot function that draws it.
PLOT_FUNCTIONS = {"map": "plot_cry_vci", "sankey": "plot_cry_vci_sankey"}

DEFAULT_THRESHOLD = 0.02

# How many states a window has to hold for the figure to still say something:
# one column per state on the map, one node per state on the Sankey.
MAX_STATES = 60

# Width of the window the dialog opens on, when the run is wider than that.
DEFAULT_WINDOW_STATES = 10

# The banner CRYSTAL prints for the VCI step, and one printed state: both have
# to be there for a plot to be possible, since a run can set VCI up and stop.
_BANNER = "VIBRATIONAL CONFIGURATION INTERACTION (VCI)"
_STATE = "VCI STATE ("

# The PES scan names the modes it couples by their CRYSTAL index, which is how
# the VCI-active modes can be recovered: the configurations themselves are
# written over a dense 1..nmodes range that says nothing about which modes of
# the full 3N those are.
_COUPLE = re.compile(r"COUPLE OF MODES:\s+(\d+)\s+(\d+)")


@dataclass(frozen=True)
class VCIRun:
    """What the VCI block of an output holds, enough to drive the dialog.

    ``modes`` is the CRYSTAL index of each VCI-active mode when the output lets
    them be recovered, else empty — in which case the plots number the modes
    from 1, which is all the configuration list itself says.
    """

    basis: str                  # "HO" or "VSCF"
    nstates: int
    nconfs: int
    nmodes: int
    irreps: Tuple[int, ...]     # empty when the VCI matrix is not blocked
    modes: Tuple[int, ...]
    zpe: Optional[float]
    # Every state as (ENE - ZPE in cm^-1, irrep), ascending in energy, so that a
    # window can be counted — per symmetry block too — without going back to the
    # output. The irrep is 0 when the VCI matrix was not blocked.
    levels: Tuple[Tuple[float, int], ...] = ()

    @property
    def label(self) -> str:
        """``"VCI@VSCF"`` — the flavour, as it is usually written."""
        return f"VCI@{self.basis}"

    @property
    def summary(self) -> str:
        """One line describing the run, for the dialog's header."""
        parts = [f"{self.label}: {self.nstates} states over {self.nconfs} configurations",
                 f"{self.nmodes} modes"]
        if self.irreps:
            parts.append(f"{len(self.irreps)} symmetry blocks")
        if self.levels:
            parts.append(f"up to {self.levels[-1][0]:.0f} cm⁻¹")
        if self.zpe is not None:
            parts.append(f"ZPE {self.zpe:.1f} cm⁻¹")
        return ", ".join(parts)

    @property
    def energies(self) -> Tuple[float, ...]:
        """Every state's ENE - ZPE, ascending."""
        return tuple(energy for energy, _irrep in self.levels)

    def count_in(self, fmin: float, fmax: float, irrep: Optional[int] = None) -> int:
        """How many states fall in ``[fmin, fmax]``, both edges included.

        ``irrep`` restricts the count the same way it restricts the plot, so the
        number the dialog shows is the number of columns that will be drawn.
        """
        low, high = sorted((fmin, fmax))
        return sum(1 for energy, block in self.levels
                   if low <= energy <= high and (irrep is None or block == irrep))

    def default_window(self, states: int = DEFAULT_WINDOW_STATES) -> Tuple[float, float]:
        """A window onto roughly the ``states`` lowest states, rounded outwards.

        A run can span thousands of states, so opening on all of them would give
        a figure nobody wants; the bottom of the spectrum is where one starts
        looking, and the window is then moved.
        """
        energies = self.energies
        if not energies:
            return 0.0, 4000.0
        top = energies[min(states, len(energies)) - 1]
        # A round number reads better in the spin box, and leaves room for the
        # states just above the cut rather than slicing between two neighbours.
        return 0.0, float(math.ceil(top / 50.0) * 50) or 50.0


def has_vci(path: Optional[str]) -> bool:
    """Whether ``path`` carries VCI states, without parsing them.

    A text scan, because the real parse costs a second or two on a large run and
    this is asked every time a file is loaded. Both the banner and at least one
    printed state are required: a run can reach the VCI step and be cut short.
    """
    if not path:
        return False
    try:
        with open(path, "r", errors="ignore") as handle:
            text = handle.read()
    except OSError:
        return False
    return _BANNER in text and _STATE in text


def _vci_modes(path: str, nmodes: int) -> Tuple[int, ...]:
    """CRYSTAL indices of the VCI-active modes, or ``()`` if not recoverable.

    Taken from the two-mode PES scan, which names its modes by CRYSTAL index.
    Only accepted when as many distinct modes turn up as the configurations are
    wide: a run whose scan was restricted, or which coupled no pair at all
    (a single-mode VCI), would otherwise mislabel every configuration.
    """
    found = set()
    try:
        with open(path, "r", errors="ignore") as handle:
            for line in handle:
                match = _COUPLE.search(line)
                if match:
                    found.update(int(g) for g in match.groups())
    except OSError:
        return ()
    return tuple(sorted(found)) if len(found) == nmodes else ()


def load_vci(path: str):
    """The output with its VCI block parsed, or ``None`` if it has none.

    CRYSTALClear chatters to stdout while parsing, which is swallowed here.
    """
    try:
        from CRYSTALClear.crystal_io import Crystal_output
    except Exception:  # noqa: BLE001 - CRYSTALClear missing/broken
        return None

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            out = Crystal_output(path)
            out.get_vci()
    except Exception:  # noqa: BLE001 - no VCI block, or an unreadable file
        return None
    if getattr(out, "VCI_state", None) is None or not len(out.VCI_state):
        return None
    return out


def vci_run(path: Optional[str]):
    """``(VCIRun, Crystal_output)`` for ``path``, or ``(None, None)``.

    The parsed output comes back with the description because it is what the
    plots are drawn from: parsing a large run twice would be a needless wait.
    """
    if not path:
        return None, None
    out = load_vci(path)
    if out is None:
        return None, None

    irreps = tuple(sorted({int(i) for i in getattr(out, "VCI_irrep", []) if int(i) > 0}))
    nmodes = int(out.VCI_list_conf.shape[1])
    run = VCIRun(
        basis=getattr(out, "VCI_basis", None) or "HO",
        nstates=int(len(out.VCI_energy)),
        nconfs=int(out.VCI_nconfs),
        nmodes=nmodes,
        irreps=irreps,
        modes=_vci_modes(path, nmodes),
        zpe=getattr(out, "VCI_zpe", None),
        levels=tuple(sorted(
            (float(energy), int(block)) for energy, block
            in zip(out.VCI_energy, getattr(out, "VCI_irrep", [0] * len(out.VCI_energy)))
        )),
    )
    return run, out


def plot_vci(
    out,
    *,
    representation: str = "map",
    frange: Optional[Sequence[float]] = None,
    irrep: Optional[int] = None,
    threshold: float = DEFAULT_THRESHOLD,
    signed: bool = False,
    annotate: bool = False,
    weight: str = "square",
    modes: Tuple[int, ...] = (),
    max_states: int = MAX_STATES,
):
    """Draw ``out``'s VCI coefficients and return the figure.

    ``frange`` is the wavenumber window, in cm⁻¹ relative to the ZPE, the states
    are taken from; ``None`` means every state of the run, which ``max_states``
    then refuses if there are too many to read.

    ``modes`` is passed through as CRYSTALClear's ``list_mode``, which relabels
    the configurations with the CRYSTAL mode indices instead of a dense
    1..nmodes range; empty means "leave them numbered from 1".

    The options that only one representation understands are not shared:
    ``signed`` and ``annotate`` belong to the map, ``weight`` to the Sankey.
    """
    from crystalline.crystalio.plotting import _plot_module, _to_figure

    if representation not in PLOT_FUNCTIONS:
        raise ValueError(f"unknown VCI representation {representation!r}")

    CCplt = _plot_module()  # imports CRYSTALClear with a non-interactive backend
    function = getattr(CCplt, PLOT_FUNCTIONS[representation])

    shared = dict(
        frange=list(frange) if frange is not None else None,
        irrep=irrep,
        threshold=threshold,
        list_mode=list(modes) if modes else None,
        max_states=max_states,
    )
    if representation == "map":
        return _to_figure(function(out, signed=signed, annotate=annotate, **shared))
    return _to_figure(function(out, weight=weight, **shared))


__all__ = [
    "DEFAULT_THRESHOLD",
    "DEFAULT_WINDOW_STATES",
    "MAX_STATES",
    "PLOT_FUNCTIONS",
    "REPRESENTATIONS",
    "VCIRun",
    "has_vci",
    "load_vci",
    "plot_vci",
    "vci_run",
]
