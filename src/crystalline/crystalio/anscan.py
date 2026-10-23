"""The anharmonic scan of one normal mode, and the figure built from it.

ANSCAN walks a single normal mode away from equilibrium, fits a polynomial to
the energies it finds, and solves the one-dimensional vibrational problem in
that potential. What one wants to see is all of it at once: the scanned points,
the fitted potential, the levels it supports and the states sitting on them —
which is what ``CRYSTALClear.plot.plot_cry_anscan`` draws, and this module is
the single point of contact with it.

Everything is expressed in the dimensionless coordinate :math:`\\xi` the scan
is run on, the ``[DISPLAC]`` column of the output, so the abscissa needs no
choosing. Two things do:

* **how tall to draw the wavefunctions.** They are normalised, so their height
  in cm⁻¹ is arbitrary and has to be set against the level spacing of the run —
  a factor of 500 on a stiff CO₂ stretch, 30 on a soft double well. Rather than
  ask, :attr:`AnscanRun.scale_wf` reads it off the levels themselves.
* **the companion file.** The wavefunction coefficients are not in the ``.out``
  at all but in ANSCANWF.DAT, so a plot needs a second file; :func:`find_wavefunctions`
  looks for it next to the output before anyone is prompted.

Kept Qt-free, like the sibling modules, so the mapping from "what the user
picked" to "which CRYSTALClear call" stays testable on its own.
"""

from __future__ import annotations

import contextlib
import glob
import io
import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple

# The banner ANSCAN prints for the scan, and the header of the table of states:
# both have to be there for a plot to be possible, since a run can walk the mode
# and stop before it solves anything.
_BANNER = "SCAN ALONG NORMAL MODES"
_STATES = "ANHARMONIC VIBRATIONAL STATES"

# " MODE(CM**-1)     7( 487.1)" — which mode was scanned, and its harmonic
# frequency, printed immediately above the scanned potential.
_MODE = re.compile(r"MODE\(CM\*\*-1\)\s+(\d+)\(\s*(-?[\d.]+)\)")

# What CRYSTAL says when the scan finds two minima and switches to the basis
# init_dwell builds. Worth repeating to the user: on that branch only the cubic
# and quartic derivatives are kept, so the fitted curve can sit visibly above
# the points it was fitted to.
_DWELL = "DOUBLE-WELL POTENTIAL DETECTED"

# Names ANSCANWF.DAT is found under: CRYSTAL writes it under its own name, and
# a run kept alongside others is usually renamed after the input.
_WF_NAMES = ("ANSCANWF.DAT", "anscanwf.dat")
_WF_SUFFIXES = (".anscanwf", ".ANSCANWF", ".anscanwf.dat", ".ANSCANWF.DAT")

# Filter for the prompt, when the file is not where it can be guessed.
WF_FILTER = "ANSCAN wavefunctions (*.anscanwf *.DAT *.dat);;All files (*)"


@dataclass(frozen=True)
class AnscanRun:
    """What the ANSCAN block of an output holds, enough to drive the dialog."""

    mode: int                       # CRYSTAL index of the scanned mode
    frequency: float                # its harmonic frequency, cm^-1, signed
    nstates: int                    # levels CRYSTAL printed
    nwf: int                        # of which these many have a wavefunction
    rangescan: Tuple[float, float]  # first and last displacement, in xi
    spacing: float                  # mean level spacing over the drawable states
    double_well: bool

    @property
    def imaginary(self) -> bool:
        """Whether the scanned mode is imaginary, i.e. a saddle at equilibrium."""
        return self.frequency < 0.0

    @property
    def summary(self) -> str:
        """One line describing the run, for the dialog's header."""
        frequency = (f"{abs(self.frequency):.1f}i" if self.imaginary
                     else f"{self.frequency:.1f}")
        parts = [f"Mode {self.mode} at {frequency} cm⁻¹",
                 f"{self.nstates} levels, {self.nwf} with a wavefunction",
                 f"scanned over ξ = {self.rangescan[0]:g} to {self.rangescan[1]:g}"]
        if self.double_well:
            parts.append("double well")
        return ", ".join(parts)

    @property
    def scale_wf(self) -> float:
        """A wavefunction height that fills the gap to the next level.

        The states are normalised, so what they are worth in cm⁻¹ is a choice,
        and the only scale in the figure that means anything is the spacing of
        the levels they are drawn on. Peak amplitude of a low state is a little
        under one, so the spacing itself is about right: it lands on the 500 and
        30 that the two runs at hand were drawn with by hand.
        """
        return float(self.spacing) if self.spacing > 0 else 1.0

    @property
    def scale_prob(self) -> float:
        """Likewise for a density, which is the square of a number under one."""
        return 2.0 * self.scale_wf


def has_anscan(path: Optional[str]) -> bool:
    """Whether ``path`` carries a solved anharmonic scan, without parsing it.

    A text scan, since this is asked every time a file is loaded. Both the
    banner and the table of states are required: a run can walk the mode and be
    cut short before it solves the vibrational problem, and there would then be
    no levels to draw.
    """
    if not path:
        return False
    try:
        with open(path, "r", errors="ignore") as handle:
            text = handle.read()
    except OSError:
        return False
    return _BANNER in text and _STATES in text


def find_wavefunctions(path: str) -> Optional[str]:
    """Path to the ANSCANWF.DAT belonging to ``path``, or ``None``.

    The coefficients live outside the output, so a plot needs a second file.
    Tried in the order that keeps a directory of several runs unambiguous: the
    output's own stem first, then the name CRYSTAL writes, and only then a
    unique ``*.anscanwf`` lying about.
    """
    if not path:
        return None
    folder = os.path.dirname(os.path.abspath(path))
    stem = os.path.splitext(os.path.basename(path))[0]

    for suffix in _WF_SUFFIXES:
        candidate = os.path.join(folder, stem + suffix)
        if os.path.isfile(candidate):
            return candidate
    for name in _WF_NAMES:
        candidate = os.path.join(folder, name)
        if os.path.isfile(candidate):
            return candidate
    # Only when it leaves no doubt: two runs side by side would otherwise be
    # drawn with each other's wavefunctions, which nothing downstream can catch.
    loose = sorted(glob.glob(os.path.join(folder, "*.anscanwf")))
    return loose[0] if len(loose) == 1 else None


def _mode_line(path: str) -> Tuple[int, float, bool]:
    """``(mode, frequency, double_well)`` read straight off the output.

    The frequency is taken from the header of the scanned potential rather than
    from the parsed value, because it is the one CRYSTAL scanned with; a missing
    header leaves zeros, which the caller replaces with what the parse found.
    """
    mode, frequency, dwell = 0, 0.0, False
    try:
        with open(path, "r", errors="ignore") as handle:
            for line in handle:
                match = _MODE.search(line)
                if match and not mode:
                    mode, frequency = int(match.group(1)), float(match.group(2))
                if _DWELL in line:
                    dwell = True
    except OSError:
        pass
    return mode, frequency, dwell


def load_anscan(path: str, wavefunctions: str):
    """The output with its ANSCAN block parsed, or ``None`` if it has none.

    CRYSTALClear chatters to stdout while parsing, which is swallowed here.
    """
    try:
        from CRYSTALClear.crystal_io import Crystal_output
    except Exception:  # noqa: BLE001 - CRYSTALClear missing/broken
        return None

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            out = Crystal_output(path)
            out.get_anscan(wavefunctions)
    except Exception:  # noqa: BLE001 - no ANSCAN block, or an unreadable file
        return None
    if getattr(out, "wf", None) is None or not len(out.wf):
        return None
    if not getattr(out, "energy", None):
        return None
    return out


def anscan_run(path: Optional[str], wavefunctions: Optional[str]):
    """``(AnscanRun, Crystal_output)`` for ``path``, or ``(None, None)``.

    The parsed output comes back with the description because it is what the
    plot is drawn from, and ``get_anscan`` re-reads the phonon block on its way
    through — not something to pay for twice.
    """
    if not path or not wavefunctions:
        return None, None
    out = load_anscan(path, wavefunctions)
    if out is None:
        return None, None

    energy = [float(e) for e in out.energy]
    nwf = min(int(out.wf.shape[1]), len(energy))
    # Over the states that will be drawn, not the whole ladder: the top of a
    # hundred-level list is nowhere near the window the figure opens on.
    gaps = [b - a for a, b in zip(energy[:nwf], energy[1:nwf])]
    mode, frequency, dwell = _mode_line(path)

    run = AnscanRun(
        mode=mode,
        # The header prints one decimal, so the parsed value is the better
        # number whenever the two agree on which mode was scanned.
        frequency=float(getattr(out, "harm_freq", frequency) or frequency),
        nstates=len(energy),
        nwf=nwf,
        rangescan=(float(out.rangescan[0]), float(out.rangescan[1])),
        spacing=sum(gaps) / len(gaps) if gaps else 0.0,
        double_well=dwell,
    )
    return run, out


def plot_anscan(
    out,
    *,
    scale_wf: Optional[float] = None,
    scale_prob: Optional[float] = None,
    harmpot: bool = False,
    scanpot: bool = True,
    nstates: Optional[int] = None,
    legend: bool = True,
    linewidth: Optional[float] = None,
):
    """Draw ``out``'s anharmonic scan and return the figure.

    ``scale_wf`` and ``scale_prob`` are heights in cm⁻¹, ``None`` meaning that
    curve is left out; :attr:`AnscanRun.scale_wf` is where a sensible one comes
    from. ``nstates`` counts up from the ground state.
    """
    from crystalline.crystalio.plotting import _plot_module, _to_figure

    CCplt = _plot_module()  # imports CRYSTALClear with a non-interactive backend
    figure = _to_figure(CCplt.plot_cry_anscan(
        out,
        scale_wf=scale_wf,
        scale_prob=scale_prob,
        harmpot=harmpot,
        scanpot=scanpot,
        nstates=nstates,
        legend=legend,
    ))
    if linewidth is not None:
        from crystalline.crystalio.pes import scale_line_widths

        scale_line_widths(figure.axes[0], linewidth)    # drawn as a PES cut is
    return figure


__all__ = [
    "WF_FILTER",
    "AnscanRun",
    "anscan_run",
    "find_wavefunctions",
    "has_anscan",
    "load_anscan",
    "plot_anscan",
]
