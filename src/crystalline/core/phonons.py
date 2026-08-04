"""Phonon-mode model: frequencies plus per-atom displacement eigenvectors.

A CRYSTAL phonon calculation yields, for each normal mode, a frequency and a
set of atomic displacement vectors (the eigenvector). Animating a mode means
displacing every atom from its equilibrium position along that eigenvector,
scaled by ``amplitude * sin(phase)``.

**Away from Gamma.** A SCELPHONO run computes force constants in a supercell
and reports modes at every q commensurate with it. Such a mode is a *travelling
wave*: cell ``n`` (an integer lattice translation from the reference cell) lags
the reference by ``q·n``, and the eigenvector is complex — CRYSTAL prints its
real and imaginary parts as the "in phase" and "anti-phase" blocks. The
displacement of an atom is therefore

    u(n, t) = Re[ e * exp(i(2*pi*q·n - w*t)) ]

which is what :func:`displaced_positions` evaluates, with the ``exp(2*pi*i q·n)``
factor folded into the eigenvector when a cell operation replicates a mode onto
image atoms (``core.cells``). At Gamma the factor is 1, the eigenvector is real,
and everything reduces to the in-phase motion this module started with.

``qpoint`` is always in fractional coordinates of the reciprocal basis of the
cell the mode's geometry is currently expressed in — a cell operation that
changes that basis (the conventional-cell expansion) transforms it to match.

This module holds the data only; the actual per-frame geometry is produced by
:func:`displaced_positions`, which ``viz.phonon_animator`` calls. Kept Qt- and
PyVista-free so it can be unit-tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Optional, Tuple

import numpy as np

# Below this, a q component counts as zero (or as an exact fraction). CRYSTAL
# prints q as an integer triple over the shrinking factors, so the values we
# parse are exact ratios of small integers; this only absorbs float noise.
_Q_TOL = 1e-6

# Largest denominator a q component is read as a fraction over. Well past any
# shrinking factor a supercell calculation is run with, so it recovers the exact
# ratio rather than rounding a fine grid down to a coarser one.
_MAX_DENOMINATOR = 128


@dataclass(frozen=True)
class PhononMode:
    """A single normal mode.

    Attributes
    ----------
    frequency:
        Mode frequency in cm^-1 (negative denotes an imaginary/soft mode).
    eigenvector:
        (N, 3) displacement vectors, one row per atom. Complex away from Gamma,
        where each atom carries its own phase (see the module docstring).
    ir_active, raman_active:
        Selection-rule activity as reported by CRYSTAL, or ``None`` when the
        output carries no IR/Raman analysis (dispersion runs, older outputs).
    ir_intensity:
        Harmonic IR intensity in km/mol, when available.
    qpoint:
        (3,) fractional coordinates of the mode's q, in the reciprocal basis of
        the cell the eigenvector is defined on. ``None`` means Gamma — the only
        possibility for a plain FREQCALC, and the default so that every existing
        caller keeps describing a Gamma-point mode.
    cell_phase:
        (N,) Bloch phase ``2*pi*q·n`` in radians of the cell each atom was
        replicated into, relative to the reference cell (``None`` = all zero, an
        unreplicated mode or a Gamma one). This is the wave itself, and the only
        form of it that survives a still image: every cell of a travelling wave
        has the *same* amplitude and differs only here. Recovering it from the
        eigenvector afterwards is impossible — ``arg`` of an oscillation is
        defined only modulo pi, which would draw a q = 1/4 wave as a q = 1/2 one
        — so the cell operations carry it alongside.
    """

    frequency: float
    eigenvector: np.ndarray
    ir_active: Optional[bool] = None
    raman_active: Optional[bool] = None
    ir_intensity: Optional[float] = None
    qpoint: Optional[np.ndarray] = None
    cell_phase: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        ev = np.asarray(self.eigenvector)
        # Complex is kept complex; anything else (ints, lists) becomes float, so
        # a mode always holds one of the two dtypes the animation understands.
        ev = ev.astype(complex if np.iscomplexobj(ev) else float, copy=False)
        if ev.ndim != 2 or ev.shape[1] != 3:
            raise ValueError(f"eigenvector must be (N, 3), got {ev.shape}")
        object.__setattr__(self, "eigenvector", ev)
        if self.qpoint is not None:
            q = np.asarray(self.qpoint, dtype=float).ravel()
            if q.size != 3:
                raise ValueError(f"qpoint must have 3 components, got {q.size}")
            object.__setattr__(self, "qpoint", q)
        if self.cell_phase is not None:
            phase = np.asarray(self.cell_phase, dtype=float).ravel()
            if phase.size != ev.shape[0]:
                raise ValueError(
                    f"cell_phase must have one entry per atom ({ev.shape[0]}), got {phase.size}"
                )
            object.__setattr__(self, "cell_phase", phase)

    @property
    def n_atoms(self) -> int:
        return self.eigenvector.shape[0]

    @property
    def is_imaginary(self) -> bool:
        return self.frequency < 0.0

    @property
    def is_gamma(self) -> bool:
        """Whether the mode sits at the zone centre (every cell in phase)."""
        return self.qpoint is None or bool(np.all(np.abs(self.qpoint) < _Q_TOL))

    @property
    def qpoint_label(self) -> str:
        """``"Γ"`` or ``"(0, 0, 1/2)"`` — the q-point as a panel would show it."""
        return qpoint_label(self.qpoint)

    def with_qpoint(self, qpoint) -> "PhononMode":
        """Copy of this mode carrying a different q-point.

        Used when a cell operation re-expresses the mode in another lattice
        basis, in which the same physical q has different fractional
        coordinates.
        """
        return replace(self, qpoint=qpoint)

    @property
    def has_activity(self) -> bool:
        """Whether selection-rule activity is known for this mode."""
        return self.ir_active is not None or self.raman_active is not None

    @property
    def peak_displacement(self) -> float:
        """Largest per-atom displacement in the raw eigenvector.

        Eigenvectors arrive normalised over the *whole* cell (their 3N-component
        norm is 1), so the per-atom motion shrinks like 1/sqrt(N) and a fixed
        amplitude that looks right for a molecule is invisible for a big cell.
        :func:`displaced_positions` divides by this to make the user-facing
        amplitude mean "peak displacement of the most-displaced atom".

        A complex (non-Gamma) atom traces an ellipse rather than a line, so this
        is measured on ``|e|``: the true peak excursion is the ellipse's
        semi-major axis, which ``|e|`` bounds and equals whenever the atom moves
        along a line — every atom of a Gamma mode included.
        """
        if self.eigenvector.size == 0:
            return 0.0
        peak = float(np.max(np.linalg.norm(np.abs(self.eigenvector), axis=1)))
        return peak if np.isfinite(peak) else 0.0

    def with_eigenvector(self, eigenvector: np.ndarray, cell_phase=None) -> "PhononMode":
        """Copy of this mode carrying a different eigenvector (e.g. tiled onto
        a supercell), keeping the frequency and the IR/Raman labels.

        ``cell_phase`` replaces the per-atom Bloch phase; it must be given
        whenever the atom count changes, since the old one no longer indexes the
        new eigenvector. It is dropped rather than kept stale when omitted.
        """
        return replace(self, eigenvector=eigenvector, cell_phase=cell_phase)


class PhononModes:
    """An ordered collection of :class:`PhononMode` sharing one geometry."""

    def __init__(self, modes: list[PhononMode]) -> None:
        self._modes = list(modes)

    def __len__(self) -> int:
        return len(self._modes)

    def __getitem__(self, i: int) -> PhononMode:
        return self._modes[i]

    def __iter__(self):
        return iter(self._modes)

    @property
    def frequencies(self) -> np.ndarray:
        return np.array([m.frequency for m in self._modes])

    @property
    def has_activity(self) -> bool:
        """Whether IR/Raman activity was reported for any mode.

        False for outputs without the selection-rule analysis (dispersion runs),
        where filtering by activity would silently hide everything.
        """
        return any(m.has_activity for m in self._modes)

    @property
    def qpoint(self) -> Optional[np.ndarray]:
        """The q these modes share, or ``None`` for Gamma (and for no modes)."""
        return self._modes[0].qpoint if self._modes else None

    @property
    def is_gamma(self) -> bool:
        """Whether these are zone-centre modes (the only kind a FREQCALC has)."""
        return not self._modes or self._modes[0].is_gamma

    @property
    def qpoint_label(self) -> str:
        """``"Γ"`` or ``"(0, 0, 1/2)"`` — for a q-point selector."""
        return qpoint_label(self.qpoint)


def displaced_positions(
    equilibrium: np.ndarray,
    mode: PhononMode,
    amplitude: float,
    phase: float,
) -> np.ndarray:
    """Return atom positions for one animation frame.

    The eigenvector is rescaled so that ``amplitude`` is the peak displacement
    of the *most-displaced atom*, in Angstrom. Without this the raw eigenvector
    (normalised over all 3N components) makes the motion fade out as 1/sqrt(N),
    so one amplitude setting cannot serve both a molecule and a large cell.

    A complex eigenvector — a mode away from Gamma, whose per-atom phases were
    baked in when the mode was replicated onto the displayed cells — is
    evaluated as ``Im[e * exp(i*phase)] = Re(e) sin(phase) + Im(e) cos(phase)``.
    That is the travelling wave of the module docstring with the time origin
    picked so a real eigenvector still moves as ``e * sin(phase)``: the animation
    of a Gamma mode is unchanged, and any mode still starts from rest at
    ``phase = 0`` in the sense that the *reference* cell does.

    Parameters
    ----------
    equilibrium:
        (N, 3) equilibrium cartesian positions.
    mode:
        The mode being animated.
    amplitude:
        Peak displacement of the most-displaced atom, in Angstrom.
    phase:
        Animation phase in radians; the frame factor is ``sin(phase)``.
    """
    eq = np.asarray(equilibrium, dtype=float)
    if eq.shape != mode.eigenvector.shape:
        raise ValueError(
            f"geometry {eq.shape} does not match eigenvector {mode.eigenvector.shape}"
        )
    peak = mode.peak_displacement
    scale = amplitude / peak if peak > 0.0 else 0.0  # a null mode simply doesn't move
    return eq + scale * frame_displacement(mode.eigenvector, phase)


def frame_displacement(eigenvector: np.ndarray, phase: float) -> np.ndarray:
    """The real (N, 3) displacement pattern of ``eigenvector`` at ``phase``.

    Split out from :func:`displaced_positions` because the same combination is
    what a *still* picture of a mode should draw — see the arrow field in
    ``viz.phonon_animator``, which is this at ``phase = pi/2``.
    """
    ev = np.asarray(eigenvector)
    if not np.iscomplexobj(ev):
        return np.sin(phase) * ev
    return np.real(ev) * np.sin(phase) + np.imag(ev) * np.cos(phase)


def phase_factors(qpoint, offsets) -> Optional[np.ndarray]:
    """``exp(2*pi*i q·n)`` for each lattice translation ``n`` in ``offsets``.

    ``offsets`` is an (N, 3) array of integer lattice translations — the cell an
    image atom was copied into, relative to its parent's. Returns ``None`` when
    the factors would all be 1 (Gamma, or no translations to speak of), which
    lets callers skip the multiplication entirely and keep a real eigenvector
    real.
    """
    if qpoint is None or offsets is None:
        return None
    q = np.asarray(qpoint, dtype=float).ravel()
    if q.size != 3 or np.all(np.abs(q) < _Q_TOL):
        return None
    n = np.asarray(offsets, dtype=float)
    if n.ndim != 2 or n.shape[1] != 3:
        return None
    return np.exp(2.0j * np.pi * (n @ q))


def qpoint_label(qpoint, max_denominator: int = 24) -> str:
    """``"Γ"`` or ``"(0, 0, 1/2)"`` — a q-point as a person reads it.

    CRYSTAL samples q on a grid of simple fractions (an integer triple over the
    shrinking factors), so the components are shown as those fractions rather
    than as 0.333333.
    """
    if qpoint is None:
        return "Γ"
    q = np.asarray(qpoint, dtype=float).ravel()
    if q.size != 3:
        return "Γ"
    if np.all(np.abs(q) < _Q_TOL):
        return "Γ"
    return "(" + ", ".join(_fraction_text(v, max_denominator) for v in q) + ")"


def _fraction_text(value: float, max_denominator: int) -> str:
    """``0.25`` -> ``"1/4"``; anything not a simple fraction stays decimal."""
    if abs(value) < _Q_TOL:
        return "0"
    frac = Fraction(float(value)).limit_denominator(max_denominator)
    if abs(float(frac) - value) > _Q_TOL:
        return f"{value:.3f}"
    return str(frac.numerator) if frac.denominator == 1 else f"{frac.numerator}/{frac.denominator}"


def commensurate_repeats(qpoint, limit: int = 12) -> Tuple[int, int, int]:
    """The smallest tiling in which a mode at ``qpoint`` is a whole wave.

    A mode at ``q = (0, 0, 1/4)`` repeats itself every four cells along **c**:
    drawn on a single cell it is a snapshot with nothing to compare against, and
    only a 1x1x4 tiling shows the wave. Each axis therefore gets the denominator
    of its q component (1 at Gamma).

    ``limit`` caps how many cells an axis may be repeated, so a fine q grid can't
    ask for a structure too big to draw. The cap truncates the wave rather than
    abandoning it: eight cells of a sixteen-cell period still read as a wave,
    where the "no tiling needed" a rounded-off denominator would give reads as a
    Gamma mode.
    """
    if qpoint is None:
        return (1, 1, 1)
    q = np.asarray(qpoint, dtype=float).ravel()
    if q.size != 3:
        return (1, 1, 1)
    reps = []
    for value in q:
        if abs(value) < _Q_TOL:
            reps.append(1)
            continue
        # A generous denominator first (CRYSTAL's shrinking factors are small,
        # but the value has been through a float), then the display cap.
        denominator = Fraction(float(value)).limit_denominator(_MAX_DENOMINATOR).denominator
        reps.append(int(min(max(denominator, 1), limit)))
    return (reps[0], reps[1], reps[2])


__all__ = [
    "PhononMode",
    "PhononModes",
    "commensurate_repeats",
    "displaced_positions",
    "frame_displacement",
    "phase_factors",
    "qpoint_label",
]
