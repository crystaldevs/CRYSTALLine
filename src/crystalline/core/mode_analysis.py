"""What a normal mode is *made of*: which atoms move, and how localised it is.

A frequency list alone doesn't say what a mode is. Scrolling 120 modes and
animating each one to find "the N–H stretches" or "the modes that live on the
adsorbate rather than the substrate" is the routine chore this module removes.

The physics is one line. CRYSTAL prints normal modes as **cartesian
displacements normalised to one** over the whole cell (``sum_a |e_a|^2 = 1``) —
they are orthogonal in the mass-weighted metric, not the plain one, which is
what identifies the convention. The share of the mode's kinetic energy carried
by atom ``a`` is therefore

    w_a = m_a |e_a|^2 / sum_b m_b |e_b|^2

Summing ``w`` over the atoms of one element gives that element's share of the
mode; summing ``w^2`` gives its localisation. ``1 / sum_a w_a^2`` — the inverse
participation number — is the *effective number of atoms in motion*: 1 for a
mode on a single atom, N for a mode spread evenly over the cell. It is reported
as-is rather than as a "localised"/"delocalised" verdict, which would need an
arbitrary threshold.

Qt- and PyVista-free, like the rest of ``core``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from ase.data import atomic_masses, chemical_symbols

from crystalline.core.phonons import PhononMode

# Elements below this share of the mode are dropped from the composition: they
# are rounding noise in a label, not chemistry.
_MIN_SHARE = 0.01


@dataclass(frozen=True)
class ModeCharacter:
    """Which elements carry a mode, and over how many atoms it is spread.

    Attributes
    ----------
    composition:
        ``((symbol, share), …)`` in descending share, shares summing to ~1 over
        the elements that clear :data:`_MIN_SHARE`. Empty for a null mode.
    effective_atoms:
        ``1 / sum_a w_a^2`` — how many atoms are effectively in motion. Compare
        with :attr:`n_atoms`: close to 1 is a mode on one atom, close to
        ``n_atoms`` is a mode spread across the cell.
    n_atoms:
        Atoms in the cell, so ``effective_atoms`` can be read as a fraction.
    """

    composition: Tuple[Tuple[str, float], ...]
    effective_atoms: float
    n_atoms: int

    def composition_text(self, limit: int = 3) -> str:
        """``"H 91%, N 8%"`` — the leading elements, at most ``limit`` of them."""
        parts = [f"{sym} {share * 100:.0f}%" for sym, share in self.composition[:limit]]
        return ", ".join(parts)

    def summary(self, limit: int = 3) -> str:
        """One line for a panel: composition plus how far the mode is spread."""
        if not self.composition:
            return "no motion"
        return (
            f"{self.composition_text(limit)}"
            f"  ·  {self.effective_atoms:.1f} of {self.n_atoms} atoms move"
        )


def mode_character(mode: PhononMode, numbers: Sequence[int]) -> ModeCharacter:
    """Element composition and localisation of ``mode``.

    ``numbers`` are the atomic numbers of the geometry the mode is defined on,
    in the same order as the eigenvector's rows; masses come from those (CRYSTAL
    runs on the dominant isotope by default, which the standard atomic weights
    reproduce closely enough for a percentage).

    A mode whose eigenvector is all zeros — or that doesn't match the geometry —
    comes back empty rather than raising: the panel labels what it can and
    leaves the rest blank.
    """
    return mode_characters([mode], numbers)[0]


def mode_characters(modes: Iterable[PhononMode], numbers: Sequence[int]) -> List[ModeCharacter]:
    """:func:`mode_character` of every mode in ``modes``, on one geometry.

    What a file's whole mode list is labelled with. The geometry's elements and
    masses are worked out once, and each mode's energy is summed per element by
    numpy rather than atom by atom in Python — which, for a few hundred modes of
    a few hundred atoms, was most of the time the Phonons panel took to fill.
    """
    numbers = np.asarray(numbers, dtype=int)
    symbols, element = _elements(numbers)
    masses = atomic_masses[numbers]
    return [_character(mode, numbers, masses, symbols, element) for mode in modes]


def _elements(numbers: np.ndarray):
    """``(symbols, element)``: the elements present, in order of first appearance,
    and for each atom the index of its element among them.

    First appearance, as the composition was built when it was summed atom by
    atom: elements with equal shares then keep the order they always had.
    """
    if len(numbers) == 0:
        return [], np.zeros(0, dtype=int)
    present, first, inverse = np.unique(numbers, return_index=True, return_inverse=True)
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return [chemical_symbols[z] for z in present[order]], rank[np.ravel(inverse)]


def _character(mode: PhononMode, numbers: np.ndarray, masses: np.ndarray, symbols, element):
    """One mode's :class:`ModeCharacter`, given its geometry's elements and masses."""
    eigenvector = np.asarray(mode.eigenvector)
    if len(numbers) == 0 or eigenvector.shape != (len(numbers), 3):
        # Modes and geometry out of step (e.g. a stale selection).
        return ModeCharacter(composition=(), effective_atoms=0.0, n_atoms=len(numbers))
    # |e|^2, so a complex (non-Gamma) eigenvector is measured by the energy its
    # atoms carry over a cycle rather than by whichever phase we caught it at.
    energy = masses * np.sum(np.abs(eigenvector) ** 2, axis=1)
    total = float(energy.sum())
    if not np.isfinite(total) or total <= 0.0:
        # A null mode (all-zero eigenvector) has no composition.
        return ModeCharacter(composition=(), effective_atoms=0.0, n_atoms=len(numbers))
    weights = energy / total
    shares = np.bincount(element, weights=weights, minlength=len(symbols))
    composition = tuple(
        sorted(
            ((sym, float(share)) for sym, share in zip(symbols, shares) if share >= _MIN_SHARE),
            key=lambda item: item[1],
            reverse=True,
        )
    )
    # sum w^2 is bounded below by 1/N, so the reciprocal never blows up.
    effective = float(1.0 / np.sum(weights**2))
    return ModeCharacter(
        composition=composition, effective_atoms=effective, n_atoms=len(weights)
    )


__all__ = ["ModeCharacter", "mode_character", "mode_characters"]
