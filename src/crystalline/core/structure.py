"""In-memory structure model, backed by :class:`ase.Atoms`.

``Structure`` is the single source of truth the UI edits and the renderer
draws. It wraps ``ase.Atoms`` (chosen for its light, ergonomic editing API and
because CRYSTALClear already depends on ase) and adds:

* a small, explicit editing vocabulary (add / remove / move atoms) — the
  multi-atom forms fire a single change notification per user action,
* a change-notification hook so views can refresh without ``core`` importing Qt.

Keeping this Qt-free is deliberate: the same model is exercised directly in
unit tests with no display server.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, chemical_symbols

ChangeListener = Callable[["Structure"], None]


class Structure:
    """A periodic (or molecular) atomic structure the GUI edits.

    Parameters
    ----------
    atoms:
        An existing ``ase.Atoms`` to wrap. If ``None``, starts empty.
    """

    def __init__(self, atoms: Optional[Atoms] = None) -> None:
        self._atoms: Atoms = atoms if atoms is not None else Atoms()
        self._listeners: list[ChangeListener] = []

    # ── construction helpers ────────────────────────────────────────────
    @classmethod
    def empty(cls) -> "Structure":
        """An empty structure with no cell (a blank molecular canvas)."""
        return cls(Atoms())

    @classmethod
    def from_ase(cls, atoms: Atoms) -> "Structure":
        return cls(atoms.copy())

    def to_ase(self) -> Atoms:
        """Return a *copy* of the underlying ase.Atoms (safe to hand out)."""
        return self._atoms.copy()

    def restore(self, atoms: Atoms) -> None:
        """Replace the entire state (atoms + cell) with a snapshot, then notify.

        Used by undo: a snapshot taken with :meth:`to_ase` is put back wholesale,
        so the atom count may change (undoing a delete/duplicate). The
        ``Structure`` object identity — and thus its listeners — is preserved.
        """
        self._atoms = atoms.copy()
        self._notify()

    # ── change notification (no Qt in core) ─────────────────────────────
    def add_listener(self, fn: ChangeListener) -> None:
        """Register ``fn`` to be called after every mutating edit."""
        if fn not in self._listeners:
            self._listeners.append(fn)

    def remove_listener(self, fn: ChangeListener) -> None:
        if fn in self._listeners:
            self._listeners.remove(fn)

    def _notify(self) -> None:
        for fn in list(self._listeners):
            fn(self)

    # ── read-only accessors ─────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._atoms)

    @property
    def positions(self) -> np.ndarray:
        """(N, 3) cartesian positions in Angstrom."""
        return self._atoms.get_positions()

    @property
    def symbols(self) -> list[str]:
        return list(self._atoms.get_chemical_symbols())

    @property
    def numbers(self) -> np.ndarray:
        return self._atoms.get_atomic_numbers()

    @property
    def cell(self) -> np.ndarray:
        """(3, 3) lattice vectors (rows). All-zero for a non-periodic system."""
        return np.asarray(self._atoms.cell)

    @property
    def pbc(self) -> np.ndarray:
        return self._atoms.get_pbc()

    @property
    def is_periodic(self) -> bool:
        return bool(np.any(self._atoms.get_pbc()))

    @property
    def cellpar(self) -> np.ndarray:
        """Lattice parameters ``[a, b, c, alpha, beta, gamma]`` (Å and degrees)."""
        return np.asarray(self._atoms.cell.cellpar(), dtype=float)

    # ── editing vocabulary ──────────────────────────────────────────────
    def set_cell(self, cell, periodic: bool = True) -> None:
        """Set the (3,3) lattice and toggle periodicity."""
        self._atoms.set_cell(cell)
        self._atoms.set_pbc(periodic)
        self._notify()

    def set_lattice_parameters(
        self,
        a: float,
        b: float,
        c: float,
        alpha: float,
        beta: float,
        gamma: float,
        scale_atoms: bool = True,
    ) -> None:
        """Reshape the cell from lattice parameters (lengths in Å, angles in degrees).

        ``scale_atoms`` (the default) moves the atoms with the cell so their
        fractional coordinates are preserved — the usual intent when editing
        lattice parameters. The system is marked periodic.
        """
        self._atoms.set_cell([a, b, c, alpha, beta, gamma], scale_atoms=scale_atoms)
        self._atoms.set_pbc(True)
        self._notify()

    def add_atom(self, symbol: str, position: Iterable[float]) -> int:
        """Append an atom, returning its index."""
        _validate_symbol(symbol)
        self._atoms += Atoms(symbol, positions=[list(position)])
        self._notify()
        return len(self._atoms) - 1

    def add_atoms(self, symbols: Iterable[str], positions: Iterable[Iterable[float]]) -> list[int]:
        """Append several atoms at once (one change notification), returning their indices.

        Used when importing atoms from a file into the current structure.
        """
        symbols = list(symbols)
        positions = [list(p) for p in positions]
        if len(symbols) != len(positions):
            raise ValueError("symbols and positions must have the same length")
        for symbol in symbols:
            _validate_symbol(symbol)
        if not symbols:
            return []
        start = len(self._atoms)
        self._atoms += Atoms(symbols, positions=positions)
        self._notify()
        return list(range(start, start + len(symbols)))

    def move_atom(self, index: int, position: Iterable[float]) -> None:
        """Set an atom's cartesian position (used by drag-to-edit)."""
        self._check_index(index)
        pos = self._atoms.get_positions()
        pos[index] = list(position)
        self._atoms.set_positions(pos)
        self._notify()

    # ── batch editing (multi-atom selection) ────────────────────────────
    # Each of these mutates several atoms but fires a single change
    # notification, so the view redraws once per user action.
    def remove_atoms(self, indices: Iterable[int]) -> None:
        """Delete several atoms at once."""
        unique = sorted({int(i) for i in indices}, reverse=True)
        for i in unique:
            self._check_index(i)
        if not unique:
            return
        for i in unique:  # high-to-low so earlier indices stay valid
            del self._atoms[i]
        self._notify()

    def translate_atoms(self, indices: Iterable[int], vector: Iterable[float]) -> None:
        """Shift the given atoms by a cartesian ``vector`` (Angstrom)."""
        unique = {int(i) for i in indices}
        for i in unique:
            self._check_index(i)
        if not unique:
            return
        delta = np.asarray(list(vector), dtype=float)
        pos = self._atoms.get_positions()
        for i in unique:
            pos[i] = pos[i] + delta
        self._atoms.set_positions(pos)
        self._notify()

    def set_symbols(self, indices: Iterable[int], symbol: str) -> None:
        """Change the element of several atoms at once."""
        _validate_symbol(symbol)
        unique = {int(i) for i in indices}
        for i in unique:
            self._check_index(i)
        if not unique:
            return
        syms = self._atoms.get_chemical_symbols()
        for i in unique:
            syms[i] = symbol
        self._atoms.set_chemical_symbols(syms)
        self._notify()

    def duplicate_atoms(
        self, indices: Iterable[int], offset: Iterable[float] = (0.0, 0.0, 0.0)
    ) -> list[int]:
        """Append copies of the given atoms, shifted by ``offset``.

        Returns the indices of the new atoms (a contiguous block at the end),
        which the UI selects so the copies can be moved straight away.
        """
        ordered = sorted({int(i) for i in indices})
        for i in ordered:
            self._check_index(i)
        if not ordered:
            return []
        delta = np.asarray(list(offset), dtype=float)
        syms = self._atoms.get_chemical_symbols()
        pos = self._atoms.get_positions()
        copies = Atoms(
            [syms[i] for i in ordered],
            positions=[pos[i] + delta for i in ordered],
        )
        start = len(self._atoms)
        self._atoms += copies
        self._notify()
        return list(range(start, start + len(ordered)))

    # ── internals ───────────────────────────────────────────────────────
    def _check_index(self, index: int) -> None:
        if not 0 <= index < len(self._atoms):
            raise IndexError(f"atom index {index} out of range (n={len(self._atoms)})")


def _validate_symbol(symbol: str) -> None:
    if symbol not in atomic_numbers:
        raise ValueError(f"unknown element symbol: {symbol!r}")


__all__ = ["Structure", "ChangeListener", "chemical_symbols"]
