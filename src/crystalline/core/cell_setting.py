"""Which setting of its space group a cell is written in, and how CRYSTAL names it.

A space-group *type* — number 14, say — can be written in many cells. P2₁/c,
P2₁/n and P2₁/a are one group seen from three choices of axes, and a crystal
computed in P2₁/n has lattice parameters (c = 8.97 Å, β = 105.5°) that its
P2₁/c description does not share (c = 10.43 Å, β = 124.1°). Both are right;
only one of them is the cell the calculation was run in.

The app offers both views, as :data:`COMPUTED` and :data:`STANDARD`:

* **as in the file** — the cell exactly as it is held, in whichever setting of
  its group that cell is. :func:`find_setting` recognises it by asking spglib,
  setting by setting, which one describes the cell with no change of basis.
  (The key is ``"computed"``, though a CIF's cell is not a computed one and
  neither is an edited structure's; the label shown names where the cell came
  from rather than what was done to it.)
* **standard** — pymatgen's conventional standard cell.

Writing either one into a ``.d12`` needs the setting named the way CRYSTAL
reads it (manual, geometry input, record ``IFLAG IFHR IFSO``):

* ``IFLAG = 0`` takes the space-group *number*, which means the standard
  setting of the International Tables; ``IFLAG = 1`` takes the Hermann–Mauguin
  symbol, with a blank between the parts for each symmetry direction, which
  can name a non-standard setting (``P 1 21/N 1``).
* ``IFSO = 0`` puts the origin at the *second* setting of the International
  Tables where a group has two, and ``IFSO = 1`` at the first. Getting this
  wrong is silent and total: silicon written in origin choice 1 and read in
  origin choice 2 has sixteen atoms in the cell instead of eight.
* ``IFHR = 1`` reads a rhombohedral group on rhombohedral axes, ``0`` on
  hexagonal ones.

Nothing is offered on trust: :func:`regenerates` expands the asymmetric unit
with the setting's own operations and checks that the crystal comes back.

Qt-free, like the rest of ``core``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from crystalline.core.structure import Structure

# The two cells the app can report and write.
COMPUTED = "computed"
STANDARD = "standard"
CELL_CHOICES: Tuple[Tuple[str, str], ...] = (
    (COMPUTED, "As in the file"),
    (STANDARD, "Standard setting"),
)
DEFAULT_CHOICE = COMPUTED

_SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def choice_label(choice: str) -> str:
    """``"As in the file"`` / ``"Standard setting"`` for a stored choice."""
    return dict(CELL_CHOICES).get(choice, choice)


@dataclass(frozen=True)
class Setting:
    """One setting of a space group, as it applies to a particular cell.

    ``hall_number`` is spglib's index of the setting (1–530), which fixes the
    axes, the cell choice and the origin together. ``origin_shift`` is what to
    add to the cell's fractional coordinates to put them on the setting's own
    origin — zero whenever the cell already sits there, which a cell CRYSTAL
    wrote always does.
    """

    hall_number: int
    number: int
    choice: str
    full_symbol: str          # spglib's full H-M symbol: "P 1 2_1/n 1"
    short_symbol: str         # the setting's own short symbol: "P2_1/n"
    spaced_symbol: str = ""   # spglib's spaced symbol of the setting: "P b n m"
    origin_shift: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # ── what kind of setting ────────────────────────────────────────────
    @property
    def origin_choice(self) -> Optional[int]:
        """1 or 2 for the groups the International Tables give two origins."""
        return int(self.choice[0]) if self.choice[:1] in ("1", "2") else None

    @property
    def axes(self) -> str:
        """The axis/cell choice alone, the origin choice stripped off."""
        return self.choice[1:] if self.origin_choice is not None else self.choice

    @property
    def is_standard_axes(self) -> bool:
        """Whether the space-group number alone names these axes.

        The standard setting of the International Tables — up to the origin
        choice and the hexagonal/rhombohedral axes, which CRYSTAL takes from
        ``IFSO`` and ``IFHR`` instead of from the symbol.
        """
        # "b" is the standard of the monoclinic groups with no cell choice to
        # make (P2, P2₁, Pm); "b1" of those that have one.
        return self.axes in ("", "b", "b1", "H", "R")

    @property
    def rhombohedral_axes(self) -> bool:
        return self.axes == "R"

    @property
    def unique_axis(self) -> Optional[str]:
        """``"a"``, ``"b"`` or ``"c"`` for a monoclinic group, else ``None``."""
        if not 3 <= self.number <= 15:
            return None
        axis = self.axes.lstrip("-")[:1]
        return axis if axis in ("a", "b", "c") else "b"

    @property
    def shifted(self) -> bool:
        return bool(np.any(np.abs(np.asarray(self.origin_shift)) > 1e-6))

    # ── how it is named ─────────────────────────────────────────────────
    @property
    def symbol(self) -> str:
        """``"P2_1/n"`` — the short symbol of *this setting*, not of the type.

        spglib's own short symbol names the type (it says ``P2_1/c`` for every
        setting of group 14), so this is built from the setting's full symbol:
        a monoclinic one loses its ``1`` directions, the rest their blanks.
        """
        if self.unique_axis is not None:
            parts = self.full_symbol.split()
            if len(parts) == 4:
                parts = [parts[0]] + [p for p in parts[1:] if p != "1"]
            return "".join(parts)
        return "".join((self.spaced_symbol or self.short_symbol).split())

    @property
    def label(self) -> str:
        """``"P2₁/n"`` — the setting's symbol as a person reads it."""
        return _pretty(self.symbol)

    def crystal_record(self) -> Optional[Tuple[str, str]]:
        """``("IFLAG IFHR IFSO", group)`` as a ``.d12`` writes them, or ``None``.

        ``None`` only for a setting there is no symbol for.
        """
        ifhr = 1 if self.rhombohedral_axes else 0
        ifso = 1 if self.origin_choice == 1 else 0
        if self.is_standard_axes:
            return f"0 {ifhr} {ifso}", str(self.number)
        symbol = self._crystal_symbol()
        if symbol is None:
            return None
        return f"1 {ifhr} {ifso}", symbol

    def _crystal_symbol(self) -> Optional[str]:
        """The H-M symbol as CRYSTAL reads it (manual, geometry input, note 10).

        "Given precisely as it appears in the International Tables, with the
        first letter in column one and a blank separating operators referring
        to different symmetry directions", in capitals as the manual's examples
        are, and with a screw axis written ``21``. A monoclinic setting is
        written in full — ``P 1 21/N 1``, ``P 1 1 21/N``, ``P 21/N 1 1`` —
        which is how the manual names the unique axis; the short ``P 21/N``
        would always be read as b-unique.

        The five groups whose glide the 2016 tables renamed ``e`` are spelled
        with the letters of the older tables, which are the ones CRYSTAL lists
        (Appendix A.1: Abm2, Aba2, Cmca, Cmma, Ccca).
        """
        if self.unique_axis is not None:
            text = self.full_symbol
        elif 16 <= self.number <= 74 and len(self.spaced_symbol.split()) == 4:
            text = _OLD_GLIDES.get((self.number, self.axes), self.spaced_symbol)
        else:
            return None
        if any("e" in part for part in text.split()[1:]):
            return None
        return text.replace("_", "").upper()

    def lattice_parameters(self, lattice) -> Tuple[float, ...]:
        """The parameters CRYSTAL reads for this setting, from a pymatgen ``Lattice``.

        Only those the group leaves free (manual, geometry input): one for a
        cubic cell, ``a, c`` for tetragonal and hexagonal axes, ``a, α`` on
        rhombohedral ones, and for a monoclinic cell the angle at its unique
        axis — β for b, γ for c, α for a.
        """
        a, b, c = lattice.a, lattice.b, lattice.c
        alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma
        n = self.number
        if n <= 2:
            return (a, b, c, alpha, beta, gamma)
        if n <= 15:
            angle = {"a": alpha, "b": beta, "c": gamma}[self.unique_axis or "b"]
            return (a, b, c, angle)
        if n <= 74:
            return (a, b, c)
        if n <= 142:
            return (a, c)
        if n <= 167 and self.rhombohedral_axes:
            return (a, alpha)
        if n <= 194:
            return (a, c)
        return (a,)


# The six orthorhombic settings of the five groups whose glide became "e" in
# the 2016 International Tables, in the older letters CRYSTAL knows them by
# (ITA Table 4.3.2.1). Keyed by spglib's setting — the axis permutation, with
# any origin choice stripped off.
_OLD_GLIDES: Dict[Tuple[int, str], str] = {
    (39, ""): "A b m 2", (39, "ba-c"): "B m a 2", (39, "cab"): "B 2 c m",
    (39, "-cba"): "C 2 m b", (39, "bca"): "C m 2 a", (39, "a-cb"): "A c 2 m",
    (41, ""): "A b a 2", (41, "ba-c"): "B b a 2", (41, "cab"): "B 2 c b",
    (41, "-cba"): "C 2 c b", (41, "bca"): "C c 2 a", (41, "a-cb"): "A c 2 a",
    (64, ""): "C m c a", (64, "ba-c"): "C c m b", (64, "cab"): "A b m a",
    (64, "-cba"): "A c a m", (64, "bca"): "B b c m", (64, "a-cb"): "B m a b",
    (67, ""): "C m m a", (67, "ba-c"): "C m m b", (67, "cab"): "A b m m",
    (67, "-cba"): "A c m m", (67, "bca"): "B m c m", (67, "a-cb"): "B m a m",
    (68, ""): "C c c a", (68, "ba-c"): "C c c b", (68, "cab"): "A b a a",
    (68, "-cba"): "A c a a", (68, "bca"): "B b c b", (68, "a-cb"): "B b a b",
}


def _pretty(symbol: str) -> str:
    """``"P2_1/n"`` -> ``"P2₁/n"``, ``"Fd-3m"`` -> ``"Fd3̄m"``."""
    import re

    text = re.sub(r"_(\d)", lambda m: m.group(1).translate(_SUBSCRIPTS), symbol)
    return re.sub(r"-(\d)", lambda m: m.group(1) + "̄", text)


# ── spglib's table of settings ────────────────────────────────────────────
@lru_cache(maxsize=None)
def _halls_by_number() -> Dict[int, Tuple[int, ...]]:
    import spglib

    table: Dict[int, List[int]] = {}
    for hall in range(1, 531):
        table.setdefault(int(spglib.get_spacegroup_type(hall).number), []).append(hall)
    return {number: tuple(halls) for number, halls in table.items()}


def halls_of(number: int) -> Tuple[int, ...]:
    """Every Hall setting of space-group type ``number``, spglib's order."""
    return _halls_by_number().get(int(number), ())


@lru_cache(maxsize=None)
def setting_type(hall: int) -> Tuple[int, str, str, str, str]:
    """``(number, choice, full, short, spaced)`` symbols of Hall setting ``hall``.

    ``spaced`` is spglib's own spelling of the setting with a blank between
    directions (``"P b n m"``); for a monoclinic setting spglib prefixes it
    with the standard symbol (``"P 2_1/c = P 1 2_1/n 1"``), which is dropped.
    """
    import spglib

    kind = spglib.get_spacegroup_type(int(hall))
    spaced = str(kind.international).split(" = ")
    spaced = spaced[1] if len(spaced) > 1 else spaced[0]
    return (int(kind.number), str(kind.choice), str(kind.international_full),
            str(kind.international_short), spaced)


@lru_cache(maxsize=None)
def operations(hall: int) -> Tuple[np.ndarray, np.ndarray]:
    """``(rotations, translations)`` of Hall setting ``hall``, in its own basis."""
    import spglib

    ops = spglib.get_symmetry_from_database(int(hall))
    return (np.asarray(ops["rotations"], dtype=int),
            np.asarray(ops["translations"], dtype=float))


def setting_from_hall(hall: int, origin_shift=(0.0, 0.0, 0.0)) -> Setting:
    number, choice, full, short, spaced = setting_type(hall)
    shift = tuple(float(v) for v in np.asarray(origin_shift, dtype=float).ravel())
    return Setting(hall_number=int(hall), number=number, choice=choice,
                   full_symbol=full, short_symbol=short, spaced_symbol=spaced,
                   origin_shift=shift)


# ── recognising the setting of a cell ─────────────────────────────────────
def find_setting(structure: Structure, symprec: float = 1e-2) -> Optional[Setting]:
    """The setting of its space group that ``structure``'s cell is written in.

    ``None`` when there is no such setting: not a 3D crystal, no symmetry found,
    or a cell that is not a conventional cell of any setting — the primitive
    cell of a centred lattice, say, or a cell spanning several conventional
    ones. A caller then falls back on the standard cell.

    Each setting of the group is tried with spglib as the target; one that
    needs no change of basis is the cell's own. A cell whose origin sits away
    from the setting's is still accepted, with the shift recorded — moving the
    origin moves no atom relative to another. Whatever is found is checked by
    applying the setting's own operations to the atoms.
    """
    if not _is_crystal(structure):
        return None
    lattice, positions, numbers = _cell(structure)
    key = (lattice.round(8).tobytes(), positions.round(8).tobytes(),
           numbers.tobytes(), float(symprec))
    if key not in _SETTINGS_SEEN:
        if len(_SETTINGS_SEEN) >= _SETTINGS_CACHE_SIZE:
            _SETTINGS_SEEN.pop(next(iter(_SETTINGS_SEEN)))
        _SETTINGS_SEEN[key] = _search(lattice, positions, numbers, symprec)
    return _SETTINGS_SEEN[key]


# A dialog asks again on every keystroke — the deck preview is rebuilt live —
# and a search over every setting of a monoclinic group is eighteen symmetry
# searches. The answer depends on nothing but the cell, so it is kept.
_SETTINGS_CACHE_SIZE = 16
_SETTINGS_SEEN: Dict[tuple, Optional[Setting]] = {}


def _search(lattice, positions, numbers, symprec) -> Optional[Setting]:
    """Try every setting of the cell's group against the cell's own operations.

    spglib is asked for each setting as its target, but its answer is used only
    for where the origin might be: when the group leaves the axes
    interchangeable (P222, P2₁2₁2₁, P1̄ …) spglib reorders them to its own
    liking even for a cell already in that setting, so "no change of basis" is
    not the test. The test is whether the setting's operations, with the
    origin moved, are exactly the operations of the cell as it stands.
    """
    try:
        import spglib

        cell = (lattice, positions, numbers)
        dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
        if dataset is None:
            return None
        own = (np.asarray(dataset.rotations, dtype=int),
               np.asarray(dataset.translations, dtype=float))
        found = []
        for hall in halls_of(int(dataset.number)):
            candidate = spglib.get_symmetry_dataset(cell, symprec=symprec, hall_number=hall)
            for shift in _origin_candidates(hall, candidate, own):
                setting = setting_from_hall(hall, shift)
                if _same_operations(setting, own, lattice, symprec):
                    found.append((setting.shifted, hall, setting))
                    break
    except Exception:  # noqa: BLE001 - no setting is a usable answer; the caller falls back
        return None
    if not found:
        return None
    return min(found, key=lambda item: (item[0], item[1]))[2]


def _origin_candidates(hall: int, dataset, own) -> List[np.ndarray]:
    """Where the setting's origin might sit in the cell, most likely first.

    Where it already is (no shift) — which is where any cell written by
    CRYSTAL has it — then spglib's shift for this setting, and for a
    centrosymmetric group the inversion centre, which pins the origin of every
    centrosymmetric setting on its own.
    """
    candidates = [np.zeros(3)]
    if dataset is not None:
        transform = np.asarray(dataset.transformation_matrix, dtype=float)
        shift = np.asarray(dataset.origin_shift, dtype=float)
        if abs(np.linalg.det(transform)) > 1e-8:
            candidates.append(np.linalg.solve(transform, shift))
        candidates.append(shift)
    rotations, translations = operations(hall)
    own_rotations, own_translations = own
    inversion = -np.eye(3, dtype=int)
    mine = [t for r, t in zip(rotations, translations) if np.array_equal(r, inversion)]
    theirs = [t for r, t in zip(own_rotations, own_translations) if np.array_equal(r, inversion)]
    if mine and theirs:
        # (-I) acting about origin s: t + (-2I)s ≡ τ, so s ≡ (t - τ)/2.
        candidates.append((mine[0] - theirs[0]) / 2.0)
    out: List[np.ndarray] = []
    for shift in candidates:
        shift = _wrapped(shift)
        if not any(np.allclose(shift, other, atol=1e-6) for other in out):
            out.append(shift)
    return out


def holds(setting: Setting, structure: Structure, symprec: float = 1e-2) -> bool:
    """Whether ``setting``'s operations are exactly the crystal's own symmetry."""
    try:
        import spglib

        lattice, positions, numbers = _cell(structure)
        dataset = spglib.get_symmetry_dataset((lattice, positions, numbers), symprec=symprec)
    except Exception:  # noqa: BLE001
        return False
    if dataset is None:
        return False
    own = (np.asarray(dataset.rotations, dtype=int),
           np.asarray(dataset.translations, dtype=float))
    return _same_operations(setting, own, lattice, symprec)


def _same_operations(setting: Setting, own, lattice, symprec) -> bool:
    """Whether the setting's operations, moved to the cell's origin, are the cell's own.

    With the setting's coordinates ``x' = x + s``, its operation ``(R, t)`` acts
    on the cell's coordinates as ``(R, t + (R - I) s)``. Comparing operator
    lists is exact and cheap, where testing every image of every atom is
    neither — and it is the same question, since spglib's list is the
    crystal's complete symmetry in this very cell.
    """
    rotations, translations = operations(setting.hall_number)
    own_rotations, own_translations = own
    if len(rotations) != len(own_rotations):
        return False
    shift = np.asarray(setting.origin_shift, dtype=float)
    unmatched = list(range(len(own_rotations)))
    for rotation, translation in zip(rotations, translations):
        moved = translation + (rotation - np.eye(3)) @ shift
        hit = None
        for index in unmatched:
            if not np.array_equal(own_rotations[index], rotation):
                continue
            delta = own_translations[index] - moved
            delta -= np.round(delta)
            if np.linalg.norm(delta @ lattice) < 2.0 * symprec:
                hit = index
                break
        if hit is None:
            return False
        unmatched.remove(hit)
    return True


def standard_cell(structure: Structure, symprec: float = 1e-2) -> Optional[Structure]:
    """pymatgen's conventional standard cell of ``structure``, or ``None``.

    What the app has always reported and written: the cell of spglib's refined
    structure, re-oriented to pymatgen's conventions (an orthorhombic cell with
    a < b < c, for one), which need not be the standard *setting* of the
    International Tables.
    """
    if not _is_crystal(structure):
        return None
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        pmg = AseAtomsAdaptor().get_structure(structure.to_ase())
        conventional = SpacegroupAnalyzer(pmg, symprec=symprec).get_conventional_standard_structure()
        return Structure.from_ase(AseAtomsAdaptor().get_atoms(conventional))
    except Exception:  # noqa: BLE001
        return None


# ── the asymmetric unit, and the check that it rebuilds the crystal ───────
def asymmetric_unit(setting: Setting, structure: Structure,
                    symprec: float = 1e-2) -> Tuple[np.ndarray, np.ndarray]:
    """``(atomic numbers, fractional coordinates)``: one atom per orbit.

    Coordinates are in the cell as held, moved onto the setting's origin.
    """
    lattice, positions, numbers = _cell(structure)
    fractional = positions + np.asarray(setting.origin_shift)
    rotations, translations = operations(setting.hall_number)
    seen = np.zeros(len(fractional), dtype=bool)
    keep: List[int] = []
    for index in range(len(fractional)):
        if seen[index]:
            continue
        keep.append(index)
        images = fractional[index] @ rotations.transpose(0, 2, 1) + translations
        for image in images:
            seen |= _matches(image, fractional, numbers, numbers[index], lattice, symprec)
    return numbers[keep], _wrapped(fractional[keep])


def regenerates(setting: Setting, structure: Structure, atomic_numbers: Sequence[int],
                coordinates: np.ndarray, symprec: float = 1e-2) -> bool:
    """Does ``setting`` applied to these atoms give ``structure`` back — no more, no less?

    This is what CRYSTAL will do with the deck, so it is the one check that
    cannot be fooled by a plausible but wrong setting.
    """
    lattice, positions, numbers = _cell(structure)
    target = positions + np.asarray(setting.origin_shift)
    rotations, translations = operations(setting.hall_number)
    built_numbers: List[int] = []
    built: List[np.ndarray] = []
    for z, point in zip(atomic_numbers, np.asarray(coordinates, dtype=float)):
        for image in point @ rotations.transpose(0, 2, 1) + translations:
            if built and np.any(_matches(image, np.asarray(built), np.asarray(built_numbers),
                                         int(z), lattice, symprec)):
                continue
            built.append(image)
            built_numbers.append(int(z))
    if len(built) != len(target):
        return False
    return _all_present(np.asarray(built), np.asarray(built_numbers),
                        target, numbers, lattice, symprec)


# ── plumbing ──────────────────────────────────────────────────────────────
def _is_crystal(structure: Structure) -> bool:
    try:
        cell = np.asarray(structure.cell, dtype=float)
        return (len(structure) > 0 and bool(np.all(structure.pbc))
                and abs(np.linalg.det(cell)) > 1e-8)
    except Exception:  # noqa: BLE001
        return False


def _cell(structure: Structure):
    atoms = structure.to_ase()
    return (np.asarray(atoms.get_cell(), dtype=float),
            np.asarray(atoms.get_scaled_positions(wrap=False), dtype=float),
            np.asarray(atoms.get_atomic_numbers(), dtype=int))


def _wrapped(fractional: np.ndarray) -> np.ndarray:
    """Into ``(-0.5, 0.5]``, with a hair of tolerance so ``0.5`` stays ``0.5``."""
    out = np.asarray(fractional, dtype=float)
    out = out - np.round(out - 1e-9)
    out[np.abs(out) < 1e-12] = 0.0
    return out


def _matches(point, fractional, numbers, z, lattice, symprec) -> np.ndarray:
    """Mask of the atoms of element ``z`` sitting at ``point`` (periodically)."""
    delta = np.asarray(fractional, dtype=float) - np.asarray(point, dtype=float)
    delta -= np.round(delta)
    distance = np.linalg.norm(delta @ lattice, axis=1)
    return (np.asarray(numbers) == int(z)) & (distance < 2.0 * symprec)


def _all_present(points, point_numbers, fractional, numbers, lattice, symprec) -> bool:
    """Whether every point has an atom of its element at it."""
    for point, z in zip(points, point_numbers):
        if not np.any(_matches(point, fractional, numbers, z, lattice, symprec)):
            return False
    return True


__all__ = [
    "CELL_CHOICES",
    "COMPUTED",
    "DEFAULT_CHOICE",
    "STANDARD",
    "Setting",
    "asymmetric_unit",
    "choice_label",
    "find_setting",
    "halls_of",
    "holds",
    "operations",
    "regenerates",
    "setting_from_hall",
    "standard_cell",
]
