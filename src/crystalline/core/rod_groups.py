"""The rod group of a polymer, as CRYSTAL numbers it, and its asymmetric unit.

A polymer repeats along one direction only, so its symmetry is a *rod group*:
operations that leave the chain's axis in place, translating along it at most.
CRYSTAL's POLYMER input takes one of 99 (manual, Appendix A.3), each named
after the space group it is a subset of, in a stated setting with the chain
along *z*. The operators are generated for that space group and then turned
90° about *y*, so that the chain runs along *x*, CRYSTAL's polymer axis, and
the rod axis passes through the origin of *y* and *z*. A water chain in rod
group 4, P 2₁11, shows it: CRYSTAL prints its operators as the identity and
(x + ½, −y, −z), the 2₁ screw of space group P 1 1 2₁ turned onto *x*.

The operators are spglib's, for the setting the manual names. That every one
of them translates along the chain only, never across it, is what makes those
settings rod groups, and is checked when they are built.

Finding the group of a chain needs three things the space-group finders do
not give, since a polymer has no lattice across it:

* **where the axis is** — through the centroid of the atoms projected across
  the chain, which every operation of a rod group leaves where it is;
* **how the chain is turned about it** — two-fold axes across the chain and
  mirrors along it both show as reflection lines of that projection, so those
  lines are the orientations worth trying, every one of them;
* **where the origin is along it** — for operations that reverse the chain,
  solved from where they send one atom.

In each orientation the groups are tried from the largest down, and the
largest the chain satisfies in any of them is taken; :func:`regenerates` checks that its asymmetric unit
rebuilds the chain before a deck is written in it.

Qt-free, like the rest of ``core``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import numpy as np

# (IGR, polymer symbol with x along the chain, space-group symbol with z along
# it, space-group number) — Appendix A.3.
TABLE: Tuple[Tuple[int, str, str, int], ...] = (
    (1, "P1", "P 1", 1), (2, "P-1", "P -1", 2), (3, "P211", "P 1 1 2", 3),
    (4, "P2_111", "P 1 1 21", 4), (5, "P121", "P 1 2 1", 3), (6, "P112", "P 2 1 1", 3),
    (7, "Pm11", "P 1 1 m", 6), (8, "P1m1", "P 1 m 1", 6), (9, "P1a1", "P 1 c 1", 7),
    (10, "P11m", "P m 1 1", 6), (11, "P11a", "P c 1 1", 7),
    (12, "P2/m11", "P 1 1 2/m", 10), (13, "P2_1/m11", "P 1 1 21/m", 11),
    (14, "P12/m1", "P 1 2/m 1", 10), (15, "P12/a1", "P 1 2/c 1", 13),
    (16, "P112/m", "P 2/m 1 1", 10), (17, "P112/a", "P 2/c 1 1", 13),
    (18, "P222", "P 2 2 2", 16), (19, "P2_122", "P 2 2 21", 17), (20, "P2mm", "P m m 2", 25),
    (21, "P2_1am", "P m c 21", 26), (22, "P2_1ma", "P c m 21", 26), (23, "P2aa", "P c c 2", 27),
    (24, "Pm2m", "P m 2 m", 25), (25, "Pm2a", "P c 2 m", 28), (26, "Pmm2", "P 2 m m", 25),
    (27, "Pma2", "P 2 c m", 28), (28, "Pmmm", "P m m m", 47), (29, "P2/m2/a2/a", "P c c m", 49),
    (30, "P2_1/m2/m2/a", "P c m m", 51), (31, "P2_1/m2/a2/m", "P m c m", 51),
    (32, "P4", "P 4", 75), (33, "P4_1", "P 41", 76), (34, "P4_2", "P 42", 77),
    (35, "P4_3", "P 43", 78), (36, "P-4", "P -4", 81), (37, "P4/m", "P 4/m", 83),
    (38, "P4_2/m", "P 42/m", 84), (39, "P422", "P 4 2 2", 89), (40, "P4_122", "P 41 2 2", 91),
    (41, "P4_222", "P 42 2 2", 93), (42, "P4_322", "P 43 2 2", 95), (43, "P4mm", "P 4 m m", 99),
    (44, "P4_2am", "P 42 c m", 101), (45, "P4aa", "P 4 c c", 103),
    (46, "P4_2ma", "P 42 m c", 105), (47, "P-42m", "P -4 2 m", 111),
    (48, "P-42a", "P -4 2 c", 112), (49, "P-4m2", "P -4 m 2", 115),
    (50, "P-4a2", "P -4 c 2", 116), (51, "P4/mmm", "P 4/m m m", 123),
    (52, "P4/m2/a2/a", "P 4/m c c", 124), (53, "P4_2/m2/m2/a", "P 42/m m c", 131),
    (54, "P4_2/m2/a2/m", "P 42/m c m", 132),
    (55, "P3", "P 3", 143), (56, "P3_1", "P 31", 144), (57, "P3_2", "P 32", 145),
    (58, "P-3", "P -3", 147), (59, "P312", "P 3 1 2", 149), (60, "P3_112", "P 31 1 2", 151),
    (61, "P3_212", "P 32 1 2", 153), (62, "P321", "P 3 2 1", 150),
    (63, "P3_121", "P 31 2 1", 152), (64, "P3_221", "P 32 2 1", 154),
    (65, "P3m1", "P 3 m 1", 156), (66, "P3a1", "P 3 c 1", 158), (67, "P31m", "P 3 1 m", 157),
    (68, "P31a", "P 3 1 c", 159), (69, "P-31m", "P -3 1 m", 162), (70, "P-31a", "P -3 1 c", 163),
    (71, "P-3m1", "P -3 m 1", 164), (72, "P-3a1", "P -3 c 1", 165),
    (73, "P6", "P 6", 168), (74, "P6_1", "P 61", 169), (75, "P6_5", "P 65", 170),
    (76, "P6_2", "P 62", 171), (77, "P6_4", "P 64", 172), (78, "P6_3", "P 63", 173),
    (79, "P-6", "P -6", 174), (80, "P6/m", "P 6/m", 175), (81, "P6_3/m", "P 63/m", 176),
    (82, "P622", "P 6 2 2", 177), (83, "P6_122", "P 61 2 2", 178),
    (84, "P6_522", "P 65 2 2", 179), (85, "P6_222", "P 62 2 2", 180),
    (86, "P6_422", "P 64 2 2", 181), (87, "P6_322", "P 63 2 2", 182),
    (88, "P6mm", "P 6 m m", 183), (89, "P6aa", "P 6 c c", 184), (90, "P6_3am", "P 63 c m", 185),
    (91, "P6_3ma", "P 63 m c", 186), (92, "P-6m2", "P -6 m 2", 187),
    (93, "P-6a2", "P -6 c 2", 188), (94, "P-62m", "P -6 2 m", 189),
    (95, "P-62a", "P -6 2 c", 190), (96, "P6/mmm", "P 6/m m m", 191),
    (97, "P6/m2/a2/a", "P 6/m c c", 192), (98, "P6_3/m2/a2/m", "P 63/m c m", 193),
    (99, "P6_3/m2/m2/a", "P 63/m m c", 194),
)

# The space group's z axis onto the polymer's x, its y kept: a turn of 90°
# about y. Rows are the polymer axes in the space group's frame.
_TO_POLYMER = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])


@dataclass(frozen=True)
class RodGroup:
    """One of CRYSTAL's 99 rod groups, its operators in the polymer frame.

    ``rotations`` are Cartesian, ``translations`` the shift of each along the
    chain, in fractions of its repeat.
    """

    number: int
    symbol: str
    space_group: str
    rotations: Tuple[np.ndarray, ...]
    translations: Tuple[float, ...]

    @property
    def order(self) -> int:
        return len(self.rotations)


@lru_cache(maxsize=None)
def rod_group(number: int) -> RodGroup:
    """Rod group ``number`` (1–99), its operators generated as CRYSTAL generates them."""
    import spglib

    igr, symbol, setting, space_group = TABLE[int(number) - 1]
    hall = _hall_number(setting, space_group)
    ops = spglib.get_symmetry_from_database(hall)
    basis = _basis(space_group)
    inverse = np.linalg.inv(basis)
    rotations, translations = [], []
    for rotation, translation in zip(ops["rotations"], ops["translations"]):
        across = (np.asarray(translation[:2], dtype=float) + 0.5) % 1.0 - 0.5
        if np.abs(across).max() > 1e-9:
            raise ValueError(f"{setting} translates across the chain: not a rod group")
        cartesian = basis @ np.asarray(rotation, dtype=float) @ inverse
        rotations.append(np.round(_TO_POLYMER @ cartesian @ _TO_POLYMER.T, 12))
        translations.append(float(translation[2]) % 1.0)
    return RodGroup(igr, symbol, setting, tuple(rotations), tuple(translations))


def _hall_number(setting: str, space_group: int) -> int:
    """The spglib setting of ``space_group`` whose symbol is ``setting``."""
    import spglib

    wanted = setting.replace(" ", "")
    short = None
    for hall in range(1, 531):
        kind = spglib.get_spacegroup_type(hall)
        if kind.number != space_group:
            continue
        if kind.international_full.replace(" ", "").replace("_", "") == wanted:
            return hall
        if short is None and kind.international.replace(" ", "").replace("_", "") == wanted:
            short = hall
    if short is None:
        raise ValueError(f"no setting {setting} of space group {space_group}")
    return short


def _basis(space_group: int) -> np.ndarray:
    """The space group's axes as columns: hexagonal for 143–194, else orthonormal."""
    if 143 <= space_group <= 194:
        return np.array([[1.0, -0.5, 0.0], [0.0, np.sqrt(3.0) / 2.0, 0.0], [0.0, 0.0, 1.0]])
    return np.eye(3)


# ── finding a chain's group ───────────────────────────────────────────────
@dataclass(frozen=True)
class RodSymmetry:
    """A chain in its rod group, laid out as the POLYMER input wants it.

    ``positions`` are every atom of one repeat, *x* fractional along the chain
    and *y*, *z* in Å from the rod axis; ``numbers`` their atomic numbers.
    """

    group: RodGroup
    a: float
    numbers: np.ndarray
    positions: np.ndarray
    tolerance: float


def analyse(numbers: Sequence[int], positions: np.ndarray, a: float,
            tolerance: float = 1e-2) -> RodSymmetry:
    """The highest rod group the chain has, among CRYSTAL's 99.

    ``positions`` are Cartesian, in Å, with *x* along the chain and the chain
    repeating every ``a``; ``tolerance`` is in Å. Rod group 1 when nothing more
    holds, which it always does.
    """
    numbers = np.asarray(numbers, dtype=int)
    cartesian = np.asarray(positions, dtype=float)
    across = cartesian[:, 1:] - cartesian[:, 1:].mean(axis=0)
    base = np.column_stack([(cartesian[:, 0] / a) % 1.0, across])
    groups = sorted((rod_group(igr) for igr, *_rest in TABLE),
                    key=lambda group: (-group.order, group.number))
    # Every orientation is tried, not the first that fits: a screw along the
    # chain fits however the chain is turned, and would end the search before
    # the turn that lines up its two-fold axes across it.
    best = RodSymmetry(rod_group(1), a, numbers, base, tolerance)
    for angle in _orientations(numbers, across, tolerance):
        turned = base.copy()
        turned[:, 1:] = across @ _turn(angle).T
        for group in groups:
            if group.order <= best.group.order:
                break
            origin = _origin(group, numbers, turned, a, tolerance)
            if origin is None:
                continue
            shifted = turned.copy()
            shifted[:, 0] = (shifted[:, 0] - origin) % 1.0
            best = RodSymmetry(group, a, numbers, shifted, tolerance)
            break
    return best


def of_atoms(atoms, tolerance: float = 1e-2) -> RodSymmetry:
    """:func:`analyse` for a polymer held as ``ase.Atoms``: periodic along one axis."""
    axis = next(i for i, periodic in enumerate(atoms.get_pbc()) if periodic)
    along = np.asarray(atoms.get_cell(), dtype=float)[axis]
    a = float(np.linalg.norm(along))
    e1 = along / a
    helper = np.eye(3)[int(np.argmin(np.abs(e1)))]
    e2 = np.cross(e1, helper)
    e2 /= np.linalg.norm(e2)
    frame = np.column_stack([e1, e2, np.cross(e1, e2)])
    return analyse(atoms.get_atomic_numbers(), atoms.get_positions() @ frame, a, tolerance)


def _turn(angle: float) -> np.ndarray:
    """The rotation about the chain by ``angle`` (radians), on (y, z)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def _orientations(numbers, across, tolerance) -> List[float]:
    """Turns about the chain worth trying: each lays a reflection line of the
    chain's projection along *y*, and then along *z*; none if it has none."""
    radii = np.linalg.norm(across, axis=1)
    angles = np.arctan2(across[:, 1], across[:, 0])
    candidates = [angles[i] for i in range(len(across)) if radii[i] > tolerance]
    for i in range(len(across)):
        for j in range(i + 1, len(across)):
            if (numbers[i] == numbers[j] and radii[i] > tolerance
                    and abs(radii[i] - radii[j]) < tolerance):
                candidates.append(0.5 * (angles[i] + angles[j]))
    lines: List[float] = []
    for angle in candidates:
        angle = float(angle % np.pi)
        if any(abs((angle - seen + np.pi / 2) % np.pi - np.pi / 2) < 1e-4 for seen in lines):
            continue
        if _reflects(numbers, across, angle, tolerance):
            lines.append(angle)
    turns = [0.0]
    for line in lines:
        turns += [-line, np.pi / 2 - line]
    return turns


def _reflects(numbers, points, angle, tolerance) -> bool:
    """Whether the projection is its own mirror image across the line at ``angle``."""
    c, s = np.cos(2 * angle), np.sin(2 * angle)
    mirrored = points @ np.array([[c, s], [s, -c]]).T
    distance = np.linalg.norm(mirrored[:, None, :] - points[None, :, :], axis=2)
    same = numbers[:, None] == numbers[None, :]
    return bool(np.all(np.any((distance < tolerance) & same, axis=1)))


def _origin(group: RodGroup, numbers, positions, a, tolerance) -> Optional[float]:
    """Where along the chain the group's origin goes, or ``None`` if it fits nowhere.

    Operations that keep the chain's direction care nothing for the origin and
    are checked first, as they are cheap; one that reverses it fixes the origin
    to one of a few places, which are then tried against all of them.
    """
    keeps = [k for k, rotation in enumerate(group.rotations) if rotation[0, 0] > 0]
    if not all(_holds(group, k, numbers, positions, a, tolerance) for k in keeps):
        return None
    reverses = [k for k in range(group.order) if k not in keeps]
    if not reverses:
        return 0.0
    k = reverses[0]
    shift = group.translations[k]
    candidates = set()
    for j in np.flatnonzero(numbers == numbers[0]):
        middle = 0.5 * (positions[j, 0] + positions[0, 0] - shift)
        candidates.update({round(middle % 1.0, 9), round((middle + 0.5) % 1.0, 9)})
    for origin in sorted(candidates):
        shifted = positions.copy()
        shifted[:, 0] = (shifted[:, 0] - origin) % 1.0
        if all(_holds(group, k, numbers, shifted, a, tolerance) for k in reverses):
            return origin
    return None


def _image(group: RodGroup, k: int, positions: np.ndarray) -> np.ndarray:
    """Where operation ``k`` sends each atom, as fractional x and Å y, z."""
    rotation = group.rotations[k]
    image = np.empty_like(positions)
    image[:, 0] = rotation[0, 0] * positions[:, 0] + group.translations[k]
    image[:, 1:] = positions[:, 1:] @ rotation[1:, 1:].T
    return image


def _matches(image, image_numbers, positions, numbers, a, tolerance) -> np.ndarray:
    """``(m, n)``: whether image ``i`` lands on atom ``j``, of the same element."""
    along = (image[:, None, 0] - positions[None, :, 0] + 0.5) % 1.0 - 0.5
    gap = np.sqrt((along * a) ** 2
                  + np.sum((image[:, None, 1:] - positions[None, :, 1:]) ** 2, axis=2))
    return (gap < tolerance) & (np.asarray(image_numbers)[:, None] == numbers[None, :])


def _holds(group, k, numbers, positions, a, tolerance) -> bool:
    hits = _matches(_image(group, k, positions), numbers, positions, numbers, a, tolerance)
    return bool(np.all(hits.any(axis=1)))


# ── the asymmetric unit, and the check that it is enough ──────────────────
def asymmetric_unit(symmetry: RodSymmetry) -> Tuple[np.ndarray, np.ndarray]:
    """``(numbers, positions)``: one atom of each orbit of the group."""
    group, positions = symmetry.group, symmetry.positions
    seen = np.zeros(len(positions), dtype=bool)
    chosen: List[int] = []
    for index in range(len(positions)):
        if seen[index]:
            continue
        chosen.append(index)
        one = positions[index:index + 1]
        for k in range(group.order):
            seen |= _matches(_image(group, k, one), symmetry.numbers[index:index + 1],
                             positions, symmetry.numbers, symmetry.a, symmetry.tolerance)[0]
    return symmetry.numbers[chosen], positions[chosen]


def regenerates(symmetry: RodSymmetry) -> bool:
    """Whether the group and its asymmetric unit rebuild the chain, atom for atom."""
    numbers, unit = asymmetric_unit(symmetry)
    group = symmetry.group
    images = np.concatenate([_image(group, k, unit) for k in range(group.order)])
    kinds = np.tile(numbers, group.order)
    images[:, 0] %= 1.0
    # Images that coincide are one atom: an atom on a mirror is its own image.
    same = _matches(images, kinds, images, kinds, symmetry.a, symmetry.tolerance)
    distinct = [i for i in range(len(images)) if not same[i, :i].any()]
    if len(distinct) != len(symmetry.positions):
        return False
    hits = _matches(images[distinct], kinds[distinct], symmetry.positions,
                    symmetry.numbers, symmetry.a, symmetry.tolerance)
    return bool(np.all(hits.any(axis=1)) and np.all(hits.any(axis=0)))


__all__ = ["RodGroup", "RodSymmetry", "TABLE", "analyse", "asymmetric_unit",
           "of_atoms", "regenerates", "rod_group"]
