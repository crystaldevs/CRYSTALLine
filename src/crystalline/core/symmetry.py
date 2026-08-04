"""Point symmetry of a structure: the axes, planes and centre that hold one
point of it fixed.

A symmetry *operator* is a matrix and a shift; what can be **drawn** is the
element it acts about — the set of points it leaves where they were. Every
element is one of the three shapes the viewer already draws for measurements:

* a centre of inversion                            → a **point**,
* a rotation axis (a rotoinversion's axis as well) → a **line**,
* a mirror plane                                   → a **surface**.

Only *point* symmetry is reported: the operators that hold one common point
fixed, drawn through it. For a molecule that is its point group, about its
centre. For a crystal it is the point symmetry at the most symmetric point of
the cell — everything drawn really does map the structure onto itself, which the
crystal class on its own would not: the "2-fold axis" of a P2₁/c crystal is a
screw axis, and turning the structure about it does *not* reproduce it. Screw
axes, glide planes and the copy of every element the lattice repeats are
therefore left out; they are what turns a cubic cell into several hundred lines,
and they are not what "the symmetry of this structure" means to look at.

Qt-free and renderer-free like the rest of ``core``: the classification and the
clipping maths are unit-tested on their own, and the renderer only turns the
result into meshes.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np

from crystalline.core.structure import Structure

# What an element looks like on screen — the same three shapes a measurement has.
AXIS, PLANE, POINT = "axis", "plane", "point"

# Two directions or positions closer than this (Å) are the same one.
_TOL = 1e-3
# A rotation this small (radians) is the identity.
_ANGLE_TOL = 1e-6
# Beyond this order a "rotation" is numerical noise rather than symmetry.
_MAX_ORDER = 12

_OVERLINE = "̄"  # combining macron: "4" + this reads as 4-bar


@dataclass(eq=False)
class SymmetryElement:
    """One drawable symmetry element, ready to list in a panel and draw in 3D.

    ``origin`` is the centre the symmetry acts about — the same point for every
    element — and ``direction`` is the axis direction (for :data:`AXIS`) or the
    plane normal (for :data:`PLANE`); a :data:`POINT` has no direction.

    ``label`` is the symbol of the defining operator, in the notation its group
    is named in: Hermann–Mauguin for a crystal ("4", "m", "1̄") and Schoenflies
    for a molecule ("C4", "σ", "i"). ``noun`` says the same thing in words.
    ``labels`` holds every symbol sharing the element, since one line on screen
    usually carries several operators — a 4-fold axis is a 2-fold axis and often
    a 4̄ axis as well.
    """

    kind: str
    label: str
    origin: np.ndarray
    noun: str = ""               # what it is, spelled out ("4-fold rotation axis")
    direction: Optional[np.ndarray] = None
    order: int = 1
    proper: bool = True          # False for mirrors, inversion and rotoinversions
    labels: Tuple[str, ...] = ()
    site: str = ""               # which way it points, in words

    def __post_init__(self) -> None:
        self.origin = np.asarray(self.origin, dtype=float)
        if self.direction is not None:
            self.direction = np.asarray(self.direction, dtype=float)
        if not self.labels:
            self.labels = (self.label,)

    def summary(self) -> str:
        """``"4 — 4-fold rotation axis ∥ [001]  (with 4̄, 2)"``: one row of the list.

        The symbol alone ("m", "1̄", "σ") is what a crystallographer reads at a
        glance and what is written beside the element in 3D, but it is no help to
        anyone who does not already know it — so the row says both.
        """
        extra = [s for s in self.labels if s != self.label]
        tail = f"  (with {', '.join(extra)})" if extra else ""
        return f"{_before_dash(self.label)}— {self.noun} {self.site}{tail}".rstrip()


@dataclass(frozen=True)
class SymmetryAnalysis:
    """What :func:`analyse` found: the elements, and where and what they are."""

    elements: List[SymmetryElement] = field(default_factory=list)
    group: Optional[str] = None       # "m3̄m", "D6h" — the group the elements form
    centre: Optional[np.ndarray] = None   # cartesian point they all pass through
    centre_site: str = ""             # that point in fractional (or Å) coordinates

    def summary(self) -> str:
        """``"Point symmetry m3̄m · 23 elements through (0, 0, 0)"`` — a status line.

        Named as *point* symmetry rather than as the group of the crystal: for
        anything but a symmorphic group in a special setting the two differ, and
        what is drawn is always the former.
        """
        if not self.elements:
            return (f"Point symmetry {self.group} — no elements" if self.group
                    else "No point symmetry")
        count = f"{len(self.elements)} element{'s' if len(self.elements) != 1 else ''}"
        parts = [f"Point symmetry {self.group}"] if self.group else []
        parts.append(f"{count} through {self.centre_site}")
        return " · ".join(parts)


# ── public entry point ────────────────────────────────────────────────────
def analyse(structure: Structure, symprec: float = 1e-2) -> SymmetryAnalysis:
    """Find the point symmetry of ``structure``, sorted for display.

    ``symprec`` is the tolerance the symmetry search works to (Å): loosen it to
    recognise a nearly-symmetric geometry, tighten it to insist on an exact one.
    An empty analysis comes back for an empty structure, and whenever the search
    itself fails — no symmetry found is a legitimate answer, not an error to
    raise at a panel that is only trying to draw something.
    """
    atoms = structure.to_ase()
    if len(atoms) == 0:
        return SymmetryAnalysis()
    try:
        operations, cell, naming = _operations(atoms, structure, symprec)
    except Exception:  # noqa: BLE001 - a symmetry search that fails shows nothing
        return SymmetryAnalysis()
    if not operations:
        return SymmetryAnalysis()

    centre, kept = _centre_and_operators(operations, cell, np.asarray(atoms.get_positions()))
    schoenflies = cell is None  # a molecule is named C2v, not mm2; so are its elements
    elements = [element for element in (_classify(r, centre, schoenflies) for r, _ in kept)
                if element is not None]
    return SymmetryAnalysis(
        elements=_merge(elements, cell),
        group=naming(kept),
        centre=centre,
        centre_site=describe_point(structure, centre),
    )


def describe_point(structure: Structure, point) -> str:
    """A point in the coordinates that structure is read in.

    Fractional for a crystal — where a centre at ``(½, ½, ½)`` says something —
    and Angstrom for a molecule, which has no cell to be a fraction of.
    """
    point = np.asarray(point, dtype=float)
    if not _is_periodic(structure):
        return "(" + ", ".join(f"{v:.2f}" for v in point) + ") Å"
    fractional = point @ np.linalg.inv(np.asarray(structure.cell, dtype=float))
    return "(" + ", ".join(_fraction(v) for v in fractional) + ")"


# ── operators ─────────────────────────────────────────────────────────────
def _operations(atoms, structure: Structure, symprec: float):
    """``(operations, cell, naming)``: the symmetry operators in **cartesian** Å.

    spglib and pymatgen both work in their own frame — fractional coordinates for
    the one, a mass-centred molecule for the other — and the drawing code wants
    neither. Converting once here is what lets a single classifier serve both.

    ``cell`` is ``None`` for a molecule, and decides both whether an operator may
    differ from another by a lattice translation and whether directions can be
    named as lattice indices. ``naming`` turns the operators that survive into
    the name of the group they form.
    """
    if _is_periodic(structure):
        cell = np.asarray(atoms.get_cell(), dtype=float)
        dataset = _dataset(atoms, symprec)
        if dataset is None:
            return [], cell, lambda _kept: None
        rotations = np.asarray(_field(dataset, "rotations"), dtype=float)
        translations = np.asarray(_field(dataset, "translations"), dtype=float)
        # Fractional (column-vector) operator to cartesian: with the lattice
        # vectors as the *rows* of ``cell``, a cartesian column is Aᵀ·x.
        basis = cell.T
        inverse = np.linalg.inv(basis)
        operations = [
            (basis @ rotation @ inverse, basis @ translation, rotation)
            for rotation, translation in zip(rotations, translations)
        ]
        return [(r, t) for r, t, _ in operations], cell, _crystal_naming(operations)

    analyzer, molecule = _point_group_analyzer(atoms, symprec)
    # PointGroupAnalyzer works on the mass-centred molecule, so its operators are
    # about the origin; put them back on the molecule as it is drawn.
    centre = np.asarray(molecule.center_of_mass, dtype=float)
    operations = []
    for operation in analyzer.get_symmetry_operations():
        rotation = np.asarray(operation.rotation_matrix, dtype=float)
        operations.append((rotation, centre - rotation @ centre))
    # A molecule's point group is exactly what pymatgen just named it.
    return operations, None, lambda _kept: analyzer.sch_symbol


def _crystal_naming(operations):
    """A namer for the subset of crystal operators that survives the centre.

    The name is the Hermann–Mauguin symbol of the group those operators form —
    which is the crystal class only when the whole group holds the point fixed,
    and the point symmetry at that site otherwise (``4̄3m`` at the atom of a
    diamond cell, whose class is ``m-3m``).
    """
    def naming(kept) -> Optional[str]:
        import spglib

        cartesian = {id(r): fractional for r, _t, fractional in operations}
        rotations = [cartesian[id(rotation)] for rotation, _ in kept]
        try:
            symbol = spglib.get_pointgroup(np.asarray(rotations, dtype="intc").round())[0]
            return _overlined(str(symbol).strip()) or None
        except Exception:  # noqa: BLE001 - naming the group is a nicety, not the job
            return None

    return naming


def _is_periodic(structure: Structure) -> bool:
    """Whether the structure has a usable lattice to find crystal operators in."""
    if not structure.is_periodic:
        return False
    cell = np.asarray(structure.cell, dtype=float)
    return cell.shape == (3, 3) and abs(np.linalg.det(cell)) > 1e-8


def _dataset(atoms, symprec: float):
    """spglib's analysis of the cell: every symmetry operator it has."""
    import spglib

    return spglib.get_symmetry_dataset(
        (
            np.asarray(atoms.get_cell(), dtype=float),
            np.asarray(atoms.get_scaled_positions(), dtype=float),
            np.asarray(atoms.get_atomic_numbers(), dtype=int),
        ),
        symprec=symprec,
    )


def _field(dataset, name: str):
    """Read a field of an spglib dataset, which is an object in 2.5+ and a dict before."""
    return dataset[name] if isinstance(dataset, dict) else getattr(dataset, name)


def _point_group_analyzer(atoms, symprec: float):
    """``(analyzer, molecule)`` for a non-periodic system.

    The molecule comes back too because the analyzer's operators are about *its*
    centre of mass, which is what puts them back on the atoms as drawn.
    """
    from pymatgen.core import Molecule
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer

    molecule = Molecule(
        [str(s) for s in atoms.get_chemical_symbols()],
        np.asarray(atoms.get_positions(), dtype=float),
    )
    # The point-group search is a geometric match over the whole molecule, not a
    # per-atom displacement test: spglib's default 0.01 Å is far too tight for it
    # (pymatgen's own default is 0.3), and would report C1 for everything.
    return PointGroupAnalyzer(molecule, tolerance=max(symprec, 0.1)), molecule


# ── which point the symmetry is drawn about ───────────────────────────────
def _centre_and_operators(operations, cell, positions: np.ndarray):
    """``(centre, operators)``: the most symmetric point, and what holds it fixed.

    An operator ``x → Wx + w`` holds a point fixed where ``(W − I)x = −w`` — a
    line for a rotation, a plane for a mirror, a single point for an inversion,
    and nowhere at all for a screw or a glide. The point to draw the symmetry
    about is the one the most operators hold fixed at once, so the search offers
    each operator's own fixed point (and, in a crystal, the same operator moved
    by each neighbouring lattice translation, which fixes a different point) as a
    candidate and keeps the best-scoring one.

    That is the honest answer for a crystal, where the *class* usually claims
    more symmetry than any one point of the structure actually has.
    """
    matrices = np.asarray([np.asarray(r, dtype=float) for r, _ in operations])
    shifts = np.asarray([np.asarray(w, dtype=float) for _, w in operations])
    inverse = None if cell is None else np.linalg.inv(np.asarray(cell, dtype=float))

    best_centre, best_score = None, -1
    centroid = positions.mean(axis=0) if len(positions) else np.zeros(3)
    for candidate in _candidates(matrices, shifts, cell, inverse):
        fixed = _fixed_by(candidate, matrices, shifts, inverse)
        score = int(fixed.sum())
        # Among equally symmetric points, the one nearest the structure: an
        # element drawn through the middle of what it acts on reads best.
        if score > best_score or (
            score == best_score
            and np.linalg.norm(candidate - centroid) < np.linalg.norm(best_centre - centroid)
        ):
            best_centre, best_score = candidate, score

    fixed = _fixed_by(best_centre, matrices, shifts, inverse)
    kept = [operations[i] for i in np.flatnonzero(fixed)]
    return _refine(best_centre, kept, centroid, cell, inverse), kept


def _candidates(matrices, shifts, cell, inverse) -> List[np.ndarray]:
    """Points worth testing: what each operator holds fixed, cell by cell.

    Taking the least-norm fixed point of an operator gives a point *on* its axis
    or *in* its plane, which is all that is needed — the scoring finds the ones
    that several operators share. In a crystal the same operator shifted by a
    lattice vector fixes a different point (the inversion centres at the corner
    and at the middle of a cell edge are both real), so each neighbouring
    translation is offered as well.
    """
    translations = (
        [np.zeros(3)] if cell is None
        else [offsets @ cell for offsets in itertools.product((0, 1), repeat=3)]
    )
    seen: set = set()
    candidates: List[np.ndarray] = []
    for matrix, shift in zip(matrices, shifts):
        fixed = np.linalg.pinv(matrix - np.eye(3))
        for translation in translations:
            point = fixed @ -(shift + translation)
            if inverse is not None:  # keep it in the cell, where the structure is
                fractional = point @ inverse
                point = point - np.floor(fractional + _TOL) @ cell
            key = tuple(np.round(point, 3) + 0.0)
            if key not in seen:
                seen.add(key)
                candidates.append(point)
    return candidates


def _fixed_by(point: np.ndarray, matrices, shifts, inverse) -> np.ndarray:
    """Which operators leave ``point`` where it is (as a boolean mask).

    In a crystal an operator that returns the point to a *periodic image* of
    itself leaves it fixed just as well: the image is the same point of the
    structure. That is why the residual is measured modulo the lattice.
    """
    residuals = np.einsum("nij,j->ni", matrices, point) + shifts - point
    if inverse is None:
        return np.linalg.norm(residuals, axis=1) < _TOL
    fractional = residuals @ inverse
    return np.all(np.abs(fractional - np.round(fractional)) < 1e-4, axis=1)


def _refine(centre, kept, centroid: np.ndarray, cell, inverse) -> np.ndarray:
    """Slide the centre along whatever the kept operators leave free.

    A single mirror plane holds every one of its own points fixed, and a lone
    rotation axis every point along it — so "the centre" is only pinned down as
    far as the symmetry pins it. Moving it to the point of that line or plane
    nearest the structure keeps the elements drawn through the structure instead
    of wherever the arithmetic happened to land.
    """
    if not kept:
        return centre
    matrices = np.vstack([np.asarray(r, dtype=float) - np.eye(3) for r, _ in kept])
    # Each operator's shift, taken with the lattice translation that made it fix
    # the centre in the first place — otherwise the solve pulls against it.
    offsets = []
    for rotation, shift in kept:
        residual = rotation @ centre + shift - centre
        lattice = np.zeros(3) if inverse is None else np.round(residual @ inverse) @ cell
        offsets.append(-(shift - lattice))
    solution, *_ = np.linalg.lstsq(
        matrices, np.concatenate(offsets) - matrices @ centroid, rcond=None
    )
    return centroid + solution


# ── classification ────────────────────────────────────────────────────────
def _classify(
    rotation: np.ndarray, centre: np.ndarray, schoenflies: bool
) -> Optional[SymmetryElement]:
    """The element one rotation matrix acts about, or ``None`` if it draws nothing.

    ``None`` is the identity, which holds everything still and so marks out no
    axis, plane or point in particular.

    An improper operator is ``-1`` times a proper rotation, and the order of that
    rotation says which kind it is: 1 gives the centre of inversion, 2 a plane (a
    half-turn followed by inversion is a reflection through the plane it turns
    about), and 3, 4 or 6 a rotoinversion axis.

    ``schoenflies`` picks the notation: C₂/σ/i/S₄ for a molecule, whose group is
    named that way too, and Hermann–Mauguin 2/m/1̄/4̄ for a crystal, whose is.
    """
    rotation = np.asarray(rotation, dtype=float)
    proper = np.linalg.det(rotation) > 0
    order, direction = _order_and_axis(rotation if proper else -rotation)

    if proper:
        if order <= 1:
            return None  # the identity
        return SymmetryElement(
            AXIS, f"C{_subscript(order)}" if schoenflies else str(order),
            centre, f"{order}-fold rotation axis", direction, order, True,
        )
    if order <= 1:
        return SymmetryElement(
            POINT, "i" if schoenflies else "1" + _OVERLINE,
            centre, "centre of inversion", None, 1, False,
        )
    if order == 2:
        return SymmetryElement(
            PLANE, "σ" if schoenflies else "m", centre, "mirror plane", direction, 2, False,
        )
    if schoenflies:
        # The same axis, named for the reflection it is built from rather than
        # the inversion: 3̄ is S₆, 4̄ is S₄, 6̄ is S₃.
        reflection = _rotoreflection_order(order)
        label = f"S{_subscript(reflection)}"
        noun = f"{reflection}-fold rotoreflection axis"
    else:
        label, noun = f"{order}{_OVERLINE}", f"{order}-fold rotoinversion axis"
    return SymmetryElement(AXIS, label, centre, noun, direction, order, False)


_SUBSCRIPTS = "₀₁₂₃₄₅₆₇₈₉"


def _subscript(number: int) -> str:
    """``2`` → ``"₂"``: the order of a Schoenflies symbol is written under it."""
    return "".join(_SUBSCRIPTS[int(digit)] for digit in str(int(number)))


def _rotoreflection_order(order: int) -> int:
    """The Schoenflies ``Sₙ`` index of the Hermann–Mauguin rotoinversion ``n̄``.

    Rotoinversion and rotoreflection are the same set of operations described
    from different ends, but their indices only agree for a quarter of the cases:
    3̄ = S₆, 4̄ = S₄, 6̄ = S₃, 5̄ = S₁₀.
    """
    if order % 2 == 1:
        return 2 * order
    return order if order % 4 == 0 else order // 2


def _order_and_axis(rotation: np.ndarray) -> Tuple[int, Optional[np.ndarray]]:
    """``(n, u)`` for a proper rotation: its order and its axis (``(1, None)`` for
    the identity). The axis is given a fixed sign — a drawn line has no sense of
    rotation, and ±u must not read as two different elements."""
    rotation = np.asarray(rotation, dtype=float)
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < _ANGLE_TOL:
        return 1, None
    order = int(round(2.0 * np.pi / angle))
    if order < 2 or order > _MAX_ORDER:
        return 1, None

    axis = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ])
    if np.linalg.norm(axis) < 1e-8:  # a half-turn: the antisymmetric part vanishes
        axis = _eigenvector(rotation, 1.0)
        if axis is None:
            return 1, None
    return order, _canonical(axis / np.linalg.norm(axis))


def _eigenvector(matrix: np.ndarray, eigenvalue: float) -> Optional[np.ndarray]:
    """The real unit eigenvector of ``matrix`` for ``eigenvalue`` (``None`` if none)."""
    values, vectors = np.linalg.eig(matrix)
    index = int(np.argmin(np.abs(values - eigenvalue)))
    if abs(values[index] - eigenvalue) > 1e-6:
        return None
    vector = np.real(vectors[:, index])
    norm = float(np.linalg.norm(vector))
    return None if norm < 1e-8 else vector / norm


def _canonical(vector: np.ndarray) -> np.ndarray:
    """``vector`` with a fixed sign, so ±u are recognised as the same line."""
    vector = np.asarray(vector, dtype=float)
    rounded = np.round(vector, 6)
    index = int(np.argmax(np.abs(rounded)))
    return -vector if rounded[index] < 0 else vector


# ── merging coincident operators ──────────────────────────────────────────
def _merge(elements: Sequence[SymmetryElement], cell) -> List[SymmetryElement]:
    """Collapse operators sharing a line (or a plane) into one element, and sort.

    A 4-fold axis, the 2-fold rotation it contains and the 4̄ about the same line
    are one line on screen; listing them separately would put three identical
    rows in the panel and draw the same tube three times. The highest-order,
    proper operator names the merged element and the rest ride along in
    ``labels``.
    """
    merged: dict = {}
    for element in elements:
        key = ((element.kind, *np.round(element.direction, 3) + 0.0)
               if element.direction is not None else (element.kind,))
        current = merged.get(key)
        if current is None:
            merged[key] = element
        elif (element.order, element.proper) > (current.order, current.proper):
            merged[key] = replace(element, labels=_labels(element, current))
        else:
            merged[key] = replace(current, labels=_labels(current, element))

    ordered = sorted(merged.values(), key=_sort_key)
    for element in ordered:
        element.site = _site_text(element, cell)
    return ordered


def _labels(primary: SymmetryElement, other: SymmetryElement) -> Tuple[str, ...]:
    """``primary``'s labels plus ``other``'s, without repeats and in order."""
    seen = list(primary.labels) if primary.labels else [primary.label]
    for label in (other.labels or (other.label,)):
        if label not in seen:
            seen.append(label)
    return tuple(seen)


def _sort_key(element: SymmetryElement) -> tuple:
    """Axes first, then planes, then the centre; each by descending order."""
    kind_order = {AXIS: 0, PLANE: 1, POINT: 2}[element.kind]
    direction = element.direction if element.direction is not None else np.zeros(3)
    return (kind_order, -element.order, *np.round(direction, 3))


# ── describing which way an element points ────────────────────────────────
def _site_text(element: SymmetryElement, cell) -> str:
    """``"∥ [001]"`` — which way the element faces.

    Lattice indices for a periodic structure (what a crystallographer reads a
    direction as), plain cartesian components for a molecule, which has no cell
    for an index to refer to. Where the element *is* needs no saying: they all
    pass through the one centre.
    """
    if element.kind == POINT:
        return ""
    if cell is None:
        # ``+ 0.0`` so a component that came out as -0.0 does not read as "-0.00".
        vector = "[" + ", ".join(f"{v:.2f}" for v in np.round(element.direction, 2) + 0.0) + "]"
        return ("∥ " if element.kind == AXIS else "⊥ ") + vector
    indices = _integer_indices(element.direction @ np.linalg.inv(np.asarray(cell, dtype=float)))
    # A plane is named by the lattice direction of its normal, in round brackets —
    # the (hkl) of the crystallographic plane it is parallel to.
    return f"⊥ ({indices})" if element.kind == PLANE else f"∥ [{indices}]"


def _integer_indices(direction: np.ndarray, limit: int = 12) -> str:
    """``"001"`` / ``"1 1̄ 0"``: the direction as the smallest whole lattice indices.

    Falls back to two decimals when the direction is not a rational one (possible
    for a nearly-symmetric cell at a loose tolerance).
    """
    direction = np.asarray(direction, dtype=float)
    magnitudes = np.abs(direction)
    smallest = magnitudes[magnitudes > 1e-6]
    if smallest.size == 0:
        return "0 0 0"
    scaled = direction / smallest.min()
    for factor in range(1, limit + 1):
        candidate = scaled * factor
        if np.all(np.abs(candidate - np.round(candidate)) < 1e-3):
            values = np.round(candidate).astype(int)
            divisor = np.gcd.reduce(np.abs(values[values != 0]))
            values = values // max(int(divisor), 1)
            if np.all(np.abs(values) < 10):
                return "".join(_index_text(v) for v in values)
            return " ".join(_index_text(v) for v in values)
    return " ".join(f"{v:.2f}" for v in direction)


def _index_text(value: int) -> str:
    """A lattice index, with the crystallographic overbar for a negative one."""
    return f"{abs(int(value))}{_OVERLINE}" if value < 0 else str(int(value))


def _overlined(symbol: str) -> str:
    """``"m-3m"`` → ``"m3̄m"``: spglib spells a bar as a leading minus, ASCII-only."""
    return re.sub(r"-(\d)", lambda match: match.group(1) + _OVERLINE, symbol)


_THIN_SPACE = "\u2009"  # invisible in source, so spelled as an escape


def _before_dash(label: str) -> str:
    """``label`` plus the space that separates it from its description.

    A combining overbar has no width of its own and is drawn over whatever
    follows, which for "1̄ — centre" is the space: the dash ends up touching the
    symbol. A thin space on top of the ordinary one keeps them apart, and is
    added only where a bar would otherwise close the gap.
    """
    return label + (_THIN_SPACE + " " if label.endswith(_OVERLINE) else " ")


# Fractions a symmetry centre actually lands on, as characters rather than
# decimals: "¼" reads as an exact site where "0.250" reads as a measurement.
_FRACTIONS = ((0.0, "0"), (0.125, "⅛"), (0.25, "¼"), (1 / 3, "⅓"), (0.375, "⅜"), (0.5, "½"),
              (0.625, "⅝"), (2 / 3, "⅔"), (0.75, "¾"), (0.875, "⅞"), (1.0, "0"))


def _fraction(value: float, tol: float = 5e-3) -> str:
    value = float(value - np.floor(value + tol))  # a centre is only defined per cell
    for exact, text in _FRACTIONS:
        if abs(value - exact) < tol:
            return text
    return f"{value:.3f}"


# ── geometry: where an element meets the drawn box ────────────────────────
def segment_in_box(origin, direction, bounds) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Clip the infinite line ``origin + t·direction`` to an axis-aligned box.

    ``bounds`` is ``(xmin, xmax, ymin, ymax, zmin, zmax)`` — pyvista's order.
    Returns the two end points of the surviving segment, or ``None`` when the
    line misses the box.
    """
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    lo = np.asarray(bounds, dtype=float)[0::2]
    hi = np.asarray(bounds, dtype=float)[1::2]
    near, far = -np.inf, np.inf
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:  # parallel to this pair of faces
            if origin[axis] < lo[axis] - _TOL or origin[axis] > hi[axis] + _TOL:
                return None
            continue
        first = (lo[axis] - origin[axis]) / direction[axis]
        second = (hi[axis] - origin[axis]) / direction[axis]
        near = max(near, min(first, second))
        far = min(far, max(first, second))
    if not np.isfinite(near) or not np.isfinite(far) or far - near <= _TOL:
        return None
    return origin + direction * near, origin + direction * far


__all__ = [
    "SymmetryElement",
    "SymmetryAnalysis",
    "AXIS",
    "PLANE",
    "POINT",
    "analyse",
    "describe_point",
    "segment_in_box",
]
