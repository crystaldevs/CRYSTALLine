"""Point symmetry: what holds a point of the structure fixed, and where it is.

The central test here is :func:`_maps_onto_itself`: every element reported must
be an operation that really does reproduce the structure. That is the whole
claim the panel makes, and it is what separates point symmetry from the crystal
*class* — a P2₁/c crystal is class 2/m, but turning it about that axis does not
reproduce it (the axis is a screw), so no 2-fold may be reported.
"""

import numpy as np
import pytest

pytest.importorskip("spglib")
pytest.importorskip("ase")

from ase import Atoms  # noqa: E402
from ase.build import bulk  # noqa: E402
from ase.spacegroup import crystal  # noqa: E402

from crystalline.core import symmetry as S  # noqa: E402
from crystalline.core.structure import Structure  # noqa: E402


# ── is a reported element really a symmetry of the structure? ─────────────
def _operation_matrix(element) -> np.ndarray:
    """The cartesian matrix of the operation ``element`` stands for."""
    if element.kind == S.POINT:
        return -np.eye(3)
    axis = element.direction / np.linalg.norm(element.direction)
    if element.kind == S.PLANE:
        return np.eye(3) - 2.0 * np.outer(axis, axis)
    angle = 2 * np.pi / element.order
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rotation = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)
    return rotation if element.proper else -rotation


def _maps_onto_itself(structure: Structure, element, tol: float = 1e-4) -> bool:
    """Whether applying ``element`` about its centre reproduces the structure.

    Every atom has to land on an atom of the same element — on a periodic image
    of one, for a crystal, which is the same atom as far as the structure is
    concerned.
    """
    atoms = structure.to_ase()
    positions, numbers = atoms.get_positions(), atoms.get_atomic_numbers()
    matrix = _operation_matrix(element)
    moved = (positions - element.origin) @ matrix.T + element.origin

    if structure.is_periodic:
        inverse = np.linalg.inv(np.asarray(atoms.get_cell(), dtype=float))
        moved, positions = (moved @ inverse) % 1.0, (positions @ inverse) % 1.0
    for index in range(len(moved)):
        offsets = moved[index] - positions
        if structure.is_periodic:
            offsets = (offsets + 0.5) % 1.0 - 0.5
        hits = (np.linalg.norm(offsets, axis=1) < tol) & (numbers == numbers[index])
        if not hits.any():
            return False
    return True


def _sites(analysis, kind=None):
    return sorted(e.site for e in analysis.elements if kind is None or e.kind == kind)


# ── crystals ──────────────────────────────────────────────────────────────
_CRYSTALS = {
    "P2_1/c": crystal(["C", "O"], [(0.1, 0.2, 0.3), (0.3, 0.1, 0.05)], spacegroup=14,
                      cellpar=[5, 6, 7, 90, 100, 90]),
    "Pnma": crystal("Si", [(0.1, 0.25, 0.2)], spacegroup=62, cellpar=[7, 5, 8, 90, 90, 90]),
    "Pm-3m": crystal(["Sr", "Ti", "O"], [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
                     spacegroup=221, cellpar=[3.9] * 3 + [90] * 3),
    "Fm-3m": bulk("NaCl", "rocksalt", a=5.64, cubic=True),
    "Fd-3m": crystal(["Si"], [(0.0, 0.0, 0.0)], spacegroup=227, cellpar=[5.43] * 3 + [90] * 3),
    "P6_3/mmc": crystal(["Mg"], [(1 / 3, 2 / 3, 0.25)], spacegroup=194,
                        cellpar=[3.21, 3.21, 5.21, 90, 90, 120]),
}


@pytest.mark.parametrize("name", sorted(_CRYSTALS))
def test_every_reported_element_really_is_a_symmetry(name):
    structure = Structure.from_ase(_CRYSTALS[name])
    analysis = S.analyse(structure)

    assert analysis.elements, f"{name} has symmetry to report"
    for element in analysis.elements:
        assert _maps_onto_itself(structure, element), f"{element.summary()} does not hold"


def test_molecular_elements_really_are_symmetries():
    pytest.importorskip("pymatgen")
    for molecule in (_water(), _ammonia()):
        analysis = S.analyse(molecule)
        assert analysis.elements
        for element in analysis.elements:
            assert _maps_onto_itself(molecule, element, tol=1e-3), element.summary()


def test_a_screw_axis_is_not_reported_as_a_rotation():
    """P2₁/c is crystal class 2/m, but its 2-fold is a screw and its mirror a
    glide: neither holds any point of the structure, so only the centre remains."""
    analysis = S.analyse(Structure.from_ase(_CRYSTALS["P2_1/c"]))

    assert [e.kind for e in analysis.elements] == [S.POINT]
    assert analysis.group == "1̄"


def test_a_cubic_perovskite_keeps_its_whole_class():
    """Nothing is lost when the group is symmorphic: every operator holds the
    special position fixed, so all 23 elements of m-3m are there."""
    analysis = S.analyse(Structure.from_ase(_CRYSTALS["Pm-3m"]))

    assert analysis.group == "m3̄m"
    assert len(analysis.elements) == 23  # 13 axes + 9 mirrors + the centre
    assert len([e for e in analysis.elements if e.order == 4]) == 3
    assert len([e for e in analysis.elements if e.order == 3]) == 4
    assert len([e for e in analysis.elements if e.kind == S.PLANE]) == 9
    assert len([e for e in analysis.elements if e.kind == S.POINT]) == 1


def test_diamond_reports_the_site_symmetry_not_the_class():
    """Fd-3m is class m-3m, but no point of a diamond cell has all of it: the
    atom sites are 4̄3m, which has no inversion centre."""
    analysis = S.analyse(Structure.from_ase(_CRYSTALS["Fd-3m"]))

    assert analysis.group == "4̄3m"
    assert not [e for e in analysis.elements if e.kind == S.POINT]
    assert {e.label for e in analysis.elements if e.kind == S.AXIS} == {"C₃", "S₄"}


def test_coincident_operators_merge_into_one_element():
    """A 4-fold axis, the 2-fold it contains and the S₄ about it are one line."""
    analysis = S.analyse(Structure.from_ase(_CRYSTALS["Fm-3m"]))
    fourfold = [e for e in analysis.elements if e.label == "C₄"]

    assert len(fourfold) == 3  # a cubic cell has three, not one per operator
    for element in fourfold:
        assert set(element.labels) == {"C₄", "C₂", "S₄"}


def test_axes_and_planes_are_named_by_lattice_direction():
    analysis = S.analyse(Structure.from_ase(_CRYSTALS["Pm-3m"]))

    assert "∥ [001]" in _sites(analysis, S.AXIS)
    assert "∥ [111]" in _sites(analysis, S.AXIS)
    assert "⊥ (110)" in _sites(analysis, S.PLANE)
    assert _sites(analysis, S.POINT) == [""]  # a centre points nowhere


def test_the_centre_is_a_point_of_the_cell_every_element_passes_through():
    analysis = S.analyse(Structure.from_ase(_CRYSTALS["Fm-3m"]))

    assert analysis.centre_site == "(0, 0, 0)"
    for element in analysis.elements:
        assert np.allclose(element.origin, analysis.centre)


def test_the_summary_names_the_group_and_where_it_sits():
    analysis = S.analyse(Structure.from_ase(_CRYSTALS["Fm-3m"]))
    assert analysis.summary() == "Point symmetry m3̄m · 23 elements through (0, 0, 0)"


# ── molecules ─────────────────────────────────────────────────────────────
def _water() -> Structure:
    # From ase's own geometries, which are exactly symmetric — a hand-typed one
    # is symmetric only to the decimals it was typed to, and the tolerance this
    # is checked at is tighter than that.
    from ase.build import molecule

    return Structure.from_ase(molecule("H2O"))


def _ammonia() -> Structure:
    from ase.build import molecule

    return Structure.from_ase(molecule("NH3"))


def test_a_molecule_gets_its_point_group_about_its_own_centre():
    pytest.importorskip("pymatgen")
    analysis = S.analyse(_water())

    assert analysis.group == "C2v"
    assert len(analysis.elements) == 3  # the 2-fold axis and the two mirror planes
    assert {e.label for e in analysis.elements} == {"C₂", "σᵥ"}  # both planes hold the axis
    assert analysis.centre_site.endswith("Å")  # no cell, so no fractions
    # The centre sits on the molecule, not at the world origin it happens to
    # be drawn near.
    assert abs(float(analysis.centre[2])) < 1.0


def test_a_molecule_is_labelled_in_schoenflies_throughout():
    """Benzene's principal axis carries C₆ and everything it contains, named the
    way its group (D6h) is — a rotoinversion 3̄ read as the rotoreflection S₆."""
    pytest.importorskip("pymatgen")
    from ase.build import molecule

    analysis = S.analyse(Structure.from_ase(molecule("C6H6")))
    principal = next(e for e in analysis.elements if e.order == 6 and e.proper)

    assert principal.label == "C₆"
    assert set(principal.labels) == {"C₆", "C₃", "C₂", "S₃", "S₆"}
    assert principal.noun == "6-fold rotation axis"
    assert {e.label for e in analysis.elements if e.kind == S.PLANE} == {"σᵥ", "σₕ", "σd"}
    assert [e.label for e in analysis.elements if e.kind == S.POINT] == ["i"]


def test_a_rotoreflection_is_named_and_described_by_its_own_order():
    """Methane's 4̄ axes are S₄, and there is no centre of inversion in Td."""
    pytest.importorskip("pymatgen")
    from ase.build import molecule

    analysis = S.analyse(Structure.from_ase(molecule("CH4")))
    improper = [e for e in analysis.elements if e.kind == S.AXIS and not e.proper]

    assert len(improper) == 3
    for element in improper:
        assert element.label == "S₄"
        assert element.noun == "4-fold rotoreflection axis"
    assert not [e for e in analysis.elements if e.kind == S.POINT]


def test_a_crystal_keeps_hermann_mauguin_with_a_real_overbar():
    """spglib spells a bar as a leading minus ("m-3m"); crystallography puts it
    over the digit. The group keeps that spelling; its operators do not — see
    the Schoenflies tests below."""
    analysis = S.analyse(Structure.from_ase(_CRYSTALS["Fm-3m"]))

    assert "-" not in analysis.group
    assert analysis.group == "m3" + "̄" + "m"


def test_the_symbol_stays_clear_of_its_description():
    """A combining overbar is drawn over the character after it, so a label that
    ends in one needs more than a single space before the dash."""
    # No Schoenflies symbol ends in a bar, so the spacing is checked on a
    # Hermann-Mauguin label built for the purpose.
    barred = S.SymmetryElement(S.POINT, "1" + "̄", [0, 0, 0], "centre of inversion")
    assert barred.summary().startswith("1" + "̄" + "\u2009 — centre of inversion")

    analysis = S.analyse(Structure.from_ase(_CRYSTALS["Fm-3m"]))
    axis = next(e for e in analysis.elements if e.label == "C₄")
    assert axis.summary().startswith("C₄ — 4-fold rotation axis")  # no bar, one space


def test_a_molecules_directions_are_cartesian_not_lattice_indices():
    pytest.importorskip("pymatgen")
    analysis = S.analyse(_ammonia())
    assert analysis.group == "C3v"
    assert any(e.site.startswith("∥ [0.00, 0.00, 1.00]") for e in analysis.elements)


# ── edge cases ────────────────────────────────────────────────────────────
def test_a_structure_with_no_symmetry_yields_no_elements():
    atoms = Atoms("HHeLi", positions=[[0, 0, 0], [1.1, 0.3, 0.2], [0.4, 1.7, 0.9]],
                  cell=np.eye(3) * 8.0, pbc=True)
    assert S.analyse(Structure.from_ase(atoms)).elements == []


def test_empty_structure_is_analysed_without_raising():
    analysis = S.analyse(Structure.empty())
    assert analysis.elements == []
    assert analysis.group is None
    assert analysis.summary() == "No point symmetry"


def test_tolerance_recognises_a_slightly_distorted_cell():
    """A cell nudged off its symmetric geometry loses elements at a tight
    tolerance and regains them at a loose one."""
    atoms = crystal("Si", [(0.0, 0.0, 0.0)], spacegroup=221, cellpar=[4.0] * 3 + [90] * 3)
    atoms.set_cell([4.0, 4.0, 4.05, 90, 90, 90], scale_atoms=True)  # cubic → tetragonal
    structure = Structure.from_ase(atoms)

    tight = S.analyse(structure, symprec=1e-3)
    loose = S.analyse(structure, symprec=0.2)

    assert len(tight.elements) < len(loose.elements)
    assert not [e for e in tight.elements if e.order == 3]  # the body diagonals go first
    assert [e for e in loose.elements if e.order == 3]


def test_a_slab_keeps_the_symmetry_its_vacuum_axis_does_not_break():
    """CRYSTAL writes a slab with a formal 500 Å cell edge; the mirror in the
    plane of the slab is still there to find."""
    slab = Atoms("Al2", positions=[[0.0, 0.0, 0.0], [1.4, 1.4, 0.0]],
                 cell=[[2.8, 0, 0], [0, 2.8, 0], [0, 0, 500.0]], pbc=[True, True, False])
    analysis = S.analyse(Structure.from_ase(slab))

    assert analysis.elements
    for element in analysis.elements:
        assert _maps_onto_itself(Structure.from_ase(slab), element)


# ── clipping to the drawn box ─────────────────────────────────────────────
_BOX = (0.0, 2.0, 0.0, 2.0, 0.0, 2.0)


def test_segment_in_box_clips_a_line_to_the_faces_it_crosses():
    start, end = S.segment_in_box([1.0, 1.0, -5.0], [0.0, 0.0, 1.0], _BOX)
    assert np.allclose(start, [1.0, 1.0, 0.0])
    assert np.allclose(end, [1.0, 1.0, 2.0])


def test_segment_in_box_misses_a_line_outside_it():
    assert S.segment_in_box([9.0, 9.0, 0.0], [0.0, 0.0, 1.0], _BOX) is None




# ── the chemist's notation ────────────────────────────────────────────────
def test_operators_are_named_the_way_a_chemist_writes_them():
    """Not 2, m, 1̄ and 4̄: the operators of a crystal are C₂, σ, i and S₄, as
    they are for a molecule. The group keeps its own convention — a crystal is
    still m3̄m — but nobody reads an operator in two notations at once."""
    analysis = S.analyse(Structure.from_ase(_CRYSTALS["Fm-3m"]))
    labels = {e.label for e in analysis.elements}

    assert labels == {"C₄", "C₃", "C₂", "σₕ", "σd", "i"}
    assert not labels & {"2", "3", "4", "m", "1" + "̄", "4" + "̄"}


@pytest.mark.parametrize("name, expected", [
    # Textbook contents of each group, mirrors included.
    ("H2O", {"C₂": 1, "σᵥ": 2}),
    ("NH3", {"C₃": 1, "σᵥ": 3}),
    ("C6H6", {"C₆": 1, "C₂": 6, "σᵥ": 3, "σd": 3, "σₕ": 1, "i": 1}),
    ("C2H6", {"C₃": 1, "C₂": 3, "σd": 3, "i": 1}),
    ("CH4", {"S₄": 3, "C₃": 4, "σd": 6}),
])
def test_the_mirrors_are_told_apart(name, expected):
    """σₕ across the principal axis, σᵥ holding it, σd bisecting the two-fold
    axes across it. In D₆ₕ every vertical plane holds a two-fold axis, so the
    ones through atoms are the σᵥ and the ones between them the σd."""
    pytest.importorskip("pymatgen")
    from ase.build import molecule

    analysis = S.analyse(Structure.from_ase(molecule(name)))
    counted = {}
    for element in analysis.elements:
        counted[element.label] = counted.get(element.label, 0) + 1

    assert counted == expected


def test_a_group_with_no_principal_axis_leaves_its_mirrors_unnamed():
    """Ethylene's three two-fold axes are equivalent, so no plane is the
    horizontal one. A name would be a guess, and σ is the honest answer."""
    pytest.importorskip("pymatgen")
    from ase.build import molecule

    analysis = S.analyse(Structure.from_ase(molecule("C2H4")))

    assert {e.label for e in analysis.elements if e.kind == S.PLANE} == {"σ"}


def test_every_label_that_can_carry_a_subscript_does():
    """C₂, S₄, σᵥ, σₕ — Unicode has no subscript d, so σd is written flat."""
    pytest.importorskip("pymatgen")
    from ase.build import molecule

    for name in ("C6H6", "CH4", "H2O"):
        for element in S.analyse(Structure.from_ase(molecule(name))).elements:
            assert not any(ch in "0123456789" for ch in element.label), element.label
            assert "_" not in element.label, element.label


def test_a_label_becomes_html_where_unicode_runs_out():
    """Unicode has a subscript for every digit, and for h and v, so the labels
    themselves are plain strings. It has none for d, so σd is the one symbol
    that needs markup — and a widget that draws rich text gets all of them that
    way, or the rows look uneven."""
    assert S.rich("σd ⊥ (101)") == "σ<sub>d</sub> ⊥ (101)"
    assert S.rich("C₄ ∥ [010]") == "C<sub>4</sub> ∥ [010]"
    assert S.rich("σₕ") == "σ<sub>h</sub>"
    assert S.rich("σᵥ") == "σ<sub>v</sub>"
    assert S.rich("S₆") == "S<sub>6</sub>"
    assert S.rich("i") == "i"


def test_markup_in_a_label_cannot_reach_the_widget_as_markup():
    """The text is drawn as HTML, so anything that looks like a tag has to be
    escaped on the way in."""
    assert S.rich("<b>2") == "&lt;b&gt;2"
