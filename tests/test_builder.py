"""Building a structure from nothing: periodicity, lattice, space group, atoms."""

import numpy as np
import pytest

from crystalline.core import builder as b


def _spglib_number(structure):
    import spglib

    atoms = structure.to_ase()
    dataset = spglib.get_symmetry_dataset(
        (atoms.cell, atoms.get_scaled_positions(), atoms.numbers), symprec=1e-3)
    return dataset.number


def test_rock_salt_from_its_asymmetric_unit():
    """Two atoms and Fm3̄m make the eight of the cubic cell."""
    built = b.build(b.BuildSpec(b.CRYSTAL, "cubic", {"a": 4.21}, 225,
                                (("Mg", (0, 0, 0)), ("O", (0.5, 0.5, 0.5)))))
    atoms = built.to_ase()
    assert len(atoms) == 8 and atoms.get_chemical_formula() == "Mg4O4"
    assert np.allclose(atoms.cell.cellpar(), [4.21, 4.21, 4.21, 90, 90, 90])
    assert _spglib_number(built) == 225


def test_a_hexagonal_crystal_takes_gamma_from_its_system():
    built = b.build(b.BuildSpec(b.CRYSTAL, "hexagonal", {"a": 3.25, "c": 5.2}, 186,
                                (("Zn", (1 / 3, 2 / 3, 0)), ("O", (1 / 3, 2 / 3, 0.38)))))
    assert np.allclose(built.to_ase().cell.cellpar(), [3.25, 3.25, 5.2, 90, 90, 120])
    assert _spglib_number(built) == 186


def test_a_lattice_the_group_cannot_have_is_refused_with_why():
    with pytest.raises(b.BuildError, match="186"):
        b.build(b.BuildSpec(b.CRYSTAL, "cubic", {"a": 4.0}, 186, (("C", (0, 0, 0)),)))


def test_a_slab_lies_in_xy_with_cristal_s_vacuum_along_z():
    """Graphene: x and y fractional, z in Å."""
    built = b.build(b.BuildSpec(b.SLAB, "hexagonal", {"a": 2.46}, 1,
                                (("C", (0, 0, 0)), ("C", (1 / 3, 2 / 3, 1.5)))))
    atoms = built.to_ase()
    assert list(atoms.pbc) == [True, True, False]
    assert np.allclose(atoms.cell[2], [0, 0, b.VACUUM])
    assert np.isclose(atoms.cell.cellpar()[5], 120.0)
    assert np.isclose(atoms.positions[1, 2], 1.5)                  # Å, not a fraction
    assert np.isclose(np.linalg.norm(atoms.positions[1, :2]), 2.46 / np.sqrt(3))


def test_a_polymer_runs_along_x_with_its_side_atoms_in_angstrom():
    built = b.build(b.BuildSpec(b.POLYMER, "chain", {"a": 2.5}, 1,
                                (("C", (0, 0, 0)), ("H", (0.5, 0.9, 0.6)))))
    atoms = built.to_ase()
    assert list(atoms.pbc) == [True, False, False]
    assert np.allclose(atoms.positions[1], [1.25, 0.9, 0.6])


def test_a_molecule_has_no_cell_and_cartesian_atoms():
    built = b.build(b.BuildSpec(b.MOLECULE, "none", {}, 1,
                                (("O", (0, 0, 0)), ("H", (0.96, 0, 0)))))
    atoms = built.to_ase()
    assert not atoms.pbc.any() and np.allclose(atoms.cell, 0)
    assert np.allclose(atoms.positions[1], [0.96, 0, 0])


@pytest.mark.parametrize("spec, why", [
    (b.BuildSpec(b.CRYSTAL, "cubic", {"a": 3.0}, 1, ()), "at least one atom"),
    (b.BuildSpec(b.CRYSTAL, "cubic", {"a": 3.0}, 1, (("Xx", (0, 0, 0)),)), "not an element"),
    (b.BuildSpec(b.CRYSTAL, "triclinic",
                 dict(a=3, b=3, c=3, alpha=170, beta=170, gamma=170), 1,
                 (("C", (0, 0, 0)),)), "do not make a cell"),
])
def test_what_is_not_a_structure_is_refused(spec, why):
    with pytest.raises(b.BuildError, match=why):
        b.build(spec)


def test_tied_parameters_follow_the_free_ones():
    rhombohedral = b.system(b.CRYSTAL, "rhombohedral")
    values = rhombohedral.complete({"a": 4.0, "alpha": 60.0})
    assert values == dict(a=4.0, b=4.0, c=4.0, alpha=60.0, beta=60.0, gamma=60.0)


@pytest.mark.parametrize("number, system", [(1, "triclinic"), (14, "monoclinic"),
                                            (62, "orthorhombic"), (139, "tetragonal"),
                                            (166, "hexagonal"), (194, "hexagonal"),
                                            (225, "cubic")])
def test_a_space_group_names_the_system_its_cell_is_written_in(number, system):
    assert b.system_of_space_group(number) == system


# ── the dialog ────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _dialog():
    from crystalline.ui.panels.build_dialog import BuildDialog

    return BuildDialog()


def test_the_dialog_offers_only_what_the_periodicity_has(qapp):
    dialog = _dialog()
    dialog.periodicity.setCurrentIndex(dialog.periodicity.findData(b.SLAB))
    keys = [dialog.lattice.itemData(i) for i in range(dialog.lattice.count())]
    assert keys == ["square", "rectangular", "hexagonal", "oblique"]
    assert not dialog._parameters["gamma"].isHidden()
    assert dialog._parameters["c"].isHidden()
    assert all(widget.isHidden() for widget in dialog._group_row)   # no space group
    header = dialog._table.horizontalHeaderItem(3).text()
    assert header == "z (Å)"


def test_tied_parameters_are_shown_and_locked(qapp):
    dialog = _dialog()
    dialog.lattice.setCurrentIndex(dialog.lattice.findData("hexagonal"))
    gamma = dialog._parameters["gamma"]
    assert gamma.value() == 120.0 and not gamma.isEnabled()


def test_a_space_group_sets_its_lattice_system(qapp):
    dialog = _dialog()
    dialog.space_group.setValue(194)
    assert dialog.lattice.currentData() == "hexagonal"
    assert dialog._group_symbol.text().startswith("P6_3/m")


def test_the_dialog_builds_what_it_shows_and_says_why_it_cannot(qapp):
    dialog = _dialog()
    dialog.space_group.setValue(225)
    dialog._parameters["a"].setValue(4.21)
    dialog._table.item(0, 0).setText("Mg")
    dialog._add_row("O", (0.5, 0.5, 0.5))
    dialog.accept()
    assert len(dialog.structure()) == 8

    refusing = _dialog()
    refusing._table.item(0, 0).setText("Qq")
    refusing.accept()
    assert refusing.structure() is None and "not an element" in refusing._error.text()
