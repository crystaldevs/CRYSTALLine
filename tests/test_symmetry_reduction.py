"""Lowering a crystal's symmetry on purpose.

Checked against crystallography rather than against what the code emits: the
subgroups of m3̄m, of 6/mmm and of 4/mmm are in the tables, and a reduction that
disagrees with them is wrong however plausible it looks.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from crystalline.core import symmetry_reduction as reduction
from crystalline.core.structure import Structure


def _mgo() -> Structure:
    return Structure.from_ase(bulk("MgO", "rocksalt", a=4.21))


def _perovskite() -> Structure:
    a = 3.905
    return Structure.from_ase(Atoms(
        "SrTiO3",
        scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0),
                          (0.5, 0, 0.5), (0, 0.5, 0.5)],
        cell=[[a, 0, 0], [0, a, 0], [0, 0, a]], pbc=True))


def _operator(symmetry, matrix):
    return next(op for op in symmetry.operators
                if np.array_equal(op.rotation, np.asarray(matrix, dtype=int)))


# ── what the structure starts with ────────────────────────────────────────
def test_the_full_symmetry_is_the_one_the_structure_has():
    symmetry = reduction.analyse(_mgo())
    assert symmetry is not None
    full = symmetry.full()
    assert (full.number, full.symbol) == (225, "Fm-3m")
    assert full.order == 48          # the order of m3̄m
    assert full.sites == 2           # one Mg orbit, one O orbit


def test_a_molecule_has_no_crystal_symmetry_to_reduce():
    """The feature is meaningless without a lattice, and must say so rather
    than raise at a panel that is only trying to draw itself."""
    from ase.build import molecule

    assert reduction.analyse(Structure.from_ase(molecule("H2O"))) is None


def test_every_operator_is_named_with_the_direction_it_acts_along():
    """"2" is four different operators in a cubic crystal; "2 ∥ [110]" is one."""
    symmetry = reduction.analyse(_mgo())
    labels = [op.label for op in symmetry.operators]
    assert len(labels) == 48
    assert "1̄" in labels
    axes = [label for label in labels if label.startswith("2 ")]
    assert len(axes) > 1 and all("∥" in label for label in axes)


# ── the group theory, against the tables ──────────────────────────────────
def test_removing_the_inversion_from_m3m_leaves_432_or_43m():
    """The two maximal subgroups of m3̄m without a centre, and no others.

    This is the case that decides the whole interface: one operator is asked
    about, twenty-four go, and there are *two* answers — so a tick box that
    quietly picked one would be choosing the physics for the user.
    """
    symmetry = reduction.analyse(_mgo())
    inversion = _operator(symmetry, -np.eye(3, dtype=int))
    options = reduction.subgroups_without(symmetry, [inversion.rotation])

    assert {(option.number, option.symbol) for option in options} == {
        (209, "F432"), (216, "F-43m")}
    for option in options:
        assert option.order == 24, "half of m3̄m, not one operator fewer"
        assert option.writable


def test_the_kept_operators_are_closed_under_composition():
    """Whatever comes back has to be a group, or it is not a symmetry at all."""
    symmetry = reduction.analyse(_mgo())
    inversion = _operator(symmetry, -np.eye(3, dtype=int))
    for option in reduction.subgroups_without(symmetry, [inversion.rotation]):
        kept = {reduction._key(rotation) for rotation in option.rotations}
        for left in option.rotations:
            for right in option.rotations:
                assert reduction._key(left @ right) in kept


def test_what_was_removed_is_reported_without_repeating_itself():
    """A cubic group loses two dozen operators at a time and most share a
    symbol; the list has to read as a sentence, not a stutter."""
    symmetry = reduction.analyse(_mgo())
    inversion = _operator(symmetry, -np.eye(3, dtype=int))
    option = reduction.subgroups_without(symmetry, [inversion.rotation])[0]
    assert "1̄" in option.dropped
    assert len(option.dropped) == len(set(option.dropped))
    assert any("×" in entry for entry in option.dropped), "repeats should be counted"


def test_a_second_removal_descends_from_where_the_first_left_off():
    """Having given up the inversion, the next operator dropped must leave
    something inside what is left — not somewhere else in the full group."""
    symmetry = reduction.analyse(_mgo())
    inversion = _operator(symmetry, -np.eye(3, dtype=int))
    first = reduction.subgroups_without(symmetry, [inversion.rotation])[0]

    still_there = [op for op in symmetry.operators
                   if not op.is_identity
                   and any(np.array_equal(op.rotation, r) for r in first.rotations)]
    second = reduction.subgroups_without(
        symmetry, [still_there[0].rotation], within=first.rotations)
    assert second
    kept_first = {reduction._key(r) for r in first.rotations}
    for option in second:
        assert option.order < first.order
        assert {reduction._key(r) for r in option.rotations} <= kept_first


def test_an_explicit_set_of_operators_is_closed_before_it_is_named():
    """A caller handing over something that is not a group gets the group it
    generates, not a silent lie about what was asked for."""
    symmetry = reduction.analyse(_mgo())
    inversion = _operator(symmetry, -np.eye(3, dtype=int))
    identity = np.eye(3, dtype=int)
    closed = reduction.descend(symmetry, [identity, inversion.rotation])
    assert closed.order == 2                     # {1, 1̄} is already a group
    assert closed.symbol.startswith(("P-1", "F-1", "C-1", "A-1", "I-1", "R-3"))


def test_removing_nothing_gives_the_group_back():
    symmetry = reduction.analyse(_mgo())
    assert reduction.subgroups_without(symmetry, [])[0].order == symmetry.full().order


# ── what it buys: sites that can move independently ───────────────────────
def test_lowering_the_symmetry_frees_atoms_that_were_equivalent():
    """The whole point. In Pm3̄m the three oxygens of SrTiO3 are one orbit; a
    descent to orthorhombic makes them three, and a geometry optimisation that
    could only breathe can now distort."""
    symmetry = reduction.analyse(_perovskite())
    assert symmetry.full().sites == 3            # Sr, Ti, and one O orbit

    orthorhombic = reduction.descend(symmetry, [
        np.diag((1, 1, 1)), np.diag((1, -1, -1)),
        np.diag((-1, 1, -1)), np.diag((-1, -1, 1))])
    assert orthorhombic.number == 16             # P222
    assert orthorhombic.sites == 5, "the three oxygens should have come apart"


def test_dropping_all_point_symmetry_lists_every_atom():
    """No symmetry means every atom is its own site — and that description is
    always writable, whatever cell it is in."""
    symmetry = reduction.analyse(_mgo())
    p1 = reduction.descend(symmetry, [np.eye(3, dtype=int)])
    assert (p1.number, p1.order) == (1, 1)
    assert p1.sites == len(symmetry.positions)
    assert p1.writable


# ── the guard against a plausible wrong deck ──────────────────────────────
@pytest.mark.parametrize("name, atoms", [
    ("rocksalt", bulk("MgO", "rocksalt", a=4.21)),
    ("diamond", bulk("Si", "diamond", a=5.43)),
    ("hcp", bulk("Mg", "hcp", a=3.21, c=5.21)),
])
def test_a_writable_subgroup_really_does_rebuild_the_crystal(name, atoms):
    """``writable`` is not a hope: it is the answer to "declare this group and
    this asymmetric unit — do you get the crystal back?"."""
    symmetry = reduction.analyse(Structure.from_ase(atoms))
    inversion = next((op for op in symmetry.operators
                      if np.array_equal(op.rotation, -np.eye(3, dtype=int))), None)
    if inversion is None:
        pytest.skip(f"{name} has no centre of symmetry to remove")
    for option in reduction.subgroups_without(symmetry, [inversion.rotation]):
        if option.writable:
            assert reduction.reproduces(symmetry, option), name


def test_a_subgroup_whose_setting_is_not_our_cell_is_marked_unwritable():
    """The group theory is exact; the setting is the risk. A monoclinic
    subgroup of a cubic crystal is a real group whose standard setting is not
    the cubic cell we hold, and a deck written from it would be a plausible
    file describing a different crystal. It has to be caught, not written.
    """
    symmetry = reduction.analyse(_mgo())
    # {1, 2 ∥ [001]} — a real subgroup, in a cell that is not its setting
    monoclinic = reduction.descend(symmetry, [np.diag((-1, -1, 1))])
    assert monoclinic.order == 2
    assert not monoclinic.writable
    assert "cell setting" in monoclinic.summary()


def test_the_asymmetric_unit_is_a_representative_of_every_orbit():
    symmetry = reduction.analyse(_perovskite())
    full = symmetry.full()
    numbers, coords = reduction.asymmetric_unit(symmetry, full.rotations)
    assert len(numbers) == full.sites == len(coords)
    assert sorted(int(z) for z in numbers) == [8, 22, 38]   # one O, one Ti, one Sr


# ── it has to be quick enough to sit behind a tick box ────────────────────
def test_the_subgroup_search_is_fast_enough_to_be_interactive():
    """m3̄m has 48 operators and ~18000 generator sets to close. Done with
    matrix products that is over two seconds, which is felt; done on a
    multiplication table it is a few tens of milliseconds."""
    import time

    symmetry = reduction.analyse(_mgo())
    reduction._subgroups_of.cache_clear()
    start = time.perf_counter()
    reduction._all_subgroups(symmetry.operators)
    assert time.perf_counter() - start < 1.0


# ── what reaches the deck ─────────────────────────────────────────────────
def _deck(structure, kept=()):
    from crystalline.core.crystal_input import (
        CrystalInputSpec, GeometryOptions, build_input,
    )

    spec = CrystalInputSpec(geometry=GeometryOptions(kept_rotations=kept))
    return build_input(structure, spec).splitlines()


def _as_tuples(rotations):
    return tuple(tuple(map(tuple, np.asarray(r, dtype=int))) for r in rotations)


def test_a_reduced_deck_keeps_every_atom_where_it_was():
    """Nothing moves. The atoms, the cell and the coordinates are what they
    were; only how many of them are independent has changed."""
    structure = _perovskite()
    symmetry = reduction.analyse(structure)
    p222 = reduction.descend(symmetry, [
        np.diag((1, 1, 1)), np.diag((1, -1, -1)),
        np.diag((-1, 1, -1)), np.diag((-1, -1, 1))])

    full, reduced = _deck(structure), _deck(structure, _as_tuples(p222.rotations))
    assert full[3] == "221" and reduced[3] == "16"
    assert full[5] == "3" and reduced[5] == "5"      # the asymmetric unit grew
    # ...and the sites the full deck listed are still there, unmoved
    for line in full[6:9]:
        assert line in reduced


def test_a_reduced_deck_writes_the_lattice_its_group_expects():
    """Group 16 is orthorhombic, so CRYSTAL wants a, b and c — even though this
    particular cell still happens to be cubic."""
    structure = _perovskite()
    symmetry = reduction.analyse(structure)
    p222 = reduction.descend(symmetry, [
        np.diag((1, 1, 1)), np.diag((1, -1, -1)),
        np.diag((-1, 1, -1)), np.diag((-1, -1, 1))])
    reduced = _deck(structure, _as_tuples(p222.rotations))
    assert len(reduced[4].split()) == 3, reduced[4]
    assert len(_deck(structure)[4].split()) == 1, "the cubic deck needs only a"


def test_a_reduction_that_cannot_be_written_falls_back_to_no_symmetry():
    """Never back to the *full* group: that would silently hand back the
    constraint the user asked to be rid of. Group 1 with every atom is the
    honest answer when the subgroup's setting is not this cell."""
    structure = _mgo()
    symmetry = reduction.analyse(structure)
    monoclinic = reduction.descend(symmetry, [np.diag((-1, -1, 1))])
    assert not monoclinic.writable

    deck = _deck(structure, _as_tuples(monoclinic.rotations))
    assert deck[3] == "1", "must not quietly restore Fm-3m"
    # Every atom of the structure as it stands — the primitive cell here, which
    # is a smaller but equally complete description than the conventional one.
    assert int(deck[5]) == len(structure)


def test_asking_for_nothing_in_particular_still_uses_the_full_symmetry():
    """The ordinary case has to be untouched by any of this."""
    assert _deck(_mgo())[3] == "225"


# ── the reduction belongs to the crystal ──────────────────────────────────
def test_a_reduction_rides_on_the_structure_through_undo_and_redo():
    """It is a property of *this crystal as the user has decided to treat it*,
    so it has to survive an edit, an undo and a redo like the atoms do — which
    it does by living in ``atoms.info``, the dict ``copy()`` carries along."""
    structure = _mgo()
    assert structure.reduced_symmetry == ()

    before = structure.to_ase()                       # an undo snapshot
    structure.set_reduced_symmetry([np.eye(3, dtype=int), np.diag((-1, -1, 1))])
    after = structure.to_ase()
    assert len(structure.reduced_symmetry) == 2

    structure.restore(before)
    assert structure.reduced_symmetry == (), "undo should take the reduction with it"
    structure.restore(after)
    assert len(structure.reduced_symmetry) == 2, "and redo should bring it back"

    structure.set_reduced_symmetry(None)
    assert structure.reduced_symmetry == ()


def test_a_structure_edit_does_not_lose_the_reduction():
    structure = _mgo()
    structure.set_reduced_symmetry([np.eye(3, dtype=int), np.diag((-1, -1, 1))])
    structure.translate_atoms([0], (0.0, 0.0, 0.01))
    assert len(structure.reduced_symmetry) == 2


def test_the_builder_takes_the_reduction_from_the_structure():
    """Not from a control in the dialog: a reduction the user applied to the
    crystal is part of the crystal, so a deck built from it must carry it
    without anyone remembering to say so."""
    from crystalline.core.crystal_input import build_input

    structure = _perovskite()
    symmetry = reduction.analyse(structure)
    p222 = reduction.descend(symmetry, [
        np.diag((1, 1, 1)), np.diag((1, -1, -1)),
        np.diag((-1, 1, -1)), np.diag((-1, -1, 1))])
    structure.set_reduced_symmetry([tuple(map(tuple, r)) for r in p222.rotations])

    from crystalline.core.crystal_input import CrystalInputSpec, GeometryOptions

    spec = CrystalInputSpec(geometry=GeometryOptions(
        kept_rotations=tuple(structure.reduced_symmetry)))
    assert build_input(structure, spec).splitlines()[3] == "16"


# ── the dialog ────────────────────────────────────────────────────────────
def test_the_dialog_offers_both_answers_and_greys_the_unusable_ones():
    """Unticking the inversion has two answers, and the dialog must show both
    rather than choose. An unwritable subgroup is shown greyed — it is real,
    and hiding it would look like a bug in the group theory."""
    pytest.importorskip("PySide6")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from crystalline.ui.panels.symmetry_reduction import SymmetryReductionDialog

    QApplication.instance() or QApplication([])
    structure = _perovskite()
    dialog = SymmetryReductionDialog(structure)

    row = next(i for i in range(dialog._operators.count())
               if np.array_equal(np.asarray(dialog._operators.item(i).data(Qt.UserRole)),
                                 -np.eye(3, dtype=int)))
    dialog._operators.item(row).setCheckState(Qt.Unchecked)

    offered = [dialog._choices.item(i) for i in range(dialog._choices.count())]
    assert len(offered) == 2, "m3̄m without a centre leaves two groups, not one"
    for item, option in zip(offered, dialog._options):
        enabled = bool(item.flags() & Qt.ItemIsEnabled)
        assert enabled == option.writable, "greyed exactly when it cannot be written"


def test_the_dialog_will_not_apply_a_subgroup_it_cannot_write():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from crystalline.ui.panels.symmetry_reduction import SymmetryReductionDialog

    QApplication.instance() or QApplication([])
    dialog = SymmetryReductionDialog(_mgo())
    before = dialog._current.order
    dialog._options = [reduction.descend(dialog._symmetry, [np.diag((-1, -1, 1))])]
    dialog._choices.addItem("unwritable")
    dialog._choices.setCurrentRow(0)
    dialog._take_choice()
    assert dialog._current.order == before, "an unwritable group must not be applied"


def test_the_identity_cannot_be_unticked():
    """Every group contains it, so a tick box that could clear it would be a
    control that does nothing — the thing we have been removing all session."""
    pytest.importorskip("PySide6")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from crystalline.ui.panels.symmetry_reduction import SymmetryReductionDialog

    QApplication.instance() or QApplication([])
    dialog = SymmetryReductionDialog(_mgo())
    identity = next(i for i in range(dialog._operators.count())
                    if np.array_equal(np.asarray(dialog._operators.item(i).data(Qt.UserRole)),
                                      np.eye(3, dtype=int)))
    assert not (dialog._operators.item(identity).flags() & Qt.ItemIsUserCheckable)


def test_a_structure_with_no_symmetry_says_so_instead_of_showing_nothing():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from ase.build import molecule

    from crystalline.ui.panels.symmetry_reduction import SymmetryReductionDialog

    QApplication.instance() or QApplication([])
    dialog = SymmetryReductionDialog(Structure.from_ase(molecule("H2O")))
    assert "no crystal symmetry" in dialog._header.text()
    assert dialog.chosen() == ()


# ── the wiring that was missing: panel -> window -> everything else ───────
def test_the_panel_asks_rather_than_setting_its_own_copy():
    """The panel is handed a *derived* analysis cell — folded to one cell and
    replaced on every edit. A reduction recorded there changes nothing anyone
    can see, which is exactly how it looked: the left panel and the deck both
    carried on as before. So the panel emits, and the window applies."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from crystalline.ui.panels.symmetry_panel import SymmetryPanel

    QApplication.instance() or QApplication([])
    assert hasattr(SymmetryPanel, "reduction_changed")

    import ast
    import inspect

    source = inspect.getsource(SymmetryPanel._open_reduction)
    assert "reduction_changed.emit" in source
    tree = ast.parse(inspect.getsource(SymmetryPanel))
    called = {node.func.attr for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert "set_reduced_symmetry" not in called, "the panel must not set it on its copy"


def test_the_window_applies_the_reduction_to_the_structure_it_holds():
    import inspect

    from crystalline.ui.main_window import MainWindow

    source = inspect.getsource(MainWindow._apply_symmetry_reduction)
    assert "self.structure.set_reduced_symmetry" in source
    # ...and the window wires the panel's signal to it
    assert "reduction_changed" in inspect.getsource(MainWindow._connect_signals)


def test_a_derived_analysis_cell_carries_the_reduction():
    """Every copy the app makes — the analysis cell, the builder's structure,
    an undo snapshot — has to keep it, or the choice evaporates on the next
    edit. It does, because it lives in the dict ``Atoms.copy()`` carries."""
    from crystalline.core.cells import to_analysis_cell

    structure = _perovskite()
    symmetry = reduction.analyse(structure)
    p222 = reduction.descend(symmetry, [
        np.diag((1, 1, 1)), np.diag((1, -1, -1)),
        np.diag((-1, 1, -1)), np.diag((-1, -1, 1))])
    structure.set_reduced_symmetry([tuple(map(tuple, r)) for r in p222.rotations])

    derived = to_analysis_cell(structure, structure.cell)
    assert len(derived.reduced_symmetry) == 4


def test_the_info_panel_reports_both_the_real_group_and_the_declared_one():
    """They are different facts: the space group is the symmetry these atoms
    have, and the reduction is the one they are being given."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QFormLayout

    from crystalline.ui.panels.info_panel import InfoPanel

    QApplication.instance() or QApplication([])
    structure = _perovskite()
    symmetry = reduction.analyse(structure)
    p222 = reduction.descend(symmetry, [
        np.diag((1, 1, 1)), np.diag((1, -1, -1)),
        np.diag((-1, 1, -1)), np.diag((-1, -1, 1))])

    panel = InfoPanel()
    panel.show_structure(structure)
    before = _rows(panel, QFormLayout)
    assert "Treated as:" not in before, "nothing to say when nothing is reduced"

    structure.set_reduced_symmetry([tuple(map(tuple, r)) for r in p222.rotations])
    panel.show_structure(structure)
    after = _rows(panel, QFormLayout)
    assert after["Space group:"].startswith("Pm-3m"), "the real symmetry still shows"
    assert after["Treated as:"] == "P222 (No. 16)"
    assert after["Independent sites:"] == "5 (was 3)"

    # ...directly under the space group, not at the foot of the panel: the two
    # answer the same question, and the second only reads while the first is
    # still in the eye.
    labels = list(after)
    assert labels.index("Treated as:") == labels.index("Space group:") + 1
    assert labels.index("Independent sites:") == labels.index("Treated as:") + 1


def _rows(panel, form_layout):
    form = panel._crystal_form
    rows = {}
    for index in range(form.rowCount()):
        label = form.itemAt(index, form_layout.LabelRole)
        field = form.itemAt(index, form_layout.FieldRole)
        if label and field:
            rows[label.widget().text()] = field.widget().text()
    return rows


def test_opening_the_reduction_from_the_panel_actually_runs():
    """The path a user takes, with the modal step stubbed out.

    Nothing exercised this before: the dialog blocks, so the tests reached
    around it and checked the pieces on either side. That left a plain
    ``AttributeError`` — ``dialog.Accepted``, which PySide6 does not put on the
    instance — to be found by hand, by the safety net, in the app.
    """
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QDialog

    from crystalline.ui import safety
    from crystalline.ui.panels import symmetry_reduction as dialog_module
    from crystalline.ui.panels.symmetry_panel import SymmetryPanel

    QApplication.instance() or QApplication([])
    structure = _perovskite()
    symmetry = reduction.analyse(structure)
    p222 = reduction.descend(symmetry, [
        np.diag((1, 1, 1)), np.diag((1, -1, -1)),
        np.diag((-1, 1, -1)), np.diag((-1, -1, 1))])
    chosen = tuple(tuple(map(tuple, r)) for r in p222.rotations)

    class _Accepting(dialog_module.SymmetryReductionDialog):
        def exec(self):
            self._current = p222
            return QDialog.Accepted

    original = dialog_module.SymmetryReductionDialog
    dialog_module.SymmetryReductionDialog = _Accepting
    try:
        panel = SymmetryPanel(structure)
        asked = []
        panel.reduction_changed.connect(asked.append)
        safety.strict(True)          # a swallowed error would hide the bug again
        try:
            panel._open_reduction()
        finally:
            safety.strict(False)
    finally:
        dialog_module.SymmetryReductionDialog = original

    assert asked == [chosen], "the panel should have asked the window to apply it"


def test_cancelling_the_reduction_asks_for_nothing():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QDialog

    from crystalline.ui import safety
    from crystalline.ui.panels import symmetry_reduction as dialog_module
    from crystalline.ui.panels.symmetry_panel import SymmetryPanel

    QApplication.instance() or QApplication([])

    class _Rejecting(dialog_module.SymmetryReductionDialog):
        def exec(self):
            return QDialog.Rejected

    original = dialog_module.SymmetryReductionDialog
    dialog_module.SymmetryReductionDialog = _Rejecting
    try:
        panel = SymmetryPanel(_perovskite())
        asked = []
        panel.reduction_changed.connect(asked.append)
        safety.strict(True)
        try:
            panel._open_reduction()
        finally:
            safety.strict(False)
    finally:
        dialog_module.SymmetryReductionDialog = original

    assert asked == []
