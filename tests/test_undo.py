"""Undo history — the Qt-free stack and its integration with Structure.

``MainWindow`` can't be built headless (its QtInteractor segfaults off-screen),
so the undo *wiring* is reproduced here against a real ``Structure`` with the
exact same suppress/record/undo dance MainWindow uses.
"""

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyvista")

from ase.build import bulk  # noqa: E402

from crystalline.core.structure import Structure  # noqa: E402
from crystalline.core.undo import UndoHistory  # noqa: E402
from crystalline.ui.main_window import MainWindow  # noqa: E402


def test_undo_history_records_and_pops_in_order():
    h = UndoHistory()
    h.reset("s0")
    assert not h.can_undo()

    h.record("s1")  # edit 1: s0 becomes undoable
    h.record("s2")  # edit 2: s1 becomes undoable
    assert h.can_undo() and len(h) == 2

    assert h.undo() == "s1"
    assert h.undo() == "s0"
    assert not h.can_undo()
    assert h.undo() is None  # nothing left


def test_redo_reapplies_undone_steps():
    h = UndoHistory()
    h.reset("s0")
    h.record("s1")
    h.record("s2")
    assert not h.can_redo()

    assert h.undo() == "s1"
    assert h.can_redo()
    assert h.undo() == "s0"
    assert not h.can_undo()

    assert h.redo() == "s1"  # forward again
    assert h.redo() == "s2"
    assert not h.can_redo()
    assert h.redo() is None  # nothing left to redo


def test_new_edit_after_undo_clears_redo():
    h = UndoHistory()
    h.reset("s0")
    h.record("s1")
    assert h.undo() == "s0"
    assert h.can_redo()

    h.record("s2")  # a fresh edit branches off — the old redo timeline is gone
    assert not h.can_redo()
    assert h.redo() is None
    assert h.undo() == "s0"  # s2's prior state (s0) is undoable


def test_undo_history_reset_clears_timeline():
    h = UndoHistory()
    h.reset("a")
    h.record("b")
    assert h.can_undo()
    h.reset("c")  # a wholesale replacement (file load / view change)
    assert not h.can_undo() and len(h) == 0


def test_undo_history_respects_limit():
    h = UndoHistory(limit=2)
    h.reset("s0")
    for s in ("s1", "s2", "s3"):
        h.record(s)
    # only the two most recent steps survive (s0 was dropped)
    assert len(h) == 2
    assert h.undo() == "s2"
    assert h.undo() == "s1"
    assert h.undo() is None


class _UndoHarness:
    """Replicates MainWindow's undo wiring against a Structure (no Qt)."""

    def __init__(self, structure: Structure) -> None:
        self.structure = structure
        self._history = UndoHistory()
        self._suppress = False
        structure.add_listener(self._on_change)
        self._history.reset(structure.to_ase())

    def _on_change(self, s: Structure) -> None:
        if self._suppress:
            return
        self._history.record(s.to_ase())

    def can_undo(self) -> bool:
        return self._history.can_undo()

    def undo(self) -> None:
        atoms = self._history.undo()
        if atoms is None:
            return
        self._suppress = True
        try:
            self.structure.restore(atoms)
        finally:
            self._suppress = False


def test_undo_reverts_move_delete_and_add():
    s = Structure.empty()
    s.set_cell(np.eye(3) * 10, periodic=True)
    s.add_atom("C", [0.0, 0.0, 0.0])
    s.add_atom("O", [1.2, 0.0, 0.0])
    harness = _UndoHarness(s)  # baseline = 2 atoms (C, O)

    s.move_atom(1, [3.0, 0.0, 0.0])  # edit 1
    s.add_atom("H", [5.0, 0.0, 0.0])  # edit 2 (now 3 atoms)
    s.remove_atoms([0])  # edit 3 (now 2 atoms: O, H)
    assert s.symbols == ["O", "H"]

    harness.undo()  # undo the delete -> C, O(moved), H
    assert s.symbols == ["C", "O", "H"]
    assert np.allclose(s.positions[1], [3.0, 0.0, 0.0])

    harness.undo()  # undo the add -> C, O(moved)
    assert s.symbols == ["C", "O"]
    assert np.allclose(s.positions[1], [3.0, 0.0, 0.0])

    harness.undo()  # undo the move -> O back at [1.2, 0, 0]
    assert s.symbols == ["C", "O"]
    assert np.allclose(s.positions[1], [1.2, 0.0, 0.0])

    assert not harness.can_undo()  # back to the baseline


def test_restore_preserves_cell_and_notifies():
    s = Structure.empty()
    s.set_cell(np.diag([4.0, 5.0, 6.0]), periodic=True)
    s.add_atom("Na", [0.0, 0.0, 0.0])
    snapshot = s.to_ase()

    s.add_atom("Cl", [2.0, 0.0, 0.0])
    assert len(s) == 2

    events = []
    s.add_listener(lambda st: events.append(len(st)))
    s.restore(snapshot)
    assert len(s) == 1 and s.symbols == ["Na"]
    assert np.allclose(np.asarray(s.cell), np.diag([4.0, 5.0, 6.0]))
    assert events == [1]  # restore fires exactly one notification


# ── undoing a change of view, not just of atoms ──────────────────────────

class _ViewHarness:
    """MainWindow's undo and view machinery, on stand-ins for its widgets.

    The methods here are the window's own — ``_apply_cell_view``,
    ``_replace_structure`` and ``_compose_view`` included — so what is exercised
    is the funnel every Cell action really goes through, not a copy of it that
    can drift. Only the leaves are stubbed: a viewport that draws nothing, a
    panel that holds a structure, an export action that updates nothing.
    """

    _snapshot = MainWindow._snapshot
    _capture_undo = MainWindow._capture_undo
    _reset_undo = MainWindow._reset_undo
    _undo = MainWindow._undo
    _redo = MainWindow._redo
    _apply_history = MainWindow._apply_history
    _derivation_matches = MainWindow._derivation_matches
    _update_boundary_control = MainWindow._update_boundary_control
    _update_supercell_action = MainWindow._update_supercell_action
    _apply_cell_view = MainWindow._apply_cell_view
    _show_derived = MainWindow._show_derived
    _on_boundary_toggled = MainWindow._on_boundary_toggled
    _rebuild_keeping_edits = MainWindow._rebuild_keeping_edits
    _retile = MainWindow._retile
    _clean_view = MainWindow._clean_view
    _start_again_from_file = MainWindow._start_again_from_file
    _switch_keeping_edits = MainWindow._switch_keeping_edits
    _reshape_lattice = MainWindow._reshape_lattice
    _compose_view = MainWindow._compose_view
    _replace_structure = MainWindow._replace_structure
    _set_supercell = MainWindow._set_supercell
    _set_cell_view = MainWindow._set_cell_view
    _note_edited = MainWindow._note_edited

    def __init__(self, source: Structure) -> None:
        from crystalline.core.cells import CellView
        from crystalline.viz.render_settings import RenderSettings

        self._source = source
        self.structure = None
        self._cell_view = CellView.PRIMITIVE
        self._supercell = (1, 1, 1)
        self._show_boundary = False
        self._shown_edited = False
        self._modes = None
        self._modes_natom = None
        self._orbital = None
        self._adps = None
        self._adp_index = None
        self._unit_cell = None
        self._bond_structure = None
        self._history = UndoHistory()
        self._suppress_undo = True          # no history while the first view builds
        settings = RenderSettings()
        self.viewport = type("V", (), {
            "renderer": type("R", (), {"settings": settings,
                                       "set_reference_cell": lambda *a: None})(),
            "show_structure": lambda *a, **k: None,
        })()
        self.structure_panel = type("P", (), {
            "set_structure": lambda *a: None, "clear_selection": lambda *a: None,
        })()
        self.phonon_panel = type("Ph", (), {
            "clear": lambda *a: None, "set_supercell": lambda *a, **k: None,
        })()
        self._tile_restore = None
        self._apply_cell_view()
        self._suppress_undo = False
        self._reset_undo()

    # The undo half of MainWindow._on_structure_changed; the rest of that method
    # is panels and redraws, which have nothing to say about history.
    def _on_structure_changed(self, s: Structure) -> None:
        self._capture_undo(s)

    def _analysis_cell(self):
        return self.structure

    def _update_view_actions(self):
        pass

    def _update_undo_action(self):
        pass

    def _update_cell_view_controls(self):
        pass

    def _refresh_adp_tensors(self, *a, **k):
        pass

    def _update_adp_controls(self, *a, **k):
        pass

    def _show_modes(self, modes):
        pass

    def _update_export_actions(self):
        pass

    # One file and no tab bar: a slot meant for a tab runs as it is.
    _tab = None

    def _routed(self, tab, slot):
        return slot


def _harness():
    s = Structure.from_ase(bulk("MgO", "rocksalt", a=4.21))
    window = _ViewHarness(s)
    window.structure.add_listener(window._on_structure_changed)
    return window


def test_a_supercell_can_be_undone():
    """It could not: every Cell action ended the timeline instead of joining it."""
    window = _harness()
    before = len(window.structure)

    window._set_supercell((2, 1, 1))
    window._apply_cell_view()
    assert len(window.structure) == 2 * before
    assert window._history.can_undo()

    window._undo()

    assert window._supercell == (1, 1, 1)
    assert len(window.structure) == before


def test_a_lattice_parameter_change_can_be_undone():
    """These are applied to the *source*, which no listener watches, so the
    change never reached the history at all — and the re-derive after it wiped
    what was there."""
    window = _harness()
    a0 = float(window._source.cellpar[0])

    window._source.set_lattice_parameters(5.0, 5.0, 5.0, 90.0, 90.0, 90.0)
    window._apply_cell_view()
    assert float(window._source.cellpar[0]) == pytest.approx(5.0)

    window._undo()

    assert float(window._source.cellpar[0]) == pytest.approx(a0)
    assert float(window.structure.cellpar[0]) == pytest.approx(a0)


def test_edits_made_before_a_supercell_are_still_undoable():
    """The worst of it: three atoms dragged into place and then a supercell, and
    the drags could no longer be undone — the history went with the re-derive."""
    window = _harness()
    original = np.array(window.structure.positions[1])

    window.structure.move_atom(1, [1.0, 1.0, 1.0])
    window._set_supercell((2, 1, 1))
    window._apply_cell_view()

    window._undo()                      # the supercell
    assert window._supercell == (1, 1, 1)
    window._undo()                      # and now the move underneath it

    assert np.allclose(window.structure.positions[1], original)
    assert not window._history.can_undo()


def test_an_edit_made_on_a_supercell_comes_back_with_it():
    """Undo steps back onto the view the edit was made in, atoms and all."""
    window = _harness()
    window._set_supercell((2, 1, 1))
    window._apply_cell_view()
    window.structure.add_listener(window._on_structure_changed)

    window.structure.add_atom("H", [0.5, 0.5, 0.5])
    n_with_h = len(window.structure)

    window._undo()                      # the added atom, still in the supercell
    assert window._supercell == (2, 1, 1)
    assert len(window.structure) == n_with_h - 1

    window._redo()
    assert window._supercell == (2, 1, 1)
    assert len(window.structure) == n_with_h


def test_a_rebuild_that_changes_nothing_leaves_no_step_to_undo():
    """Switching q-point falls back to a full rebuild; it changes the modes, not
    the structure, and must not leave an undo step that puts nothing back."""
    window = _harness()

    window._apply_cell_view()
    window._apply_cell_view()

    assert not window._history.can_undo()


def test_switching_cell_setting_is_not_an_undo_step():
    """Which of the two settings a crystal is drawn in is a way of looking at
    it, not a change to it. Ctrl+Z belongs to the edits."""
    from crystalline.core.cells import CellView

    window = _harness()
    assert not window._history.can_undo()

    window._set_cell_view(CellView.CRYSTALLOGRAPHIC)

    assert not window._history.can_undo(), "the switch left a step to undo"
    assert window._cell_view is CellView.CRYSTALLOGRAPHIC


def test_an_edit_after_a_switch_undoes_to_that_switch_not_through_it():
    """The baseline moves with the switch. Without that, the next edit would
    push the *old* setting onto the stack and undoing the edit would flip the
    cell as well."""
    from crystalline.core.cells import CellView

    window = _harness()
    window._set_cell_view(CellView.CRYSTALLOGRAPHIC)
    window.structure.add_listener(window._on_structure_changed)
    n = len(window.structure)

    window.structure.add_atom("H", [0.5, 0.5, 0.5])
    window._undo()

    assert len(window.structure) == n
    assert window._cell_view is CellView.CRYSTALLOGRAPHIC, "the cell setting moved"


def test_undoing_back_past_a_switch_returns_to_the_setting_of_that_edit():
    """An edit made in the primitive cell is dropped by a switch — it lives in
    the history and nowhere else — so stepping back to it has to bring its own
    setting with it."""
    from crystalline.core.cells import CellView

    window = _harness()
    original = np.array(window.structure.positions[1])
    window.structure.move_atom(1, [1.0, 1.0, 1.0])

    window._set_cell_view(CellView.CRYSTALLOGRAPHIC)
    window._undo()

    assert window._cell_view is CellView.PRIMITIVE
    assert np.allclose(window.structure.positions[1], [1.0, 1.0, 1.0])  # the edit is back
    window._undo()
    assert np.allclose(window.structure.positions[1], original)


def test_an_undo_does_not_leave_the_view_looking_edited():
    """Reported: open a file, make a supercell, undo it, switch to the primitive
    cell — and the next undo switched back to the conventional one.

    Restoring a snapshot goes through the structure's listeners, so it looks
    like an edit from the outside and set the "there are edits to lose" flag.
    The switch after it was then recorded as destructive, which put a step on
    the stack that had no business being there. A snapshot carries what that
    flag was when it was taken.
    """
    from crystalline.core.cells import CellView

    window = _harness()
    window._cell_view = CellView.CRYSTALLOGRAPHIC
    window._apply_cell_view()
    window._reset_undo()

    window._set_supercell((2, 1, 1))
    window._apply_cell_view()
    window._undo()
    assert not window._shown_edited, "an undo left the cell looking edited"

    window._set_cell_view(CellView.PRIMITIVE)
    assert not window._history.can_undo(), "the switch was recorded as a step"

    window._undo()
    assert window._cell_view is CellView.PRIMITIVE  # and nothing moved under us



def test_completing_the_boundary_keeps_the_edits_and_undoes_one_step_at_a_time():
    """Completing molecules is a way of drawing the crystal: it re-derived the
    view from the file and threw away what had been done to the atoms."""
    from crystalline.core.cells import to_analysis_cell

    def crystal(window):
        """The folded cell as a set of wrapped fractional positions."""
        atoms = to_analysis_cell(window.structure, window._unit_cell).to_ase()
        return sorted(tuple(np.round(f % 1.0, 4)) for f in atoms.get_scaled_positions(wrap=False))

    window = _harness()
    pristine = crystal(window)
    window.structure.translate_atoms([0], (0.0, 0.0, 0.3))
    edited = crystal(window)
    assert edited != pristine

    window._on_boundary_toggled(True)
    assert crystal(window) == edited
    assert window._shown_edited                      # still work a re-derive would lose
    window._on_boundary_toggled(False)
    assert crystal(window) == edited

    window._undo()                                   # completion back on
    assert window._show_boundary and crystal(window) == edited
    window._undo()                                   # and off, the edit still there
    assert not window._show_boundary and crystal(window) == edited
    window._undo()                                   # the edit itself
    assert crystal(window) == pristine



def _folded(window):
    from crystalline.core.cells import to_analysis_cell

    atoms = to_analysis_cell(window.structure, window._unit_cell).to_ase()
    return sorted(tuple(np.round(f % 1.0, 4)) for f in atoms.get_scaled_positions(wrap=False))


def test_a_supercell_of_an_edited_cell_keeps_the_edit_and_undoes_back_to_it():
    """Taking a supercell re-derived the view from the file, over the edit."""
    window = _harness()
    window.structure.translate_atoms([0], (0.0, 0.0, 0.3))
    edited = _folded(window)

    assert window._retile((2, 1, 1))
    assert window._supercell == (2, 1, 1)
    assert len(window.structure) == 2 * len(window._source)
    assert _folded(window) == edited

    window._undo()
    assert window._supercell == (1, 1, 1) and _folded(window) == edited


def test_an_edit_in_one_tile_is_carried_into_a_whole_multiple_and_asked_about_otherwise(
        monkeypatch):
    from crystalline.ui import main_window

    window = _harness()
    window._retile((2, 1, 1))
    window.structure.remove_atoms([0])               # one atom gone, from one tile
    count = len(window.structure)

    assert window._retile((4, 1, 1))                 # twice the 2×1×1, edit and all
    assert len(window.structure) == 2 * count

    asked = []
    monkeypatch.setattr(main_window.QMessageBox, "question",
                        lambda *a, **k: (asked.append(a), main_window.QMessageBox.Cancel)[1])
    assert not window._retile((3, 1, 1))             # not a multiple: asked, declined
    assert asked and window._supercell == (4, 1, 1)
    assert len(window.structure) == 2 * count



def test_an_edit_in_the_primitive_cell_is_carried_into_the_conventional_and_back():
    """The switch re-derived from the file, over the edit. MgO's primitive cell
    holds 2 atoms and its face-centred cube 8: one atom moved in the first is
    four atoms moved alike in the second, and those reduce to it again."""
    from crystalline.core.cells import CellView

    window = _harness()
    window.structure.translate_atoms([0], (0.0, 0.0, 0.3))
    primitive = _folded(window)

    window._set_cell_view(CellView.CRYSTALLOGRAPHIC)
    assert window._cell_view is CellView.CRYSTALLOGRAPHIC
    assert len(window.structure) == 4 * len(window._source)       # boundary off: the cube
    assert window._shown_edited

    window._set_cell_view(CellView.PRIMITIVE)
    assert window._cell_view is CellView.PRIMITIVE
    assert _folded(window) == primitive


def test_an_edit_that_breaks_the_centring_is_asked_about_before_it_is_dropped(monkeypatch):
    from crystalline.core.cells import CellView
    from crystalline.ui import main_window

    window = _harness()
    window._set_cell_view(CellView.CRYSTALLOGRAPHIC)
    window.structure.translate_atoms([0], (0.0, 0.0, 0.3))         # one of four copies alone
    edited = _folded(window)
    asked = []
    monkeypatch.setattr(main_window.QMessageBox, "question",
                        lambda *a, **k: (asked.append(a), main_window.QMessageBox.Cancel)[1])

    window._set_cell_view(CellView.PRIMITIVE)

    assert asked and window._cell_view is CellView.CRYSTALLOGRAPHIC
    assert _folded(window) == edited



def test_a_lattice_change_reshapes_the_cell_on_screen_edits_and_all():
    """It reshaped the file's cell and derived the view again, over the edits;
    and it reshaped the *primitive* cell while the conventional one was shown."""
    from crystalline.core.cells import CellView, reshaped_cell

    window = _harness()
    window._set_cell_view(CellView.CRYSTALLOGRAPHIC)
    a = float(np.linalg.norm(window._unit_cell[0]))
    window.structure.translate_atoms([0], (0.0, 0.0, 0.3))
    edited = _folded(window)
    file_cell = np.array(window._source.cell)

    window._reshape_lattice(reshaped_cell(window._unit_cell, [a + 0.2, a + 0.2, a + 0.2, 90, 90, 90]))

    assert np.linalg.norm(window._unit_cell[0]) == pytest.approx(a + 0.2)
    assert _folded(window) == edited                        # same fractions, edit included
    assert np.allclose(window._source.cell, file_cell)      # the file's cell left alone

    window._undo()
    assert np.linalg.norm(window._unit_cell[0]) == pytest.approx(a)
    assert _folded(window) == edited
