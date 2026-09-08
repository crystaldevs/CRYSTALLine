"""Undo history — the Qt-free stack and its integration with Structure.

``MainWindow`` can't be built headless (its QtInteractor segfaults off-screen),
so the undo *wiring* is reproduced here against a real ``Structure`` with the
exact same suppress/record/undo dance MainWindow uses.
"""

import numpy as np

from crystalline.core.structure import Structure
from crystalline.core.undo import UndoHistory


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
