"""Tests for StructurePanel's multi-selection model and editing gate.

Pure-Qt widget (no VTK), so it runs headless; skips without PySide6.
"""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from crystalline.core.structure import Structure  # noqa: E402
from crystalline.ui.panels.structure_panel import StructurePanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _panel(qapp, n=4):
    s = Structure.empty()
    for i in range(n):
        s.add_atom("C", [i, 0, 0])
    return StructurePanel(s), s


def test_multiselect_toggle_replace_and_range_filter(qapp):
    panel, _ = _panel(qapp)
    emitted = []
    panel.selection_changed.connect(lambda idx: emitted.append(list(idx)))

    panel.set_selection([1, 3])
    assert panel.selected_indices() == [1, 3]

    panel.select_atom(2, additive=True)  # add
    assert panel.selected_indices() == [1, 2, 3]

    panel.select_atom(1, additive=True)  # toggle off
    assert panel.selected_indices() == [2, 3]

    panel.select_atom(0, additive=False)  # replace
    assert panel.selected_indices() == [0]

    panel.set_selection([0, 9, 5])  # out-of-range filtered away
    assert panel.selected_indices() == [0]

    assert emitted  # selection_changed fired for the UI/highlights


def test_editing_gate_controls(qapp):
    panel, _ = _panel(qapp)
    # editing off by default: add + coordinate editor disabled
    assert not panel.add_btn.isEnabled()

    panel.set_editing_enabled(True)
    assert panel.add_btn.isEnabled()

    panel.set_selection([2])  # single selection while editing -> editor on
    assert panel.editor.isEnabled()

    panel.set_selection([1, 2])  # multi-selection -> single-atom editor off
    assert not panel.editor.isEnabled()
