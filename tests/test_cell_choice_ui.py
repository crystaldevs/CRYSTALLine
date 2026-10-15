"""The Cell box in the Info panel and in the input builder."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("spglib")

from test_cell_setting import _p21n  # noqa: E402

from crystalline.core.cell_setting import COMPUTED, STANDARD  # noqa: E402
from crystalline.core.structure import Structure  # noqa: E402


def _rows(panel):
    form = panel._crystal_form
    rows = {}
    for row in range(form.rowCount()):
        label = form.itemAt(row, form.ItemRole.LabelRole)
        field = form.itemAt(row, form.ItemRole.FieldRole)
        if label is not None and field is not None:
            rows[label.widget().text().rstrip(":")] = field.widget().text()
    return rows


def test_the_info_panel_switches_between_the_two_cells(qapp):
    from crystalline.ui.panels.info_panel import InfoPanel

    panel = InfoPanel()
    panel.set_cell_choice(COMPUTED)
    panel.show_structure(_p21n())
    computed = _rows(panel)
    assert computed["Space group"] == "P2₁/n (No. 14)"
    assert computed["a, b, c (Å)"] == "8.2400, 13.4487, 8.9677"

    seen = []
    panel.cell_choice_changed.connect(seen.append)
    panel._cell.setCurrentIndex(panel._cell.findData(STANDARD))   # as a click would
    standard = _rows(panel)
    assert standard["Space group"] == "P2₁/c (No. 14)"
    assert standard["a, b, c (Å)"] == "8.2400, 13.4487, 10.4338"
    assert standard["Cell volume (Å³)"] == computed["Cell volume (Å³)"]
    assert seen == [STANDARD]


def test_setting_the_choice_from_outside_does_not_announce_it(qapp):
    from crystalline.ui.panels.info_panel import InfoPanel

    panel = InfoPanel()
    seen = []
    panel.cell_choice_changed.connect(seen.append)
    panel.set_cell_choice(STANDARD)
    panel.set_cell_choice(COMPUTED)
    assert seen == []
    assert panel.cell_choice() == COMPUTED


def test_the_cell_box_is_off_for_a_slab(qapp):
    from ase.build import fcc100

    from crystalline.ui.panels.info_panel import InfoPanel

    slab = fcc100("Cu", size=(2, 2, 3), vacuum=10.0)
    slab.pbc = (True, True, False)
    panel = InfoPanel()
    panel.show_structure(Structure.from_ase(slab))
    assert not panel._cell.isEnabled()
    panel.show_structure(_p21n())
    assert panel._cell.isEnabled()


def test_the_builder_starts_on_the_cell_it_is_given_and_writes_it(qapp):
    from crystalline.ui.panels.input_builder import InputBuilderDialog

    dialog = InputBuilderDialog(_p21n(), cell_choice=COMPUTED)
    lines = dialog._preview.toPlainText().splitlines()
    assert lines[2:4] == ["1 0 0", "P 1 21/N 1"]
    assert dialog._cell_note.text() == "Cell as in the file: P2₁/n (No. 14)."

    dialog._cell.setCurrentIndex(dialog._cell.findData(STANDARD))
    lines = dialog._preview.toPlainText().splitlines()
    assert lines[2:4] == ["0 0 0", "14"]
    assert dialog._cell_note.text() == "Standard setting: P2₁/c (No. 14)."

    standard = InputBuilderDialog(_p21n(), cell_choice=STANDARD)
    assert standard._preview.toPlainText().splitlines()[3] == "14"


def test_the_builder_cell_box_follows_the_symmetry_switch(qapp):
    from crystalline.ui.panels.input_builder import InputBuilderDialog

    dialog = InputBuilderDialog(_p21n(), cell_choice=COMPUTED)
    assert dialog._cell.isEnabled()
    dialog._symmetry.setChecked(False)
    assert not dialog._cell.isEnabled()
    assert dialog._preview.toPlainText().splitlines()[3] == "1"   # P1
    dialog._symmetry.setChecked(True)
    assert dialog._cell.isEnabled()
