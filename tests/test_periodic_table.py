"""The visual periodic-table element picker (needs PySide6)."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QPushButton  # noqa: E402
from ase.data import chemical_symbols  # noqa: E402

from crystalline.ui.periodic_table import (  # noqa: E402
    _F_ROWS,
    _MAIN_ROWS,
    PeriodicTableDialog,
    _category,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_table_places_every_element_exactly_once():
    placed = [s for row in _MAIN_ROWS for s in row if s]
    placed += [s for row in _F_ROWS for s in row]
    assert len(placed) == len(set(placed)) == 118  # H..Og, no gaps or duplicates
    assert set(placed) == set(chemical_symbols[1:119])


def test_every_element_gets_a_known_category():
    for symbol in chemical_symbols[1:119]:
        assert _category(symbol)  # never raises / empty; unknowns fall back to d-block


def test_dialog_builds_a_button_per_element(qapp):
    dialog = PeriodicTableDialog(current="Fe")
    assert len(dialog.findChildren(QPushButton)) == 118


def test_choosing_an_element_emits_and_accepts(qapp):
    dialog = PeriodicTableDialog()
    received = []
    dialog.element_selected.connect(received.append)

    dialog._choose("Zr")

    assert dialog.selected == "Zr"
    assert received == ["Zr"]
    assert dialog.result() == PeriodicTableDialog.Accepted  # picking closes the dialog
