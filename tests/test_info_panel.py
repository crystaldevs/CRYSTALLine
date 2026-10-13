"""The Info panel's CRYSTAL-output rows, which arrive a moment after the structure."""

import pytest

pytest.importorskip("PySide6")

from ase.build import bulk  # noqa: E402
from PySide6.QtWidgets import QApplication, QFormLayout  # noqa: E402

from crystalline.core.structure import Structure  # noqa: E402
from crystalline.ui.panels.info_panel import InfoPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _output_rows(panel):
    """The CRYSTAL-output group's rows as (label, value), or None while it is hidden."""
    if panel._output_group.isHidden():
        return None
    form = panel._output_form
    rows = []
    for row in range(form.rowCount()):
        label = form.itemAt(row, QFormLayout.LabelRole)
        value = form.itemAt(row, QFormLayout.FieldRole)
        rows.append((label.widget().text() if label else "", value.widget().text()))
    return rows


def _rock_salt():
    return Structure.from_ase(bulk("MgO", "rocksalt", a=4.21))


def test_rows_still_to_read_are_said_to_be_coming(qapp):
    """Shown with the structure before the rows about the run are read, the
    group says they are on their way rather than being missing."""
    panel = InfoPanel()
    panel.set_output_props({}, reading=True)
    panel.show_structure(_rock_salt(), {})
    assert _output_rows(panel) == [("", "Reading the output…")]

    panel.set_output_props({"Code": "CRYSTAL23", "Task": "FREQCALC"})
    assert _output_rows(panel) == [("Code:", "CRYSTAL23"), ("Task:", "FREQCALC")]


def test_an_edit_while_they_are_read_keeps_saying_so(qapp):
    """An edit redraws the panel from the structure with the rows it has — none
    yet — and the group must not vanish before they arrive."""
    panel = InfoPanel()
    panel.set_output_props({}, reading=True)
    panel.show_structure(_rock_salt(), {})
    panel.show_structure(_rock_salt(), {})              # as _refresh_info does after an edit
    assert _output_rows(panel) == [("", "Reading the output…")]


def test_an_output_with_no_rows_shows_no_group(qapp):
    """Read, and nothing to show: the group goes, as for a geometry file."""
    panel = InfoPanel()
    panel.set_output_props({}, reading=True)
    panel.show_structure(_rock_salt(), {})
    panel.set_output_props({})
    assert _output_rows(panel) is None


def test_the_rows_alone_do_not_analyse_the_structure_again(qapp, monkeypatch):
    """They describe the run, not the geometry: the crystallography stands."""
    from crystalline.ui.panels import info_panel

    panel = InfoPanel()
    panel.show_structure(_rock_salt(), {})
    monkeypatch.setattr(info_panel, "analyze", lambda *a, **k: pytest.fail("analysed again"))
    panel.set_output_props({"Code": "CRYSTAL23"})
    assert _output_rows(panel) == [("Code:", "CRYSTAL23")]
