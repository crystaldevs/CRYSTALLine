"""The Info panel's CRYSTAL-output rows, which arrive a moment after the structure."""

import pytest

pytest.importorskip("PySide6")

from ase.build import bulk  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from _panels import output_rows  # noqa: E402

from crystalline.core.structure import Structure  # noqa: E402
from crystalline.ui.panels.info_panel import InfoPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _rock_salt():
    return Structure.from_ase(bulk("MgO", "rocksalt", a=4.21))


def test_the_rows_alone_do_not_analyse_the_structure_again(qapp, monkeypatch):
    """They describe the run, not the geometry: the crystallography stands."""
    from crystalline.ui.panels import info_panel

    panel = InfoPanel()
    panel.show_structure(_rock_salt(), {})
    monkeypatch.setattr(info_panel, "analyze", lambda *a, **k: pytest.fail("analysed again"))
    panel.set_output_props({"Code": "CRYSTAL23"})
    assert output_rows(panel) == [("Code:", "CRYSTAL23")]
