"""The h k l boxes a plane is named with, shared by the density slice and the
Geometry panel's lattice planes — one widget, so a plane is asked for the same
way in both."""

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from crystalline.ui.widgets.miller import MILLER_RANGE, MillerIndices  # noqa: E402

_HEXAGONAL = np.array([[3.25, 0.0, 0.0], [-1.625, 2.8146, 0.0], [0.0, 0.0, 5.21]])


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_three_boxes_that_say_which_index_is_which(qapp):
    widget = MillerIndices((1, 0, 2))
    assert widget.indices() == (1, 0, 2)
    assert [box.prefix() for box in widget.boxes] == ["", "", ""]
    shown = [label.text() for label in widget.findChildren(QLabel) if not label.isHidden()]
    assert sorted(shown) == ["h", "k", "l"]
    assert all(box.minimum() == -MILLER_RANGE and box.maximum() == MILLER_RANGE
               for box in widget.boxes)


def test_hexagonal_axes_show_the_fourth_index(qapp):
    widget = MillerIndices((1, 0, 0))
    assert widget.i_label.isHidden()
    widget.set_cell(_HEXAGONAL)
    assert widget.hexagonal and not widget.i_label.isHidden()
    assert not widget.i_name.isHidden() and widget.i_name.text() == "i"
    assert widget.i_label.text() == "-1"
    widget.boxes[1].setValue(1)
    assert widget.i_label.text() == "-2"
    widget.set_cell(4.0 * np.eye(3))
    assert widget.i_label.isHidden() and widget.i_name.isHidden()


def test_any_edit_is_announced_once(qapp):
    widget = MillerIndices()
    seen = []
    widget.changed.connect(lambda: seen.append(widget.indices()))
    widget.boxes[0].setValue(2)
    assert seen == [(2, 0, 1)]


def test_the_density_dialog_and_the_geometry_panel_share_it(qapp):
    from crystalline.core.structure import Structure
    from crystalline.ui.panels.density_dialog import DensityDialog
    from crystalline.ui.panels.geometry_panel import GeometryPanel

    dialog = DensityDialog(miller_cell=_HEXAGONAL)
    assert isinstance(dialog.miller[0].parent(), MillerIndices)
    assert dialog.miller_h is dialog.miller[0]            # remembered settings still find them
    panel = GeometryPanel(Structure.empty())
    assert isinstance(panel._miller, MillerIndices)
