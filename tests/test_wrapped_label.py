"""A wrapped label has to count its own lines.

Qt's ``QLabel`` reports the height of one line as its minimum however many it
draws, so a dialog built from it can be made shorter than its own contents —
and the squeeze lands on whatever is below, which in the bands dialog was a
table painting across a checkbox.
"""

import os

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from crystalline.ui.panels.electronic_dialog import ElectronicDialog  # noqa: E402
from crystalline.ui.widgets.wrapped_label import WrappedLabel  # noqa: E402

_TEXT = ("Band and DOS files from a PROPERTIES run. CRYSTAL writes their energies "
         "relative to the Fermi level; choose absolute energies to move everything, "
         "the Fermi line included, up by E_F.")


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _shown(label, width):
    label.show()
    label.resize(width, 10)
    return label


def test_a_plain_label_asks_for_one_line_however_many_it_draws(qapp):
    """The Qt behaviour this widget exists to correct — pinned so that a future
    Qt that fixes it is noticed rather than silently making this moot."""
    plain = _shown(QLabel(_TEXT), 300)
    plain.setWordWrap(True)

    assert plain.heightForWidth(300) > plain.minimumSizeHint().height()


def test_a_wrapped_label_asks_for_the_height_it_will_draw(qapp):
    label = _shown(WrappedLabel(_TEXT), 300)

    assert label.minimumSizeHint().height() == label.heightForWidth(300)


def test_it_asks_for_more_as_it_is_narrowed(qapp):
    label = _shown(WrappedLabel(_TEXT), 600)
    wide = label.minimumSizeHint().height()

    label.resize(200, 10)

    assert label.minimumSizeHint().height() > wide


def test_it_asks_for_nothing_unusual_before_it_is_shown(qapp):
    """Off screen its width is Qt's default, not the one it will be given, and
    answering for that opens every dialog with space under its note."""
    hidden, plain = WrappedLabel(_TEXT), QLabel(_TEXT)
    plain.setWordWrap(True)

    assert hidden.minimumSizeHint() == plain.minimumSizeHint()


def test_the_bands_dialog_cannot_be_shrunk_onto_itself(qapp):
    """At its smallest, the projections table used to be drawn over the
    checkbox below it: the dialog's minimum was one wrapped line short, the
    grid gave the table less than the 150 px it insists on, and ``setGeometry``
    clamps to that minimum rather than honouring the cell."""
    dialog = ElectronicDialog(None, "")
    dialog.show()
    dialog.resize(dialog.minimumSizeHint())
    qapp.processEvents()

    table = dialog._table.geometry()
    checkbox = dialog.overlay.geometry()

    assert table.bottom() < checkbox.top()
    dialog.close()
