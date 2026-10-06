"""The two things drawn over the 3D view: the drop hint and the busy scrim.

Both are windows of their own, so that they can be drawn over the 3D views,
which are too. And on macOS a hidden window is not a gone one: it kept the last
picture painted into it, and that picture reached the screen again for a frame
each time a tab was closed — "Open 13 files" over the 3D view, long after the
files had been dropped. So an overlay taken down is shrunk to a pixel before it
is hidden, leaving its window no room to show a picture in.

Shrunk, not destroyed: destroying the window was the first answer, and every
drop crashed the app. While files are dragged in, the hint is up under the
pointer, macOS takes its window for the drop target, and the hint is taken
down in the middle of the drag; with the window freed, the drag's next message
reached for it — a segfault in ``-[NSDragDestination _setTarget:]``.
"""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSize, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def host(qapp):
    widget = QWidget()
    widget.resize(420, 280)
    widget.show()
    qapp.processEvents()
    yield widget
    widget.hide()


def _native_size(widget) -> QSize:
    """The size of the overlay's own window, which is what reaches the screen."""
    return widget.windowHandle().geometry().size()


def test_a_drop_hint_taken_down_keeps_its_window_but_no_room_to_show_it(host):
    from crystalline.ui.widgets import DropHint

    hint = DropHint(host)
    hint.show_hint("Open 13 files", "each in a new tab")
    window = hint.internalWinId()
    assert hint.isVisible() and window
    assert _native_size(hint) == host.size()

    hint.hide_hint()                                     # the drop, or the drag leaving
    assert not hint.isVisible()
    assert hint.internalWinId() == window                # still there for the drag to reach
    assert _native_size(hint) == QSize(1, 1)             # with nowhere to show a picture

    hint.show_hint("Open 2 files", "each in a new tab")  # the next drag
    assert hint.isVisible() and _native_size(hint) == host.size()


def test_a_hidden_overlay_stays_shrunk_while_its_parent_resizes(host, qapp):
    """It follows its parent's size only while it is up: fitted again while
    hidden, it would have its old picture's room back."""
    from crystalline.ui.widgets import DropHint

    hint = DropHint(host)
    hint.show_hint("Open 13 files", "each in a new tab")
    hint.hide_hint()
    host.resize(500, 320)
    qapp.processEvents()
    assert _native_size(hint) == QSize(1, 1)
    hint.show_hint("Open 13 files", "each in a new tab")
    assert _native_size(hint) == QSize(500, 320)


def test_the_busy_scrim_taken_down_keeps_its_window_but_no_room_to_show_it(host):
    from crystalline.ui.widgets import BusyOverlay

    busy = BusyOverlay(host)
    for _ in range(2):                                   # and comes back each time
        busy.start("Building the orbital…")
        assert busy.isVisible() and _native_size(busy) == host.size()
        busy.stop()
        assert not busy.isVisible() and busy.internalWinId()
        assert _native_size(busy) == QSize(1, 1)


def test_an_overlay_asks_for_a_window_of_its_own(host):
    """It used to become one by accident — Qt makes every sibling native when the
    first VTK view asks for a window. Drawn into its parent instead, it would sit
    underneath the 3D views."""
    from crystalline.ui.widgets import BusyOverlay, DropHint

    for overlay in (DropHint(host), BusyOverlay(host)):
        assert overlay.testAttribute(Qt.WA_NativeWindow)


def test_an_overlay_is_never_destroyed_to_take_it_down():
    """Checked on the source, since the crash needs a real drag on macOS."""
    import inspect

    from crystalline.ui.widgets.overlay import ParentOverlay

    assert ".destroy(" not in inspect.getsource(ParentOverlay._take_down)
