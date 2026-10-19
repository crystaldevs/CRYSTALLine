"""The two things drawn over the 3D view: the drop hint and the busy scrim.

Both are windows of their own, so that they can be drawn over the 3D views,
which are too. And on macOS a hidden window is not a gone one: it kept the last
picture painted into it, and that picture reached the screen again for a frame
each time a tab was closed — "Open 13 files" over the 3D view, long after the
files had been dropped. So an overlay taken down swaps its window for an empty
one — on the next turn of the loop, since the hint is taken down from inside the
drag session, which holds the very views it is dispatching to.
"""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
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


def test_a_drop_hint_taken_down_leaves_no_picture_behind(host, qapp):
    from crystalline.ui.widgets import DropHint

    hint = DropHint(host)
    hint.show_hint("Open 13 files", "each in a new tab")
    assert hint.isVisible() and hint.internalWinId()
    painted = hint.internalWinId()

    hint.hide_hint()                                     # the drop, or the drag leaving
    assert not hint.isVisible()
    assert hint.internalWinId() == painted               # still there: the drag is dispatching

    qapp.processEvents()                                 # the drag over, the window goes
    assert hint.internalWinId() and hint.internalWinId() != painted   # an empty one in its place

    hint.show_hint("Open 2 files", "each in a new tab")  # the next drag
    assert hint.isVisible() and hint.internalWinId()     # a window again: over the views


def test_the_busy_scrim_taken_down_leaves_no_picture_behind(host, qapp):
    from crystalline.ui.widgets import BusyOverlay

    busy = BusyOverlay(host)
    seen = set()
    for _ in range(2):                                   # and comes back each time
        busy.start("Building the orbital…")
        assert busy.isVisible() and busy.internalWinId()
        seen.add(int(busy.internalWinId()))
        busy.stop()
        qapp.processEvents()
        assert not busy.isVisible()
    assert len(seen) == 2                                # never the window it painted into


def test_an_overlay_is_not_pulled_out_from_under_a_drag(host, qapp):
    """Taken down inside dragLeaveEvent and dropEvent, which the platform runs
    in the middle of its drag session — and the session holds the native views
    it is dispatching to. Destroying one there took the process with it."""
    from crystalline.ui.widgets import DropHint

    hint = DropHint(host)
    hint.show_hint("Open 13 files", "each in a new tab")
    window = hint.internalWinId()

    hint.hide_hint()

    assert hint.internalWinId() == window                # nothing destroyed in the handler
    qapp.processEvents()
    assert hint.internalWinId() != window                # and done once the drag is over


def test_an_overlay_has_its_window_before_anything_puts_it_up(host, qapp):
    """The first thing to put the drop hint up is a drag, and the window is
    better not made inside the session either."""
    from crystalline.ui.widgets import DropHint

    hint = DropHint(host)
    host.resize(host.width(), host.height() - 1)          # the parent moves: it is watched
    qapp.processEvents()

    assert hint.internalWinId()                           # ready, and nothing dragging
    assert not hint.isVisible()                           # but not up


def test_an_overlay_asks_for_a_window_of_its_own(host):
    """It used to become one by accident — Qt makes every sibling native when the
    first VTK view asks for a window — and a window is made again on showing only
    for a widget that asks. Without it, an overlay shown after being taken down
    would be drawn into its parent, underneath the 3D views."""
    from crystalline.ui.widgets import BusyOverlay, DropHint

    for overlay in (DropHint(host), BusyOverlay(host)):
        assert overlay.testAttribute(Qt.WA_NativeWindow)


# ── what shows under the tint ────────────────────────────────────────────
class _Colour:
    def __init__(self, rgb):
        self.int_rgb = rgb


class _View(QWidget):
    """Stands for a 3D view: it paints itself on screen, so Qt holds no
    picture of it — which is what left stale ones under the overlays."""

    def __init__(self, parent, rgb):
        super().__init__(parent)
        self.setAttribute(Qt.WA_PaintOnScreen, True)
        self.background_color = _Colour(rgb)


def _hint_over(host, rgb, qapp):
    from crystalline.ui.widgets import DropHint

    view = _View(host, rgb)
    view.setGeometry(0, 0, host.width(), host.height())
    view.show()
    hint = DropHint(host)
    hint.show_hint("Open beta-PVDF_known.cif", "in a new tab")
    qapp.processEvents()
    return hint, view


def _corner_pixel(hint):
    """A pixel inside the hint, clear of its frame and its card."""
    return hint.grab().toImage().pixelColor(4, 4)


def test_the_drop_hint_paints_the_view_s_background_under_its_tint(host, qapp):
    """Inside the hint used to be whatever Qt last held for that spot: an
    earlier hint's card, a panel that stood there before the view widened.
    Painted from the view's own background now, so the tint lies on that."""
    red_hint, red_view = _hint_over(host, (200, 30, 30), qapp)
    red = _corner_pixel(red_hint)
    red_hint.hide_hint()
    red_view.deleteLater()

    blue_hint, _blue_view = _hint_over(host, (30, 30, 200), qapp)
    blue = _corner_pixel(blue_hint)
    blue_hint.hide_hint()

    # Each follows its view's background, under the same faint tint.
    assert red.red() > 150 and red.blue() < 100
    assert blue.blue() > 150 and blue.red() < 100


def test_only_what_paints_itself_is_painted_for(host, qapp):
    """Ordinary widgets are in Qt's picture already, current; and a 3D view
    in another window is not under this overlay at all."""
    from crystalline.ui.widgets import DropHint

    QWidget(host).setGeometry(0, 0, 100, 100)                # an ordinary widget
    elsewhere = QWidget()
    _View(elsewhere, (0, 255, 0)).setGeometry(0, 0, 100, 100)
    elsewhere.show()
    hint = DropHint(host)
    hint.show_hint("Open x")
    qapp.processEvents()

    assert hint._views_under() == []
    hint.hide_hint()
    elsewhere.close()


def test_the_busy_scrim_paints_the_view_s_background_too(host, qapp):
    from crystalline.ui.widgets import BusyOverlay

    view = _View(host, (200, 30, 30))
    view.setGeometry(0, 0, host.width(), host.height())
    view.show()
    scrim = BusyOverlay(host)
    scrim.start("Reading…")
    qapp.processEvents()

    pixel = scrim.grab().toImage().pixelColor(4, 4)
    scrim.stop()

    assert pixel.red() > pixel.blue() + 40    # the red view, under the scrim
