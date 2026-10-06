"""The tab bar whose tabs slide as a whole on a trackpad swipe."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QInputDevice, QPointingDevice, QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QTabBar, QTabWidget, QWidget  # noqa: E402

from crystalline.ui.widgets import SlidingTabBar  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _crowded_tabs():
    """More closable tabs than the bar has room for, so it scrolls."""
    tabs = QTabWidget()
    tabs.setTabBar(SlidingTabBar(tabs))
    tabs.setTabsClosable(True)
    tabs.setUsesScrollButtons(True)
    for n in range(12):
        tabs.addTab(QWidget(), f"run_{n:02d}.out")
    tabs.resize(320, 120)
    tabs.show()
    QApplication.processEvents()
    return tabs


def _swipe(bar, dx: int, times: int = 6):
    """A trackpad's pixel scroll across the bar, as macOS sends one."""
    pad = QPointingDevice("trackpad", 7, QInputDevice.DeviceType.TouchPad,
                          QPointingDevice.PointerType.Finger,
                          QInputDevice.Capability.Position | QInputDevice.Capability.PixelScroll,
                          2, 0)
    for _ in range(times):
        pos = QPointF(bar.width() / 2, bar.height() / 2)
        event = QWheelEvent(pos, QPointF(bar.mapToGlobal(pos.toPoint())), QPoint(dx, 0),
                            QPoint(2 * dx, 0), Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate,
                            False, Qt.MouseEventNotSynthesized, pad)
        QApplication.sendEvent(bar, event)
        QApplication.processEvents()


def _where_the_buttons_sit(bar):
    """Each close button's place on its own tab."""
    side = QTabBar.ButtonPosition.RightSide
    return [bar.tabButton(i, side).pos() - bar.tabRect(i).topLeft() for i in range(bar.count())]


def test_a_swipe_slides_the_close_buttons_with_their_tabs(qapp):
    """The names used to slide along under close buttons that stayed put, so
    each × ended up beside another file's name."""
    tabs = _crowded_tabs()
    bar = tabs.tabBar()
    assert isinstance(bar, SlidingTabBar)
    before = _where_the_buttons_sit(bar)
    first = bar.tabRect(0).left()

    _swipe(bar, -14)

    assert bar.tabRect(0).left() < first                     # the bar did scroll
    assert _where_the_buttons_sit(bar) == before             # and every × went with its tab
    assert bar.currentIndex() == 0                           # a swipe scrolls, it does not switch
    tabs.close()


def test_the_file_tabs_and_the_plot_tabs_slide_as_a_whole(qapp):
    import inspect

    from crystalline.ui.main_window import MainWindow
    from crystalline.ui.panels.plot_view import PlotPanel

    panel = PlotPanel()
    assert isinstance(panel._tabs.tabBar(), SlidingTabBar)
    assert "setTabBar(SlidingTabBar(" in inspect.getsource(MainWindow.__init__)


def test_a_crowded_tab_keeps_a_readable_part_of_its_name(qapp):
    """Qt squeezes every name to three characters — "t…t" — before scrolling;
    a tab here keeps about a dozen, and a name shorter than that stays whole."""
    from crystalline.ui.widgets.tab_bar import READABLE_CHARACTERS

    long_name = "thiourea_pbed3_Ahlrichs-pVTZ_freqcalc_80K.out"
    ours, plain = SlidingTabBar(), QTabBar()
    for bar in (ours, plain):
        bar.setElideMode(Qt.ElideMiddle)
        bar.addTab(long_name)
        bar.addTab("mgo.out")

    qt_minimum = plain.minimumTabSizeHint(0).width()
    ours_minimum = ours.minimumTabSizeHint(0).width()
    room = ours.fontMetrics().averageCharWidth() * (READABLE_CHARACTERS - 3)
    assert ours_minimum == qt_minimum + room                 # a dozen characters, not three
    assert ours_minimum < ours.tabSizeHint(0).width()        # still elided: the name is long
    assert ours.minimumTabSizeHint(1).width() == ours.tabSizeHint(1).width()   # short: whole

    ours.setElideMode(Qt.ElideNone)                          # not eliding: Qt's own answer
    assert ours.minimumTabSizeHint(0) == QTabBar.minimumTabSizeHint(ours, 0)


def test_a_crowded_bar_lays_its_tabs_out_that_wide(qapp):
    tabs = _crowded_tabs()
    tabs.setElideMode(Qt.ElideMiddle)
    for index in range(tabs.count()):
        tabs.setTabText(index, f"thiourea_pbed3_freqcalc_{index:02d}.out")
    QApplication.processEvents()
    bar = tabs.tabBar()
    assert bar.tabRect(0).width() >= bar.minimumTabSizeHint(0).width()
    assert bar.tabRect(0).width() < bar.tabSizeHint(0).width()   # elided, and the bar scrolls
    tabs.close()
