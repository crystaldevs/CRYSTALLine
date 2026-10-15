"""The tab bar of fixed-width tabs, which slide as a whole on a trackpad swipe."""

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


def _bar(*names, elide=Qt.ElideMiddle):
    bar = SlidingTabBar()
    bar.setElideMode(elide)
    bar.setTabsClosable(True)
    for name in names:
        bar.addTab(name)
    return bar


_NAMES = ("mgo.out", "thiourea_pbed3_Ahlrichs-pVTZ_freqcalc_80K.out", "H2O-Ice-Ih.cif")


def test_every_tab_is_as_wide_whatever_its_name(qapp):
    """As in Firefox. Sized to its name, a tab moved whenever the bar's width
    changed — the file tabs reflowed each time the Phonons panel came or went,
    and the tab just clicked slid out from under the pointer."""
    from crystalline.ui.widgets.tab_bar import TAB_CHARACTERS

    bar = _bar(*_NAMES)

    widths = {bar.tabSizeHint(i).width() for i in range(bar.count())}
    assert len(widths) == 1, widths
    # room for the same number of characters, whatever the name
    plain = QTabBar()
    plain.setTabsClosable(True)
    plain.addTab("")
    assert widths.pop() == (plain.tabSizeHint(0).width()
                            + bar.fontMetrics().averageCharWidth() * TAB_CHARACTERS)


def test_the_row_stays_put_while_the_bar_narrows_around_it(qapp):
    """What fixed widths are for: the same tabs at the same places, as long as
    the narrower bar still holds them."""
    tabs = QTabWidget()
    tabs.setTabBar(SlidingTabBar(tabs))
    tabs.setTabsClosable(True)
    tabs.setElideMode(Qt.ElideMiddle)
    tabs.setUsesScrollButtons(True)
    for name in _NAMES:
        tabs.addTab(QWidget(), name)
    bar = tabs.tabBar()
    row = bar.tabSizeHint(0).width() * bar.count()

    def places(width):
        tabs.resize(width, 120)
        tabs.show()
        QApplication.processEvents()
        return [bar.tabRect(i) for i in range(bar.count())]

    wide = places(row * 2)
    assert places(row + 20) == wide                     # narrower, still fits: nothing moves
    tabs.close()


def test_a_crowded_tab_keeps_a_readable_part_of_its_name(qapp):
    """A row that does not fit shrinks before it scrolls, as Firefox's does —
    but not to Qt's three characters, "t…t": a dozen tells two runs apart."""
    from crystalline.ui.widgets.tab_bar import READABLE_CHARACTERS, TAB_CHARACTERS

    bar = _bar(*_NAMES)

    room = bar.fontMetrics().averageCharWidth() * (TAB_CHARACTERS - READABLE_CHARACTERS)
    for index in range(bar.count()):
        assert bar.minimumTabSizeHint(index).width() == bar.tabSizeHint(index).width() - room
    assert len({bar.minimumTabSizeHint(i).width() for i in range(bar.count())}) == 1


def test_a_bar_that_does_not_elide_keeps_every_name_whole(qapp):
    """Not eliding is asking for whole names, and a fixed width would cut them
    off instead: such a bar is sized as Qt sizes it."""
    ours, plain = _bar(*_NAMES, elide=Qt.ElideNone), QTabBar()
    plain.setElideMode(Qt.ElideNone)
    plain.setTabsClosable(True)
    for name in _NAMES:
        plain.addTab(name)

    for index in range(len(_NAMES)):
        assert ours.tabSizeHint(index) == plain.tabSizeHint(index)
        assert ours.minimumTabSizeHint(index) == plain.minimumTabSizeHint(index)


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
