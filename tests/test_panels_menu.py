"""Reopening panels closed with their × button.

A dock closes with one click and, without the View → Panels entries, stays
closed for the rest of the session — the panel is simply gone. Every dock is
registered so it can always be brought back.
"""

import os

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QDockWidget,
    QMainWindow,
    QWidget,
)

from crystalline.ui import menus  # noqa: E402
from crystalline.ui.main_window import MainWindow  # noqa: E402

_DOCKS = (
    ("_info_dock", Qt.LeftDockWidgetArea),
    ("_display_dock", Qt.LeftDockWidgetArea),
    ("_geometry_dock", Qt.LeftDockWidgetArea),
    ("_symmetry_dock", Qt.LeftDockWidgetArea),
    ("_phonon_dock", Qt.RightDockWidgetArea),
    ("_plot_dock", Qt.BottomDockWidgetArea),
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _StubWindow(QMainWindow):
    """The dock bookkeeping the View menu needs, without VTK.

    The docks are built with the window's own ``_dock``, not with a bare
    ``QDockWidget``: the features it sets are the whole reason this menu needs
    care. Built loosely here, this harness passed while every entry in the real
    menu was greyed out.
    """

    def __init__(self) -> None:
        super().__init__()
        self._axis_actions = []
        self.viewport = None
        for name, area in _DOCKS:
            setattr(self, name, self._dock(name, QWidget(self), area))
        self._plot_dock.hide()  # as the real window starts
        # Whether the file on screen has frequencies. The Phonons panel is the
        # one that can have nothing to show (see tests/test_phonon_dock.py);
        # the real window answers this from the tab on screen.
        self.phonons = True

    _dock = MainWindow._dock
    _panel_docks = MainWindow._panel_docks
    _panel_available = MainWindow._panel_available
    _restore_all_panels = MainWindow._restore_all_panels
    _show_display_panel = MainWindow._show_display_panel

    def _phonons_available(self) -> bool:
        return self.phonons

    # The View menu also steps between file tabs; there are none here.
    def _next_tab(self) -> None:
        pass

    def _previous_tab(self) -> None:
        pass


def _window(qapp):
    window = _StubWindow()
    menus._build_view_menu(window)
    return window


def test_every_panel_has_a_menu_entry(qapp):
    window = _window(qapp)

    assert list(window._panel_actions) == [
        "Info", "Display", "Geometry", "Point symmetry", "Phonons", "Plots"
    ]
    assert all(action.isCheckable() for action in window._panel_actions.values())


def test_a_panel_can_be_put_away_and_brought_back(qapp):
    """What the entries are for. The docks carry no × of their own, so if these
    do not work a panel cannot be put away at all — which is what happened:
    Qt disables ``toggleViewAction`` for a dock that is not closable, so every
    entry here was greyed out."""
    window = _window(qapp)
    action = window._panel_actions["Geometry"]

    assert action.isEnabled(), "the entry is greyed out; no panel can be hidden"
    assert action.isChecked()

    action.trigger()                       # as clicking a ticked entry does
    assert not action.isChecked()
    assert window._geometry_dock.isHidden()

    action.trigger()                       # and clicking it again brings it back
    assert action.isChecked()
    assert not window._geometry_dock.isHidden()


def test_the_panels_have_no_close_button_of_their_own(qapp):
    """The menu is the only route, deliberately: a dock shut by accident or
    dragged out of place was work to get back."""
    window = _window(qapp)

    for _title, dock in window._panel_docks():
        assert dock.features() == QDockWidget.NoDockWidgetFeatures


def test_the_ticks_catch_up_with_panels_shown_from_elsewhere(qapp):
    """A plot arriving reveals the Plots panel, and Restore all brings
    everything back. The entries are brought up to date when the menu opens,
    not tracked signal by signal — a tabbed dock reports itself invisible
    whenever another tab is selected, which would untick a panel that is
    perfectly well there."""
    window = _window(qapp)
    assert not window._panel_actions["Plots"].isChecked()  # starts hidden

    window._plot_dock.show()
    window._info_dock.hide()
    menus.sync_panel_actions(window)

    assert window._panel_actions["Plots"].isChecked()
    assert not window._panel_actions["Info"].isChecked()


def test_restore_all_brings_back_every_closed_panel(qapp):
    window = _window(qapp)
    for name in ("_info_dock", "_phonon_dock", "_plot_dock"):
        getattr(window, name).close()
    assert [t for t, d in window._panel_docks() if d.isHidden()] == ["Info", "Phonons", "Plots"]

    window._restore_all_panels()

    assert [t for t, d in window._panel_docks() if d.isHidden()] == []


def test_restore_all_redocks_panels_that_were_floating(qapp):
    """A dock dragged out and then closed would otherwise come back floating,
    possibly off-screen if the window has moved since."""
    window = _window(qapp)
    window._phonon_dock.setFloating(True)
    window._phonon_dock.close()

    window._restore_all_panels()

    assert not window._phonon_dock.isFloating()
    assert not window._phonon_dock.isHidden()


def test_restore_all_leaves_open_panels_where_they_are(qapp):
    """Someone who deliberately floated a panel keeps it floating."""
    window = _window(qapp)
    window._display_dock.setFloating(True)

    window._restore_all_panels()

    assert window._display_dock.isFloating()


def test_a_hidden_title_bar_never_asks_for_a_negative_height(qapp):
    """Each dock holds a stack of the open tabs' panels, empty until the first
    tab exists. A bare QWidget as the title bar measures (-1, -1), and the dock
    added that -1 to its empty content's height: a minimum size of (0, -1),
    which Qt warns about on macOS ("Negative sizes (0,-1) are not possible")."""
    from PySide6.QtWidgets import QStackedWidget

    from crystalline.ui.main_window import _no_title_bar

    window = QMainWindow()
    dock = QDockWidget("Info", window)
    dock.setWidget(QStackedWidget())            # no panel in it yet
    window.addDockWidget(Qt.LeftDockWidgetArea, dock)
    dock.setTitleBarWidget(_no_title_bar(dock))

    assert dock.titleBarWidget().sizeHint().height() == 0
    minimum = dock.layout().minimumSize()
    assert minimum.width() >= 0 and minimum.height() >= 0


# ── a panel the file has nothing to put in ────────────────────────────────
def test_the_phonons_entry_is_greyed_for_a_file_with_no_modes(qapp):
    """Ticking it would bring back a panel listing nothing. Greyed, it says
    instead that there is nothing to bring back."""
    window = _window(qapp)
    window.phonons = False

    menus.sync_panel_actions(window)

    assert not window._panel_actions["Phonons"].isEnabled()
    assert all(window._panel_actions[title].isEnabled()
               for title in ("Info", "Display", "Geometry", "Plots"))

    window.phonons = True
    menus.sync_panel_actions(window)
    assert window._panel_actions["Phonons"].isEnabled()


def test_restore_all_does_not_restore_a_panel_with_nothing_in_it(qapp):
    """"All panels" means the ones this file has: a geometry does not get an
    empty list of modes back."""
    window = _window(qapp)
    window.phonons = False
    for name in ("_info_dock", "_phonon_dock", "_plot_dock"):
        getattr(window, name).close()

    window._restore_all_panels()

    assert [t for t, d in window._panel_docks() if d.isHidden()] == ["Phonons"]


def test_control_tab_steps_through_the_tabs_on_macos(qapp, monkeypatch):
    """⌘Tab, Qt's own binding there, is the system's app switcher."""
    from PySide6.QtGui import QAction, QKeySequence

    monkeypatch.setattr(menus.sys, "platform", "darwin")
    window = _window(qapp)
    actions = {a.text(): a for a in window.findChildren(QAction)}

    next_keys = [k.toString() for k in actions["Next tab"].shortcuts()]
    previous_keys = [k.toString() for k in actions["Previous tab"].shortcuts()]
    assert "Meta+Tab" in next_keys
    assert "Meta+Shift+Backtab" in previous_keys        # what Shift makes of Tab
    for standard, keys in ((QKeySequence.NextChild, next_keys),
                           (QKeySequence.PreviousChild, previous_keys)):
        assert all(k.toString() in keys for k in QKeySequence.keyBindings(standard))


def test_other_platforms_keep_their_own_keys(qapp, monkeypatch):
    from PySide6.QtGui import QKeySequence

    monkeypatch.setattr(menus.sys, "platform", "linux")
    assert menus.tab_shortcuts(QKeySequence.NextChild) == \
        list(QKeySequence.keyBindings(QKeySequence.NextChild))
