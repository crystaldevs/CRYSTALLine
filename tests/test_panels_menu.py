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
from PySide6.QtWidgets import QApplication, QDockWidget, QMainWindow  # noqa: E402

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
    """The dock bookkeeping the View menu needs, without VTK."""

    def __init__(self) -> None:
        super().__init__()
        self._axis_actions = []
        self.viewport = None
        for name, area in _DOCKS:
            dock = QDockWidget(name, self)
            self.addDockWidget(area, dock)
            setattr(self, name, dock)
        self._plot_dock.hide()  # as the real window starts

    _panel_docks = MainWindow._panel_docks
    _restore_all_panels = MainWindow._restore_all_panels
    _show_display_panel = MainWindow._show_display_panel


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


def test_a_panel_closed_with_its_button_can_be_reopened(qapp):
    """The case this exists for: the × is easy to hit by accident."""
    window = _window(qapp)
    action = window._panel_actions["Geometry"]

    window._geometry_dock.close()
    assert window._geometry_dock.isHidden()
    assert not action.isChecked()  # the menu tracks the dock, however it was closed

    action.trigger()
    assert not window._geometry_dock.isHidden()
    assert action.isChecked()


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
