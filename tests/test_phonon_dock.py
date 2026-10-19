"""The Phonons panel comes and goes with the file's frequencies.

A geometry — an input file, an optimisation, a structure built in the window —
has no modes, and the panel is a list of them with the controls that animate
one. Left on screen it lists nothing, which reads as a frequency run whose
modes could not be read; that is a different thing and is said in words.

The dock is shared by every tab (one dock, a stack of panels inside it), so
what it follows is the tab on screen, as the Plots window does.
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
    QTabBar,
    QTabWidget,
    QWidget,
)

from crystalline.ui.main_window import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _StubPanel:
    """The one question the window asks the panel."""

    def __init__(self, modes: bool) -> None:
        self._modes = modes

    def has_modes(self) -> bool:
        return self._modes


class _StubTab:
    def __init__(self, modes: bool = True, built: bool = True) -> None:
        self.phonon_panel = _StubPanel(modes)
        self._built = built

    def built(self) -> bool:
        return self._built


class _StubWindow(QMainWindow):
    """Only the dock and the tab on screen, so no VTK is needed."""

    def __init__(self, tab=None) -> None:
        super().__init__()
        self._tab = tab
        self._phonon_dock = QDockWidget("Phonons", self)
        self.addDockWidget(Qt.RightDockWidgetArea, self._phonon_dock)

    _phonons_available = MainWindow._phonons_available
    _sync_phonon_dock = MainWindow._sync_phonon_dock
    _panel_available = MainWindow._panel_available


def test_a_frequency_run_gets_the_panel(qapp):
    window = _StubWindow(_StubTab(modes=True))

    window._sync_phonon_dock()

    assert not window._phonon_dock.isHidden()


def test_a_geometry_does_not(qapp):
    """What this is all for: an optimisation or an input file has no modes."""
    window = _StubWindow(_StubTab(modes=False))

    window._sync_phonon_dock()

    assert window._phonon_dock.isHidden()


def test_the_panel_follows_the_tab_on_screen(qapp):
    """One dock serves every tab, so switching from a frequency run to a
    geometry has to put it away again — and back on the way there."""
    window = _StubWindow(_StubTab(modes=True))
    window._sync_phonon_dock()
    assert not window._phonon_dock.isHidden()

    window._tab = _StubTab(modes=False)
    window._sync_phonon_dock()
    assert window._phonon_dock.isHidden()

    window._tab = _StubTab(modes=True)
    window._sync_phonon_dock()
    assert not window._phonon_dock.isHidden()


@pytest.mark.parametrize(
    "tab, why",
    [
        (None, "the window opens on no tab at all"),
        (_StubTab(modes=True, built=False), "a tab read in behind has no panel yet"),
    ],
)
def test_a_tab_with_nothing_in_it_yet_is_not_asked_for_modes(qapp, tab, why):
    """A tab opened behind the one being worked in is not built until it is
    looked at, so its panel does not exist to be asked."""
    window = _StubWindow(tab)

    window._sync_phonon_dock()

    assert window._phonon_dock.isHidden(), why


def test_only_the_phonons_panel_can_be_missing(qapp):
    """The others always have something to show — a structure, its symmetry,
    the display settings — so only this one's menu entry is ever greyed."""
    window = _StubWindow(_StubTab(modes=False))

    assert not window._panel_available("Phonons")
    for title in ("Info", "Display", "Geometry", "Point symmetry", "Plots"):
        assert window._panel_available(title), title


# ── named like every other panel ──────────────────────────────────────────
class _StubDockWindow(QMainWindow):
    """The two docks of the right-hand area, built the window's own way."""

    def __init__(self) -> None:
        super().__init__()
        self._phonon_dock = self._dock("Phonons", QWidget(self), Qt.RightDockWidgetArea)
        self._symmetry_dock = self._dock(
            "Point symmetry", QWidget(self), Qt.RightDockWidgetArea
        )
        self.tabifyDockWidget(self._phonon_dock, self._symmetry_dock)
        self._symmetry_dock.hide()   # until Cell ▸ Point symmetry analysis asks

    _dock = MainWindow._dock
    _settle_docks = MainWindow._settle_docks

    def _panel_docks(self) -> list:
        return [("Phonons", self._phonon_dock),
                ("Point symmetry", self._symmetry_dock)]

    def tabs(self) -> list:
        """Every dock tab on screen, as the user reads them."""
        return [bar.tabText(i)
                for bar in self.findChildren(QTabBar) if bar.isVisible()
                for i in range(bar.count())]


def test_the_panel_is_named_by_a_tab_below_it_like_the_others(qapp):
    """A dock that is away is not tabified with anything as far as Qt is
    concerned, so putting this one away before the docks are settled left it
    its own title bar — and it came back named along the top, beside the file
    tabs, while every other panel is named by a tab along the bottom.
    """
    window = _StubDockWindow()
    window._settle_docks()
    window._phonon_dock.hide()        # the window opens with no modes
    window.show()
    qapp.processEvents()
    assert window.tabs() == []

    window._phonon_dock.show()        # a frequency run is opened
    qapp.processEvents()

    assert window.tabs() == ["Phonons"]
    assert window._phonon_dock.titleBarWidget() is not None, "named twice: tab and title bar"


def test_point_symmetry_still_tabs_alongside_it(qapp):
    """The two share the right-hand area, and the panel coming and going must
    not cost them that."""
    window = _StubDockWindow()
    window._settle_docks()
    window._phonon_dock.hide()
    window.show()

    window._phonon_dock.show()
    window._symmetry_dock.show()
    qapp.processEvents()

    assert window.tabs() == ["Phonons", "Point symmetry"]
