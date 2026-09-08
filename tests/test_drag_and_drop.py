"""Dropping a file on the window: what it is taken for, and when it is acted on.

The window itself can't be built here — a real ``MainWindow`` needs a live
``QtInteractor``, which segfaults under the offscreen platform plugin — so the
handlers under test are bound onto a plain object, which is enough: they only
touch the event, the hint and the two loaders.
"""

import os

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QMimeData, QPoint, QPointF, QUrl, Qt  # noqa: E402
from PySide6.QtGui import QDragEnterEvent, QDropEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from crystalline.crystalio import file_action  # noqa: E402
from crystalline.ui.main_window import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# A drag event does not take ownership of its mime data, so a QMimeData built
# and dropped on the floor here is collected while the event still points at it
# — which segfaults on the first read, in the code under test rather than in the
# test. Qt owns it in the real thing; these are held for the session instead.
_MIME_KEEPALIVE = []


def _mime(*paths):
    data = QMimeData()
    data.setUrls([QUrl.fromLocalFile(p) for p in paths])
    _MIME_KEEPALIVE.append(data)
    return data


def _enter(paths):
    return QDragEnterEvent(QPoint(4, 4), Qt.CopyAction, _mime(*paths),
                           Qt.LeftButton, Qt.NoModifier)


def _drop(paths):
    return QDropEvent(QPointF(4, 4), Qt.CopyAction, _mime(*paths),
                      Qt.LeftButton, Qt.NoModifier)


class _Hint:
    def __init__(self):
        self.shown = None

    def show_hint(self, title, detail=""):
        self.shown = (title, detail)

    def hide_hint(self):
        self.shown = None


def _window():
    """A stand-in carrying the window's drag-and-drop handlers and nothing else."""

    class _Stub:
        dragEnterEvent = MainWindow.dragEnterEvent
        dragMoveEvent = MainWindow.dragMoveEvent
        dropEvent = MainWindow.dropEvent
        _handle_drop = MainWindow._handle_drop
        _dropped_file = MainWindow._dropped_file
        _dropped_paths = staticmethod(MainWindow._dropped_paths)

        def __init__(self):
            self._drop_hint = _Hint()
            self.opened = []
            self.imported = []
            self.messages = []

        def _load_path(self, path):
            self.opened.append(path)
            return True

        def _import_path(self, path):
            self.imported.append(path)
            return True

        def statusBar(self):
            outer = self

            class _Bar:
                @staticmethod
                def showMessage(text, _timeout=0):
                    outer.messages.append(text)

            return _Bar()

    return _Stub()


# ── what a file is taken for ──────────────────────────────────────────────
@pytest.mark.parametrize(
    "name, expected",
    [
        ("run.out", "open"),
        ("RUN.OUT", "open"),          # the check is case-insensitive
        ("geom.gui", "open"),
        ("fort.34", "open"),
        ("relaxed.34", "open"),       # a renamed fort.34 is still one
        ("cell.cif", "open"),         # a whole structure: an open, not an import
        ("fragment.xyz", "import"),
        ("protein.pdb", "import"),
        ("figure.png", None),
        ("orbitals.molden", None),    # readable, but not as a structure
        ("notes", None),
    ],
)
def test_a_file_is_taken_for_what_it_can_carry(name, expected):
    assert file_action(os.path.join("/tmp", name)) == expected


# ── the drag ──────────────────────────────────────────────────────────────
def test_dragging_a_usable_file_is_accepted_and_says_what_it_would_do(qapp):
    window = _window()

    event = _enter(["/tmp/run.out"])
    window.dragEnterEvent(event)
    assert event.isAccepted()
    title, detail = window._drop_hint.shown
    assert title == "Open run.out"
    assert "replaces" in detail

    event = _enter(["/tmp/fragment.xyz"])
    window.dragEnterEvent(event)
    title, detail = window._drop_hint.shown
    assert title == "Add the atoms in fragment.xyz"
    assert "undoable" in detail


def test_dragging_something_unusable_is_refused(qapp):
    window = _window()

    event = _enter(["/tmp/holiday.png"])
    window.dragEnterEvent(event)

    assert not event.isAccepted()
    assert window._drop_hint.shown is None


def test_a_drag_that_carries_no_files_at_all_is_refused(qapp):
    """Dragging selected text over the window must not look like a drop target."""
    window = _window()
    empty = QMimeData()
    _MIME_KEEPALIVE.append(empty)
    event = QDragEnterEvent(QPoint(4, 4), Qt.CopyAction, empty,
                            Qt.LeftButton, Qt.NoModifier)

    window.dragEnterEvent(event)

    assert not event.isAccepted()


def test_the_drag_is_re_accepted_on_every_move(qapp):
    """Qt asks again as the pointer moves; answering only dragEnterEvent leaves
    the drop refused whatever the enter said."""
    window = _window()

    usable = _drop(["/tmp/run.out"])
    window.dragMoveEvent(usable)
    assert usable.isAccepted()

    unusable = _drop(["/tmp/holiday.png"])
    window.dragMoveEvent(unusable)
    assert not unusable.isAccepted()


# ── the drop ──────────────────────────────────────────────────────────────
def test_the_drop_is_acted_on_after_the_drag_not_during_it(qapp):
    """The regression that segfaulted: opening a file tears the VTK scene down
    and builds another, and doing that inside the platform's drag session —
    which is what a drop on the 3D view is — crashes. The drop is accepted
    immediately and the file opened on the next turn of the event loop."""
    window = _window()
    event = _drop(["/tmp/run.out"])

    window.dropEvent(event)
    assert event.isAccepted()
    assert window.opened == []          # not yet: still inside the drag

    qapp.processEvents()
    assert window.opened == ["/tmp/run.out"]
    assert window._drop_hint.shown is None


def test_an_xyz_is_imported_rather_than_opened(qapp):
    window = _window()

    window.dropEvent(_drop(["/tmp/fragment.xyz"]))
    qapp.processEvents()

    assert window.imported == ["/tmp/fragment.xyz"]
    assert window.opened == []


def test_dropping_several_files_opens_one_and_says_so(qapp):
    """Opening is a replacement, so opening four in a row would leave three of
    them having flashed past. Silently ignoring them is worse than saying so:
    a multiple selection dragged in one gesture looks like it should all
    arrive."""
    window = _window()

    window.dropEvent(_drop(["/tmp/holiday.png", "/tmp/run.out", "/tmp/other.cif"]))
    qapp.processEvents()

    assert window.opened == ["/tmp/run.out"]      # the first usable one
    assert len(window.messages) == 1
    assert "2 other dropped files ignored" in window.messages[0]


def test_a_single_file_is_opened_without_comment(qapp):
    window = _window()

    window.dropEvent(_drop(["/tmp/run.out"]))
    qapp.processEvents()

    assert window.opened == ["/tmp/run.out"]
    assert window.messages == []


def test_dropping_something_unusable_does_nothing(qapp):
    window = _window()
    event = _drop(["/tmp/holiday.png"])

    window.dropEvent(event)
    qapp.processEvents()

    assert not event.isAccepted()
    assert window.opened == [] and window.imported == []


# ── the 3D view has to let the drop through ───────────────────────────────
def test_the_vtk_widget_is_not_a_drop_target():
    """Qt gives a drop to the innermost widget under the pointer that accepts
    drops. ``QtInteractor`` accepts them and reads whatever is dropped as a
    *mesh*, so without this a CRYSTAL output dropped on the 3D view — the
    obvious place to drop one — never reaches the window, and pyvistaqt tries
    to add it to the scene.

    Checked on ``__init__``'s own names rather than by building a Viewport,
    which needs a live QtInteractor.
    """
    from crystalline.ui.viewport import Viewport

    assert "setAcceptDrops" in Viewport.__init__.__code__.co_names


def test_the_window_accepts_drops_at_all():
    from crystalline.ui.main_window import MainWindow as MW

    assert "setAcceptDrops" in MW.__init__.__code__.co_names


# ── the hint ──────────────────────────────────────────────────────────────
def test_a_long_file_name_is_elided_to_fit(qapp):
    """A CRYSTAL output's name is routinely sixty characters of run parameters.
    Painted whole it ran out past the frame on both sides."""
    from PySide6.QtWidgets import QWidget

    from crystalline.ui.widgets import DropHint

    host = QWidget()
    host.resize(420, 280)
    hint = DropHint(host)
    long_name = "Open " + "brucite_pbe0-d3_pob_tight_10M_VSCF_VPT2_INTENS" * 2 + ".out"
    hint.show_hint(long_name, "replaces the structure on screen")

    hint.grab()  # paints; the elision happens there and must not raise
    assert hint.width() == host.width()


def test_the_hint_ignores_the_pointer(qapp):
    """It sits over the drop target. Taking the pointer would take the drag with
    it, and the drop would look like it had silently failed."""
    from PySide6.QtWidgets import QWidget

    from crystalline.ui.widgets import DropHint

    host = QWidget()          # held: a collected parent takes the hint with it
    hint = DropHint(host)

    assert hint.testAttribute(Qt.WA_TransparentForMouseEvents)
    assert not hint.acceptDrops()
