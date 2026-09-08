"""MainWindow wiring: menus/toolbars build, and every slot they connect exists.

Also guards ``_connect_signals`` — which runs *before* the menus are built — from
reaching for an attribute that doesn't exist yet.

``MainWindow`` itself can't be built headless (its VTK interactor needs a GL
context), so the menu code is exercised against a stub window instead — which is
exactly what the split into :mod:`crystalline.ui.menus` buys. A second, static
check then pins the stub to reality: every ``window.<attr>`` the menus touch must
be a real method of ``MainWindow`` or an attribute its ``__init__`` assigns.
"""

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QAction  # noqa: E402
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDockWidget, QMainWindow, QMenu  # noqa: E402

from crystalline.ui import menus  # noqa: E402

EXPECTED_MENUS = ["&File", "&Cell", "&Edit", "&View", "&Plot", "&Help"]


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _StubWindow(QMainWindow):
    """A MainWindow stand-in: every slot the menus connect to is a no-op."""

    def __init__(self) -> None:
        super().__init__()
        self._show_boundary = True
        self._axis_actions: list = []
        self.viewport = self  # only align_view_along/can_align_axes are used
        # The View menu builds a toggle per panel, so the docks must be real —
        # __getattr__'s no-op would return None where a list is iterated.
        self._docks = {}
        for title in ("Info", "Display", "Geometry", "Phonons", "Plots"):
            dock = QDockWidget(title, self)
            self.addDockWidget(Qt.LeftDockWidgetArea, dock)
            self._docks[title] = dock

    def _panel_docks(self):
        return list(self._docks.items())

    def align_view_along(self, axis: int) -> None:
        pass

    def can_align_axes(self) -> bool:
        return True

    def __getattr__(self, name: str):
        # QMainWindow.__getattr__ is only reached for genuinely missing names;
        # hand back a no-op so any slot the menus connect to resolves.
        if name.startswith("_"):
            return lambda *args, **kwargs: None
        raise AttributeError(name)


def test_menu_bar_is_built_with_every_menu_and_action(qapp):
    window = _StubWindow()
    menus.build_menus(window)

    titles = [m.title() for m in window.menuBar().findChildren(QMenu)]
    for expected in EXPECTED_MENUS:
        # the Plot menu renames itself when CRYSTALClear is missing
        assert any(t.startswith(expected) for t in titles), f"{expected} missing from {titles}"

    labels = {a.text() for a in window.findChildren(QAction) if a.text()}
    for expected in (
        "Open…",
        "Import atoms into structure…",
        "Save structure as .gui…",
        "Save structure as .cif…",
        "Export image…",
        "Undo",
        "Redo",
        "Editing mode",
        "Select all",
        "Delete selected",
        "Restore geometry",
        "Lattice parameters…",
        "Supercell…",
        "Point symmetry analysis",
        "Display settings",
        "About CRYSTALLine",
    ):
        assert expected in labels, f"{expected!r} missing from the menu bar"


def test_import_action_starts_disabled(qapp):
    """'Import atoms into structure' is off until a structure is loaded — it needs
    something to import *into*. ``_update_import_action`` enables it after a load."""
    window = _StubWindow()
    menus.build_menus(window)
    assert window._import_action.text() == "Import atoms into structure…"
    assert window._import_action.isEnabled() is False


def test_import_atoms_action_starts_disabled(qapp):
    """'Import atoms' needs a structure first, so the menus build it disabled;
    MainWindow re-enables it via _update_import_action once a file is loaded."""
    import inspect

    from crystalline.ui.main_window import MainWindow

    window = _StubWindow()
    menus.build_menus(window)
    assert window._import_action.text() == "Import atoms into structure…"
    assert not window._import_action.isEnabled()  # disabled until a structure loads

    assert callable(MainWindow._update_import_action)
    # ...on _load_path, which is the open sequence itself: _open_file is now only
    # the file dialog in front of it, so that a dropped file follows the same one.
    src = inspect.getsource(MainWindow._load_path)
    assert "_update_import_action" in src  # re-enabled on open


def test_actions_the_window_drives_later_are_stashed_on_it(qapp):
    """MainWindow enables/disables these by name after construction."""
    window = _StubWindow()
    menus.build_menus(window)

    for attr in (
        "_export_anim_action",
        "_lattice_action",
        "_supercell_action",
        "_boundary_action",
        "_undo_action",
        "_redo_action",
        "_edit_mode_action",
        "_edit_tool_actions",
        "_axis_buttons",
        "_plot_kinds",
        "_plot_actions",
    ):
        assert attr in vars(window), f"menus.build_menus did not set {attr}"
    assert len(window._axis_actions) == 3  # a/b/c view alignment
    assert len(window._axis_buttons) == 3


def test_rotate_buttons_sit_beside_the_axis_buttons(qapp):
    window = _StubWindow()
    menus.build_menus(window)

    # left / right / up / down + the two in-screen-plane (roll) buttons
    assert len(window._rotate_buttons) == 6
    assert [b.text() for b in window._rotate_buttons] == ["◀", "▶", "▲", "▼", "↺", "↻"]
    assert all(b.autoRepeat() for b in window._rotate_buttons)  # hold to keep turning

    # they orbit the view, and unlike a/b/c alignment they need no cell
    turned = []
    window.rotate_view = lambda az=0.0, el=0.0, roll=0.0: turned.append((az, el, roll))
    for button in window._rotate_buttons:
        button.click()
    assert turned == [
        (-15.0, 0.0, 0.0),
        (15.0, 0.0, 0.0),
        (0.0, 15.0, 0.0),
        (0.0, -15.0, 0.0),
        (0.0, 0.0, -15.0),  # anticlockwise in the screen plane
        (0.0, 0.0, 15.0),  # clockwise
    ]


def test_viewport_provides_the_rotation_the_toolbar_calls():
    """The buttons call viewport.rotate_view(azimuth, elevation, roll) — it must
    exist with that shape (the Viewport itself can't be built headless)."""
    import inspect

    from crystalline.ui.viewport import Viewport

    assert callable(Viewport.rotate_view)
    parameters = inspect.signature(Viewport.rotate_view).parameters
    assert list(parameters) == ["self", "azimuth", "elevation", "roll"]
    assert all(parameters[name].default == 0.0 for name in ("azimuth", "elevation", "roll"))


def test_roll_sign_makes_the_structure_turn_the_way_the_button_says():
    """``rotate_view(roll=+x)`` must turn the structure *clockwise* on screen.

    vtkCamera.Roll turns the view-up vector, so the scene goes the opposite way
    — hence the negation in Viewport.rotate_view. Pin the convention here: a
    sign flip is silent otherwise.
    """
    vtk = pytest.importorskip("vtk")

    camera = vtk.vtkCamera()
    camera.SetPosition(0, 0, 10)
    camera.SetFocalPoint(0, 0, 0)
    camera.SetViewUp(0, 1, 0)

    camera.Roll(-90)  # what rotate_view(roll=+90) does under the hood

    # Screen-up now points along world -x, so a feature that was at the top of
    # the screen (world +y) has moved to the right: a clockwise turn.
    assert np.allclose(camera.GetViewUp(), [-1, 0, 0], atol=1e-9)


def test_viewport_eventfilter_routes_the_delete_key(qapp):
    """Del/Backspace over the 3D view (which grabs focus after picking) must be
    turned into a ``delete_requested`` signal — the VTK widget otherwise swallows
    the plain key so the menu's ``Del`` shortcut never fires. The Viewport can't
    be built headless, so drive ``eventFilter`` directly (it returns before the
    QWidget ``super().eventFilter`` for handled keys)."""
    import types
    from unittest.mock import Mock

    from PySide6.QtCore import QEvent, Qt

    from crystalline.ui.viewport import Viewport

    interactor = object()
    for key in (Qt.Key_Delete, Qt.Key_Backspace):
        stub = types.SimpleNamespace(interactor=interactor, _drag=Mock(), delete_requested=Mock())
        event = types.SimpleNamespace(type=lambda: QEvent.KeyPress, key=lambda k=key: k)
        consumed = Viewport.eventFilter(stub, interactor, event)
        assert consumed is True  # event is eaten so VTK doesn't also act on it
        stub.delete_requested.emit.assert_called_once_with()


def test_viewport_eventfilter_nudges_selection_while_editing(qapp):
    """An arrow key over the 3D view in editing mode is turned into a
    ``nudge_requested`` signal and consumed (so it moves atoms, not the camera)."""
    import types
    from unittest.mock import Mock

    from PySide6.QtCore import QEvent, Qt

    from crystalline.ui.viewport import Viewport

    interactor = object()
    event = types.SimpleNamespace(
        type=lambda: QEvent.KeyPress, key=lambda: Qt.Key_Right, modifiers=lambda: Qt.NoModifier
    )
    editing = types.SimpleNamespace(
        interactor=interactor, _drag=Mock(), _editing=True, nudge_requested=Mock(),
        _nudge_vector=lambda key, mods: "VEC",
    )
    assert Viewport.eventFilter(editing, interactor, event) is True  # consumed
    editing.nudge_requested.emit.assert_called_once_with("VEC")


def test_viewport_nudge_vector_moves_in_the_camera_screen_plane(qapp):
    """Right/Up track the camera's horizontal/vertical axes; Shift is coarser."""
    import types

    import numpy as np
    from PySide6.QtCore import Qt

    from crystalline.ui.viewport import Viewport, _NUDGE_STEP, _NUDGE_STEP_COARSE

    # Camera looking down -z with +y up: screen-right is +x, screen-up is +y.
    camera = types.SimpleNamespace(
        GetDirectionOfProjection=lambda: (0.0, 0.0, -1.0),
        GetViewUp=lambda: (0.0, 1.0, 0.0),
    )
    stub = types.SimpleNamespace(interactor=types.SimpleNamespace(camera=camera))

    right = Viewport._nudge_vector(stub, Qt.Key_Right, Qt.NoModifier)
    up = Viewport._nudge_vector(stub, Qt.Key_Up, Qt.NoModifier)
    assert np.allclose(right, [_NUDGE_STEP, 0, 0])
    assert np.allclose(up, [0, _NUDGE_STEP, 0])
    coarse = Viewport._nudge_vector(stub, Qt.Key_Right, Qt.ShiftModifier)
    assert np.allclose(coarse, [_NUDGE_STEP_COARSE, 0, 0])


def test_connect_signals_only_uses_attributes_that_exist_by_then():
    """``__init__`` calls ``_connect_signals()`` before ``menus.build_menus()``,
    so anything the menus create (``_edit_mode_action`` and friends) is not there
    yet — connecting to it would blow up at startup, where no test can reach it."""
    from crystalline.ui.main_window import MainWindow

    tree = ast.parse(Path(inspect.getsourcefile(MainWindow)).read_text())
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "MainWindow")
    methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")

    # Walk __init__ in order, stopping at the _connect_signals() call.
    assigned = set()
    for statement in init.body:
        calls = [
            n
            for n in ast.walk(statement)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_connect_signals"
        ]
        if calls:
            break
        for node in ast.walk(statement):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Attribute) and getattr(target.value, "id", None) == "self":
                        assigned.add(target.attr)

    connect = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_connect_signals")
    used = {
        node.attr
        for node in ast.walk(connect)
        if isinstance(node, ast.Attribute) and getattr(node.value, "id", None) == "self"
    }
    missing = sorted(used - assigned - methods)
    assert not missing, (
        "_connect_signals uses attributes not yet assigned when it runs: "
        f"{missing} — connect them after menus.build_menus(), or go through a method"
    )


def test_every_slot_the_menus_wire_up_exists_on_main_window():
    """Guards the stub above from drifting away from the real window."""
    from crystalline.ui.main_window import MainWindow

    source = Path(inspect.getsourcefile(menus)).read_text()
    used = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "window":
                used.add(node.attr)

    # attributes MainWindow.__init__ assigns, plus the ones menus itself sets
    window_source = Path(inspect.getsourcefile(MainWindow)).read_text()
    assigned = {
        target.attr
        for node in ast.walk(ast.parse(window_source))
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Attribute) and getattr(target.value, "id", None) == "self"
    }
    set_by_menus = {
        target.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Attribute) and getattr(target.value, "id", None) == "window"
    }

    known = set(dir(MainWindow)) | assigned | set_by_menus
    missing = sorted(used - known)
    assert not missing, f"menus.py wires up names MainWindow does not have: {missing}"
