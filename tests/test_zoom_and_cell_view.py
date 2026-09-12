"""Continuous wheel zoom, and the graphical primitive/crystallographic switch."""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from crystalline.core.structure import Structure  # noqa: E402
from crystalline.ui.drag_controller import install_atom_drag  # noqa: E402
from crystalline.viz.renderer import StructureRenderer  # noqa: E402


def _scene():
    structure = Structure.empty()
    structure.set_cell(np.eye(3) * 8, periodic=False)
    structure.add_atom("C", [4, 4, 4])
    plotter = pv.Plotter(off_screen=True, window_size=(800, 600))
    renderer = StructureRenderer(plotter)
    renderer.set_structure(structure)
    plotter.reset_camera()
    style = install_atom_drag(
        plotter, renderer,
        on_commit=lambda i, pos: None,
        on_click=lambda i, additive: None,
    )
    return plotter, style


def _distance(plotter):
    camera = plotter.renderer.GetActiveCamera()
    return float(np.linalg.norm(
        np.asarray(camera.GetPosition()) - np.asarray(camera.GetFocalPoint())
    ))


# ── zoom ──────────────────────────────────────────────────────────────────
def test_zoom_moves_the_camera_in_proportion_to_the_factor():
    """The whole point: the amount asked for is the amount applied, so a small
    scroll zooms a small amount instead of VTK's fixed step."""
    plotter, style = _scene()
    try:
        # The app draws orthographic by default, where the zoom is the parallel
        # scale; this covers the perspective half.
        plotter.disable_parallel_projection()
        before = _distance(plotter)

        style.zoom_by(2.0)
        halved = _distance(plotter)
        assert halved == pytest.approx(before / 2.0, rel=1e-6)

        style.zoom_by(1.0 / 2.0)  # and back
        assert _distance(plotter) == pytest.approx(before, rel=1e-6)

        style.zoom_by(1.02)  # a nudge is a nudge, not a step
        assert _distance(plotter) == pytest.approx(before / 1.02, rel=1e-6)
    finally:
        plotter.close()


def test_zoom_under_parallel_projection_changes_the_scale_not_the_distance():
    """Under parallel projection the zoom *is* the camera's parallel scale, so
    dollying would appear to do nothing at all."""
    plotter, style = _scene()
    try:
        plotter.enable_parallel_projection()  # the app's own default
        camera = plotter.renderer.GetActiveCamera()
        before_scale = camera.GetParallelScale()
        before_distance = _distance(plotter)

        style.zoom_by(2.0)

        assert camera.GetParallelScale() == pytest.approx(before_scale / 2.0, rel=1e-6)
        assert _distance(plotter) == pytest.approx(before_distance, rel=1e-6)
    finally:
        plotter.close()


def test_zooming_out_still_stops_at_the_cap():
    """The zoom-out cap that keeps the structure findable must apply to the
    wheel too, not only to VTK's own dolly."""
    plotter, style = _scene()
    try:
        plotter.disable_parallel_projection()  # the cap is on camera distance
        cap = _distance(plotter) * 1.5
        style.set_max_distance(cap)

        style.zoom_by(1.0 / 100.0)  # a huge pull-back

        assert _distance(plotter) == pytest.approx(cap, rel=1e-6)
    finally:
        plotter.close()


def test_a_nonsense_factor_is_ignored():
    plotter, style = _scene()
    try:
        plotter.disable_parallel_projection()
        before = _distance(plotter)
        for factor in (0.0, -1.0, float("nan"), float("inf")):
            style.zoom_by(factor)
        assert _distance(plotter) == pytest.approx(before, rel=1e-9)
    finally:
        plotter.close()


def test_the_wheel_factor_follows_the_scroll_amount():
    """VTK dollies a fixed ~21% per wheel *event* whatever was scrolled, which is
    what makes the wheel feel notched and a trackpad leap. The factor has to come
    from the delta."""
    pytest.importorskip("PySide6")
    from crystalline.ui import viewport as vp
    from crystalline.ui import wheel_zoom

    asked = []

    class _StubViewport:
        _zoom_from_wheel = vp.Viewport._zoom_from_wheel
        _drag = type("D", (), {"zoom_by": staticmethod(asked.append)})()
        # the method also reports that the camera is moving; not what this test
        # is about (see test_a_wheel_zoom_counts_as_moving_the_camera_...)
        _set_camera_busy = staticmethod(lambda _busy: None)
        _wheel_idle = type("T", (), {"start": staticmethod(lambda: None)})()

    class _Wheel:
        def __init__(self, y, x=0):
            self._y, self._x = y, x

        def angleDelta(self):
            return type("P", (), {"y": lambda _s: self._y, "x": lambda _s: self._x})()

    view = _StubViewport()

    view._zoom_from_wheel(_Wheel(120))          # one notch in
    view._zoom_from_wheel(_Wheel(60))           # half a notch
    view._zoom_from_wheel(_Wheel(-120))         # one notch out
    view._zoom_from_wheel(_Wheel(0))            # nothing at all

    assert len(asked) == 3                      # the empty scroll did nothing
    full, half, out = asked
    # The rule itself lives in wheel_zoom now, shared with the zone view; what
    # this test pins is that the viewport goes through it.
    assert full == pytest.approx(wheel_zoom.ZOOM_PER_NOTCH)
    assert 1.0 < half < full                    # proportionally less, not the same
    assert out == pytest.approx(1.0 / full)     # and symmetric


def test_an_absurd_delta_cannot_teleport_the_view():
    pytest.importorskip("PySide6")
    from crystalline.ui import viewport as vp
    from crystalline.ui import wheel_zoom

    asked = []

    class _StubViewport:
        _zoom_from_wheel = vp.Viewport._zoom_from_wheel
        _drag = type("D", (), {"zoom_by": staticmethod(asked.append)})()
        # the method also reports that the camera is moving; not what this test
        # is about (see test_a_wheel_zoom_counts_as_moving_the_camera_...)
        _set_camera_busy = staticmethod(lambda _busy: None)
        _wheel_idle = type("T", (), {"start": staticmethod(lambda: None)})()

    class _Wheel:
        def angleDelta(self):
            return type("P", (), {"y": lambda _s: 100000, "x": lambda _s: 0})()

    _StubViewport()._zoom_from_wheel(_Wheel())

    assert asked[0] == pytest.approx(wheel_zoom.MAX_ZOOM_PER_EVENT)


# ── the cell-view switch ──────────────────────────────────────────────────
def test_switching_cell_view_redraws_and_is_a_no_op_when_unchanged():
    pytest.importorskip("PySide6")
    from crystalline.core.cells import CellView
    from crystalline.ui.main_window import MainWindow

    rebuilt = []

    class _StubWindow:
        _set_cell_view = MainWindow._set_cell_view
        _update_cell_view_controls = MainWindow._update_cell_view_controls

        def _apply_cell_view(self):
            rebuilt.append(True)

    window = _StubWindow()
    window._cell_view = CellView.CRYSTALLOGRAPHIC
    window._cell_view_actions = {}
    window._cell_view_buttons = {}

    window._set_cell_view(CellView.CRYSTALLOGRAPHIC)   # already there
    assert rebuilt == []

    window._set_cell_view(CellView.PRIMITIVE)
    assert window._cell_view is CellView.PRIMITIVE
    assert len(rebuilt) == 1


def test_the_menu_and_toolbar_follow_the_cell_actually_shown(qapp):
    """They are a display of the current view, so a change made anywhere — the
    menu, the switch, or loading a file — has to be reflected in both. And
    setting them must not echo back: the switch would fight the click."""
    from PySide6.QtGui import QAction

    from crystalline.core.cells import CellView
    from crystalline.ui.main_window import MainWindow
    from crystalline.ui.widgets import ToggleSwitch

    class _StubWindow:
        _update_cell_view_controls = MainWindow._update_cell_view_controls

    window = _StubWindow()
    window._cell_view = CellView.PRIMITIVE
    window._cell_view_actions = {view: QAction("x", checkable=True) for view in CellView}
    window._conventional_switch = ToggleSwitch()
    window._conventional_switch.setChecked(True)  # out of step, to be corrected

    fired = []
    window._conventional_switch.toggled.connect(lambda *_: fired.append(True))
    window._cell_view_actions[CellView.PRIMITIVE].triggered.connect(
        lambda *_: fired.append(True)
    )

    window._update_cell_view_controls()

    assert window._cell_view_actions[CellView.PRIMITIVE].isChecked()
    assert not window._cell_view_actions[CellView.CRYSTALLOGRAPHIC].isChecked()
    # one switch, not a pair: on means the conventional cell
    assert not window._conventional_switch.isChecked()
    assert fired == []


def test_the_conventional_switch_is_a_single_on_off_control(qapp):
    """Two cells, each the negation of the other — a conv/prim pair spends two
    controls saying what one can."""
    from crystalline.core.cells import CellView
    from crystalline.ui import menus

    class _Window:
        def __init__(self):
            self.chosen = []

        def _set_cell_view(self, view):
            self.chosen.append(view)

    window = _Window()
    window._cell_view = CellView.CRYSTALLOGRAPHIC

    from crystalline.ui.widgets import ToggleSwitch

    button = ToggleSwitch()
    button.setChecked(True)
    button.toggled.connect(
        lambda on: window._set_cell_view(
            CellView.CRYSTALLOGRAPHIC if on else CellView.PRIMITIVE
        )
    )

    button.setChecked(False)
    button.setChecked(True)

    assert window.chosen == [CellView.PRIMITIVE, CellView.CRYSTALLOGRAPHIC]


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


# ── camera-busy reporting ─────────────────────────────────────────────────
def _busy_stub():
    """A stand-in for the Viewport carrying just its camera-busy machinery.

    A real ``Viewport`` needs a live ``QtInteractor``, which segfaults under the
    offscreen platform plugin, so the methods under test are bound onto a plain
    QObject instead.
    """
    from PySide6.QtCore import QObject, QPoint, QTimer, Signal

    from crystalline.ui.viewport import Viewport

    class _Stub(QObject):
        camera_busy = Signal(bool)
        _set_camera_busy = Viewport._set_camera_busy
        _zoom_from_wheel = Viewport._zoom_from_wheel

    stub = _Stub()
    stub._camera_busy = False
    stub.zoomed = []
    stub._drag = type("D", (), {"zoom_by": lambda _self, f: stub.zoomed.append(f)})()
    stub._wheel_idle = QTimer(stub)
    stub._wheel_idle.setSingleShot(True)
    stub._wheel_idle.setInterval(10)
    stub._wheel_idle.timeout.connect(lambda: stub._set_camera_busy(False))
    stub.seen = []
    stub.camera_busy.connect(stub.seen.append)
    stub.wheel = type("W", (), {"angleDelta": lambda _self: QPoint(0, 120)})()
    return stub


def test_camera_busy_is_announced_once_per_transition(qapp):
    """It gates other work (the phonon animation), so it must be an edge, not a
    stream: a drag emits one True however many moves it is made of."""
    stub = _busy_stub()

    stub._set_camera_busy(True)
    stub._set_camera_busy(True)
    stub._set_camera_busy(False)
    stub._set_camera_busy(False)

    assert stub.seen == [True, False]


def test_a_wheel_zoom_counts_as_moving_the_camera_until_it_goes_quiet(qapp):
    """A wheel zoom has no end event of its own — a trackpad sends a long
    stream of them — so it is bracketed by a timeout instead. Without this the
    animation would keep competing with the render through the whole gesture."""
    from PySide6.QtCore import QCoreApplication, QElapsedTimer

    stub = _busy_stub()

    stub._zoom_from_wheel(stub.wheel)
    assert stub.seen == [True]
    assert stub.zoomed  # ...and it still zooms

    clock = QElapsedTimer()
    clock.start()
    while stub.seen == [True] and clock.elapsed() < 2000:
        QCoreApplication.processEvents()
    assert stub.seen == [True, False]
