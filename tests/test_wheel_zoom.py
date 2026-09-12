"""The wheel-zoom rule every 3D view shares.

It lives in one module precisely so the structure viewport and the Brillouin
zone cannot end up with different wheels under the same hand — so the test is
partly that both really do use it.
"""

import ast
import inspect
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402

from crystalline.ui import wheel_zoom  # noqa: E402


def _wheel(units: int) -> QWheelEvent:
    """A wheel event carrying ``units`` eighths of a degree, as Qt reports."""
    return QWheelEvent(QPoint(0, 0), QPoint(0, 0), QPoint(0, 0),
                       QPoint(0, units), Qt.NoButton, Qt.NoModifier,
                       Qt.NoScrollPhase, False)


def test_one_notch_zooms_by_the_stated_amount():
    factor = wheel_zoom.zoom_factor(_wheel(int(wheel_zoom.WHEEL_UNITS_PER_NOTCH)))
    assert factor == pytest.approx(wheel_zoom.ZOOM_PER_NOTCH)


def test_a_small_scroll_zooms_proportionally_less():
    """The whole point: VTK's own style moves the same distance for any delta,
    which is what makes a trackpad's stream of small events leap."""
    notch = wheel_zoom.WHEEL_UNITS_PER_NOTCH
    small = wheel_zoom.zoom_factor(_wheel(int(notch / 10)))
    full = wheel_zoom.zoom_factor(_wheel(int(notch)))
    assert 1.0 < small < full
    # ten small scrolls travel as far as one notch, so the hand's total
    # movement decides the zoom rather than the number of events it became
    assert small ** 10 == pytest.approx(full, rel=1e-6)


def test_scrolling_back_undoes_scrolling_forward():
    forward = wheel_zoom.zoom_factor(_wheel(120))
    backward = wheel_zoom.zoom_factor(_wheel(-120))
    assert forward * backward == pytest.approx(1.0)


def test_an_absurd_delta_cannot_teleport_the_view():
    assert wheel_zoom.zoom_factor(_wheel(100000)) == wheel_zoom.MAX_ZOOM_PER_EVENT
    assert wheel_zoom.zoom_factor(_wheel(-100000)) == 1 / wheel_zoom.MAX_ZOOM_PER_EVENT


def test_a_dead_event_does_nothing():
    assert wheel_zoom.zoom_factor(_wheel(0)) == 1.0


def test_a_parallel_camera_zooms_by_scale_not_by_dollying():
    """Under parallel projection a dolly does nothing at all, so the zone view
    — which is parallel on purpose — would simply not zoom."""
    import vtk

    camera = vtk.vtkCamera()
    camera.ParallelProjectionOn()
    camera.SetParallelScale(2.0)
    wheel_zoom.apply_zoom(camera, 2.0)
    assert camera.GetParallelScale() == pytest.approx(1.0)


def test_a_nonsense_factor_leaves_the_camera_alone():
    import vtk

    camera = vtk.vtkCamera()
    camera.ParallelProjectionOn()
    camera.SetParallelScale(2.0)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        wheel_zoom.apply_zoom(camera, bad)
    assert camera.GetParallelScale() == pytest.approx(2.0)


def test_both_three_dimensional_views_use_this_module():
    """A second opinion about the wheel is the bug this module exists to stop."""
    for module in ("ui/viewport.py", "ui/panels/zone_picker.py"):
        source = Path(inspect.getfile(wheel_zoom)).parent.parent / module
        tree = ast.parse(source.read_text())
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        assert "wheel_zoom" in names, f"{module} rolls its own wheel zoom"
