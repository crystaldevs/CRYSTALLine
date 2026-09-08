"""AtomDragStyle: drag moves atoms, and survives an interactor-style reset.

Uses a real off-screen VTK interactor so events dispatch through whatever style
is currently active (exactly like real mouse events). Picking is stubbed
(``vtkPropPicker.Pick`` segfaults off-screen), which is orthogonal to what these
tests cover. Skips without pyvista.
"""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
import vtk  # noqa: E402

from crystalline.core.structure import Structure  # noqa: E402
from crystalline.viz.renderer import StructureRenderer  # noqa: E402
from crystalline.ui.drag_controller import install_atom_drag  # noqa: E402


def _scene():
    s = Structure.empty()
    s.set_cell(np.eye(3) * 8, periodic=False)
    s.add_atom("C", [4, 4, 4])
    p = pv.Plotter(off_screen=True, window_size=(800, 600))
    r = StructureRenderer(p)
    r.set_structure(s)
    style = install_atom_drag(
        p, r,
        on_commit=lambda i, pos: s.move_atom(i, pos),
        on_click=lambda i, additive: None,
        on_click_empty=lambda: None,
        on_grab=lambda i: None,
        editing=True,
    )
    style._pick_atom = lambda x, y: 0  # bypass the off-screen-segfaulting picker
    p.reset_camera()
    p.render()
    return p, r, s, style


def _drag_distance(p, s, style):
    """Dispatch a press-move-release through the interactor's ACTIVE style."""
    iren = p.iren.interactor
    before = s.positions[0].copy()
    ren = p.renderer
    ren.SetWorldPoint(*before, 1.0)
    ren.WorldToDisplay()
    sx, sy, _ = ren.GetDisplayPoint()
    iren.SetControlKey(0)
    iren.SetEventPosition(int(sx), int(sy))
    iren.InvokeEvent("LeftButtonPressEvent")
    iren.SetEventPosition(int(sx + 80), int(sy))
    iren.InvokeEvent("MouseMoveEvent")
    iren.SetEventPosition(int(sx + 80), int(sy))
    iren.InvokeEvent("LeftButtonReleaseEvent")
    return float(np.linalg.norm(s.positions[0] - before))


def test_drag_moves_atom_when_our_style_is_active():
    p, r, s, style = _scene()
    assert _drag_distance(p, s, style) > 0.05
    p.close()


def test_reactivate_restores_our_style_after_reset():
    # Reproduces the reported bug's mechanism: a QtInteractor can swap back to a
    # default trackball style (e.g. after the menu/focus change from toggling
    # editing), after which atom-drag silently does nothing. reactivate() —
    # which set_editing_enabled()/show_structure() now call — must put our style
    # back as the active one.
    p, r, s, style = _scene()
    iren = p.iren.interactor

    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())  # style reset
    assert _drag_distance(p, s, style) == 0.0  # bug: drag does nothing
    assert not isinstance(iren.GetInteractorStyle(), type(style))

    style.reactivate()
    assert isinstance(iren.GetInteractorStyle(), type(style))  # our style is back
    p.close()


def test_style_survives_pyvistas_update_style():
    # pyvista re-applies its *tracked* interactor style via update_style() while
    # handling ordinary mouse events (its chart-interaction check). Our style
    # must be the tracked one, or the first interaction reverts to pyvista's
    # default and atom picking/dragging silently dies. Installing through
    # pyvista's `iren.style` keeps it tracked; a direct SetInteractorStyle does not.
    p, r, s, style = _scene()
    p.iren.update_style()  # what pyvista does under the hood on mouse events
    assert isinstance(p.iren.interactor.GetInteractorStyle(), type(style))
    assert _drag_distance(p, s, style) > 0.05  # still draggable after the reversion
    p.close()


def test_zoom_out_is_capped():
    p, r, s, style = _scene()
    cam = p.camera
    focal = np.array(cam.GetFocalPoint())
    pos = np.array(cam.GetPosition())
    d0 = np.linalg.norm(pos - focal)
    style.set_max_distance(d0 * 6)

    # yank the camera far out, then clamp
    direction = (pos - focal) / d0
    cam.SetPosition(*(focal + direction * d0 * 100))
    style._clamp_zoom()
    capped = np.linalg.norm(np.array(cam.GetPosition()) - focal)
    assert abs(capped - d0 * 6) < 1e-3

    # a within-cap distance is left untouched
    cam.SetPosition(*(focal + direction * d0 * 3))
    style._clamp_zoom()
    assert abs(np.linalg.norm(np.array(cam.GetPosition()) - focal) - d0 * 3) < 1e-3
    p.close()


def test_camera_moves_are_announced_so_other_work_can_stand_down():
    """The style brackets every camera gesture, which is what lets the phonon
    animation get out of the way of a drag (see PhononPanel.hold). VTK raises
    these events itself for rotate/pan/dolly — they are read off the style
    rather than inferred from button presses, so a gesture that starts any
    other way is still bracketed."""
    import vtk

    from crystalline.ui.drag_controller import install_atom_drag

    seen = []
    plotter, renderer, _structure, _style = _scene()
    style = install_atom_drag(
        plotter, renderer,
        on_commit=lambda i, pos: None,
        on_camera=seen.append,
    )

    style.InvokeEvent(vtk.vtkCommand.StartInteractionEvent)
    assert seen == [True]
    style.InvokeEvent(vtk.vtkCommand.EndInteractionEvent)
    assert seen == [True, False]
