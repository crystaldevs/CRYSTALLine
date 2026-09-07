"""Projection math for drag-to-move — tested against a real (off-screen) camera.

No Qt here: an off-screen ``pyvista.Plotter`` gives a genuine vtkRenderer with a
finalised camera, which is all the projection helpers need.
"""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from crystalline.core.structure import Structure
from crystalline.viz.renderer import StructureRenderer
from crystalline.ui.drag_controller import world_to_depth, screen_to_world


@pytest.fixture
def scene():
    pv.OFF_SCREEN = True
    s = Structure.empty()
    s.set_cell(np.eye(3) * 8, periodic=False)
    s.add_atom("C", [2, 3, 1])
    s.add_atom("O", [4, 3, 1])
    p = pv.Plotter(off_screen=True, window_size=(800, 600))
    r = StructureRenderer(p)
    r.set_structure(s)
    p.reset_camera()
    p.render()
    yield p, r, s
    p.close()


def test_screen_world_roundtrip_is_identity(scene):
    p, r, s = scene
    ren = p.renderer
    for world in ([2, 3, 1], [4, 3, 1], [3, 3, 1], [2, 3, 2]):
        w = np.array(world, dtype=float)
        ren.SetWorldPoint(*w, 1.0)
        ren.WorldToDisplay()
        dx, dy, _ = ren.GetDisplayPoint()
        depth = world_to_depth(ren, w)
        back = screen_to_world(ren, dx, dy, depth)
        assert np.allclose(back, w, atol=1e-6)


def test_drag_stays_in_screen_parallel_plane(scene):
    p, r, s = scene
    ren = p.renderer
    depth = world_to_depth(ren, r.atom_position(0))
    here = screen_to_world(ren, 206, 300, depth)
    right = screen_to_world(ren, 306, 300, depth)
    delta = right - here
    # a horizontal screen move maps to an in-plane world move (no depth drift)
    assert abs(delta[0]) > 0.1  # actually moved
    assert abs(delta[2]) < 1e-6  # depth unchanged


def test_preview_move_does_not_touch_model(scene):
    p, r, s = scene
    original = s.positions[0].copy()
    r.preview_atom_position(0, np.array([5.0, 5.0, 5.0]))
    # atom moved on screen, model untouched (drag commits only on release)
    assert np.allclose(s.positions[0], original)
    assert np.allclose(r.rendered_atom_position(0), [5.0, 5.0, 5.0])
