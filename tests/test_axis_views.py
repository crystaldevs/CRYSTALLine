"""The a/b/c view-alignment buttons.

Looking down a lattice axis needs an up direction, and which one is chosen is
what makes the three views agree or disagree with each other. Taking the other
lattice vectors in plain index order gave the b view a→up — the *previous* axis
where a and c both got the next one — which mirrored that one view against the
other two and threw the a/b/c gizmo to the opposite side of the screen.
"""

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyvista")

from crystalline.ui.viewport import Viewport  # noqa: E402

_CUBIC = np.eye(3) * 5.0
# ZnO's hexagonal cell: a and b are 120° apart, so the up vector has to be
# projected perpendicular to the view direction rather than used as-is.
_HEXAGONAL = np.array(
    [[2.85292, -1.647134, 0.0], [0.0, 3.294269, 0.0], [0.0, 0.0, 5.270251]]
)


def _screen_frame(cell, axis):
    """Which lattice vector points up and which points right, for this view."""
    direction = cell[axis] / np.linalg.norm(cell[axis])
    up = Viewport._up_for(direction, cell, axis)
    right = np.cross(direction, up)  # VTK: right = view direction × up
    def nearest(vector):
        dots = [float(np.dot(cell[i] / np.linalg.norm(cell[i]), vector)) for i in range(3)]
        return "abc"[int(np.argmax(dots))]
    return nearest(up), nearest(right)


@pytest.mark.parametrize("cell", [_CUBIC, _HEXAGONAL], ids=["cubic", "hexagonal"])
def test_the_three_axis_views_are_cyclic(cell):
    """Down a puts b up and c right; down b puts c up and a right; down c puts a
    up and b right. Any other pattern makes one view disagree with the others."""
    assert _screen_frame(cell, 0) == ("b", "c")
    assert _screen_frame(cell, 1) == ("c", "a")
    assert _screen_frame(cell, 2) == ("a", "b")


@pytest.mark.parametrize("cell", [_CUBIC, _HEXAGONAL], ids=["cubic", "hexagonal"])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_the_up_vector_is_a_unit_vector_across_the_view(cell, axis):
    direction = cell[axis] / np.linalg.norm(cell[axis])
    up = Viewport._up_for(direction, cell, axis)

    assert np.linalg.norm(up) == pytest.approx(1.0)
    assert float(np.dot(up, direction)) == pytest.approx(0.0, abs=1e-12)


def test_a_degenerate_lattice_vector_falls_through_to_the_next_one():
    """A slab's aperiodic direction can be a zero (or vacuum) vector; the up
    choice has to skip it rather than normalising a zero."""
    cell = np.array([[4.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 6.0]])

    up = Viewport._up_for(np.array([1.0, 0.0, 0.0]), cell, 0)

    assert np.linalg.norm(up) == pytest.approx(1.0)
    assert np.allclose(np.abs(up), [0.0, 0.0, 1.0])  # fell through b to c


def test_without_a_cell_the_up_vector_comes_from_a_world_axis():
    """A molecule has no a/b/c; the view still has to be well defined."""
    for axis in range(3):
        direction = np.eye(3)[axis]
        up = Viewport._up_for(direction, None, axis)

        assert np.linalg.norm(up) == pytest.approx(1.0)
        assert float(np.dot(up, direction)) == pytest.approx(0.0, abs=1e-12)
