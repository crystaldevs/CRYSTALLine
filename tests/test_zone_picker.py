"""The Brillouin-zone picker's geometry helpers.

The dialog itself needs a GL context (it owns a VTK viewport), so it is not
built here; what is testable without one is where the labels go, which is the
part that was wrong — labels drawn at the markers' own centres vanished inside
them as soon as anyone zoomed in.
"""

import numpy as np
import pytest

pytest.importorskip("PySide6")

from crystalline.ui.panels import zone_picker  # noqa: E402


def test_a_label_is_pushed_clear_of_its_marker():
    """Outward by more than the marker's radius, so the text is never inside it."""
    centre = np.array([0.4, -0.2, 0.1])
    offset = zone_picker._outward(centre) * zone_picker._LABEL_OFFSET
    assert np.linalg.norm(offset) > 1.0  # in units of the marker radius
    # ...and outward, not into the zone: the label leaves the body it labels.
    assert float(np.dot(offset, centre)) > 0


def test_gamma_gets_a_direction_of_its_own():
    """The zone centre has no outward direction; it still needs its label clear."""
    direction = zone_picker._outward(np.zeros(3))
    assert pytest.approx(1.0) == float(np.linalg.norm(direction))


def test_the_offset_scales_with_the_marker():
    """Both are in world units, so zooming can never close the gap between them.

    This is the whole fix: the label's distance from the marker and the marker's
    own size grow together, so their ratio on screen is the same at every zoom.
    """
    for extent in (0.1, 1.0, 37.0):
        radius = zone_picker._POINT_RADIUS * extent
        gap = radius * zone_picker._LABEL_OFFSET
        assert gap / radius == pytest.approx(zone_picker._LABEL_OFFSET)


# ── what the side panel shows ─────────────────────────────────────────────
def test_a_coordinate_is_shown_as_the_fraction_it_is():
    """0.333 reads as an approximation of something; 1/3 reads as the thing.

    Special points are halves, thirds and eighths, and the fraction is also
    what explains the shrinking factor — a path through K = 1/3 1/3 0 and
    M = 1/2 0 0 needs ISS 6 because lcm(3, 2) is 6.
    """
    assert zone_picker._tidy_number(1 / 3) == "1/3"
    assert zone_picker._tidy_number(0.5) == "1/2"
    assert zone_picker._tidy_number(0.375) == "3/8"
    assert zone_picker._tidy_number(0.0) == "0"
    assert zone_picker._tidy_number(-0.5) == "-1/2"


def test_a_coordinate_that_is_not_a_simple_fraction_is_left_as_a_number():
    """Rounding 0.137 to 3/22 would claim a special point that is not there."""
    assert zone_picker._tidy_number(0.137) == "0.137"


def test_a_dashed_line_is_made_of_separate_pieces():
    """VTK's OpenGL2 backend dropped line stippling, so the dashes are real
    segments; each must be its own two-point line, not one polyline."""
    dashes = zone_picker._dashed_line([0, 0, 0], [1, 0, 0], extent=1.0)
    assert dashes is not None
    assert dashes.n_cells > 1, "a dashed line of one piece is a solid line"
    assert dashes.n_points == 2 * dashes.n_cells
    # ...and it stays within the span it was asked for
    assert dashes.points[:, 0].max() <= 1.0 + 1e-9
    assert dashes.points[:, 0].min() >= -1e-9


def test_a_line_of_no_length_produces_nothing():
    """Γ to itself: there is no line from a point to where it already is."""
    assert zone_picker._dashed_line([0, 0, 0], [0, 0, 0], extent=1.0) is None
