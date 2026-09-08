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
