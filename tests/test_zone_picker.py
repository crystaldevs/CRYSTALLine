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


def test_a_label_sits_outside_the_zone_in_its_point_s_own_direction():
    """Labels ride on a ring around the polyhedron, not on its surface.

    The high-symmetry points of a cubic lattice all lie within about thirty
    degrees of each other, so labels nudged off their markers stay as bunched
    as the points are. Pushing them all to one radius spreads them by the only
    thing that differs — their direction.
    """
    assert zone_picker._LABEL_RADIUS > 1.0, "the ring must clear the zone"
    centre = np.array([0.4, -0.2, 0.1])
    direction = zone_picker._outward(centre)
    assert float(np.dot(direction, centre)) > 0  # outward, not into the zone
    assert np.linalg.norm(direction) == pytest.approx(1.0)


def test_gamma_gets_a_direction_of_its_own():
    """The zone centre has no outward direction; it still needs its label clear."""
    direction = zone_picker._outward(np.zeros(3))
    assert pytest.approx(1.0) == float(np.linalg.norm(direction))


def test_the_ring_scales_with_the_zone():
    """Everything is a fraction of the zone's own extent, so the picture looks
    the same whatever the lattice parameters are."""
    for extent in (0.1, 1.0, 37.0):
        ring = extent * zone_picker._LABEL_RADIUS
        marker = extent * zone_picker._POINT_RADIUS
        assert ring > extent > marker
        assert ring / extent == pytest.approx(zone_picker._LABEL_RADIUS)


# ── what the side panel shows ─────────────────────────────────────────────
def test_a_coordinate_is_shown_as_the_fraction_it_is():
    """0.333 reads as an approximation of something; 1/3 reads as the thing.

    Special points are halves, thirds and eighths, and the fraction is also
    what explains the shrinking factor — a path through K = 1/3 1/3 0 and
    M = 1/2 0 0 needs ISS 6 because lcm(3, 2) is 6.
    """
    assert zone_picker._tidy_number(1 / 3) == "⅓"
    assert zone_picker._tidy_number(0.5) == "½"
    assert zone_picker._tidy_number(0.375) == "⅜"
    assert zone_picker._tidy_number(0.0) == "0"
    assert zone_picker._tidy_number(-0.5) == "-½"


def test_a_fraction_with_no_glyph_falls_back_to_a_slash():
    """The glyphs cover the denominators special points use; 1/7 is not one of
    them, and inventing a glyph for it is not an option."""
    assert zone_picker._tidy_number(1 / 7) == "1/7"


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


# ── Gamma, and the tools beside the picture ───────────────────────────────
def test_gamma_is_written_as_gamma_for_a_reader():
    """Stored as G because that is what a deck carries; shown as Γ because
    that is what the point is called."""
    from crystalline.core.brillouin import display_label

    assert display_label("G") == "Γ"
    assert display_label("X") == "X"
    assert display_label("W") == "W"


def test_a_typed_gamma_is_understood_as_the_stored_label():
    """The combo offers Γ, so Γ has to be readable back — otherwise the editor
    shows a point it then refuses to accept."""
    from crystalline.ui.panels.band_path_editor import read_kpoint

    points = {"G": (0.0, 0.0, 0.0), "X": (0.5, 0.0, 0.5)}
    assert read_kpoint("Γ", points) == ("G", (0.0, 0.0, 0.0))
    assert read_kpoint("G", points) == ("G", (0.0, 0.0, 0.0))
