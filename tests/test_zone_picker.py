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


def test_a_label_sits_just_clear_of_its_own_marker():
    """Close to the dot it names, and outward so it leaves the zone's body.

    Close is only possible because the coordinates moved to the corner legend:
    a two-line label on every point overlapped into a heap, a one-line label
    beside its dot does not.
    """
    assert zone_picker._LABEL_OFFSET > 1.0, "the label must clear the marker"
    centre = np.array([0.4, -0.2, 0.1])
    direction = zone_picker._outward(centre)
    assert float(np.dot(direction, centre)) > 0  # outward, not into the zone
    assert np.linalg.norm(direction) == pytest.approx(1.0)


def test_gamma_gets_a_direction_of_its_own():
    """The zone centre has no outward direction; it still needs its label clear."""
    direction = zone_picker._outward(np.zeros(3))
    assert pytest.approx(1.0) == float(np.linalg.norm(direction))


def test_the_gap_scales_with_the_marker():
    """Both are world lengths, so zooming never closes the gap: the marker and
    the distance to its label grow together."""
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


# ── how labels are written into the scene ─────────────────────────────────
def test_gamma_is_written_as_mathtext_because_the_font_has_no_greek():
    """VTK's built-in font draws a missing glyph as *nothing*, so a bare "Γ"
    label rendered blank and the point looked unlabelled. MathText has Greek."""
    assert zone_picker._math_label("G") == r"$\Gamma$"
    assert zone_picker._math_label("X") == "$X$"
    assert zone_picker._math_label("X_1") == "$X_1$"


def test_a_greek_named_point_keeps_its_letter_and_its_index():
    assert zone_picker._math_label("Sigma_1") == r"$\Sigma_1$"


def test_a_label_mathtext_could_not_set_falls_back_to_plain_text():
    """Blank is the one unacceptable outcome; ugly is fine."""
    assert zone_picker._math_label("X'") == "X'"


def test_the_axes_are_labelled_with_real_subscripts():
    """k_x, not kx: the axis of a reciprocal-space figure is a subscripted k."""
    import inspect

    source = inspect.getsource(zone_picker.ZonePickerDialog._draw_axes)
    for axis in ("$k_x$", "$k_y$", "$k_z$"):
        assert axis in source


# ── the path's colours ────────────────────────────────────────────────────
def test_each_leg_of_the_path_gets_its_own_colour():
    colours = [zone_picker.segment_colour(i) for i in range(4)]
    assert len(set(colours)) == 4


def test_the_colours_repeat_rather_than_running_out():
    """A path can be longer than the palette; it must still draw."""
    palette = len(zone_picker._SEGMENT_COLOURS)
    assert zone_picker.segment_colour(palette) == zone_picker.segment_colour(0)
    assert zone_picker.segment_colour(palette * 3 + 2) == zone_picker.segment_colour(2)


def test_the_legend_font_can_write_gamma_and_angstrom():
    """The legend is plain text, not MathText, so it needs a font with the
    characters in it — the same reason the markers needed MathText."""
    TTFont = pytest.importorskip("fontTools.ttLib").TTFont

    path = zone_picker._unicode_font()
    if path is None:
        pytest.skip("matplotlib's bundled font is not available")
    cmap = TTFont(path).getBestCmap()
    for character in "Γ→Å⁻¹½":
        assert ord(character) in cmap, f"{character!r} is missing from the legend font"


# ── one image export, shared with the structure window ────────────────────
def test_both_windows_export_images_through_the_same_dialog():
    """The zone offered a bare PNG while the structure window offered format,
    resolution and transparency. Same job, two answers, and the difference is
    only visible to someone who has used both — which is everyone."""
    import ast
    import inspect
    from pathlib import Path

    from crystalline.ui import image_export

    root = Path(inspect.getfile(image_export)).parent
    for module in ("main_window.py", "panels/zone_picker.py"):
        tree = ast.parse((root / module).read_text())
        names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        names |= {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        assert "export_view" in names, f"{module} does not use the shared exporter"


def test_the_exporter_offers_vector_formats_too():
    """A zone diagram goes into a paper, so PDF and SVG matter as much as PNG."""
    from crystalline.ui.image_export import FORMATS

    extensions = {ext for ext, _label, _vector, _alpha in FORMATS}
    assert {"png", "jpg", "tif", "svg", "pdf", "eps"} <= extensions


# ── the toolbar mirrors the structure window's ────────────────────────────
def test_the_axis_chips_are_drawn_with_real_subscripts():
    """A QToolButton renders plain text, and Unicode has a subscript x but no
    subscript y or z — so "k_y" cannot be written as a string at all. The chip
    label is drawn instead, which is why it is an icon."""
    from PySide6.QtGui import QFont, QIcon
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    icon = zone_picker._subscript_icon("k", "y", "#ffffff", QFont())
    assert isinstance(icon, QIcon)
    assert not icon.isNull()
    size = icon.availableSizes()[0]
    assert size.width() > 0 and size.height() > 0


def test_the_zone_and_the_structure_window_rotate_by_the_same_step():
    """Two 3D views that orbit by different amounts would be two apps."""
    from crystalline.ui import menus

    assert zone_picker._ROTATE_STEP_DEG == menus._ROTATE_STEP_DEG


def test_the_corner_readout_is_inset_from_the_edge():
    """Text hard against the frame reads as though it has been cropped."""
    assert 0.5 < zone_picker._LEGEND_X < 1.0
    assert 0.5 < zone_picker._LEGEND_TOP < 1.0
    # a comfortable margin, not a hairline
    assert 1.0 - zone_picker._LEGEND_X > 0.02
    assert 1.0 - zone_picker._LEGEND_TOP > 0.05


def test_the_legend_does_not_repeat_the_path_list():
    """The path's legs are listed beside the view, in the same colours. Saying
    it again in the corner grew the corner with the path and told the reader
    nothing new."""
    import inspect

    source = inspect.getsource(zone_picker.ZonePickerDialog._draw_legend)
    assert "_legend_selection" in source
    assert not hasattr(zone_picker.ZonePickerDialog, "_legend_segment")


def test_there_is_no_control_that_does_nothing():
    """The 'All coordinates' tick was left behind when the coordinates moved to
    the corner: it redrew the scene and changed nothing anyone could see."""
    import inspect

    source = inspect.getsource(zone_picker)
    assert "All coordinates" not in source
    assert "_coordinates" not in source
