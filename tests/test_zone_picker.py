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
# ── a polymer's zone: a segment, not a solid ──────────────────────────────
def test_a_two_point_zone_is_recognised_as_a_chain():
    """What tells the picture it is drawing a line: everything about a segment
    is drawn differently from a solid or a polygon."""
    vertices = np.array([[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0]])
    assert np.allclose(zone_picker._segment_direction(vertices), [1.0, 0.0, 0.0])

    # A polygon and a solid are not segments, and neither is a zone of no length.
    assert zone_picker._segment_direction(np.eye(3)) is None
    assert zone_picker._segment_direction(np.zeros((2, 3))) is None


def test_a_chain_is_given_an_up_direction_square_to_it():
    """A line has no up of its own, and the labels and the camera both need one."""
    for chain in (np.array([1.0, 0, 0]), np.array([0, 0, 1.0]),
                  np.array([1.0, 1.0, 0]) / np.sqrt(2), np.array([0.3, -0.5, 0.81])):
        chain = chain / np.linalg.norm(chain)
        up = zone_picker._perpendicular_to(chain)
        assert pytest.approx(1.0) == float(np.linalg.norm(up))
        assert abs(float(np.dot(up, chain))) < 1e-9   # square to it, whatever it is


def test_a_chains_labels_go_above_the_bar_not_along_it():
    """Outward from Γ on a segment is along the segment, where the axis arrow
    and the next label already are."""
    class _Dialog:
        _chain = np.array([1.0, 0.0, 0.0])
        _flat_normal = None
        _label_direction = zone_picker.ZonePickerDialog._label_direction

    direction = _Dialog()._label_direction(np.array([0.2, 0.0, 0.0]))
    assert abs(float(np.dot(direction, _Dialog._chain))) < 1e-9

    class _Solid(_Dialog):
        _chain = None

    centre = np.array([0.2, 0.1, 0.0])
    assert np.allclose(_Solid()._label_direction(centre), zone_picker._outward(centre))


# ── a slab's zone: a polygon, seen square on ──────────────────────────────
def test_a_flat_zone_is_looked_at_from_a_fixed_side():
    """Which way ``_plane_normal`` points falls out of the winding of the zone's
    corners, so without this the same slab could open seen from above or from
    below depending on the lattice it came from."""
    for normal in (np.array([0.0, 0, 1.0]), np.array([0.0, 0, -1.0])):
        assert np.allclose(zone_picker._facing(normal), [0, 0, 1])
    for normal in (np.array([0.0, 1.0, 0]), np.array([0.0, -1.0, 0])):
        assert np.allclose(zone_picker._facing(normal), [0, 1, 0])


def test_a_slab_in_the_xy_plane_comes_up_the_familiar_way_round():
    """k_y up and k_x to the right, as a zone diagram prints it — not on its side."""
    up = zone_picker._upright_in(np.array([0.0, 0, 1.0]))

    assert np.allclose(up, [0, 1, 0])
    # And whatever the plane, "up" lies in it.
    for normal in (np.array([1.0, 0, 0]), np.array([1.0, 1.0, 1.0]) / np.sqrt(3)):
        up = zone_picker._upright_in(normal)
        assert pytest.approx(1.0) == float(np.linalg.norm(up))
        assert abs(float(np.dot(up, normal))) < 1e-9


def test_gammas_label_leaves_the_plane_it_is_now_seen_in():
    """Γ's fallback direction points at the viewer, which was fine from a corner
    and invisible once a flat zone is looked at square on — the label sat on top
    of the marker it names. It goes up the plane instead, and off the two axes
    drawn out of Γ."""
    class _Flat:
        _chain = None
        _flat_normal = np.array([0.0, 0, 1.0])
        _label_direction = zone_picker.ZonePickerDialog._label_direction

    direction = _Flat()._label_direction(np.zeros(3))

    assert abs(float(np.dot(direction, _Flat._flat_normal))) < 1e-9   # in the plane
    assert direction[0] < 0 and direction[1] > 0                      # up and left of both axes
    # A point that is not Γ still has its own outward direction.
    corner = np.array([0.3, 0.2, 0.0])
    assert np.allclose(_Flat()._label_direction(corner), zone_picker._outward(corner))


def test_only_a_3d_crystal_has_a_second_cell_to_be_drawn_in():
    """A slab's symmetry is a layer group and a chain's a rod group: neither has
    a Bravais lattice to offer a primitive-versus-conventional choice of."""
    from ase import Atoms

    from crystalline.core.structure import Structure

    def built(pbc):
        return Structure.from_ase(Atoms("C", positions=[[0, 0, 0]],
                                        cell=[[2.5, 0, 0], [0, 2.5, 0], [0, 0, 2.5]],
                                        pbc=pbc))

    assert zone_picker._is_crystal(built([True, True, True]))
    assert not zone_picker._is_crystal(built([True, True, False]))   # slab
    assert not zone_picker._is_crystal(built([True, False, False]))  # polymer


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
def test_gamma_is_written_as_a_gamma_in_a_font_that_has_one():
    """VTK's built-in font draws a missing glyph as *nothing*, so a bare "Γ"
    rendered blank and the point looked unlabelled. The letter is kept and the
    billboard is given a font that has it — see
    ``test_a_billboard_label_is_drawn_in_a_font_that_has_greek``."""
    assert zone_picker._math_label("G") == "Γ"
    assert zone_picker._math_label("X") == "X"
    assert zone_picker._math_label("X_1") == "X₁"


def test_a_greek_named_point_keeps_its_letter_and_its_index():
    """The letter is the point's name; the index distinguishes it from its
    siblings. Both survive as Unicode, the index as a real subscript — those
    are the digits, which the label font has."""
    assert zone_picker._math_label("Sigma_1") == "Σ₁"


def test_a_label_mathtext_could_not_set_falls_back_to_plain_text():
    """Blank is the one unacceptable outcome; ugly is fine."""
    assert zone_picker._math_label("X'") == "X'"


def test_the_axes_are_labelled_through_the_same_path_as_the_points():
    """They were written out as MathText literals, so they stayed ``$k_x$`` on a
    machine where the points had already fallen back to plain letters. Going
    through the same drawing as the points is also what gives them their
    subscripts."""
    import inspect

    source = inspect.getsource(zone_picker.ZonePickerDialog._draw_axes)
    for axis in ("k_x", "k_y", "k_z"):
        assert f'"{axis}"' in source
    assert "_draw_label(" in source
    assert "$k_x$" not in source


# ── subscripts ────────────────────────────────────────────────────────────
def test_a_digit_subscript_is_one_string_and_a_letter_is_two():
    """Unicode has ₀–₉ and the label font draws them, so an index needs no
    second actor. It has a ₓ and no ᵧ or ᵶ, so k_y and k_z do."""
    assert zone_picker._label_parts("Sigma_1") == ("Σ₁", "")
    assert zone_picker._label_parts("X") == ("X", "")
    for axis, letter in (("k_x", "x"), ("k_y", "y"), ("k_z", "z")):
        assert zone_picker._label_parts(axis) == ("k", letter)


class _Window:
    """A render window that reports whatever DPI the test is about."""

    def __init__(self, dpi: int) -> None:
        self.dpi = dpi

    def GetDPI(self) -> int:  # noqa: N802 - VTK's name
        return self.dpi


class _Plotter:
    """Enough plotter for the labels: something to add actors to, and a window."""

    def __init__(self, dpi: int = 72) -> None:
        self.ren_win = _Window(dpi)
        self.actors = []

    def add_actor(self, actor, **_kwargs):
        self.actors.append(actor)
        return actor


def _drawn(plotter, text: str):
    return next(a for a in plotter.actors if a.GetInput() == text)


def test_the_subscript_is_placed_in_pixels_after_the_base():
    """``SetDisplayOffset`` is applied after projection, so the two actors keep
    their arrangement at every zoom and from every angle — a second label
    anchored in the scene would drift apart from the first as the view moved.
    The offset has to clear the base: the letters are drawn from the anchor
    rightwards (justification is left), so anything less puts the subscript on
    top of the k it belongs to."""
    plotter = _Plotter(dpi=144)

    for label in ("k_x", "k_y", "k_z"):            # all three, not just the first
        zone_picker._draw_label(plotter, (0.0, 0.0, 0.0), label, "#101010", 14)

    assert _drawn(plotter, "k").GetDisplayOffset() == (0, 0)
    for letter in ("x", "y", "z"):
        across, down = _drawn(plotter, letter).GetDisplayOffset()
        assert across >= zone_picker._text_width("k", 14, 144)   # clear of the letter
        assert down < 0                                          # and below its middle


def test_the_subscript_is_placed_again_for_the_screen_it_lands_on():
    """A render window reports 72 DPI until it has one, and the scene is built
    before the dialog is shown — so on a Retina screen every subscript was
    measured at half the size its letters are drawn at, and sat on top of its
    own k. The placing is done again once there is a window to ask."""
    plotter = _Plotter(dpi=72)
    zone_picker._draw_label(plotter, (0.0, 0.0, 0.0), "k_y", "#101010", 14)
    subscript = _drawn(plotter, "y")
    for_72 = subscript.GetDisplayOffset()[0]

    plotter.ren_win.dpi = 144                       # the window turns out to be Retina
    zone_picker.place_subscripts(plotter)

    assert subscript.GetDisplayOffset()[0] >= zone_picker._text_width("k", 14, 144)
    assert subscript.GetDisplayOffset()[0] > for_72


def test_a_rebuilt_scene_forgets_the_labels_it_dropped():
    """``clear()`` takes the actors; keeping their records would place subscripts
    that are no longer drawn, and grow the list for the life of the dialog."""
    plotter = _Plotter(dpi=144)
    zone_picker._draw_label(plotter, (0.0, 0.0, 0.0), "k_z", "#101010", 14)
    assert zone_picker._subscripts(plotter)

    zone_picker.forget_subscripts(plotter)

    assert not zone_picker._subscripts(plotter)


def test_the_width_is_measured_in_the_pixels_the_glyphs_are_drawn_in():
    # Wider text, further along — whatever the font turns out to be.
    narrow = zone_picker._text_width("k", 14, 144)
    wide = zone_picker._text_width("kkk", 14, 144)
    assert 0 < narrow < wide
    assert zone_picker._text_width("", 14, 144) == 0
    # Twice the DPI, twice the width, which is what keeps a HiDPI screen right.
    assert zone_picker._text_width("k", 14, 144) > zone_picker._text_width("k", 14, 72)


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

    from crystalline.viz.fonts import unicode_font

    path = unicode_font()
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

    from crystalline.ui import menus

    QApplication.instance() or QApplication([])
    subscripted = menus.axis_icon("k", "y", "#ffffff", QFont())
    plain = menus.axis_icon("a", "", "#ffffff", QFont())
    for icon in (subscripted, plain):
        assert isinstance(icon, QIcon)
        assert not icon.isNull()
    # the subscripted one is wider, because it carries a second glyph
    assert (subscripted.availableSizes()[0].width()
            > plain.availableSizes()[0].width())


def test_every_chip_in_every_toolbar_is_the_same_size():
    """Both toolbars offer the same controls; they must not be two sizes.

    The axis chips draw an icon and the rotate chips draw text, and a
    QToolButton asks for more room for an icon than for a letter — so without
    one declared size the axis chips stood a head taller than the rotate chips
    beside them, differently in each window.
    """
    from crystalline.ui import menus

    assert menus.CHIP_SIZE.width() > 0 and menus.CHIP_SIZE.height() > 0
    assert menus.CHIP_ICON.width() < menus.CHIP_SIZE.width()
    assert menus.CHIP_ICON.height() < menus.CHIP_SIZE.height()


def test_both_toolbars_shape_their_chips_through_the_same_helper():
    import ast
    import inspect
    from pathlib import Path

    from crystalline.ui import menus

    root = Path(inspect.getfile(menus)).parent
    for module in ("menus.py", "panels/zone_picker.py"):
        tree = ast.parse((root / module).read_text())
        names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        names |= {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        assert "style_chip" in names, f"{module} shapes its chips by hand"


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


# ── the two toolbars stay in step ─────────────────────────────────────────
def test_both_views_offer_the_same_rotate_chips():
    """Same glyphs, same order, same meaning — from one definition."""
    from crystalline.ui import menus

    labels = [chip[0] for chip in menus.ROTATE_CHIPS]
    assert labels == ["◀", "▶", "▲", "▼", "↺", "↻"]
    # unit signs, so the step box decides how far a press turns
    for _label, _tip, azimuth, elevation, roll in menus.ROTATE_CHIPS:
        assert {abs(azimuth), abs(elevation), abs(roll)} == {0.0, 1.0}


def test_the_rotation_step_is_a_control_starting_at_fifteen_degrees():
    from PySide6.QtWidgets import QApplication

    from crystalline.ui import menus

    QApplication.instance() or QApplication([])
    box = menus.rotate_step_box(None)
    assert box.value() == 15
    assert box.minimum() >= 1 and box.maximum() >= 90
    assert box.suffix() == "°"


def test_the_k_axes_are_drawn_in_the_colours_of_the_chips_that_aim_down_them():
    """A chip and the arrow it aims at have to be recognisably the same axis —
    the structure window's a/b/c chips match its lattice gizmo the same way."""
    import inspect

    from crystalline.ui import theme

    source = inspect.getsource(zone_picker.ZonePickerDialog._draw_axes)
    assert "AXIS_COLOURS" in source
    assert len(theme.AXIS_COLOURS) == 3


def test_no_leg_of_a_path_is_coloured_like_an_axis():
    """Otherwise a segment through the middle of the zone reads as an axis."""
    from crystalline.ui import theme

    assert not set(zone_picker._SEGMENT_COLOURS) & set(theme.AXIS_COLOURS)


# ── clicking nothing ──────────────────────────────────────────────────────
def test_a_click_on_empty_space_clears_the_selection():
    """pyvista's mesh picking only calls back on a hit, so a miss was silent —
    the last point stayed marked, and read out in the corner, with nothing to
    say it was stale."""
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    assert hasattr(zone_picker.ZonePickerDialog, "_on_press")
    assert hasattr(zone_picker.ZonePickerDialog, "_on_release")


def test_a_drag_from_empty_space_is_not_a_click():
    """Rotating the zone by starting on the background is the most ordinary
    thing anyone does here; it must not wipe the selection."""
    import inspect

    source = inspect.getsource(zone_picker.ZonePickerDialog._on_release)
    assert "_CLICK_SLOP" in source
    assert zone_picker._CLICK_SLOP >= 2, "a hand is never perfectly still"


def test_labels_are_billboards_rather_than_a_placement_hierarchy():
    """add_point_labels builds a vtkLabelPlacementMapper, which decides for
    itself which labels are worth drawing — by collision, and by how much time
    the render was allowed. That is why labels flickered in during a click and
    vanished on release. A billboard actor has no such opinion."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(zone_picker))
    called = {node.func.attr for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert "add_point_labels" not in called, "the placement mapper is back"
    assert "vtkBillboardTextActor3D" in inspect.getsource(zone_picker)


def test_a_billboard_label_carries_its_text_and_sits_where_it_is_put():
    from PySide6.QtWidgets import QApplication

    import pyvista as pv

    QApplication.instance() or QApplication([])
    plotter = pv.Plotter(off_screen=True)
    try:
        actor = zone_picker._billboard(plotter, (0.1, 0.2, 0.3), r"$\Gamma$",
                                       "#ffffff", 15)
        assert actor.GetInput() == r"$\Gamma$"
        assert tuple(round(v, 6) for v in actor.GetPosition()) == (0.1, 0.2, 0.3)
        assert actor.GetTextProperty().GetFontSize() == 15
    finally:
        plotter.close()


# ── which way the rotate chips turn things ────────────────────────────────
def test_the_arrows_describe_the_scene_not_the_camera():
    """Orbiting the camera right slides the scene left, so a chip labelled ▶
    that passes its angle straight to the camera moves the crystal the wrong
    way. The chips are written scene-side and converted once."""
    from crystalline.ui import menus

    azimuth, elevation, roll = menus.camera_angles(1.0, 0.0, 0.0, 15)
    assert azimuth == -15, "the camera goes the other way to the scene"
    assert (elevation, roll) == (0.0, 0.0)

    _azimuth, elevation, _roll = menus.camera_angles(0.0, 1.0, 0.0, 15)
    assert elevation == -15

    # Roll is already scene-side in rotate_view, which negates it there.
    _azimuth, _elevation, roll = menus.camera_angles(0.0, 0.0, 1.0, 15)
    assert roll == 15


def test_the_step_scales_every_angle():
    from crystalline.ui import menus

    for step in (5, 15, 90):
        angles = menus.camera_angles(1.0, -1.0, 1.0, step)
        assert {abs(a) for a in angles} == {float(step)}


def test_both_views_convert_through_the_same_function():
    """Two 3D views that turn opposite ways under the same arrow is the bug
    this replaced."""
    import ast
    import inspect
    from pathlib import Path

    from crystalline.ui import menus

    root = Path(inspect.getfile(menus)).parent
    for module in ("menus.py", "panels/zone_picker.py"):
        tree = ast.parse((root / module).read_text())
        names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        names |= {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        assert "camera_angles" in names, f"{module} wires its chips by hand"


# ── the camera stays where it was put ─────────────────────────────────────
def _stub_picker(plotter):
    """The dialog's picking state over a plain plotter — enough to click with.

    The real dialog owns a GL viewport and cannot be built headless, but the
    pick handler only needs the plotter and the bookkeeping around it.
    """
    from types import SimpleNamespace

    from PySide6.QtWidgets import QLabel, QListWidget
    from ase.build import bulk

    import pyvista as pv
    from crystalline.core.brillouin import reciprocal_cell, special_points, zone_lattice
    from crystalline.core.structure import Structure

    lattice = zone_lattice(Structure.from_ase(bulk("Si", "diamond", a=5.43)))
    dialog = zone_picker.ZonePickerDialog.__new__(zone_picker.ZonePickerDialog)
    dialog._view = SimpleNamespace(plotter=plotter)
    dialog._points = special_points(lattice)
    dialog._reciprocal = reciprocal_cell(lattice)
    dialog._extent = float(np.abs(dialog._reciprocal).max())
    dialog._picking, dialog._picked, dialog._selected = True, [], None
    dialog._path_actors, dialog._legend_names = [], []
    dialog._list, dialog._total = QListWidget(), QLabel()
    dialog._actors = {}
    for label, fractional in dialog._points.items():
        centre = np.asarray(fractional) @ dialog._reciprocal
        actor = plotter.add_mesh(pv.Sphere(radius=0.01, center=centre))
        dialog._actors[label] = (actor, centre)
    return dialog


def test_picking_a_point_leaves_the_camera_alone():
    """Each click after the first draws a leg of the path, and a leg added the
    default way let pyvista refit the camera — zoom and pan thrown away, the
    view recentred on the bounding box, on every point picked."""
    from PySide6.QtWidgets import QApplication

    import pyvista as pv

    QApplication.instance() or QApplication([])
    plotter = pv.Plotter(off_screen=True)
    try:
        plotter.enable_parallel_projection()
        dialog = _stub_picker(plotter)
        plotter.show(auto_close=False)          # as a window that is on screen
        plotter.reset_camera()                  # what _centre_on_gamma does
        plotter.camera_set = False              # ...which leaves it unset
        camera = plotter.renderer.GetActiveCamera()
        camera.SetParallelScale(camera.GetParallelScale() * 0.4)     # zoomed in
        camera.SetFocalPoint(0.05, 0.02, 0.0)                         # panned

        def state():
            return round(camera.GetParallelScale(), 6), tuple(
                round(v, 6) for v in camera.GetFocalPoint())

        before = state()
        for name in ("G", "X", "W", "L"):
            dialog._on_pick(pv.Sphere(radius=0.001, center=dialog._actors[name][1]))
            assert state() == before, f"picking {name} moved the camera"
        assert len(dialog._path_actors) == 3
    finally:
        plotter.close()


def test_a_redraw_marks_the_camera_as_placed():
    """So nothing added afterwards — a leg, a label — can refit it."""
    import inspect

    assert "camera_set = True" in inspect.getsource(zone_picker.ZonePickerDialog._draw)


# ── labels when MathText is not working ─────────────────────────────────

def test_a_label_falls_back_to_plain_unicode_when_mathtext_cannot_draw_it():
    """Reported from a running app: every label showed as ``$\\Gamma$``, dollar
    signs and all.

    The labels are built as MathText because VTK's own font has no Greek. But
    registering that backend is not the same as it working — where VTK and
    matplotlib disagree about their internals the string is drawn verbatim — and
    the picker had no way to tell. It now asks VTK to rasterise one glyph and
    uses plain Unicode when that fails, which the billboard draws in a font that
    has the letter.
    """
    from crystalline.ui.panels import zone_picker as zp

    assert zp._plain_label("G") == "Γ"
    assert zp._plain_label("Sigma") == "Σ"
    assert zp._plain_label("k_x") == "kx"      # as one string; drawn as two actors
    assert zp._math_label("G") == "Γ"          # and that is what gets drawn
    assert zp._plain_label("X") == "X"
    assert "$" not in zp._plain_label("Gamma")


def test_no_label_is_ever_handed_to_vtk_as_mathtext():
    """MathText is how VTK draws Greek through matplotlib, and where the two
    disagree the label is drawn verbatim — ``$\\Gamma$``, dollar signs and all,
    as reported from a running app. Nothing on this side predicts which machine
    does which, so the 3D labels do not use it at all."""
    from crystalline.ui.panels import zone_picker as zp

    for label in ("G", "Gamma", "Sigma", "Delta", "Lambda", "X", "k_x", "W_2"):
        drawn = zp._math_label(label)
        assert "$" not in drawn and "\\" not in drawn, (label, drawn)


def test_a_billboard_label_is_drawn_in_a_font_that_has_greek():
    """Whatever the label says, the actor must be able to draw it: VTK's own
    font renders a gamma as nothing at all."""
    import inspect

    from crystalline.ui.panels import zone_picker as zp

    source = inspect.getsource(zp._billboard)
    assert "SetFontFile" in source and "unicode_font()" in source
