"""The Display dock round-trips every render setting.

The panel rebuilds a whole :class:`RenderSettings` from its widgets on each
change, so a field that gets a widget but is left out of that rebuild silently
reverts to its default the moment anything else is touched. One round-trip over
a fully non-default settings object catches that for every field at once,
including ones added later.
"""

import os
from dataclasses import fields

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from crystalline.ui.panels.display_settings import DisplayPanel  # noqa: E402
from crystalline.viz.render_settings import RenderSettings  # noqa: E402

# Every field set away from its default, and within each widget's range.
_TWEAKED = RenderSettings(
    atom_scale=0.75,
    atom_opacity=0.6,
    atom_colors=((8, "#123456"),),
    show_bonds=False,
    bond_radius=0.2,
    bond_tolerance=1.4,
    bond_color="#abcdef",
    show_hydrogen_bonds=False,
    hydrogen_bond_color="#fedcba",
    hydrogen_bond_width=3.5,
    show_cell=False,
    cell_color="#aa00aa",
    cell_line_width=4.0,
    show_lattice_vectors=False,
    show_adp_ellipsoids=True,
    adp_probability=0.9,
    adp_opacity=0.4,
    adp_temperature_index=2,
    show_mode_arrows=True,
    mode_arrow_scale=2.5,
    mode_arrow_color="#00ff00",
    show_atom_labels=True,
    atom_label_size=22,
    atom_label_color="#0000ff",
    show_polyhedra=False,
    polyhedra_opacity=0.8,
    polyhedra_min_vertices=6,
    polyhedra_edge_width=3.0,
    measure_point_color="#111111",
    measure_line_color="#222222",
    measure_plane_color="#333333",
    background_color="#444444",
    parallel_projection=False,
    show_orientation_axes=True,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_every_setting_survives_a_trip_through_the_panel(qapp):
    emitted = []
    panel = DisplayPanel(_TWEAKED, emitted.append)
    # The ellipsoid controls are driven by the loaded file: with no temperatures
    # the panel forces them off, so give it some before checking the round-trip.
    panel.set_adp_temperatures(["10 K", "150 K", "300 K"])
    panel._show_adp.setChecked(_TWEAKED.show_adp_ellipsoids)
    emitted.clear()

    panel._emit()

    assert len(emitted) == 1
    for field in fields(RenderSettings):
        assert getattr(emitted[0], field.name) == getattr(_TWEAKED, field.name), field.name


def test_loading_settings_does_not_emit(qapp):
    """Building the panel must not look like a user change, or opening the dock
    would push a redraw before anything was touched."""
    emitted = []
    DisplayPanel(_TWEAKED, emitted.append)

    assert emitted == []


def test_toggling_a_control_reports_the_change(qapp):
    emitted = []
    panel = DisplayPanel(_TWEAKED, emitted.append)

    panel._show_arrows.setChecked(False)

    assert emitted[-1].show_mode_arrows is False
    assert emitted[-1].mode_arrow_scale == _TWEAKED.mode_arrow_scale  # nothing else moved


def test_ellipsoids_cannot_be_switched_on_without_adp_data(qapp):
    """Most frequency runs carry no ADPs; the controls say so rather than
    offering a toggle that would draw nothing."""
    emitted = []
    panel = DisplayPanel(_TWEAKED, emitted.append)
    panel.set_adp_temperatures(["10 K", "300 K"])
    panel._show_adp.setChecked(True)
    emitted.clear()

    panel.set_adp_temperatures([])  # a file with no ADP section

    assert not panel._show_adp.isChecked()
    assert emitted[-1].show_adp_ellipsoids is False  # stale ellipsoids cleared
    assert "no ADP data" in panel._show_adp.toolTip()
    assert not panel._adp_temp.isEnabled()


def test_the_temperature_picker_follows_the_loaded_file(qapp):
    emitted = []
    panel = DisplayPanel(_TWEAKED, emitted.append)

    panel.set_adp_temperatures(["10 K", "82.5 K", "300 K"])
    assert panel._adp_temp.isEnabled()
    assert [panel._adp_temp.itemText(i) for i in range(panel._adp_temp.count())] == [
        "10 K", "82.5 K", "300 K"
    ]
    # the index the settings asked for is honoured on the first file
    assert panel._adp_temp.currentIndex() == _TWEAKED.adp_temperature_index

    emitted.clear()
    panel._adp_temp.setCurrentIndex(1)
    assert emitted[-1].adp_temperature_index == 1

    # A file with fewer temperatures must not leave the index out of range.
    panel.set_adp_temperatures(["150 K"])
    panel._emit()
    assert emitted[-1].adp_temperature_index == 0


def test_opening_a_file_with_adps_shows_them():
    """A run that went to the trouble of computing ADPs is one whose ADPs you
    want to see, so the loader switches the ellipsoids on."""
    emitted = []
    panel = DisplayPanel(RenderSettings(), emitted.append)
    assert not panel._show_adp.isChecked()  # nothing loaded yet

    panel.set_adp_temperatures(["10 K", "300 K"], autoshow=True)

    assert panel._show_adp.isChecked()
    assert emitted[-1].show_adp_ellipsoids is True  # and they are drawn, not just ticked


def test_a_view_change_does_not_overrule_switching_them_off():
    """Cell view, supercell and boundary changes re-list the same temperatures;
    none of them is a reason to turn the ellipsoids back on."""
    emitted = []
    panel = DisplayPanel(RenderSettings(), emitted.append)
    panel.set_adp_temperatures(["10 K", "300 K"], autoshow=True)
    panel._show_adp.setChecked(False)  # the user doesn't want them

    panel.set_adp_temperatures(["10 K", "300 K"])  # a supercell change, say

    assert not panel._show_adp.isChecked()


def test_opening_a_file_without_adps_leaves_them_off():
    emitted = []
    panel = DisplayPanel(RenderSettings(), emitted.append)

    panel.set_adp_temperatures([], autoshow=True)

    assert not panel._show_adp.isChecked()
    assert not panel._adp_temp.isEnabled()


def test_the_default_probability_is_the_one_the_picker_starts_on():
    panel = DisplayPanel(RenderSettings(), lambda _s: None)

    assert RenderSettings().adp_probability == 0.99
    assert panel._adp_probability.currentText() == "99 %"


def test_a_slider_drag_is_coalesced_into_one_change(qapp):
    """A slider spans 1000 steps, and each step can cost the renderer a full scene
    rebuild — dragging one used to emit hundreds of times. The continuous controls
    wait for the drag to settle; discrete ones must stay immediate."""
    emitted = []
    panel = DisplayPanel(RenderSettings(), emitted.append)

    for value in (0.55, 0.60, 0.65, 0.70):
        panel._atom_scale.setValue(value)
    assert emitted == []  # nothing reported while the drag is still moving

    qapp.processEvents()
    _wait_for_settle(qapp, panel)

    assert len(emitted) == 1  # one change for the whole drag...
    assert emitted[-1].atom_scale == pytest.approx(0.70)  # ...carrying where it ended


def test_a_discrete_control_reports_immediately_and_flushes_a_pending_drag(qapp):
    """Waiting would add latency to a single click for nothing. And a slider tick
    still sitting on the timer must not overtake it, or the older settings would
    arrive last and undo the click."""
    emitted = []
    panel = DisplayPanel(RenderSettings(), emitted.append)

    panel._atom_scale.setValue(0.8)  # pending, not yet reported
    panel._show_bonds.setChecked(False)  # a click: immediate

    assert len(emitted) == 1
    assert emitted[-1].show_bonds is False
    assert emitted[-1].atom_scale == pytest.approx(0.8)  # the pending value came along

    _wait_for_settle(qapp, panel)
    assert len(emitted) == 1  # the flushed tick does not fire again afterwards


def _wait_for_settle(qapp, panel, timeout_ms: int = 2000) -> None:
    """Pump the event loop until the panel's debounce timer has fired."""
    from time import monotonic

    deadline = monotonic() + timeout_ms / 1000.0
    while panel._settle.isActive() and monotonic() < deadline:
        qapp.processEvents()
    qapp.processEvents()


# ── layout: the thing that made the panel look unfinished ─────────────────
def test_every_control_starts_at_the_same_x(qapp):
    """Rows used to be QFormLayout rows with right-aligned labels, so a control
    began wherever its own label happened to end — "Show bonds", "Radius (Å)"
    and "Show hydrogen bonds" each started somewhere different, and that ragged
    edge was the loudest thing on the panel.

    A fixed label column puts every control on one line down the whole panel.
    """
    from PySide6.QtWidgets import QGridLayout

    from crystalline.ui.panels import controls
    from crystalline.ui.panels.display_settings import DisplayPanel

    panel = DisplayPanel(RenderSettings(), lambda s: None)
    panel.set_elements([11, 17])

    grids = panel.findChildren(QGridLayout)
    assert grids, "the panel should be built on grids"
    for grid in grids:
        assert grid.columnMinimumWidth(0) == controls.LABEL_COLUMN
        assert grid.columnStretch(1) >= 1  # the control column takes the slack


def test_checkboxes_carry_their_own_text(qapp):
    """Split across a label column and a bare box, a column of checkboxes put
    every tick at a different x. A checkbox reads as one thing."""
    from crystalline.ui.panels import display_settings as ds

    panel = ds.DisplayPanel(RenderSettings(), lambda s: None)

    assert panel._show_hbonds.text() == "Hydrogen bonds"
    assert panel._show_labels.text() == "Element labels"


def test_the_panel_never_scrolls_sideways(qapp):
    """Sideways it would hide the controls it exists to present."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QScrollArea

    from crystalline.ui.panels.display_settings import DisplayPanel

    panel = DisplayPanel(RenderSettings(), lambda s: None)
    scroll = panel.findChild(QScrollArea)

    assert scroll is not None
    assert scroll.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff


def test_a_section_title_is_shown_as_written(qapp):
    """Regression, three times over: the title went through a QGroupBox, where Qt
    reads "&" as a mnemonic and "Cell & axes" drew as "Cell _axes"; then through
    a QLabel, where the "&&" that fixed it drew literally; and now through a
    QToolButton, which is a mnemonic context again.

    Checked on what Qt will *display*, not on the string handed over, since the
    right answer differs by widget."""
    from PySide6.QtWidgets import QToolButton

    from crystalline.ui.panels.display_settings import DisplayPanel

    panel = DisplayPanel(RenderSettings(), lambda s: None)
    headers = [
        button for button in panel.findChildren(QToolButton)
        if button.property("role") == "section"
    ]
    assert headers, "sections should have clickable headers"

    # A QToolButton renders "&&" as one "&" and a lone "&" as an underline.
    shown = [button.text().replace("&&", "&") for button in headers]
    assert "MEASUREMENTS & SYMMETRY" in shown
    assert not any(header.count("&") != header.count("&&") * 2
                   for header in (b.text() for b in headers))


def test_sections_fold_and_unfold(qapp):
    """Every section starts folded: the panel opens as a short list of the things
    in the view, each a click away from its settings. Three open sections used
    to make it a wall of controls before anything had been asked for."""
    from PySide6.QtWidgets import QToolButton

    from crystalline.ui.panels.display_settings import DisplayPanel

    panel = DisplayPanel(RenderSettings(), lambda s: None)
    headers = {
        button.text().replace("&&", "&"): button for button in panel.findChildren(QToolButton)
        if button.property("role") == "section"
    }

    assert headers and not any(button.isChecked() for button in headers.values())
    assert list(headers) == ["ATOMS", "BONDS", "UNIT CELL", "COORDINATION POLYHEDRA",
                             "THERMAL ELLIPSOIDS", "PHONON ARROWS",
                             "MEASUREMENTS & SYMMETRY", "VIEW"]

    view = headers["VIEW"]
    view.setChecked(True)
    assert view.arrowType() == Qt.DownArrow
    view.setChecked(False)
    assert view.arrowType() == Qt.RightArrow


def test_what_can_be_hidden_is_switched_from_its_folded_header(qapp):
    """With every section folded, a "Show bonds" inside the section was out of
    sight; the switch on the header needs no unfolding."""
    emitted = []
    panel = DisplayPanel(RenderSettings(), emitted.append)

    for box, field in ((panel._show_bonds, "show_bonds"), (panel._show_cell, "show_cell"),
                       (panel._show_poly, "show_polyhedra"),
                       (panel._show_arrows, "show_mode_arrows")):
        assert not box.text() and box.toolTip()          # a bare switch, explained
        assert box.isVisibleTo(panel) or not panel.isVisible()
        box.setChecked(not box.isChecked())
        assert getattr(emitted[-1], field) == box.isChecked()


def test_the_cell_bonds_and_labels_can_be_restyled(qapp):
    """They were drawn in fixed colours and widths no control reached."""
    emitted = []
    panel = DisplayPanel(RenderSettings(), emitted.append)

    panel._cell_width.setValue(4.5)
    panel._emit()

    assert emitted[-1].cell_line_width == 4.5
    for attr in ("_cell_color", "_bond_color", "_hbond_color", "_atom_label_color"):
        assert attr in panel._color_buttons


def test_label_colour_follows_the_background_until_one_is_chosen(qapp):
    """A fixed near-black was unreadable on the dark ground of the dark theme."""
    from crystalline.viz.renderer import _readable_on

    panel = DisplayPanel(RenderSettings(), lambda s: None)
    swatch = panel._color_buttons["_atom_label_color"]

    panel.set_background("#15181c")
    assert panel._atom_label_color == ""                       # still automatic
    assert _readable_on("#15181c") in swatch.styleSheet()      # and shown as white


def test_reset_restores_the_defaults_but_keeps_the_background(qapp):
    """The background follows the app's light or dark appearance; a reset that
    turned a dark viewport white would read as a fault."""
    emitted = []
    panel = DisplayPanel(_TWEAKED, emitted.append)

    panel.reset()

    expected = RenderSettings()
    for field in fields(RenderSettings):
        if field.name in ("background_color", "adp_temperature_index", "show_adp_ellipsoids"):
            continue
        assert getattr(emitted[-1], field.name) == getattr(expected, field.name), field.name
    assert emitted[-1].background_color == _TWEAKED.background_color


def test_new_appearance_settings_reach_the_actors_without_a_rebuild():
    pv = pytest.importorskip("pyvista")
    from dataclasses import replace

    import numpy as np
    from ase.build import bulk

    from crystalline.core.structure import Structure
    from crystalline.viz.renderer import StructureRenderer

    plotter = pv.Plotter(off_screen=True)
    try:
        renderer = StructureRenderer(plotter)
        renderer.set_settings(replace(RenderSettings(), show_atom_labels=True,
                                      background_color="#15181c"))
        renderer.set_structure(Structure.from_ase(bulk("MgO", "rocksalt", a=4.21, cubic=True)))
        rebuilds = []
        original = renderer._rebuild
        renderer._rebuild = lambda: (rebuilds.append(1), original())

        renderer.set_settings(replace(renderer.settings, cell_color="#ff0000",
                                      cell_line_width=5.0, bond_color="#00ff00"))

        assert rebuilds == []
        assert np.allclose(renderer._cell_actor.GetProperty().GetColor(), (1, 0, 0))
        assert renderer._cell_actor.GetProperty().GetLineWidth() == 5.0
        assert np.allclose(renderer._bond_actor.GetProperty().GetColor(), (0, 1, 0))
        labels = renderer._label_actor.GetMapper().GetInputAlgorithm().GetTextProperty()
        assert np.allclose(labels.GetColor(), (1, 1, 1))          # white on the dark ground

        renderer.set_settings(replace(renderer.settings, background_color="white"))
        assert np.allclose(labels.GetColor(), (0, 0, 0))          # and black on a light one
    finally:
        plotter.close()


def test_reset_sits_in_a_footer_that_does_not_scroll_away(qapp):
    from PySide6.QtWidgets import QScrollArea

    panel = DisplayPanel(RenderSettings(), lambda s: None)
    scroll = panel.findChild(QScrollArea)

    assert not scroll.widget().isAncestorOf(panel._reset_button)
    assert panel.isAncestorOf(panel._reset_button)


def test_the_projection_is_named_orthographic(qapp):
    panel = DisplayPanel(RenderSettings(), lambda s: None)

    assert [panel._projection.itemText(i) for i in range(panel._projection.count())] == [
        "Perspective", "Orthographic"]
