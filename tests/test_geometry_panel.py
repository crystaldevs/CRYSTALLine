"""The Geometry panel: measuring the selection, and the atom-edit gate."""

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from crystalline.core import measure as M  # noqa: E402
from crystalline.core.structure import Structure  # noqa: E402
from crystalline.ui.panels.geometry_panel import GeometryPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def sections_remembered(monkeypatch):
    """Which sections are open is kept in QSettings: keep the tests' in a dict,
    out of the user's settings — and the user's out of the tests."""
    from crystalline.ui import preferences

    stored = {}
    monkeypatch.setattr(preferences, "section_open",
                        lambda name, default=True: stored.get(name, default))
    monkeypatch.setattr(preferences, "set_section_open",
                        lambda name, open_: stored.__setitem__(name, bool(open_)))
    return stored


def _water() -> Structure:
    s = Structure.empty()
    s.add_atom("O", [0.0, 0.0, 0.0])
    s.add_atom("H", [0.9572, 0.0, 0.0])
    s.add_atom("H", [-0.2400, 0.9266, 0.0])
    return s


def test_measuring_two_atoms_lists_a_distance(qapp):
    panel = GeometryPanel(_water())
    emitted = []
    panel.annotations_changed.connect(lambda anns: emitted.append(list(anns)))

    panel.set_selection([0, 1])
    panel._measure_selection()

    assert len(panel.measurements()) == 1
    result = panel.measurements()[0]
    assert result.kind == M.DISTANCE
    assert result.value == pytest.approx(0.9572, abs=1e-4)
    # listed, but not drawn until ticked: that is the user's to decide
    assert emitted and emitted[-1] == []
    assert panel._items()[0].checkState(0) == Qt.Unchecked


def test_measuring_three_atoms_gives_an_angle(qapp):
    panel = GeometryPanel(_water())
    panel.set_selection([1, 0, 2])

    panel._measure_selection()
    assert panel.measurements()[-1].kind == M.ANGLE
    assert panel.measurements()[-1].value == pytest.approx(104.5, abs=0.05)


def test_five_atoms_are_for_a_plane_fit_not_a_measurement(qapp):
    from ase.build import molecule

    panel = GeometryPanel(Structure.from_ase(molecule("C6H6")))
    panel.set_selection(range(5))
    assert not panel._measure_btn.isEnabled()
    assert "Lattice planes" in panel._hint.text()
    assert panel._fit_plane_btn.isEnabled()          # where the plane went


def test_a_new_measurement_is_listed_and_drawn_only_once_ticked(qapp):
    """Whether a measurement goes into the view is the user's to decide: drawn
    straight away, every bond measured for its force constant put a line and a
    label over it."""
    panel = GeometryPanel(_water())
    shown = []
    panel.annotations_changed.connect(lambda anns: shown.append(list(anns)))
    panel.set_selection([0, 1])
    panel._measure_selection()
    assert panel.shown_annotations() == [] and len(panel.measurements()) == 1
    panel._items()[0].setCheckState(0, Qt.Checked)     # the tick is the first column's
    assert len(panel.shown_annotations()) == 1 and len(shown[-1]) == 1


def test_unticking_a_measurement_hides_it_without_deleting_it(qapp):
    panel = GeometryPanel(_water())
    shown = []
    panel.annotations_changed.connect(lambda anns: shown.append(list(anns)))
    panel.set_selection([0, 1])
    panel._measure_selection()
    panel._items()[0].setCheckState(0, Qt.Checked)
    assert len(panel.shown_annotations()) == 1

    panel._items()[0].setCheckState(0, Qt.Unchecked)
    assert panel.shown_annotations() == []       # nothing drawn
    assert len(panel.measurements()) == 1        # but still listed
    assert shown[-1] == []                       # and the viewport was told


def test_clearing_and_removing_measurements(qapp):
    panel = GeometryPanel(_water())
    panel.set_selection([0, 1])
    panel._measure_selection()
    panel.set_selection([1, 0, 2])
    panel._measure_selection()
    assert len(panel.measurements()) == 2

    panel._list.topLevelItem(0).setSelected(True)
    panel._remove_selected_measurements()
    assert len(panel.measurements()) == 1
    assert panel.measurements()[0].kind == M.ANGLE  # the right one went

    panel.clear_measurements()
    assert panel.measurements() == [] and panel.shown_annotations() == []


def test_a_new_structure_drops_stale_measurements(qapp):
    panel = GeometryPanel(_water())
    panel.set_selection([0, 1])
    panel._measure_selection()

    panel.set_structure(_water())  # e.g. a supercell or a freshly loaded file
    assert panel.measurements() == []


def test_atom_tools_are_gated_on_editing_mode(qapp):
    """The trap this panel exists to expose: selecting atoms is not enough —
    editing mode must be on before Delete does anything."""
    panel = GeometryPanel(_water())
    panel.set_selection([0, 1])
    assert not panel._delete_btn.isEnabled()  # selected, but editing is off

    panel.set_editing_enabled(True)
    assert panel._delete_btn.isEnabled()
    assert panel._duplicate_btn.isEnabled()
    assert panel._set_element_btn.isEnabled()
    assert panel._add_btn.isEnabled()  # adding needs no selection

    panel.set_selection([])
    assert not panel._delete_btn.isEnabled()  # editing on, nothing selected
    assert panel._add_btn.isEnabled()


def test_the_editing_checkbox_reflects_and_drives_the_edit_menu(qapp):
    panel = GeometryPanel(_water())
    toggles = []
    panel.editing_toggled.connect(toggles.append)

    # driven from the panel -> MainWindow is told
    panel._edit_check.setChecked(True)
    assert toggles == [True]

    # driven from the menu -> the checkbox follows, without echoing back
    panel.set_editing_enabled(False)
    assert panel._edit_check.isChecked() is False
    assert toggles == [True]


def test_edit_buttons_emit_intent_rather_than_editing_directly(qapp):
    """MainWindow runs the same slots the Edit menu uses, so undo behaves."""
    panel = GeometryPanel(_water())
    panel.set_editing_enabled(True)
    panel.set_selection([0])

    seen = []
    panel.delete_requested.connect(lambda: seen.append("delete"))
    panel.duplicate_requested.connect(lambda: seen.append("duplicate"))
    panel.translate_requested.connect(lambda: seen.append("translate"))
    panel.set_element_requested.connect(lambda s: seen.append(("element", s)))
    panel.add_atom_requested.connect(lambda s: seen.append(("add", s)))

    panel._element.setCurrentText("Fe")
    panel._delete_btn.click()
    panel._duplicate_btn.click()
    panel._translate_btn.click()
    panel._set_element_btn.click()
    panel._add_btn.click()

    assert seen == ["delete", "duplicate", "translate", ("element", "Fe"), ("add", "Fe")]
    assert len(_water()) == 3  # the panel itself changed nothing


def test_setting_a_per_item_colour(qapp, monkeypatch):
    """The 'Colour…' button recolours only the selected measurement(s); each item
    keeps its own colour, and the change reaches the emitted annotations."""
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QColorDialog

    panel = GeometryPanel(_water())
    panel.set_selection([0, 1])
    panel._measure_selection()  # a distance
    panel.set_selection([1, 0, 2])
    panel._measure_selection()  # an angle
    for item in panel._items():
        item.setCheckState(0, Qt.Checked)               # both drawn, so the redraw has them
    assert [m.color for m in panel.measurements()] == [None, None]  # default (group) colour

    emitted = []
    panel.annotations_changed.connect(lambda anns: emitted.append(list(anns)))

    # Recolour just the distance (row 0); the colour dialog is stubbed.
    monkeypatch.setattr(QColorDialog, "getColor", staticmethod(lambda *a, **k: QColor("#123456")))
    panel._list.topLevelItem(0).setSelected(True)
    panel._list.topLevelItem(1).setSelected(False)
    panel._set_measurement_colour()

    assert panel.measurements()[0].color == "#123456"  # the distance is recoloured
    assert panel.measurements()[1].color is None        # the angle is untouched
    assert emitted[-1][0].color == "#123456"            # and the redraw sees it


def test_colour_button_needs_a_selected_measurement(qapp):
    panel = GeometryPanel(_water())
    panel.set_selection([0, 1])
    panel._measure_selection()
    panel._list.clearSelection()
    panel._sync_buttons()
    assert not panel._color_btn.isEnabled()  # nothing selected → nothing to recolour
    panel._list.topLevelItem(0).setSelected(True)
    assert panel._color_btn.isEnabled()


def test_the_thickness_slider_sets_every_line_or_the_selected_ones(qapp):
    panel = GeometryPanel(_water())
    emitted = []
    panel.annotations_changed.connect(lambda anns: emitted.append(list(anns)))
    assert panel.measurement_thickness() == pytest.approx(M.DEFAULT_THICKNESS)
    panel.set_selection([0, 1])
    panel._measure_selection()
    panel.set_selection([1, 0, 2])
    panel._measure_selection()
    for item in panel._items():
        item.setCheckState(0, Qt.Checked)               # both drawn, so the redraw has them
    assert [m.thickness for m in panel.measurements()] == pytest.approx([0.07, 0.07])

    # nothing selected in the list: every line follows, live
    panel._thickness.slider.setValue(375)          # 0.20 Å: 0.02 + 0.375 × 0.48
    assert [m.thickness for m in panel.measurements()] == pytest.approx([0.2, 0.2])
    assert [m.thickness for m in emitted[-1]] == pytest.approx([0.2, 0.2])
    assert panel._thickness.box.value() == pytest.approx(0.2)      # the box follows

    # one selected: only that one, and picking another shows its value
    panel._list.topLevelItem(1).setSelected(True)
    panel._thickness.box.setValue(0.35)
    assert [m.thickness for m in panel.measurements()] == pytest.approx([0.2, 0.35])
    assert panel._thickness.slider.value() == round((0.35 - 0.02) / 0.48 * 1000)
    panel._list.clearSelection()
    panel._list.topLevelItem(0).setSelected(True)
    assert panel._thickness.box.value() == pytest.approx(0.2)
    assert [m.thickness for m in panel.measurements()] == pytest.approx([0.2, 0.35])


def test_new_measurements_take_the_thickness_shown(qapp):
    panel = GeometryPanel(_water())
    panel._thickness.box.setValue(0.3)       # nothing measured yet: sets the next one's
    panel.set_selection([0, 1])
    panel._measure_selection()
    assert panel.measurements()[0].thickness == pytest.approx(0.3)


# ── placing an atom at typed coordinates ──────────────────────────────────
def _nacl() -> Structure:
    from ase.build import bulk

    return Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64, cubic=True))


def test_position_boxes_need_exactly_one_atom_and_editing_mode(qapp):
    """Coordinates name a single atom: with none or several there is nothing a
    typed position could mean, and like every other atom tool they are gated on
    editing mode."""
    panel = GeometryPanel(_nacl())

    panel.set_selection([0])
    assert not panel._coord_boxes[0].isEnabled()   # editing still off

    panel.set_editing_enabled(True)
    assert all(spin.isEnabled() for spin in panel._coord_boxes)

    panel.set_selection([0, 1])
    assert not panel._coord_boxes[0].isEnabled()   # ambiguous with two
    panel.set_selection([])
    assert not panel._coord_boxes[0].isEnabled()


def test_position_is_shown_as_fractional_for_a_periodic_cell(qapp):
    """A crystal is specified in fractions of its cell, so that is what the boxes
    read and write — Na at the origin of rocksalt, Cl at (½, 0, 0)."""
    structure = _nacl()
    panel = GeometryPanel(structure)
    panel.set_editing_enabled(True)

    panel.set_selection([0])
    assert "fractional" in panel._coord_label.text()
    assert [s.value() for s in panel._coord_boxes] == pytest.approx([0.0, 0.0, 0.0])

    cl = next(i for i, sym in enumerate(structure.symbols) if sym == "Cl")
    panel.set_selection([cl])
    fractional = structure.positions[cl] @ np.linalg.inv(np.asarray(structure.cell))
    assert [s.value() for s in panel._coord_boxes] == pytest.approx(list(fractional), abs=1e-4)


def test_a_slab_keeps_its_aperiodic_axis_in_angstrom(qapp):
    """CRYSTAL writes a slab as fractional x, y and a height in Å, because the
    aperiodic axis carries its formal 500 Å vacuum vector — a fraction of which
    is not a coordinate anyone can read. The boxes follow the same convention."""
    structure = Structure.empty()
    structure.set_cell([[3.0, 0, 0], [0, 3.0, 0], [0, 0, 500.0]], periodic=True)
    structure._atoms.set_pbc((True, True, False))
    structure.add_atom("Pt", [1.5, 0.75, 2.1])

    panel = GeometryPanel(structure)
    panel.set_editing_enabled(True)
    panel.set_selection([0])

    assert panel._coord_label.text().endswith("(fractional / Å)")
    assert [s.value() for s in panel._coord_boxes] == pytest.approx([0.5, 0.25, 2.1])


def test_a_molecule_falls_back_to_angstrom(qapp):
    """No cell means no fraction to be of."""
    panel = GeometryPanel(_water())
    panel.set_editing_enabled(True)
    panel.set_selection([1])

    assert panel._coord_label.text().endswith("(Å)")
    assert [s.value() for s in panel._coord_boxes] == pytest.approx([0.9572, 0.0, 0.0])


def test_coordinates_round_trip_through_the_display_frame(qapp):
    """The conversion has no visible failure mode — a wrong transpose still gives
    plausible numbers — so it is pinned on a deliberately non-orthogonal cell."""
    from ase.build import bulk

    structure = Structure.from_ase(bulk("Zn", "hcp", a=2.66, c=4.95))
    panel = GeometryPanel(structure)

    for position in structure.positions:
        assert panel._from_display(panel._to_display(position)) == pytest.approx(
            position, abs=1e-12
        )


def test_changing_a_box_applies_immediately_and_carries_no_button(qapp):
    """There is no Move button: committing a value *is* the request. The signal
    carries cartesian coordinates, since that is what the model stores."""
    structure = _nacl()
    panel = GeometryPanel(structure)
    panel.set_editing_enabled(True)
    asked = []
    panel.set_position_requested.connect(asked.append)

    assert not hasattr(panel, "_move_btn")

    panel.set_selection([0])
    panel._coord_boxes[0].setValue(0.25)  # a quarter of the way along a

    assert len(asked) == 1
    assert asked[0] == pytest.approx(
        list(np.asarray([0.25, 0.0, 0.0]) @ np.asarray(structure.cell))
    )


def test_filling_the_boxes_from_the_model_is_not_mistaken_for_an_edit(qapp):
    """The boxes apply on ``valueChanged`` and are refreshed after every structure
    change, so writing an atom's own position back into them must not ask to move
    it there again — that would be an endless loop of edits."""
    panel = GeometryPanel(_nacl())
    panel.set_editing_enabled(True)
    asked = []
    panel.set_position_requested.connect(asked.append)

    panel.set_selection([0])
    panel.sync_position()      # what MainWindow calls after every edit
    panel.set_selection([1])   # and what a selection change calls

    assert asked == []


def test_position_boxes_follow_the_model_when_the_atom_moves(qapp):
    """They display a live atom, so they must re-read whenever it moves — a drag,
    an arrow-key nudge or an undo — rather than keep offering a stale position."""
    structure = _nacl()
    panel = GeometryPanel(structure)
    panel.set_editing_enabled(True)
    panel.set_selection([2])

    structure.translate_atoms([2], [0.4, 0.0, 0.0])  # moved behind the panel's back
    panel.sync_position()

    expected = structure.positions[2] @ np.linalg.inv(np.asarray(structure.cell))
    assert [s.value() for s in panel._coord_boxes] == pytest.approx(list(expected), abs=1e-4)


def test_the_position_boxes_do_not_widen_the_dock(qapp):
    """Regression: a box wide enough for "-1000.0000" made the panel's preferred
    width ~465px, and a dock takes its width from that — the Geometry dock came
    out wider than the 3D view beside it. The boxes must not drive it."""
    from PySide6.QtWidgets import QScrollArea

    panel = GeometryPanel(_nacl())
    # the panel's own width without the boxes, plus the scroll bar the panel
    # keeps room for. That width is now the Measure row's — Colour, Remove,
    # Clear all and Export side by side: 357 px with the macOS style, 384 here,
    # where the offscreen style makes every button at least 80 px wide.
    scroll_bar = panel.findChild(QScrollArea).verticalScrollBar().sizeHint().width()
    assert panel.sizeHint().width() <= 370 + scroll_bar


def test_a_typed_position_moves_the_periodic_images_too(qapp):
    """A typed move is the same operation as a dragged one.

    ``Viewport.move_atom_to`` is what a finished drag calls, so it carries the
    atom's periodic images along; typing coordinates goes through it for exactly
    that reason. Exercised unbound against an off-screen renderer, since
    ``Viewport`` itself needs a live QtInteractor.
    """
    pv = pytest.importorskip("pyvista")

    from crystalline.ui.viewport import Viewport
    from crystalline.viz.renderer import StructureRenderer

    pv.OFF_SCREEN = True
    structure = Structure.empty()
    structure.set_cell(np.eye(3) * 5.0, periodic=True)
    structure.add_atom("Na", [0.0, 0.0, 0.0])
    structure.add_atom("Na", [5.0, 0.0, 0.0])  # a periodic image of the first
    structure.add_atom("Cl", [2.5, 0.0, 0.0])  # unrelated, must not move

    plotter = pv.Plotter(off_screen=True)
    renderer = StructureRenderer(plotter)
    renderer.set_structure(structure)
    try:
        class _StubViewport:
            move_atom_to = Viewport.move_atom_to
            atom_moved = type("S", (), {"emit": staticmethod(lambda *_: None)})()

        viewport = _StubViewport()
        viewport._structure = structure
        viewport.renderer = renderer

        assert renderer.periodic_image_indices(0) == [1]  # the image is recognised

        viewport.move_atom_to(0, [0.75, 0.0, 0.0])

        assert structure.positions[0] == pytest.approx([0.75, 0.0, 0.0])  # exactly there
        assert structure.positions[1] == pytest.approx([5.75, 0.0, 0.0])  # image followed
        assert structure.positions[2] == pytest.approx([2.5, 0.0, 0.0])   # untouched
    finally:
        plotter.close()


# ── lattice planes (hkl) ──────────────────────────────────────────────────
def _type_miller(panel, h, k, l):
    for spin, value in zip(panel._miller_boxes, (h, k, l)):
        spin.setValue(value)


def _plane_panel():
    structure = _nacl()
    panel = GeometryPanel(structure)
    panel.set_miller_cell(np.asarray(structure.cell))
    drawn = []
    panel.lattice_planes_changed.connect(lambda planes, cell: drawn.append((planes, cell)))
    return panel, structure, drawn


def test_a_plane_is_drawn_from_its_indices_alone(qapp):
    panel, structure, drawn = _plane_panel()
    _type_miller(panel, 1, 1, 1)
    panel._add_plane_btn.click()                   # no atom selected, nothing fitted

    # Typed indices alone put the plane through the origin; where else it might
    # go is said by picking an atom, not by a figure in units of d.
    (plane,) = panel.lattice_planes()
    assert plane.miller == (1, 1, 1) and plane.offset == 0.0 and not plane.family
    planes, cell = drawn[-1]
    assert planes == [plane] and np.allclose(cell, structure.cell)


def test_a_plane_through_the_selected_atom_passes_through_it(qapp):
    from crystalline.core import lattice_planes as lp

    panel, structure, _drawn = _plane_panel()
    _type_miller(panel, 1, 1, 1)
    assert not panel._plane_atom_btn.isEnabled()   # needs exactly one atom
    panel.set_selection([1, 2])
    assert not panel._plane_atom_btn.isEnabled()
    panel.set_selection([1])
    assert panel._plane_atom_btn.isEnabled()
    panel._plane_atom_btn.click()

    (plane,) = panel.lattice_planes()
    assert plane.offset == pytest.approx(0.5)      # Cl at (½ 0 0): h x + k y + l z = ½
    assert 1 in lp.atoms_on(np.asarray(structure.cell), plane, structure.positions)


def test_a_family_and_the_atoms_on_it_can_be_selected(qapp):
    panel, structure, _drawn = _plane_panel()
    _type_miller(panel, 1, 1, 1)
    panel._family_check.setChecked(True)
    panel._add_plane_btn.click()
    assert panel.lattice_planes()[0].family

    chosen = []
    panel.select_atoms_requested.connect(chosen.append)
    assert not panel._plane_select_btn.isEnabled()  # a plane has to be picked in the list
    panel._plane_list.item(0).setSelected(True)
    panel._plane_select_btn.click()
    # The family runs through the origin, so it is the Na that lie on it.
    assert chosen and {structure.symbols[i] for i in chosen[0]} == {"Na"}


def test_each_plane_gets_its_own_colour_and_unticking_hides_it(qapp):
    panel, _structure, drawn = _plane_panel()
    _type_miller(panel, 1, 0, 0)
    panel._add_plane_btn.click()
    _type_miller(panel, 0, 1, 0)
    panel._add_plane_btn.click()
    first, second = panel.lattice_planes()
    assert first.color and second.color and first.color != second.color

    panel._plane_list.item(0).setCheckState(Qt.Unchecked)
    assert drawn[-1][0] == [second]
    assert panel.lattice_planes() == [first, second]  # hidden, not deleted


def test_recolouring_and_removing_planes(qapp, monkeypatch):
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QColorDialog

    panel, _structure, drawn = _plane_panel()
    panel._add_plane_btn.click()
    panel._add_plane_btn.click()
    panel._plane_list.item(1).setSelected(True)
    monkeypatch.setattr(QColorDialog, "getColor", staticmethod(lambda *a, **k: QColor("#123456")))
    panel._plane_colour_btn.click()
    assert panel.lattice_planes()[1].color == "#123456"
    assert drawn[-1][0][1].color == "#123456"

    panel._plane_remove_btn.click()
    assert len(panel.lattice_planes()) == 1
    panel.clear_lattice_planes()
    assert panel.lattice_planes() == [] and drawn[-1][0] == []


def test_zero_indices_name_no_plane(qapp):
    panel, _structure, _drawn = _plane_panel()
    _type_miller(panel, 0, 0, 0)
    assert not panel._add_plane_btn.isEnabled()
    assert not panel._plane_atom_btn.isEnabled()


def test_hexagonal_planes_are_written_with_four_indices(qapp):
    from ase.build import bulk

    zinc = bulk("Zn", "hcp", a=2.66, c=4.95)
    panel = GeometryPanel(Structure.from_ase(zinc))
    panel.set_miller_cell(zinc.cell[:])
    _type_miller(panel, 1, 0, 0)
    assert not panel._i_label.isHidden() and panel._i_label.value() == -1
    assert not panel._i_label.isEnabled()          # fixed by h and k, not typed
    panel._add_plane_btn.click()
    assert panel._plane_list.item(0).text().startswith("(1 0 -1 0) · d ")

    panel.set_miller_cell(np.asarray(_nacl().cell))
    assert panel._i_label.isHidden()


def test_planes_outlive_a_change_of_view_but_not_of_crystal(qapp):
    panel, structure, drawn = _plane_panel()
    panel._add_plane_btn.click()
    bigger = Structure.from_ase(structure.to_ase().repeat((2, 2, 2)))
    panel.set_structure(bigger)                      # a supercell, say
    panel.set_miller_cell(np.asarray(structure.cell))
    assert len(panel.lattice_planes()) == 1 and len(drawn[-1][0]) == 1
    assert panel._plane_list.item(0).text().endswith("· 16 atoms")  # recounted on the supercell
    panel.clear_lattice_planes()                     # what a new file does
    assert panel.lattice_planes() == []


def test_a_molecule_offers_no_lattice_planes_but_a_fit(qapp):
    panel = GeometryPanel(_water())
    panel.set_miller_cell(None)
    assert not panel._add_plane_btn.isEnabled()     # no lattice, so no lattice plane
    assert not panel._miller_boxes[0].isEnabled()
    panel.set_selection([0, 1, 2])
    assert panel._fit_plane_btn.isEnabled()
    panel._fit_plane_btn.click()
    assert panel._plane_list.item(0).text() == "Fit · rms 0.000 Å · 3 atoms"
    assert "through 3 atoms" in panel._plane_list.item(0).toolTip()


def test_a_plane_fitted_to_atoms_is_kept_as_fitted_and_named_by_its_nearest_hkl(qapp):
    from crystalline.core import lattice_planes as lp

    panel, structure, drawn = _plane_panel()
    cell = np.asarray(structure.cell)
    on_111 = [int(i) for i in lp.atoms_on(cell, lp.LatticePlane((1, 1, 1), 0.5), structure.positions)]
    assert not panel._fit_plane_btn.isEnabled()
    panel.set_selection(on_111[:2])
    assert not panel._fit_plane_btn.isEnabled()          # two atoms fix no plane
    panel.set_selection(on_111)
    panel._fit_plane_btn.click()

    (plane,) = panel.lattice_planes()
    assert isinstance(plane, lp.FittedPlane) and plane.color
    assert drawn[-1][0] == [plane]
    assert panel._plane_list.item(0).text() == "Fit ≈ (1 1 1) 0.0° · rms 0.000 Å · 3 atoms"
    tip = panel._plane_list.item(0).toolTip()
    assert "through 3 atoms" in tip and "(1 1 1), 0.0° away" in tip


def test_a_fitted_plane_selects_the_atoms_on_it_and_takes_the_opacity(qapp):
    panel, structure, drawn = _plane_panel()
    panel._plane_opacity.slider.setValue(600)
    panel.set_selection([0, 1, 2])
    panel._fit_plane_btn.click()
    assert panel.lattice_planes()[0].opacity == pytest.approx(0.6)

    chosen = []
    panel.select_atoms_requested.connect(chosen.append)
    panel._plane_list.item(0).setSelected(True)
    panel._plane_select_btn.click()
    assert chosen and set(chosen[0]) >= {0, 1, 2}


def test_measure_offers_no_plane_fit_any_more(qapp):
    panel = GeometryPanel(_water())
    assert not hasattr(panel, "_plane_btn")
    assert "plane" not in panel._measure_btn.toolTip().split(".")[0]


def test_the_opacity_slider_sets_every_plane_or_the_selected_ones(qapp):
    panel, _structure, drawn = _plane_panel()
    assert panel.plane_opacity() == pytest.approx(0.35)
    _type_miller(panel, 1, 0, 0)
    panel._add_plane_btn.click()
    _type_miller(panel, 0, 1, 0)
    panel._add_plane_btn.click()

    # nothing selected in the list: every plane follows the slider, live
    panel._plane_opacity.slider.setValue(700)
    assert [p.opacity for p in panel.lattice_planes()] == pytest.approx([0.7, 0.7])
    assert [p.opacity for p in drawn[-1][0]] == pytest.approx([0.7, 0.7])
    assert panel._plane_opacity.box.value() == pytest.approx(0.7)   # the box follows

    # one selected: only that one
    panel._plane_list.item(1).setSelected(True)
    panel._plane_opacity.box.setValue(0.15)
    assert [p.opacity for p in panel.lattice_planes()] == pytest.approx([0.7, 0.15])
    assert panel._plane_opacity.slider.value() == 150


def test_picking_a_plane_shows_its_opacity_and_new_planes_take_the_slider(qapp):
    panel, _structure, drawn = _plane_panel()
    panel._plane_opacity.slider.setValue(900)
    panel._add_plane_btn.click()
    assert panel.lattice_planes()[0].opacity == pytest.approx(0.9)   # added at the value shown
    panel._plane_opacity.slider.setValue(200)
    panel._add_plane_btn.click()
    assert [p.opacity for p in panel.lattice_planes()] == pytest.approx([0.2, 0.2])

    panel._plane_list.item(0).setSelected(True)
    panel._plane_opacity.box.setValue(0.55)
    count = len(drawn)
    panel._plane_list.clearSelection()
    panel._plane_list.item(1).setSelected(True)          # shows 0.2, changes nothing
    assert panel._plane_opacity.slider.value() == 200
    assert len(drawn) == count
    assert [p.opacity for p in panel.lattice_planes()] == pytest.approx([0.55, 0.2])


def test_the_opacity_row_waits_for_a_lattice_or_a_fit(qapp):
    panel = GeometryPanel(_water())
    panel.set_miller_cell(None)
    assert not panel._plane_opacity.isEnabled()
    panel.set_selection([0, 1, 2])
    panel._fit_plane_btn.click()
    assert panel._plane_opacity.isEnabled()


# ── sections ──────────────────────────────────────────────────────────────
def test_the_panel_is_three_sections_that_fold_and_are_remembered(qapp, sections_remembered):
    panel = GeometryPanel(_water())
    titles = [section.header.text() for section in panel._sections.values()]
    assert titles == ["MEASURE", "LATTICE PLANES", "ATOMS"]
    # Folded the first time: three sections of tools open at once fill the dock
    # and bury whichever one is in use.
    assert not any(panel.section_open(key) for key in panel._sections)

    panel._sections["atoms"].header.click()
    assert panel.section_open("atoms") and not panel._sections["atoms"].body.isHidden()
    assert sections_remembered == {"geometry/atoms": True}

    other = GeometryPanel(_water())                  # another tab's, or the next session's
    assert other.section_open("atoms") and not other.section_open("measure")


def test_a_panel_shown_again_takes_up_the_sections_as_last_left(qapp):
    """Each tab has its own panel: folding a section in one, then switching to
    another tab, finds it folded there too."""
    first, second = GeometryPanel(_water()), GeometryPanel(_water())
    first._sections["measure"].header.click()        # opened in the tab on screen
    assert not second.section_open("measure")        # the hidden tab's has not been shown since
    second.show()
    qapp.processEvents()
    assert second.section_open("measure")
    second.hide()


def test_folding_a_section_never_widens_the_dock(qapp, sections_remembered):
    sections_remembered.update({"geometry/measure": False, "geometry/planes": False,
                                "geometry/atoms": False})
    folded = GeometryPanel(_nacl())
    sections_remembered.clear()
    unfolded = GeometryPanel(_nacl())
    assert folded.minimumSizeHint().width() == unfolded.minimumSizeHint().width()


def test_the_plane_buttons_are_colour_remove_and_clear_all(qapp):
    """Three actions on what is listed; the atom-driven ones sit with the other
    ways a plane gets made, not among them."""
    panel, _structure, _drawn = _plane_panel()
    _type_miller(panel, 1, 1, 1)
    panel._add_plane_btn.click()
    _type_miller(panel, 1, 0, 0)
    panel._add_plane_btn.click()
    assert len(panel.lattice_planes()) == 2

    assert not panel._plane_clear_btn.isEnabled() or panel._planes  # live only with planes
    panel._plane_clear_btn.click()

    assert panel.lattice_planes() == [] and panel._plane_list.count() == 0
    assert not panel._plane_clear_btn.isEnabled()   # nothing left to clear


def test_clearing_needs_no_selection_but_removing_does(qapp):
    panel, _structure, _drawn = _plane_panel()
    _type_miller(panel, 1, 1, 1)
    panel._add_plane_btn.click()

    assert panel._plane_clear_btn.isEnabled()       # something to clear
    assert not panel._plane_remove_btn.isEnabled()  # but nothing picked to remove
    panel._plane_list.item(0).setSelected(True)
    assert panel._plane_remove_btn.isEnabled()


def test_the_position_slider_moves_a_plane_along_its_normal(qapp):
    """A plane is placed by eye against the atoms far more often than it is
    calculated, so where it sits is a slider beside its opacity rather than a
    figure typed before it is drawn."""
    panel, _structure, drawn = _plane_panel()
    _type_miller(panel, 1, 1, 1)
    panel._add_plane_btn.click()
    assert panel.lattice_planes()[0].offset == 0.0

    panel._plane_list.item(0).setSelected(True)
    panel._plane_offset.set_value(0.5)

    assert panel.lattice_planes()[0].offset == pytest.approx(0.5)
    assert drawn[-1][0][0].offset == pytest.approx(0.5)   # and the view was told
    assert "at 0.50 d" in panel._plane_list.item(0).text()   # and the row says so


def test_the_position_slider_reaches_the_other_side_of_the_origin(qapp):
    """A plane is drawn only where it cuts the cell, and with a negative index
    the cell lies on the negative side of the plane through the origin: from 0
    to 1, (-1 0 0) left the cell altogether and drew nothing."""
    panel, _structure, drawn = _plane_panel()
    _type_miller(panel, -1, 0, 0)
    panel._add_plane_btn.click()

    panel._plane_list.item(0).setSelected(True)
    panel._plane_offset.set_value(-0.5)

    assert panel.lattice_planes()[0].offset == pytest.approx(-0.5)
    assert drawn[-1][0][0].offset == pytest.approx(-0.5)
    assert "at -0.50 d" in panel._plane_list.item(0).text()
    panel._plane_offset.set_value(-1.0)                   # the neighbour on that side
    assert panel.lattice_planes()[0].offset == pytest.approx(-1.0)
    assert panel._plane_offset.slider.value() == panel._plane_offset.slider.minimum()


def test_picking_a_plane_shows_where_it_sits(qapp):
    panel, _structure, _drawn = _plane_panel()
    _type_miller(panel, 1, 1, 1)
    panel._add_plane_btn.click()
    _type_miller(panel, 1, 0, 0)
    panel._add_plane_btn.click()
    panel._plane_list.item(0).setSelected(True)
    panel._plane_offset.set_value(0.25)

    panel._plane_list.item(0).setSelected(False)
    panel._plane_list.item(1).setSelected(True)
    assert panel._plane_offset.value() == pytest.approx(0.0)   # the other one, untouched
    panel._plane_list.item(1).setSelected(False)
    panel._plane_list.item(0).setSelected(True)
    assert panel._plane_offset.value() == pytest.approx(0.25)


def test_a_fitted_plane_has_no_position_to_slide(qapp):
    """It is not a member of a family, so there is no d to move it by — and the
    slider must not try to replace a field the dataclass has not got."""
    panel = GeometryPanel(_water())
    panel.set_miller_cell(None)
    panel.set_selection([0, 1, 2])
    panel._fit_plane_btn.click()

    panel._plane_list.item(0).setSelected(True)
    assert not panel._plane_offset.isEnabled()
    panel._plane_offset.set_value(0.4)              # must not raise
    assert panel.lattice_planes()[0].rms == pytest.approx(0.0)


# ── local modes: a LOCMODES run's force constants beside the measurements ──
def _peroxide():
    """H2O2 as on screen: its four atoms, and O2 again in the cell along −a, at
    the far end of the H3···O2 contact LOCMODES lists. Source atoms 0–3, then 1."""
    s = Structure.empty()
    for element, position in (("O", [0.0, 0.0, 0.0]), ("O", [1.45, 0.0, 0.0]),
                              ("H", [-0.3, 0.9, 0.2]), ("H", [1.75, -0.2, 0.9]),
                              ("O", [1.45 - 5.0, 0.0, 0.0])):
        s.add_atom(element, position)
    return s, [0, 1, 2, 3, 1]


def _peroxide_modes():
    """The local modes of ``data/locmodes/h2o2_synthetic.out``, built by hand."""
    from crystalline.core.local_modes import LocalCoordinate, LocalModes

    origin = (0, 0, 0)
    positions = [[0.0, 0.0, 0.0], [1.45, 0.0, 0.0], [-0.3, 0.9, 0.2], [1.75, -0.2, 0.9]]
    coordinates = (
        LocalCoordinate(1, M.DISTANCE, (0, 1), (origin, origin), ("O", "O"),
                        1.45, 4.2, None, 8.0, 950.0),
        LocalCoordinate(2, M.DISTANCE, (0, 2), (origin, origin), ("O", "H"),
                        0.9695, 6.5, None, 0.95, 3400.0),
        LocalCoordinate(4, M.DISTANCE, (2, 1), (origin, (-1, 0, 0)), ("H", "O"),
                        3.3782, 0.15, None, 0.98, 500.0),
        LocalCoordinate(5, M.ANGLE, (2, 0, 1), (origin,) * 3, ("H", "O", "O"),
                        108.0247, 0.7, 0.6, 1.0, 1300.0),
    )
    return LocalModes(positions, np.eye(3) * 5.0, coordinates)


def _locmodes_panel():
    structure, source = _peroxide()
    panel = GeometryPanel(structure)
    panel.set_local_modes(_peroxide_modes())
    panel.set_source_index(source)
    return panel


def _measure(panel, *atoms):
    panel.set_selection(list(atoms))
    panel._measure_selection()
    return panel.measurements()[-1]


def _row(panel, row):
    from crystalline.ui.panels.geometry_panel import _HEADERS

    item = panel._items()[row]
    return [item.text(column) for column in range(len(_HEADERS))]


def test_a_measured_internal_coordinate_has_its_local_mode_beside_it(qapp):
    """What a LOCMODES run adds: measure a bond, and if it is one of the internal
    coordinates the run computed, its local force constant, mass and frequency
    are there beside the value."""
    panel = _locmodes_panel()
    bond = _measure(panel, 0, 1)
    assert [c.number for c in bond.local] == [1]
    assert _row(panel, 0)[1:] == ["1.450 Å", "4.200", "", "8.000", "950.0"]   # no kᵃ(d) for a bond
    angle = _measure(panel, 1, 0, 2)                      # O2–O1–H3: the same angle, backwards
    assert [c.number for c in angle.local] == [5]
    assert _row(panel, 1)[1:] == ["108.025°", "0.700", "0.600", "1.000", "1300.0"]


def test_a_measurement_that_is_none_of_them_has_nothing_beside_it(qapp):
    panel = _locmodes_panel()
    assert _measure(panel, 0, 3).local == ()               # O1···H4: not an internal coordinate
    assert _row(panel, 0)[2:] == ["", "", "", ""]
    assert "Not one of the internal coordinates" in panel._items()[0].toolTip(0)


def test_a_contact_into_the_next_cell_is_found_on_the_atom_there(qapp):
    """LOCMODES names the far atom of H3···O2 by its cell, (-1 0 0): it is the O2
    drawn a cell along −a, not the one bonded to O1."""
    panel = _locmodes_panel()
    assert [c.number for c in _measure(panel, 2, 4).local] == [4]
    assert _measure(panel, 2, 1).local == ()
    tip = panel._items()[0].toolTip(0)
    assert "LOCMODES internal coordinate 4, a bond: H3–O2 (-1 0 0)" in tip
    assert "CRYSTAL's value 3.3782 Å" in tip


def test_the_local_mode_columns_are_there_only_for_a_locmodes_run(qapp):
    from crystalline.ui.panels.geometry_panel import _LOCAL_COLUMNS

    structure, source = _peroxide()
    panel = GeometryPanel(structure)
    hidden = lambda: [panel._list.isColumnHidden(c) for c in _LOCAL_COLUMNS]  # noqa: E731
    assert hidden() == [True] * 4
    panel.set_local_modes(_peroxide_modes())
    assert hidden() == [False] * 4
    panel.set_local_modes(None)                            # the next file opened in the tab
    assert hidden() == [True] * 4


def test_sorting_moves_each_measurement_with_its_row(qapp):
    """The table sorts by whichever heading is clicked, and the measurement behind
    each row goes with it: Remove takes away the row picked, and nothing else.
    Rows with nothing in the column stay at the bottom, whichever way it runs."""
    from crystalline.ui.panels.geometry_panel import _COL_K

    panel = _locmodes_panel()
    _measure(panel, 0, 1)            # O1–O2, kᵃ 4.2
    _measure(panel, 0, 3)            # O1···H4, none
    _measure(panel, 0, 2)            # O1–H3, kᵃ 6.5
    for row in (1, 2):                                    # O1···H4 and O1–H3 drawn, O1–O2 not
        panel._items()[row].setCheckState(0, Qt.Checked)

    panel._list.header().setSortIndicator(_COL_K, Qt.DescendingOrder)
    assert [_row(panel, r)[_COL_K] for r in range(3)] == ["6.500", "4.200", ""]
    assert [round(m.value, 4) for m in panel.measurements()] == [0.9695, 1.45, pytest.approx(2.0, abs=0.1)]
    assert [m.value for m in panel.shown_annotations()] == [panel.measurements()[0].value,
                                                            panel.measurements()[2].value]
    _measure(panel, 2, 4)            # H3···O2(−a), kᵃ 0.15: sorted into place
    assert [_row(panel, r)[_COL_K] for r in range(4)] == ["6.500", "4.200", "0.150", ""]

    panel._list.topLevelItem(0).setSelected(True)
    panel._remove_selected_measurements()
    assert [_row(panel, r)[_COL_K] for r in range(3)] == ["4.200", "0.150", ""]
    assert [m.local[0].number for m in panel.measurements()[:2]] == [1, 4]


def test_a_third_click_on_a_heading_stops_sorting(qapp):
    from crystalline.ui.panels.geometry_panel import _COL_K

    panel = _locmodes_panel()
    _measure(panel, 0, 1)
    _measure(panel, 0, 2)
    header = panel._list.header()
    header.setSortIndicator(_COL_K, Qt.DescendingOrder)
    header.setSortIndicator(-1, Qt.AscendingOrder)         # cleared: left as it stands
    _measure(panel, 2, 4)
    assert [_row(panel, r)[_COL_K] for r in range(3)] == ["6.500", "4.200", "0.150"]


def test_atoms_out_of_step_with_their_source_are_not_claimed(qapp):
    """An edit that adds or removes atoms leaves the index from the last view
    behind: nothing is matched rather than the wrong thing."""
    panel = _locmodes_panel()
    panel.set_source_index([0, 1, 2])                      # three, for five atoms on screen
    assert _measure(panel, 0, 1).local == ()


def test_a_point_lists_its_coordinates_as_its_value(qapp):
    panel = _locmodes_panel()
    _measure(panel, 3)
    assert _row(panel, 0)[:2] == ["H(3)", "(1.750, -0.200, 0.900)"]
    assert _row(panel, 0)[2:] == ["", "", "", ""]


# ── exporting the table ───────────────────────────────────────────────────
def _read_csv(path):
    import csv

    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.reader(handle))


def test_the_table_is_exported_as_it_is_shown(qapp, tmp_path):
    """Every row, in the order on screen, with the numbers in full — and for a
    LOCMODES run the internal coordinate each measurement is, as CRYSTAL gives it."""
    from crystalline.ui.panels.geometry_panel import _COL_K

    panel = _locmodes_panel()
    _measure(panel, 0, 1)            # O1–O2, kᵃ 4.2
    _measure(panel, 0, 3)            # O1···H4, none
    _measure(panel, 2, 4)            # H3···O2(−a), kᵃ 0.15
    panel._list.header().setSortIndicator(_COL_K, Qt.DescendingOrder)

    target = tmp_path / "peroxide_measurements.csv"
    panel.export_csv(str(target))
    header, *rows = _read_csv(target)

    assert header == list(M.EXPORT_COLUMNS) + list(M.EXPORT_LOCAL_COLUMNS)
    assert [row[1] for row in rows] == ["O(0)–O(1)", "H(2)–O(4)", "O(0)–H(3)"]   # as sorted
    assert rows[0][:4] == ["distance", "O(0)–O(1)", "1.450000", "Å"]
    assert rows[0][4:] == ["1", "O1–O2", "1.45", "4.2", "", "8.0", "950.0",     # no K^A(D) for a bond
                           "as listed"]
    assert rows[1][4:6] == ["4", "H3–O2 (-1 0 0)"]
    assert rows[2][4:] == [""] * len(M.EXPORT_LOCAL_COLUMNS)                    # none of them


def test_a_near_linear_angle_exports_as_one_row_per_bend(qapp, tmp_path):
    """LOCMODES gives the two bends of a near-linear angle a force constant each;
    a spreadsheet wants a number in a cell, so the angle is written twice."""
    from crystalline.core.local_modes import LocalCoordinate, LocalModes

    modes = _peroxide_modes()
    second = LocalCoordinate(6, M.ANGLE, (2, 0, 1), ((0, 0, 0),) * 3, ("H", "O", "O"),
                             -171.5, 0.058, 0.230, 1.379, 1005.2)
    structure, source = _peroxide()
    panel = GeometryPanel(structure)
    panel.set_local_modes(LocalModes(modes.positions, modes.lattice,
                                     modes.coordinates + (second,)))
    panel.set_source_index(source)
    _measure(panel, 2, 0, 1)
    target = tmp_path / "bends.csv"
    panel.export_csv(str(target))
    _, *rows = _read_csv(target)
    assert [(row[4], row[7]) for row in rows] == [("5", "0.7"), ("6", "0.058")]
    assert rows[0][:4] == rows[1][:4]                       # the same measurement on both


def test_a_table_without_local_modes_exports_without_their_columns(qapp, tmp_path):
    """Any other file: what was measured, and a point's position in columns of its own."""
    panel = GeometryPanel(_water())
    panel.set_selection([0, 1])
    panel._measure_selection()
    panel.set_selection([2])
    panel._measure_selection()
    target = tmp_path / "water.csv"
    panel.export_csv(str(target))
    header, distance, point = _read_csv(target)
    assert header == list(M.EXPORT_COLUMNS) + list(M.EXPORT_POSITION_COLUMNS)
    assert distance == ["distance", "O(0)–H(1)", "0.957200", "Å", "", "", ""]
    assert point == ["point", "H(2)", "", "", "-0.240000", "0.926600", "0.000000"]
    assert target.read_bytes().startswith(b"\xef\xbb\xbf")  # a spreadsheet reads it as UTF-8


def test_the_export_button_follows_the_table(qapp):
    """And Clear all with it: both stayed greyed out after the first measurement
    until the selection next changed."""
    panel = GeometryPanel(_water())
    assert not panel._export_btn.isEnabled()
    panel.set_selection([0, 1])
    panel._measure_selection()
    assert panel._export_btn.isEnabled() and panel._clear_btn.isEnabled()
    panel.clear_measurements()
    assert not panel._export_btn.isEnabled() and not panel._clear_btn.isEnabled()


def test_the_export_is_offered_beside_the_run_and_named_after_it(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    panel = GeometryPanel(_water())
    panel.set_source_file("/runs/dafadnp/DAFADNP_FREQ_LOCMODES.out")
    panel.set_selection([0, 1])
    panel._measure_selection()
    offered = []
    chosen = tmp_path / "picked"                           # typed without the extension

    def ask(_parent, _title, suggested, _filter):
        offered.append(suggested)
        return str(chosen), "CSV files (*.csv)"

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(ask))
    panel._export_btn.click()
    assert offered == ["/runs/dafadnp/DAFADNP_FREQ_LOCMODES_measurements.csv"]
    assert (tmp_path / "picked.csv").is_file()


def test_the_measurement_buttons_span_the_table_however_wide_the_dock(qapp):
    """Colour, Remove, Clear all and Export share the table's width: the first
    starts where the table starts, the last ends where it ends, and they grow
    together as the dock is widened — as the planes' row does."""
    from PySide6.QtCore import QPoint

    panel = GeometryPanel(_water())
    buttons = (panel._color_btn, panel._remove_btn, panel._clear_btn, panel._export_btn)

    def edges(widget):
        left = widget.mapTo(panel, QPoint(0, 0)).x()
        return left, left + widget.width()

    for width in (420, 640):
        panel.resize(width, 700)
        panel.show()
        qapp.processEvents()
        table = edges(panel._list)
        assert edges(buttons[0])[0] == table[0]
        assert edges(buttons[-1])[1] == table[1]
        widths = [b.width() for b in buttons]
        assert max(widths) - min(widths) <= 2           # shared out evenly
    assert buttons[0].width() > 100                     # and wider at 640 than at their least
    panel.hide()


def test_a_measurement_found_by_symmetry_is_marked_as_such(qapp, tmp_path):
    """Its local mode is a listed coordinate's, carried over by an operation of
    the crystal: in italics in the table, the operation in the tooltip and in
    the export, so it is never taken for a coordinate the output names."""
    from crystalline.core.local_modes import LocalCoordinate, LocalModes
    from crystalline.ui.panels.geometry_panel import _COL_K

    pair = [[0.6, 0.1, 0.2], [1.5, 0.4, 0.5], [-0.6, -0.1, -0.2], [-1.5, -0.4, -0.5]]
    structure = Structure.empty()
    for element, position in zip("OHOH", pair):
        structure.add_atom(element, position)
    oh = LocalCoordinate(1, M.DISTANCE, (0, 1), ((0, 0, 0),) * 2, ("O", "H"),
                         0.95, 7.0, None, 0.95, 3500.0)
    panel = GeometryPanel(structure)
    panel.set_local_modes(LocalModes(pair, np.eye(3) * 5.0, (oh,), (8, 1, 8, 1), 2))
    panel.set_source_index([0, 1, 2, 3])

    listed = _measure(panel, 0, 1)
    mirrored = _measure(panel, 2, 3)                      # the inversion's image, not listed
    assert [c.symmetry for c in listed.local] == [None]
    assert [(c.number, c.symmetry) for c in mirrored.local] == [(1, "-x, -y, -z")]
    rows = panel._items()
    assert rows[1].text(_COL_K) == rows[0].text(_COL_K) == "7.000"
    assert rows[1].font(_COL_K).italic() and not rows[0].font(_COL_K).italic()
    assert "Not listed itself" in rows[1].toolTip(0) and "-x, -y, -z" in rows[1].toolTip(0)

    target = tmp_path / "pair.csv"
    panel.export_csv(str(target))
    _, first, second = _read_csv(target)
    assert (first[-1], second[-1]) == ("as listed", "by symmetry (-x, -y, -z)")
