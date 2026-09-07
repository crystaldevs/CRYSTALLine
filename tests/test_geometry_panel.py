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
    # new measurements are shown in 3D straight away
    assert emitted and len(emitted[-1]) == 1


def test_measuring_three_atoms_gives_an_angle_and_a_plane_on_request(qapp):
    panel = GeometryPanel(_water())
    panel.set_selection([1, 0, 2])

    panel._measure_selection()
    assert panel.measurements()[-1].kind == M.ANGLE
    assert panel.measurements()[-1].value == pytest.approx(104.5, abs=0.05)

    panel._measure_plane()  # the same three atoms, as a plane instead
    assert panel.measurements()[-1].kind == M.PLANE


def test_unticking_a_measurement_hides_it_without_deleting_it(qapp):
    panel = GeometryPanel(_water())
    shown = []
    panel.annotations_changed.connect(lambda anns: shown.append(list(anns)))
    panel.set_selection([0, 1])
    panel._measure_selection()
    assert len(panel.shown_annotations()) == 1

    panel._items()[0].setCheckState(Qt.Unchecked)
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

    panel._list.item(0).setSelected(True)
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
    panel.set_selection([0, 1, 2])
    panel._measure_plane()      # a plane
    assert [m.color for m in panel.measurements()] == [None, None]  # default (group) colour

    emitted = []
    panel.annotations_changed.connect(lambda anns: emitted.append(list(anns)))

    # Recolour just the distance (row 0); the colour dialog is stubbed.
    monkeypatch.setattr(QColorDialog, "getColor", staticmethod(lambda *a, **k: QColor("#123456")))
    panel._list.item(0).setSelected(True)
    panel._list.item(1).setSelected(False)
    panel._set_measurement_colour()

    assert panel.measurements()[0].color == "#123456"  # the distance is recoloured
    assert panel.measurements()[1].color is None        # the plane is untouched
    assert emitted[-1][0].color == "#123456"            # and the redraw sees it


def test_colour_button_needs_a_selected_measurement(qapp):
    panel = GeometryPanel(_water())
    panel.set_selection([0, 1])
    panel._measure_selection()
    panel._list.clearSelection()
    panel._sync_buttons()
    assert not panel._color_btn.isEnabled()  # nothing selected → nothing to recolour
    panel._list.item(0).setSelected(True)
    assert panel._color_btn.isEnabled()


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
    with_boxes = GeometryPanel(_nacl()).sizeHint().width()
    assert with_boxes <= 360  # the panel's own width before the boxes existed


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
