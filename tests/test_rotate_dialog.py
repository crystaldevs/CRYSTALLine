"""The Rotate selected dialog, its preview, and the edit it makes."""

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QDoubleSpinBox  # noqa: E402

from crystalline.ui.panels.rotate_dialog import (  # noqa: E402
    ATOM,
    CUSTOM,
    ORIGIN,
    THROUGH_ATOMS,
    RotateDialog,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


_POSITIONS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [3.0, 3.0, 3.0]])
_SYMBOLS = ["C", "C", "F", "H"]
_CELL = np.diag([2.5, 8.6, 4.8])


def _dialog(selected=(0, 1, 2), cell=_CELL, preview=None):
    return RotateDialog(list(selected), _POSITIONS, _SYMBOLS, cell=cell, preview=preview)


def _choose(box, data):
    box.setCurrentIndex(box.findData(data))


def test_the_lattice_vectors_are_offered_only_with_a_lattice(qapp):
    crystal = _dialog()
    with_cell = [crystal.axis.itemData(i) for i in range(crystal.axis.count())]
    molecule = _dialog(cell=None)
    without = [molecule.axis.itemData(i) for i in range(molecule.axis.count())]

    assert with_cell[:6] == ["a", "b", "c", "x", "y", "z"]
    assert "a" not in without and without[:3] == ["x", "y", "z"]


def test_two_atoms_are_needed_for_an_axis_through_them(qapp):
    one = _dialog(selected=(0,))
    assert one.axis.findData(THROUGH_ATOMS) == -1


def test_by_default_it_turns_about_a_through_the_centroid(qapp):
    dialog = _dialog()
    dialog.angle.setValue(90)
    axis, angle, centre = dialog.rotation()

    assert np.allclose(axis, _CELL[0])
    assert angle == 90
    assert np.allclose(centre, _POSITIONS[:3].mean(axis=0))


def test_an_axis_through_two_atoms_passes_through_them(qapp):
    dialog = _dialog()
    _choose(dialog.axis, THROUGH_ATOMS)
    dialog.axis_from.setCurrentIndex(1)
    dialog.axis_to.setCurrentIndex(2)

    axis, _angle, centre = dialog.rotation()
    assert np.allclose(axis, _POSITIONS[2] - _POSITIONS[1])
    assert np.allclose(centre, _POSITIONS[1])
    assert dialog.centre.isHidden() and not dialog._axis_atoms.isHidden()


def test_the_same_atom_twice_is_no_axis(qapp):
    dialog = _dialog()
    _choose(dialog.axis, THROUGH_ATOMS)
    dialog.axis_to.setCurrentIndex(dialog.axis_from.currentIndex())

    assert dialog.rotation() is None
    ok = dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok)
    assert not ok.isEnabled()


def test_the_point_can_be_an_atom_the_origin_or_typed(qapp):
    dialog = _dialog()
    _choose(dialog.centre, ATOM)
    dialog.centre_atom.setCurrentIndex(2)
    assert np.allclose(dialog.centre_point(), _POSITIONS[2])
    _choose(dialog.centre, ORIGIN)
    assert np.allclose(dialog.centre_point(), 0)
    _choose(dialog.centre, CUSTOM)
    for spin, value in zip(dialog.point.findChildren(QDoubleSpinBox), (1, 2, 3)):
        spin.setValue(value)
    assert np.allclose(dialog.centre_point(), [1, 2, 3])


def test_the_slider_and_the_box_agree(qapp):
    dialog = _dialog()
    dialog.slider.setValue(45)
    assert dialog.angle.value() == 45
    dialog.angle.setValue(270)
    assert dialog.slider.value() == -90          # where it points


def test_every_change_is_previewed_and_a_cancel_puts_the_atoms_back(qapp):
    seen = []
    dialog = _dialog(preview=lambda axis, angle, centre: seen.append((axis is None, angle)))
    dialog.angle.setValue(30)
    dialog.angle.setValue(60)
    dialog.reject()

    assert seen[0] == (False, 30) and seen[1] == (False, 60)
    assert seen[-1] == (True, 0.0)


def test_an_accepted_rotation_is_not_undone_by_closing(qapp):
    seen = []
    dialog = _dialog(preview=lambda axis, angle, centre: seen.append(axis is None))
    dialog.angle.setValue(30)
    dialog.accept()

    assert seen and not seen[-1]


# ── the window's command ──────────────────────────────────────────────────


class _Renderer:
    def __init__(self):
        self.previews = []

    def periodic_image_indices(self, index):
        return {0: [3], 3: [0]}.get(index, [])

    def preview_positions(self, positions):
        self.previews.append(np.array(positions))


def _window(structure, selected):
    from types import SimpleNamespace

    halos = []
    return SimpleNamespace(
        _editing=True,
        structure=structure,
        _unit_cell=np.diag([5.0, 5.0, 5.0]),
        structure_panel=SimpleNamespace(selected_indices=lambda: list(selected)),
        viewport=SimpleNamespace(renderer=_Renderer(), set_selection=halos.append),
        halos=halos,
    )


def test_the_command_applies_the_rotation_once_images_and_all(qapp, monkeypatch):
    from ase import Atoms

    from crystalline.core.structure import Structure
    from crystalline.ui import main_window
    from crystalline.ui.panels import rotate_dialog

    structure = Structure.from_ase(Atoms("CCCC", positions=[[0, 1, 0], [1, 1, 0], [9, 9, 9], [5, 1, 0]],
                                         cell=np.diag([5.0, 5.0, 5.0]), pbc=True))
    edits = []
    structure.add_listener(lambda *a: edits.append(a))

    class Accepting:
        def __init__(self, selected, positions, symbols, cell=None, preview=None, parent=None):
            self.preview = preview

        def exec(self):
            self.preview(np.array([0, 0, 1.0]), 90.0, np.zeros(3))   # dialled in
            return QDialog.Accepted

        def rotation(self):
            return np.array([0, 0, 1.0]), 90.0, np.zeros(3)

    monkeypatch.setattr(rotate_dialog, "RotateDialog", Accepting)
    window = _window(structure, [0, 1])

    main_window.MainWindow._rotate_selected(window)

    assert len(edits) == 1                                        # one undo step
    p = structure.positions
    assert np.allclose(p[0], [-1, 0, 0]) and np.allclose(p[1], [-1, 1, 0])
    assert np.allclose(p[3] - p[0], [5, 0, 0])                    # the image followed
    assert np.allclose(p[2], [9, 9, 9])                           # the rest stayed
    assert window.viewport.renderer.previews                      # it was shown first
    assert window.halos == [[0, 1]]                               # selection kept


def test_a_cancelled_rotation_changes_nothing(qapp, monkeypatch):
    from ase import Atoms

    from crystalline.core.structure import Structure
    from crystalline.ui import main_window
    from crystalline.ui.panels import rotate_dialog

    structure = Structure.from_ase(Atoms("CC", positions=[[0, 1, 0], [1, 1, 0]]))
    before = structure.positions.copy()

    class Cancelling:
        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return QDialog.Rejected

    monkeypatch.setattr(rotate_dialog, "RotateDialog", Cancelling)
    main_window.MainWindow._rotate_selected(_window(structure, [0, 1]))

    assert np.allclose(structure.positions, before)

