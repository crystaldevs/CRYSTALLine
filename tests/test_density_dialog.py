"""The dialog that chooses which field to draw, and how."""

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from crystalline.crystalio import density as D  # noqa: E402
from test_density import POTENTIAL_TITLE, _field, _write_cube, _write_fort31  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def folder(tmp_path):
    """A calculation's folder as an ECH3 plus a POT3 run leaves it."""
    n = 9
    steps = np.array([[0.3, 0.0, 0.0], [-0.15, 0.26, 0.0], [0.0, 0.0, 0.35]])
    base = D.ScalarField(values=np.zeros((n, n, n)), origin=np.zeros(3), steps=steps)
    points = base.points()
    centre = points[n // 2, n // 2, n // 2]
    rho = 4.0 * np.exp(-((points - centre) ** 2).sum(axis=-1) / 0.4)
    _write_cube(tmp_path / "DENS_CUBE.DAT", D.replace(base, values=rho))
    _write_cube(tmp_path / "POT_CUBE.DAT",
                D.replace(base, values=np.broadcast_to(
                    np.linspace(-1.0, 1.0, n)[:, None, None], (n, n, n)).copy()),
                title=POTENTIAL_TITLE)
    _write_fort31(tmp_path / "fort.31", D.replace(base, values=rho))
    return tmp_path


def _dialog(folder):
    from crystalline.ui.panels.density_dialog import DensityDialog

    return DensityDialog(folder=str(folder))


def test_the_files_of_the_folder_are_found_without_being_asked_for(qapp, folder):
    dialog = _dialog(folder)

    names = [dialog.file.itemText(i) for i in range(dialog.file.count())]

    assert "DENS_CUBE.DAT" in names and "POT_CUBE.DAT" in names


def test_the_charge_density_is_chosen_first_and_the_potential_second(qapp, folder):
    """So that "colour the density by the potential" is one click away."""
    dialog = _dialog(folder)

    assert dialog.file.currentText() == "DENS_CUBE.DAT"
    assert dialog.second.currentText() == "POT_CUBE.DAT"


def test_the_isovalue_opens_on_a_surface_that_clears_the_atoms(qapp, folder):
    """Not at a fraction of the peak, which tracks the cores, nor at the level
    enclosing most of the charge, which tracks the valence tail and fills the
    cell — beryllium reaches nine tenths of its charge at 0.035 e/bohr³. A
    fraction of the cell's volume is what behaves, and it has to be a large
    enough fraction for the surface to come out of the drawn atom."""
    from crystalline.ui.panels.density_dialog import DENSE_FRACTION

    dialog = _dialog(folder)
    field = dialog._field

    value = dialog.isovalue.value()

    wrapped = float((field.values >= value).mean())
    assert wrapped == pytest.approx(DENSE_FRACTION, abs=0.02)
    assert DENSE_FRACTION >= 0.1        # smaller hides inside the atoms
    assert value < field.peak          # so there is a surface to draw


def test_a_signed_field_opens_on_a_fraction_of_its_peak_instead(qapp, folder):
    """A difference is near zero over most of the cell, so a volume fraction of
    it would be noise."""
    from crystalline.ui.panels.density_dialog import _SUBTRACT

    dialog = _dialog(folder)
    dialog.second.setCurrentIndex(
        [dialog.second.itemText(i) for i in range(dialog.second.count())].index("POT_CUBE.DAT"))
    dialog.companion.setCurrentIndex(
        [dialog.companion.itemData(i) for i in range(dialog.companion.count())].index(_SUBTRACT))

    assert dialog.request()[0].signed


def test_an_empty_folder_leaves_ok_disabled_and_says_why(qapp, tmp_path):
    from PySide6.QtWidgets import QDialogButtonBox
    from crystalline.ui.panels.density_dialog import DensityDialog

    DensityDialog.last_folder = ""      # do not fall back to an earlier test's
    dialog = DensityDialog(folder=str(tmp_path))

    assert not dialog._buttons.button(QDialogButtonBox.Ok).isEnabled()
    assert "ECH3" in dialog.info.text()


def test_choosing_to_subtract_hands_back_the_difference_not_the_two(qapp, folder):
    """The difference is the field being drawn: its sign and its peak are what
    the isovalue is read against."""
    from crystalline.ui.panels.density_dialog import _SUBTRACT

    dialog = _dialog(folder)
    dialog.second.setCurrentIndex(
        [dialog.second.itemText(i) for i in range(dialog.second.count())].index("fort.31"))
    dialog.companion.setCurrentIndex(
        [dialog.companion.itemData(i) for i in range(dialog.companion.count())].index(_SUBTRACT))

    field, options = dialog.request()

    assert field.signed                       # a difference takes both signs
    # The same density, read back from the two formats it was written in.
    assert np.allclose(field.values, 0.0, atol=1e-5)
    assert options.colour_by is None


def test_choosing_to_paint_keeps_the_density_and_carries_the_potential(qapp, folder):
    from crystalline.ui.panels.density_dialog import _PAINT

    dialog = _dialog(folder)
    dialog.companion.setCurrentIndex(
        [dialog.companion.itemData(i) for i in range(dialog.companion.count())].index(_PAINT))

    field, options = dialog.request()

    assert field.kind == D.CHARGE
    assert options.colour_by is not None and options.colour_by.kind == D.POTENTIAL


def test_a_second_field_on_another_grid_is_reported_rather_than_combined(qapp, folder,
                                                                        tmp_path):
    from crystalline.ui.panels.density_dialog import _PAINT

    _write_cube(folder / "other.cube", _field(shape=(4, 4, 4)))
    dialog = _dialog(folder)
    dialog.file.setCurrentIndex(
        [dialog.file.itemText(i) for i in range(dialog.file.count())].index("DENS_CUBE.DAT"))
    dialog.second.setCurrentIndex(
        [dialog.second.itemText(i) for i in range(dialog.second.count())].index("other.cube"))
    dialog.companion.setCurrentIndex(
        [dialog.companion.itemData(i) for i in range(dialog.companion.count())].index(_PAINT))

    assert "different grid" in dialog.info.text()


def test_the_slice_controls_replace_the_isovalue_and_not_the_other_way(qapp, folder):
    """A row that does not apply goes away rather than sitting there greyed."""
    dialog = _dialog(folder)

    dialog.view.setCurrentIndex(
        [dialog.view.itemData(i) for i in range(dialog.view.count())].index(D.SLICE))

    assert dialog._isovalue_row[1].isHidden()
    assert not dialog._miller_row[1].isHidden()
    assert not dialog.cutaway.isHidden()
    assert dialog._positive_row[1].isHidden()   # the map colours a slice


def test_a_plane_is_named_by_miller_indices_and_reports_its_spacing(qapp, folder):
    from crystalline.ui.panels.density_dialog import DensityDialog

    dialog = DensityDialog(folder=str(folder), miller_cell=4.21 * np.eye(3))
    dialog.view.setCurrentIndex(
        [dialog.view.itemData(i) for i in range(dialog.view.count())].index(D.SLICE))
    for box, value in zip(dialog.miller, (1, 1, 1)):
        box.setValue(value)
    dialog.offset.setValue(0.5)

    _field_out, options = dialog.request()

    assert options.miller == (1, 1, 1) and options.offset == pytest.approx(0.5)
    assert "d = 2.431 Å" in dialog.info.text()


def test_000_is_not_a_plane_and_cannot_be_drawn(qapp, folder):
    from PySide6.QtWidgets import QDialogButtonBox

    dialog = _dialog(folder)
    dialog.view.setCurrentIndex(
        [dialog.view.itemData(i) for i in range(dialog.view.count())].index(D.SLICE))
    for box in dialog.miller:
        box.setValue(0)

    assert not dialog._buttons.button(QDialogButtonBox.Ok).isEnabled()
    assert "not a plane" in dialog.info.text()


def test_the_options_carry_every_choice_to_the_renderer(qapp, folder):
    dialog = _dialog(folder)
    dialog.isovalue.setValue(0.25)
    dialog.opacity.setValue(0.5)
    dialog.clip.setChecked(False)

    _field_out, options = dialog.request()

    assert options.isovalue == pytest.approx(0.25)
    assert options.opacity == pytest.approx(0.5)
    assert options.clip_to_cell is False


class _StubWindow(QWidget):
    """A MainWindow stand-in carrying the real handlers.

    ``MainWindow`` cannot be built headless — its VTK interactor needs a GL
    surface, and building one part-way through a test session segfaults it —
    but the handler can be run against a stub that provides exactly the
    collaborators it is allowed to use. Anything it reaches for that is not
    here is an AttributeError, which is what shipped once: the status bar is
    ``statusBar()``, not ``status``.
    """

    from crystalline.ui.main_window import MainWindow as _real

    _open_density = _real._open_density
    _clear_density = _real._clear_density
    _update_density_actions = _real._update_density_actions
    _conventional_cell = _real._conventional_cell
    del _real

    def __init__(self, folder):
        super().__init__()
        from PySide6.QtGui import QAction

        self._output_path = str(folder / "run.out")
        self._density_shown = False
        self._clear_density_action = QAction("Clear field", self)
        self._messages = []
        self.drawn = []
        self.viewport = type("V", (), {})()
        self.viewport.renderer = self

    # the renderer's half of the contract
    def set_density(self, field, options=None, miller_cell=None):
        self.drawn.append((field, options))

    def face_density_plane(self):
        self.faced = True

    # the window's
    def statusBar(self):  # noqa: N802 - Qt's name
        return self

    def showMessage(self, text, timeout=0):  # noqa: N802 - Qt's name
        self._messages.append(text)

    def _restore_dialog(self, dialog, key):
        self.restored = key

    def _remember_dialog(self, dialog, key):
        self.remembered = key


def test_the_menu_entry_draws_the_field_and_says_what_it_drew(qapp, folder, monkeypatch):
    """End to end through the handler, which is where a wrong attribute or a
    bad argument shows up — reading the source of it cannot catch either."""
    from PySide6.QtWidgets import QDialog

    from crystalline.ui.panels import density_dialog

    monkeypatch.setattr(density_dialog.DensityDialog, "exec",
                        lambda self: QDialog.Accepted)
    window = _StubWindow(folder)

    window._open_density()

    assert window._density_shown
    assert window._clear_density_action.isEnabled()
    assert window.drawn and window.drawn[-1][0].kind == D.CHARGE
    assert "density" in window._messages[-1].lower()

    window._clear_density()

    assert not window._density_shown
    assert window.drawn[-1][0] is None


def test_a_slice_reports_the_plane_it_was_cut_across(qapp, folder, monkeypatch):
    from PySide6.QtWidgets import QDialog

    from crystalline.ui.panels import density_dialog

    def _accept(self):
        self.view.setCurrentIndex(
            [self.view.itemData(i) for i in range(self.view.count())].index(D.SLICE))
        for box, value in zip(self.miller, (1, 1, 0)):
            box.setValue(value)
        return QDialog.Accepted

    monkeypatch.setattr(density_dialog.DensityDialog, "exec", _accept)
    window = _StubWindow(folder)

    window._open_density()

    assert window._messages[-1].endswith("(1 1 0) plane")
    assert window.faced          # a plane is turned face-on, or it reads as a line


def test_opening_another_file_takes_the_previous_field_off_the_view():
    """A rebuild redraws whatever field the renderer holds, so a new file drew
    the old one's density — and a slice's cutaway — over its own atoms. Loading
    has to clear it before the new structure goes in, as it does the orbital.
    ``MainWindow`` cannot be built headless, so the order is read from the
    source, as the menu tests do for the rest of ``_load_path``."""
    import inspect

    from crystalline.ui.main_window import MainWindow

    source = inspect.getsource(MainWindow._load_path)

    assert "self._clear_density()" in source
    assert source.index("self._clear_density()") < source.index("self._source = result.structure")


def test_a_cleared_field_stays_cleared_through_a_rebuild(qapp):
    """What loading relies on: once cleared, a rebuild does not bring it back."""
    import pyvista as pv
    from ase import Atoms

    from crystalline.core.structure import Structure
    from crystalline.viz.renderer import StructureRenderer

    cell = 4.0 * np.eye(3)
    n = 9
    field = D.ScalarField(values=np.random.default_rng(0).random((n, n, n)),
                          origin=np.zeros(3), steps=cell / (n - 1))
    plotter = pv.Plotter(off_screen=True)
    try:
        renderer = StructureRenderer(plotter)
        renderer.set_structure(Structure.from_ase(Atoms("Na", positions=[(0, 0, 0)],
                                                        cell=cell, pbc=True)))
        renderer.set_density(field, D.DensityOptions(view=D.SLICE, miller=(0, 0, 1)),
                             miller_cell=cell)
        assert renderer._density_actors

        renderer.set_density(None)
        renderer.set_structure(Structure.from_ase(Atoms("Cl", positions=[(1, 1, 1)],
                                                        cell=cell, pbc=True)))

        assert renderer._density_actors == []
        assert not any(actor.GetMapper().GetNumberOfClippingPlanes()
                       for actor in plotter.renderer.actors.values()
                       if hasattr(actor, "GetMapper") and actor.GetMapper() is not None
                       and hasattr(actor.GetMapper(), "GetNumberOfClippingPlanes"))
    finally:
        plotter.close()


def test_the_colour_bar_is_offered_only_where_there_is_a_map(qapp, folder):
    from crystalline.ui.panels.density_dialog import _PAINT

    dialog = _dialog(folder)
    assert dialog.colour_bar.isHidden()                   # a plain surface

    dialog.view.setCurrentIndex(
        [dialog.view.itemData(i) for i in range(dialog.view.count())].index(D.SLICE))
    assert not dialog.colour_bar.isHidden()               # a plane

    dialog.view.setCurrentIndex(
        [dialog.view.itemData(i) for i in range(dialog.view.count())].index(D.ISOSURFACE))
    dialog.companion.setCurrentIndex(
        [dialog.companion.itemData(i) for i in range(dialog.companion.count())].index(_PAINT))
    assert not dialog.colour_bar.isHidden()               # a painted surface

    dialog.colour_bar.setChecked(True)
    assert dialog.request()[1].colour_bar


def test_every_setting_is_remembered_the_next_time_the_dialog_opens(qapp, folder):
    """The remembered settings are read off the dialog's attributes, and boxes
    kept only in a list — the Miller indices — were skipped."""
    from crystalline.ui.panels import dialog_state

    first = _dialog(folder)
    first.view.setCurrentIndex(
        [first.view.itemData(i) for i in range(first.view.count())].index(D.SLICE))
    for box, value in zip(first.miller, (1, 1, 1)):
        box.setValue(value)
    first.offset.setValue(0.5)
    first.colour_bar.setChecked(True)
    first.cutaway.setChecked(False)

    second = _dialog(folder)
    dialog_state.restore(second, dialog_state.capture(first))
    options = second.request()[1]

    assert options.miller == (1, 1, 1)
    assert options.offset == pytest.approx(0.5)
    assert options.colour_bar and not options.cutaway
