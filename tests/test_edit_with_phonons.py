"""Regression: editing geometry while phonon modes are loaded must not snap
atoms back to the (now stale) equilibrium — the bug where a dragged atom
reverted to its original spot on drop.

Needs PySide6 (PhononPanel) and pyvista (renderer); skips otherwise.
"""

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyvista")

import pyvista as pv  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from crystalline.core.phonons import PhononMode, PhononModes  # noqa: E402
from crystalline.core.structure import Structure  # noqa: E402
from crystalline.viz.phonon_animator import PhononAnimator  # noqa: E402
from crystalline.viz.renderer import StructureRenderer  # noqa: E402
from crystalline.ui.main_window import MainWindow  # noqa: E402
from crystalline.ui.panels.phonon_panel import PhononPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_edit_does_not_revert_atom_when_modes_loaded(qapp):
    s = Structure.empty()
    s.set_cell(np.eye(3) * 8, periodic=False)
    s.add_atom("C", [4, 4, 4])
    s.add_atom("O", [5, 4, 4])

    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(s)
    panel = PhononPanel(PhononAnimator(renderer))
    panel.set_modes(s.positions.copy(), PhononModes(
        [PhononMode(100.0, np.array([[1, 0, 0], [-1, 0, 0]], float))]
    ))

    # Wire up MainWindow's reaction to a structure edit.
    s.add_listener(lambda st: (renderer.refresh(), panel.invalidate_on_edit(st.positions)))

    s.move_atom(0, [6.5, 4.0, 4.0])  # a committed drag

    # The rendered atom must sit at the new position, not snap back.
    actor_world = renderer.rendered_atom_position(0)
    assert np.allclose(actor_world, s.positions[0])
    assert not np.allclose(actor_world, [4, 4, 4])


# ── the modes after an edit that changes the atom count ──────────────────

class _StubWindow:
    """Only what the mode-reconciliation path reads — no VTK window, no docks.

    ``MainWindow`` cannot be constructed headless, and these three methods talk
    to each other through the window's state, so they are exercised unbound
    against the state they actually touch (the same approach as
    ``test_compose_view``).
    """

    from crystalline.core.cells import CellView as _CellView

    _reconcile_modes_with_edit = MainWindow._reconcile_modes_with_edit
    _reload_modes = MainWindow._reload_modes
    _show_modes = MainWindow._show_modes
    _compose_view = MainWindow._compose_view

    def __init__(self, source, modes, panel):
        self._source = source
        self.structure = source
        self._modes = modes
        self._adps = None
        self._adp_index = None
        self._show_boundary = False
        self._cell_view = self._CellView.PRIMITIVE
        self._supercell = (1, 1, 1)
        self._modes_natom = None
        self.phonon_panel = panel

    def _update_export_actions(self):
        pass

    def edited(self):
        """What ``_on_structure_changed`` does with the modes, in order."""
        self.phonon_panel.invalidate_on_edit(self.structure.positions)
        self._reconcile_modes_with_edit(self.structure)


def _loaded_with_modes():
    """A two-atom structure with one mode, as a freshly opened output would be."""
    s = Structure.empty()
    s.set_cell(np.eye(3) * 8, periodic=False)
    s.add_atom("C", [4, 4, 4])
    s.add_atom("O", [5, 4, 4])
    modes = PhononModes([PhononMode(100.0, np.array([[1, 0, 0], [-1, 0, 0]], float))])

    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(s)
    panel = PhononPanel(PhononAnimator(renderer))
    window = _StubWindow(s, modes, panel)
    window._show_modes(modes)
    panel.select_mode(0)
    return s, panel, window


def test_adding_an_atom_drops_the_modes(qapp):
    """A mode has one displacement per atom, so it cannot describe a structure
    with another atom in it."""
    s, panel, window = _loaded_with_modes()

    s.add_atom("H", [2, 2, 2])
    window.edited()

    assert not panel.has_modes()
    assert not panel.isEnabled()


def test_putting_the_geometry_back_brings_the_modes_back(qapp):
    """Undo restores the geometry the modes were read for, so the panel has to
    offer them again. It could not: it had thrown them away, and stayed empty
    and grey for the rest of the session however the structure was put back.
    (An undo reaches the window as this same notification.)"""
    s, panel, window = _loaded_with_modes()

    s.add_atom("H", [2, 2, 2])
    window.edited()
    assert not panel.has_modes()  # the state to recover from

    s.remove_atoms([2])
    window.edited()

    assert panel.has_modes()
    assert panel.isEnabled()
    assert panel.mode_list.count() == 1
    # Anchored to the geometry on screen, not to the one the modes arrived with.
    assert np.allclose(panel._equilibrium, s.positions)


def test_moving_an_atom_keeps_the_modes(qapp):
    """The count is what matters. A displaced geometry is still one the modes
    describe, and re-reading them for every drag would be wasted work."""
    s, panel, window = _loaded_with_modes()

    s.move_atom(0, [4.4, 4.0, 4.0])
    window.edited()

    assert panel.has_modes() and panel.isEnabled()
    assert panel.current_mode_index() == 0  # and the selected mode survives
