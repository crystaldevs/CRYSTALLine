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
