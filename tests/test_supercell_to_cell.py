"""Making the supercell on screen the structure's own cell, for defects."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")

from ase.build import bulk  # noqa: E402

from crystalline.core.cells import CellView, tile_supercell, without_boundary_copies  # noqa: E402
from crystalline.core.structure import Structure  # noqa: E402
from crystalline.ui.main_window import MainWindow  # noqa: E402


def _window(reps=(2, 2, 2)):
    unit = Structure.from_ase(bulk("MgO", "rocksalt", a=4.21, cubic=True))
    shown, _modes = tile_supercell(unit, reps)
    calls, messages = [], []
    window = SimpleNamespace(
        structure=shown,
        _source=Structure.from_ase(bulk("MgO", "rocksalt", a=4.21)),
        _unit_cell=np.asarray(unit.to_ase().cell, dtype=float),
        _supercell=tuple(reps),
        _cell_view=CellView.CRYSTALLOGRAPHIC,
        _tile_restore=("something",),
        _apply_cell_view=lambda: calls.append("derived"),
        _update_cell_view_controls=lambda: calls.append("controls"),
        statusBar=lambda: SimpleNamespace(showMessage=lambda text, _ms=0: messages.append(text)),
        calls=calls, messages=messages,
    )
    window._set_supercell = lambda r: setattr(window, "_supercell", tuple(r))
    return window


def test_a_vacancy_in_the_supercell_becomes_one_in_the_crystal():
    window = _window()
    window.structure.remove_atoms([0])                     # one Mg of 32
    window.structure.set_reduced_symmetry((((1, 0, 0), (0, 1, 0), (0, 0, 1)),))

    MainWindow._supercell_to_cell(window)

    source = window._source.to_ase()
    assert len(source) == 63 and source.get_chemical_formula() == "Mg31O32"
    assert np.allclose(source.cell.cellpar(), [8.42, 8.42, 8.42, 90, 90, 90])
    assert window._supercell == (1, 1, 1) and window._cell_view is CellView.PRIMITIVE
    assert window._tile_restore is None
    assert window._source.reduced_symmetry == ()           # named in the old cell
    assert window.calls == ["derived", "controls"]


def test_the_supercell_s_copies_at_the_boundary_are_not_counted_twice():
    from crystalline.core.cells import complete_boundary

    window = _window()
    window.structure, _modes = complete_boundary(window.structure)
    assert len(window.structure) > 64

    MainWindow._supercell_to_cell(window)

    assert len(window._source) == 64


def test_with_no_supercell_there_is_nothing_to_do_and_it_says_so():
    window = _window(reps=(1, 1, 1))
    before = window._source.to_ase()

    MainWindow._supercell_to_cell(window)

    assert window._source.to_ase() == before and window.calls == []
    assert "Supercell" in window.messages[0]
