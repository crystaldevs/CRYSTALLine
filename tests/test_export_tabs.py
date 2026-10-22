"""The measurement export with several files open: each tab writes its own.

On the real window (``_window.py``: only the 3D views are stood in for), so
what is checked is the window's own plumbing — the tab a file is read into,
which panel gets its name, which panel the dock shows when Export to CSV… is
pressed — rather than its source. The tabs hold shipped outputs, read as the
window reads any file.

A LOCMODES run's local modes go into its export too: the run is
``data/locmodes/h2o2_synthetic.out``, made-up force constants for hydrogen
peroxide in a 5 Å box. It holds no geometry block a structure could be read
from, so what the reader makes of it is put together here: the structure at the
positions its Cartesian block gives, and the local modes read from it by the
real reader.
"""

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")

from _window import (  # noqa: E402
    atoms_on_screen,
    close_window,
    export,
    make_window,
    measure,
    measure_a_bond,
    open_file,
    show_tab,
    visible_geometry_panel,
)
from sample_data import path as sample_path  # noqa: E402

from crystalline.core import measure as M  # noqa: E402

_THIOUREA = "thiourea/thiourea_pbed3_Ahlrichs-pVTZ_freqcalc_80K.out"
_COESITE = "coesite/coesite_ela.out"


@pytest.fixture
def window(monkeypatch, tmp_path):
    made = make_window(monkeypatch, tmp_path)
    yield made
    close_window(made)


def _beside(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(os.path.dirname(path), f"{stem}_measurements.csv")


def _row(bond):
    return [M.DISTANCE, bond.label, f"{bond.value:.6f}", "Å"]


# ── each tab its own ──────────────────────────────────────────────────────
def test_each_tab_exports_what_was_measured_in_it(window, monkeypatch, tmp_path):
    """Measured in turn in two tabs, then exported from each: every file holds
    its own tab's measurements, and is offered beside its own file."""
    open_file(window, sample_path(_COESITE))
    coesite = window._tab
    in_coesite = measure_a_bond(window)
    open_file(window, sample_path(_THIOUREA))
    thiourea = window._tab
    assert thiourea is not coesite
    in_thiourea = measure_a_bond(window)

    offered, (header, *rows) = export(window, monkeypatch, tmp_path / "thiourea.csv")
    assert offered == _beside(sample_path(_THIOUREA))
    assert header == list(M.EXPORT_COLUMNS)
    assert rows == [_row(in_thiourea)]

    show_tab(window, coesite)
    assert visible_geometry_panel(window) is coesite.geometry_panel
    offered, (header, *rows) = export(window, monkeypatch, tmp_path / "coesite.csv")
    assert offered == _beside(sample_path(_COESITE))
    assert rows == [_row(in_coesite)]


def test_a_file_read_behind_the_tab_on_screen_exports_its_own_once_shown(
        window, monkeypatch, tmp_path):
    """Opened with another in front, the file waits unbuilt; its panel is made
    and filled the first time its tab is clicked — and it is that panel, not the
    one on screen when it was read, that gets the file's name."""
    open_file(window, sample_path(_THIOUREA))
    thiourea = window._tab
    open_file(window, sample_path(_COESITE), front=False)
    coesite = window._tabs[-1]
    assert window._tab is thiourea and not coesite.built()

    show_tab(window, coesite)
    assert coesite.built() and window._tab is coesite
    bond = measure_a_bond(window)
    offered, (_, *rows) = export(window, monkeypatch, tmp_path / "coesite.csv")
    assert offered == _beside(sample_path(_COESITE)) and rows == [_row(bond)]

    show_tab(window, thiourea)
    measure_a_bond(window)
    offered, (_, *rows) = export(window, monkeypatch, tmp_path / "thiourea.csv")
    assert offered == _beside(sample_path(_THIOUREA)) and len(rows) == 1


def test_closing_a_tab_leaves_the_next_ones_export_as_it_was(window, monkeypatch, tmp_path):
    """Closing the tab in front shows the one behind it, whose panel and table
    are its own, untouched by the tab that went."""
    open_file(window, sample_path(_THIOUREA))
    thiourea = window._tab
    bond = measure_a_bond(window)
    open_file(window, sample_path(_COESITE))
    measure_a_bond(window)
    measure_a_bond(window)

    window._close_current_tab()
    assert window._tab is thiourea and visible_geometry_panel(window) is thiourea.geometry_panel
    offered, (_, *rows) = export(window, monkeypatch, tmp_path / "thiourea.csv")
    assert offered == _beside(sample_path(_THIOUREA)) and rows == [_row(bond)]


def test_an_angle_clicked_in_the_view_is_exported_at_the_atom_clicked_second(
        window, monkeypatch, tmp_path):
    """A neighbour, then the atom, then another neighbour: the angle at the
    atom. The selection used to reach the panel in index order, and the angle
    exported was the one at whichever atom sat in the middle of the list."""
    open_file(window, sample_path(_THIOUREA))
    positions = np.asarray(window.structure.positions, dtype=float)
    nearest = np.argsort(np.linalg.norm(positions - positions[0], axis=1))
    atoms = [int(nearest[1]), 0, int(nearest[2])]
    assert atoms != sorted(atoms)                    # else index order would do as well
    measure(window, atoms)
    _, (_, row) = export(window, monkeypatch, tmp_path / "angle.csv")

    first, second = positions[atoms[0]] - positions[0], positions[atoms[2]] - positions[0]
    cosine = first @ second / np.linalg.norm(first) / np.linalg.norm(second)
    at_vertex = np.degrees(np.arccos(cosine))
    symbols = [window.structure.symbols[i] for i in atoms]
    assert row[:2] == [M.ANGLE, "–".join(f"{s}({i})" for s, i in zip(symbols, atoms))]
    assert float(row[2]) == pytest.approx(at_vertex, abs=1e-4)


def test_the_export_button_follows_the_tab_on_screen(window):
    """Greyed while the tab on screen has nothing measured, whatever the others
    hold: it is that tab's panel the dock shows."""
    open_file(window, sample_path(_THIOUREA))
    thiourea = window._tab
    measure_a_bond(window)
    open_file(window, sample_path(_COESITE))
    assert not visible_geometry_panel(window)._export_btn.isEnabled()
    show_tab(window, thiourea)
    assert visible_geometry_panel(window)._export_btn.isEnabled()


# ── a LOCMODES run's local modes, exported with it ─────────────────────────
_PEROXIDE = "locmodes/h2o2_synthetic.out"


def _peroxide_read():
    """What reading the synthetic run would give, had it a geometry block."""
    from ase import Atoms

    from crystalline.core.structure import Structure
    from crystalline.crystalio import LoadedFile, read_local_modes
    from crystalline.ui.main_window import _ReadFile

    modes = read_local_modes(sample_path(_PEROXIDE))
    atoms = Atoms("OOHH", positions=modes.positions, cell=modes.lattice, pbc=True)
    return _ReadFile(loaded=LoadedFile(structure=Structure.from_ase(atoms)), output_path=None,
                     adps=None, output_props={}, capabilities={}, local_modes=modes)


def _open_peroxide(window, front=True):
    open_file(window, sample_path(_PEROXIDE), _peroxide_read(), front=front)
    return window._tabs[-1]


def _coordinate(window, number):
    modes = window.geometry_panel._local_modes
    return next(c for c in modes.coordinates if c.number == number)


def _measure_coordinate(window, number):
    """Measure LOCMODES coordinate ``number`` on its atoms in the tab on screen."""
    c = _coordinate(window, number)
    atoms = atoms_on_screen(window, window.geometry_panel._local_modes.points(c), c.atoms)
    assert atoms is not None, f"coordinate {number} is not on screen"
    return measure(window, atoms)


_LOCAL_HEADER = list(M.EXPORT_COLUMNS) + list(M.EXPORT_LOCAL_COLUMNS)


def test_only_the_run_with_local_modes_exports_their_columns(window, monkeypatch, tmp_path):
    """Measured in turn in two tabs, then exported from each: every file holds
    its own tab's measurements, with the LOCMODES columns only for the run that
    has local modes, and is offered beside its own file."""
    peroxide = _open_peroxide(window)
    _measure_coordinate(window, 1)                       # O1–O2
    _measure_coordinate(window, 4)                       # H3···O2, into the cell along −a
    open_file(window, sample_path(_THIOUREA))
    thiourea = window._tab
    assert thiourea is not peroxide
    bond = measure_a_bond(window)

    offered, (header, *rows) = export(window, monkeypatch, tmp_path / "thiourea.csv")
    assert offered == _beside(sample_path(_THIOUREA))
    assert header == list(M.EXPORT_COLUMNS)
    assert rows == [[M.DISTANCE, bond.label, f"{bond.value:.6f}", "Å"]]

    show_tab(window, peroxide)
    assert visible_geometry_panel(window) is peroxide.geometry_panel
    offered, (header, *rows) = export(window, monkeypatch, tmp_path / "peroxide.csv")
    assert offered == _beside(sample_path(_PEROXIDE))
    assert header == _LOCAL_HEADER
    assert [row[4:] for row in rows] == [
        ["1", "O1–O2", "1.45", "4.2", "", "8.0", "950.0", "as listed"],
        ["4", "H3–O2 (-1 0 0)", "3.3782", "0.15", "", "0.98", "500.0", "as listed"],
    ]


def test_a_file_read_behind_the_tab_on_screen_gets_its_local_modes_when_first_shown(
        window, monkeypatch, tmp_path):
    """Opened with another in front, the run waits unbuilt; its panel is made
    and filled the first time its tab is clicked — and it is that panel, not the
    one on screen when it was read, that gets the local modes and the name."""
    open_file(window, sample_path(_THIOUREA))
    thiourea = window._tab
    peroxide = _open_peroxide(window, front=False)
    assert window._tab is thiourea and not peroxide.built()
    assert thiourea.geometry_panel._local_modes is None

    show_tab(window, peroxide)
    assert peroxide.built() and window._tab is peroxide
    found = _measure_coordinate(window, 5)               # the H3–O1–O2 angle
    assert [c.number for c in found.local] == [5]
    offered, (header, *rows) = export(window, monkeypatch, tmp_path / "peroxide.csv")
    assert offered == _beside(sample_path(_PEROXIDE)) and header == _LOCAL_HEADER
    assert [row[4] for row in rows] == ["5"]

    show_tab(window, thiourea)
    measure_a_bond(window)
    offered, (header, *rows) = export(window, monkeypatch, tmp_path / "thiourea.csv")
    assert offered == _beside(sample_path(_THIOUREA)) and header == list(M.EXPORT_COLUMNS)
    assert len(rows) == 1


def test_a_supercell_in_one_tab_leaves_the_other_tabs_atoms_as_they_were(
        window, monkeypatch, tmp_path):
    """The atoms a measurement is matched on come from the tab's own view: a
    supercell taken in one tab re-indexes that tab's atoms, and the same run
    open in another tab still finds its local modes on its own."""
    first = _open_peroxide(window)
    second = _open_peroxide(window)
    assert window._tab is second
    window._set_supercell((2, 1, 1))
    window._apply_cell_view()
    across = _measure_coordinate(window, 4)              # now between images of the supercell
    assert [c.number for c in across.local] == [4]
    natoms_second = len(window.structure)

    show_tab(window, first)
    assert len(window.structure) < natoms_second         # still the unit cell
    bond = _measure_coordinate(window, 1)
    assert [c.number for c in bond.local] == [1]
    offered, (_, *rows) = export(window, monkeypatch, tmp_path / "first.csv")
    assert [row[4] for row in rows] == ["1"]

    show_tab(window, second)
    offered, (_, *rows) = export(window, monkeypatch, tmp_path / "second.csv")
    assert [row[4] for row in rows] == ["4"]


def test_closing_a_tab_leaves_the_next_ones_local_modes_as_they_were(
        window, monkeypatch, tmp_path):
    """Closing the tab in front shows the one behind it, whose panel, table and
    local modes are its own, untouched by the tab that went."""
    peroxide = _open_peroxide(window)
    _measure_coordinate(window, 2)                       # O1–H3
    open_file(window, sample_path(_COESITE))
    measure_a_bond(window)
    measure_a_bond(window)

    window._close_current_tab()
    assert window._tab is peroxide and visible_geometry_panel(window) is peroxide.geometry_panel
    offered, (header, *rows) = export(window, monkeypatch, tmp_path / "peroxide.csv")
    assert offered == _beside(sample_path(_PEROXIDE)) and header == _LOCAL_HEADER
    assert [row[4:6] for row in rows] == [["2", "O1–H3"]]


def test_an_angle_clicked_in_the_view_finds_its_local_mode_at_the_atom_clicked_second(
        window, monkeypatch, tmp_path):
    """H3, then O1, then O2: the angle at O1, the run's coordinate 5. The
    selection used to reach the panel in index order, and the angle exported
    was the one at whichever atom sat in the middle of the list."""
    _open_peroxide(window)
    c = _coordinate(window, 5)
    atoms = atoms_on_screen(window, window.geometry_panel._local_modes.points(c), c.atoms)
    assert atoms != sorted(atoms)                    # else index order would do as well
    measure(window, atoms)
    _, (_, row) = export(window, monkeypatch, tmp_path / "angle.csv")
    symbols = [window.structure.symbols[i] for i in atoms]
    assert row[:2] == [M.ANGLE, "–".join(f"{s}({i})" for s, i in zip(symbols, atoms))]
    assert float(row[2]) == pytest.approx(108.0247, abs=1e-3)
    assert row[4:6] == ["5", "H3–O1–O2"]
