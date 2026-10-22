"""The measurement export with several files open: each tab writes its own.

On the real window (``_window.py``: only the 3D views are stood in for), so
what is checked is the window's own plumbing — the tab a file is read into,
which panel gets its name, which panel the dock shows when Export to CSV… is
pressed — rather than its source. The tabs hold shipped outputs, read as the
window reads any file.
"""

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")

from _window import (  # noqa: E402
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
