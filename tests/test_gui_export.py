"""CRYSTAL .gui export: packed cells, molecules and the load round-trip."""

import numpy as np
import pytest

pytest.importorskip("pymatgen")
pytest.importorskip("ase")
pytest.importorskip("CRYSTALClear")

from ase import Atoms  # noqa: E402
from ase.build import bulk, fcc111  # noqa: E402

from crystalline.core.cells import (  # noqa: E402
    CellView,
    as_view,
    complete_boundary,
    tile_supercell,
)
from crystalline.core.structure import Structure  # noqa: E402
from crystalline.crystalio.loader import load_structure, save_structure_gui  # noqa: E402


def _dimensionality(path) -> int:
    """First field of a .gui header: 3=bulk, 2=slab, 1=polymer, 0=molecule."""
    return int(path.read_text().split("\n", 1)[0].split()[0])


def _packed_water_crystal() -> Structure:
    """A molecular crystal as displayed: boundary-completed, with image atoms."""
    at = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        cell=np.eye(3) * 6.0,
        pbc=True,
    )
    base = as_view(Structure.from_ase(at), CellView.CRYSTALLOGRAPHIC)
    analysis, _ = tile_supercell(base, (1, 1, 1), None)
    packed, _ = complete_boundary(analysis, None)
    return packed


def test_periodic_gui_round_trips(tmp_path):
    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    out = tmp_path / "nacl.gui"
    save_structure_gui(nacl, str(out))
    assert _dimensionality(out) == 3
    back = load_structure(str(out))
    assert back.to_ase().get_chemical_formula() == nacl.to_ase().get_chemical_formula()


def test_packed_boundary_cell_saves(tmp_path):
    """The displayed cell is boundary-completed (duplicate periodic images); the
    saver must merge those coincident sites instead of failing symmetry finding."""
    packed = _packed_water_crystal()
    assert len(packed) > 3  # genuinely packed with image duplicates

    out = tmp_path / "packed.gui"
    save_structure_gui(packed, str(out))  # previously raised SymmetryUndeterminedError
    assert load_structure(str(out)).to_ase().get_chemical_formula() == "H2O"


def test_non_periodic_molecule_writes_0d_gui(tmp_path):
    water = Structure.empty()
    water.add_atom("O", [0.0, 0.0, 0.0])
    water.add_atom("H", [0.96, 0.0, 0.0])
    water.add_atom("H", [-0.24, 0.93, 0.0])
    out = tmp_path / "water.gui"
    save_structure_gui(water, str(out))  # previously raised LinAlgError: singular matrix
    assert _dimensionality(out) == 0
    assert len(load_structure(str(out))) == 3


def test_slab_keeps_2d_periodicity(tmp_path):
    slab = Structure.from_ase(fcc111("Al", size=(2, 2, 3), vacuum=10.0))
    out = tmp_path / "slab.gui"
    save_structure_gui(slab, str(out))
    assert _dimensionality(out) == 2
    assert len(load_structure(str(out))) == len(slab)


def test_edited_supercell_keeps_every_atom(tmp_path):
    """Symmetry reduction must not throw away atoms of a genuinely edited cell."""
    base = as_view(Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64)), CellView.CRYSTALLOGRAPHIC)
    super_cell, _ = tile_supercell(base, (2, 2, 2), None)
    edited = Structure.from_ase(super_cell.to_ase())
    edited.move_atom(0, edited.positions[0] + np.array([0.4, 0.2, 0.1]))

    out = tmp_path / "edited.gui"
    save_structure_gui(edited, str(out))
    assert len(load_structure(str(out))) == len(edited)
