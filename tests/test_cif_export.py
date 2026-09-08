"""CIF export: periodic cells (pymatgen CifWriter) and molecules (ASE fallback)."""

import pytest

pytest.importorskip("pymatgen")
pytest.importorskip("ase")

from ase.build import bulk  # noqa: E402

from crystalline.core.cells import (  # noqa: E402
    CellView,
    as_view,
    complete_boundary,
    tile_supercell,
)
from crystalline.core.structure import Structure  # noqa: E402
from crystalline.crystalio.loader import load, load_structure, save_structure_cif  # noqa: E402


def test_cif_round_trips_through_load(tmp_path):
    """A .cif can be opened as a full structure (not just imported as atoms)."""
    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    cif = tmp_path / "nacl.cif"
    save_structure_cif(nacl, str(cif))

    loaded = load_structure(str(cif))
    assert len(loaded) == 8 and loaded.is_periodic
    assert loaded.to_ase().get_chemical_formula() == "Cl4Na4"

    result = load(str(cif))  # the GUI entry point: geometry only, never phonons
    assert result.modes is None and not result.has_phonons
    assert len(result.structure) == 8


def test_periodic_cif_has_cell_and_symmetry(tmp_path):
    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    out = tmp_path / "nacl.cif"
    save_structure_cif(nacl, str(out))
    text = out.read_text()
    assert "_cell_length_a" in text  # lattice written
    assert "_symmetry_space_group_name" in text or "_space_group" in text  # symmetry detected


def test_packed_boundary_cell_saves(tmp_path):
    """The displayed cell is boundary-completed (duplicate periodic images); the
    saver must merge those coincident sites instead of failing symmetry finding."""
    src = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64))
    base = as_view(src, CellView.CRYSTALLOGRAPHIC)
    analysis, _ = tile_supercell(base, (1, 1, 1), None)
    packed, _ = complete_boundary(analysis, None)
    assert len(packed) > len(base)  # genuinely packed with image duplicates

    out = tmp_path / "packed.cif"
    save_structure_cif(packed, str(out))  # previously raised SymmetryUndeterminedError
    assert "_cell_length_a" in out.read_text()


def test_non_periodic_molecule_falls_back_to_ase(tmp_path):
    water = Structure.empty()
    water.add_atom("O", [0.0, 0.0, 0.0])
    water.add_atom("H", [0.96, 0.0, 0.0])
    water.add_atom("H", [-0.24, 0.93, 0.0])
    out = tmp_path / "water.cif"
    save_structure_cif(water, str(out))  # no lattice -> ASE writer, must not raise
    assert out.stat().st_size > 0
    assert "O" in out.read_text()
