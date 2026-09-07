"""Tests for crystallographic analysis (needs pymatgen)."""

import numpy as np
import pytest

from crystalline.core.crystallography import analyze
from crystalline.core.structure import Structure

pytest.importorskip("pymatgen")
from ase.build import bulk  # noqa: E402


def test_analyze_fcc_nacl():
    info = analyze(Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64)))
    assert info.periodic
    assert info.formula == "NaCl"
    assert info.space_group_symbol == "Fm-3m"
    assert info.space_group_number == 225
    assert info.crystal_system == "cubic"
    # conventional cell params (not the 2-atom primitive) are reported
    assert abs(info.a - 5.64) < 1e-3 and abs(info.alpha - 90.0) < 1e-3
    assert info.z == 4
    assert info.density and info.density > 0
    # every reported row is a (label, value) string pair
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in info.rows())


def test_analyze_molecule_is_non_periodic():
    mol = Structure.empty()
    mol.add_atom("O", [0.0, 0.0, 0.0])
    mol.add_atom("H", [0.96, 0.0, 0.0])
    mol.add_atom("H", [-0.24, 0.93, 0.0])

    info = analyze(mol)
    assert not info.periodic
    assert info.space_group_symbol is None
    assert "0D" in info.dimensionality or "Molecule" in info.dimensionality
    assert info.n_atoms == 3


def test_analyze_empty_structure():
    info = analyze(Structure.empty())
    assert info.n_atoms == 0
    assert info.rows()  # still yields at least the placeholder rows


def _crystal_slab(atoms) -> Structure:
    """A 2D slab as CRYSTAL writes it: aperiodic c formally set to 500 Å."""
    cell = np.asarray(atoms.get_cell(), dtype=float)
    cell[2] = [0.0, 0.0, 500.0]
    atoms.set_cell(cell)
    atoms.set_pbc((True, True, False))
    return Structure.from_ase(atoms)


def test_analyze_slab_uses_layer_group_not_space_group():
    from ase.build import graphene

    info = analyze(_crystal_slab(graphene()))
    assert info.ndim == 2
    assert "2D" in info.dimensionality
    # a slab is described by its layer group, never a 3D space group
    assert info.layer_group_symbol == "p6/mmm"
    assert info.layer_group_number == 80
    assert info.point_group == "6/mmm"
    assert info.space_group_symbol is None


def test_analyze_slab_drops_the_500_angstrom_artifacts():
    from ase.build import fcc111

    info = analyze(_crystal_slab(fcc111("Pt", size=(1, 1, 3), vacuum=0.0)))
    # the formal 500 Å c, the vacuum-inflated volume and the bogus density are gone
    assert info.c is None and info.volume is None and info.density is None
    # in-plane metrics are reported instead
    assert info.a is not None and info.b is not None and info.area is not None
    assert abs(info.a - 2.7719) < 1e-3

    labels = [label for label, _ in info.rows()]
    assert "Layer group" in labels and "Cell area (Å²)" in labels
    assert "Space group" not in labels
    assert not any("Density" in lbl or "volume" in lbl for lbl in labels)
    assert not any("500" in value for _, value in info.rows())


def test_analyze_polymer_reports_single_repeat_length():
    from ase import Atoms

    chain = Atoms(
        "C2H2",
        positions=[[0, 0, 0], [1.3, 0.2, 0], [0, -1, 0], [1.3, 1.2, 0]],
        cell=[2.5, 500.0, 500.0],
        pbc=(True, False, False),
    )
    info = analyze(Structure.from_ase(chain))
    assert info.ndim == 1 and "1D" in info.dimensionality
    assert abs(info.a - 2.5) < 1e-6
    assert info.b is None and info.c is None  # the two 500 Å axes are not reported
    assert info.volume is None and info.density is None
    assert not any(lbl.startswith(("a, b", "α")) for lbl, _ in info.rows())
