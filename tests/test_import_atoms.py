"""Importing atoms from xyz/pdb/cif to add into the current structure."""

import pytest

pytest.importorskip("ase")

from ase.build import molecule  # noqa: E402
from ase.io import write as ase_write  # noqa: E402

from crystalline.crystalio.loader import read_atoms  # noqa: E402


@pytest.mark.parametrize(
    "fmt,ext",
    [("xyz", "xyz"), ("proteindatabank", "pdb"), ("cif", "cif")],
)
def test_read_atoms_from_common_formats(tmp_path, fmt, ext):
    water = molecule("H2O")
    path = tmp_path / f"water.{ext}"
    ase_write(str(path), water, format=fmt)

    atoms = read_atoms(str(path))
    assert len(atoms) == 3
    assert sorted(set(atoms.get_chemical_symbols())) == ["H", "O"]


def test_read_atoms_first_frame_of_multiframe_xyz(tmp_path):
    frames = [molecule("H2O"), molecule("CH4")]
    path = tmp_path / "traj.xyz"
    ase_write(str(path), frames, format="xyz")  # 2 frames
    atoms = read_atoms(str(path))
    assert len(atoms) == 3  # first frame (water), not the 5-atom methane
