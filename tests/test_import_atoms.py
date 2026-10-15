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


# ── what the window does with them ────────────────────────────────────────
def _importing_window():
    """A stand-in carrying the window's import, and what it reports to."""
    pytest.importorskip("PySide6")

    from crystalline.core.structure import Structure
    from crystalline.ui.main_window import MainWindow

    class _Action:
        def setChecked(self, on):
            pass

    class _Panel:
        def __init__(self):
            self.selected = None

        def set_elements(self, _numbers):
            pass

        def set_selection(self, indices):
            self.selected = list(indices)

    class _Window:
        _import_path = MainWindow._import_path

        def __init__(self):
            self.structure = Structure.from_ase(molecule("CH4"))
            self.display_panel = _Panel()
            self.structure_panel = _Panel()
            self._editing = False
            self._edit_mode_action = _Action()
            self.said = []

        def statusBar(self):
            outer = self

            class _Bar:
                @staticmethod
                def showMessage(text, _timeout=0):
                    outer.said.append(text)

            return _Bar()

    return _Window()


def test_an_import_says_how_many_atoms_arrived_and_where(tmp_path):
    """An import can be invisible — three atoms at their own coordinates, among
    thirty already there — so the window says what it did and that they are
    selected, which is how they are found."""
    path = tmp_path / "water.xyz"
    ase_write(str(path), molecule("H2O"), format="xyz")
    window = _importing_window()

    assert window._import_path(str(path)) is True

    assert len(window.structure) == 5 + 3
    assert window.structure_panel.selected == [5, 6, 7]   # the imported ones
    assert window.said == ["Imported 3 atoms from water.xyz  ·  selected, "
                           "and in the undo history"]


def test_one_atom_is_reported_in_the_singular(tmp_path):
    from ase import Atoms

    path = tmp_path / "lone.xyz"
    ase_write(str(path), Atoms("He", positions=[(0.0, 0.0, 0.0)]), format="xyz")
    window = _importing_window()

    window._import_path(str(path))

    assert "Imported 1 atom from lone.xyz" in window.said[0]
