"""LOCMODES: the internal coordinates of a run, read, and found again by measuring.

The reader is checked on ``data/locmodes/h2o2_synthetic.out``: hydrogen peroxide
in a 5 Å box, laid out as a CRYSTAL23 output lays out the blocks it reads, with
made-up force constants — not a real run. A real one is checked, on the machine
that has it, by a test that stays out of the repository.
"""

import numpy as np
import pytest

from crystalline.core import measure as M
from crystalline.core.local_modes import LocalCoordinate, LocalModes

from sample_data import path as sample_path

_SYNTHETIC = "locmodes/h2o2_synthetic.out"

# The synthetic run's geometry: O1–O2 along x, an H on each, in a 5 Å cube.
_LATTICE = np.eye(3) * 5.0
_POSITIONS = np.array([[0.0, 0.0, 0.0], [1.45, 0.0, 0.0], [-0.3, 0.9, 0.2], [1.75, -0.2, 0.9]])
_ORIGIN = ((0, 0, 0),)


def _bond(number, a, b, cell=(0, 0, 0), k=4.2, value=1.45):
    return LocalCoordinate(number, M.DISTANCE, (a, b), _ORIGIN + (cell,), ("O", "O"),
                           value, k, None, 8.0, 950.0)


def _angle(number, a, b, c, k=0.7):
    return LocalCoordinate(number, M.ANGLE, (a, b, c), _ORIGIN * 3, ("H", "O", "O"),
                           108.02, k, 0.6, 1.0, 1300.0)


def _modes(*coordinates):
    return LocalModes(_POSITIONS, _LATTICE, tuple(coordinates))


# ── finding a measurement among them ──────────────────────────────────────
def test_a_measured_bond_is_found_from_either_end():
    modes = _modes(_bond(1, 0, 1))
    points = _POSITIONS[[0, 1]]
    assert [c.number for c in modes.find(M.DISTANCE, (0, 1), points)] == [1]
    assert [c.number for c in modes.find(M.DISTANCE, (1, 0), points[::-1])] == [1]


def test_the_same_bond_a_cell_away_is_the_same_bond():
    """The structure on screen is a conventional cell, boundary-completed, maybe
    a supercell: the bond measured may be any image of the one CRYSTAL lists."""
    modes = _modes(_bond(1, 0, 1))
    shifted = _POSITIONS[[0, 1]] + np.array([1, -2, 0]) @ _LATTICE
    assert [c.number for c in modes.find(M.DISTANCE, (0, 1), shifted)] == [1]


def test_one_atom_a_cell_away_is_another_contact():
    """O1 and the O2 of the next cell are the same two atoms, and not the same
    bond: the cell index is part of what names it."""
    modes = _modes(_bond(1, 0, 1))
    stretched = np.array([_POSITIONS[0], _POSITIONS[1] + _LATTICE[0]])
    assert modes.find(M.DISTANCE, (0, 1), stretched) == ()


def test_a_contact_into_the_next_cell_is_found_there():
    contact = _bond(4, 2, 1, cell=(-1, 0, 0), k=0.15, value=3.3782)
    modes = _modes(contact)
    points = np.array([_POSITIONS[2], _POSITIONS[1] - _LATTICE[0]])
    assert modes.find(M.DISTANCE, (2, 1), points) == (contact,)
    assert np.allclose(modes.points(contact), points)


def test_an_angle_keeps_its_vertex_whichever_end_it_is_measured_from():
    modes = _modes(_angle(5, 2, 0, 1))
    points = _POSITIONS[[2, 0, 1]]
    assert [c.number for c in modes.find(M.ANGLE, (2, 0, 1), points)] == [5]
    assert [c.number for c in modes.find(M.ANGLE, (1, 0, 2), points[::-1])] == [5]
    # the same three atoms with another at the vertex is another angle
    assert modes.find(M.ANGLE, (0, 2, 1), _POSITIONS[[0, 2, 1]]) == ()


def test_a_near_linear_angle_is_two_bends():
    """LOCMODES takes a near-linear angle as two bends over the same atoms, each
    with a force constant of its own: a measurement of it is both."""
    first, second = _angle(9, 2, 0, 1, k=0.134), _angle(8, 2, 0, 1, k=0.058)
    modes = _modes(first, second)
    assert modes.find(M.ANGLE, (2, 0, 1), _POSITIONS[[2, 0, 1]]) == (second, first)  # by number


def test_a_measurement_of_another_kind_is_not_found():
    modes = _modes(_bond(1, 0, 1))
    assert modes.find(M.ANGLE, (0, 1), _POSITIONS[[0, 1]]) == ()
    assert modes.find(M.POINT, (0,), _POSITIONS[[0]]) == ()


def test_a_coordinate_names_as_many_atoms_as_its_kind_takes():
    with pytest.raises(ValueError):
        LocalCoordinate(1, M.ANGLE, (0, 1), _ORIGIN * 2, ("O", "O"), 1.0, 1.0, None, 1.0, 1.0)


# ── reading a run ─────────────────────────────────────────────────────────
def test_a_run_is_read_with_every_coordinate_and_its_local_mode():
    from crystalline.crystalio import read_local_modes

    modes = read_local_modes(sample_path(_SYNTHETIC))
    assert [c.number for c in modes.coordinates] == [1, 2, 3, 4, 5, 6, 7]
    assert [c.kind for c in modes.coordinates] == [M.DISTANCE] * 4 + [M.ANGLE] * 2 + [M.DIHEDRAL]
    bond, contact, angle = modes.coordinates[0], modes.coordinates[3], modes.coordinates[4]
    assert (bond.atoms, bond.elements, bond.force_constant) == ((0, 1), ("O", "O"), 4.2)
    assert bond.angular_force_constant is None             # a bond has no kᵃ(d)
    assert (bond.mass, bond.frequency) == (8.0, 950.0)
    assert contact.cells == ((0, 0, 0), (-1, 0, 0))
    assert (angle.atoms, angle.force_constant, angle.angular_force_constant) == ((2, 0, 1), 0.7, 0.6)
    assert np.allclose(modes.lattice, _LATTICE) and np.allclose(modes.positions, _POSITIONS)


def test_every_bond_read_has_the_length_crystal_gives_it():
    """What the positions are for: the cell indices are counted from CRYSTAL's
    Cartesian block, and from there each bond is as long as it says."""
    from crystalline.crystalio import read_local_modes

    modes = read_local_modes(sample_path(_SYNTHETIC))
    for c in modes.coordinates:
        if c.kind == M.DISTANCE:
            a, b = modes.points(c)
            assert np.linalg.norm(b - a) == pytest.approx(c.value, abs=1e-4)


def test_an_output_without_local_modes_has_none():
    from crystalline.crystalio import has_local_modes, read_local_modes

    plain = sample_path("thiourea/thiourea_pbed3_Ahlrichs-pVTZ_freqcalc_80K.out")
    assert not has_local_modes(plain) and read_local_modes(plain) is None
    assert read_local_modes(None) is None                  # a geometry file has no output


@pytest.mark.parametrize("text, cell", [
    ("( 0 0 0)", (0, 0, 0)),
    ("( 0 0-1)", (0, 0, -1)),        # two columns each: a minus runs into its neighbour
    ("(-1-1-1)", (-1, -1, -1)),
    ("( 1-1 1)", (1, -1, 1)),
])
def test_a_cell_index_is_three_signed_integers_however_spaced(text, cell):
    from crystalline.crystalio.locmodes import _cell

    assert _cell(text) == cell


def test_a_row_that_does_not_hold_together_is_left_out():
    """A row whose columns have slipped, as CRYSTALClear once read the
    dihedral whose last cell index is printed (-1-1-1): the cell index is then
    a number and the angle is the force constant. Such a row is not guessed at."""
    from crystalline.crystalio.locmodes import _coordinate

    row = {"ic index": 410, "Atom 1 label": 95, "Atom 1": "C", "Atom 2 label": 75, "Atom 2": "N",
           "Atom 2 cell index": "( 0-1 0)", "Atom 3 label": 19, "Atom 3": "O",
           "Atom 3 cell index": "( 0-1 0)", "Atom 4 label": 33, "Atom 4": "H",
           "Atom 4 cell index": 177.3083, "Angle": 0.424, "k^a": 0.178, "k^a(d)": 8.943,
           "m^a": 347.596, "w^a": float("nan")}
    assert _coordinate(row, M.DIHEDRAL, 4, "Angle", True, natoms=104) is None
    row.update({"Atom 4 cell index": "(-1-1-1)", "Angle": 177.3083, "k^a": 0.424,
                "k^a(d)": 0.178, "m^a": 8.943, "w^a": 347.596})
    repaired = _coordinate(row, M.DIHEDRAL, 4, "Angle", True, natoms=104)
    assert repaired.cells[-1] == (-1, -1, -1) and repaired.atoms == (94, 74, 18, 32)


def test_a_number_that_overflowed_its_field_leaves_its_row_out():
    """CRYSTAL prints ***** for a number too wide for its column, which
    CRYSTALClear reads as NaN: not a number, so not guessed at."""
    from crystalline.crystalio.locmodes import _coordinate

    row = {"ic index": 1, "Atom 1 label": 1, "Atom 1": "O", "Atom 2 label": 2, "Atom 2": "O",
           "Atom 2 cell index": "( 0 0 0)", "Bond Length": 1.45, "k^a": 4.2, "m^a": 8.0,
           "w^a": float("nan")}
    assert _coordinate(row, M.DISTANCE, 2, "Bond Length", False, natoms=4) is None


def test_the_tables_are_read_without_crystalclears_own_symmetry(monkeypatch):
    """CRYSTALClear groups the coordinates by symmetry unless told not to,
    with a check of the geometry that raises on a file it cannot read (the
    synthetic run has no dimension block). The Measure section finds
    equivalent coordinates itself, so it asks for the tables alone."""
    from CRYSTALClear.crystal_io import Crystal_output

    from crystalline.crystalio import read_local_modes

    asked = []
    real = Crystal_output.get_locmodes

    def get_locmodes(self, **options):
        asked.append(options)
        return real(self, **options)

    monkeypatch.setattr(Crystal_output, "get_locmodes", get_locmodes)
    assert read_local_modes(sample_path(_SYNTHETIC)).coordinates
    assert asked == [{"symmetry": False}]


def test_a_row_naming_an_atom_the_cell_has_not_got_is_left_out():
    from crystalline.crystalio.locmodes import _coordinate

    row = {"ic index": 1, "Atom 1 label": 1, "Atom 1": "O", "Atom 2 label": 9, "Atom 2": "O",
           "Atom 2 cell index": "( 0 0 0)", "Bond Length": 1.45, "k^a": 4.2, "m^a": 8.0,
           "w^a": 950.0}
    assert _coordinate(row, M.DISTANCE, 2, "Bond Length", False, natoms=4) is None


# ── the window ────────────────────────────────────────────────────────────
def test_a_file_is_read_with_its_local_modes(monkeypatch):
    """Read with the rest of the file, off the main thread; and when there is a
    LOCMODES section that will not read, why not, for the window to say."""
    pytest.importorskip("PySide6")
    import crystalline.crystalio as crystalio
    from crystalline.ui import main_window as mw

    monkeypatch.setattr(crystalio, "read_local_modes", lambda path: ("the modes of", path))
    plain = sample_path("thiourea/thiourea_pbed3_Ahlrichs-pVTZ_freqcalc_80K.out")
    read = mw._read_file(plain)
    assert read.local_modes is None and read.local_modes_note is None   # none to read
    assert read.capabilities["locmodes"] is False

    monkeypatch.setattr(crystalio, "has_local_modes", lambda path: True)
    read = mw._read_file(plain)
    assert read.local_modes == ("the modes of", plain) and read.local_modes_note is None
    assert read.capabilities["locmodes"] is True

    def unreadable(_path):
        raise crystalio.LocalModesError("the LOCMODES section lists no internal coordinates")

    monkeypatch.setattr(crystalio, "read_local_modes", unreadable)
    read = mw._read_file(plain)
    assert read.local_modes is None and "no internal coordinates" in read.local_modes_note


def test_the_window_hands_the_measure_section_its_local_modes_and_atoms():
    """Checked on the source, like the rest of the window's plumbing: a real
    window needs a live 3D view."""
    pytest.importorskip("PySide6")
    import inspect

    from crystalline.ui.main_window import MainWindow

    assert "self.geometry_panel.set_local_modes(read.local_modes)" in inspect.getsource(
        MainWindow._fill_tab)
    assert "self.geometry_panel.set_source_index(source_index)" in inspect.getsource(
        MainWindow._apply_cell_view)


# ── by symmetry ───────────────────────────────────────────────────────────
# Two O–H units related by the inversion through the origin, in a 5 Å cube: a
# crystal with that symmetry, and nothing else. LOCMODES lists the first O–H only.
_PAIR = np.array([[0.6, 0.1, 0.2], [1.5, 0.4, 0.5], [-0.6, -0.1, -0.2], [-1.5, -0.4, -0.5]])
_PAIR_TYPES = (8, 1, 8, 1)


def _oh(number, a, b, frequency=3500.0):
    return LocalCoordinate(number, M.DISTANCE, (a, b), _ORIGIN * 2, ("O", "H"),
                           0.95, 7.0, None, 0.95, frequency)


def _pair(*coordinates, count=2):
    return LocalModes(_PAIR, _LATTICE, tuple(coordinates), _PAIR_TYPES, count)


def test_a_coordinate_not_listed_is_found_by_symmetry():
    """The second O–H is the image of the first under the inversion: the same
    local mode, though LOCMODES never names it."""
    modes = _pair(_oh(1, 0, 1))
    assert modes.find(M.DISTANCE, (2, 3), _PAIR[[2, 3]]) == ()     # not listed
    found = modes.find_by_symmetry(M.DISTANCE, (2, 3), _PAIR[[2, 3]])
    assert [(c.number, c.symmetry, c.force_constant) for c in found] == [(1, "-x, -y, -z", 7.0)]
    assert modes.find_by_symmetry(M.DISTANCE, (3, 2), _PAIR[[3, 2]])  # from either end
    assert modes.find_by_symmetry(M.DISTANCE, (0, 3), _PAIR[[0, 3]]) == ()  # O1···H4: none


def test_symmetry_is_used_only_when_it_is_the_calculations():
    """The inversion carries no listed coordinate onto another, so nothing in the
    data can bear it out. It is used when CRYSTAL says it used as many operators
    as the geometry has — the same group — and not when it does not say, or
    used fewer: a run in a subgroup, whose local modes need not keep it."""
    for count, kept in ((2, ["-x, -y, -z"]), (None, []), (1, [])):
        assert [o.symbol for o in _pair(_oh(1, 0, 1), count=count).operations()] == kept


def test_an_operation_the_local_modes_contradict_is_not_used():
    """Both O–H listed, with different frequencies: then the inversion is not a
    symmetry of the local modes, whatever the geometry says."""
    modes = _pair(_oh(1, 0, 1), _oh(2, 2, 3, frequency=3400.0))
    assert modes.operations() == []


def test_a_listed_coordinate_is_found_as_listed_not_by_symmetry():
    modes = _pair(_oh(1, 0, 1))
    assert [c.symmetry for c in modes.find(M.DISTANCE, (0, 1), _PAIR[[0, 1]])] == [None]


def test_an_operation_is_written_as_the_international_tables_write_it():
    from crystalline.core.local_modes import _symbol

    assert _symbol(np.diag([-1, 1, -1]), np.array([0.5, 0.5, 0.5])) == "-x+1/2, y+1/2, -z+1/2"
    assert _symbol(np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]]),
                   np.array([0.0, 0.0, 1 / 3])) == "-y, x-y, z+1/3"


@pytest.mark.parametrize("line, count", [
    (" ****    4 SYMMOPS - TRANSLATORS IN FRACTIONAL UNITS", 4),   # what CRYSTALClear's get_symmops misses
    (" ****   48 SYMMOPS - TRANSLATORS IN FRACTIONAL UNITS", 48),
])
def test_crystals_operator_count_is_read_however_spaced(line, count):
    from crystalline.crystalio.locmodes import _operation_count

    assert _operation_count(["a line", line, "another"], 3) == count
    assert _operation_count(["no table here"], 1) is None
