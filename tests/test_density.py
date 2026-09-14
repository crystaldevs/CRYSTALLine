"""Reading a charge density, a spin density or a potential off a 3D grid."""

import numpy as np
import pytest

from crystalline.crystalio import density as D


def _field(shape=(3, 4, 5), steps=None, origin=(0.0, 0.0, 0.0)):
    """A field whose value at each point is a distinct number.

    Distinct, so that a transposed or misordered read cannot pass: every
    permutation of the axes gives a different array.
    """
    values = np.arange(np.prod(shape), dtype=float).reshape(shape) * 0.001
    if steps is None:
        steps = np.array([[1.0, 0.0, 0.0], [0.5, 0.9, 0.0], [0.0, 0.0, 1.2]])
    return D.ScalarField(values=values, origin=np.array(origin, float),
                         steps=np.asarray(steps, float))


CHARGE_TITLE = " Charge density - 3D GRID - GAUSSIAN CUBE FORMAT TEST"
POTENTIAL_TITLE = " Potential - 3D GRID - GAUSSIAN CUBE FORMAT TEST"
SPIN_TITLE = " Spin density - 3D GRID - GAUSSIAN CUBE FORMAT TEST"


def _write_cube(path, field, natoms=1, title=CHARGE_TITLE):
    """A CUBE file holding ``field`` — header in bohr, last index fastest.

    The first line is what CRYSTAL writes there, since that — not the file's
    name — is what says which property the cube holds.
    """
    lines = [title, "written by the test"]
    origin = field.origin / D.BOHR
    lines.append(f"{natoms:5d} {origin[0]:12.6f} {origin[1]:12.6f} {origin[2]:12.6f}")
    for count, step in zip(field.shape, field.steps / D.BOHR):
        lines.append(f"{count:5d} {step[0]:12.6f} {step[1]:12.6f} {step[2]:12.6f}")
    for index in range(natoms):
        lines.append(f"{8:5d} {8.0:12.6f} {index:12.6f} {0.0:12.6f} {0.0:12.6f}")
    flat = field.values.ravel(order="C")
    for start in range(0, flat.size, 6):
        lines.append(" ".join(f"{v:13.5E}" for v in flat[start:start + 6]))
    path.write_text("\n".join(lines) + "\n")


def _write_fort31(path, field, title="charge density"):
    """A fort.31 holding ``field`` — Appendix D p. 447, first index fastest."""
    lines = [title, " ".join(str(n) for n in field.shape)]
    lines.append(" ".join(f"{v:14.7E}" for v in field.origin / D.BOHR))
    for step in field.steps / D.BOHR:
        lines.append(" ".join(f"{v:14.7E}" for v in step))
    flat = field.values.ravel(order="F")
    for start in range(0, flat.size, 5):
        lines.append(" ".join(f"{v:14.7E}" for v in flat[start:start + 5]))
    path.write_text("\n".join(lines) + "\n")


def test_a_cube_is_read_back_point_for_point(tmp_path):
    original = _field()
    path = tmp_path / "DENS_CUBE.DAT"
    _write_cube(path, original)

    read = D.read_cube(path)

    assert read.shape == original.shape
    assert np.allclose(read.values, original.values)
    assert np.allclose(read.steps, original.steps)
    assert np.allclose(read.origin, original.origin)


def test_a_cube_header_is_in_bohr_and_comes_back_in_angstrom(tmp_path):
    """Both formats are written in atomic units; the app works in Angstrom."""
    path = tmp_path / "DENS_CUBE.DAT"
    _write_cube(path, _field(steps=np.eye(3)))

    read = D.read_cube(path)

    assert np.allclose(np.diag(read.steps), 1.0)          # an Angstrom step
    assert not np.allclose(np.diag(read.steps), 1 / D.BOHR)


def test_a_negative_axis_count_means_that_axis_is_already_in_angstrom(tmp_path):
    """Part of the CUBE format, and the opposite of what CRYSTAL writes."""
    field = _field(steps=np.eye(3))
    path = tmp_path / "other.cube"
    _write_cube(path, field)
    lines = path.read_text().splitlines()
    lines[3] = "   -3      2.000000     0.000000     0.000000"
    path.write_text("\n".join(lines) + "\n")

    read = D.read_cube(path)

    assert np.isclose(read.steps[0, 0], 2.0)               # taken as Angstrom
    assert not np.isclose(read.steps[0, 0], 2.0 * D.BOHR)  # and not converted
    assert np.isclose(read.steps[1, 1], 1.0)               # this one came from bohr


def test_a_cube_carries_its_atoms(tmp_path):
    path = tmp_path / "DENS_CUBE.DAT"
    _write_cube(path, _field(), natoms=3)

    read = D.read_cube(path)

    assert read.numbers is not None and len(read.numbers) == 3
    assert read.positions.shape == (3, 3)
    assert np.isclose(read.positions[1, 0], 1.0 * D.BOHR)  # bohr -> Angstrom


def test_a_fort31_is_read_back_point_for_point(tmp_path):
    original = _field()
    path = tmp_path / "fort.31"
    _write_fort31(path, original)

    read = D.read_fort31(path)

    assert np.allclose(read.values, original.values)
    assert read.kind == D.CHARGE


def test_the_two_formats_of_one_run_hold_the_same_field(tmp_path):
    """A run writes both, so each is a check on the other.

    This fixture asserts our two readers agree with our two writers, not that
    CRYSTAL orders its fort.31 the way we assume — the manual does not say.
    Run it on a real pair of files and it becomes that check.
    """
    original = _field()
    _write_cube(tmp_path / "DENS_CUBE.DAT", original)
    _write_fort31(tmp_path / "fort.31", original)

    assert D.agree(D.read_field(tmp_path / "DENS_CUBE.DAT"),
                   D.read_field(tmp_path / "fort.31"))


def test_a_truncated_file_says_so_rather_than_drawing_a_wrong_field(tmp_path):
    path = tmp_path / "DENS_CUBE.DAT"
    _write_cube(path, _field())
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-3]) + "\n")

    with pytest.raises(D.DensityError, match="promises"):
        D.read_cube(path)


def test_a_fortran_exponent_with_no_e_is_still_a_number(tmp_path):
    """Fortran drops the E when the exponent needs all three columns."""
    assert np.isclose(D._repair_exponent("0.1234-102"), 1.234e-103)
    assert np.isclose(D._floats(["1.0 0.5000-105 2.0"])[1], 0.5e-105)


def test_the_format_is_told_from_the_file_not_the_name(tmp_path):
    """DENS_CUBE.DAT and BAND.DAT share an extension."""
    _write_cube(tmp_path / "anything.dat", _field())
    _write_fort31(tmp_path / "fort.31", _field())
    (tmp_path / "BAND.DAT").write_text("# NKPT 10\n1 2 3\n4 5 6\n7 8 9\n1 1 1\n2 2 2\n")

    assert D.file_format(tmp_path / "anything.dat") == "cube"
    assert D.file_format(tmp_path / "fort.31") == "fort31"
    assert D.file_format(tmp_path / "BAND.DAT") is None


def test_the_cubes_of_a_folder_are_found_with_the_cube_before_the_fort31(tmp_path):
    """A cube carries the atoms and states its own value order."""
    _write_cube(tmp_path / "DENS_CUBE.DAT", _field())
    _write_fort31(tmp_path / "fort.31", _field())

    found = D.find_fields(tmp_path)

    assert [D.Path(p).name for p in found] == ["DENS_CUBE.DAT", "fort.31"]


def test_a_cube_is_known_by_its_first_line_and_never_by_its_name(tmp_path):
    """Outputs are renamed as soon as they leave the scratch directory, and
    guessing from the tutorial's ``mgo_pot3.cube`` took every potential for a
    charge density. Here every name lies."""
    for name, title, kind in (("POT_CUBE.DAT", CHARGE_TITLE, D.CHARGE),
                              ("DENS_CUBE.DAT", POTENTIAL_TITLE, D.POTENTIAL),
                              ("mgo_ech3.cube", SPIN_TITLE, D.SPIN),
                              ("SPIN_CUBE.DAT", " Something else entirely", D.FIELD)):
        _write_cube(tmp_path / name, _field(), title=title)
        assert D.read_cube(tmp_path / name).kind == kind, name


def test_the_tutorials_own_file_names_are_read_correctly():
    """The case that exposed the name guessing, on the real files if present."""
    import os

    base = "/Users/davidemitoli/QMMC2026/OneElectronProperties/output"
    if not os.path.isdir(base):
        pytest.skip("no QMMC output folder")
    assert D.read_field(os.path.join(base, "mgo_pot3.cube")).kind == D.POTENTIAL
    assert D.read_field(os.path.join(base, "mgo_ech3.cube")).kind == D.CHARGE


def test_a_field_nothing_identifies_is_drawn_by_what_its_values_do(tmp_path):
    positive = _field()
    signed = D.replace(positive, values=positive.values - 0.01)
    _write_cube(tmp_path / "a", positive, title=" unlabelled")
    _write_cube(tmp_path / "b", signed, title=" unlabelled")

    assert D.read_cube(tmp_path / "a").kind == D.FIELD
    assert not D.read_cube(tmp_path / "a").signed
    assert D.read_cube(tmp_path / "b").signed


def test_a_charge_density_is_unsigned_and_a_spin_density_is_not(tmp_path):
    _write_cube(tmp_path / "one", _field(), title=CHARGE_TITLE)
    _write_cube(tmp_path / "two", _field(), title=SPIN_TITLE)

    assert not D.read_cube(tmp_path / "one").signed
    assert D.read_cube(tmp_path / "two").signed


def test_a_folder_is_searched_by_extension_and_read_by_content(tmp_path):
    """The extension says it is a grid file; only its content says what of."""
    _write_cube(tmp_path / "DENS_CUBE.DAT", _field(), title=POTENTIAL_TITLE)   # a lying stem
    _write_cube(tmp_path / "mgo_pot3.cube", _field(), title=CHARGE_TITLE)      # another
    _write_cube(tmp_path / "notes.txt", _field(), title=CHARGE_TITLE)          # not a grid extension
    (tmp_path / "density.cube").write_text("this is not a cube\n" * 10)

    found = [D.Path(p).name for p in D.find_fields(tmp_path)]

    assert found == ["mgo_pot3.cube", "DENS_CUBE.DAT"]   # the charge density first


def test_a_difference_needs_the_same_grid(tmp_path):
    first = _field()
    other = _field(shape=(3, 4, 6))

    with pytest.raises(D.DensityError, match="different grids"):
        D.difference(first, other)


def test_a_difference_is_signed_even_between_two_densities():
    first, second = _field(), _field()
    delta = D.difference(first, D.replace(second, values=second.values * 2))

    assert delta.signed
    assert np.allclose(delta.values, -first.values)
    assert delta.kind == D.DIFFERENCE and delta.of == D.CHARGE
    assert delta.unit == "e/bohr³"


def test_the_grid_points_run_along_the_lattice_vectors_not_the_axes():
    """A sheared cell is the case an origin-and-spacing grid gets wrong."""
    steps = np.array([[1.0, 0.0, 0.0], [-0.5, 0.866, 0.0], [0.0, 0.0, 1.6]])
    field = _field(shape=(2, 2, 2), steps=steps)

    points = field.points()

    assert np.allclose(points[0, 1, 0], steps[1])   # one step along b, not along y
    assert np.allclose(points[1, 1, 1], steps.sum(axis=0))


def test_the_lattice_is_settled_against_the_structure_rather_than_guessed():
    """Neither format says whether the far face of the cell was sampled."""
    cell = np.diag([4.0, 4.0, 4.0])
    inclusive = _field(shape=(5, 5, 5), steps=np.eye(3))          # 4 steps of 1.0
    exclusive = _field(shape=(4, 4, 4), steps=np.eye(3))          # 4 steps of 1.0

    assert np.allclose(D.lattice_of(inclusive, cell), cell)
    assert np.allclose(D.lattice_of(exclusive, cell), cell)
    # With no cell to check against, the inclusive reading is taken.
    assert np.allclose(D.lattice_of(exclusive), np.diag([3.0, 3.0, 3.0]))


def test_a_density_can_be_asked_for_in_electrons_per_angstrom_cubed():
    field = _field()

    converted = field.in_angstrom_cubed()

    assert np.allclose(converted.values, field.values / D.BOHR ** 3)


def test_the_grids_of_the_open_structure_come_first_by_their_lattice(tmp_path):
    """A folder shared by several systems: the lattice in a grid's header says
    which one it belongs to — its name does not, and neither does its age."""
    ours = _field(steps=0.5 * np.eye(3))                    # a 1 x 1.5 x 2 Å cell
    theirs = _field(steps=0.6 * np.eye(3))
    _write_cube(tmp_path / "theirs_dens.cube", theirs)
    _write_cube(tmp_path / "ours_is_misnamed.cube", ours)
    import os
    os.utime(tmp_path / "ours_is_misnamed.cube", (1e9 - 100, 1e9 - 100))
    os.utime(tmp_path / "theirs_dens.cube", (1e9, 1e9))    # the newer

    cell = ours.steps * (np.array(ours.shape) - 1)[:, None]
    found = [D.Path(p).name for p in D.find_fields(tmp_path, cell=cell)]

    assert found[0] == "ours_is_misnamed.cube"
    assert D._same_lattice(cell[[2, 0, 1]], cell)          # the same cell, reordered
