"""The two cells a crystal can be reported and written in: the file's, or standard.

Every deck here is checked the way that matters — rebuilt by
``_crystal_reader`` following the CRYSTAL manual's reading of ``IFLAG IFHR
IFSO`` — rather than by comparing text.
"""

import numpy as np
import pytest

pytest.importorskip("spglib")
pytest.importorskip("pymatgen")

from ase import Atoms  # noqa: E402

from _crystal_reader import crystal_reads, same_crystal, setting_read  # noqa: E402
from crystalline.core import cell_setting  # noqa: E402
from crystalline.core.cell_setting import COMPUTED, STANDARD  # noqa: E402
from crystalline.core.crystal_input import (  # noqa: E402
    CrystalInputSpec,
    GeometryOptions,
    build_input,
    geometry_note,
)
from crystalline.core.crystallography import analyze  # noqa: E402
from crystalline.core.structure import Structure  # noqa: E402


# ── building a crystal in a chosen setting ────────────────────────────────
def _lattice_for(setting):
    from pymatgen.core import Lattice

    n = setting.number
    if n <= 2:
        p = (5.1, 6.3, 7.2, 81, 97, 103)
    elif n <= 15:
        p = (5.1, 6.3, 7.2) + {"a": (103, 90, 90), "b": (90, 103, 90),
                               "c": (90, 90, 103)}[setting.unique_axis]
    elif n <= 74:
        p = (5.1, 7.7, 6.3, 90, 90, 90)          # deliberately not a < b < c
    elif n <= 142:
        p = (5.1, 5.1, 7.3, 90, 90, 90)
    elif n <= 167 and setting.rhombohedral_axes:
        p = (5.5, 5.5, 5.5, 72, 72, 72)
    elif n <= 194:
        p = (5.1, 5.1, 7.3, 90, 90, 120)
    else:
        p = (6.1, 6.1, 6.1, 90, 90, 90)
    return Lattice.from_parameters(*p).matrix


def _crystal(hall, lattice=None, sites=None):
    """A crystal in Hall setting ``hall``, from atoms at general positions."""
    setting = cell_setting.setting_from_hall(hall)
    rotations, translations = cell_setting.operations(hall)
    lattice = _lattice_for(setting) if lattice is None else np.asarray(lattice, dtype=float)
    sites = sites or ((11, (0.1234, 0.2871, 0.3456)), (17, (0.4133, 0.0987, 0.2468)))
    points, numbers = [], []
    for z, point in sites:
        for image in np.asarray(point) @ rotations.transpose(0, 2, 1) + translations:
            image = image - np.floor(image)
            if not any(np.linalg.norm(((image - q + 0.5) % 1 - 0.5) @ lattice) < 1e-4 for q in points):
                points.append(image)
                numbers.append(z)
    return Structure.from_ase(Atoms(numbers=numbers, scaled_positions=points,
                                    cell=lattice, pbc=True))


def _p21n():
    """A P2₁/n molecular crystal, in the cell of a real run."""
    from pymatgen.core import Lattice

    lattice = Lattice.from_parameters(8.24000122, 13.44872633, 8.96772645, 90, 105.483151, 90)
    return _crystal(82, lattice.matrix, ((8, (-0.2878619, 0.1494426, 0.4103222)),
                                         (7, (0.1485990, 0.1245213, -0.4333973)),
                                         (6, (-0.0124827, 0.1523440, -0.4923920)),
                                         (1, (0.2472852, 0.1732136, -0.4048284))))


def _deck(structure, cell):
    return build_input(structure, CrystalInputSpec(geometry=GeometryOptions(cell_setting=cell)))


def _atoms(structure):
    atoms = structure.to_ase()
    return atoms.cell[:], atoms.numbers, atoms.get_scaled_positions()


# ── recognising the setting ───────────────────────────────────────────────
def test_a_p21n_cell_is_recognised_as_p21n():
    setting = cell_setting.find_setting(_p21n())
    assert setting.hall_number == 82
    assert setting.symbol == "P2_1/n"
    assert setting.label == "P2₁/n"
    assert not setting.shifted


def test_crystal_names_each_kind_of_setting_as_the_manual_says():
    record = lambda hall: cell_setting.setting_from_hall(hall).crystal_record()  # noqa: E731
    assert record(81) == ("0 0 0", "14")                 # P 1 21/c 1: by number
    assert record(82) == ("1 0 0", "P 1 21/N 1")         # b unique, cell choice 2
    assert record(84) == ("1 0 0", "P 1 1 21/A")         # c unique: full symbol
    assert record(88) == ("1 0 0", "P 21/N 1 1")         # a unique
    assert record(294) == ("1 0 0", "P B N M")           # Pnma on other axes
    assert record(525) == ("0 0 1", "227")               # Fd-3m, origin choice 1
    assert record(526) == ("0 0 0", "227")               # origin choice 2 is IFSO=0
    assert record(460) == ("0 0 0", "167")               # R-3c, hexagonal axes
    assert record(461) == ("0 1 0", "167")               # R-3c, rhombohedral axes
    assert record(307) == ("1 0 0", "A C A M")           # Cmce, as CRYSTAL spells it (Cmca)


def test_every_symbol_written_names_exactly_one_setting():
    for hall in range(1, 531):
        setting = cell_setting.setting_from_hall(hall)
        flags, group = setting.crystal_record()
        assert setting_read(flags, group).hall_number == hall, (hall, flags, group)


def test_a_shifted_origin_is_found_and_moved_back():
    structure = _p21n()
    atoms = structure.to_ase()
    atoms.translate(np.array([0.13, 0.21, -0.07]) @ atoms.cell[:])
    moved = Structure.from_ase(atoms)
    setting = cell_setting.find_setting(moved)
    assert setting is not None and setting.symbol == "P2_1/n" and setting.shifted
    assert same_crystal(_deck(moved, COMPUTED), *_atoms(moved))


def test_the_primitive_cell_of_a_centred_lattice_has_no_setting():
    from ase.build import bulk

    assert cell_setting.find_setting(Structure.from_ase(bulk("Si", "diamond", a=5.43))) is None


# ── the decks ─────────────────────────────────────────────────────────────
def test_the_computed_deck_keeps_the_computed_cell():
    structure = _p21n()
    deck = _deck(structure, COMPUTED)
    record = deck.splitlines()[2:5]
    assert record[:2] == ["1 0 0", "P 1 21/N 1"]
    assert record[2] == "8.240001 13.448726 8.967726 105.483151"
    assert same_crystal(deck, *_atoms(structure))


def test_the_standard_deck_is_pymatgens_cell_and_still_the_same_crystal():
    structure = _p21n()
    deck = _deck(structure, STANDARD)
    assert deck.splitlines()[2:4] == ["0 0 0", "14"]
    standard = cell_setting.standard_cell(structure)
    assert same_crystal(deck, *_atoms(standard))


def test_a_pnma_crystal_with_b_longest_is_not_written_as_another_crystal():
    # pymatgen sorts the axes a < b < c, which takes Pnma into another setting;
    # writing that cell as "62" rebuilt 28 atoms instead of 20.
    structure = _crystal(292, _lattice_for(cell_setting.setting_from_hall(292)))
    for cell in (STANDARD, COMPUTED):
        reference = structure if cell == COMPUTED else cell_setting.standard_cell(structure)
        assert same_crystal(_deck(structure, cell), *_atoms(reference)), cell


def test_silicon_is_written_with_the_origin_choice_its_coordinates_use():
    # IFSO=0 is CRYSTAL's second origin choice; pymatgen's cell is in the first.
    from ase.build import bulk

    structure = Structure.from_ase(bulk("Si", "diamond", a=5.43, cubic=True))
    deck = _deck(structure, STANDARD)
    assert deck.splitlines()[2:4] == ["0 0 1", "227"]
    setting, _lattice, numbers, _points = crystal_reads(deck)
    assert setting.origin_choice == 1 and len(numbers) == 8


@pytest.mark.parametrize("hall", range(1, 531, 1))
def test_every_setting_rebuilds_in_both_cells(hall):
    structure = _crystal(hall)
    computed = _deck(structure, COMPUTED)
    assert "as in the file" in geometry_note(structure, GeometryOptions(cell_setting=COMPUTED))
    assert same_crystal(computed, *_atoms(structure))
    standard = cell_setting.standard_cell(structure)
    assert same_crystal(_deck(structure, STANDARD), *_atoms(standard))


def test_a_cell_with_no_setting_falls_back_to_the_standard_one_and_says_so():
    from ase.build import bulk

    structure = Structure.from_ase(bulk("Si", "diamond", a=5.43))
    note = geometry_note(structure, GeometryOptions(cell_setting=COMPUTED))
    assert "standard cell is written" in note
    assert _deck(structure, COMPUTED) == _deck(structure, STANDARD)


def test_the_note_names_the_setting_written():
    structure = _p21n()
    assert geometry_note(structure, GeometryOptions(cell_setting=COMPUTED)) == \
        "Cell as in the file: P2₁/n (No. 14)."
    assert geometry_note(structure, GeometryOptions(cell_setting=STANDARD)) == \
        "Standard setting: P2₁/c (No. 14)."
    assert "P1" in geometry_note(structure, GeometryOptions(use_symmetry=False))


def test_a_slab_has_no_note():
    from ase.build import fcc100

    slab = fcc100("Cu", size=(2, 2, 3), vacuum=10.0)
    slab.pbc = (True, True, False)
    assert geometry_note(Structure.from_ase(slab)) == ""


# ── the Info panel's summary ──────────────────────────────────────────────
def test_analyze_reports_the_computed_cell_in_its_own_setting():
    structure = _p21n()
    computed = analyze(structure, cell=COMPUTED)
    standard = analyze(structure, cell=STANDARD)
    assert computed.space_group_symbol == "P2_1/n" and computed.space_group_number == 14
    assert computed.c == pytest.approx(8.96772645, abs=1e-5)
    assert computed.beta == pytest.approx(105.483151, abs=1e-4)
    assert standard.space_group_symbol == "P2_1/c"
    assert standard.c == pytest.approx(10.4338, abs=1e-3)
    assert standard.beta == pytest.approx(124.076, abs=1e-3)
    # the same crystal either way
    assert computed.volume == pytest.approx(standard.volume)
    assert computed.z == standard.z == 4
    assert computed.cell_note is None


def test_analyze_says_when_the_computed_cell_is_not_conventional():
    from ase.build import bulk

    info = analyze(Structure.from_ase(bulk("Si", "diamond", a=5.43)), cell=COMPUTED)
    assert info.space_group_number == 227
    assert info.cell_note is not None
    assert ("Cell", info.cell_note) in info.rows()
    assert info.alpha == pytest.approx(60.0)          # the primitive cell, as shown


def test_the_default_is_still_the_standard_cell():
    structure = _p21n()
    assert analyze(structure) == analyze(structure, cell=STANDARD)
    assert build_input(structure) == _deck(structure, STANDARD)
