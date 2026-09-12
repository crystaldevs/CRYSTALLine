"""The special points as CRYSTAL names them (manual tables 14.1 and 14.2).

Every band deck this app writes and every band file it reads belongs to
CRYSTAL, so its letters are the ones that have to appear on a plot's axis and
in a deck. The standard tables ASE and pymatgen follow agree for the common
lattices and disagree elsewhere — sometimes giving the same letter to a
different point, which is the failure this module exists to prevent.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.lattice import BCC, BCT, CUB, FCC, HEX, MCL, MCLC, ORC, ORCC, ORCF, ORCI, RHL, TET, TRI

from crystalline.core import crystal_points as convention
from crystalline.core.brillouin import special_points
from crystalline.core.structure import Structure

_THIRD = 1.0 / 3.0


def _structure(lattice) -> Structure:
    """One atom at the origin, so the cell keeps the lattice's full symmetry."""
    return Structure.from_ase(Atoms("H", cell=lattice.tocell(), pbc=True))


# ── the tables themselves ─────────────────────────────────────────────────
@pytest.mark.parametrize("lattice, label, coordinates", [
    ("CUB", "X", (0, .5, 0)),
    ("FCC", "W", (.5, .25, .75)),
    ("BCC", "P", (.25, .25, .25)),        # the tutorial's web page prints ¼¼½
    ("HEX", "L", (.5, 0, .5)),            # ...and 0½0 for this one
    ("RHL", "T", (.5, .5, -.5)),
    ("TET", "R", (0, .5, .5)),
    ("BCT", "P", (.5, .5, .5)),
    ("ORC", "U", (.5, 0, .5)),
    ("ORCF", "T", (1, .5, .5)),
    ("ORCC", "R", (0, .5, .5)),
    ("ORCI", "X", (.5, -.5, .5)),
    ("MCL", "B", (.5, 0, 0)),
    ("MCLC", "Y", (0, .5, .5)),
])
def test_a_table_entry_is_what_the_manual_prints(lattice, label, coordinates):
    assert convention.TABLES[lattice][label] == pytest.approx(coordinates)


def test_gamma_belongs_to_every_lattice():
    """The tables leave it out and the manual says why: G is the zone centre
    whatever the Bravais lattice."""
    assert all("G" not in table for table in convention.TABLES.values())
    assert convention.points_for(_structure(FCC(3.6)))["G"] == (0.0, 0.0, 0.0)


def test_triclinic_is_left_to_the_standard_tables():
    """CRYSTAL names no point of it, so nothing here should either."""
    structure = _structure(TRI(3.0, 4.0, 5.0, 70.0, 80.0, 85.0))
    assert convention.bravais_key(structure) is None
    assert convention.points_for(structure) == {}


# ── which table a cell belongs to ─────────────────────────────────────────
def test_the_lattice_is_taken_from_the_cell_not_from_the_space_group():
    """Both settings of a face-centred crystal are drawn in the zone picker,
    and the conventional one is a *simple* cubic lattice: its zone is a cube
    with M, R and X on it. Classified by the crystal's space group instead, it
    was handed X, L and W — points that are not on the zone being drawn."""
    from ase.build import bulk

    conventional = Structure.from_ase(bulk("Cu", "fcc", a=3.6, cubic=True))
    primitive = Structure.from_ase(bulk("Cu", "fcc", a=3.6))
    assert convention.bravais_key(conventional) == "CUB"
    assert convention.bravais_key(primitive) == "FCC"
    assert set(special_points(conventional)) == {"G", "M", "R", "X"}


# ── what the app now says ─────────────────────────────────────────────────
@pytest.mark.parametrize("name, lattice", [
    ("CUB", CUB(3.0)), ("FCC", FCC(3.6)), ("BCC", BCC(2.9)),
    ("HEX", HEX(2.29, 3.59)), ("RHL", RHL(4.0, 54.0)),
    ("TET", TET(3.0, 5.0)), ("BCT", BCT(3.0, 5.0)),
    ("ORC", ORC(3., 4., 5.)), ("ORCF", ORCF(3., 4., 5.)),
    ("ORCI", ORCI(3., 4., 5.)), ("ORCC", ORCC(3., 4., 5.)),
    ("MCL", MCL(3., 4., 5., 70.)), ("MCLC", MCLC(3., 4., 5., 70.)),
])
def test_every_point_crystal_names_is_offered_where_crystal_puts_it(name, lattice):
    points = special_points(_structure(lattice))
    for label, coordinates in convention.TABLES[name].items():
        assert label in points, f"{name}: {label} is missing"
        assert points[label] == pytest.approx(coordinates), f"{name}: {label}"


@pytest.mark.parametrize("name, lattice, label, ours, crystal", [
    # Same letter, a different point: the ones that mattered.
    ("BCT", BCT(3.0, 5.0), "P", (.25, .25, .25), (.5, .5, .5)),
    ("ORCI", ORCI(3., 4., 5.), "X", (-.34, .34, .34), (.5, -.5, .5)),
    ("MCL", MCL(3., 4., 5., 70.), "Y", (0, 0, .5), (0, .5, 0)),
    ("MCL", MCL(3., 4., 5., 70.), "Z", (.5, 0, 0), (0, 0, .5)),
    ("MCLC", MCLC(3., 4., 5., 70.), "M", (.5, 0, .5), (.5, .5, .5)),
])
def test_a_letter_that_meant_another_point_now_means_crystals(name, lattice, label,
                                                              ours, crystal):
    points = special_points(_structure(lattice))
    assert points[label] == pytest.approx(crystal)
    assert points[label] != pytest.approx(ours, abs=1e-3)


@pytest.mark.parametrize("lattice, expected", [
    (FCC(3.6), {"G", "K", "L", "U", "W", "X"}),
    (CUB(3.0), {"G", "M", "R", "X"}),
])
def test_the_lattices_that_already_agreed_keep_every_label(lattice, expected):
    """Cubic, hexagonal, primitive tetragonal and orthorhombic already followed
    CRYSTAL; the extra points the standard tables offer (fcc K and U, which
    CRYSTAL cannot name but can still be reached by coordinates) stay."""
    assert set(special_points(_structure(lattice))) == expected


def test_a_point_crystal_knows_under_another_name_is_not_offered_twice():
    """Rhombohedral F is (0,½,½) to CRYSTAL and (½,½,0) in the standard tables:
    one point, and one dot on the zone, under CRYSTAL's name."""
    points = special_points(_structure(RHL(4.0, 54.0)))
    assert points["F"] == pytest.approx((0, .5, .5))
    rotations = convention.reciprocal_rotations(_structure(RHL(4.0, 54.0)))
    duplicates = [label for label, point in points.items()
                  if label != "F" and convention.same_point(point, (0, .5, .5), rotations)]
    assert duplicates == []


def test_a_slab_keeps_its_two_dimensional_names():
    """CRYSTAL's tables are of the 3D Bravais lattices. A slab's zone is a
    polygon, and its points are the ones ASE names for the 2D lattice."""
    from ase.build import graphene

    atoms = graphene(vacuum=8.0)
    atoms.pbc = [True, True, False]
    assert set(special_points(Structure.from_ase(atoms))) == {"G", "K", "M"}


# ── into the deck ─────────────────────────────────────────────────────────
def test_a_deck_carries_only_letters_crystal_reads_the_same_way():
    """A deck's letters are read back — by CRYSTAL with ISS=0, and by this app
    when it labels a plot. A monoclinic path named by the standard tables would
    have said Y where CRYSTAL means Z."""
    from crystalline.core.properties_input import (
        BandOptions, PropertiesSpec, build_properties_input,
    )
    from crystalline.crystalio.electronic import _band_blocks

    structure = _structure(MCL(3., 4., 5., 70.))
    # An explicit path of CRYSTAL's own points: the conventional path from the
    # standard tables cannot be written for a monoclinic lattice at all, since
    # it visits points whose coordinates depend on the cell parameters and
    # CRYSTAL reads a path as integers over a shrinking factor.
    table = convention.points_for(structure)
    route = [("G", "Y"), ("Y", "Z"), ("Z", "B")]
    deck = build_properties_input(structure, PropertiesSpec(band=BandOptions(
        enabled=True, labels=route,
        segments=[(table[a], table[b]) for a, b in route])))
    rotations = convention.reciprocal_rotations(structure)
    blocks = _band_blocks(deck, "test.d3")
    assert blocks, "the deck has a BAND block"
    for block in blocks:
        for label, corner in zip(block.labels, block.corners):
            if not label:
                continue                      # a corner CRYSTAL does not name
            assert label in table, label
            assert convention.same_point(np.asarray(corner) / block.shrink,
                                         table[label], rotations), label


# ── every lattice can have a deck ─────────────────────────────────────────
_LATTICES = [("CUB", CUB(3.0)), ("FCC", FCC(3.6)), ("BCC", BCC(2.9)),
             ("HEX", HEX(2.29, 3.59)), ("RHL", RHL(4.0, 54.0)),
             ("TET", TET(3.0, 5.0)), ("BCT", BCT(3.0, 5.0)),
             ("ORC", ORC(3., 4., 5.)), ("ORCF", ORCF(3., 4., 5.)),
             ("ORCI", ORCI(3., 4., 5.)), ("ORCC", ORCC(3., 4., 5.)),
             ("MCL", MCL(3., 4., 5., 70.)), ("MCLC", MCLC(3., 4., 5., 70.)),
             ("TRI", TRI(3., 4., 5., 70., 80., 85.))]

# The seven whose standard path visits points whose coordinates depend on the
# cell parameters — 0.411306 for this monoclinic one — which CRYSTAL cannot
# write as whole numbers over a shrinking factor.
_UNWRITABLE = {"RHL", "BCT", "ORCF", "ORCI", "ORCC", "MCL", "MCLC"}


@pytest.mark.parametrize("name, lattice", _LATTICES)
def test_every_bravais_lattice_can_be_written_as_a_deck(name, lattice):
    """Seven of the fourteen used to refuse outright."""
    from crystalline.core.properties_input import (
        BandOptions, PropertiesSpec, build_properties_input,
    )

    deck = build_properties_input(_structure(lattice),
                                  PropertiesSpec(band=BandOptions(enabled=True)))
    assert deck.splitlines()[0] == "BAND"
    assert int(deck.splitlines()[2].split()[0]) >= 1      # NLINE


@pytest.mark.parametrize("name, lattice", [(n, lat) for n, lat in _LATTICES
                                           if n in _UNWRITABLE])
def test_an_unwritable_path_falls_back_to_crystals_own_points(name, lattice):
    """Every point CRYSTAL tabulates is a simple fraction, so a path made of
    them always writes — and every corner is a point the program can name."""
    from crystalline.core.properties_input import band_path, band_path_kind, band_shrink

    structure = _structure(lattice)
    assert band_path_kind(structure) == "points"
    labels, segments = band_path(structure)
    table = convention.points_for(structure)
    assert band_shrink(segments) <= 4, "simple fractions"
    for (start, end), (first, second) in zip(labels, segments):
        assert start == "G" and np.allclose(first, 0.0)
        assert end in table and np.allclose(second, table[end])
    assert {end for _start, end in labels} == set(table) - {"G"}


@pytest.mark.parametrize("name, lattice", [(n, lat) for n, lat in _LATTICES
                                           if n not in _UNWRITABLE])
def test_a_lattice_whose_standard_path_writes_keeps_it(name, lattice):
    """The fallback is for the lattices that need it, and no others."""
    from crystalline.core.properties_input import band_path, band_path_kind

    structure = _structure(lattice)
    assert band_path_kind(structure) == "standard"
    labels, _segments = band_path(structure)
    assert any(start != "G" for start, _end in labels), "a walk, not a star"
