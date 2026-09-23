"""Layer-group symmetry for slabs, and the decks written from it.

Checked against crystallography and against a real CRYSTAL deck
(``MSSC2025/.../2D/1_graphene/graphene_testgeom.d12``), not against what the
code happens to emit: graphene is p6/mmm with one carbon in the asymmetric
unit, and a slab deck that says otherwise is wrong however plausible it reads.
"""

import numpy as np
import pytest
from ase.build import fcc100, fcc111, graphene

from crystalline.core import slab_symmetry as layers
from crystalline.core.structure import Structure


def _slab(atoms, pbc=(True, True, False)) -> Structure:
    atoms = atoms.copy()
    atoms.pbc = list(pbc)
    return Structure.from_ase(atoms)


def _graphene() -> Structure:
    return _slab(graphene(vacuum=8.0))


# ── the group ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name, structure, number, symbol, sites", [
    ("graphene", _graphene(), 80, "p6/mmm", 1),
    ("Cu(111)", _slab(fcc111("Cu", size=(1, 1, 4), vacuum=10.0)), 72, "p-3m1", 2),
    ("Cu(100)", _slab(fcc100("Cu", size=(1, 1, 3), vacuum=10.0)), 61, "p4/mmm", 2),
])
def test_a_slab_gets_its_layer_group_and_its_orbits(name, structure, number, symbol, sites):
    symmetry = layers.analyse(structure)
    assert symmetry is not None, name
    assert (symmetry.number, symmetry.symbol) == (number, symbol), name
    numbers, coords = layers.asymmetric_unit(symmetry)
    assert len(numbers) == sites == len(coords), name
    assert len(numbers) <= len(structure)


def test_a_layer_straddling_the_cell_boundary_keeps_its_mirror():
    """spglib's standardisation can leave a slab across the cell edge, with
    atoms at z = 0.02 and z = 0.98 that are really neighbours. Asked about a
    cell like that the detector misses the mirror through the layer's centre —
    Cu(100) comes back p4mm (55) instead of p4/mmm (61), half the symmetry.
    """
    symmetry = layers.analyse(_slab(fcc100("Cu", size=(1, 1, 3), vacuum=10.0)))
    assert symmetry.number == 61, "the layer must be sat whole inside the cell"


def test_the_vacuum_need_not_be_the_third_axis():
    """A slab stacked along a is the same slab. The standard setting of a layer
    group always puts the aperiodic direction on c, and carrying the input's
    axis index into that setting reported p1 for a perfectly symmetric slab."""
    base = fcc111("Cu", size=(1, 1, 4), vacuum=10.0)
    rolled = base.copy()
    rolled.set_cell(np.asarray(base.get_cell())[[2, 0, 1]], scale_atoms=False)
    rolled.set_positions(base.get_positions())     # the same vectors, reordered

    symmetry = layers.analyse(_slab(rolled, pbc=(False, True, True)))
    assert symmetry is not None
    assert (symmetry.number, symmetry.symbol) == (72, "p-3m1")
    assert symmetry.normal_axis == 2, "the standard setting puts the vacuum on c"


def test_anything_that_is_not_a_slab_is_declined():
    from ase.build import bulk, molecule

    assert layers.analyse(Structure.from_ase(bulk("MgO", "rocksalt", a=4.21))) is None
    assert layers.analyse(Structure.from_ase(molecule("H2O"))) is None
    assert not layers.is_slab(Structure.from_ase(bulk("Cu", "fcc", a=3.6)))


# ── what CRYSTAL is told ──────────────────────────────────────────────────
@pytest.mark.parametrize("structure, expected", [
    (_graphene(), 1),                                                   # hexagonal: a
    (_slab(fcc100("Cu", size=(1, 1, 3), vacuum=10.0)), 1),              # square: a
])
def test_the_cell_record_carries_only_what_the_group_leaves_free(structure, expected):
    symmetry = layers.analyse(structure)
    assert len(layers.cell_record(symmetry)) == expected


def test_a_flat_monolayer_sits_at_zero():
    """Height is measured from the layer group's own origin, which is the
    middle of the layer — that is where a horizontal mirror lies. Measured from
    the cell edge instead, graphene lands at z = -8 Å and p6/mmm mirrors it
    into a second sheet sixteen Ångström away: a different material.
    """
    symmetry = layers.analyse(_graphene())
    _numbers, coords = layers.asymmetric_unit(symmetry)
    deck = layers.deck_coordinates(symmetry, coords)
    assert abs(deck[0][2]) < 1e-6, "a flat layer is at the origin, not at the cell edge"


def test_the_group_is_only_offered_once_it_rebuilds_the_slab():
    """The group theory is exact; the setting is the risk. A number naming the
    wrong arrangement of axes gives a plausible file describing another
    surface, so it is checked by doing what CRYSTAL will do."""
    for structure in (_graphene(),
                      _slab(fcc111("Cu", size=(1, 1, 4), vacuum=10.0)),
                      _slab(fcc100("Cu", size=(1, 1, 3), vacuum=10.0))):
        assert layers.regenerates(layers.analyse(structure))


# ── the deck ──────────────────────────────────────────────────────────────
def _deck(structure, **geometry):
    from crystalline.core.crystal_input import (
        CrystalInputSpec, GeometryOptions, build_input,
    )

    lines = build_input(
        structure, CrystalInputSpec(geometry=GeometryOptions(**geometry))).splitlines()
    return lines[1:lines.index("BASISSET")]


def test_a_slab_deck_is_written_in_its_real_layer_group():
    """It was written in layer group 1 with every atom listed, throwing away
    symmetry the app had already found and making CRYSTAL work through orbits
    it could have been told about."""
    deck = _deck(_graphene())
    assert deck[0] == "SLAB"
    assert deck[1] == "80"
    assert len(deck[2].split()) == 1, "a hexagonal cell needs only a"
    assert deck[3] == "1"
    # ...and the atom line is fractional, fractional, Ångström
    z_number, x, y, z = deck[4].split()
    assert int(z_number) == 6
    assert 0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0
    assert abs(float(z)) < 1e-6


def test_the_asymmetric_unit_is_smaller_than_the_slab():
    structure = _slab(fcc111("Cu", size=(1, 1, 4), vacuum=10.0))
    deck = _deck(structure)
    assert deck[1] == "72"
    assert int(deck[3]) == 2 < len(structure)


def test_symmetry_off_still_writes_every_atom_in_group_one():
    """The escape hatch has to keep working: group 1 is always true."""
    structure = _slab(fcc111("Cu", size=(1, 1, 4), vacuum=10.0))
    deck = _deck(structure, use_symmetry=False)
    assert deck[1] == "1"
    assert int(deck[3]) == len(structure)
    assert len(deck[2].split()) == 3, "an oblique cell needs a, b and gamma"


def test_a_slab_with_no_findable_group_falls_back_rather_than_guessing():
    """A jumbled layer has no symmetry to write; the deck must still be valid."""
    rng = np.random.default_rng(0)
    atoms = fcc111("Cu", size=(2, 2, 2), vacuum=10.0)
    atoms.positions += rng.normal(scale=0.25, size=atoms.positions.shape)
    structure = _slab(atoms)

    deck = _deck(structure)
    assert deck[0] == "SLAB"
    assert int(deck[3]) == len(structure), "every atom, since none are equivalent"


def test_crystals_and_molecules_are_untouched_by_any_of_this():
    from ase.build import bulk, molecule

    assert _deck(Structure.from_ase(bulk("MgO", "rocksalt", a=4.21)))[0] == "CRYSTAL"
    assert _deck(Structure.from_ase(molecule("H2O")))[0] == "MOLECULE"


# ── reciprocal space, in two dimensions ───────────────────────────────────
@pytest.mark.parametrize("name, structure, corners, points", [
    ("graphene", _graphene(), 6, {"G", "K", "M"}),
    ("Cu(111)", _slab(fcc111("Cu", size=(1, 1, 4), vacuum=10.0)), 6, {"G", "K", "M"}),
    ("Cu(100)", _slab(fcc100("Cu", size=(1, 1, 3), vacuum=10.0)), 4, {"G", "M", "X"}),
])
def test_a_slab_zone_is_a_polygon_with_in_plane_points(name, structure, corners, points):
    """A slab's zone is two-dimensional. Built from the full 3x3 cell it comes
    out a thin *solid* whose thickness is an artefact of how much vacuum was
    left around the layer, carrying points along a k_z that does not exist."""
    from crystalline.core import brillouin

    vertices, faces = brillouin.brillouin_zone(structure)
    assert len(faces) == 1, f"{name}: a polygon has one face"
    assert len(vertices) == corners, name
    assert set(brillouin.special_points(structure)) == points, name


def test_every_zone_vertex_lies_in_the_plane():
    from crystalline.core import brillouin

    structure = _graphene()
    vertices, _faces = brillouin.brillouin_zone(structure)
    normal = np.cross(*brillouin.plane_reciprocal(structure))
    normal /= np.linalg.norm(normal)
    assert np.abs(vertices @ normal).max() < 1e-9


def test_a_slab_band_path_never_leaves_the_plane():
    """Read as a 3D crystal a slab's conventional path runs G-M-K-G-A-L-H-A...,
    six of nine segments along a reciprocal direction it does not have."""
    from crystalline.core.properties_input import band_path

    for structure in (_graphene(),
                      _slab(fcc111("Cu", size=(1, 1, 4), vacuum=10.0)),
                      _slab(fcc100("Cu", size=(1, 1, 3), vacuum=10.0))):
        labels, segments = band_path(structure)
        assert not {name for pair in labels for name in pair} & {"A", "L", "H", "R"}
        for start, end in segments:
            assert abs(start[2]) < 1e-9 and abs(end[2]) < 1e-9


def test_a_slab_keeps_its_own_cell_for_the_zone():
    """There is no 'primitive standard' cell for a layer group in the 3D sense;
    running a slab through that standardiser rebuilds it around the vacuum."""
    from crystalline.core.brillouin import PRIMITIVE, zone_lattice

    structure = _graphene()
    assert zone_lattice(structure, PRIMITIVE) is structure


def test_a_slab_point_is_placed_with_its_own_reciprocal_vectors():
    from crystalline.core import brillouin

    structure = _graphene()
    points = brillouin.special_points(structure)
    normal = np.cross(*brillouin.plane_reciprocal(structure))
    normal /= np.linalg.norm(normal)
    for label, fractional in points.items():
        cartesian = brillouin.to_cartesian(structure, fractional)
        assert abs(float(cartesian @ normal)) < 1e-9, label


def test_crystals_keep_their_three_dimensional_zone():
    from ase.build import bulk

    from crystalline.core import brillouin

    vertices, faces = brillouin.brillouin_zone(
        Structure.from_ase(bulk("MgO", "rocksalt", a=4.21)))
    assert (len(vertices), len(faces)) == (24, 14)


# ── the reduction dialog knows what it cannot do ──────────────────────────
def test_symmetry_reduction_declines_slabs_rather_than_pretending():
    """It reported a *space* group for a slab — the vacuum counted as a lattice
    vector — and the slab deck ignored the answer, so the whole control did
    nothing. A slab's symmetry belongs to this module instead."""
    from crystalline.core import symmetry_reduction

    assert symmetry_reduction.analyse(_graphene()) is None
    assert symmetry_reduction.analyse(
        _slab(fcc111("Cu", size=(1, 1, 4), vacuum=10.0))) is None


def test_the_reduction_dialog_says_why_a_slab_cannot_be_reduced():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from crystalline.ui.panels.symmetry_reduction import SymmetryReductionDialog

    QApplication.instance() or QApplication([])
    dialog = SymmetryReductionDialog(_graphene())
    assert "layer group" in dialog._header.text()
    assert dialog.chosen() == ()


# ── a slab's band path has to be writable too ─────────────────────────────
def _plane_slab(lattice, height: float = 20.0) -> Structure:
    from ase import Atoms

    cell = np.asarray(lattice.tocell())
    cell[2] = [0.0, 0.0, height]
    return Structure.from_ase(Atoms("C", cell=cell, pbc=[True, True, False]))


@pytest.mark.parametrize("name, lattice", [
    ("SQR", "SQR"), ("RECT", "RECT"), ("CRECT", "CRECT"),
    ("HEX2D", "HEX2D"), ("OBL", "OBL"),
])
def test_every_plane_lattice_can_be_written_as_a_deck(name, lattice):
    """An oblique or centred rectangular slab's conventional path runs through
    points whose coordinates depend on the cell's angle — 0.4319 for one — and
    CRYSTAL reads a path as whole numbers over a shrinking factor."""
    from ase.lattice import CRECT, HEX2D, OBL, RECT, SQR

    from crystalline.core.properties_input import (
        BandOptions, PropertiesSpec, band_path, band_shrink, build_properties_input,
    )

    built = {"SQR": SQR(3.0), "RECT": RECT(3.0, 4.5), "CRECT": CRECT(3.0, 70.0),
             "HEX2D": HEX2D(2.46), "OBL": OBL(3.0, 4.0, 75.0)}[lattice]
    structure = _plane_slab(built)
    _labels, segments = band_path(structure)
    assert band_shrink(segments) <= 6
    for start, end in segments:
        assert start[2] == 0.0 and end[2] == 0.0, "a slab has no k_z"
    deck = build_properties_input(structure, PropertiesSpec(band=BandOptions(enabled=True)))
    assert deck.splitlines()[0] == "BAND"


def test_a_slab_whose_path_cannot_be_written_falls_back_to_its_own_points():
    from ase.lattice import OBL

    from crystalline.core.properties_input import band_path, band_path_kind

    from crystalline.core import brillouin

    structure = _plane_slab(OBL(3.0, 4.0, 75.0))
    assert band_path_kind(structure) == "points"
    labels, segments = band_path(structure)
    points = brillouin.special_points(brillouin.zone_lattice(structure))
    for (start, end), (_first, second) in zip(labels, segments):
        assert start == "G"
        assert np.allclose(second, points[end]), end


def test_a_slab_whose_path_writes_keeps_it():
    from crystalline.core.properties_input import band_path_kind

    assert band_path_kind(_graphene()) == "standard"


def test_a_slab_deck_is_labelled_from_its_plane_lattice():
    """Not from the 3D tables: a slab's letters are the ones its own zone shows."""
    from crystalline.core.properties_input import (
        BandOptions, PropertiesSpec, build_properties_input,
    )

    from crystalline.core import brillouin

    structure = _graphene()
    deck = build_properties_input(structure, PropertiesSpec(band=BandOptions(enabled=True)))
    written = {name for line in deck.splitlines()
               for name in line.split()[6:8] if name.isalpha()}
    assert written <= set(brillouin.special_points(brillouin.zone_lattice(structure)))
    assert written == {"G", "M", "K"}
