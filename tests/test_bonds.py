"""Tests for chemically-aware connectivity (CrystalNN); needs pymatgen."""

import numpy as np
import pytest

from crystalline.core.bonds import (
    connectivity,
    hydrogen_bonds_from_positions,
    replicate_polyhedra,
)
from crystalline.core.structure import Structure

pytest.importorskip("pymatgen")
from ase import Atoms  # noqa: E402
from ase.build import bulk  # noqa: E402
from ase.data import chemical_symbols  # noqa: E402


def test_connectivity_cation_polyhedra_only():
    # NaCl: Na is the cation centre (Na-Cl octahedra); Cl (anion) gets none.
    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    conn = connectivity(nacl, min_vertices=4)
    assert conn is not None
    assert len(conn.bonds) > 0
    centers = {chemical_symbols[z] for z, _centre, _pts in conn.polyhedra}
    assert centers == {"Na"}  # only the cation is a polyhedron centre
    # each polyhedron is an octahedron (6 ligands) centred on its Na
    assert all(len(pts) == 6 for _z, _centre, pts in conn.polyhedra)
    for _z, centre, pts in conn.polyhedra:
        assert np.allclose(np.mean(pts, axis=0), centre, atol=1e-6)


def test_connectivity_none_for_non_periodic():
    mol = Structure.empty()
    mol.add_atom("C", [0, 0, 0])
    mol.add_atom("O", [1.16, 0, 0])
    assert connectivity(mol) is None


def test_connectivity_skips_reduced_dimensionality_cells():
    """CrystalNN's Voronoi hangs on the 500 Å vacuum of a slab, so a partially
    periodic cell must be declined (the caller then uses the distance fallback)."""
    from ase.build import fcc111

    slab = fcc111("Pt", size=(1, 1, 3), vacuum=0.0)
    cell = np.asarray(slab.get_cell(), dtype=float)
    cell[2] = [0.0, 0.0, 500.0]
    slab.set_cell(cell)
    slab.set_pbc((True, True, False))
    assert connectivity(Structure.from_ase(slab)) is None  # declined, not hung


def _hbonds(structure: Structure) -> np.ndarray:
    """The hydrogen bonds of ``structure`` as drawn — atoms exactly as positioned."""
    return hydrogen_bonds_from_positions(structure.positions, structure.numbers)


def _water_dimer(separation: float) -> Structure:
    """Two waters along x: the first donates an O–H toward the second's O."""
    return Structure.from_ase(Atoms(
        "OH2OH2",
        positions=[
            [0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0],
            [separation, 0, 0], [separation + 0.96, 0.2, 0], [separation - 0.24, -0.93, 0],
        ],
        pbc=False,
    ))


def test_hydrogen_bond_found_in_a_water_dimer():
    hbonds = _hbonds(_water_dimer(2.85))  # O···O ~2.85 Å, near-linear
    assert len(hbonds) == 1
    h, acceptor = hbonds[0]
    assert np.allclose(h, [0.96, 0.0, 0.0])  # the donated hydrogen
    assert 1.3 < np.linalg.norm(acceptor - h) <= 2.6  # H···A in range


def test_no_hydrogen_bond_when_acceptor_too_far():
    assert len(_hbonds(_water_dimer(4.5))) == 0  # O···O too long


def test_ch_does_not_donate_a_hydrogen_bond():
    # Methane near an O: C–H is not a hydrogen-bond donor (C isn't electronegative).
    system = Atoms(
        "CH4O",
        positions=[
            [0, 0, 0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63],
            [-0.63, 0.63, -0.63], [0.63, -0.63, -0.63], [2.0, 0, 0],
        ],
        pbc=False,
    )
    assert len(_hbonds(Structure.from_ase(system))) == 0


def test_hydrogen_bonds_empty_without_hydrogen():
    assert len(_hbonds(Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64)))) == 0


def test_hydrogen_bond_is_directional_prefers_linear_over_closer_bent():
    """The classic surprise: a proton bonds to a farther but in-line acceptor and
    not a closer but off-axis one — hydrogen bonds are directional, not nearest."""
    donor = [-1.0, 0.0, 0.0]
    h = [0.0, 0.0, 0.0]  # O–H points along +x
    near_bent = [0.4, 1.55, 0.0]   # 1.60 Å from H, but D–H···A ≈ 105° (off-axis)
    far_linear = [2.2, 0.0, 0.0]   # 2.20 Å from H, but D–H···A = 180° (in-line)
    system = Atoms("OHOO", positions=[donor, h, near_bent, far_linear], pbc=False)

    hbonds = _hbonds(Structure.from_ase(system))
    assert len(hbonds) == 1
    assert np.allclose(hbonds[0][1], far_linear)  # the linear one, though it's farther


def test_hydrogen_bonds_on_a_displayed_cell_end_on_visible_atoms():
    """On a boundary-completed (displayed) cell the bonds connect the atoms as
    drawn — every endpoint is an atom actually on screen, not a periodic image
    out in space, which is what searching the images too would produce."""
    from crystalline.core.cells import complete_boundary, tile_supercell

    chain = Atoms(
        "OH2",
        positions=[[0, 0, 0], [0.97, 0, 0], [-0.3, 0.9, 0]],
        cell=[2.8, 10, 10],
        pbc=(True, False, False),
    )
    shown, _ = complete_boundary(*tile_supercell(Structure.from_ase(chain), (1, 1, 1)))

    drawn = _hbonds(shown)
    assert len(drawn) > 0
    for _h, acceptor in drawn:
        assert np.any(np.all(np.isclose(shown.positions, acceptor, atol=1e-6), axis=1))


def test_hydrogen_bonds_from_positions_tracks_the_atoms():
    """The positions-based entry point (used for animation frames) re-derives the
    contacts from wherever the atoms currently are."""
    system = _water_dimer(2.85)
    positions, numbers = system.positions, system.numbers

    assert len(hydrogen_bonds_from_positions(positions, numbers)) == 1

    moved = positions.copy()
    moved[3:] += [3.0, 0.0, 0.0]  # push the acceptor water far away
    assert len(hydrogen_bonds_from_positions(moved, numbers)) == 0


def test_replicate_puts_a_polyhedron_on_every_shown_image():
    """Boundary completion draws a corner atom at every cell position it touches;
    each of those images must get the centre's polyhedron, translated."""
    cell = np.eye(3) * 4.0
    ligands = np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0]], dtype=float)
    analysed = [(11, np.zeros(3), ligands)]  # one Na centred at the origin
    # the origin atom as drawn by boundary completion: 8 corners of the cell
    corners = np.array(
        [[x, y, z] for x in (0.0, 4.0) for y in (0.0, 4.0) for z in (0.0, 4.0)], dtype=float
    )
    numbers = np.full(len(corners), 11)

    out = replicate_polyhedra(analysed, cell, corners, numbers)
    assert len(out) == 8  # one per shown image, not one for the whole cell
    centres = sorted(tuple(np.round(pts.mean(axis=0), 6)) for _z, pts in out)
    assert centres == sorted(tuple(np.round(c, 6)) for c in corners)
    assert all(len(pts) == len(ligands) for _z, pts in out)


def test_replicate_ignores_atoms_that_are_not_lattice_images():
    """An atom of the same element at a non-lattice offset is a different site."""
    cell = np.eye(3) * 4.0
    ligands = np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0]], dtype=float)
    analysed = [(11, np.zeros(3), ligands)]
    shown = np.array([[0.0, 0, 0], [1.7, 0.3, 0.0]], dtype=float)  # 2nd is not an image

    out = replicate_polyhedra(analysed, cell, shown, np.array([11, 11]))
    assert len(out) == 1
    assert np.allclose(out[0][1].mean(axis=0), [0, 0, 0])


def test_replicate_survives_a_degenerate_cell():
    ligands = np.zeros((4, 3))
    analysed = [(11, np.zeros(3), ligands)]
    out = replicate_polyhedra(analysed, np.zeros((3, 3)), np.zeros((1, 3)), np.array([11]))
    assert len(out) == 1  # no lattice to image with: returned as analysed


# ── which neighbours are a polyhedron's vertices ─────────────────────────
def _srtio3():
    from ase.spacegroup import crystal

    return Structure.from_ase(crystal(["Sr", "Ti", "O"], [(0, 0, 0), (.5, .5, .5), (.5, .5, 0)],
                                      spacegroup=221, cellpar=[3.905] * 3 + [90] * 3))


def test_a_cation_is_never_a_vertex_of_another_cations_polyhedron():
    """Ti (1.54) is more electronegative than Sr (0.95), and being more
    electronegative than the centre was the whole test: SrTiO3's Sr swallowed
    its 8 Ti neighbours and became a 20-vertex solid."""
    from crystalline.core.bonds import is_ligand

    assert not is_ligand(0.95, 1.54)          # Ti around Sr
    assert is_ligand(0.95, 3.44)              # O around Sr
    assert is_ligand(1.31, 2.20)              # H around Mg, as in a hydride
    assert not is_ligand(3.44, 1.31)          # Mg around O: O is not a centre
    assert not is_ligand(None, 3.44)
    assert not is_ligand(float("nan"), 3.44)


def test_srtio3_has_its_cuboctahedra_and_octahedra():
    pytest.importorskip("pymatgen")
    conn = connectivity(_srtio3())
    assert conn is not None

    counts = {int(z): len(ligands) for z, _centre, ligands in conn.polyhedra}

    assert counts == {38: 12, 22: 6}


def test_the_distance_fallback_draws_the_same_polyhedra_as_crystalnn():
    """The two paths used different ligand rules, so the picture depended on
    which of them a structure's size happened to select."""
    from crystalline.viz.renderer import _fallback_polyhedra

    found = _fallback_polyhedra(_srtio3().to_ase(), 1.1, 4)

    assert {int(z): len(ligands) for z, _centre, ligands in found} == {38: 12, 22: 6}
