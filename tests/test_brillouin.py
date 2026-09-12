"""The first Brillouin zone, and picking a band path on it.

The zone is computed as the Wigner-Seitz cell of the reciprocal lattice — a
Voronoi cell — rather than from the fourteen per-Bravais-lattice constructions
crystallography books give. So the tests are the shapes those books draw: if the
general method is right it reproduces every one of them.
"""

import numpy as np
import pytest
from ase.build import bulk

from crystalline.core.brillouin import (
    CONVENTIONAL,
    PRIMITIVE,
    brillouin_zone,
    nearest_special_point,
    reciprocal_cell,
    special_points,
    to_cartesian,
    zone_lattice,
)
from crystalline.core.structure import Structure


@pytest.mark.parametrize("name, atoms, faces, vertices, sides", [
    ("fcc — truncated octahedron", bulk("Cu", "fcc", a=3.6), 14, 24, {4, 6}),
    ("bcc — rhombic dodecahedron", bulk("Fe", "bcc", a=2.87), 12, 14, {4}),
    ("simple cubic — cube", bulk("Po", "sc", a=3.35), 6, 8, {4}),
    ("hcp — hexagonal prism", bulk("Mg", "hcp", a=3.21, c=5.21), 8, 12, {4, 6}),
])
def test_the_zone_is_the_shape_the_textbooks_draw(name, atoms, faces, vertices, sides):
    zone_vertices, zone_faces = brillouin_zone(Structure.from_ase(atoms))
    assert len(zone_faces) == faces, name
    assert len(zone_vertices) == vertices, name
    assert {len(face) for face in zone_faces} == sides, name


def test_every_face_is_a_closed_planar_polygon():
    """Voronoi hands a ridge's vertices back in no order; drawn as given they
    make a star rather than a polygon."""
    structure = Structure.from_ase(bulk("Cu", "fcc", a=3.6))
    vertices, faces = brillouin_zone(structure)
    for face in faces:
        assert len(face) == len(set(face)), "a vertex appears twice in one face"
        points = vertices[face]
        centre = points.mean(axis=0)
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        normal /= np.linalg.norm(normal)
        assert np.abs((points - centre) @ normal).max() < 1e-8, "face is not planar"
        # consecutive vertices are neighbours: the perimeter of a wound polygon
        # is far shorter than that of the same points in a scrambled order
        loop = np.vstack([points, points[:1]])
        perimeter = np.linalg.norm(np.diff(loop, axis=0), axis=1).sum()
        assert perimeter < 1.4 * _hull_perimeter(points)


def _hull_perimeter(points: np.ndarray) -> float:
    """Perimeter of the convex hull of a planar polygon, as a lower bound."""
    centre = points.mean(axis=0)
    spokes = points - centre
    normal = np.cross(spokes[0], spokes[1])
    normal /= np.linalg.norm(normal)
    axis1 = spokes[0] / np.linalg.norm(spokes[0])
    axis2 = np.cross(normal, axis1)
    angles = np.arctan2(spokes @ axis2, spokes @ axis1)
    ordered = points[np.argsort(angles)]
    loop = np.vstack([ordered, ordered[:1]])
    return float(np.linalg.norm(np.diff(loop, axis=0), axis=1).sum())


def test_the_zone_is_centred_on_gamma_and_encloses_it():
    """The first zone is the points closer to the origin than to any other
    reciprocal lattice point, so the origin is inside it and it is symmetric."""
    structure = Structure.from_ase(bulk("Cu", "fcc", a=3.6))
    vertices, _faces = brillouin_zone(structure)
    assert np.linalg.norm(vertices.mean(axis=0)) < 1e-9
    # every vertex has its opposite: the lattice has inversion symmetry
    for vertex in vertices:
        assert np.min(np.linalg.norm(vertices + vertex, axis=1)) < 1e-6


def test_no_zone_vertex_is_nearer_another_lattice_point_than_the_origin():
    """The defining property, checked directly rather than via the construction."""
    structure = Structure.from_ase(bulk("Mg", "hcp", a=3.21, c=5.21))
    vertices, _faces = brillouin_zone(structure)
    reciprocal = reciprocal_cell(structure)
    span = range(-2, 3)
    neighbours = np.array([(i, j, k) for i in span for j in span for k in span
                           if (i, j, k) != (0, 0, 0)], dtype=float) @ reciprocal
    for vertex in vertices:
        assert np.linalg.norm(vertex) <= np.linalg.norm(neighbours - vertex, axis=1).min() + 1e-6


def test_a_structure_with_no_lattice_is_refused():
    from ase.build import molecule

    with pytest.raises(ValueError, match="no usable lattice"):
        brillouin_zone(Structure.from_ase(molecule("H2O")))


# ── the special points ────────────────────────────────────────────────────
def test_gamma_is_called_g_as_crystal_writes_it():
    points = special_points(Structure.from_ase(bulk("Cu", "fcc", a=3.6)))
    assert "G" in points and "\\Gamma" not in points
    assert points["G"] == (0.0, 0.0, 0.0)


def _bounding_planes(structure, reach=2):
    """The perpendicular bisectors that bound the zone, as reciprocal vectors."""
    reciprocal = reciprocal_cell(structure)
    span = range(-reach, reach + 1)
    return np.array([(i, j, k) for i in span for j in span for k in span
                     if (i, j, k) != (0, 0, 0)], dtype=float) @ reciprocal


def _surface_margin(structure, cartesian):
    """How far outside the zone a point is: 0 on the surface, <0 inside."""
    planes = _bounding_planes(structure)
    return float((planes @ cartesian - 0.5 * np.einsum("ij,ij->i", planes, planes)).max())


@pytest.mark.parametrize("name, atoms", [
    ("fcc", bulk("Cu", "fcc", a=3.6)),
    ("bcc", bulk("Fe", "bcc", a=2.87)),
    ("hcp", bulk("Mg", "hcp", a=3.21, c=5.21)),
])
def test_every_labelled_point_lies_exactly_on_the_zone_surface(name, atoms):
    """The invariant that catches a zone drawn from the wrong cell.

    A high-symmetry point other than Γ is on the zone boundary by definition —
    it is a face centre, an edge or a vertex. Label a zone from one lattice and
    draw it from another and these points drift off the surface, which is
    exactly what a conventional-cell file used to do: the picture was a cube
    and W, K and U were outside it entirely.
    """
    structure = Structure.from_ase(atoms)
    points = special_points(structure)
    assert points, name
    for label, fractional in points.items():
        cartesian = to_cartesian(structure, fractional)
        if np.linalg.norm(cartesian) < 1e-9:
            continue  # Γ, at the centre
        assert abs(_surface_margin(structure, cartesian)) < 1e-9, f"{name}: {label}"


def test_an_unclassifiable_lattice_gives_no_labels_rather_than_raising():
    """A zone with no labels is still usable — it can be drawn and clicked. A
    crash on the way to drawing it is not."""
    from ase import Atoms

    odd = Structure.from_ase(Atoms("H", positions=[[0, 0, 0]],
                                   cell=[[3.1, 0.2, 0.1], [0.3, 3.3, 0.2], [0.1, 0.4, 3.7]],
                                   pbc=True))
    assert isinstance(special_points(odd), dict)


def test_the_nearest_special_point_is_found_and_far_ones_are_not():
    structure = Structure.from_ase(bulk("Cu", "fcc", a=3.6))
    x = to_cartesian(structure, special_points(structure)["X"])
    assert nearest_special_point(structure, x) == "X"
    assert nearest_special_point(structure, x * 100) is None


# ── which cell the zone belongs to ────────────────────────────────────────
def _mgo(cubic):
    from ase.build import bulk as _bulk

    return Structure.from_ase(_bulk("MgO", "rocksalt", a=4.21, cubic=cubic))


@pytest.mark.parametrize("cubic", [False, True])
def test_the_primitive_zone_is_the_truncated_octahedron_whatever_the_file_says(cubic):
    """MgO's zone is the same shape whether the file holds the primitive cell
    or the conventional one — the lattice is fcc either way.

    This is the bug the switch exists to make impossible. Handed a conventional
    cubic MgO, the zone used to come out a cube (6 faces) while the labels
    stayed fcc, putting W, K and U outside the picture.
    """
    lattice = zone_lattice(_mgo(cubic), PRIMITIVE)
    vertices, faces = brillouin_zone(lattice)
    assert len(faces) == 14
    assert len(vertices) == 24
    assert {len(face) for face in faces} == {4, 6}
    assert set(special_points(lattice)) == {"G", "K", "L", "U", "W", "X"}


@pytest.mark.parametrize("cubic", [False, True])
def test_the_conventional_zone_is_the_cube_with_its_own_labels(cubic):
    """The other picture, and an honest one: the cubic cell's zone is a cube,
    and it carries the simple-cubic labels rather than fcc's."""
    lattice = zone_lattice(_mgo(cubic), CONVENTIONAL)
    vertices, faces = brillouin_zone(lattice)
    assert len(faces) == 6
    assert len(vertices) == 8
    assert set(special_points(lattice)) == {"G", "M", "R", "X"}


def test_each_setting_labels_the_zone_it_draws():
    """Both settings are self-consistent — that is the point of resolving the
    cell once and being literal about it everywhere after."""
    for setting in (PRIMITIVE, CONVENTIONAL):
        lattice = zone_lattice(_mgo(True), setting)
        for label, fractional in special_points(lattice).items():
            cartesian = to_cartesian(lattice, fractional)
            if np.linalg.norm(cartesian) < 1e-9:
                continue
            assert abs(_surface_margin(lattice, cartesian)) < 1e-9, f"{setting}: {label}"


def test_an_unclassifiable_lattice_keeps_its_own_cell():
    """No standard setting exists, and the Wigner-Seitz cell is still fine."""
    from ase import Atoms

    odd = Structure.from_ase(Atoms("H", positions=[[0, 0, 0]],
                                   cell=[[3.1, 0.2, 0.1], [0.3, 3.3, 0.2], [0.1, 0.4, 3.7]],
                                   pbc=True))
    lattice = zone_lattice(odd, PRIMITIVE)
    assert brillouin_zone(lattice)[0].shape[1] == 3
