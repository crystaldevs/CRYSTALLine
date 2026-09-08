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
    brillouin_zone,
    nearest_special_point,
    reciprocal_cell,
    special_points,
    to_cartesian,
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


def test_special_points_land_inside_the_zone():
    structure = Structure.from_ase(bulk("Cu", "fcc", a=3.6))
    vertices, _faces = brillouin_zone(structure)
    radius = np.linalg.norm(vertices, axis=1).max()
    for label, fractional in special_points(structure).items():
        assert np.linalg.norm(to_cartesian(structure, fractional)) <= radius + 1e-6, label


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
