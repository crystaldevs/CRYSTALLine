"""The first Brillouin zone, and the special points on it.

The zone is the Wigner–Seitz cell of the reciprocal lattice: the set of points
closer to the origin than to any other reciprocal lattice point. That is exactly
a Voronoi cell, so it is computed as one rather than by the case-by-case
constructions crystallography books give per Bravais lattice — those are a table
of fourteen special cases, and this is one function that is right for all of
them, including the triclinic ones no book draws.

Everything here is in cartesian reciprocal space (1/Angstrom, without the 2π —
the same convention as the fractional coordinates a k-path is written in), and
Qt- and PyVista-free like the rest of ``core``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from crystalline.core.structure import Structure

# The two cells a Brillouin zone can be asked for. They are genuinely different
# pictures of the same crystal: MgO's primitive (fcc) zone is the truncated
# octahedron every band structure is plotted on, while its conventional (cubic)
# zone is a cube — the *folded* zone of a lattice four times smaller, on which
# the fcc labels L, W and K do not exist at all.
PRIMITIVE = "primitive"
CONVENTIONAL = "conventional"

# Reciprocal lattice points out to this many cells in each direction are fed to
# the Voronoi construction. The first zone is bounded by the perpendicular
# bisectors to the *nearest* neighbours, and for any lattice — however oblique —
# those lie within two cells; three is margin.
_NEIGHBOUR_RANGE = 2

# Two zone vertices closer than this (1/Angstrom) are the same vertex. Voronoi
# output repeats them per face, and a hair of floating-point difference would
# otherwise leave a mesh full of near-duplicate points.
_MERGE_TOLERANCE = 1e-6


def zone_lattice(structure: Structure, setting: str = PRIMITIVE) -> Structure:
    """The cell whose zone is drawn — resolve it once, then use it throughout.

    Everything else here is *literal*: it uses the lattice of the structure it
    is handed and nothing else. That is the whole defence against the mistake
    this function exists to prevent — drawing the zone from one cell while
    labelling it from another, which silently produces a cube with fcc points
    floating outside it.

    ``PRIMITIVE`` gives the standard primitive cell that the high-symmetry
    labels are defined on, whatever setting the file was written in; it is the
    zone of the actual Bravais lattice, and the one to compare with a textbook.
    ``CONVENTIONAL`` gives the crystallographic cell's own, smaller zone.

    A structure that cannot be classified is returned untouched — an
    unclassifiable lattice still has a perfectly good Wigner–Seitz cell.
    """
    if not structure.is_periodic:
        return structure
    if setting == CONVENTIONAL:
        from crystalline.core.cells import to_conventional

        try:
            return to_conventional(structure)
        except Exception:  # noqa: BLE001 - the original cell is a usable answer
            return structure
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        pmg = AseAtomsAdaptor.get_structure(structure.to_ase())
        primitive = SpacegroupAnalyzer(pmg, symprec=1e-2).get_primitive_standard_structure()
        return Structure.from_ase(AseAtomsAdaptor.get_atoms(primitive))
    except Exception:  # noqa: BLE001
        return structure


def reciprocal_cell(structure: Structure) -> np.ndarray:
    """The reciprocal lattice vectors as rows, without the 2π.

    With this convention a k-point's fractional coordinates multiply these rows
    directly, which is how every path in this app is expressed.
    """
    cell = np.asarray(structure.cell, dtype=float)
    if cell.shape != (3, 3) or abs(np.linalg.det(cell)) < 1e-12:
        raise ValueError("This structure has no usable lattice.")
    return np.linalg.inv(cell).T


def brillouin_zone(structure: Structure) -> Tuple[np.ndarray, List[List[int]]]:
    """The first Brillouin zone as ``(vertices, faces)``.

    ``faces`` indexes into ``vertices``, each face a closed polygon wound in
    order so it can be drawn directly. Raises ``ValueError`` for a structure
    with no lattice.
    """
    from scipy.spatial import Voronoi

    reciprocal = reciprocal_cell(structure)
    span = range(-_NEIGHBOUR_RANGE, _NEIGHBOUR_RANGE + 1)
    offsets = np.array([(i, j, k) for i in span for j in span for k in span], dtype=float)
    points = offsets @ reciprocal
    origin = int(np.argmin(np.linalg.norm(points, axis=1)))

    voronoi = Voronoi(points)
    # Every ridge with the origin on one side is a face of its Voronoi cell.
    raw_faces = [
        ridge for pair, ridge in zip(voronoi.ridge_points, voronoi.ridge_vertices)
        if origin in pair and -1 not in ridge
    ]
    if not raw_faces:
        raise ValueError("Could not construct the Brillouin zone for this lattice.")

    used = sorted({index for face in raw_faces for index in face})
    vertices, remap = _merged(voronoi.vertices[used], used)
    faces = []
    for face in raw_faces:
        polygon = _ordered([remap[index] for index in face], vertices)
        if len(polygon) >= 3:
            faces.append(polygon)
    return vertices, faces


def _merged(points: np.ndarray, used: List[int]) -> Tuple[np.ndarray, Dict[int, int]]:
    """Collapse coincident vertices, returning the survivors and an index map."""
    unique: List[np.ndarray] = []
    remap: Dict[int, int] = {}
    for position, original in zip(points, used):
        for index, kept in enumerate(unique):
            if np.linalg.norm(position - kept) < _MERGE_TOLERANCE:
                remap[original] = index
                break
        else:
            remap[original] = len(unique)
            unique.append(position)
    return np.asarray(unique, dtype=float), remap


def _ordered(indices: List[int], vertices: np.ndarray) -> List[int]:
    """Wind a face's vertices into polygon order.

    Voronoi hands back the vertices of a ridge as a set, in no particular
    order; drawn as given they make a star rather than a polygon. They are
    coplanar, so sorting them by angle about their own centroid, in the plane's
    own basis, is enough.
    """
    indices = list(dict.fromkeys(indices))
    if len(indices) < 3:
        return indices
    points = vertices[indices]
    centre = points.mean(axis=0)
    spokes = points - centre
    normal = np.cross(spokes[0], spokes[1])
    if np.linalg.norm(normal) < 1e-12:  # degenerate: leave it alone
        return indices
    normal /= np.linalg.norm(normal)
    axis1 = spokes[0] / np.linalg.norm(spokes[0])
    axis2 = np.cross(normal, axis1)
    angles = np.arctan2(spokes @ axis2, spokes @ axis1)
    return [indices[i] for i in np.argsort(angles)]


def special_points(structure: Structure) -> Dict[str, Tuple[float, float, float]]:
    """The lattice's labelled high-symmetry points, in fractional coordinates.

    ``G`` for Γ, as CRYSTAL writes it. Empty when the lattice cannot be
    classified — the caller then has a zone to click on but no labels for it,
    which is still usable.

    Literal about the cell it is given: the points come back in *this*
    structure's own reciprocal basis, so they always land on the zone
    :func:`brillouin_zone` draws for the same structure. Ask for the labels of
    a cell and you are drawing another one and they will not match — see
    :func:`zone_lattice`, which is how a caller chooses.
    """
    try:
        points = structure.to_ase().cell.bandpath().special_points
        return {
            _tidy(label): tuple(float(v) for v in point)
            for label, point in points.items()
        }
    except Exception:  # noqa: BLE001 - no labels is a usable state, a crash is not
        return {}


def display_label(label: str) -> str:
    """``G`` -> ``Γ``, for anything a person reads.

    The stored label stays ``G`` because that is what goes into a deck: a
    CRYSTAL title record is echoed back through Fortran I/O, and there is
    nothing to gain by sending two bytes of UTF-8 through it. On screen there
    is no such constraint, and the point is called Gamma.
    """
    return "Γ" if label == "G" else str(label)


def _tidy(label: str) -> str:
    name = str(label).replace("\\", "").strip()
    return "G" if name.lower() == "gamma" else name


def to_cartesian(structure: Structure, fractional) -> np.ndarray:
    """Fractional reciprocal coordinates -> cartesian, for drawing."""
    return np.asarray(fractional, dtype=float) @ reciprocal_cell(structure)


def nearest_special_point(
    structure: Structure, cartesian, tolerance: Optional[float] = None
) -> Optional[str]:
    """The label of the special point nearest ``cartesian``, if one is close.

    ``tolerance`` defaults to a tenth of the zone's own size, so "close" scales
    with the lattice instead of being an absolute distance that means something
    different for every crystal.
    """
    points = special_points(structure)
    if not points:
        return None
    target = np.asarray(cartesian, dtype=float)
    reciprocal = reciprocal_cell(structure)
    if tolerance is None:
        tolerance = 0.1 * float(np.linalg.norm(reciprocal, axis=1).max())
    best, best_distance = None, np.inf
    for label, fractional in points.items():
        distance = float(np.linalg.norm(np.asarray(fractional) @ reciprocal - target))
        if distance < best_distance:
            best, best_distance = label, distance
    return best if best_distance <= tolerance else None


__all__ = [
    "CONVENTIONAL",
    "PRIMITIVE",
    "brillouin_zone",
    "display_label",
    "nearest_special_point",
    "reciprocal_cell",
    "special_points",
    "to_cartesian",
    "zone_lattice",
]
