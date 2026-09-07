"""Chemically-aware, periodic connectivity: bonds and coordination polyhedra.

Ball-and-stick tools usually bond atoms whose distance is within a covalent-radius
tolerance. That over-bonds ionic crystals — a big cation like Ca (covalent radius
1.76 Å) ends up "bonded" to nearby Ca and Si, so its coordination polyhedron
swallows other cations. Instead we use pymatgen's ``CrystalNN`` near-neighbour
algorithm, which is periodic-aware and finds real cation–anion bonds (Si→4 O,
Ca→6–8 O for alite), and place bonds/polyhedra using the *image* coordinates of
each neighbour so they correctly cross the cell boundary.

pymatgen is imported lazily; the caller falls back to distance-based bonding when
this returns ``None`` (pymatgen missing, non-periodic, or too large to analyse).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from crystalline.core.structure import Structure

# CrystalNN is O(N · neighbours); above this it's too slow for an interactive
# rebuild, so the renderer falls back to the fast distance method.
_MAX_ANALYSIS_ATOMS = 400


@dataclass
class Connectivity:
    """Bonds and coordination polyhedra for one structure.

    ``bonds`` is an ``(M, 2, 3)`` array of cartesian segment endpoints (an
    endpoint may lie outside the cell — a neighbour's periodic image).
    ``polyhedra`` is a list of ``(atomic_number, centre, ligand_positions)`` for
    each cation-centred coordination polyhedron; ``centre`` is the central
    atom's position, which lets a polyhedron be translated onto the centre's
    periodic images (see :func:`replicate_polyhedra`).
    """

    bonds: np.ndarray
    polyhedra: List[Tuple[int, np.ndarray, np.ndarray]] = field(default_factory=list)


def connectivity(structure: Structure, min_vertices: int = 4) -> Optional[Connectivity]:
    """Return :class:`Connectivity` via CrystalNN, or ``None`` if not applicable.

    ``None`` means "fall back to distance-based bonding": non-periodic system,
    too many atoms, or pymatgen/CrystalNN unavailable or failing.
    """
    if not structure.is_periodic or len(structure) == 0:
        return None
    if not np.all(structure.pbc):
        # Slab/polymer: CrystalNN's Voronoi tessellation chokes (hangs) on the
        # formal 500 Å vacuum vector CRYSTAL fills aperiodic directions with. The
        # caller falls back to a bounded distance search, which handles it fine.
        return None
    if len(structure) > _MAX_ANALYSIS_ATOMS:
        return None

    try:
        from pymatgen.analysis.local_env import CrystalNN
        from pymatgen.io.ase import AseAtomsAdaptor
    except Exception:  # noqa: BLE001 - pymatgen missing/broken
        return None

    try:
        pmg = AseAtomsAdaptor().get_structure(structure.to_ase())
        near = CrystalNN()
        electroneg = [site.specie.X for site in pmg]

        segments: List[np.ndarray] = []
        seen: set = set()
        polyhedra: List[Tuple[int, np.ndarray, np.ndarray]] = []
        for i, site in enumerate(pmg):
            infos = near.get_nn_info(pmg, i)
            ligands = np.array([info["site"].coords for info in infos], dtype=float)
            for coord in ligands:
                # de-duplicate the symmetric i–j bond by its rounded endpoints
                key = tuple(sorted((_round(site.coords), _round(coord))))
                if key in seen:
                    continue
                seen.add(key)
                segments.append(np.array([site.coords, coord], dtype=float))
            # a coordination polyhedron only for cation centres (the centre is
            # less electronegative than its ligands) with enough vertices
            if len(ligands) >= min_vertices:
                neighbour_x = [electroneg[info["site_index"]] for info in infos]
                if electroneg[i] is not None and all(x is not None for x in neighbour_x):
                    if electroneg[i] < np.mean(neighbour_x):
                        centre = np.asarray(site.coords, dtype=float)
                        polyhedra.append((int(site.specie.Z), centre, ligands))
    except Exception:  # noqa: BLE001 - CrystalNN can choke on odd cells
        return None

    bonds = np.asarray(segments, dtype=float) if segments else np.empty((0, 2, 3))
    return Connectivity(bonds=bonds, polyhedra=polyhedra)


def replicate_polyhedra(
    polyhedra: List[Tuple[int, np.ndarray, np.ndarray]],
    cell: np.ndarray,
    positions: np.ndarray,
    numbers: np.ndarray,
    tol: float = 1e-3,
) -> List[Tuple[int, np.ndarray]]:
    """Put a polyhedron on **every shown centre**, not just the analysed ones.

    Coordination is analysed on the clean unit cell, but the *displayed* cell is
    usually boundary-completed: each atom sitting on a face/edge/corner is drawn
    again at every cell position it touches. Those image atoms are not in the
    analysed cell, so they used to come out bare — a rutile cell shows 16 Ti but
    drew only the 2 polyhedra of the analysed cell.

    A periodic image of a coordination polyhedron is just that polyhedron
    translated by a lattice vector, so for each analysed centre this finds every
    displayed atom of the same element that sits an **integer** number of lattice
    vectors away, and emits the hull shifted onto it. Returns
    ``[(atomic_number, ligand_positions), …]`` ready for hulling.

    Degenerate (non-invertible) cells can't define images; the polyhedra then
    come back as analysed, one per centre.
    """
    plain = [(z, ligands) for z, _centre, ligands in polyhedra]
    cell = np.asarray(cell, dtype=float)
    if not polyhedra or cell.shape != (3, 3) or abs(np.linalg.det(cell)) < 1e-8:
        return plain

    inverse = np.linalg.inv(cell)  # cartesian -> fractional (cell rows are vectors)
    positions = np.asarray(positions, dtype=float)
    numbers = np.asarray(numbers, dtype=int)

    out: List[Tuple[int, np.ndarray]] = []
    for z, centre, ligands in polyhedra:
        same_element = positions[numbers == int(z)]
        if len(same_element) == 0:
            out.append((z, ligands))
            continue
        shifts = same_element - np.asarray(centre, dtype=float)
        fractional = shifts @ inverse
        is_image = np.all(np.abs(fractional - np.round(fractional)) < tol, axis=1)
        if not is_image.any():
            out.append((z, ligands))  # centre isn't shown (edited away?): keep it as analysed
            continue
        for shift in shifts[is_image]:
            out.append((z, ligands + shift))
    return out


def _round(xyz: np.ndarray) -> tuple:
    return tuple(np.round(xyz, 3))


# ── hydrogen bonds ─────────────────────────────────────────────────────────
# A hydrogen bond is a D–H···A contact: H is covalently bound to an electronegative
# donor D, and points at a nearby electronegative acceptor A. We keep the classic
# N/O/F donor-acceptor set and a geometric test (H···A distance + D–H···A angle),
# which is robust, fast and needs no extra dependencies.
_HBOND_ELEMENTS = frozenset({7, 8, 9})  # N, O, F
_HBOND_MAX_DH = 1.3      # Å: covalent D–H upper bound (N–H≈1.0, O–H≈0.97, F–H≈0.92)
_HBOND_MIN_HA = 1.3      # Å: H···A must be a non-covalent contact
_HBOND_MAX_HA = 2.6      # Å: H···A upper bound for a real hydrogen bond
_HBOND_MIN_ANGLE = 120.0  # degrees: minimum D–H···A angle
_NO_HBONDS = np.empty((0, 2, 3), dtype=float)  # the "no hydrogen bonds" result


def hydrogen_bonds_from_positions(positions, numbers, reference=None) -> np.ndarray:
    """Hydrogen bonds among atoms at ``positions``, treated non-periodically.

    The form the renderer uses for the frame on screen. ``reference`` decides
    *which* contacts count while ``positions`` decides where they are drawn: an
    animation passes the equilibrium geometry so the same set of bonds is drawn
    throughout the cycle instead of appearing and vanishing as atoms swing past
    the distance and angle cut-offs.

    Non-periodic by design: the displayed cell is already boundary-completed, so
    searching its periodic images too would draw each contact twice — once to a
    visible atom and once to an image out in space.
    """
    positions = np.asarray(positions, dtype=float)
    numbers = np.asarray(numbers, dtype=int)
    if len(positions) == 0 or not np.any(numbers == 1):
        return _NO_HBONDS.copy()
    source = positions if reference is None else np.asarray(reference, dtype=float)
    if len(source) != len(positions):  # stale reference: fall back to the live frame
        source = positions
    return hydrogen_bond_segments(positions, hydrogen_bond_pairs(source, numbers))


def hydrogen_bond_pairs(positions, numbers) -> np.ndarray:
    """``(M, 2)`` ``[H, acceptor]`` indices of the hydrogen bonds at ``positions``.

    Split out from the segment builder so a caller that redraws the same geometry
    many times — an animation — can run this scan once and then only gather
    coordinates. It is by far the most expensive part of drawing hydrogen bonds.
    """
    positions = np.asarray(positions, dtype=float)
    numbers = np.asarray(numbers, dtype=int)
    if len(positions) == 0 or not np.any(numbers == 1):
        return np.empty((0, 2), dtype=int)
    contacts = _hydrogen_bond_contacts(numbers, *_kdtree_pairs(positions, _HBOND_MAX_HA))
    return np.asarray([[h, a] for h, a, _ in contacts], dtype=int).reshape(-1, 2)


def hydrogen_bond_segments(positions, pairs) -> np.ndarray:
    """``(M, 2, 3)`` segments for already-known ``pairs`` — a pure array gather."""
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    if len(pairs) == 0:
        return _NO_HBONDS.copy()
    positions = np.asarray(positions, dtype=float)
    return np.stack([positions[pairs[:, 0]], positions[pairs[:, 1]]], axis=1)


def _hydrogen_bond_contacts(numbers, src, dst, vec) -> List[tuple]:
    """``(H index, acceptor index, H→A vector)`` for each qualifying contact."""
    if len(src) == 0:
        return []
    # Group each hydrogen's neighbours: its covalent donor(s) and candidate acceptors.
    dist = np.linalg.norm(vec, axis=1)
    is_h = numbers[src] == 1
    contacts: List[tuple] = []
    for h in np.unique(src[is_h]):
        sel = src == h
        n_idx, n_vec, n_dist = dst[sel], vec[sel], dist[sel]
        electroneg = np.isin(numbers[n_idx], list(_HBOND_ELEMENTS))
        donor = electroneg & (n_dist <= _HBOND_MAX_DH)
        if not donor.any():
            continue  # this H is not on an electronegative donor (e.g. a C–H)
        d_vec = n_vec[donor][np.argmin(n_dist[donor])]  # H→D for the closest donor
        acceptors = electroneg & (n_dist > _HBOND_MIN_HA) & (n_dist <= _HBOND_MAX_HA)
        for a_idx, a_vec in zip(n_idx[acceptors], n_vec[acceptors]):
            if _angle_between(d_vec, a_vec) >= _HBOND_MIN_ANGLE:
                contacts.append((int(h), int(a_idx), a_vec))
    return contacts


def _kdtree_pairs(positions: np.ndarray, cutoff: float):
    """``(src, dst, vec)`` pairs within ``cutoff`` by cartesian distance (no PBC)."""
    from scipy.spatial import cKDTree

    positions = np.asarray(positions, dtype=float)
    pairs = cKDTree(positions).query_pairs(cutoff, output_type="ndarray")
    if len(pairs) == 0:
        return np.empty(0, int), np.empty(0, int), np.empty((0, 3), float)
    i = np.concatenate([pairs[:, 0], pairs[:, 1]])  # both directions, like neighbor_list
    j = np.concatenate([pairs[:, 1], pairs[:, 0]])
    return i, j, positions[j] - positions[i]


def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """Angle (degrees) between vectors ``u`` and ``v`` — here the D–H···A angle."""
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    cos = float(np.dot(u, v) / (nu * nv))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


__all__ = [
    "hydrogen_bond_pairs",
    "hydrogen_bond_segments",
    "Connectivity",
    "connectivity",
    "replicate_polyhedra",
    "hydrogen_bonds_from_positions",
]
