"""Primitive ↔ crystallographic (conventional) cell views of a structure.

Building the crystallographic cell the way a crystallographer expects to see it
is two steps: get the right *lattice* (with the right number of atoms), then
place those atoms into the drawn cell box sensibly.

**Lattice — the centring expansion.** For a centred lattice (F, I, C, R…) the
primitive cell CRYSTAL works in holds fewer atoms than the conventional cell
(an FCC crystal's 2-atom primitive cell vs. its 8-atom cubic cell). We expand
with the integer matrix ``M`` where ``conventional_vectors = M · primitive_vectors``
(``det M`` = the centring multiplicity: 1 for P, 2 for I/C, 3 for R, 4 for F),
applied via ``ase.build.make_supercell``. A P lattice (dry ice's Pa-3, say) has
``M = I`` and is left untouched. Not pymatgen's
``get_conventional_standard_structure``: that re-orders atoms (destroying the
ordering phonon eigenvectors rely on) and re-wraps them. The tiling instead
keeps every conventional atom mapped to the primitive atom it came from, which
is what lets phonon modes animate on the conventional cell too (see
:func:`expand_modes_to_conventional`).

**Placement — molecule-aware wrapping.** CRYSTAL centres its coordinates on the
origin (fractional roughly ``[-0.5, 0.5]``), but the viewport draws the cell box
as ``[0, 1)``, so half the atoms fall outside it and any molecule straddling a
face renders as broken fragments (dry ice's four CO₂ show up as one molecule
plus three bare carbons). :func:`_wrap_molecules` reconstructs whole molecules
across the periodic boundary and translates each into the box. Wrapping is only
ever integer lattice translations, so atom order — and thus the phonon
correspondence — is preserved.

**Phonons — the phase that comes with the copy.** Every operation here copies
atoms into other cells, and a phonon mode has to be copied with them. At Gamma
that is a plain replication: all cells move together. Away from Gamma (the modes
a SCELPHONO run reports at the q commensurate with its supercell) cell ``n``
lags the reference by ``2*pi*q·n``, so each image atom's eigenvector is
multiplied by that phase — see :func:`_lattice_offsets`, which recovers ``n``
from the *drawn* positions, and :func:`_replicate_modes`. Tiling is what makes
such a mode visible at all: on one cell there is nothing for the wave to
modulate.

pymatgen/ase are imported lazily so ``core`` stays importable (and unit-testable)
without them, matching ``crystalio.loader``. No function mutates its argument.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple

import numpy as np

from crystalline.core.phonons import PhononModes, phase_factors
from crystalline.core.structure import Structure

# ase array key used to carry each atom's source index through the supercell
# expansion, so we can map conventional atoms back to their primitive parent.
_SOURCE_INDEX_KEY = "__crystalline_source_index__"


class CellView(str, Enum):
    """Which cell the viewport draws.

    ``CRYSTALLOGRAPHIC`` is the conventional cell (the default); ``PRIMITIVE``
    is the cell as loaded from the CRYSTAL file.
    """

    CRYSTALLOGRAPHIC = "crystallographic"
    PRIMITIVE = "primitive"


def _conventional_expansion(
    structure: Structure, symprec: float
) -> Tuple["object", np.ndarray, np.ndarray]:
    """Build the conventional cell of ``structure`` by centring expansion.

    Returns ``(ase.Atoms, mapping, matrix)`` where ``mapping[i]`` is the index of
    the primitive-cell atom that conventional atom ``i`` is an image of, and
    ``matrix`` is the integer ``M`` with ``conventional_vectors = M ·
    primitive_vectors`` (the identity when nothing was expanded). ``M`` is what
    re-expresses a phonon q in the conventional basis. A non-periodic system, an
    already-conventional cell, or a P lattice all yield the loaded atoms
    unchanged with an identity mapping.
    """
    source = structure.to_ase()
    identity = (source, np.arange(len(source)), np.eye(3, dtype=int))

    if not structure.is_periodic or np.allclose(structure.cell, 0.0):
        return identity

    # Lazy imports: keep `core` importable without pymatgen/ase build tools.
    from ase.build import make_supercell
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    pmg = AseAtomsAdaptor().get_structure(source)
    sga = SpacegroupAnalyzer(pmg, symprec=symprec)

    # Only expand when the loaded cell really is the primitive one; if it is
    # already the conventional cell (or otherwise not primitive), it is its own
    # conventional cell — expanding again would wrongly multiply the atoms.
    primitive = sga.find_primitive()
    if primitive is None or len(source) != len(primitive):
        return identity

    # M maps primitive → conventional lattice vectors, in the lattice basis
    # (orientation-independent, so it applies to the loaded cell as-is).
    conv_to_prim = np.asarray(sga.get_conventional_to_primitive_transformation_matrix(), dtype=float)
    matrix = np.rint(np.linalg.inv(conv_to_prim)).astype(int)
    if abs(int(round(np.linalg.det(matrix)))) <= 1:
        return identity  # P lattice: conventional == primitive

    tagged = source.copy()
    tagged.set_array(_SOURCE_INDEX_KEY, np.arange(len(source)))
    conventional = make_supercell(tagged, matrix)

    if _SOURCE_INDEX_KEY not in conventional.arrays:
        # make_supercell didn't carry the tag through: fall back rather than
        # return a mapping we can't trust.
        return identity
    mapping = np.asarray(conventional.get_array(_SOURCE_INDEX_KEY), dtype=int)
    del conventional.arrays[_SOURCE_INDEX_KEY]
    return conventional, mapping, matrix


def _wrap_molecules(atoms, mult: float = 1.15):
    """Return ``atoms`` with whole molecules translated into the cell box.

    Molecules are found by covalent-radius bonding under the minimum-image
    convention, each connected component is reassembled so it is contiguous
    (never split across a cell face), then shifted so its anchor atom sits in
    ``[0, 1)`` fractional coordinates. A component that percolates the lattice
    (an extended framework, not a discrete molecule) can't be made whole, so its
    atoms are simply wrapped individually — the standard depiction for those.

    Only integer lattice translations are applied, so atom order is preserved
    (and with it the phonon-eigenvector correspondence). Non-periodic systems
    are returned unchanged.
    """
    import numpy as np

    if not atoms.pbc.any() or np.allclose(np.asarray(atoms.cell), 0.0):
        return atoms

    try:
        from ase.neighborlist import natural_cutoffs, neighbor_list

        i, j, offset = neighbor_list("ijD", atoms, natural_cutoffs(atoms, mult=mult))
    except Exception:  # noqa: BLE001 - never let a bonding hiccup break rendering
        return atoms

    n = len(atoms)
    cell = np.asarray(atoms.cell, dtype=float)
    # Only wrap along periodic axes. A slab/polymer's aperiodic axis carries
    # CRYSTAL's formal 500 Å vector; wrapping a slab centred on z = 0 into [0, 1)
    # there would fling its negative-z atoms to z ≈ 500, splitting it into two
    # sheets half a micron apart.
    periodic = np.asarray(atoms.pbc, dtype=bool)
    neighbours: list[list] = [[] for _ in range(n)]
    for a, b, disp in zip(i, j, offset):
        neighbours[a].append((b, disp))

    positions = atoms.get_positions()
    unwrapped = positions.copy()
    visited = [False] * n
    for start in range(n):
        if visited[start]:
            continue
        visited[start] = True
        component = [start]
        stack = [start]
        while stack:  # depth-first reassembly via minimum-image displacements
            u = stack.pop()
            for v, disp in neighbours[u]:
                if not visited[v]:
                    visited[v] = True
                    unwrapped[v] = unwrapped[u] + disp
                    component.append(v)
                    stack.append(v)
        comp = np.asarray(component)
        frac = np.linalg.solve(cell.T, unwrapped[comp].T).T
        if np.any(frac.max(0) - frac.min(0) >= 1.0):
            # Extended framework: reassembly is meaningless — wrap each atom, but
            # only along periodic axes (aperiodic coordinates stay put).
            wrapped = frac.copy()
            wrapped[:, periodic] = frac[:, periodic] % 1.0
            unwrapped[comp] = wrapped @ cell
        else:
            shift = np.floor(frac[0])
            shift[~periodic] = 0.0  # never translate along an aperiodic axis
            unwrapped[comp] -= shift @ cell  # anchor into [0, 1)

    out = atoms.copy()
    out.set_positions(unwrapped)
    return out


def _lattice_offsets(positions, source_positions, mapping, cell) -> Optional[np.ndarray]:
    """Which cell each image atom was drawn into, relative to its parent's.

    Returns an (N, 3) array of integer lattice translations ``n`` — image atom
    ``i`` sits at ``parent + n_i · cell`` — or ``None`` when there is no lattice
    to count in (a molecule, a degenerate cell) or the positions can't be
    reconciled with one.

    This is what carries a non-Gamma mode onto image atoms: the phase an atom
    lags its parent by is ``2*pi*q·n``, so the answer has to be read off the
    *drawn* positions, after every wrap and reassembly the operation performed —
    not from the order the images were generated in.
    """
    cell = np.asarray(cell, dtype=float)
    if cell.shape != (3, 3) or abs(np.linalg.det(cell)) < 1e-8:
        return None
    delta = np.asarray(positions, dtype=float) - np.asarray(source_positions, dtype=float)[mapping]
    frac = np.linalg.solve(cell.T, delta.T).T
    offsets = np.rint(frac)
    if np.any(np.abs(frac - offsets) > 1e-3):
        # Not a whole number of lattice vectors: the image isn't a pure
        # translation of its parent, so no phase can be assigned. Better an
        # in-phase replica (what Gamma modes always got) than a wrong phase.
        return None
    return offsets


def _replicate_modes(modes, mapping, offsets) -> PhononModes:
    """Copy each mode's eigenvector onto image atoms, phase included.

    ``mapping[i]`` is the parent of image atom ``i`` and ``offsets[i]`` the
    lattice translation between them. A Gamma mode is replicated verbatim (every
    cell moves in phase); away from Gamma each image is multiplied by its
    ``exp(2*pi*i q·n)``, which is what turns a repeated cell into a wave.

    The phase is also *recorded* per atom, added to whatever the parent already
    carried, so a mode that reaches the screen through several operations
    (conventional cell, then tiling, then boundary completion) knows the total
    Bloch phase of the cell each atom ended up in — see
    :attr:`~crystalline.core.phonons.PhononMode.cell_phase`.
    """
    replicated = []
    for mode in modes:
        eigenvector = mode.eigenvector[mapping]
        inherited = None if mode.cell_phase is None else mode.cell_phase[mapping]
        factors = phase_factors(mode.qpoint, offsets)
        if factors is None:
            phase = inherited
        else:
            eigenvector = eigenvector * factors[:, None]
            added = 2.0 * np.pi * (np.asarray(offsets, dtype=float) @ mode.qpoint)
            phase = added if inherited is None else inherited + added
        replicated.append(mode.with_eigenvector(eigenvector, cell_phase=phase))
    return PhononModes(replicated)


# Default for the optional ``per_atom`` argument of the cell operations. A
# distinct object rather than None, so that passing ``per_atom=None`` — a file
# that simply has no ADPs — still returns the three-value form the caller asked
# for, instead of a two-value one it would fail to unpack.
_NO_PAYLOAD = object()


def to_conventional(structure: Structure, symprec: float = 1e-2) -> Structure:
    """Return the crystallographic (conventional) cell of ``structure``.

    The lattice is expanded for centred cells and the atoms are wrapped
    molecule-wise into the cell box. Non-periodic systems are returned unchanged
    as a fresh copy.

    Parameters
    ----------
    structure:
        The structure to expand (typically the primitive cell as loaded).
    symprec:
        Symmetry tolerance in Angstrom handed to the space-group analyser.
    """
    atoms, _mapping, _matrix = _conventional_expansion(structure, symprec)
    return Structure.from_ase(_wrap_molecules(atoms))


def expand_modes_to_conventional(
    structure: Structure,
    modes: PhononModes,
    symprec: float = 1e-2,
    per_atom=_NO_PAYLOAD,
) -> Tuple[Structure, PhononModes]:
    """Return the conventional cell together with phonon modes defined on it.

    Each conventional atom is an image of a primitive atom. At Gamma every image
    of a primitive cell moves in phase, so a conventional-cell eigenvector is
    just the primitive eigenvector replicated onto each atom's parent; away from
    Gamma each image also picks up the ``exp(2*pi*i q·n)`` of the primitive cell
    it lives in. When no expansion happens (P lattice, molecule), the modes come
    back unchanged. The returned structure is molecule-wrapped to match
    :func:`to_conventional`; wrapping is pure lattice translation, which the
    phases are read off *after* so they follow the atoms into the drawn cell.

    The q-point is handed on in the conventional cell's own reciprocal basis
    (``q_conv = M·q_prim`` for ``conventional = M · primitive``), so a later
    tiling of that cell can go on counting translations in the cell it sees.

    ``per_atom`` (ADP tensors, say) is replicated the same way — see
    :func:`tile_supercell` for what passing it does to the return value.
    """
    atoms, mapping, matrix = _conventional_expansion(structure, symprec)
    wrapped = _wrap_molecules(atoms)
    offsets = _lattice_offsets(
        wrapped.get_positions(), structure.positions, mapping, structure.cell
    )
    expanded = _replicate_modes(modes, mapping, offsets)
    expanded = PhononModes(
        [
            m.with_qpoint(None if m.qpoint is None else matrix @ m.qpoint)
            for m in expanded
        ]
    )
    return _with_payload(Structure.from_ase(wrapped), expanded, per_atom, mapping)


def as_view(structure: Structure, view: CellView) -> Structure:
    """Return ``structure`` rendered in the requested :class:`CellView`.

    Always a fresh copy, so editing the returned structure never mutates the
    source the view was derived from.
    """
    if view is CellView.PRIMITIVE:
        return Structure.from_ase(structure.to_ase())
    return to_conventional(structure)


def tile_supercell(
    structure: Structure,
    reps: Tuple[int, int, int],
    modes: Optional[PhononModes] = None,
    per_atom=_NO_PAYLOAD,
) -> Tuple[Structure, Optional[PhononModes]]:
    """Tile ``structure`` ``reps`` = ``(na, nb, nc)`` times along its lattice vectors.

    Applied on top of whichever cell view is active, so the supercell inherits
    its molecule-wrapped atoms (each image cell stays whole). Non-periodic axes
    are never tiled. If phonon ``modes`` are given they are replicated onto the
    image atoms and returned alongside the supercell; the atom order is
    preserved so each image atom's eigenvector matches its parent. At Gamma
    every image cell moves in phase; away from it, image cell ``n`` lags by
    ``2*pi*q·n``, which is the whole point of tiling such a mode — a wave needs
    more than one cell to be a wave. :func:`~crystalline.core.phonons.commensurate_repeats`
    gives the smallest tiling that shows one whole period.

    ``per_atom`` is any array indexed by atom along its first axis — ADP tensors,
    say — replicated the same way. *Passing* it adds a third element to the
    returned tuple (as ``numpy``'s ``return_index`` does) rather than changing
    the shape of the common two-value call. Passing ``None`` counts as passing:
    a caller with nothing to replicate still gets three values, the third being
    ``None``.
    """
    atoms = structure.to_ase()
    pbc = atoms.get_pbc()
    reps = tuple(int(r) if pbc[k] else 1 for k, r in enumerate(reps))
    if reps == (1, 1, 1) or not pbc.any():
        return _with_payload(Structure.from_ase(atoms), modes, per_atom, None)

    tagged = atoms.copy()
    tagged.set_array(_SOURCE_INDEX_KEY, np.arange(len(atoms)))
    supercell = tagged.repeat(reps)

    mapping = None
    if _SOURCE_INDEX_KEY in supercell.arrays:
        mapping = np.asarray(supercell.get_array(_SOURCE_INDEX_KEY), dtype=int)
        del supercell.arrays[_SOURCE_INDEX_KEY]

    tiled_modes = modes
    if modes is not None and mapping is not None:
        offsets = _lattice_offsets(
            supercell.get_positions(), atoms.get_positions(), mapping, structure.cell
        )
        tiled_modes = _replicate_modes(modes, mapping, offsets)
    elif modes is not None:
        # Couldn't recover the mapping: drop the modes rather than hand back
        # eigenvectors whose atom count no longer matches the supercell.
        tiled_modes = None
    return _with_payload(Structure.from_ase(supercell), tiled_modes, per_atom, mapping)


def _with_payload(structure, modes, per_atom, mapping):
    """Assemble the return of a cell operation, with ``per_atom`` if asked for.

    Whether the third element appears is decided by *passing* the argument, not
    by its value: ``per_atom=None`` is a legitimate "this file has no ADPs", and
    a caller that asked for three values must get three either way. Only the
    :data:`_NO_PAYLOAD` sentinel — the default, meaning the argument was never
    given — returns the plain two-value form.

    ``mapping[i]`` is the source atom image atom ``i`` came from; ``None`` means
    the operation was the identity (or its mapping couldn't be recovered), in
    which case a payload that already matches the atom count passes through and
    anything else is dropped rather than misaligned.

    A payload too short for the mapping is dropped as well. That is not a
    hypothetical: a conventional-cell expansion can add atoms, and indexing the
    pre-expansion ADPs by a post-expansion mapping would either raise or, worse,
    hand back tensors belonging to the wrong atoms.
    """
    if per_atom is _NO_PAYLOAD:
        return structure, modes
    if per_atom is None:
        return structure, modes, None
    payload = np.asarray(per_atom)
    if mapping is None:
        payload = payload if len(payload) == len(structure) else None
    elif len(mapping) and int(np.max(mapping)) >= len(payload):
        payload = None
    else:
        payload = payload[mapping]
    return structure, modes, payload


def _boundary_completion(structure: Structure, tol: float, mult: float = 1.15):
    """Add periodic images of every molecule/atom that pokes into the cell.

    Returns ``(ase.Atoms, mapping)`` where the atoms are the cell's own atoms
    PLUS the atoms of any periodic image whose molecule has at least one atom
    within ``[-tol, 1+tol]`` fractional (i.e. inside or just touching the box).
    Whole molecules are kept together, so a corner CO2 shows up complete at every
    corner it belongs to. ``mapping[i]`` is the index of the original atom that
    image atom ``i`` is a copy of (for replicating phonon eigenvectors).
    """
    import itertools

    atoms = structure.to_ase()
    n = len(atoms)
    identity = (atoms, np.arange(n))
    if not structure.is_periodic or np.allclose(structure.cell, 0.0) or n == 0:
        return identity

    try:
        from ase import Atoms
        from ase.neighborlist import natural_cutoffs, neighbor_list

        i, j, disp = neighbor_list("ijD", atoms, natural_cutoffs(atoms, mult=mult))
    except Exception:  # noqa: BLE001 - never let a bonding hiccup break rendering
        return identity

    cell = np.asarray(atoms.cell, dtype=float)
    positions = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()
    neighbours: list[list] = [[] for _ in range(n)]
    for a, b, d in zip(i, j, disp):
        neighbours[a].append((b, d))

    # Reassemble contiguous molecules (whole across the periodic boundary).
    unwrapped = positions.copy()
    visited = [False] * n
    components: list = []
    for start in range(n):
        if visited[start]:
            continue
        visited[start] = True
        component = [start]
        stack = [start]
        while stack:
            u = stack.pop()
            for v, d in neighbours[u]:
                if not visited[v]:
                    visited[v] = True
                    unwrapped[v] = unwrapped[u] + d
                    component.append(v)
                    stack.append(v)
        components.append(np.asarray(component))

    # A component that percolates the lattice (extended framework) is imaged atom
    # by atom instead of as one blob. Crucially, such atoms are imaged from their
    # ORIGINAL (in-cell) positions, not the DFS-``unwrapped`` ones: unwrapping an
    # extended network drifts by lattice vectors around its loops, which would
    # place the drawn atoms off their true sites — visibly detaching the bonds
    # and coordination polyhedra (analysed on the clean cell) from the atoms.
    base_pos = unwrapped.copy()
    clusters: list = []
    for comp in components:
        frac = np.linalg.solve(cell.T, unwrapped[comp].T).T
        if np.any(frac.max(0) - frac.min(0) >= 1.0):
            base_pos[comp] = positions[comp]  # canonical sites, no unwrap drift
            clusters.extend(np.array([idx]) for idx in comp)
        else:
            clusters.append(comp)

    out_pos: list = []
    out_num: list = []
    out_parent: list = []
    seen: set = set()
    # Only image along periodic axes: a slab/polymer's aperiodic direction carries
    # CRYSTAL's formal 500 Å vector, and shifting by it would spawn phantom copies
    # of the whole system 500 Å away in the vacuum.
    per_axis = [(-1, 0, 1) if p else (0,) for p in atoms.get_pbc()]
    shifts = list(itertools.product(*per_axis))
    for comp in clusters:
        cluster_pos = base_pos[comp]
        for shift in shifts:
            image = cluster_pos + np.asarray(shift, dtype=float) @ cell
            frac = np.linalg.solve(cell.T, image.T).T
            touches = np.any(np.all((frac >= -tol) & (frac <= 1.0 + tol), axis=1))
            if not touches:
                continue
            for k, idx in enumerate(comp):
                key = tuple(np.round(image[k], 3))
                if key in seen:
                    continue
                seen.add(key)
                out_pos.append(image[k])
                out_num.append(int(numbers[idx]))
                out_parent.append(int(idx))

    if not out_pos:
        return identity
    result = Atoms(
        numbers=out_num, positions=np.asarray(out_pos), cell=cell, pbc=atoms.get_pbc()
    )
    return result, np.asarray(out_parent, dtype=int)


def complete_boundary(
    structure: Structure,
    modes: Optional[PhononModes] = None,
    tol: float = 2e-2,
    per_atom=_NO_PAYLOAD,
) -> Tuple[Structure, Optional[PhononModes]]:
    """Return ``structure`` with boundary molecules/atoms completed by images.

    Every molecule that partially belongs to the unit cell is drawn whole, at
    each cell position it touches (all corners/edges/faces). Phonon ``modes`` are
    replicated onto the image atoms — in phase at Gamma, and with the phase of
    the cell the image was placed in away from it, so a boundary atom vibrates
    with the neighbours it is drawn among rather than with its parent across the
    cell. ``per_atom`` is replicated too — see :func:`tile_supercell` for what
    that argument does to the return value.
    """
    atoms, mapping = _boundary_completion(structure, tol)
    new_modes = modes
    if modes is not None:
        offsets = _lattice_offsets(
            atoms.get_positions(), structure.positions, mapping, structure.cell
        )
        new_modes = _replicate_modes(modes, mapping, offsets)
    return _with_payload(Structure.from_ase(atoms), new_modes, per_atom, mapping)


def to_analysis_cell(structure: Structure, unit_cell) -> Structure:
    """Fold a *displayed* structure back into a single unit cell for symmetry work.

    The shown structure may be a supercell and/or boundary-completed, so it can't
    be handed to a symmetry finder directly: a supercell's box has the wrong shape
    (spglib would report the box's point group, not the crystal's), and boundary
    images sit on top of their originals once wrapped. Re-imposing ``unit_cell``
    and wrapping every atom into it folds the tiles and images back onto their
    originals; coincident copies are then dropped, leaving one clean cell. Any
    edits made to the displayed atoms are carried along by the fold.

    A slab's (or polymer's) dimensionality is preserved: only the periodic axes
    are folded, so the aperiodic 500 Å direction is never wrapped (which would fling
    atoms across the vacuum) and the system does not silently become 3D.

    Non-periodic structures (and degenerate cells) are returned untouched. Always
    a fresh copy — never mutates its argument.
    """
    if not structure.is_periodic:
        return Structure.from_ase(structure.to_ase())
    cell = np.asarray(unit_cell, dtype=float)
    if cell.shape != (3, 3) or abs(np.linalg.det(cell)) < 1e-8:
        return Structure.from_ase(structure.to_ase())

    atoms = structure.to_ase()
    pbc = [bool(p) for p in atoms.get_pbc()]  # keep the original dimensionality
    atoms.set_cell(cell)
    frac = atoms.get_scaled_positions(wrap=True)  # wraps periodic axes into [0, 1) only
    cart = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()
    # Tiles and boundary images fold to the *same* fractional site along the periodic
    # axes (they differ by whole lattice vectors), so a coarse grid key on those axes
    # dedupes them — the ``% 1.0`` also folds an atom drawn at both the 0 and 1
    # boundary. An aperiodic axis has no images, so it keys on the raw position to
    # keep distinct layers (e.g. the sheets of a slab) apart.
    seen: set = set()
    keep = []
    for i in range(len(atoms)):
        key = [int(numbers[i])]
        for axis in range(3):
            if pbc[axis]:
                key.append(round(float(frac[i][axis]), 3) % 1.0)
            else:
                key.append(round(float(cart[i][axis]), 2))
        key = tuple(key)
        if key not in seen:
            seen.add(key)
            keep.append(i)
    return Structure.from_ase(atoms[keep])


__all__ = [
    "CellView",
    "to_conventional",
    "expand_modes_to_conventional",
    "as_view",
    "tile_supercell",
    "complete_boundary",
    "to_analysis_cell",
]
