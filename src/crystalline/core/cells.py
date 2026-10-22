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
    structure: Structure, symprec: float, matrix=None
) -> Tuple["object", np.ndarray, np.ndarray]:
    """Build the conventional cell of ``structure`` by centring expansion.

    Returns ``(ase.Atoms, mapping, matrix)`` where ``mapping[i]`` is the index of
    the primitive-cell atom that conventional atom ``i`` is an image of, and
    ``matrix`` is the integer ``M`` with ``conventional_vectors = M ·
    primitive_vectors`` (the identity when nothing was expanded). ``M`` is what
    re-expresses a phonon q in the conventional basis. A non-periodic system, an
    already-conventional cell, or a P lattice all yield the loaded atoms
    unchanged with an identity mapping.

    ``matrix``, when given, is used instead of being found: an edited primitive
    cell is expanded as the cell it was edited from was (see
    :func:`conventional_matrix`), since the edit may have taken away the very
    symmetry the expansion is found by.
    """
    source = structure.to_ase()
    identity = (source, np.arange(len(source)), np.eye(3, dtype=int))

    if not structure.is_periodic or np.allclose(structure.cell, 0.0):
        return identity

    # Lazy import: keep `core` importable without ase's build tools.
    from ase.build import make_supercell

    if matrix is None:
        matrix = _centring_matrix(source, symprec)
    else:
        matrix = np.asarray(matrix, dtype=int)
    if matrix is None or abs(int(round(np.linalg.det(matrix)))) <= 1:
        # A P lattice, a cell that is already conventional, or one that is not
        # the primitive cell of a standard one: it is drawn as it is, since
        # expanding it would multiply the atoms for nothing.
        return identity

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


def _bond_graph(atoms, mult: float = 1.15):
    """Each atom's bonds, as ``(neighbour, lattice shift)`` lists; ``None`` if not found.

    Covalent radii scaled by ``mult``, less the contacts between two cations
    (:func:`~crystalline.core.bonds.cation_contact`) — the bonds the renderer
    draws, so that what is completed at the boundary is what is seen bonded.
    """
    try:
        from ase.neighborlist import natural_cutoffs, neighbor_list

        from crystalline.core.bonds import cation_contact

        i, j, shift = neighbor_list("ijS", atoms, natural_cutoffs(atoms, mult=mult))
        keep = ~cation_contact(atoms.numbers[i], atoms.numbers[j]) if len(i) else np.zeros(0, bool)
    except Exception:  # noqa: BLE001 - never let a bonding hiccup break rendering
        return None
    graph: list[list] = [[] for _ in range(len(atoms))]
    for a, b, s in zip(i[keep], j[keep], shift[keep]):
        graph[a].append((int(b), np.asarray(s, dtype=int)))
    return graph


def _components(graph):
    """The bonded components: ``(indices, images, rank)`` for each.

    ``images`` (``(k, 3)`` integers) are the lattice shifts that put each atom
    next to the atoms it is bonded to, following the bonds out from the first.
    ``rank`` is how many independent directions the component repeats along:
    following the bonds, a bond to an atom already placed either lands on it —
    a ring — or one lattice vector away, which is the component bonded to its
    own copy. A molecule has no such bond (rank 0), a chain's all lie along one
    lattice vector (1), a layer's in a plane (2), a framework's span space (3).
    How far the atoms happen to spread says none of this: an 8-atom cell of
    silicon spans less than a cell, and is a framework all the same.
    """
    n = len(graph)
    image = np.zeros((n, 3), dtype=int)
    visited = [False] * n
    found = []
    for start in range(n):
        if visited[start]:
            continue
        visited[start] = True
        component, stack, loops = [start], [start], []
        while stack:
            u = stack.pop()
            for v, s in graph[u]:
                if not visited[v]:
                    visited[v] = True
                    image[v] = image[u] + s
                    component.append(v)
                    stack.append(v)
                else:
                    loop = image[u] + s - image[v]
                    if loop.any():
                        loops.append(loop)
        rank = int(np.linalg.matrix_rank(np.asarray(loops, dtype=float))) if loops else 0
        found.append((np.asarray(component), image[component].copy(), rank))
    return found


def _wrap_molecules(atoms, mult: float = 1.15):
    """Return ``atoms`` with whole molecules translated into the cell box.

    Molecules are found by covalent-radius bonding under the minimum-image
    convention, each one put together whole (never split across a cell face),
    then shifted so its anchor atom sits in ``[0, 1)`` fractional coordinates.
    Anything that repeats — a chain, a layer, a framework (see
    :func:`_components`) — cannot be made whole, so its atoms are wrapped
    individually.

    Only integer lattice translations are applied, so atom order is preserved
    (and with it the phonon-eigenvector correspondence). Non-periodic systems
    are returned unchanged.
    """
    if not atoms.pbc.any() or np.allclose(np.asarray(atoms.cell), 0.0):
        return atoms
    graph = _bond_graph(atoms, mult)
    if graph is None:
        return atoms
    cell = np.asarray(atoms.cell, dtype=float)
    # Only wrap along periodic axes. A slab/polymer's aperiodic axis carries
    # CRYSTAL's formal 500 Å vector; wrapping a slab centred on z = 0 into [0, 1)
    # there would fling its negative-z atoms to z ≈ 500, splitting it into two
    # sheets half a micron apart.
    periodic = np.asarray(atoms.pbc, dtype=bool)
    frac = np.linalg.solve(cell.T, atoms.get_positions().T).T
    for indices, images, rank in _components(graph):
        if rank:
            frac[np.ix_(indices, periodic)] %= 1.0                     # atom by atom
            continue
        whole = frac[indices] + images
        anchor = np.floor(whole[0])
        anchor[~periodic] = 0.0           # never translate along an aperiodic axis
        frac[indices] = whole - anchor
    out = atoms.copy()
    out.set_positions(frac @ cell)
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


def _centring_matrix(atoms, symprec: float) -> Optional[np.ndarray]:
    """The integer ``M`` with ``conventional_vectors = M · loaded_vectors``.

    From spglib's dataset *of the loaded cell*: its ``transformation_matrix``
    ``P`` takes that basis to the standard conventional one, ``(a_s b_s c_s) =
    (a b c) P⁻¹``, so ``M = (P⁻¹)ᵀ`` for vectors held as rows. Not pymatgen's
    conventional-to-primitive matrix, which relates pymatgen's own two standard
    cells and fits a loaded cell only when CRYSTAL chose the same primitive
    vectors. For β-PVDF's A-centred cell, with a along the chain, it does not,
    and gives a doubled cell at 117°, 117° and 55° instead of the orthorhombic
    box.

    ``None`` when there is no symmetry to go on, or when ``M`` is not an integer
    matrix — the loaded cell is then not the primitive cell of a standard one.
    """
    try:
        import spglib

        dataset = spglib.get_symmetry_dataset(
            (atoms.cell.array, atoms.get_scaled_positions(wrap=False), atoms.numbers),
            symprec=symprec)
        if dataset is None:
            return None
        exact = np.linalg.inv(np.asarray(dataset.transformation_matrix, dtype=float)).T
    except Exception:  # noqa: BLE001 - no symmetry found: the cell is drawn as loaded
        return None
    matrix = np.rint(exact)
    if not np.allclose(exact, matrix, atol=1e-6):
        return None
    return _facing_the_loaded_cell(matrix.astype(int), np.asarray(atoms.cell.array))


# The half turns of a cell about one of its own axes: two vectors reversed.
_HALF_TURNS = ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))


def _facing_the_loaded_cell(matrix: np.ndarray, loaded: np.ndarray) -> np.ndarray:
    """``matrix``, turned half a turn if that points its vectors the loaded cell's way.

    spglib's basis can have two vectors reversed: the same lattice, the same
    setting, but a box drawn on the far side of the origin from the atoms
    CRYSTAL placed. Only the half turns that leave the cell's lengths and angles
    exactly as they are qualify, so a setting's own conventions — an obtuse
    monoclinic β, say — are kept; of those, the one whose vectors lie closest to
    the loaded cell's, each to the one of the same name, is taken.
    """
    vectors = matrix @ loaded
    metric = vectors @ vectors.T
    tolerance = 1e-6 * max(1.0, float(np.abs(metric).max()))
    best, best_score = matrix, None
    for signs in _HALF_TURNS:
        turn = np.diag(signs)
        if not np.allclose(turn @ metric @ turn, metric, atol=tolerance):
            continue
        turned = turn @ vectors
        score = sum(float(np.dot(t, l) / (np.linalg.norm(t) * np.linalg.norm(l)))
                    for t, l in zip(turned, loaded))
        if best_score is None or score > best_score + 1e-9:
            best, best_score = turn @ matrix, score
    return best


def conventional_matrix(structure: Structure, symprec: float = 1e-2) -> np.ndarray:
    """The integer ``M`` that expands ``structure`` into its conventional cell (or ``I``).

    For carrying an edit into the conventional cell: the expansion of the cell
    as read, applied to the cell as edited.
    """
    return _conventional_expansion(structure, symprec)[2]


def primitive_of(conventional: Structure, reference: Structure, mapping,
                 primitive_cell, tolerance: float = 1e-3) -> Optional[Structure]:
    """An edited conventional cell as the primitive cell it was expanded from, if it still is one.

    ``reference`` is the conventional cell as derived, before the edit, atom
    for atom with ``conventional``; ``mapping[i]`` is the primitive atom that
    its atom ``i`` is a copy of. The edited cell is still a centred one when
    every primitive atom's copies kept the centring translations between them
    — the edit made to every copy alike — and then one copy of each is the
    edited primitive cell. ``None`` when an edit broke that: one of two
    centring-related chains turned, say. The edited crystal then has no
    primitive cell smaller than the conventional one.
    """
    mapping = np.asarray(mapping, dtype=int)
    if len(conventional) != len(reference) or len(conventional) != len(mapping):
        return None
    edited = np.asarray(conventional.positions, dtype=float)
    before = np.asarray(reference.positions, dtype=float)
    numbers = np.asarray(conventional.numbers)
    box = np.asarray(conventional.cell, dtype=float)
    box_before = np.asarray(reference.cell, dtype=float)
    first = []
    for atom in range(int(mapping.max()) + 1 if len(mapping) else 0):
        copies = np.nonzero(mapping == atom)[0]
        if not len(copies):
            return None
        lead = copies[0]
        # Compared as fractions of each one's own box, since a lattice change
        # scales the one and not the other; and equal up to a whole box, since
        # each cell puts its atoms into the box on its own, and the same copy
        # can land one box over in one of them and not in the other.
        now = np.linalg.solve(box.T, (edited[copies] - edited[lead]).T).T
        then = np.linalg.solve(box_before.T, (before[copies] - before[lead]).T).T
        drift = now - then
        if not np.allclose(drift, np.rint(drift), atol=tolerance) or np.any(numbers[copies] != numbers[lead]):
            return None
        first.append(lead)
    atoms = conventional.to_ase()[first]
    atoms.set_cell(np.asarray(primitive_cell, dtype=float))
    for key in (_PARENT_KEY, _SHIFT_KEY):
        atoms.arrays.pop(key, None)
    return Structure.from_ase(atoms)


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
    matrix=None,
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
    atoms, mapping, matrix = _conventional_expansion(structure, symprec, matrix)
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


def reshaped_cell(cell, cellpar) -> np.ndarray:
    """``cell`` given the lattice parameters ``cellpar`` (a, b, c in Å; α, β, γ in °), facing as before.

    Built in ase's standard orientation (a along x, b in the xy plane) and then
    turned the way ``cell`` faces, so that changing a length or an angle does not
    also turn the crystal on screen: a cell drawn with a along y keeps a along
    y. Only the shape changes.
    """
    from ase.geometry import cell_to_cellpar, cellpar_to_cell

    cell = np.asarray(cell, dtype=float)
    standard_now = cellpar_to_cell(cell_to_cellpar(cell))
    turn = np.linalg.solve(standard_now, cell)        # cell = standard_now · turn
    return cellpar_to_cell(np.asarray(cellpar, dtype=float)) @ turn


def repeating_tile(tiled: Structure, reps: Tuple[int, int, int], unit_cell,
                   tolerance: float = 1e-3) -> Optional[Structure]:
    """The first tile of ``tiled``, if every tile is that one moved by its lattice shift.

    ``tiled`` is a supercell in the order :func:`tile_supercell` makes one — the
    tiles one after the other, the last axis turning fastest — possibly edited
    since. When the edit was made the same in every tile (a molecule turned
    with all its periodic images, say), the supercell is still ``reps`` copies
    of one cell, and that cell — edit included — is returned. ``None`` when
    the tiles differ: an atom moved or removed in one tile and not the others,
    which only the supercell can hold.
    """
    reps = tuple(int(r) for r in reps)
    count = int(np.prod(reps))
    if count == 1:
        return Structure.from_ase(tiled.to_ase())
    if len(tiled) % count:
        return None
    n = len(tiled) // count
    positions = np.asarray(tiled.positions, dtype=float)
    numbers = np.asarray(tiled.numbers)
    unit_cell = np.asarray(unit_cell, dtype=float)
    first = positions[:n]
    block = 0
    for shift in np.ndindex(*reps):
        rows = slice(block * n, (block + 1) * n)
        if not np.array_equal(numbers[rows], numbers[:n]):
            return None
        if not np.allclose(positions[rows], first + np.asarray(shift) @ unit_cell, atol=tolerance):
            return None
        block += 1
    atoms = tiled.to_ase()[:n]
    atoms.set_cell(unit_cell)
    return Structure.from_ase(atoms)


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
    """Draw the cell's contents with their coordination complete.

    What is drawn depends on what the bonds make of each component (see
    :func:`_components`):

    * a **molecule** is drawn whole at every cell position where any of its
      atoms lies in the box, so a corner CO2 shows up complete at every corner
      it belongs to;
    * a **framework** — a component bonded through every periodic direction,
      silicon or rock salt — is every atom lying in the box, with its copies on
      the faces, edges and corners, and the ligands of each cation among them:
      rock salt's corner Na with all six of its Cl, the corners of the
      octahedra drawn round it. Silicon has no ligands — Si bonds to Si — and is
      the 18 atoms of its cell;
    * a **chain or a layer** in a crystal of more dimensions is drawn as a
      framework is, and then each coordination centre in the box is given all
      the atoms it is bonded to, inside the box or not, and each terminal atom
      (one bond: H, F) on those is added too. A polymer chain runs through the
      cell and one bond past each face, whole groups and all; a brucite layer
      gives every Mg its Mg(OH)6 octahedron.

    A coordination centre has more than one bond, and is bonded to something at
    least as electronegative as itself: the Mg of brucite and the carbons of a
    chain are, the O of brucite is not — completing it would bring in Mg from
    the next cells round. A framework's cations are completed more narrowly,
    with their ligands only (:func:`~crystalline.core.bonds.is_ligand`), and
    only up to :data:`~crystalline.core.bonds.POLYHEDRON_MAX_VERTICES` of them:
    the atoms a coordination polyhedron is drawn between, so every corner of
    one is an atom, and not the twelve oxygens round a perovskite's A site. "In the box" is within ``tol`` of it, so an atom on a
    face counts; along an aperiodic direction everything is in it, and nothing
    is ever shifted along one.

    Returns ``(ase.Atoms, mapping)``: the cell's own atoms first, in order, then
    the copies; ``mapping[i]`` is the atom that drawn atom ``i`` is a copy of
    (for replicating phonon eigenvectors).
    """
    import itertools

    from crystalline.core.bonds import POLYHEDRON_MAX_VERTICES, electronegativity, is_ligand

    atoms = structure.to_ase()
    n = len(atoms)
    identity = (atoms, np.arange(n))
    if not structure.is_periodic or np.allclose(structure.cell, 0.0) or n == 0:
        return identity
    graph = _bond_graph(atoms, mult)
    if graph is None:
        return identity

    from ase import Atoms

    cell = np.asarray(atoms.cell, dtype=float)
    periodic = np.asarray(atoms.get_pbc(), dtype=bool)
    numbers = atoms.get_atomic_numbers()
    frac = np.linalg.solve(cell.T, atoms.get_positions().T).T
    # Only along periodic axes: a slab/polymer's aperiodic direction carries
    # CRYSTAL's formal 500 Å vector, and shifting by it would spawn phantom copies
    # of the whole system 500 Å away in the vacuum.
    shifts = np.asarray(list(itertools.product(*[(-1, 0, 1) if p else (0,) for p in periodic])),
                        dtype=int)

    def in_box(points):                     # (..., 3) fractional -> (...) bool
        inside = (points >= -tol) & (points <= 1.0 + tol)
        return np.all(inside | ~periodic, axis=-1)

    x = np.array([electronegativity(int(z)) for z in numbers])
    centre = [len(graph[a]) > 1 and any(x[b] >= x[a] - 1e-9 for b, _s in graph[a])
              for a in range(n)]
    ligands = [[(b, s) for b, s in graph[a] if is_ligand(x[a], x[b])] for a in range(n)]
    ligands = [found if len(found) <= POLYHEDRON_MAX_VERTICES else [] for found in ligands]
    drawn = set()
    for indices, images, rank in _components(graph):
        if rank == 0:
            # One copy per shift that brings any of the molecule into the box.
            whole = frac[indices] + images
            touching = in_box(whole[None, :, :] + shifts[:, None, :]).any(axis=1)
            for s in shifts[touching]:
                drawn.update((int(a), tuple(int(v) for v in img + s))
                             for a, img in zip(indices, images))
            continue
        hits = in_box(frac[indices][None, :, :] + shifts[:, None, :])          # (S, k)
        core = {(int(indices[k]), tuple(int(v) for v in shifts[s]))
                for s, k in zip(*np.nonzero(hits))}
        drawn |= core
        if rank >= int(periodic.sum()):                    # a framework: its cations' ligands
            shell = {(b, tuple(int(v) for v in np.asarray(o) + t))
                     for a, o in core for b, t in ligands[a]}
        else:                                              # a chain or a layer: its centres' bonds
            shell = {(b, tuple(int(v) for v in np.asarray(o) + t))
                     for a, o in core if centre[a] for b, t in graph[a]}
        drawn |= shell
        drawn |= {(c, tuple(int(v) for v in np.asarray(o) + t))
                  for b, o in shell - core for c, t in graph[b] if len(graph[c]) == 1}

    if not drawn:
        return identity
    # The cell's own atoms first, in their order, then the copies.
    order = sorted(drawn, key=lambda pair: (any(pair[1]), pair[1], pair[0]))
    parents = np.array([a for a, _o in order], dtype=int)
    offsets = np.array([o for _a, o in order], dtype=float)
    result = Atoms(numbers=numbers[parents],
                   positions=atoms.get_positions()[parents] + offsets @ cell,
                   cell=cell, pbc=atoms.get_pbc())
    return result, parents


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
    _label_copies(atoms, structure, mapping)
    return _with_payload(Structure.from_ase(atoms), new_modes, per_atom, mapping)


# Per-atom arrays a completed structure carries, so that the completion can be
# taken back off it after the drawn atoms have been edited: which atom of the
# cell each drawn atom is a copy of (plus one, so that 0 — what ase gives an
# atom added later — means "none: an atom of its own"), and the whole lattice
# shift it was drawn at.
_PARENT_KEY = "crystalline_copy_of"
_SHIFT_KEY = "crystalline_copy_shift"


def _label_copies(atoms, structure: Structure, mapping) -> None:
    """Record on ``atoms`` which atom of ``structure`` each is a copy of, and where."""
    if mapping is None:
        mapping = np.arange(len(atoms))
    mapping = np.asarray(mapping, dtype=int)
    shifts = np.zeros((len(atoms), 3), dtype=int)
    cell = np.asarray(structure.cell, dtype=float)
    if len(atoms) and abs(np.linalg.det(cell)) > 1e-8:
        delta = atoms.get_positions() - np.asarray(structure.positions)[mapping]
        shifts = np.rint(np.linalg.solve(cell.T, delta.T).T).astype(int)
    atoms.set_array(_PARENT_KEY, mapping + 1)
    atoms.set_array(_SHIFT_KEY, shifts)


def own_atoms(structure: Structure) -> Structure:
    """``structure`` labelled as nothing but its own atoms: no copies drawn.

    What a view without boundary completion holds, labelled the way
    :func:`complete_boundary` labels its copies, so that edits to either can be
    carried through a change of the other (:func:`without_boundary_copies`).
    """
    atoms = structure.to_ase()
    _label_copies(atoms, structure, None)
    return Structure.from_ase(atoms)


def without_boundary_copies(structure: Structure):
    """The atoms a completed (and maybe edited) structure stands for, copies taken off.

    Returns ``(structure, parents)``: one atom per atom of the cell that still
    has a copy drawn — in the order of the cell, at the place its own copy is,
    or where any surviving copy says it is once its lattice shift is taken off
    — then any atom added since, which belongs to no cell atom. ``parents[i]``
    is the cell atom that atom ``i`` is, or ``-1`` for an added one. A structure
    that was never labelled (see :func:`own_atoms`) has no copies to take off,
    and comes back as it is with ``parents`` of ``None``.
    """
    atoms = structure.to_ase()
    if _PARENT_KEY not in atoms.arrays or _SHIFT_KEY not in atoms.arrays:
        return Structure.from_ase(atoms), None
    parent = np.asarray(atoms.get_array(_PARENT_KEY), dtype=int) - 1
    shift = np.asarray(atoms.get_array(_SHIFT_KEY), dtype=int)
    cell = np.asarray(atoms.cell, dtype=float)
    positions = atoms.get_positions()
    chosen = {}
    for i, p in enumerate(parent):
        if p < 0:
            continue
        own = not np.any(shift[i])
        if p not in chosen or (own and np.any(shift[chosen[p]])):
            chosen[p] = i
    order = sorted(chosen)
    added = [i for i, p in enumerate(parent) if p < 0]
    keep = [chosen[p] for p in order] + added
    result = atoms[keep]
    result.set_positions(np.array([positions[i] - shift[i] @ cell for i in keep]).reshape(-1, 3))
    for key in (_PARENT_KEY, _SHIFT_KEY):
        del result.arrays[key]
    parents = np.array(order + [-1] * len(added), dtype=int)
    return Structure.from_ase(result), parents


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
    folded = atoms[keep]
    for name in (_PARENT_KEY, _SHIFT_KEY):   # the drawing's labels; not the crystal's
        folded.arrays.pop(name, None)
    return Structure.from_ase(folded)


__all__ = [
    "CellView",
    "to_conventional",
    "conventional_matrix",
    "expand_modes_to_conventional",
    "primitive_of",
    "as_view",
    "tile_supercell",
    "complete_boundary",
    "own_atoms",
    "repeating_tile",
    "reshaped_cell",
    "to_analysis_cell",
    "without_boundary_copies",
]
