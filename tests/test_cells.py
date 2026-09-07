"""Tests for the primitive ↔ crystallographic cell views (no display needed).

These exercise the real symmetry analysis, so they need pymatgen; they skip if
it (or ase's structure builders) is unavailable.
"""

import numpy as np
import pytest

from crystalline.core.cells import (
    CellView,
    as_view,
    complete_boundary,
    expand_modes_to_conventional,
    tile_supercell,
    to_analysis_cell,
    to_conventional,
)
from crystalline.core.phonons import PhononMode, PhononModes
from crystalline.core.structure import Structure

pytest.importorskip("pymatgen")
from ase import Atoms  # noqa: E402  (after importorskip)
from ase.build import bulk  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402


def _dry_ice() -> Structure:
    """Solid CO2 (Pa-3, space group 205): 4 CO2 molecules in a cubic P cell.

    Coordinates as CRYSTAL prints them — centred on the origin (fractional
    roughly [-0.4, 0.5]), so three of the four CO2 straddle the [0, 1) cell box
    and render as broken fragments until molecule-aware wrapping reassembles
    them. This is the real regression case.
    """
    a = 5.67747679
    d = 0.6752559463681  # C–O projected onto each cubic axis
    p = 2.838738395  # a / 2
    positions = np.array(
        [
            [0, 0, 0], [p, 0, p], [p, p, 0], [0, p, p],           # 4 carbons
            [d, d, d], [p - d, -d, -(p - d)], [-(p - d), p - d, -d], [-d, -(p - d), p - d],
            [-d, -d, -d], [-(p - d), d, p - d], [p - d, -(p - d), d], [d, p - d, -(p - d)],
        ]
    )
    atoms = Atoms(["C"] * 4 + ["O"] * 8, positions=positions, cell=np.diag([a, a, a]), pbc=True)
    return Structure.from_ase(atoms)


def _o_neighbours_per_c(structure: Structure) -> list:
    """How many O atoms sit within a C–O bond length of each C (no PBC).

    2 for every carbon means the four CO2 molecules render whole, as the drawn
    cell sees them.
    """
    pos = structure.positions
    syms = structure.symbols
    carbons = [i for i, s in enumerate(syms) if s == "C"]
    oxygens = [i for i, s in enumerate(syms) if s == "O"]
    tree = cKDTree(pos[oxygens])
    return [len(tree.query_ball_point(pos[c], 1.4)) for c in carbons]


def test_conventional_expands_centred_lattice():
    # FCC rock-salt NaCl: 2-atom primitive cell -> 8-atom cubic conventional cell.
    primitive = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64))
    conventional = to_conventional(primitive)

    assert len(primitive) == 2
    assert len(conventional) == 8
    assert np.allclose(conventional.to_ase().cell.angles(), 90.0)


def test_p_lattice_keeps_all_atoms_and_order():
    # Dry ice is a P lattice: no centring expansion, so every atom and the atom
    # order survive (only lattice-translation wrapping may move atoms).
    dry = _dry_ice()
    conventional = to_conventional(dry)
    assert len(conventional) == len(dry) == 12
    assert conventional.symbols == dry.symbols


def test_conventional_reassembles_broken_molecules():
    # The regression: as loaded, only the origin CO2 is whole; the other three
    # carbons have lost their oxygens across the cell boundary. The
    # crystallographic cell must show all four CO2 intact.
    dry = _dry_ice()
    assert _o_neighbours_per_c(dry) != [2, 2, 2, 2]  # broken as loaded

    conventional = to_conventional(dry)
    assert _o_neighbours_per_c(conventional) == [2, 2, 2, 2]  # four whole CO2


def test_conventional_leaves_source_untouched():
    primitive = Structure.from_ase(bulk("Fe", "bcc", a=2.87))
    _ = to_conventional(primitive)
    assert len(primitive) == 1  # transforming must not mutate the input


def test_non_periodic_structure_is_returned_unchanged():
    mol = Structure.empty()
    mol.add_atom("O", [0.0, 0.0, 0.0])
    mol.add_atom("H", [0.76, 0.59, 0.0])

    result = to_conventional(mol)
    assert len(result) == 2
    assert not result.is_periodic
    assert result is not mol


def test_as_view_dispatches_and_copies():
    primitive = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64))

    prim_view = as_view(primitive, CellView.PRIMITIVE)
    assert len(prim_view) == 2
    assert prim_view is not primitive  # editable copy, never the source

    conv_view = as_view(primitive, CellView.CRYSTALLOGRAPHIC)
    assert len(conv_view) == 8


def test_modes_replicate_onto_conventional_cell():
    # A 2-atom primitive mode (Na +x, Cl −x) must map onto all 8 conventional
    # atoms following each atom's primitive parent, so phonons animate there too.
    primitive = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64))
    modes = PhononModes(
        [PhononMode(frequency=100.0, eigenvector=np.array([[1, 0, 0], [-1, 0, 0]], float))]
    )

    conv_struct, conv_modes = expand_modes_to_conventional(primitive, modes)

    assert len(conv_struct) == 8
    evec = conv_modes[0].eigenvector
    assert evec.shape == (8, 3)
    for sym, disp in zip(conv_struct.symbols, evec):
        expected = [1, 0, 0] if sym == "Na" else [-1, 0, 0]
        assert np.allclose(disp, expected)


def test_modes_unchanged_for_p_lattice():
    # No expansion for a P lattice -> the modes come back one-for-one.
    dry = _dry_ice()
    evec = np.random.default_rng(0).standard_normal((len(dry), 3))
    modes = PhononModes([PhononMode(frequency=50.0, eigenvector=evec)])

    conv_struct, conv_modes = expand_modes_to_conventional(dry, modes)
    assert len(conv_struct) == len(dry)
    assert np.allclose(conv_modes[0].eigenvector, evec)


def test_supercell_tiles_atoms_and_replicates_modes():
    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64))
    modes = PhononModes(
        [PhononMode(frequency=100.0, eigenvector=np.array([[1, 0, 0], [-1, 0, 0]], float))]
    )

    sup, sup_modes = tile_supercell(nacl, (2, 2, 2), modes)

    assert len(sup) == 2 * 8  # 2-atom cell, 8 image cells
    assert sup_modes[0].eigenvector.shape == (16, 3)
    for sym, disp in zip(sup.symbols, sup_modes[0].eigenvector):
        expected = [1, 0, 0] if sym == "Na" else [-1, 0, 0]
        assert np.allclose(disp, expected)


def test_supercell_keeps_the_ir_raman_labels():
    """Tiling only re-indexes the eigenvector; the selection rules the panel
    filters on must survive, or a supercell view loses the IR/Raman filter."""
    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64))
    modes = PhononModes(
        [
            PhononMode(
                frequency=100.0,
                eigenvector=np.array([[1, 0, 0], [-1, 0, 0]], float),
                ir_active=True,
                raman_active=False,
                ir_intensity=42.0,
            )
        ]
    )

    _sup, sup_modes = tile_supercell(nacl, (2, 2, 2), modes)

    assert sup_modes[0].ir_active is True
    assert sup_modes[0].raman_active is False
    assert sup_modes[0].ir_intensity == 42.0
    assert sup_modes.has_activity


def test_boundary_completion_adds_partial_molecules():
    # Dry ice: 4 whole CO2 in the cell -> completing the boundary adds the
    # corner/edge/face molecules that partially belong, all kept whole.
    dry = to_conventional(_dry_ice())  # molecule-wrapped, 12 atoms
    assert _o_neighbours_per_c(dry) == [2, 2, 2, 2]

    packed, _ = complete_boundary(dry)
    assert len(packed) > len(dry)  # image atoms added
    carbons = [s for s in packed.symbols if s == "C"]
    assert len(carbons) > 4  # more molecules than the 4 in the cell
    assert _o_neighbours_per_c(packed) == [2] * len(carbons)  # every CO2 whole

    frac = packed.to_ase().get_scaled_positions(wrap=False)
    assert frac.min() < 0.0 and frac.max() > 1.0  # atoms poke outside the box


def test_conventional_keeps_a_centred_slab_in_one_piece():
    """Regression: a slab centred on z = 0 must not be split into two sheets
    ~500 Å apart. Wrapping into the cell box may only touch the periodic axes —
    wrapping the formal 500 Å aperiodic axis flung negative-z atoms to z ≈ 500."""
    from ase.build import fcc111

    at = fcc111("Pt", size=(1, 1, 4), vacuum=0.0)
    at.translate([0, 0, -at.get_positions()[:, 2].mean()])  # centre at z = 0, as CRYSTAL does
    cell = np.asarray(at.get_cell(), dtype=float)
    cell[2] = [0.0, 0.0, 500.0]
    at.set_cell(cell)
    at.set_pbc((True, True, False))

    conv = to_conventional(Structure.from_ase(at))
    thickness = np.ptp(conv.positions[:, 2])
    assert thickness < 20.0  # the whole slab stays together (real thickness ~7 Å)


def test_boundary_completion_never_images_along_aperiodic_axis():
    """A slab must not spawn phantom copies 500 Å away in the vacuum: boundary
    completion may only image along the periodic (in-plane) axes."""
    from ase.build import fcc111

    at = fcc111("Pt", size=(1, 1, 3), vacuum=0.0)
    cell = np.asarray(at.get_cell(), dtype=float)
    cell[2] = [0.0, 0.0, 500.0]  # CRYSTAL's aperiodic placeholder
    at.set_cell(cell)
    at.set_pbc((True, True, False))
    slab = Structure.from_ase(at)

    packed, _ = complete_boundary(slab)
    # every completed atom stays with the slab, never at z ≈ 500
    assert packed.positions[:, 2].max() < 10.0
    original_zmax = slab.positions[:, 2].max()
    assert packed.positions[:, 2].max() <= original_zmax + 1e-6


def test_boundary_completion_replicates_modes():
    dry = to_conventional(_dry_ice())
    evec = np.arange(len(dry) * 3, dtype=float).reshape(len(dry), 3)
    modes = PhononModes([PhononMode(60.0, evec)])

    packed, packed_modes = complete_boundary(dry, modes)
    assert packed_modes[0].eigenvector.shape == (len(packed), 3)
    # every image atom carries its parent's eigenvector (rows drawn from evec)
    for row in packed_modes[0].eigenvector:
        assert any(np.allclose(row, evec[k]) for k in range(len(dry)))


def test_supercell_identity_and_molecule_are_noops():
    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64))
    same, _ = tile_supercell(nacl, (1, 1, 1))
    assert len(same) == len(nacl)  # 1×1×1 changes nothing

    mol = Structure.empty()
    mol.add_atom("O", [0.0, 0.0, 0.0])
    mol.add_atom("H", [0.96, 0.0, 0.0])
    tiled, _ = tile_supercell(mol, (3, 3, 3))
    assert len(tiled) == 2  # a non-periodic system is never tiled


def test_to_analysis_cell_folds_supercell_and_boundary_back():
    """A displayed (tiled + boundary-completed) cell folds back to one clean cell.

    The shown structure is unusable for symmetry analysis directly — a supercell
    box has the wrong shape and boundary images overlap their originals — so the
    Info panel folds it back into the original unit cell before analysing.
    """
    from crystalline.core.crystallography import analyze

    base = as_view(Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64)), CellView.CRYSTALLOGRAPHIC)
    unit_cell = base.to_ase().get_cell()

    analysis, _ = tile_supercell(base, (2, 2, 2))
    shown, _ = complete_boundary(analysis)
    assert len(shown) > len(base)  # tiling + images inflate the shown atom count

    folded = to_analysis_cell(shown, unit_cell)
    assert len(folded) == len(base)  # collapsed back to a single unit cell
    # and the fold recovers the crystal's true point group (m-3m), which neither
    # the supercell (wrong box) nor the boundary-completed cell (overlaps) yields.
    assert analyze(folded).point_group == analyze(base).point_group == "m-3m"


def test_to_analysis_cell_carries_edits_and_lowers_symmetry():
    """An edit made on the shown structure survives the fold and changes symmetry."""
    from crystalline.core.crystallography import analyze

    base = as_view(Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64)), CellView.CRYSTALLOGRAPHIC)
    unit_cell = base.to_ase().get_cell()

    shown, _ = complete_boundary(*tile_supercell(base, (1, 1, 1)))
    edited = Structure.from_ase(shown.to_ase())
    edited.translate_atoms([0], [0.4, 0.0, 0.0])  # nudge one atom off its site

    folded = to_analysis_cell(edited, unit_cell)
    assert analyze(folded).point_group != "m-3m"  # symmetry dropped from cubic


def test_to_analysis_cell_passes_molecules_through():
    mol = Structure.empty()
    mol.add_atom("O", [0.0, 0.0, 0.0])
    mol.add_atom("H", [0.96, 0.0, 0.0])
    out = to_analysis_cell(mol, np.zeros((3, 3)))  # degenerate cell, non-periodic
    assert len(out) == 2


# ── per-atom payloads (ADP tensors riding the same replication as the modes) ──
def _nacl() -> Structure:
    return Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64, cubic=True))


def _tensors(n: int) -> np.ndarray:
    """``(n, 3, 3)`` tensors, each distinguishable by its trace."""
    return np.stack([np.eye(3) * (i + 1) for i in range(n)])


def test_tiling_replicates_a_per_atom_payload_onto_the_images():
    nacl = _nacl()
    payload = _tensors(len(nacl))

    supercell, _modes, tiled = tile_supercell(nacl, (2, 2, 2), None, per_atom=payload)

    assert tiled.shape == (len(supercell), 3, 3)
    # every image atom carries the tensor of the source atom it copies
    for i in range(len(supercell)):
        assert np.trace(tiled[i]) in {np.trace(u) for u in payload}
    # ...and the counts work out: each source atom appears 8 times
    traces = [np.trace(u) for u in tiled]
    assert all(traces.count(np.trace(u)) == 8 for u in payload)


def test_boundary_completion_replicates_a_per_atom_payload():
    nacl = _nacl()
    payload = _tensors(len(nacl))

    packed, _modes, completed = complete_boundary(nacl, None, per_atom=payload)

    assert len(packed) > len(nacl)  # images were added
    assert completed.shape == (len(packed), 3, 3)
    # Boundary completion reorders as it walks molecules, so check the tensors
    # against the geometry rather than the order: every packed atom must carry
    # the tensor of the source atom it is a lattice translation away from.
    cell = np.asarray(nacl.cell, dtype=float)
    for position, tensor in zip(packed.positions, completed):
        fractional = np.linalg.solve(cell.T, (position - nacl.positions).T).T
        parents = np.nonzero(np.all(np.abs(fractional - np.rint(fractional)) < 1e-6, axis=1))[0]
        assert len(parents) == 1, "each packed atom images exactly one source atom"
        assert np.allclose(tensor, payload[parents[0]])


def test_omitting_the_payload_keeps_the_two_value_return():
    """The common call is unchanged: only asking for a payload adds a third
    element, as numpy's ``return_index`` does."""
    nacl = _nacl()

    assert len(tile_supercell(nacl, (2, 2, 2))) == 2
    assert len(complete_boundary(nacl)) == 2
    assert len(tile_supercell(nacl, (2, 2, 2), None, per_atom=_tensors(len(nacl)))) == 3


def test_passing_no_payload_still_returns_three_values():
    """``per_atom=None`` means "this file has nothing to replicate", not "don't
    give me the third value" — a caller that asked for three must always get
    three, or every file without ADPs fails to unpack."""
    nacl = _nacl()
    modes = PhononModes([])

    for result in (
        tile_supercell(nacl, (2, 2, 2), None, per_atom=None),
        tile_supercell(nacl, (1, 1, 1), None, per_atom=None),  # the identity path too
        complete_boundary(nacl, None, per_atom=None),
        expand_modes_to_conventional(nacl, modes, per_atom=None),
    ):
        assert len(result) == 3
        assert result[2] is None


def test_a_payload_that_does_not_fit_is_dropped_not_misaligned():
    """A conventional expansion can add atoms; indexing pre-expansion tensors by
    a post-expansion mapping would hand back the wrong atoms' data."""
    nacl = _nacl()

    _sup, _modes, dropped = tile_supercell(nacl, (2, 2, 2), None, per_atom=_tensors(1))
    assert dropped is None

    # the identity case has no mapping to check against, so it checks the length
    _same, _m, kept = tile_supercell(nacl, (1, 1, 1), None, per_atom=_tensors(len(nacl)))
    assert kept is not None and len(kept) == len(nacl)
    _same2, _m2, mismatched = tile_supercell(nacl, (1, 1, 1), None, per_atom=_tensors(3))
    assert mismatched is None


# ── modes away from Gamma: the phase that comes with each copy ────────────
# Every cell operation copies atoms into other cells. At Gamma the copies move
# in phase; away from it, cell n lags by 2*pi*q·n, and getting that wrong turns
# a travelling wave into a Gamma mode drawn on more atoms.
def _wave(structure: Structure, qpoint) -> PhononModes:
    """One complex mode on ``structure``: atom 0 along x, the rest at rest."""
    evec = np.zeros((len(structure), 3), complex)
    evec[0] = [1.0, 0.0, 0.0]
    return PhononModes([PhononMode(frequency=100.0, eigenvector=evec, qpoint=qpoint)])


def _expected_phases(shown: Structure, source: Structure, cell, qpoint) -> np.ndarray:
    """exp(2*pi*i q·n) for each shown atom, from where it was actually drawn.

    Derived from cartesian positions rather than from the tiling order, so it
    checks the phases against the picture on screen — which is the thing that
    has to be right.
    """
    phases = []
    for position in shown.positions:
        distances = np.linalg.norm(source.positions - position, axis=1)
        # the parent is the source atom this one differs from by whole cells
        offsets = np.linalg.solve(np.asarray(cell).T, (position - source.positions).T).T
        parent = int(np.argmin(np.abs(offsets - np.rint(offsets)).sum(axis=1) + 1e-9 * distances))
        phases.append(np.exp(2j * np.pi * np.dot(np.rint(offsets[parent]), qpoint)))
    return np.array(phases)


def test_tiling_a_non_gamma_mode_makes_it_a_wave():
    """q = (0, 0, 1/2) means neighbouring cells along c move in opposition —
    the one thing a single cell cannot show."""
    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64))
    modes = _wave(nacl, [0.0, 0.0, 0.5])

    sup, sup_modes = tile_supercell(nacl, (1, 1, 2), modes)

    evec = sup_modes[0].eigenvector
    n = len(nacl)
    assert evec.shape == (2 * n, 3)
    assert np.allclose(evec[:n], -evec[n:])  # the two cells in anti-phase
    assert sup_modes[0].qpoint is not None and not sup_modes[0].is_gamma


def test_tiling_a_quarter_wave_turns_the_phase_a_quarter_per_cell():
    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64))
    q = [0.0, 0.0, 0.25]

    sup, sup_modes = tile_supercell(nacl, (1, 1, 4), _wave(nacl, q))

    expected = _expected_phases(sup, nacl, nacl.cell, q)
    moving = np.linalg.norm(np.abs(sup_modes[0].eigenvector), axis=1) > 0
    assert np.allclose(sup_modes[0].eigenvector[moving, 0], expected[moving])


def test_tiling_a_gamma_mode_is_still_a_plain_replication():
    """The Gamma path must not acquire a phase (or a complex dtype) it never
    had: every existing view of a frequency calculation goes through here."""
    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64))
    evec = np.array([[1, 0, 0], [-1, 0, 0]], float)
    modes = PhononModes([PhononMode(frequency=100.0, eigenvector=evec)])

    _sup, sup_modes = tile_supercell(nacl, (2, 2, 2), modes)

    assert not np.iscomplexobj(sup_modes[0].eigenvector)
    assert np.allclose(sup_modes[0].eigenvector, np.tile(evec, (8, 1)))


def test_boundary_images_vibrate_with_the_cell_they_are_drawn_in():
    """A boundary image is drawn among *other* cells' atoms, so it has to carry
    that cell's phase — otherwise the packed cell shows atoms moving against
    neighbours they are bonded to."""
    dry = _dry_ice()
    q = [0.5, 0.0, 0.0]

    packed, packed_modes = complete_boundary(dry, _wave(dry, q))

    expected = _expected_phases(packed, dry, dry.cell, q)
    evec = packed_modes[0].eigenvector
    moving = np.linalg.norm(np.abs(evec), axis=1) > 0
    assert moving.any()
    assert np.allclose(evec[moving, 0], expected[moving])


def test_the_conventional_cell_re_expresses_q_in_its_own_basis():
    """The conventional cell has different lattice vectors, so the same physical
    q has different fractional coordinates. Tiling that cell counts translations
    in it, so the q handed on must be the one that basis expects."""
    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64))  # F lattice, 2 atoms
    q_prim = np.array([0.5, 0.5, 0.5])

    conv, conv_modes = expand_modes_to_conventional(nacl, _wave(nacl, q_prim))

    assert len(conv) == 4 * len(nacl)  # F centring: 4 primitive cells
    q_conv = conv_modes[0].qpoint
    assert q_conv is not None
    # q·R is a physical phase: reading it in either basis must give the same
    # answer for every lattice translation of the conventional cell.
    for n_conv in np.eye(3):
        translation = n_conv @ conv.cell
        n_prim = np.linalg.solve(np.asarray(nacl.cell).T, translation)
        assert np.isclose(
            np.exp(2j * np.pi * np.dot(n_conv, q_conv)),
            np.exp(2j * np.pi * np.dot(np.rint(n_prim), q_prim)),
        )


def test_a_wave_keeps_its_phases_across_expansion_and_tiling():
    """The pipeline the window runs: conventional cell, then supercell. Each
    displayed atom must end up with the phase of the *primitive* cell it is
    drawn in, however many operations put it there."""
    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64))
    q_prim = np.array([0.0, 0.0, 0.5])

    conv, conv_modes = expand_modes_to_conventional(nacl, _wave(nacl, q_prim))
    shown, shown_modes = tile_supercell(conv, (2, 1, 1), conv_modes)

    expected = _expected_phases(shown, nacl, nacl.cell, q_prim)
    evec = shown_modes[0].eigenvector
    moving = np.linalg.norm(np.abs(evec), axis=1) > 0
    assert moving.sum() == 8  # one moving atom per primitive cell drawn
    assert np.allclose(evec[moving, 0], expected[moving])


def test_replication_records_the_phase_of_every_cell():
    """The phase is the wave: a travelling mode has the same amplitude in every
    cell, so unless the phase is carried alongside, nothing on a still image can
    tell one cell from the next. It cannot be recovered from the eigenvector
    afterwards — an oscillation's phase is only defined modulo pi, which would
    make a q = 1/4 wave indistinguishable from a q = 1/2 one."""
    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64))

    sup, sup_modes = tile_supercell(nacl, (1, 1, 4), _wave(nacl, [0.0, 0.0, 0.25]))

    phase = sup_modes[0].cell_phase
    assert phase is not None and phase.shape == (len(sup),)
    # four cells, a quarter turn each, and the phase matches the eigenvector it
    # was applied to (atom 0 of the reference cell is the only one moving)
    moving = np.linalg.norm(np.abs(sup_modes[0].eigenvector), axis=1) > 0
    assert np.allclose(
        np.sort(np.mod(phase[moving], 2 * np.pi)),
        [0.0, np.pi / 2, np.pi, 3 * np.pi / 2],
    )
    assert np.allclose(
        np.exp(1j * phase[moving]),
        sup_modes[0].eigenvector[moving, 0] / sup_modes[0].eigenvector[moving, 0][0]
        * np.exp(1j * phase[moving][0]),
    )


def test_phases_accumulate_across_the_whole_pipeline():
    """Expansion, then tiling: an atom's recorded phase must be the total Bloch
    phase of the cell it ended up in, not the last leg of the journey."""
    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64))
    q = np.array([0.0, 0.0, 0.5])

    conv, conv_modes = expand_modes_to_conventional(nacl, _wave(nacl, q))
    shown, shown_modes = tile_supercell(conv, (1, 1, 2), conv_modes)

    expected = _expected_phases(shown, nacl, nacl.cell, q)  # exp(2πi q·n) per atom
    assert np.allclose(np.exp(1j * shown_modes[0].cell_phase), expected)


def test_a_gamma_mode_carries_no_phase():
    """Nothing to record when every cell moves together — and the renderer keys
    on its absence to draw one flat colour instead of a meaningless rainbow."""
    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64))
    modes = PhononModes(
        [PhononMode(frequency=100.0, eigenvector=np.array([[1, 0, 0], [-1, 0, 0]], float))]
    )

    _sup, sup_modes = tile_supercell(nacl, (2, 2, 2), modes)

    assert sup_modes[0].cell_phase is None
