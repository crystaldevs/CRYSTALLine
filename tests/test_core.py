"""Unit tests for the Qt-free domain core (no display needed)."""

import numpy as np
import pytest

from crystalline.core.structure import Structure
from crystalline.core.phonons import (
    PhononMode,
    PhononModes,
    commensurate_repeats,
    displaced_positions,
    frame_displacement,
    phase_factors,
    qpoint_label,
)


def test_add_atoms_batch_appends_and_fires_once():
    s = Structure.empty()
    s.add_atom("C", [0.0, 0.0, 0.0])
    events = []
    s.add_listener(lambda st: events.append(len(st)))

    new = s.add_atoms(["O", "H", "H"], [[1, 0, 0], [2, 0, 0], [0, 1, 0]])
    assert new == [1, 2, 3]
    assert len(s) == 4
    assert s.symbols == ["C", "O", "H", "H"]
    assert events == [4]  # a single notification for the whole batch

    assert s.add_atoms([], []) == []  # empty is a no-op
    with pytest.raises(ValueError):
        s.add_atoms(["C"], [[0, 0, 0], [1, 1, 1]])  # length mismatch


def test_add_move_remove_atom_notifies():
    s = Structure.empty()
    events = []
    s.add_listener(lambda st: events.append(len(st)))

    i = s.add_atom("C", [0.0, 0.0, 0.0])
    j = s.add_atom("O", [1.2, 0.0, 0.0])
    assert (i, j) == (0, 1)
    assert len(s) == 2
    assert s.symbols == ["C", "O"]

    s.move_atom(1, [1.5, 0.0, 0.0])
    assert np.allclose(s.positions[1], [1.5, 0.0, 0.0])

    s.set_symbols([0], "N")
    assert s.symbols[0] == "N"

    s.remove_atoms([0])
    assert len(s) == 1 and s.symbols == ["O"]

    # each mutating call fired exactly one notification
    assert events == [1, 2, 2, 2, 1]


def test_invalid_symbol_and_index():
    s = Structure.empty()
    with pytest.raises(ValueError):
        s.add_atom("Xx", [0, 0, 0])
    s.add_atom("H", [0, 0, 0])
    with pytest.raises(IndexError):
        s.move_atom(5, [0, 0, 0])


def test_batch_edits_translate_duplicate_set_remove():
    s = Structure.empty()
    for i, el in enumerate(["C", "O", "N", "H"]):
        s.add_atom(el, [i, 0, 0])
    events = []
    s.add_listener(lambda st: events.append(len(st)))

    s.translate_atoms([0, 2], [0.0, 1.0, 0.0])
    assert np.allclose(s.positions[0], [0, 1, 0])
    assert np.allclose(s.positions[2], [2, 1, 0])
    assert np.allclose(s.positions[1], [1, 0, 0])  # untouched

    new = s.duplicate_atoms([1, 3], offset=[0.0, 0.0, 5.0])
    assert new == [4, 5]
    assert s.symbols[4] == "O" and s.symbols[5] == "H"
    assert np.allclose(s.positions[4], [1, 0, 5])

    s.set_symbols([0, 4], "S")
    assert s.symbols[0] == "S" and s.symbols[4] == "S"

    s.remove_atoms([5, 0])  # order-independent, high-to-low internally
    assert len(s) == 4
    assert s.symbols == ["O", "N", "H", "S"]

    # one notification per batch action (4 actions)
    assert events == [4, 6, 6, 4]


def test_batch_edits_validate_and_ignore_empty():
    s = Structure.empty()
    s.add_atom("C", [0, 0, 0])
    calls = []
    s.add_listener(lambda st: calls.append(1))

    with pytest.raises(IndexError):
        s.remove_atoms([0, 9])  # 9 out of range -> nothing removed
    assert len(s) == 1
    with pytest.raises(ValueError):
        s.set_symbols([0], "Zz")

    # empty selection is a no-op that does not notify
    s.translate_atoms([], [1, 2, 3])
    assert s.duplicate_atoms([]) == []
    assert calls == []


def test_phonon_mode_shape_validation():
    with pytest.raises(ValueError):
        PhononMode(frequency=100.0, eigenvector=np.zeros((4,)))  # not (N, 3)
    m = PhononMode(frequency=-2.0, eigenvector=np.zeros((3, 3)))
    assert m.is_imaginary and m.n_atoms == 3


def test_set_lattice_parameters_scales_atoms_and_notifies():
    s = Structure.empty()
    s.set_cell(np.diag([4.0, 4.0, 4.0]), periodic=True)
    s.add_atom("Na", [0.0, 0.0, 0.0])
    s.add_atom("Cl", [2.0, 2.0, 2.0])  # fractional (0.5, 0.5, 0.5)
    frac_before = np.linalg.solve(s.cell.T, s.positions.T).T

    events = []
    s.add_listener(lambda st: events.append(1))
    s.set_lattice_parameters(6.0, 6.0, 6.0, 90.0, 90.0, 90.0)

    assert np.allclose(s.cellpar, [6.0, 6.0, 6.0, 90.0, 90.0, 90.0])
    # atoms moved with the cell: fractional coordinates preserved
    frac_after = np.linalg.solve(s.cell.T, s.positions.T).T
    assert np.allclose(frac_before, frac_after)
    assert np.allclose(s.positions[1], [3.0, 3.0, 3.0])  # 0.5 * 6.0
    assert events == [1]  # exactly one notification

    # a non-orthogonal angle is applied faithfully
    s.set_lattice_parameters(6.0, 6.0, 6.0, 90.0, 90.0, 120.0)
    assert np.isclose(s.cellpar[5], 120.0)


def test_displaced_positions_at_key_phases():
    eq = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    evec = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    mode = PhononMode(frequency=50.0, eigenvector=evec)

    # phase 0 -> sin(0)=0 -> equilibrium
    assert np.allclose(displaced_positions(eq, mode, amplitude=1.0, phase=0.0), eq)
    # phase pi/2 -> sin=1 -> full displacement
    peak = displaced_positions(eq, mode, amplitude=0.5, phase=np.pi / 2)
    assert np.allclose(peak, eq + 0.5 * evec)


def test_amplitude_is_the_peak_atomic_displacement_whatever_the_cell_size():
    """Eigenvectors come back normalised over all 3N components, so the motion
    of any one atom faded as 1/sqrt(N) and the default amplitude that suited a
    molecule was invisible for a large cell. Amplitude now means the peak
    displacement of the most-displaced atom, in Angstrom."""
    for natom in (2, 50, 500):
        eq = np.zeros((natom, 3))
        evec = np.zeros((natom, 3))
        evec[:, 0] = 1.0
        evec /= np.linalg.norm(evec)  # as CRYSTALClear normalises it

        peak = displaced_positions(eq, PhononMode(100.0, evec), amplitude=0.4, phase=np.pi / 2)
        assert np.isclose(np.max(np.linalg.norm(peak - eq, axis=1)), 0.4)


def test_amplitude_keeps_the_relative_motion_within_a_mode():
    # One atom moving twice as far as another must still do so after scaling.
    eq = np.zeros((3, 3))
    evec = np.array([[2.0, 0, 0], [1.0, 0, 0], [0.0, 0, 0]])
    peak = displaced_positions(eq, PhononMode(100.0, evec), amplitude=0.5, phase=np.pi / 2)
    assert np.allclose(peak[:, 0], [0.5, 0.25, 0.0])


def test_a_null_eigenvector_does_not_move_or_blow_up():
    eq = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mode = PhononMode(frequency=0.0, eigenvector=np.zeros((2, 3)))
    assert np.allclose(displaced_positions(eq, mode, amplitude=1.0, phase=np.pi / 2), eq)


def test_activity_labels_ride_along_with_the_mode():
    mode = PhononMode(
        frequency=1050.0,
        eigenvector=np.eye(3)[:2],
        ir_active=True,
        raman_active=False,
        ir_intensity=12.5,
    )
    assert mode.has_activity

    # Remapping onto a supercell keeps frequency and labels, swaps the vectors.
    tiled = mode.with_eigenvector(np.zeros((4, 3)))
    assert tiled.n_atoms == 4
    assert tiled.frequency == 1050.0
    assert tiled.ir_active is True and tiled.raman_active is False
    assert tiled.ir_intensity == 12.5

    # An output without the analysis leaves the labels unknown, not False.
    plain = PhononMode(frequency=1050.0, eigenvector=np.eye(3)[:2])
    assert plain.ir_active is None and not plain.has_activity
    assert not PhononModes([plain]).has_activity
    assert PhononModes([plain, mode]).has_activity


def test_phonon_modes_collection():
    modes = PhononModes(
        [PhononMode(10.0, np.zeros((2, 3))), PhononMode(-5.0, np.zeros((2, 3)))]
    )
    assert len(modes) == 2
    assert np.allclose(modes.frequencies, [10.0, -5.0])
    assert modes[1].is_imaginary


# ── modes away from Gamma (a SCELPHONO run's other q-points) ──────────────
# Such a mode is a travelling wave: its eigenvector is complex, each cell lags
# the last by q·n, and the animation has to evaluate the real displacement at
# every frame instead of scaling one fixed pattern.
def test_a_mode_defaults_to_gamma_and_keeps_a_real_eigenvector():
    mode = PhononMode(frequency=100.0, eigenvector=np.eye(3)[:2])
    assert mode.is_gamma and mode.qpoint is None
    assert mode.qpoint_label == "Γ"
    assert not np.iscomplexobj(mode.eigenvector)
    # An explicit zero q is Gamma too — the loader labels the first set that way.
    assert PhononMode(100.0, np.eye(3)[:2], qpoint=[0, 0, 0]).is_gamma


def test_a_complex_eigenvector_survives_and_carries_its_qpoint():
    evec = np.array([[1.0 + 0.0j, 0, 0], [0, 0.5j, 0]])
    mode = PhononMode(frequency=100.0, eigenvector=evec, qpoint=[0.0, 0.0, 0.5])

    assert np.iscomplexobj(mode.eigenvector)
    assert not mode.is_gamma
    assert mode.qpoint_label == "(0, 0, 1/2)"
    with pytest.raises(ValueError):
        PhononMode(frequency=100.0, eigenvector=evec, qpoint=[0.0, 0.5])


def test_a_complex_mode_moves_as_a_wave_over_the_cycle():
    """Re(e) and Im(e) are CRYSTAL's in-phase and anti-phase blocks; the atom's
    displacement runs through both over a cycle, which is what makes one cell of
    a tiled wave lag another rather than merely scaling it."""
    eq = np.zeros((2, 3))
    evec = np.array([[1.0, 0, 0], [1.0j, 0, 0]])  # two atoms a quarter cycle apart
    mode = PhononMode(frequency=100.0, eigenvector=evec, qpoint=[0.0, 0.0, 0.5])

    at_zero = displaced_positions(eq, mode, amplitude=1.0, phase=0.0)
    quarter = displaced_positions(eq, mode, amplitude=1.0, phase=np.pi / 2)

    # phase 0: the real atom is at rest, the imaginary one at full stretch
    assert np.allclose(at_zero[0], [0.0, 0, 0]) and np.allclose(at_zero[1], [1.0, 0, 0])
    # a quarter cycle later they have swapped roles
    assert np.allclose(quarter[0], [1.0, 0, 0]) and np.allclose(quarter[1], [0.0, 0, 0])
    # a real eigenvector still moves as e*sin(phase) — the Gamma animation is unchanged
    real = PhononMode(frequency=100.0, eigenvector=np.real(evec))
    assert np.allclose(frame_displacement(real.eigenvector, 0.3), np.real(evec) * np.sin(0.3))


def test_amplitude_still_means_the_peak_atomic_displacement_when_complex():
    eq = np.zeros((2, 3))
    evec = np.array([[0.3 + 0.4j, 0, 0], [0.1, 0, 0]])  # |e| = 0.5 for atom 0
    mode = PhononMode(frequency=100.0, eigenvector=evec, qpoint=[0.5, 0, 0])

    frames = [
        displaced_positions(eq, mode, amplitude=0.2, phase=p)
        for p in np.linspace(0, 2 * np.pi, 64)
    ]
    excursion = max(np.max(np.linalg.norm(f - eq, axis=1)) for f in frames)
    assert np.isclose(excursion, 0.2, atol=1e-3)


def test_phase_factors_lag_each_cell_by_q_dot_n():
    offsets = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]], float)
    factors = phase_factors([0.0, 0.0, 0.25], offsets)

    assert np.allclose(factors, [1, 1j, -1, -1j])
    # Gamma (and a missing q) means "no phase at all", so callers can skip it
    assert phase_factors(None, offsets) is None
    assert phase_factors([0.0, 0.0, 0.0], offsets) is None


def test_qpoint_labels_read_as_the_fractions_crystal_sampled():
    assert qpoint_label(None) == "Γ"
    assert qpoint_label([0.0, 0.0, 0.0]) == "Γ"
    assert qpoint_label([0.5, 0.0, 1 / 3]) == "(1/2, 0, 1/3)"


def test_commensurate_repeats_is_the_smallest_whole_period():
    assert commensurate_repeats(None) == (1, 1, 1)
    assert commensurate_repeats([0.0, 0.0, 0.25]) == (1, 1, 4)
    assert commensurate_repeats([0.5, 1 / 3, 0.0]) == (2, 3, 1)
    # A fine q grid must not ask for a structure too big to draw.
    assert commensurate_repeats([1 / 16, 0.0, 0.0], limit=8) == (8, 1, 1)


def test_phonon_modes_report_the_qpoint_they_share():
    gamma = PhononModes([PhononMode(10.0, np.zeros((2, 3)))])
    assert gamma.is_gamma and gamma.qpoint_label == "Γ"

    away = PhononModes(
        [PhononMode(10.0, np.zeros((2, 3), complex), qpoint=[0.0, 0.5, 0.0])]
    )
    assert not away.is_gamma and away.qpoint_label == "(0, 1/2, 0)"
    assert PhononModes([]).is_gamma  # no modes: nothing to be away from Gamma
