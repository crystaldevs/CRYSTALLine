"""What a normal mode is made of: element composition and localisation.

The convention these formulas rest on is that CRYSTAL's eigenvectors are
*cartesian* displacements (normalised over the cell), so the kinetic-energy
share of an atom carries a factor of its mass. An acoustic mode pins that down
exactly: a rigid translation of the whole cell must come out with each element's
share equal to its share of the cell's mass, and no other weighting reproduces
that.
"""

import numpy as np
import pytest
from ase.data import atomic_masses, atomic_numbers

from crystalline.core.mode_analysis import mode_character
from crystalline.core.phonons import PhononMode


def _mode(eigenvector) -> PhononMode:
    ev = np.asarray(eigenvector, dtype=float)
    norm = np.linalg.norm(ev)
    return PhononMode(frequency=100.0, eigenvector=ev / norm if norm else ev)


_CCO = [atomic_numbers["C"], atomic_numbers["C"], atomic_numbers["O"]]


def test_a_rigid_translation_splits_by_mass():
    """The acoustic-mode check that identifies the eigenvector convention."""
    translation = _mode([[0.0, 0.0, 1.0]] * 3)  # every atom moves identically

    character = mode_character(translation, _CCO)

    total = 2 * atomic_masses[atomic_numbers["C"]] + atomic_masses[atomic_numbers["O"]]
    shares = dict(character.composition)
    assert shares["C"] == pytest.approx(2 * atomic_masses[atomic_numbers["C"]] / total, abs=1e-4)
    assert shares["O"] == pytest.approx(atomic_masses[atomic_numbers["O"]] / total, abs=1e-4)


def test_a_mode_on_one_atom_is_reported_as_one_atom_moving():
    character = mode_character(_mode([[0, 0, 0], [0, 0, 0], [0.0, 0.0, 1.0]]), _CCO)

    assert character.composition[0][0] == "O"  # the element carrying the mode
    assert dict(character.composition)["O"] == pytest.approx(1.0)
    assert character.effective_atoms == pytest.approx(1.0)
    assert character.n_atoms == 3


def test_a_mode_spread_over_every_atom_reaches_the_atom_count():
    """Equal *energy* on each atom — not equal displacement — is full spreading."""
    masses = atomic_masses[np.asarray(_CCO)]
    even_energy = _mode(np.column_stack([np.zeros(3), np.zeros(3), 1.0 / np.sqrt(masses)]))

    character = mode_character(even_energy, _CCO)

    assert character.effective_atoms == pytest.approx(3.0)
    assert character.effective_atoms == pytest.approx(character.n_atoms)  # fully delocalised


def test_a_null_mode_has_no_composition():
    character = mode_character(_mode(np.zeros((3, 3))), _CCO)

    assert character.composition == ()
    assert character.summary() == "no motion"
    assert character.effective_atoms == 0.0


def test_a_mode_that_does_not_match_the_geometry_is_reported_empty():
    """Modes and geometry go out of step (a supercell, an edit); don't raise."""
    character = mode_character(_mode([[1.0, 0, 0], [0, 0, 0]]), _CCO)

    assert character.composition == ()
    assert character.n_atoms == 3


def test_trace_elements_are_dropped_from_the_composition():
    """A 0.1% contribution is rounding noise in a label, not chemistry."""
    tiny = _mode([[1.0, 0, 0], [1.0, 0, 0], [0.001, 0, 0]])

    assert [sym for sym, _share in mode_character(tiny, _CCO).composition] == ["C"]


def test_summary_reads_as_composition_then_spread():
    summary = mode_character(_mode([[0, 0, 0], [0, 0, 0], [0.0, 0.0, 1.0]]), _CCO).summary()

    assert "O 100%" in summary
    assert "1.0 of 3 atoms move" in summary


def test_a_whole_list_is_analysed_as_each_mode_would_be():
    """The batch the Phonons panel fills its list with must say, mode for mode,
    what the one-at-a-time analysis says — element order included, which for
    equal shares follows the order the elements first appear in."""
    from crystalline.core.mode_analysis import mode_characters

    rng = np.random.default_rng(3)
    numbers = [atomic_numbers[s] for s in ("O", "C", "H", "C", "N", "H", "O")]
    modes = [_mode(rng.normal(size=(7, 3))) for _ in range(20)]
    modes.append(_mode(np.zeros((7, 3))))                       # a null mode

    batch = mode_characters(modes, numbers)

    assert len(batch) == len(modes)
    for mode, character in zip(modes, batch):
        single = mode_character(mode, numbers)
        assert [s for s, _ in character.composition] == [s for s, _ in single.composition]
        assert [v for _, v in character.composition] == pytest.approx(
            [v for _, v in single.composition])
        assert character.effective_atoms == pytest.approx(single.effective_atoms)
    assert batch[-1].composition == ()

    from crystalline.core.mode_analysis import _elements

    symbols, element = _elements(np.asarray(numbers))
    assert symbols == ["O", "C", "H", "N"]                       # as they first appear
    assert list(element) == [0, 1, 2, 1, 3, 2, 0]
