"""Tests for PhononAnimator's resilience when the displayed structure changes.

Uses an off-screen PyVista plotter (the QtInteractor segfaults off-screen, but a
plain ``pyvista.Plotter`` is fine); skips if PyVista/ase are unavailable.
"""

import numpy as np
import pytest

pytest.importorskip("pyvista")
pytest.importorskip("ase")

import pyvista as pv  # noqa: E402
from ase.build import bulk  # noqa: E402

from crystalline.core.phonons import PhononMode  # noqa: E402
from crystalline.core.structure import Structure  # noqa: E402
from crystalline.viz.phonon_animator import PhononAnimator  # noqa: E402
from crystalline.viz.renderer import StructureRenderer  # noqa: E402


def _renderer():
    return StructureRenderer(pv.Plotter(off_screen=True))


def test_stale_equilibrium_does_not_crash_after_supercell_rebuild():
    # Regression: selecting a mode, then rebuilding the renderer with a different
    # atom count (a supercell), then a reset()/set_frame() — as PhononPanel does
    # on the next set_modes — must not raise on the now-stale equilibrium.
    renderer = _renderer()
    animator = PhononAnimator(renderer)

    small = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64))  # 2 atoms
    renderer.set_structure(small)
    animator.set_mode(small.positions, PhononMode(100.0, np.array([[1, 0, 0], [-1, 0, 0]], float)))
    animator.set_frame(np.pi / 2)  # animates fine at the small size

    supercell = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64).repeat((2, 2, 2)))  # 16
    renderer.set_structure(supercell)
    assert renderer.atom_count == 16

    animator.reset()  # would raise ValueError without the size guard
    animator.set_frame(np.pi / 2)  # ditto

    # once the matching equilibrium is supplied, animation resumes
    animator.set_mode(supercell.positions, PhononMode(100.0, np.ones((16, 3))))
    animator.set_frame(np.pi / 2)


def _water() -> Structure:
    """A water molecule: two O–H bonds that a stretching mode can break visually."""
    s = Structure.empty()
    s.add_atom("O", [0.0, 0.0, 0.0])
    s.add_atom("H", [0.9572, 0.0, 0.0])
    s.add_atom("H", [-0.2400, 0.9266, 0.0])
    return s


def _bond_count(renderer) -> int:
    """How many bond tubes the renderer currently draws (0 when none)."""
    from crystalline.viz.renderer import _bonded_pairs

    source = (
        renderer._positions if renderer._bond_reference is None else renderer._bond_reference
    )
    i, _j = _bonded_pairs(source, renderer._numbers, renderer.settings.bond_tolerance)
    return len(i)


def test_bonds_survive_a_large_amplitude_animation():
    # Regression: connectivity used to be re-derived from each displaced frame, so
    # a big amplitude pushed O-H past the bond-length criterion and the bonds
    # blinked out mid-cycle. They must persist for the whole cycle now.
    renderer = _renderer()
    structure = _water()
    renderer.set_structure(structure)
    resting = _bond_count(renderer)
    assert resting == 2  # both O-H bonds

    # A symmetric stretch: both hydrogens move straight out along their bonds.
    eigenvector = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-0.25, 0.97, 0.0]])
    animator = PhononAnimator(renderer)
    animator.set_mode(structure.positions, PhononMode(frequency=3600.0, eigenvector=eigenvector))
    animator.amplitude = 1.5  # far beyond the default

    for phase in PhononAnimator.phase_sequence(24):
        animator.set_frame(float(phase))
        assert _bond_count(renderer) == resting

    # Stopping restores live bonding, so editing re-bonds atoms as it should.
    animator.reset()
    assert renderer._bond_reference is None


def test_large_amplitude_would_break_bonds_without_the_reference():
    # Guards the premise of the fix: at this amplitude the displaced geometry on
    # its own really does lose a bond, so freezing the topology is what saves it.
    from crystalline.viz.renderer import _bonded_pairs

    renderer = _renderer()
    structure = _water()
    renderer.set_structure(structure)
    eigenvector = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-0.25, 0.97, 0.0]])
    from crystalline.core.phonons import displaced_positions

    stretched = displaced_positions(
        structure.positions,
        PhononMode(frequency=3600.0, eigenvector=eigenvector),
        amplitude=1.5,
        phase=np.pi / 2,
    )
    i, _ = _bonded_pairs(stretched, renderer._numbers, renderer.settings.bond_tolerance)
    assert len(i) < 2


def test_large_cell_bonds_follow_the_atoms_during_animation():
    # Regression (MOF-sized cells): above the old 300-atom limit bonds were not
    # redrawn at all while the atoms moved, so the sticks hung in space and the
    # atoms visibly detached from them as soon as the amplitude was raised.
    renderer = _renderer()
    atoms = bulk("C", "diamond", a=3.567, cubic=True) * (3, 3, 3) * (2, 1, 1)
    structure = Structure.from_ase(atoms)
    assert len(structure) > 300  # the size that used to disable live bonds
    renderer.set_structure(structure)

    equilibrium = structure.positions
    rng = np.random.default_rng(0)
    mode = PhononMode(frequency=500.0, eigenvector=rng.normal(size=equilibrium.shape) * 0.3)
    animator = PhononAnimator(renderer)
    animator.set_mode(equilibrium, mode)
    animator.amplitude = 1.5

    def bond_bounds():
        return np.array(renderer._bond_actor.GetMapper().GetInput().bounds)

    resting = bond_bounds()
    animator.set_frame(np.pi / 2)
    assert not np.allclose(resting, bond_bounds())  # the bonds moved with the atoms


def test_large_cell_keeps_every_bond_across_the_cycle():
    # The same cell also used to lose bonds wholesale once they were redrawn,
    # because connectivity came from the displaced frame.
    from crystalline.viz.renderer import _bonded_pairs

    renderer = _renderer()
    atoms = bulk("C", "diamond", a=3.567, cubic=True) * (3, 3, 3) * (2, 1, 1)
    structure = Structure.from_ase(atoms)
    renderer.set_structure(structure)
    equilibrium = structure.positions
    tolerance = renderer.settings.bond_tolerance
    expected = len(_bonded_pairs(equilibrium, structure.numbers, tolerance)[0])

    rng = np.random.default_rng(0)
    mode = PhononMode(frequency=500.0, eigenvector=rng.normal(size=equilibrium.shape) * 0.3)
    animator = PhononAnimator(renderer)
    animator.set_mode(equilibrium, mode)
    animator.amplitude = 1.5

    for phase in PhononAnimator.phase_sequence(12):
        animator.set_frame(float(phase))
        assert len(renderer._frozen_bond_pairs[0]) == expected

    # ... and re-deriving from a displaced frame really would have lost some.
    from crystalline.core.phonons import displaced_positions

    stretched = displaced_positions(equilibrium, mode, 1.5, np.pi / 2)
    assert len(_bonded_pairs(stretched, structure.numbers, tolerance)[0]) < expected
