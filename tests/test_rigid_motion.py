"""Turning a selection about an axis, with its periodic images kept as images."""

import numpy as np
import pytest

from crystalline.core.rigid_motion import (
    image_classes,
    rotate_points,
    rotated_selection,
    rotation_matrix,
)
from crystalline.core.structure import Structure


def test_a_positive_angle_turns_anticlockwise_looking_down_the_axis():
    """The right-hand rule: x turned a quarter about z points along y."""
    assert np.allclose(rotation_matrix((0, 0, 1), 90) @ [1, 0, 0], [0, 1, 0])
    assert np.allclose(rotation_matrix((0, 0, 5), -90) @ [1, 0, 0], [0, -1, 0])


def test_the_axis_needs_a_direction():
    with pytest.raises(ValueError):
        rotation_matrix((0, 0, 0), 30)


def test_points_turn_about_the_line_through_the_centre():
    turned = rotate_points([[2, 1, 0]], (0, 0, 1), 180, centre=(1, 1, 0))
    assert np.allclose(turned, [[0, 1, 0]])


def test_a_whole_turn_puts_everything_back():
    points = np.random.default_rng(0).normal(size=(5, 3))
    assert np.allclose(rotate_points(points, (1, 2, 3), 360, (0.5, 0, 1)), points)


def _images_by(table):
    return lambda i: table.get(i, [])


def test_the_selection_and_its_images_fall_into_classes():
    """0 and 3 are images of one another, both selected; 1 has an image, 4."""
    images = _images_by({0: [3], 3: [0], 1: [4], 4: [1]})
    classes = image_classes([0, 1, 3], images)

    assert classes == [[0, 3], [1, 4]]


def test_images_move_with_their_atom_and_stay_images():
    """An image left behind is no longer an image; turned on its own, it would
    move by a different amount from its atom and stop being one too."""
    lattice = np.array([5.0, 0.0, 0.0])
    positions = np.array([[0.0, 1.0, 0.0],        # selected
                          [5.0, 1.0, 0.0]])       # its image, one cell along x
    images = _images_by({0: [1], 1: [0]})

    moved = rotated_selection(positions, [0], images, (0, 0, 1), 90, centre=(0, 0, 0))

    assert np.allclose(moved[0], [-1.0, 0.0, 0.0])
    assert np.allclose(moved[1] - moved[0], lattice)      # still an image


def test_an_atom_and_its_image_both_selected_turn_as_one():
    """The one nearest the centre is the one turned; the other follows it."""
    positions = np.array([[0.0, 1.0, 0.0], [5.0, 1.0, 0.0]])
    images = _images_by({0: [1], 1: [0]})

    moved = rotated_selection(positions, [1, 0], images, (0, 0, 1), 90, centre=(0, 0, 0))

    assert np.allclose(moved[0], [-1.0, 0.0, 0.0])
    assert np.allclose(moved[1] - moved[0], [5.0, 0.0, 0.0])


def test_a_chain_turned_about_its_own_length_keeps_its_shape():
    """A zigzag along x, turned about x through its centroid: every bond and
    every distance to the axis is kept."""
    chain = np.array([[0.0, 0.0, 0.0], [1.25, 0.8, 0.0], [2.5, 0.0, 0.0], [3.75, 0.8, 0.0]])
    no_images = _images_by({})
    centre = chain.mean(axis=0)

    moved = rotated_selection(chain, range(4), no_images, (1, 0, 0), 73, centre)
    turned = np.array([moved[i] for i in range(4)])

    bonds = lambda p: np.linalg.norm(np.diff(p, axis=0), axis=1)
    assert np.allclose(bonds(turned), bonds(chain))
    radius = lambda p: np.linalg.norm((p - centre)[:, 1:], axis=1)
    assert np.allclose(radius(turned), radius(chain))
    assert np.allclose(turned[:, 0], chain[:, 0])          # nothing moves along the axis


def test_moving_several_atoms_is_one_edit():
    structure = Structure.from_ase(__import__("ase").Atoms("H3", positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]]))
    calls = []
    structure.add_listener(lambda *a: calls.append(a))

    structure.move_atoms({0: (0, 0, 1), 2: (2, 0, 1)})

    assert len(calls) == 1
    assert np.allclose(structure.positions, [[0, 0, 1], [1, 0, 0], [2, 0, 1]])

