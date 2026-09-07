"""Atomic displacement parameters: ellipsoid geometry."""

import numpy as np
import pytest

from crystalline.core.adp import (
    ADPSet,
    ellipsoid_axes,
    ellipsoid_radii,
    probability_scale,
)

# An arbitrary, definitely-anisotropic tensor with off-diagonal terms.
_U = np.array(
    [[0.0121, -0.0038, -0.0003], [-0.0038, 0.0122, -0.0014], [-0.0003, -0.0014, 0.0031]]
)


def test_probability_scale_matches_the_crystallographic_convention():
    """50% is ORTEP's, and the number every structure report is drawn at."""
    assert probability_scale(0.50) == pytest.approx(1.5382, abs=1e-4)
    assert probability_scale(0.90) == pytest.approx(2.5003, abs=1e-4)
    assert probability_scale(0.99) == pytest.approx(3.3682, abs=1e-4)
    for bad in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError):
            probability_scale(bad)


def test_ellipsoid_axes_reconstruct_the_tensor():
    """The drawn geometry must be the tensor and nothing else: scaling the
    radii back down and rebuilding from the axes has to return U exactly."""
    radii, axes, positive_definite = ellipsoid_axes(_U, 0.5)

    assert positive_definite
    assert np.allclose(axes @ axes.T, np.eye(3))          # orthonormal
    assert np.linalg.det(axes) == pytest.approx(1.0)      # a rotation, not a flip
    eigenvalues = (radii / probability_scale(0.5)) ** 2
    assert np.allclose(axes.T @ np.diag(eigenvalues) @ axes, _U)


def test_the_radii_scale_with_the_probability():
    at_50 = ellipsoid_axes(_U, 0.50)[0]
    at_99 = ellipsoid_axes(_U, 0.99)[0]

    ratio = probability_scale(0.99) / probability_scale(0.50)
    assert np.allclose(at_99, at_50 * ratio)


def test_a_non_positive_definite_tensor_is_flagged_not_drawn_imaginary():
    """Under-converged sampling produces these; a NaN radius would poison the
    mesh, so the offending axis is flattened and the caller told."""
    radii, _axes, positive_definite = ellipsoid_axes(np.diag([0.01, 0.005, -1e-4]))

    assert positive_definite is False
    assert not np.isnan(radii).any()
    assert radii.min() == 0.0


def test_adp_set_indexes_temperatures():
    tensors = np.stack([np.tile(np.eye(3) * t, (2, 1, 1)) for t in (0.001, 0.002, 0.003)])
    adps = ADPSet(temperatures=[10.0, 150.0, 300.0], tensors=tensors)

    assert len(adps) == 3
    assert adps.n_atoms == 2
    assert adps.label(2) == "300 K"
    assert np.allclose(adps.at(1), tensors[1])
    assert np.allclose(adps.at(99), tensors[2])  # out of range clamps, never raises


def test_adp_set_rejects_tensors_that_do_not_match_the_temperatures():
    with pytest.raises(ValueError):
        ADPSet(temperatures=[10.0, 300.0], tensors=np.zeros((3, 2, 3, 3)))
    with pytest.raises(ValueError):
        ADPSet(temperatures=[10.0], tensors=np.zeros((1, 2, 3)))


def test_batched_radii_agree_with_the_per_atom_axes():
    """``ellipsoid_radii`` exists only to be faster than looping ``ellipsoid_axes``
    (the renderer runs it per animation frame), so what it must not do is differ.
    Checked on positive-definite *and* NPD tensors, where the clamp applies."""
    rng = np.random.default_rng(1)
    factors = rng.normal(size=(24, 3, 3)) * 0.05
    tensors = factors @ np.swapaxes(factors, -1, -2)  # positive definite by construction
    tensors[::4] -= np.eye(3) * 0.3  # and some that are not, so the clamp is exercised
    assert not all(ellipsoid_axes(u)[2] for u in tensors)  # NPD cases really present

    for probability in (0.5, 0.9, 0.99):
        looped = np.array([ellipsoid_axes(u, probability)[0] for u in tensors])
        assert np.allclose(ellipsoid_radii(tensors, probability), looped)


def test_batched_radii_handle_an_empty_stack_and_reject_a_single_tensor():
    """A structure can legitimately have no atoms; a bare (3, 3) tensor, though,
    is a caller confusing this with ``ellipsoid_axes``."""
    assert ellipsoid_radii(np.empty((0, 3, 3))).shape == (0, 3)
    with pytest.raises(ValueError):
        ellipsoid_radii(_U)
