"""Geometry measurements: distances, angles, dihedrals and plane fits."""

import numpy as np
import pytest

from crystalline.core import measure as M

# water, experimental geometry
_WATER = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2400, 0.9266, 0.0]])
_WATER_SYMBOLS = ["O", "H", "H"]


def test_distance_and_angle_match_known_geometry():
    assert M.distance(_WATER, 0, 1) == pytest.approx(0.9572, abs=1e-4)
    assert M.angle(_WATER, 1, 0, 2) == pytest.approx(104.5, abs=0.05)


def test_angle_and_dihedral_agree_with_ase():
    """ASE is the independent reference for the sign/convention."""
    ase = pytest.importorskip("ase")
    from ase import Atoms

    rng = np.random.default_rng(7)
    for _ in range(50):
        p = rng.normal(size=(4, 3)) * 1.5
        atoms = Atoms("C4", positions=p)
        assert M.angle(p, 0, 1, 2) == pytest.approx(atoms.get_angle(0, 1, 2), abs=1e-9)
        # ASE reports the torsion in [0, 360); ours is signed in (-180, 180]
        assert M.dihedral(p, 0, 1, 2, 3) % 360 == pytest.approx(
            atoms.get_dihedral(0, 1, 2, 3), abs=1e-9
        )
    assert ase  # silence the unused-import lint


def test_dihedral_is_signed():
    p = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 1]], dtype=float)
    left = M.dihedral(p, 0, 1, 2, 3)
    p[3] = [0, 1, -1]
    right = M.dihedral(p, 0, 1, 2, 3)
    assert left == pytest.approx(-right)  # mirrored geometry, opposite sign
    assert abs(left) == pytest.approx(90.0)


def test_plane_fit_is_exact_for_coplanar_atoms():
    square = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    centroid, normal, rms = M.plane(square, [0, 1, 2, 3])
    assert centroid == pytest.approx([0.5, 0.5, 0.0])
    assert abs(np.dot(normal, [0, 0, 1])) == pytest.approx(1.0)  # the z normal
    assert rms == pytest.approx(0.0, abs=1e-12)


def test_plane_rms_reports_non_planarity():
    puckered = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 0.4]], dtype=float
    )
    _c, _n, rms = M.plane(puckered, [0, 1, 2, 3, 4])
    assert rms > 0.1


def test_measure_dispatches_on_how_many_atoms_are_selected():
    kinds = {
        1: M.POINT,
        2: M.DISTANCE,
        3: M.ANGLE,
        4: M.DIHEDRAL,
    }
    positions = np.eye(5, 3) * 1.5
    symbols = ["C"] * 5
    for count, kind in kinds.items():
        result = M.measure(positions, symbols, range(count))
        assert result is not None and result.kind == kind
        assert result.indices == tuple(range(count))
    assert M.measure(positions, symbols, range(5)).kind == M.PLANE  # 5+ can only be a plane
    assert M.measure(positions, symbols, []) is None


def test_measure_plane_needs_three_atoms():
    positions = np.eye(4, 3)
    symbols = ["C"] * 4
    assert M.measure_plane(positions, symbols, [0, 1]) is None
    result = M.measure_plane(positions, symbols, [0, 1, 2])
    assert result is not None and result.kind == M.PLANE
    assert result.origin is not None and result.normal is not None
    assert np.linalg.norm(result.normal) == pytest.approx(1.0)


def test_summaries_carry_the_symbols_and_units():
    distance = M.measure(_WATER, _WATER_SYMBOLS, [0, 1])
    assert "O(0)" in distance.summary() and "H(1)" in distance.summary()
    assert distance.summary().endswith("Å")

    angle = M.measure(_WATER, _WATER_SYMBOLS, [1, 0, 2])
    assert angle.summary().endswith("°")
    assert angle.unit == "°"


def test_measurements_keep_the_points_they_were_taken_on():
    """The renderer draws from these, so they must survive the selection changing."""
    result = M.measure(_WATER, _WATER_SYMBOLS, [0, 1])
    assert result.points.shape == (2, 3)
    assert result.points[0] == pytest.approx(_WATER[0])


def test_degenerate_geometry_gives_nan_not_an_exception():
    coincident = np.zeros((3, 3))
    assert np.isnan(M.angle(coincident, 0, 1, 2))
    collinear = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
    assert np.isnan(M.dihedral(collinear, 0, 1, 2, 3))


def test_selection_hint_describes_what_will_be_measured():
    assert "distance" in M.selection_hint(2)
    assert "angle" in M.selection_hint(3)
    assert "plane" in M.selection_hint(9)
