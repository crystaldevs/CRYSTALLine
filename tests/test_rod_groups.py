"""Rod groups: CRYSTAL's 99, their operators, and finding a chain's own.

Checked against what CRYSTAL itself prints for a water chain in rod group 4
(P 2₁11, from space group P 1 1 2₁): the identity and (x + ½, −y, −z), with the
chain along x and its axis through the origin of y and z.
"""

import numpy as np
import pytest

import sample_data as sample
from crystalline.core import rod_groups as rg

# The water chain as it was given to CRYSTAL: rod group 4, its asymmetric unit.
_WATER_A = 4.965635
_WATER = ((8, (0.0, 0.6852945, 0.0)), (1, (0.032558, 1.5213825, -0.400375)),
          (1, (0.168195, 0.2242435, 0.0)))


# Test12 of CRYSTAL's suite as a polymer: rod group 12, P2/m11, every atom 1.5 Å
# off the axis. CRYSTAL reads a alone from its lattice record.
_UREA_A = 8.2
_UREA = ((6, (0.0, 1.5, 0.0)), (8, (0.0, 1.5, 1.261401)), (7, (0.1459, 1.6459, -0.69979)),
         (1, (0.2575, 1.7575, -0.202817)), (1, (0.1441, 1.6441, -1.704975)))


def _chain(number, unit, a):
    """Every atom of a repeat, from a group and its asymmetric unit, in Å."""
    group = rg.rod_group(number)
    numbers = np.array([z for z, _ in unit])
    fractional = np.array([position for _, position in unit], dtype=float)
    images = np.concatenate([rg._image(group, k, fractional) for k in range(group.order)])
    kinds = np.tile(numbers, group.order)
    images[:, 0] %= 1.0
    keep = [i for i in range(len(images)) if not any(
        kinds[i] == kinds[j] and abs((images[i, 0] - images[j, 0] + 0.5) % 1 - 0.5) < 1e-6
        and np.allclose(images[i, 1:], images[j, 1:], atol=1e-6) for j in range(i))]
    cartesian = images[keep].copy()
    cartesian[:, 0] *= a
    return kinds[keep], cartesian


def _closest(cartesian, a):
    """The shortest distance between two atoms of the chain, across repeats too."""
    gaps = cartesian[:, None, :] - cartesian[None, :, :]
    gaps[..., 0] = (gaps[..., 0] / a + 0.5) % 1.0 * a - 0.5 * a
    distance = np.linalg.norm(gaps, axis=2)
    np.fill_diagonal(distance, np.inf)
    return float(distance.min()) if len(cartesian) > 1 else np.inf


def _disguised(cartesian, angle=0.7, along=0.31, a=1.0, off=(2.0, -1.5)):
    """The same chain turned about its axis, slid along it and moved off the origin."""
    c, s = np.cos(angle), np.sin(angle)
    moved = cartesian.copy()
    moved[:, 0] += along * a
    moved[:, 1] = cartesian[:, 1] * c - cartesian[:, 2] * s + off[0]
    moved[:, 2] = cartesian[:, 1] * s + cartesian[:, 2] * c + off[1]
    return moved


# ── the table ─────────────────────────────────────────────────────────────
def test_rod_group_4_is_what_crystal_prints_for_it():
    group = rg.rod_group(4)
    assert (group.symbol, group.space_group) == ("P2_111", "P 1 1 21")
    assert np.allclose(group.rotations[0], np.eye(3)) and group.translations[0] == 0.0
    assert np.allclose(group.rotations[1], np.diag([1, -1, -1]))
    assert group.translations[1] == pytest.approx(0.5)


def test_rod_group_12_is_what_crystal_prints_for_it():
    """The identity, the twofold axis along the chain, the inversion and the
    mirror across the chain, none with a translation."""
    group = rg.rod_group(12)
    assert (group.symbol, group.space_group) == ("P2/m11", "P 1 1 2/m")
    printed = [np.eye(3), np.diag([1, -1, -1]), -np.eye(3), np.diag([-1, 1, 1])]
    for rotation in printed:
        assert any(np.allclose(rotation, mine) for mine in group.rotations)
    assert group.order == 4 and not any(group.translations)


def test_all_99_are_built_and_keep_the_chain_on_its_axis():
    """Every operation sends x to ±x and mixes y and z only among themselves."""
    orders = []
    for number in range(1, 100):
        group = rg.rod_group(number)
        orders.append(group.order)
        for rotation in group.rotations:
            assert abs(rotation[0, 0]) == pytest.approx(1.0)
            assert np.allclose(rotation[0, 1:], 0) and np.allclose(rotation[1:, 0], 0)
            assert np.allclose(rotation @ rotation.T, np.eye(3))
    assert max(orders) == 24 and orders[95] == 24                  # P6/mmm


@pytest.mark.parametrize("number, order", [(1, 1), (2, 2), (18, 4), (28, 8), (51, 16),
                                           (59, 6), (72, 12), (96, 24), (99, 24)])
def test_each_group_has_as_many_operations_as_its_point_group(number, order):
    assert rg.rod_group(number).order == order


# ── finding a chain's group ───────────────────────────────────────────────
def test_the_water_chain_is_found_in_the_group_it_was_run_in():
    numbers, chain = _chain(4, _WATER, _WATER_A)
    symmetry = rg.analyse(numbers, _disguised(chain, a=_WATER_A), _WATER_A)

    assert symmetry.group.number == 4 and rg.regenerates(symmetry)
    unit_numbers, unit = rg.asymmetric_unit(symmetry)
    assert sorted(unit_numbers) == [1, 1, 8]


def test_the_urea_chain_is_found_in_the_group_it_was_run_in():
    numbers, chain = _chain(12, _UREA, _UREA_A)
    assert len(numbers) == 16                      # as CRYSTAL generates it
    symmetry = rg.analyse(numbers, _disguised(chain, a=_UREA_A), _UREA_A)
    assert symmetry.group.number == 12 and rg.regenerates(symmetry)
    assert len(rg.asymmetric_unit(symmetry)[0]) == 5


def test_the_manual_s_sn_chain_keeps_at_least_its_screw():
    """Test03 of the manual: (SN)x in rod group 4."""
    numbers, chain = _chain(4, ((16, (0.0, -0.844969, 0.0)),
                                (7, (0.141600540, 0.667077, -0.00093))), 4.431)
    symmetry = rg.analyse(numbers, _disguised(chain, a=4.431), 4.431)
    assert symmetry.group.order >= 2 and rg.regenerates(symmetry)


@pytest.mark.parametrize("seed", [1, 2])
def test_every_group_is_found_in_a_chain_built_from_it(seed):
    """Atoms at random places, the chain built by the group, then turned about
    its axis, slid along it and moved off the origin: the group found is at
    least as large, and its asymmetric unit rebuilds the chain."""
    rng = np.random.default_rng(seed)
    for number in range(1, 100):
        while True:
            unit = tuple((z, (rng.random(), rng.normal(0, 1.5), rng.normal(0, 1.5)))
                         for z in (6, 8))
            numbers, chain = _chain(number, unit, 3.7)
            if _closest(chain, 3.7) > 0.3:     # atoms apart, as in any real chain
                break
        moved = _disguised(chain, angle=rng.random() * np.pi, along=rng.random(), a=3.7)
        symmetry = rg.analyse(numbers, moved, 3.7, 1e-3)
        assert symmetry.group.order >= rg.rod_group(number).order, number
        assert rg.regenerates(symmetry), number


def test_a_chain_on_its_axis_takes_the_largest_group():
    """Atoms on the axis have every rotation about it; P6/mmm is the largest
    a crystal's operations allow."""
    numbers = np.array([8, 12])
    chain = np.array([[0.0, 0.0, 0.0], [1.78, 0.0, 0.0]])
    assert rg.analyse(numbers, chain, 3.567).group.number == 96


def test_a_chain_with_no_symmetry_is_rod_group_1():
    numbers = np.array([6, 7, 8, 1])
    chain = np.array([[0.1, 0.3, 0.2], [0.9, -0.7, 0.4], [1.7, 0.2, -0.9], [2.2, 1.1, 0.3]])
    symmetry = rg.analyse(numbers, chain, 3.1)
    assert symmetry.group.number == 1 and rg.regenerates(symmetry)


# ── the deck ──────────────────────────────────────────────────────────────
def _water_structure():
    from ase import Atoms

    from crystalline.core.structure import Structure

    numbers, chain = _chain(4, _WATER, _WATER_A)
    cell = np.diag([_WATER_A, 500.0, 500.0])
    return Structure.from_ase(Atoms(numbers=numbers, positions=chain, cell=cell,
                                    pbc=(True, False, False)))


def test_a_polymer_deck_is_written_in_its_rod_group():
    from crystalline.core import crystal_input as ci

    lines = ci.build_input(_water_structure(), ci.CrystalInputSpec()).splitlines()
    assert lines[1:5] == ["POLYMER", "4", f"{_WATER_A:.6f}", "3"]
    assert "-0.0000000000" not in "\n".join(lines[5:8])


def test_without_symmetry_a_polymer_is_rod_group_1_with_every_atom():
    from crystalline.core import crystal_input as ci

    spec = ci.CrystalInputSpec(geometry=ci.GeometryOptions(use_symmetry=False))
    lines = ci.build_input(_water_structure(), spec).splitlines()
    assert lines[1:5] == ["POLYMER", "1", f"{_WATER_A:.6f}", "6"]


def test_the_info_panel_names_the_rod_group():
    from crystalline.core.crystallography import analyze

    rows = dict(analyze(_water_structure()).rows())
    assert rows["Rod group"] == "P2₁11 (No. 4)"


_RUN = sample.path("H2O_polymer.out")


@pytest.mark.skipif(not _RUN, reason="the water chain run is not available")
def test_the_deck_written_from_crystal_s_output_is_the_deck_it_was_run_from():
    from crystalline.core import crystal_input as ci
    from crystalline.crystalio import load

    lines = ci.build_input(load(_RUN).structure, ci.CrystalInputSpec()).splitlines()
    assert lines[1:5] == ["POLYMER", "4", f"{_WATER_A:.6f}", "3"]
    written = np.array([[float(v) for v in line.split()[1:]] for line in lines[5:8]])
    given = np.array([position for _, position in _WATER])
    assert np.allclose(written, given, atol=1e-6)


_UREA_RUN = sample.path("Urea_polymer.out")


@pytest.mark.skipif(not _UREA_RUN, reason="the urea chain run is not available")
def test_the_urea_deck_written_from_crystal_s_output_is_the_deck_it_was_run_from():
    from crystalline.core import crystal_input as ci
    from crystalline.crystalio import load

    lines = ci.build_input(load(_UREA_RUN).structure, ci.CrystalInputSpec()).splitlines()
    assert lines[1:5] == ["POLYMER", "12", f"{_UREA_A:.6f}", "5"]
    written = np.array([[float(v) for v in line.split()[1:]] for line in lines[5:10]])
    given = np.array([position for _, position in _UREA])
    assert np.allclose(written, given, atol=1e-6)


# ── every group, read back as the manual names it ─────────────────────────
# The manual's "polymer" symbols are written along the chain (x) and, across it,
# along the space group's secondary and tertiary directions turned into the
# polymer frame: y and z for the orthogonal groups; the turned a and [110], or
# a and [1-10], for the tetragonal and hexagonal ones. That column was written
# independently of the space-group settings the operators come from, so reading
# a symbol off the operators and finding the manual's checks the turn onto x
# and every orientation across the chain, for all 99 at once.
def _directions(space_group):
    if space_group <= 74:
        return [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])]
    tertiary = (np.array([0, 1.0, -1.0]) / np.sqrt(2) if space_group <= 142
                else np.array([0, -0.5, -np.sqrt(3) / 2]))
    return [np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), tertiary]


def _order(rotation):
    power = np.eye(3)
    for k in range(1, 7):
        power = power @ rotation
        if np.allclose(power, np.eye(3)):
            return k
    raise AssertionError("not a crystallographic rotation")


def _token(group, direction, primary):
    """What the group has along ``direction``, written as Hermann–Mauguin writes it."""
    proper, improper, mirror, glide = {}, set(), False, False
    for rotation, shift in zip(group.rotations, group.translations):
        if np.linalg.det(rotation) > 0:
            if not np.allclose(rotation, np.eye(3)) and np.allclose(rotation @ direction,
                                                                       direction):
                proper.setdefault(_order(rotation), []).append((rotation, shift))
        elif np.allclose(-rotation @ direction, direction):
            improper.add(_order(-rotation))
            if _order(-rotation) == 2:
                mirror = True
                glide = glide or (not primary and abs((shift + 0.5) % 1 - 0.5) > 1e-9)
    n = max(proper) if proper else 1
    turn = str(n)
    if primary and n > 1:
        # The screw's subscript, from the turn by +360°/n about the chain.
        subscripts = set()
        for rotation, shift in proper[n]:
            turned = rotation @ np.array([0, 1.0, 0])
            if abs(np.degrees(np.arctan2(turned[2], turned[1])) % 360 - 360 / n) < 1e-6:
                subscripts.add(round((shift % 1.0) * n) % n)
        if 0 not in subscripts:
            turn = f"{n}_{subscripts.pop()}"
    plane = ("a" if glide else "m") if mirror else None
    if primary:
        if n == 3 and mirror:
            return "-6"
        if mirror and n > 1:
            return f"{turn}/{plane}"
        if n == 3 and 3 in improper:
            return "-3"
        if n == 2 and 4 in improper:
            return "-4"
        if n == 1 and 1 in improper and group.order == 2:
            return "-1"
        return plane or turn
    if n > 1:
        return f"{turn}/{plane}" if plane else turn
    return plane or "1"


def test_every_group_reads_as_the_manual_names_it():
    import re

    for number, symbol, _setting, space_group in rg.TABLE:
        group = rg.rod_group(number)
        manual = re.findall(r"-?\d(?:_\d)?(?:/[ma])?|[ma]", symbol[1:])
        mine = [_token(group, direction, index == 0)
                for index, direction in enumerate(_directions(space_group))]
        for ours, theirs in zip(mine, manual):
            # A short symbol leaves out the twofold of a 2/m: Pmmm for P2/m2/m2/m.
            assert ours == theirs or ours.endswith("/" + theirs), (number, symbol, mine)
