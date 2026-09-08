"""Crystalline orbitals: parsing CRYSTAL's ORBITALS Molden files, and evaluating them.

The evaluation has no visible failure mode — a permuted harmonic or a shell
scaled wrong still draws a plausible orbital — so it is pinned on quantities that
must come out exact: a single atomic orbital integrates to 1, and the orbitals of
one file come out mutually orthonormal.
"""

import os
import textwrap

import numpy as np
import pytest

from crystalline.core import orbitals as orb
from crystalline.crystalio import molden

# A real ORBITALS run, when one is on this machine.
_RUN = os.path.expanduser("~/Desktop/PyCrystal/corundum_orbitals")
_GAMMA = os.path.join(_RUN, "corundum_00001_K000_real.molden")

requires_run = pytest.mark.skipif(
    not os.path.isfile(_GAMMA), reason="the reference ORBITALS run is not on this machine"
)

_MINIMAL = textwrap.dedent("""\
    [Molden Format]
    [Cell] (Angs)
       4.0 4.0 4.0 90.0 90.0 90.0
    [CellAxes] (Angs)
        4.00000000    0.00000000    0.00000000
        0.00000000    4.00000000    0.00000000
        0.00000000    0.00000000    4.00000000
    [Atoms] (Fractional)
    H      1     1      0.0000000000E+00    0.0000000000E+00    0.0000000000E+00
    H      2     1      0.5000000000E+00    0.5000000000E+00    0.0000000000E+00
    [5D]
    [7F]
    [GTO]
       1
    s       1
          1.00000000        0.71270547

       2
    s       1
          1.00000000        0.71270547

    [MO]
     Sym   = 1A
     Ene   =      -0.500000
     Spin  = ALPHA+BETA
     Occup =  2.0000
        1   0.7000000E+00
        2   0.7000000E+00
     Sym   = 1A
     Ene   =       0.200000
     Spin  = ALPHA+BETA
     Occup =  0.0000
        1   0.7000000E+00
        2  -0.7000000E+00
    """)


@pytest.fixture
def minimal(tmp_path):
    path = tmp_path / "toy_00001_K000_real.molden"
    path.write_text(_MINIMAL)
    return str(path)


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


# ── parsing ───────────────────────────────────────────────────────────────
def test_parses_geometry_basis_and_orbitals(minimal):
    data = molden.load(minimal)

    assert list(data.numbers) == [1, 1]
    assert data.cell == pytest.approx(np.eye(3) * 4.0)
    assert data.positions[1] == pytest.approx([2.0, 2.0, 0.0])  # fractional resolved
    assert data.spherical_d and data.spherical_f
    assert len(data.shells) == 2 and data.n_ao == 2
    assert [mo.occupation for mo in data.orbitals] == [2.0, 0.0]
    assert data.orbitals[0].energy == pytest.approx(-0.5)
    assert data.homo_index == 0


def test_the_file_name_carries_the_k_point_and_which_part_it_is(minimal):
    data = molden.load(minimal)
    assert (data.kpoint_label, data.part) == ("000", "real")
    assert molden.k_label("run_00003_K110_complex.molden") == "110"


def test_a_non_molden_file_is_refused(tmp_path):
    path = tmp_path / "not.molden"
    path.write_text("just some text\n")
    with pytest.raises(ValueError):
        molden.load(str(path))


def test_an_sp_shell_contributes_four_orbitals_sharing_its_exponents():
    """Pople's combined shell: one s and one p over the same primitives, with the
    coefficients in two columns."""
    shell = molden.Shell(
        atom=0, kind="sp",
        exponents=np.array([1.0, 0.5]),
        coefficients=np.array([[0.3, 0.4], [0.5, 0.6]]),
    )
    assert shell.size == 4
    values = orb._shell_amplitudes(shell, np.zeros((1, 3)))
    assert len(values) == 4
    assert values[0][0] != 0.0                     # s survives at the origin
    assert all(v[0] == 0.0 for v in values[1:])    # p vanishes there


# ── the maths that has no visible failure mode ────────────────────────────
def _self_overlap(shell, component, reach, samples=150):
    axis = np.linspace(-reach, reach, samples)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    amplitude = orb._shell_amplitudes(shell, grid)[component]
    return float((amplitude ** 2).sum() * (axis[1] - axis[0]) ** 3)


@pytest.mark.parametrize(
    "kind, component", [("s", 0), ("p", 1), ("d", 0), ("d", 3), ("f", 0), ("f", 5)]
)
def test_one_atomic_orbital_integrates_to_one(kind, component):
    """Molden folds each primitive's normalisation into its coefficient, so a
    single atomic orbital must come out normalised as written.

    This is what pins the ``1/sqrt((2l-1)!!)`` on the d and f harmonics. Without
    it a d orbital comes out sqrt(3) too large *relative to s and p*, which does
    not merely rescale the picture — it changes the shape of every orbital that
    mixes them.
    """
    alpha = 0.8
    order = {"s": 0, "p": 1, "d": 2, "f": 3}[kind]
    norm = (2 * alpha / np.pi) ** 0.75 * (4 * alpha) ** (order / 2.0)
    shell = molden.Shell(
        atom=0, kind=kind, exponents=np.array([alpha]), coefficients=np.array([norm])
    )
    assert _self_overlap(shell, component, reach=7.0) == pytest.approx(1.0, abs=2e-3)


def test_every_component_of_a_shell_shares_one_normalisation():
    """The five d (and seven f) components must be scaled alike, or the lobes of
    a single orbital come out with the wrong relative sizes."""
    for kind, count in (("d", 5), ("f", 7)):
        shell = molden.Shell(
            atom=0, kind=kind, exponents=np.array([0.7]), coefficients=np.array([1.0])
        )
        values = [_self_overlap(shell, k, reach=7.0) for k in range(count)]
        assert max(values) / min(values) == pytest.approx(1.0, abs=1e-3)


# ── against a real run ────────────────────────────────────────────────────
@requires_run
def test_the_reference_run_parses_consistently():
    data = molden.load(_GAMMA)
    assert len(data.numbers) == 10                      # corundum's primitive cell
    assert data.n_ao == len(data.orbitals[0].coefficients) == 180
    assert sum(mo.is_occupied for mo in data.orbitals) == 50
    assert np.linalg.norm(data.cell, axis=1) == pytest.approx([5.012737] * 3, abs=1e-5)


@requires_run
def test_orbitals_of_one_file_come_out_orthonormal():
    """The end-to-end check. Normalisation, harmonic ordering, the sp split and
    the Bloch lattice sum all have to be right for this to hold — a permuted
    harmonic destroys the off-diagonals."""
    data = molden.load(_GAMMA)
    chosen = [40, data.homo_index, data.homo_index + 1]
    fields = {i: orb.evaluate(data, i, samples=70) for i in chosen}

    reference = fields[chosen[0]]
    axes = [
        reference.origin[k] + reference.spacing[k] * np.arange(reference.shape[k])
        for k in range(3)
    ]
    points = np.stack([m.ravel() for m in np.meshgrid(*axes, indexing="ij")], axis=1)
    fractional = points @ np.linalg.inv(data.cell)
    inside = np.all((fractional >= 0.0) & (fractional < 1.0), axis=1)
    voxel = float(np.prod(reference.spacing))

    for i in chosen:
        for j in chosen:
            overlap = float(
                (fields[i].values.ravel()[inside] * fields[j].values.ravel()[inside]).sum()
                * voxel
            )
            assert overlap == pytest.approx(1.0 if i == j else 0.0, abs=0.02)


@requires_run
def test_the_orbital_continues_across_the_cell_boundary():
    """The Bloch sum is what makes it crystalline: without it the orbital would
    die at the cell face instead of continuing into the neighbouring cell, which
    is a molecule's orbital, not a crystal's."""
    data = molden.load(_GAMMA)
    field = orb.evaluate(data, data.homo_index, samples=60)

    peak = np.abs(field.values).max()
    faces = [
        np.abs(field.values[0, :, :]).max(), np.abs(field.values[-1, :, :]).max(),
        np.abs(field.values[:, 0, :]).max(), np.abs(field.values[:, -1, :]).max(),
        np.abs(field.values[:, :, 0]).max(), np.abs(field.values[:, :, -1]).max(),
    ]
    assert min(faces) / peak > 0.05


@requires_run
def test_a_supercell_spans_a_larger_box_and_stays_normalised_per_cell():
    """Asking for more of an orbital than one cell holds spans a supercell, and
    it is re-evaluated over the bigger box rather than the single-cell result
    being repeated: away from Gamma psi picks up a phase from cell to cell."""
    data = molden.load(_GAMMA)
    one = orb.evaluate(data, data.homo_index, samples=50)
    two = orb.evaluate(data, data.homo_index, samples=50, repeat=(2, 1, 1))

    # The clipped region really is the doubled cell...
    assert np.linalg.norm(two.cell[0]) == pytest.approx(2 * np.linalg.norm(one.cell[0]))
    assert abs(np.linalg.det(two.cell)) == pytest.approx(2 * abs(np.linalg.det(one.cell)))
    # ...and the sampled box grew to hold it. Not by 2x along x: the box is the
    # cell's bounding box, and on a rhombohedral cell the other two vectors
    # contribute negative x, so doubling a widens it by less than a's own length.
    single_span = one.spacing * (np.asarray(one.shape) - 1)
    double_span = two.spacing * (np.asarray(two.shape) - 1)
    assert np.prod(double_span) > np.prod(single_span) * 1.5
    assert double_span[0] > single_span[0]
    # normalised over one cell either way, so an isovalue means the same thing
    assert orb.integrate_density(one, data.cell) == pytest.approx(1.0, abs=0.02)


@requires_run
def test_a_modulus_is_non_negative_and_still_normalised():
    """Away from Gamma the two files combine into |psi|, which has no sign — so
    it is drawn as a single surface rather than a pair of lobes. It has to be
    asked for: the signed amplitude is the default, being the one that shows the
    phase relation between cells."""
    real_path = os.path.join(_RUN, "corundum_00002_K100_real.molden")
    real = molden.load(real_path)
    imaginary = molden.load(molden.companion_part(real_path))
    k = molden.k_vector(real_path)

    field = orb.evaluate(real, real.homo_index, samples=60, imaginary=imaginary,
                         kpoint=k, draw=orb.MODULUS)

    assert field.modulus is True
    assert field.values.min() >= 0.0
    assert orb.integrate_density(field, real.cell) == pytest.approx(1.0, abs=0.02)

    amplitude = orb.evaluate(real, real.homo_index, samples=60, imaginary=imaginary,
                             kpoint=k)
    assert amplitude.modulus is False
    assert amplitude.values.min() < 0.0  # ± lobes


# ── the Bloch phase ───────────────────────────────────────────────────────
def test_the_orbital_carries_the_bloch_phase_from_cell_to_cell(minimal):
    """The defining property: ``psi(r + T) = e^(2*pi*i*k.T) psi(r)``.

    Stepping one cell along **a** at k = (1/3, 0, 0) turns the orbital by 120°,
    so the field there is what the *same* point of the first cell gives when the
    section is turned by 120° instead. That is checked directly, which pins both
    that the phase is applied and that it is applied with the right sign and
    period.

    It is the test that fails against a lattice sum with no phase in it: such a
    sum is real and lattice-periodic, so ``psi(r + a)`` would come back equal to
    ``psi(r)`` and a turned section would be a flat ``cos(theta)`` rescaling of
    it.

    The cubic toy cell is used rather than the real run because the sampled box
    is then axis-aligned, and "one cell along a" is a whole number of grid steps.
    """
    data = molden.load(minimal)
    k = np.array([1.0 / 3.0, 0.0, 0.0])
    turn = 2.0 * np.pi / 3.0          # the phase one step along a picks up

    def field(draw=orb.AMPLITUDE, **options):
        return orb.evaluate(data, 0, samples=61, repeat=(3, 1, 1), kpoint=k,
                            draw=draw, **options)

    here = field()
    step = int(round(4.0 / here.spacing[0]))   # grid steps in one 4 Å cell

    assert here.values[step:] == pytest.approx(field(phase=turn).values[:-step], abs=1e-9)
    # ...and it really is a phase, not a scaling: the field is not periodic
    assert not np.allclose(here.values[step:], here.values[:-step], atol=1e-3)
    # while its modulus is — which is why the modulus shows no phase relation
    modulus = field(draw=orb.MODULUS)
    assert modulus.values[step:] == pytest.approx(modulus.values[:-step], abs=1e-9)
    # the turn compounds: two cells along a is two turns of 120°
    assert here.values[2 * step:] == pytest.approx(
        field(phase=2 * turn).values[:-2 * step], abs=1e-9
    )
    # and three of them is a whole period, which is the identity
    assert field(phase=3 * turn).values == pytest.approx(here.values, abs=1e-9)


def test_the_phase_view_is_one_flat_colour_per_cell(minimal):
    """What makes the relation legible: ``|psi|`` is identical in every cell, so
    the colour is the only thing that changes — and it steps by exactly
    ``2*pi*k.T``, holding constant across each cell rather than drifting."""
    data = molden.load(minimal)
    k = np.array([1.0 / 3.0, 0.0, 0.0])

    field = orb.evaluate(data, 0, samples=61, repeat=(3, 1, 1), kpoint=k, draw=orb.PHASE)

    assert field.modulus is True
    assert field.values.min() >= 0.0
    step = int(round(4.0 / field.spacing[0]))     # grid steps in one 4 Å cell
    # the surface is lattice-periodic: that is the whole point of drawing |psi|
    assert field.values[step:] == pytest.approx(field.values[:-step], abs=1e-9)

    # three cells along a, three flat phases, 120° apart and in order
    phases = field.phases
    for cell in range(3):
        block = phases[cell * step + 2: (cell + 1) * step - 2]
        assert np.ptp(block) == pytest.approx(0.0, abs=1e-12), "not flat within a cell"
    seen = [float(phases[cell * step + step // 2, 0, 0]) for cell in range(3)]
    turns = np.exp(1j * np.diff(seen))
    assert turns == pytest.approx(np.full(2, np.exp(2j * np.pi / 3)), abs=1e-9)


def test_the_phase_view_falls_back_to_one_colour_without_a_k(minimal):
    """No k means no known relation, so nothing is asserted about one: the field
    still draws, in a single phase, rather than inventing a Gamma-like pattern."""
    data = molden.load(minimal)

    field = orb.evaluate(data, 0, samples=25, repeat=(2, 1, 1), kpoint=None,
                         draw=orb.PHASE, imaginary=None)

    assert field.phases is None or np.ptp(field.phases) == pytest.approx(0.0)


def test_a_real_orbital_ignores_the_draw_mode(minimal):
    """At Gamma there is no imaginary part, so the modulus and phase views have
    nothing the signed one doesn't. They collapse onto it rather than drawing a
    sign-less blob that hides the nodes for no gain."""
    data = molden.load(minimal)

    signed = orb.evaluate(data, 0, samples=25, draw=orb.AMPLITUDE)
    for mode in (orb.MODULUS, orb.PHASE):
        other = orb.evaluate(data, 0, samples=25, draw=mode)
        assert other.values == pytest.approx(signed.values)
        assert other.modulus is False
        assert other.phases is None


def test_a_tiled_orbital_is_sampled_as_finely_per_cell_as_a_single_one(minimal):
    """The grid density is per cell, not per picture.

    Left as a fixed 60x60x60 whatever the box, the very tiling asked for to see
    the phase relation would be what made the orbital too coarse to read: three
    cells along **a** would put the same points over three times the distance.
    """
    data = molden.load(minimal)

    one = orb.evaluate(data, 0, samples=41)
    three = orb.evaluate(data, 0, samples=41, repeat=(3, 1, 1))

    assert three.spacing == pytest.approx(one.spacing)
    assert three.shape[0] == 3 * (one.shape[0] - 1) + 1
    assert three.shape[1:] == one.shape[1:]


def test_a_very_large_grid_is_capped_rather_than_allocated(minimal):
    """A fine density over a big tiling would otherwise ask for tens of
    gigabytes. The density drops uniformly; the box asked for is still covered."""
    data = molden.load(minimal)

    field = orb.evaluate(data, 0, samples=120, repeat=(6, 6, 6))

    assert np.prod(field.shape) <= orb._MAX_GRID_POINTS
    extent = field.spacing * (np.asarray(field.shape) - 1)
    assert extent == pytest.approx(np.full(3, 6 * 4.0))  # still the whole 6x6x6 box


def test_gamma_needs_no_phase_and_takes_the_real_path(minimal):
    """At Gamma every image adds in step, so k changes nothing — and the
    evaluation stays on the cheaper real-only path."""
    data = molden.load(minimal)

    without = orb.evaluate(data, 0, samples=25)
    with_gamma = orb.evaluate(data, 0, samples=25, kpoint=np.zeros(3))

    assert with_gamma.values == pytest.approx(without.values)
    assert with_gamma.modulus is False


@requires_run
def test_the_k_point_is_read_against_the_net_the_orbitals_came_from():
    """The file names carry integer coordinates, not k: ``K100`` is ``(1 0 0)``
    in units of a shrinking factor that only the output states. This run does
    SCF on a 6x6x6 net and NEWK on a 3x3x3 one, and *both* tables call their
    second point ``(1 0 0)`` — so reading a single file against the first table
    found halves every k. The table has to be the one that accounts for all six
    file names.
    """
    labels = {
        "corundum_00001_K000_real.molden": (0.0, 0.0, 0.0),
        "corundum_00002_K100_real.molden": (1 / 3, 0.0, 0.0),
        "corundum_00003_K110_real.molden": (1 / 3, 1 / 3, 0.0),
        "corundum_00004_K210_real.molden": (2 / 3, 1 / 3, 0.0),
        "corundum_00005_K111_real.molden": (1 / 3, 1 / 3, 1 / 3),
        "corundum_00006_K211_real.molden": (2 / 3, 1 / 3, 1 / 3),
    }
    for name, expected in labels.items():
        assert molden.k_vector(os.path.join(_RUN, name)) == pytest.approx(expected)


def test_an_unaccompanied_molden_file_has_no_k_vector(tmp_path):
    """No output beside it, so the shrinking factor is unknowable. Γ is still
    Γ — it is the origin whatever the factor is — but nothing else is, and
    guessing would draw a different orbital that looks perfectly plausible."""
    (tmp_path / "run_00002_K100_real.molden").write_text(_MINIMAL)
    (tmp_path / "run_00001_K000_real.molden").write_text(_MINIMAL)

    assert molden.k_vector(str(tmp_path / "run_00002_K100_real.molden")) is None
    assert molden.k_vector(str(tmp_path / "run_00001_K000_real.molden")) == pytest.approx(
        np.zeros(3)
    )


def test_the_tiling_for_a_period_is_measured_in_the_cell_on_screen():
    """k is quoted against the file's own (primitive) lattice, and the view is
    usually showing a different cell. A hexagonal cell of a rhombohedral crystal
    holds three primitive cells, so it already spans a whole period of
    k = (1/3, 1/3, 1/3) — where the untransformed answer asks for 3x3x3 cells,
    twenty-seven times the structure for no more information.
    """
    primitive = np.array([[2.7149, 0.0, 4.2139],
                          [-1.3574, 2.3511, 4.2139],
                          [-1.3574, -2.3511, 4.2139]])
    # the hexagonal cell of that rhombohedron: (-a+b, a-c, a+b+c)
    transform = np.array([[-1, 1, 0], [1, 0, -1], [1, 1, 1]], dtype=float)
    hexagonal = transform @ primitive

    third = np.array([1 / 3, 1 / 3, 1 / 3])
    assert orb.commensurate_repeats_in(third, primitive, hexagonal) == (1, 1, 1)
    assert orb.commensurate_repeats_in(third, primitive, primitive) == (3, 3, 3)
    assert orb.commensurate_repeats_in(third, primitive, None) == (3, 3, 3)
    assert orb.commensurate_repeats_in(None, primitive, hexagonal) == (1, 1, 1)

    two_thirds = np.array([2 / 3, 1 / 3, 0.0])
    assert orb.commensurate_repeats_in(two_thirds, primitive, hexagonal) == (3, 3, 1)


@requires_run
def test_a_run_offers_its_k_points_and_pairs_real_with_imaginary():
    files = molden.find_orbital_files(_GAMMA)
    # 22 files on disk, but the run writes the same set twice under two stems
    assert len(files) == 11

    gamma = [f for f in files if molden.k_label(f) == "000"]
    assert len(gamma) == 1  # Gamma is real, so it has no imaginary companion
    assert molden.companion_part(gamma[0]) is None

    away = next(f for f in files if molden.k_label(f) == "100" and "real" in f)
    assert molden.companion_part(away) is not None  # k != 0 is complex


# ── the orbital must line up with the cell it is drawn over ───────────────
@requires_run
def test_the_isosurface_is_clipped_to_the_cell():
    """Regression: lobes were drawn out in the corners of the sampled box.

    The box is the cell's *bounding* box — a rhombohedral cell fills only about a
    quarter of it — so what was drawn out there belonged to neighbouring cells
    whose atoms are not on screen, and read as lobes floating in empty space.
    """
    pv = pytest.importorskip("pyvista")
    from crystalline.viz import renderer as viz

    data = molden.load(_GAMMA)
    field = orb.evaluate(data, data.homo_index, samples=70)

    grid = pv.ImageData()
    grid.dimensions = field.shape
    grid.origin = tuple(float(v) for v in field.origin)
    grid.spacing = tuple(float(v) for v in field.spacing)
    grid.point_data["psi"] = field.values.ravel(order="F")
    surface = grid.contour([0.2 * field.peak], scalars="psi")
    assert surface.n_points > 0

    clipped = viz._clip_to_cell(surface, field.cell)

    assert 0 < clipped.n_points < surface.n_points   # something survived, some went
    fractional = np.asarray(clipped.points) @ np.linalg.inv(np.asarray(field.cell))
    assert fractional.min() > -0.02 and fractional.max() < 1.02


def test_the_phase_colour_is_sampled_as_a_step_not_interpolated():
    """A per-cell phase is a step function. Carried through VTK's contour filter
    as point data it would be interpolated, and every cell boundary a lobe
    straddles would come out as a band sweeping the whole colour wheel — so it
    is looked up at the nearest grid point instead, and only the two phases
    actually present may appear on the surface.

    The step is placed between two grid planes and across the cap of the blob,
    where the surface really does cross grid edges spanning it — the case
    interpolation gets wrong. A step lying on a grid plane, or cutting the blob
    at its equator, is crossed only by edges whose two ends share a phase, and
    would pass either way.
    """
    import pyvista as pv

    from crystalline.viz.renderer import StructureRenderer

    axis = np.linspace(0.0, 10.0, 41)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    blob = np.exp(-((x - 5.0) ** 2 + (y - 5.0) ** 2 + (z - 5.0) ** 2) / 4.0)
    phases = np.where(x < 6.6, 0.0, 2.0)          # between grid planes, across the cap
    field = orb.OrbitalField(
        values=blob, origin=np.zeros(3), spacing=np.full(3, 0.25),
        shape=blob.shape, modulus=True, cell=np.eye(3) * 10.0, phases=phases,
    )

    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_orbital(field, 0.5)

    assert len(renderer._orbital_actors) == 1     # one |psi| surface, not two lobes
    surface = renderer._orbital_actors[0].mapper.dataset
    drawn = np.unique(np.asarray(surface.point_data["phase"]))
    assert set(np.round(drawn, 9)) == {0.0, 2.0}


def test_clipping_leaves_a_surface_alone_without_a_cell():
    """A degenerate or absent cell defines no faces; the orbital is kept whole
    rather than lost."""
    pv = pytest.importorskip("pyvista")
    from crystalline.viz import renderer as viz

    sphere = pv.Sphere(radius=1.0)
    assert viz._clip_to_cell(sphere, None) is sphere
    assert viz._clip_to_cell(sphere, np.zeros((3, 3))) is sphere


@requires_run
def test_an_orbital_can_be_drawn_over_the_crystallographic_cell():
    """The conventional cell is a different *region* of the same crystal, not a
    different crystal, and the Bloch sum runs over the file's own lattice either
    way — so an orbital defined on a primitive cell can be sampled over the
    conventional one, which is what the view shows by default.
    """
    from crystalline.core.cells import to_conventional
    from crystalline.core.structure import Structure
    from ase import Atoms

    data = molden.load(_GAMMA)
    primitive = Structure.from_ase(
        Atoms(numbers=data.numbers, positions=data.positions, cell=data.cell, pbc=True)
    )
    conventional = to_conventional(primitive)
    assert len(conventional) == 3 * len(primitive)  # rhombohedral -> hexagonal

    field = orb.evaluate(
        data, data.homo_index, samples=70, box_cell=conventional.cell
    )

    # it is clipped to the conventional cell, not the file's own
    assert abs(np.linalg.det(field.cell)) == pytest.approx(
        abs(np.linalg.det(conventional.cell)), rel=1e-6
    )
    # and the orbital is on the oxygens, not smeared over the box: this is the
    # check that would fail if the cell and the orbital had drifted apart
    def amplitude_at(point):
        index = [int(round((point[k] - field.origin[k]) / field.spacing[k]))
                 for k in range(3)]
        if any(i < 0 or i >= field.shape[k] for k, i in enumerate(index)):
            return None
        return abs(field.values[tuple(index)])

    oxygens = [p for p, z in zip(conventional.positions, conventional.numbers) if z == 8]
    on_atoms = [v for v in (amplitude_at(p) for p in oxygens) if v is not None]
    rng = np.random.default_rng(0)
    low = field.origin
    high = field.origin + field.spacing * (np.asarray(field.shape) - 1)
    anywhere = [amplitude_at(low + rng.random(3) * (high - low)) for _ in range(300)]
    anywhere = [v for v in anywhere if v is not None]

    assert len(on_atoms) > 10
    assert np.mean(on_atoms) > 3.0 * np.mean(anywhere)


@requires_run
def test_showing_an_orbital_leaves_the_cell_view_alone(qapp):
    """Regression, twice over.

    The orbital first came out shifted because the view pipeline expanded the
    file's rhombohedral cell to the hexagonal conventional one while the orbital
    stayed behind. Forcing the view to the primitive cell fixed that but took
    away the crystallographic cell, which is the one worth looking at — so the
    orbital is sampled over whatever cell is displayed instead, and the view is
    left alone.

    The supercell must survive too: it is how you ask to see more of an orbital.
    """
    from crystalline.core.cells import CellView
    from crystalline.ui.main_window import MainWindow

    data = molden.load(_GAMMA)
    applied = []

    class _StubWindow:
        _show_orbital_structure = MainWindow._show_orbital_structure

        def _apply_cell_view(self):
            applied.append(True)

        def _update_view_actions(self):
            pass

    window = _StubWindow()
    window._cell_view = CellView.CRYSTALLOGRAPHIC
    window._supercell = (2, 1, 1)
    window.phonon_panel = type("P", (), {"clear": staticmethod(lambda: None)})()

    window._show_orbital_structure(data)

    assert window._cell_view is CellView.CRYSTALLOGRAPHIC   # not forced away
    assert window._supercell == (2, 1, 1)                   # left alone
    assert len(window._source) == len(data.numbers)         # the molden's own cell
    assert window._source.cell == pytest.approx(data.cell)
    assert applied                                          # rebuilt through the pipeline


@requires_run
def test_tiling_to_a_period_is_measured_against_the_displayed_cell(qapp):
    """The window's half of the tiling: the dialog only asks for it.

    The number of cells depends on the cell on screen, so it cannot be settled
    in the dialog — and when the displayed cell already spans a period, tiling
    is declined rather than applied as a no-op.
    """
    from crystalline.ui.main_window import MainWindow

    data = molden.load(_GAMMA)
    primitive = np.asarray(data.cell)
    hexagonal = np.array([[-1, 1, 0], [1, 0, -1], [1, 1, 1]], dtype=float) @ primitive
    applied = []

    class _StubWindow:
        _tile_to_period = MainWindow._tile_to_period
        _tile_restore = (1, 1, 1)

        def _set_supercell(self, reps):
            self._supercell = reps

        def _apply_cell_view(self):
            applied.append(self._supercell)

    window = _StubWindow()
    window._supercell = (1, 1, 1)
    window._unit_cell = hexagonal

    # the hexagonal cell already holds a whole period of this k: nothing to do
    assert window._tile_to_period(np.array([1 / 3, 1 / 3, 1 / 3]), primitive) is None
    assert not applied
    assert window._supercell == (1, 1, 1)

    assert window._tile_to_period(np.array([2 / 3, 1 / 3, 0.0]), primitive) == (3, 3, 1)
    assert applied == [(3, 3, 1)]
    assert window._tile_restore is None   # the tiling is the user's, not the panel's


@requires_run
def test_the_orbital_is_rebuilt_over_the_cell_on_screen(qapp):
    """What keeps the two from drifting apart: the field is always evaluated over
    the displayed cell, whatever the view and supercell have made it.

    ``_run_busy`` is the seam. In the app it moves the evaluation to a thread and
    draws the result back on the UI thread; here it runs straight through, so the
    test is about *what* is computed and drawn, not about the threading.
    """
    from crystalline.ui.main_window import MainWindow

    data = molden.load(_GAMMA)
    captured = {}

    class _StubWindow:
        _refresh_orbital = MainWindow._refresh_orbital

        def _run_busy(self, work, message, done, failed_title):
            captured["message"] = message
            done(work())

        def _update_orbital_actions(self):
            pass

    window = _StubWindow()
    window._orbital = {
        "data": data, "imaginary": None,
        "index": data.homo_index, "isovalue": 0.2, "samples": 30,
    }
    window._unit_cell = data.cell * 2.0     # stand in for a conventional cell
    window._supercell = (1, 1, 1)
    window.viewport = type("V", (), {"renderer": type("R", (), {
        "set_orbital": staticmethod(lambda field, iso: captured.update(field=field, iso=iso))
    })()})()

    assert window._refresh_orbital() is True
    assert captured["iso"] == 0.2
    assert "orbital" in captured["message"].lower()   # the overlay says what it is doing
    # sampled over the cell it was told about, not the file's own
    assert abs(np.linalg.det(captured["field"].cell)) == pytest.approx(
        abs(np.linalg.det(data.cell * 2.0)), rel=1e-6
    )


# ── the picker ────────────────────────────────────────────────────────────
@requires_run
def test_the_dialog_offers_one_entry_per_k_point_not_per_file(qapp):
    """A run writes a ``_real`` and a ``_complex`` file per k away from Gamma, but
    those are two halves of the same orbitals — offering both as separate
    k-points would invite plotting half an orbital."""
    from crystalline.ui.panels.orbital_dialog import OrbitalDialog

    dialog = OrbitalDialog(molden.find_orbital_files(_GAMMA))

    assert dialog.kpoint.count() == 6            # Gamma plus five sampled k
    assert dialog.kpoint.itemText(0) == "Γ"
    assert "complex" in dialog.kpoint.itemText(1)


@requires_run
def test_the_dialog_lands_on_the_homo(qapp):
    """"The orbital just below the gap" is what someone is usually after, and
    counting to it through 180 rows is the tedious way."""
    from crystalline.ui.panels.orbital_dialog import OrbitalDialog

    dialog = OrbitalDialog(molden.find_orbital_files(_GAMMA))

    assert dialog.orbitals.count() == 180
    assert dialog.orbitals.currentRow() == molden.load(_GAMMA).homo_index
    assert "HOMO" in dialog.orbitals.currentItem().text()


@requires_run
def test_the_draw_and_phase_options_only_apply_where_the_orbital_is_complex(qapp):
    """At Gamma the orbital is real: there is nothing for the other two views to
    show and no phase to turn. Away from Gamma they apply — and the companion
    file is needed for all three, since every one of them is built from the two
    halves."""
    from crystalline.ui.panels.orbital_dialog import OrbitalDialog

    dialog = OrbitalDialog(molden.find_orbital_files(_GAMMA))

    assert not dialog.draw.isEnabled()                          # Gamma
    assert not dialog.phase.isEnabled()
    assert dialog.selection()["draw"] == orb.AMPLITUDE
    assert dialog.selection()["imaginary_path"] is None

    dialog.kpoint.setCurrentIndex(1)                            # k != 0
    assert dialog.draw.isEnabled()
    assert dialog.phase.isEnabled()
    assert dialog.selection()["imaginary_path"] is not None
    # the signed amplitude is the default: the ± lobes are what an orbital is
    # usually being looked at for
    assert dialog.selection()["draw"] == orb.AMPLITUDE

    dialog.draw.setCurrentIndex(dialog.draw.findData(orb.MODULUS))
    assert dialog.selection()["draw"] == orb.MODULUS
    assert dialog.selection()["imaginary_path"] is not None     # still both halves


@requires_run
def test_a_choice_forced_at_gamma_is_given_back_away_from_it(qapp):
    """Regression: Gamma has no phase to draw and no period to tile to, so both
    controls are forced there. Leaving them forced on the way back out made
    moving between k-points look like it had silently changed the view."""
    from crystalline.ui.panels.orbital_dialog import OrbitalDialog

    dialog = OrbitalDialog(molden.find_orbital_files(_GAMMA))

    dialog.kpoint.setCurrentIndex(1)
    assert dialog.selection()["draw"] == orb.AMPLITUDE          # the default

    dialog.draw.setCurrentIndex(dialog.draw.findData(orb.PHASE))
    dialog.kpoint.setCurrentIndex(0)                # Gamma has no phase to draw
    assert dialog.selection()["draw"] == orb.AMPLITUDE
    dialog.kpoint.setCurrentIndex(1)
    assert dialog.selection()["draw"] == orb.PHASE              # ...and gives it back


@requires_run
def test_the_dialog_offers_tiling_only_away_from_gamma(qapp):
    """One cell shows the whole orbital at Gamma; away from it a period spans
    several, exactly as for a phonon at the same q."""
    from crystalline.ui.panels.orbital_dialog import OrbitalDialog

    dialog = OrbitalDialog(molden.find_orbital_files(_GAMMA))

    assert not dialog.tile.isEnabled()
    assert dialog.selection()["tile"] is False

    dialog.kpoint.setCurrentIndex(1)
    assert dialog.tile.isEnabled()
    dialog.tile.setChecked(True)
    assert dialog.selection()["tile"] is True

    dialog.kpoint.setCurrentIndex(0)     # back to Gamma: the request lapses
    assert dialog.selection()["tile"] is False


@requires_run
def test_the_dialog_has_no_margin_control_and_defaults_to_a_fifth(qapp):
    from crystalline.ui.panels.orbital_dialog import OrbitalDialog

    dialog = OrbitalDialog(molden.find_orbital_files(_GAMMA))

    assert not hasattr(dialog, "padding")
    assert dialog.isovalue.value() == pytest.approx(0.2)
    assert sorted(dialog.selection()) == [
        "draw", "imaginary_path", "index", "isovalue", "kpoint", "path",
        "phase", "samples", "tile",
    ]
