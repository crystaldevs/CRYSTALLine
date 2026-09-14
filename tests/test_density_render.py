"""Drawing a charge density, a spin density or a potential in the 3D view."""

import numpy as np
import pytest

pytest.importorskip("pyvista")

from crystalline.crystalio import density as D  # noqa: E402

_PLOTTERS = []


def _plotter():
    """An off-screen plotter that is closed when the test ends.

    Left to the garbage collector, a render window is torn down at whatever
    moment the collector runs — often in the middle of some unrelated later
    test — and on macOS that teardown can take the interpreter with it.
    """
    import pyvista as pv

    plotter = pv.Plotter(off_screen=True)
    _PLOTTERS.append(plotter)
    return plotter


@pytest.fixture(autouse=True)
def _close_plotters():
    yield
    while _PLOTTERS:
        try:
            _PLOTTERS.pop().close()
        except Exception:  # noqa: BLE001 - closing is best effort
            pass
from crystalline.viz.renderer import _structured_grid  # noqa: E402


def _hexagonal_blob(n=15, sigma=0.6):
    """A Gaussian at the centre of a *sheared* cell.

    Hexagonal on purpose: an axis-aligned grid draws this one wrong, and wrong
    in a way that still looks like a plausible blob.
    """
    steps = np.array([[0.2, 0.0, 0.0], [-0.1, 0.173, 0.0], [0.0, 0.0, 0.25]])
    shape = (n, n, n)
    field = D.ScalarField(values=np.zeros(shape), origin=np.zeros(3), steps=steps)
    points = field.points()
    centre = points[n // 2, n // 2, n // 2]
    squared = ((points - centre) ** 2).sum(axis=-1)
    return D.replace(field, values=np.exp(-squared / sigma ** 2)), centre


def _renderer():
    import pyvista as pv

    from crystalline.viz.renderer import StructureRenderer

    return StructureRenderer(_plotter())


def test_the_grid_keeps_each_value_at_its_own_point_in_a_sheared_cell():
    """VTK varies the first index fastest; numpy's default reshape does not."""
    field, centre = _hexagonal_blob()
    n = field.shape[0]

    grid = _structured_grid(field)

    flat = (n // 2) + n * (n // 2) + n * n * (n // 2)
    assert grid.n_points == n ** 3
    assert np.allclose(grid.points[flat], centre)
    assert np.isclose(grid.point_data["value"][flat], 1.0)


def test_the_isosurface_sits_where_the_density_is_and_not_on_an_axis():
    field, centre = _hexagonal_blob()

    surface = _structured_grid(field).contour([0.5], scalars="value")

    assert surface.n_points > 0
    assert np.allclose(surface.points.mean(axis=0), centre, atol=0.05)


def test_a_charge_density_gets_one_surface_and_a_spin_density_two():
    """A density is non-negative; a signed field has both lobes to show."""
    field, _centre = _hexagonal_blob()
    signed = D.replace(field, kind=D.SPIN,
                       values=field.values - np.roll(field.values, 4, axis=0))

    charge = _renderer()
    charge.set_density(field, D.DensityOptions(isovalue=0.5))
    spin = _renderer()
    spin.set_density(signed, D.DensityOptions(isovalue=0.3))

    assert len(charge._density_actors) == 1
    assert len(spin._density_actors) == 2


def test_passing_none_clears_the_field():
    field, _centre = _hexagonal_blob()
    renderer = _renderer()
    renderer.set_density(field, D.DensityOptions(isovalue=0.5))

    renderer.set_density(None)

    assert renderer._density_actors == []


def test_an_isovalue_no_part_of_the_field_reaches_draws_nothing(recwarn):
    """Rather than raising into the redraw."""
    field, _centre = _hexagonal_blob()
    renderer = _renderer()

    renderer.set_density(field, D.DensityOptions(isovalue=10.0))

    assert renderer._density_actors == []


def test_a_supercell_shows_the_field_in_every_one_of_its_cells():
    """There is no count of cells to set: the field follows the atoms, so
    building a supercell is how more of it is shown. The copies go into one
    mesh — a hundred actors for a hundred cells would be a hundred draw calls
    for one picture."""
    field, _centre = _hexagonal_blob()
    lattice = D.lattice_of(field)
    atoms = np.array([_centre + i * lattice[0] + j * lattice[1]
                      for i in range(2) for j in range(2)])
    renderer = StructureRenderer_with(atoms, lattice)

    renderer.set_density(field, D.DensityOptions(isovalue=0.5))

    assert len(renderer._density_actors) == 1
    mesh = renderer._density_actors[0].mapper.dataset
    assert len(mesh.split_bodies()) == 4


def test_a_potential_can_be_painted_onto_a_density_surface():
    """The surface says where the electrons end, the colour what a charge feels."""
    field, _centre = _hexagonal_blob()
    # A potential that varies along a, so the painted values must too.
    ramp = np.linspace(-1.0, 1.0, field.shape[0])[:, None, None]
    potential = D.replace(field, kind=D.POTENTIAL,
                          values=np.broadcast_to(ramp, field.shape).copy())

    renderer = _renderer()
    renderer.set_density(field, D.DensityOptions(isovalue=0.5, colour_by=potential))

    assert len(renderer._density_actors) == 1
    mesh = renderer._density_actors[0].mapper.dataset
    painted = np.asarray(mesh.point_data["value"])
    assert painted.min() < 0.0 < painted.max()


def test_painting_reads_the_second_field_at_the_surfaces_own_points():
    """A nearest-point lookup, so a vertex takes the value of the grid point
    it sits on rather than of the one with the same index."""
    field, centre = _hexagonal_blob()
    marked = np.zeros(field.shape)
    marked[7, 7, 7] = 5.0
    other = D.replace(field, values=marked)

    value = D.sample(other, np.array([centre]))

    assert np.isclose(value[0], 5.0)


def test_a_point_outside_the_sampled_box_takes_the_nearest_face():
    field, _centre = _hexagonal_blob()
    far = field.origin + 1000.0 * field.steps[0]

    value = D.sample(field, np.array([far]))

    assert np.isfinite(value[0])


# ── which cells the field is copied into ─────────────────────────────────
def _cell_and_field(n=11):
    """A field with a single blob at the fractional point (1/3, 2/3, 1/4)."""
    cell = np.array([[1.9832, -1.145, 0.0], [0.0, 2.29, 0.0], [0.0, 0.0, 3.59]])
    steps = cell / (n - 1)
    field = D.ScalarField(values=np.zeros((n, n, n)), origin=np.zeros(3), steps=steps)
    points = field.points()
    centre = np.array([1 / 3, 2 / 3, 1 / 4]) @ cell
    field = D.replace(field, values=np.exp(-((points - centre) ** 2).sum(axis=-1) / 0.3))
    return cell, field, centre


def test_a_field_is_copied_onto_the_cell_an_atom_was_written_in():
    """CRYSTAL writes each atom in whichever periodic image it came out in.
    Beryllium's two atoms land one ``b`` and one ``a + c`` from the grid's own
    copies of them, and a field drawn over the home cell alone sits a whole
    translation away from the atoms on screen."""
    from crystalline.viz.renderer import StructureRenderer
    import pyvista as pv

    cell, field, centre = _cell_and_field()
    stray = centre - cell[1]                      # the same atom, one -b away

    renderer = StructureRenderer(_plotter())
    renderer._positions = np.array([stray])
    renderer._density_view = (field, D.DensityOptions(isovalue=0.4, clip_to_cell=False))
    renderer._draw_density()

    centres = [np.asarray(a.mapper.dataset.points).mean(axis=0)
               for a in renderer._density_actors]
    assert any(np.allclose(c, stray, atol=0.1) for c in centres)


def test_no_surface_is_drawn_around_an_atom_that_is_not_on_screen():
    """A cell entered by one boundary atom brings a whole cell of density with
    it, most of it around atoms nobody drew — which reads as the field being in
    the wrong place. Only the pieces an atom on screen lies against are kept."""
    from crystalline.viz.renderer import StructureRenderer
    import pyvista as pv

    cell, field, centre = _cell_and_field()
    straggler = centre - cell[1] - np.array([0.0, 0.05, 0.0])   # just outside

    renderer = StructureRenderer(_plotter())
    renderer._positions = np.array([centre, straggler])
    renderer._density_view = (field, D.DensityOptions(isovalue=0.4))
    renderer._draw_density()

    drawn = [np.asarray(body.points).mean(axis=0)
             for actor in renderer._density_actors
             for body in actor.mapper.dataset.split_bodies()]
    assert drawn
    for point in drawn:
        assert np.linalg.norm(renderer._positions - point, axis=1).min() < 1.0


def test_every_cell_of_a_supercell_gets_the_field():
    """Each holds a real share of the atoms, unlike a straggler."""
    from crystalline.viz.renderer import _density_images

    cell, field, _centre = _cell_and_field()
    positions = np.array([[i + 0.5, j + 0.5, 0.5] for i in range(2) for j in range(2)]) @ cell

    images = _density_images(field, cell, positions)

    assert len(images) == 4
    assert np.allclose(images[0], 0.0)             # the home cell comes first


def test_the_colours_of_a_slice_are_set_by_its_bulk_not_by_the_nuclei():
    """A few points on a nucleus used to take the whole top of the map."""
    from crystalline.viz.renderer import _colour_limits

    values = np.concatenate([np.linspace(-2.0, 0.0, 1000), [3.0, 3.0]])

    low, high = _colour_limits(values, symmetric=False)

    assert high < 1.0 and low > -2.0


def test_a_signed_field_keeps_zero_in_the_middle_of_the_map():
    from crystalline.viz.renderer import _colour_limits

    low, high = _colour_limits(np.array([-0.2, 0.1, 0.05, 5.0]), symmetric=True)

    assert low == -high


# ── lattice planes (hkl) ─────────────────────────────────────────────────
def test_miller_indices_give_the_textbook_normal_and_spacing():
    cubic = 4.21 * np.eye(3)
    hexagonal = np.array([[2.29, 0.0, 0.0], [-1.145, 1.9832, 0.0], [0.0, 0.0, 3.59]])

    _p, normal, spacing = D.miller_plane(cubic, (0, 0, 1))
    assert np.allclose(normal, [0, 0, 1]) and np.isclose(spacing, 4.21)
    assert np.isclose(D.miller_plane(cubic, (1, 1, 1))[2], 4.21 / np.sqrt(3))
    assert np.isclose(D.miller_plane(cubic, (2, 0, 0))[2], 4.21 / 2)
    assert np.isclose(D.miller_plane(hexagonal, (0, 0, 1))[2], 3.59)
    point, normal, spacing = D.miller_plane(cubic, (0, 0, 1), offset=0.5)
    assert np.allclose(point, [0, 0, 2.105])


def test_a_plane_needs_a_non_zero_index():
    with pytest.raises(D.DensityError, match="000"):
        D.miller_plane(np.eye(3), (0, 0, 0))


def test_the_field_is_read_periodically_anywhere_in_space():
    """A plane through a crystal runs through many cells; the grid covers one.
    CRYSTAL's grid samples the far face too, and that layer must not be
    counted twice when the field is continued."""
    cell, field, _centre = _cell_and_field()
    # Index 10 is the far face, a copy of index 0 in a real grid; this fixture's
    # blob is not periodic, so only interior points are compared.
    points = field.origin + np.array([[0, 0, 0], [3, 7, 2], [9, 9, 9]]) @ field.steps

    here = D.sample_periodic(field, points)
    elsewhere = D.sample_periodic(field, points + cell[0] - 2 * cell[2])

    grid = field.values[[0, 3, 9], [0, 7, 9], [0, 2, 9]]
    assert np.allclose(here, grid)
    assert np.allclose(elsewhere, grid)


def _slice_renderer(miller=(0, 0, 1), offset=0.0, cutaway=True):
    """A cubic rock-salt-like cell with an atom at the origin, sliced."""
    import pyvista as pv

    from crystalline.core.structure import Structure
    from crystalline.viz.renderer import StructureRenderer
    from ase import Atoms

    a = 4.0
    n = 17
    cell = a * np.eye(3)
    structure = Structure.from_ase(Atoms("NaCl", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)],
                                         cell=cell, pbc=True))
    field = D.ScalarField(values=np.zeros((n, n, n)), origin=np.zeros(3),
                          steps=cell / (n - 1))
    points = field.points()
    rho = np.zeros(field.shape)
    for centre in (np.zeros(3), 0.5 * cell.sum(axis=0)):
        for shift in np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).reshape(3, -1).T:
            rho += np.exp(-((points - centre - shift @ cell) ** 2).sum(axis=-1) / 0.3)
    field = D.replace(field, values=rho)

    renderer = StructureRenderer(_plotter())
    renderer.set_structure(structure)
    renderer.set_density(field, D.DensityOptions(view=D.SLICE, miller=miller, offset=offset,
                                                 cutaway=cutaway), miller_cell=cell)
    return renderer, cell


def _map(renderer):
    """The slice mesh itself — the actor carrying the field's values."""
    for actor in renderer._density_actors:
        mesh = actor.mapper.dataset
        if "value" in mesh.point_data:
            return mesh
    raise AssertionError("no map drawn")


def test_a_slice_is_the_plane_its_miller_indices_name():
    renderer, cell = _slice_renderer(miller=(1, 1, 0), offset=0.0)

    mesh = _map(renderer)
    points = np.asarray(mesh.points)

    normal = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    assert np.ptp(points @ normal) < 1e-6          # flat, and square to (110)
    assert np.allclose(points @ normal, 0.0, atol=1e-6)


def test_a_slice_is_one_rectangle_covering_what_is_on_screen():
    """Not a piece of the one sampled cell, which for any plane not parallel to
    a cell face is a ragged polygon."""
    renderer, cell = _slice_renderer(miller=(0, 0, 1), offset=0.5)

    points = np.asarray(_map(renderer).points)

    assert points[:, 0].min() <= 0.0 and points[:, 0].max() >= cell[0, 0]
    assert points[:, 1].min() <= 0.0 and points[:, 1].max() >= cell[1, 1]


def test_a_slice_shows_the_density_peaks_where_the_atoms_are():
    renderer, _cell = _slice_renderer(miller=(0, 0, 1), offset=0.0)
    mesh = _map(renderer)
    points = np.asarray(mesh.points)
    values = np.asarray(mesh.point_data["value"])

    at_origin = values[np.argmin(np.linalg.norm(points, axis=1))]
    at_hole = values[np.argmin(np.linalg.norm(points - [1.0, 1.0, 0.0], axis=1))]

    assert at_origin > at_hole


def test_the_crystal_in_front_of_the_plane_is_cut_away_and_the_map_is_not():
    renderer, _cell = _slice_renderer()
    density = {id(actor) for actor in renderer._density_actors}

    clipped = [actor for actor in renderer.plotter.renderer.actors.values()
               if hasattr(actor, "GetMapper") and actor.GetMapper() is not None
               and hasattr(actor.GetMapper(), "GetNumberOfClippingPlanes")
               and actor.GetMapper().GetNumberOfClippingPlanes()]

    assert clipped                                               # atoms, bonds, cell
    assert not any(id(actor) in density for actor in clipped)   # the map stays whole


def test_taking_the_field_away_restores_the_crystal():
    renderer, _cell = _slice_renderer()

    renderer.set_density(None)

    assert not any(actor.GetMapper().GetNumberOfClippingPlanes()
                   for actor in renderer.plotter.renderer.actors.values()
                   if hasattr(actor, "GetMapper") and actor.GetMapper() is not None
                   and hasattr(actor.GetMapper(), "GetNumberOfClippingPlanes"))


def test_the_cutaway_can_be_switched_off():
    renderer, _cell = _slice_renderer(cutaway=False)

    assert not any(actor.GetMapper().GetNumberOfClippingPlanes()
                   for actor in renderer.plotter.renderer.actors.values()
                   if hasattr(actor, "GetMapper") and actor.GetMapper() is not None
                   and hasattr(actor.GetMapper(), "GetNumberOfClippingPlanes"))


def test_every_atom_on_the_map_is_marked_not_only_the_ones_drawn():
    """A map spans more than the cell box; a peak with a dot beside a peak
    without one reads as two different things."""
    renderer, cell = _slice_renderer(miller=(0, 0, 1), offset=0.0)
    dots = [actor.mapper.dataset for actor in renderer._density_actors
            if "value" not in actor.mapper.dataset.point_data]

    assert dots
    centres = [np.asarray(body.points).mean(axis=0) for body in dots[0].split_bodies()]
    # The Na at the origin and its images at the other corners of the face.
    corners = [np.zeros(3), cell[0], cell[1], cell[0] + cell[1]]
    for corner in corners:
        assert any(np.allclose(c[:2], corner[:2], atol=0.05) for c in centres)


def test_a_rebuild_keeps_the_field_on_screen():
    """Any rebuild — a display setting, an edit — used to drop a drawn field
    while it was still counted as shown."""
    renderer, _cell = _slice_renderer()

    renderer._rebuild()

    assert renderer._density_actors
    assert all(id(actor) in {id(a) for a in renderer.plotter.renderer.actors.values()}
               for actor in renderer._density_actors)


def test_the_camera_turns_to_face_the_plane():
    renderer, _cell = _slice_renderer(miller=(0, 0, 1))

    renderer.face_density_plane()

    position, focal, _up = renderer.plotter.camera_position
    direction = np.asarray(position) - np.asarray(focal)
    assert np.allclose(direction / np.linalg.norm(direction), [0, 0, 1], atol=1e-6)



def test_atoms_on_the_corners_of_the_cell_get_one_whole_surface_each():
    """In MgO's primitive cell every Mg sits on a corner of CRYSTAL's grid.
    Contoured over the one cell, its surface came out in eight pieces, and
    reassembling them from copies of the cell gave equivalent atoms a whole
    shell, part of one, or several fused on top of each other."""
    cell = 4.0 * np.eye(3)
    n = 25
    field = D.ScalarField(values=np.zeros((n, n, n)), origin=np.zeros(3), steps=cell / (n - 1))
    points = field.points()
    rho = np.zeros(field.shape)
    shifts = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).reshape(3, -1).T
    for centre in (np.zeros(3), 0.5 * cell.sum(axis=0)):
        for shift in shifts:
            rho += np.exp(-((points - centre - shift @ cell) ** 2).sum(axis=-1) / 0.4)
    field = D.replace(field, values=rho)

    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float) @ cell
    atoms = np.vstack([corners, 0.5 * cell.sum(axis=0)])   # a boundary-completed view

    renderer = StructureRenderer_with(atoms, cell)
    renderer._density_view = (field, D.DensityOptions(isovalue=0.5))
    renderer._draw_density()

    bodies = [b for a in renderer._density_actors for b in a.mapper.dataset.split_bodies()]
    per_atom = []
    for atom in atoms:
        mine = [b for b in bodies if np.linalg.norm(np.asarray(b.points).mean(axis=0) - atom) < 0.1]
        per_atom.append((len(mine), sum(b.n_points for b in mine)))

    assert all(count == 1 for count, _ in per_atom)          # one surface, not pieces or copies
    corner_sizes = {size for _count, size in per_atom[:8]}
    assert len(corner_sizes) == 1                             # identical on every corner
    assert len(bodies) == len(atoms)


def StructureRenderer_with(positions, cell):
    """A renderer holding ``positions`` as the atoms on screen, in ``cell``."""
    from crystalline.viz.renderer import StructureRenderer

    renderer = StructureRenderer(_plotter())
    renderer._positions = np.asarray(positions, dtype=float)
    renderer._cell_or_none = lambda: np.asarray(cell, dtype=float)
    return renderer


def test_switching_the_cell_view_does_not_contour_the_field_again(monkeypatch):
    """Contouring and splitting the surface is the expensive part and depends on
    nothing on screen. Redone on every rebuild, it made switching between the
    conventional and the primitive cell take seconds with a density shown."""
    import crystalline.viz.renderer as R
    from ase import Atoms

    from crystalline.core.structure import Structure

    calls = []
    real = R._density_surfaces
    monkeypatch.setattr(R, "_density_surfaces",
                        lambda *args, **kwargs: calls.append(1) or real(*args, **kwargs))

    field, _centre = _hexagonal_blob()
    lattice = D.lattice_of(field)
    renderer = R.StructureRenderer(_plotter())
    one = Structure.from_ase(Atoms("H", positions=[lattice.sum(axis=0) / 2], cell=lattice, pbc=True))
    two = Structure.from_ase(Atoms("H2", positions=[lattice.sum(axis=0) / 2, lattice[0] / 2],
                                   cell=lattice, pbc=True))
    renderer.set_structure(one)
    renderer.set_density(field, D.DensityOptions(isovalue=0.5))

    renderer.set_structure(two)
    renderer.set_structure(one)

    assert len(calls) == 1
    assert renderer._density_actors


def test_a_new_isovalue_does_contour_again(monkeypatch):
    import crystalline.viz.renderer as R

    calls = []
    real = R._density_surfaces
    monkeypatch.setattr(R, "_density_surfaces",
                        lambda *args, **kwargs: calls.append(1) or real(*args, **kwargs))
    field, _centre = _hexagonal_blob()
    renderer = R.StructureRenderer(_plotter())

    renderer.set_density(field, D.DensityOptions(isovalue=0.5))
    renderer.set_density(field, D.DensityOptions(isovalue=0.3))

    assert len(calls) == 2


# ── colour bar ───────────────────────────────────────────────────────────
def test_a_bar_is_headed_with_the_quantity_its_unit_and_its_scale():
    field, _centre = _hexagonal_blob()
    potential = D.replace(field, kind=D.POTENTIAL)

    assert D.bar_title(field, D.DensityOptions(view=D.SLICE)) == "log₁₀ ρ (e/bohr³)"
    assert D.bar_title(field, D.DensityOptions(view=D.SLICE, logarithmic=False)) == "ρ (e/bohr³)"
    assert D.bar_title(field, D.DensityOptions(colour_by=potential)) == "V (hartree/e)"
    difference = D.difference(field, D.replace(field, values=field.values * 0.5))
    assert D.bar_title(difference, D.DensityOptions(view=D.SLICE)) == "Δρ (e/bohr³)"


def test_a_slice_gets_a_bar_when_asked_and_loses_it_when_cleared():
    renderer, _cell = _slice_renderer()
    field, options = renderer._density_view
    renderer.set_density(field, D.replace(options, colour_bar=True),
                         miller_cell=renderer._density_miller_cell)

    assert list(renderer.plotter.scalar_bars.keys()) == ["log₁₀ ρ (e/bohr³)"]

    renderer._rebuild()
    assert len(renderer.plotter.scalar_bars) == 1          # redrawn once, not twice

    renderer.set_density(None)
    assert len(renderer.plotter.scalar_bars) == 0


def test_a_plain_surface_has_nothing_to_key():
    field, _centre = _hexagonal_blob()
    renderer = StructureRenderer_with(np.empty((0, 3)), D.lattice_of(field))

    renderer.set_density(field, D.DensityOptions(isovalue=0.5, colour_bar=True))

    assert len(renderer.plotter.scalar_bars) == 0


def test_the_bar_text_reads_on_the_background():
    from crystalline.viz.renderer import _readable_on

    assert _readable_on("white") == "black"
    assert _readable_on("#101418") == "white"
