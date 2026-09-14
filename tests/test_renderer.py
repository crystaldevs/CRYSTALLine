"""Tests for StructureRenderer's cell wireframe (off-screen PyVista)."""

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("pyvista")
pytest.importorskip("ase")

import pyvista as pv  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.build import bulk  # noqa: E402

from crystalline.core.structure import Structure  # noqa: E402
from crystalline.viz.renderer import StructureRenderer, _bonded_pairs, _cell_edges  # noqa: E402
from crystalline.viz.render_settings import RenderSettings  # noqa: E402


def _outline_extent(renderer):
    lo = np.array(renderer._cell_actor.GetBounds()).reshape(3, 2)
    return lo[:, 1] - lo[:, 0]


def test_reference_cell_outlines_original_not_supercell():
    base = bulk("NaCl", "rocksalt", a=5.64)
    supercell = Structure.from_ase(base.repeat((2, 2, 1)))
    renderer = StructureRenderer(pv.Plotter(off_screen=True))

    # Default: the wireframe follows the (super)cell that owns the atoms.
    renderer.set_reference_cell(None)
    renderer.set_structure(supercell)
    supercell_extent = _outline_extent(renderer)

    # With a reference cell, the wireframe shrinks to the original unit cell
    # while all supercell atoms remain on screen.
    renderer.set_reference_cell(np.asarray(base.cell))
    renderer.set_structure(supercell)
    unit_extent = _outline_extent(renderer)

    assert renderer.atom_count == 4 * len(base)  # 2×2×1 supercell atoms still shown
    assert np.all(unit_extent < supercell_extent - 1e-6)  # smaller outline


def test_cell_edges_scale_with_periodic_dimension():
    a, b, c = np.eye(3)
    assert _cell_edges([a]).n_lines == 1  # 1D polymer: a single segment
    assert _cell_edges([a, b]).n_lines == 4  # 2D slab: a parallelogram
    assert _cell_edges([a, b, c]).n_lines == 12  # 3D crystal: the full box


def test_slab_cell_box_is_flat_not_500_angstroms():
    """A 2D slab must outline its in-plane cell, not a box stretched along the
    formal 500 Å vacuum vector CRYSTAL writes for the aperiodic direction."""
    from ase.build import fcc111

    slab = fcc111("Pt", size=(2, 2, 3), vacuum=0.0)
    cell = np.asarray(slab.get_cell(), dtype=float)
    cell[2] = [0.0, 0.0, 500.0]  # CRYSTAL's aperiodic placeholder
    slab.set_cell(cell)
    slab.set_pbc((True, True, False))

    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(Structure.from_ase(slab))

    extent = _outline_extent(renderer)
    assert extent[2] < 1.0  # the box has no depth along the non-periodic axis
    assert extent[0] > 1.0 and extent[1] > 1.0  # but spans the periodic plane


def test_the_abc_gizmo_lives_in_screen_space_not_in_the_scene():
    """It is pinned to a viewport corner, so it stays put whatever the camera
    does. As a scene actor it moved with every view change — the three axis
    alignments each put it somewhere different, which is what a gizmo must not
    do. Being in screen space, it contributes no actors to the renderer."""
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    periodic = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64))

    renderer.set_structure(periodic)

    assert renderer._orientation_widget is not None
    # an arrow and a label per periodic direction, plus the invisible prop that
    # widens the bounds so the labels aren't clipped at the viewport edge
    assert renderer._lattice_marker().GetParts().GetNumberOfItems() == 3 * 2 + 1


def test_the_gizmo_shows_one_arrow_per_periodic_direction():
    """A slab must not sprout a c arrow along CRYSTAL's 500 Å vacuum vector."""
    from ase.build import fcc111

    slab = fcc111("Pt", size=(2, 2, 2), vacuum=0.0)
    cell = np.asarray(slab.get_cell(), dtype=float)
    cell[2] = [0.0, 0.0, 500.0]
    slab.set_cell(cell)
    slab.set_pbc((True, True, False))
    renderer = StructureRenderer(pv.Plotter(off_screen=True))

    renderer.set_structure(Structure.from_ase(slab))

    assert renderer._lattice_marker().GetParts().GetNumberOfItems() == 2 * 2 + 1  # a and b only


def test_the_gizmo_labels_are_screen_aligned_captions():
    """3D text turns edge-on and vanishes exactly when its axis points at the
    viewer — which is the case in each of the a/b/c alignment views. Captions are
    drawn in the image plane, so they stay upright and legible from every angle."""
    import vtk

    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64)))

    parts = renderer._lattice_marker().GetParts()
    parts.InitTraversal()
    captions = [
        part for part in (parts.GetNextProp() for _ in range(parts.GetNumberOfItems()))
        if isinstance(part, vtk.vtkCaptionActor2D)
    ]

    assert [c.GetCaption() for c in captions] == ["a", "b", "c"]
    # no border and no leader line: just the letter at the arrow tip
    assert all(not c.GetBorder() and not c.GetLeader() for c in captions)


def test_a_non_periodic_structure_has_no_gizmo():
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    mol = Structure.empty()
    mol.add_atom("O", [0.0, 0.0, 0.0])
    mol.add_atom("H", [0.96, 0.0, 0.0])

    renderer.set_structure(mol)  # no lattice to point at, no crash

    assert renderer._lattice_marker() is None


def _count_actors(renderer):
    return len(list(renderer.plotter.renderer.actors))


def test_settings_toggle_bonds_cell_and_polyhedra():
    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(nacl)
    assert renderer._bond_actor is not None
    # Polyhedra are off until asked for: they enclose the atoms they are built
    # from, so a first look at a structure would be a first look at solids.
    assert renderer._polyhedra_actor is None

    renderer.set_settings(RenderSettings(show_polyhedra=True))
    assert renderer._polyhedra_actor is not None  # octahedra drawn

    renderer.set_settings(
        RenderSettings(show_bonds=False, show_cell=False, show_polyhedra=False)
    )
    assert renderer._bond_actor is None and renderer._cell_actor is None
    assert renderer._polyhedra_actor is None


def test_polyhedron_outline_shows_real_edges_only():
    """A cubic 8-coordination hull outlines its 12 edges — the triangulation's
    6 coplanar face diagonals must not show (they do unless the hull's simplices
    are consistently wound)."""
    from crystalline.viz.renderer import _POLYHEDRA_EDGE_ANGLE, _hull_mesh

    cube = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    mesh = _hull_mesh([(20, cube)], {})
    edges = mesh.extract_feature_edges(
        feature_angle=_POLYHEDRA_EDGE_ANGLE,
        boundary_edges=False,
        non_manifold_edges=False,
        manifold_edges=False,
    )
    assert edges.n_cells == 12


def test_edge_sharing_polyhedra_keep_their_shared_edge():
    """Neighbouring polyhedra share ligand atoms; welding those points would fuse
    the two solids and drop the shared edge out of the outline."""
    from crystalline.viz.renderer import _POLYHEDRA_EDGE_ANGLE, _hull_mesh

    octa = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float
    )
    shifted = octa + np.array([2.0, 0.0, 0.0])  # shares the [1,0,0] vertex
    mesh = _hull_mesh([(11, octa), (11, shifted)], {})
    edges = mesh.extract_feature_edges(
        feature_angle=_POLYHEDRA_EDGE_ANGLE,
        boundary_edges=False,
        non_manifold_edges=False,
        manifold_edges=False,
    )
    assert edges.n_cells == 24  # 12 per octahedron, none lost to the shared vertex


def test_polyhedra_get_an_outline_actor_and_a_translucent_default():
    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    assert RenderSettings().polyhedra_opacity == 0.3  # see through to the atoms inside

    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(nacl)
    renderer.set_settings(RenderSettings(show_polyhedra=True))
    assert renderer._polyhedra_edge_actor is not None
    assert renderer._polyhedra_edge_actor.GetMapper().GetInput().GetNumberOfCells() > 0

    renderer.set_settings(RenderSettings(show_polyhedra=False))
    assert renderer._polyhedra_edge_actor is None  # outline goes with the polyhedra


def test_measurement_annotations_are_drawn_and_survive_a_rebuild():
    from crystalline.core import measure as measure_mod

    water = Structure.empty()
    water.add_atom("O", [0.0, 0.0, 0.0])
    water.add_atom("H", [0.96, 0.0, 0.0])
    water.add_atom("H", [-0.24, 0.93, 0.0])

    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(water)
    assert renderer._annotation_actors == []  # nothing measured yet

    positions, symbols = water.positions, water.symbols
    renderer.set_annotations(
        [
            measure_mod.measure(positions, symbols, [0, 1]),        # distance
            measure_mod.measure(positions, symbols, [1, 0, 2]),     # angle
            measure_mod.measure_plane(positions, symbols, [0, 1, 2]),  # plane
        ]
    )
    drawn = len(renderer._annotation_actors)
    assert drawn >= 4  # two paths, a plane patch, and the labels

    # An edit or a settings change clears the plotter — annotations must come back.
    renderer.set_settings(RenderSettings(show_bonds=False))
    assert len(renderer._annotation_actors) == drawn

    renderer.set_annotations([])
    assert renderer._annotation_actors == []


def test_plane_annotation_has_no_floating_label():
    """A plane patch carries no 'rms …' label — only the patch actor is drawn."""
    from crystalline.core import measure as measure_mod

    water = Structure.empty()
    water.add_atom("O", [0.0, 0.0, 0.0])
    water.add_atom("H", [0.96, 0.0, 0.0])
    water.add_atom("H", [-0.24, 0.93, 0.0])
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(water)

    renderer.set_annotations([measure_mod.measure_plane(water.positions, water.symbols, [0, 1, 2])])
    assert len(renderer._annotation_actors) == 1  # just the patch, no label actor


def test_measurement_colours_come_from_settings():
    from crystalline.core import measure as measure_mod

    water = Structure.empty()
    water.add_atom("O", [0.0, 0.0, 0.0])
    water.add_atom("H", [0.96, 0.0, 0.0])
    renderer = StructureRenderer(
        pv.Plotter(off_screen=True), RenderSettings(measure_line_color="#ff0000")
    )
    renderer.set_structure(water)
    renderer.set_annotations([measure_mod.measure(water.positions, water.symbols, [0, 1])])

    line_actor = renderer._annotation_actors[0]
    assert np.allclose(line_actor.GetProperty().GetColor(), (1.0, 0.0, 0.0), atol=1e-3)


def test_per_item_measurement_colour_overrides_the_group_default():
    import dataclasses

    from crystalline.core import measure as measure_mod

    water = Structure.empty()
    water.add_atom("O", [0.0, 0.0, 0.0])
    water.add_atom("H", [0.96, 0.0, 0.0])
    renderer = StructureRenderer(
        pv.Plotter(off_screen=True), RenderSettings(measure_line_color="#ff0000")
    )
    renderer.set_structure(water)

    dist = measure_mod.measure(water.positions, water.symbols, [0, 1])
    recoloured = dataclasses.replace(dist, color="#00ff00")  # this item is green
    renderer.set_annotations([recoloured])

    colour = renderer._annotation_actors[0].GetProperty().GetColor()
    assert np.allclose(colour, (0.0, 1.0, 0.0), atol=1e-3)  # the item's colour, not the red default


def test_hydrogen_bonds_follow_a_phonon_animation():
    """The H-bond overlay must move with the atoms during animation (via
    ``update_positions``), not stay pinned to the equilibrium geometry."""
    dimer = Structure.empty()
    for sym, pos in (
        ("O", [0, 0, 0]), ("H", [0.97, 0, 0]), ("H", [-0.3, 0.9, 0]),
        ("O", [2.8, 0, 0]), ("H", [3.77, 0.2, 0]), ("H", [2.5, -0.9, 0]),
    ):
        dimer.add_atom(sym, pos)
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(dimer)
    assert renderer._hbond_actor is not None  # the O–H···O contact is drawn
    before = np.array(renderer._hbond_actor.GetBounds())

    moved = dimer.positions.copy()
    moved[3:] += [0.6, 0.0, 0.0]  # push the acceptor water away
    renderer.update_positions(moved)
    after = np.array(renderer._hbond_actor.GetBounds())
    assert not np.allclose(before, after)  # the dashed bond followed the atoms

    far = dimer.positions.copy()
    far[3:] += [3.0, 0.0, 0.0]  # far enough that it's no longer a hydrogen bond
    renderer.update_positions(far)
    assert renderer._hbond_actor is None


def test_annotations_never_become_pick_targets():
    """Picking must still hit atoms, not a measurement drawn over them."""
    from crystalline.core import measure as measure_mod

    water = Structure.empty()
    water.add_atom("O", [0.0, 0.0, 0.0])
    water.add_atom("H", [0.96, 0.0, 0.0])

    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(water)
    renderer.set_annotations([measure_mod.measure(water.positions, water.symbols, [0, 1])])
    assert renderer._annotation_actors
    for actor in renderer._annotation_actors:
        assert actor.GetPickable() == 0


def test_polyhedra_follow_the_atoms_during_animation():
    """A phonon animation moves atoms via update_positions; the coordination
    polyhedra (and their outline) have to travel with their ligands."""
    from crystalline.core.cells import CellView, as_view, complete_boundary, tile_supercell

    src = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    analysis, _ = tile_supercell(as_view(src, CellView.CRYSTALLOGRAPHIC), (1, 1, 1), None)
    packed, _ = complete_boundary(analysis, None)

    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_settings(RenderSettings(show_polyhedra=True))
    renderer.set_structure(packed, bond_structure=analysis)
    assert renderer._polyhedra_follow is not None  # every vertex matched an atom
    faces_before = renderer._polyhedra_mesh_obj.points.copy()
    edges_before = renderer._polyhedra_edge_mesh.points.copy()

    # A rigid shift of every atom must shift every hull vertex by exactly that.
    shift = np.array([0.13, -0.07, 0.21])
    renderer.update_positions(packed.positions + shift)
    assert np.allclose(renderer._polyhedra_mesh_obj.points - faces_before, shift)
    assert np.allclose(renderer._polyhedra_edge_mesh.points - edges_before, shift)

    # A per-atom displacement lands each vertex on its own displaced atom.
    rng = np.random.default_rng(0)
    displaced = packed.positions + rng.normal(scale=0.05, size=packed.positions.shape)
    renderer.update_positions(displaced)
    index, offset = renderer._polyhedra_follow
    assert np.allclose(renderer._polyhedra_mesh_obj.points, displaced[index] + offset)


def test_polyhedra_are_drawn_on_boundary_completed_images():
    """The displayed cell is boundary-completed, so a corner cation appears at
    several cell positions; each of those must be coordinated, not just the one
    atom the (pre-packing) analysis cell knows about."""
    from crystalline.core.cells import CellView, as_view, complete_boundary, tile_supercell

    src = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    analysis, _ = tile_supercell(as_view(src, CellView.CRYSTALLOGRAPHIC), (1, 1, 1), None)
    packed, _ = complete_boundary(analysis, None)
    assert len(packed) > len(analysis)  # genuinely packed with image atoms

    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_settings(RenderSettings(show_polyhedra=True))
    renderer.set_structure(packed, bond_structure=analysis)

    from crystalline.core.bonds import connectivity

    analysed = connectivity(analysis, RenderSettings().polyhedra_min_vertices)
    centres = {int(z) for z, _c, _l in analysed.polyhedra}
    shown_centres = sum(int((packed.numbers == z).sum()) for z in centres)
    assert shown_centres > len(analysed.polyhedra)  # the bug: images were left bare

    # every shown centre contributes a hull, so the mesh spans the packed cell
    mesh_extent = np.array(renderer._polyhedra_actor.GetBounds()).reshape(3, 2)
    atoms_extent = np.array([packed.positions.min(axis=0), packed.positions.max(axis=0)]).T
    assert np.all(mesh_extent[:, 0] <= atoms_extent[:, 0] + 1e-6)
    assert np.all(mesh_extent[:, 1] >= atoms_extent[:, 1] - 1e-6)


def test_scene_settings_and_atom_labels():
    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(nacl)
    assert renderer._label_actor is None  # labels off by default

    renderer.set_settings(
        RenderSettings(
            show_atom_labels=True,
            background_color="black",
            parallel_projection=True,
            show_orientation_axes=True,
        )
    )
    assert renderer._label_actor is not None  # element labels drawn
    assert renderer.plotter.camera.GetParallelProjection() == 1  # orthographic applied


def test_default_projection_is_parallel():
    # New structures render orthographic (parallel) by default.
    assert RenderSettings().parallel_projection is True
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64)))
    assert renderer.plotter.camera.GetParallelProjection() == 1


def test_refresh_preserves_zoom_under_parallel_projection():
    """Editing (e.g. deleting an atom) rebuilds the scene; the user's zoom must
    survive. Under parallel projection zoom is the camera's parallel scale, which
    ``camera_position`` doesn't carry — so a naive save/restore let the auto-fit
    zoom back out on every edit."""
    renderer = StructureRenderer(pv.Plotter(off_screen=True))  # parallel by default
    structure = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64).repeat((2, 2, 2)))
    renderer.set_structure(structure)

    renderer.plotter.camera.Zoom(2.5)  # user zooms in
    zoomed = renderer.plotter.camera.GetParallelScale()

    structure.remove_atoms([0])  # an edit -> refresh() rebuilds the scene
    renderer.refresh()

    assert np.isclose(renderer.plotter.camera.GetParallelScale(), zoomed)


def test_atom_labels_suppressed_for_large_cells():
    big = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64).repeat((6, 6, 6)))  # 432 atoms
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(big)
    renderer.set_settings(RenderSettings(show_atom_labels=True))
    assert renderer.atom_count > 400
    assert renderer._label_actor is None  # too many atoms to label legibly


def test_atom_colour_overrides_apply_per_element():
    from crystalline.viz.renderer import _hex_to_rgb

    assert list(_hex_to_rgb("#ff8800")) == [255, 136, 0]
    assert list(_hex_to_rgb("#f80")) == [255, 136, 0]  # short form

    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(nacl)
    renderer.set_settings(RenderSettings(atom_colors=((11, "#ff0000"),)))  # Na -> red
    rgb = renderer._rgb_for(np.array([11, 17, 11]))
    assert list(rgb[0]) == [255, 0, 0] and list(rgb[2]) == [255, 0, 0]  # both Na red
    assert list(rgb[1]) != [255, 0, 0]  # Cl keeps its Jmol colour


def test_selected_group_moves_together_on_drag():
    # A non-periodic cluster (no periodic images to confuse the group).
    s = Structure.empty()
    for k in range(6):
        s.add_atom("C", [k * 1.6, 0.0, 0.0])
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(s)

    # With no selection, only the grabbed atom's (empty) image set moves with it.
    assert renderer.active_move_group(2) == []

    # Selecting {2,3,4} and grabbing 3 drags the whole selection as one piece.
    renderer.highlight([2, 3, 4])
    assert renderer.active_move_group(3) == [2, 4]
    assert renderer.active_move_group(0) == []  # grabbing outside the selection

    renderer.preview_atom_position(3, np.array([3 * 1.6, 0.0, 2.0]))  # shift +2 in z
    for i in (2, 3, 4):
        assert np.isclose(renderer.rendered_atom_position(i)[2], 2.0)
    for i in (0, 1, 5):
        assert np.isclose(renderer.rendered_atom_position(i)[2], 0.0)  # untouched


def test_bonds_draw_with_a_single_fixed_colour():
    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(nacl)
    assert renderer._bond_actor is not None  # bonds still drawn, just no colour options


def test_hydrogen_bonds_drawn_by_default_and_toggle_off():
    from ase import Atoms

    dimer = Structure.from_ase(Atoms(
        "OH2OH2",
        positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0],
                   [2.85, 0, 0], [3.81, 0.2, 0], [2.61, -0.93, 0]],
        pbc=False,
    ))
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    assert RenderSettings().show_hydrogen_bonds is True  # on by default
    renderer.set_structure(dimer)
    assert renderer._hbond_actor is not None  # the O–H···O contact is drawn

    renderer.set_settings(RenderSettings(show_hydrogen_bonds=False))
    assert renderer._hbond_actor is None


def test_hydrogen_bonds_follow_the_atoms_during_animation():
    """Animation moves atoms via ``update_positions`` (no full rebuild); the
    hydrogen bonds must move with them and vanish when the contact breaks."""
    from ase import Atoms

    dimer = Structure.from_ase(Atoms(
        "OH2OH2",
        positions=[[0, 0, 0], [0.97, 0, 0], [-0.3, 0.9, 0],
                   [2.8, 0, 0], [3.77, 0.2, 0], [2.5, -0.9, 0]],
        pbc=False,
    ))
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(dimer)
    xmax_before = renderer._hbond_actor.GetBounds()[1]

    moved = dimer.positions.copy()
    moved[3:] += [0.6, 0.0, 0.0]  # push the acceptor water out along +x
    renderer.update_positions(moved)
    assert renderer._hbond_actor.GetBounds()[1] > xmax_before  # the bond stretched with it

    far = dimer.positions.copy()
    far[3:] += [3.0, 0.0, 0.0]  # break the contact entirely
    renderer.update_positions(far)
    assert renderer._hbond_actor is None


def test_dashed_lines_breaks_a_segment_into_pieces():
    from crystalline.viz.renderer import _dashed_lines

    seg = np.array([[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])  # one 3 Å segment
    solid = _dashed_lines(seg, dash=3.0, gap=0.0)
    dashed = _dashed_lines(seg, dash=0.3, gap=0.2)
    assert dashed.n_lines > solid.n_lines  # dashing yields many short line cells
    assert _dashed_lines(np.empty((0, 2, 3)), 0.3, 0.2) is None  # nothing to draw


def test_view_up_vector_is_orthonormal_to_direction():
    from crystalline.ui.viewport import Viewport

    cell = np.array([[4.0, 0.0, 0.0], [1.0, 3.5, 0.0], [0.5, 0.4, 5.0]])
    for axis in range(3):
        direction = cell[axis] / np.linalg.norm(cell[axis])
        up = Viewport._up_for(direction, cell, axis)
        assert abs(np.dot(up, direction)) < 1e-9  # perpendicular to the view direction
        assert abs(np.linalg.norm(up) - 1.0) < 1e-9  # unit length
    # No cell → a world axis not parallel to the view direction.
    up = Viewport._up_for(np.array([0.0, 0.0, 1.0]), None, 2)
    assert abs(np.dot(up, [0.0, 0.0, 1.0])) < 1e-9


def test_bonds_drop_cation_cation_pairs():
    # Two Na (both cations) close together must NOT bond; Na-Cl must.
    na2 = Structure.empty()
    na2.add_atom("Na", [0, 0, 0])
    na2.add_atom("Na", [2.5, 0, 0])
    i, j = _bonded_pairs(na2.positions, na2.numbers, 1.15)
    assert len(i) == 0  # cation-cation excluded

    nacl = Structure.empty()
    nacl.add_atom("Na", [0, 0, 0])
    nacl.add_atom("Cl", [2.5, 0, 0])
    i, j = _bonded_pairs(nacl.positions, nacl.numbers, 1.15)
    assert len(i) == 1  # cation-anion kept


def test_periodic_images_move_together_on_preview():
    # A cell with an atom and its exact lattice image, a distinct same-element
    # atom, and a different element sitting on a lattice point.
    s = Structure.empty()
    s.set_cell(np.diag([5.0, 5.0, 5.0]), periodic=True)
    s.add_atom("Na", [0.0, 0.0, 0.0])  # 0
    s.add_atom("Na", [5.0, 0.0, 0.0])  # 1: periodic image of 0 (a-translation)
    s.add_atom("Na", [2.0, 2.0, 2.0])  # 2: a distinct Na, not an image
    s.add_atom("Cl", [0.0, 5.0, 0.0])  # 3: on a lattice point but different element
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(s)

    assert renderer.periodic_image_indices(0) == [1]
    assert renderer.periodic_image_indices(2) == []  # no image of the interior atom

    # Previewing a drag of atom 0 shifts its image 1 by the same vector; the
    # unrelated atoms stay put.
    renderer.preview_atom_position(0, np.array([0.5, 0.0, 0.0]))
    assert np.allclose(renderer.rendered_atom_position(0), [0.5, 0.0, 0.0])
    assert np.allclose(renderer.rendered_atom_position(1), [5.5, 0.0, 0.0])
    assert np.allclose(renderer.rendered_atom_position(2), [2.0, 2.0, 2.0])
    assert np.allclose(renderer.rendered_atom_position(3), [0.0, 5.0, 0.0])


def test_periodic_images_span_a_supercell_via_reference_cell():
    base = bulk("NaCl", "rocksalt", a=5.64)
    supercell = Structure.from_ase(base.repeat((2, 2, 2)))  # 16 atoms
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    # The reference cell (original unit cell) defines the periodicity, so each
    # site has one image per supercell tile.
    renderer.set_reference_cell(np.asarray(base.cell))
    renderer.set_structure(supercell)

    images = renderer.periodic_image_indices(0)
    assert len(images) == 7  # 2×2×2 tiles → 8 copies of the site, minus itself
    assert all(supercell.numbers[i] == supercell.numbers[0] for i in images)


# ── phonon displacement arrows ────────────────────────────────────────────
def _two_atoms() -> Structure:
    s = Structure.empty()
    s.set_cell(np.eye(3) * 8, periodic=False)
    s.add_atom("C", [4.0, 4.0, 4.0])
    s.add_atom("O", [5.2, 4.0, 4.0])
    return s


def _arrow_extent(renderer):
    """Longest side of the arrow actor's bounding box, in Angstrom."""
    bounds = np.array(renderer._arrow_actor.GetBounds()).reshape(3, 2)
    return float((bounds[:, 1] - bounds[:, 0]).max())


_ARROWS_ON = RenderSettings(show_mode_arrows=True, show_polyhedra=False)


def _arrow_renderer(settings: RenderSettings = _ARROWS_ON, structure=None):
    renderer = StructureRenderer(pv.Plotter(off_screen=True), settings)
    renderer.set_structure(structure if structure is not None else _two_atoms())
    return renderer


def test_arrows_are_off_until_asked_for():
    """The animation is the primary view; arrows on top of it are opt-in."""
    assert RenderSettings().show_mode_arrows is False

    renderer = StructureRenderer(pv.Plotter(off_screen=True))  # stock settings
    renderer.set_structure(_two_atoms())
    renderer.set_mode_vectors(np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]))

    assert renderer._arrow_actor is None


def test_mode_vectors_are_drawn_and_cleared():
    renderer = _arrow_renderer()
    assert renderer._arrow_actor is None  # nothing to show until a mode is chosen

    renderer.set_mode_vectors(np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]))
    assert renderer._arrow_actor is not None

    renderer.set_mode_vectors(None)
    assert renderer._arrow_actor is None


def test_every_arrow_is_drawn_at_the_same_length():
    """Arrows mark direction, not distance. Two atoms 10 Å apart moving apart by
    very different amounts must still get identically-sized arrows — so the span
    they cover is the separation plus two whole arrows, not one and a fraction."""
    structure = Structure.empty()
    structure.set_cell(np.eye(3) * 40, periodic=False)
    structure.add_atom("C", [0.0, 0.0, 0.0])
    structure.add_atom("C", [0.0, 0.0, 10.0])
    renderer = _arrow_renderer(
        RenderSettings(show_mode_arrows=True, show_polyhedra=False, mode_arrow_scale=2.0),
        structure,
    )

    # the far atom barely moves; its arrow must be as long as the near one's
    renderer.set_mode_vectors(np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 0.2]]))

    assert _arrow_extent(renderer) == pytest.approx(10.0 + 2 * 2.0, abs=0.05)


def test_arrow_length_ignores_the_eigenvector_norm():
    """Eigenvectors are normalised over the whole cell, so their absolute size
    means nothing on screen: the setting alone fixes the drawn length."""
    renderer = _arrow_renderer(
        RenderSettings(show_mode_arrows=True, show_polyhedra=False, mode_arrow_scale=2.0)
    )

    renderer.set_mode_vectors(np.array([[0.0, 0.01, 0.0], [0.0, 0.0, 0.0]]))
    tiny = _arrow_extent(renderer)
    renderer.set_mode_vectors(np.array([[0.0, 100.0, 0.0], [0.0, 0.0, 0.0]]))
    huge = _arrow_extent(renderer)

    assert tiny == pytest.approx(huge, abs=1e-6)  # same drawn size either way
    assert huge == pytest.approx(2.0, abs=0.05)   # and it is the requested scale


def test_atoms_barely_involved_in_the_mode_get_no_arrow():
    """With every arrow the same length, an atom that hardly moves would
    otherwise be flagged as strongly as the one carrying the mode."""
    renderer = _arrow_renderer(
        RenderSettings(show_mode_arrows=True, show_polyhedra=False, mode_arrow_scale=2.0)
    )

    # second atom at 0.1% of the first: below the cut-off, so one arrow only
    renderer.set_mode_vectors(np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.001]]))

    assert _arrow_extent(renderer) == pytest.approx(2.0, abs=0.05)


def test_a_mismatched_or_null_mode_draws_no_arrows():
    """The shown structure can be swapped (supercell, cell view) under a
    selected mode; and an all-zero eigenvector has no direction to draw."""
    renderer = _arrow_renderer()

    renderer.set_mode_vectors(np.ones((5, 3)))  # five vectors, two atoms
    assert renderer._arrow_actor is None

    renderer.set_mode_vectors(np.zeros((2, 3)))
    assert renderer._arrow_actor is None


def test_arrows_can_be_switched_off_in_the_display_settings():
    renderer = _arrow_renderer()
    renderer.set_mode_vectors(np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]))
    assert renderer._arrow_actor is not None

    renderer.set_settings(RenderSettings(show_mode_arrows=False))
    assert renderer._arrow_actor is None

    renderer.set_settings(RenderSettings(show_mode_arrows=True))
    assert renderer._arrow_actor is not None  # and come back with the setting


def test_arrows_travel_with_the_atoms_during_an_animation():
    structure = _two_atoms()
    renderer = _arrow_renderer(structure=structure)
    renderer.set_mode_vectors(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]))
    before = np.array(renderer._arrow_actor.GetBounds()).reshape(3, 2)[2]  # z range

    renderer.update_positions(structure.positions + np.array([0.0, 0.0, 3.0]))
    after = np.array(renderer._arrow_actor.GetBounds()).reshape(3, 2)[2]

    assert after == pytest.approx(before + 3.0, abs=1e-6)


# ── thermal ellipsoids (ADP) ──────────────────────────────────────────────
_ADP_ON = RenderSettings(show_adp_ellipsoids=True, show_polyhedra=False)
# A realistic room-temperature ADP: ~0.1 Å r.m.s. displacement.
_TYPICAL_ADP = np.diag([0.012, 0.008, 0.020])


def _adp_renderer(settings: RenderSettings = _ADP_ON, structure=None):
    renderer = StructureRenderer(pv.Plotter(off_screen=True), settings)
    renderer.set_structure(structure if structure is not None else _two_atoms())
    return renderer


def test_ellipsoids_replace_the_atom_spheres():
    """The ellipsoid *is* the atom, as in ORTEP. Keeping the covalent-radius
    sphere as well would bury it — a 50% ellipsoid is a couple of tenths of an
    Angstrom across, the drawn atom several times that."""
    renderer = _adp_renderer()
    assert renderer._atom_actor is not None and renderer._adp_actor is None

    renderer.set_adp_tensors(np.tile(_TYPICAL_ADP, (2, 1, 1)))

    assert renderer._adp_actor is not None
    assert renderer._atom_actor is None  # every atom became an ellipsoid


def test_an_ellipsoid_is_picked_as_its_atom():
    """Picking must keep working when the sphere it used to hit is gone."""
    renderer = _adp_renderer()
    renderer.set_adp_tensors(np.tile(_TYPICAL_ADP, (2, 1, 1)))

    assert renderer.pick_atom_index(renderer._adp_actor, [5.2, 4.0, 4.0]) == 1
    assert renderer.pick_atom_index(renderer._adp_actor, [4.0, 4.0, 4.0]) == 0
    assert renderer.pick_atom_index(None, [4.0, 4.0, 4.0]) is None


def test_the_drawn_ellipsoid_has_the_tensor_s_size_and_orientation():
    """A tensor stretched along z must draw an ellipsoid stretched along z, at
    the semi-axis length the probability asks for."""
    structure = Structure.empty()
    structure.set_cell(np.eye(3) * 30, periodic=False)
    structure.add_atom("C", [0.0, 0.0, 0.0])
    probability = 0.5  # pinned, not the app default, so the arithmetic is explicit
    renderer = _adp_renderer(
        RenderSettings(show_adp_ellipsoids=True, show_polyhedra=False,
                       adp_probability=probability),
        structure,
    )

    renderer.set_adp_tensors(np.array([np.diag([0.01, 0.01, 0.04])]))

    from crystalline.core.adp import probability_scale

    extent = np.array(renderer._adp_actor.GetBounds()).reshape(3, 2)
    span = extent[:, 1] - extent[:, 0]
    expected = 2 * probability_scale(probability) * np.sqrt([0.01, 0.01, 0.04])
    assert np.allclose(span, expected, rtol=0.02)


def test_a_higher_probability_draws_a_bigger_ellipsoid():
    renderer = _adp_renderer(
        RenderSettings(show_adp_ellipsoids=True, show_polyhedra=False, adp_probability=0.5)
    )
    renderer.set_adp_tensors(np.tile(_TYPICAL_ADP, (2, 1, 1)))
    at_50 = np.array(renderer._adp_actor.GetBounds()).reshape(3, 2)

    renderer.set_settings(RenderSettings(show_adp_ellipsoids=True, show_polyhedra=False,
                                         adp_probability=0.99))
    at_99 = np.array(renderer._adp_actor.GetBounds()).reshape(3, 2)

    span_50 = (at_50[:, 1] - at_50[:, 0]).max()
    span_99 = (at_99[:, 1] - at_99[:, 0]).max()
    assert span_99 > span_50


def test_tensors_that_do_not_match_the_geometry_are_kept_but_not_drawn():
    """Switching cell view or supercell must not silently discard the file's
    ADPs, but must not draw them against the wrong atoms either."""
    renderer = _adp_renderer()

    renderer.set_adp_tensors(np.tile(_TYPICAL_ADP, (7, 1, 1)))  # seven tensors, two atoms

    assert renderer._adp_actor is None
    assert renderer._atom_actor is not None      # atoms stay drawn as spheres
    assert renderer._adp_tensors is not None     # ...and the tensors are still held


def test_an_atom_with_a_vanishing_tensor_keeps_its_sphere():
    """A null or clamped-flat tensor would render as an invisible speck; that
    atom is better drawn the ordinary way than not at all."""
    renderer = _adp_renderer()

    renderer.set_adp_tensors(np.array([_TYPICAL_ADP, np.zeros((3, 3))]))

    assert list(renderer._ellipsoid_atoms()) == [True, False]
    assert renderer._adp_actor is not None   # the first atom
    assert renderer._atom_actor is not None  # the second


def test_switching_ellipsoids_off_brings_the_spheres_back():
    renderer = _adp_renderer()
    renderer.set_adp_tensors(np.tile(_TYPICAL_ADP, (2, 1, 1)))
    assert renderer._atom_actor is None

    renderer.set_settings(RenderSettings(show_adp_ellipsoids=False))
    assert renderer._adp_actor is None
    assert renderer._atom_actor is not None

    renderer.set_adp_tensors(None)
    assert renderer._adp_actor is None


def test_ellipsoids_travel_with_the_atoms_during_an_animation():
    """A vibration moves the atoms; their displacement tensors don't change. So
    each ellipsoid must slide onto its own atom keeping its shape — not stay
    behind at the equilibrium site, and not be rebuilt every frame."""
    structure = _two_atoms()
    renderer = _adp_renderer(structure=structure)
    renderer.set_adp_tensors(np.array([_TYPICAL_ADP, _TYPICAL_ADP * 2]))
    equilibrium = structure.positions.copy()
    before = np.array(renderer._adp_mesh_obj.points)
    index, _offset = renderer._adp_follow

    # a per-atom displacement, as an eigenvector gives: the two atoms move
    # differently, so a rigid shift of the whole mesh would not do
    shift = np.array([[0.0, 0.0, 0.7], [0.3, 0.0, -0.2]])
    renderer.update_positions(equilibrium + shift)

    after = np.array(renderer._adp_mesh_obj.points)
    for atom in np.unique(index):
        vertices = index == atom
        assert np.allclose(after[vertices] - before[vertices], shift[atom])
        # shape and orientation untouched: only the centre moved
        assert np.allclose(
            after[vertices] - after[vertices].mean(axis=0),
            before[vertices] - before[vertices].mean(axis=0),
        )


def test_ellipsoids_return_to_rest_when_the_animation_stops():
    structure = _two_atoms()
    renderer = _adp_renderer(structure=structure)
    renderer.set_adp_tensors(np.tile(_TYPICAL_ADP, (2, 1, 1)))
    equilibrium = structure.positions.copy()
    at_rest = np.array(renderer._adp_mesh_obj.points)

    renderer.update_positions(equilibrium + np.array([[0.0, 0.0, 0.5], [0.0, 0.0, -0.5]]))
    renderer.update_positions(equilibrium)

    assert np.allclose(np.array(renderer._adp_mesh_obj.points), at_rest)


def test_atoms_still_drawn_as_spheres_animate_alongside_their_ellipsoid_neighbours():
    """A mixed cell — one atom with a usable tensor, one without — must animate
    as a whole, the sphere and the ellipsoid moving together."""
    structure = _two_atoms()
    renderer = _adp_renderer(structure=structure)
    renderer.set_adp_tensors(np.array([_TYPICAL_ADP, np.zeros((3, 3))]))
    assert renderer._adp_actor is not None and renderer._atom_actor is not None

    sphere_before = np.array(renderer._atom_actor.GetBounds()).reshape(3, 2)[2]
    ellipsoid_before = np.array(renderer._adp_actor.GetBounds()).reshape(3, 2)[2]

    renderer.update_positions(structure.positions + np.array([0.0, 0.0, 0.6]))

    assert np.allclose(
        np.array(renderer._atom_actor.GetBounds()).reshape(3, 2)[2], sphere_before + 0.6
    )
    assert np.allclose(
        np.array(renderer._adp_actor.GetBounds()).reshape(3, 2)[2], ellipsoid_before + 0.6
    )


def test_an_ellipsoid_can_be_grabbed_and_dragged_like_an_atom():
    """With ADPs shown the ellipsoid *is* the atom: it is what the user clicks,
    so it has to be what follows the cursor. Its sphere isn't even drawn."""
    structure = _two_atoms()
    renderer = _adp_renderer(structure=structure)
    renderer.set_adp_tensors(np.tile(_TYPICAL_ADP, (2, 1, 1)))
    assert renderer._atom_actor is None  # nothing but ellipsoids to grab

    # a click on the ellipsoid resolves to the atom underneath it
    assert renderer.pick_atom_index(renderer._adp_actor, structure.positions[1]) == 1

    index, _offset = renderer._adp_follow
    before = np.array(renderer._adp_mesh_obj.points)
    renderer.preview_atom_position(1, structure.positions[1] + np.array([0.0, 0.0, 2.0]))
    after = np.array(renderer._adp_mesh_obj.points)

    assert np.allclose(after[index == 1] - before[index == 1], [0.0, 0.0, 2.0])
    assert np.allclose(after[index == 0], before[index == 0])  # the other atom stays put


def test_dragging_carries_the_whole_move_group_s_ellipsoids():
    """Periodic images move with the atom they copy, so their ellipsoids must too
    — otherwise the boundary images tear away from their atoms."""
    from crystalline.core.cells import complete_boundary

    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    packed, _modes = complete_boundary(nacl)
    renderer = _adp_renderer(structure=packed)
    renderer.set_adp_tensors(np.tile(_TYPICAL_ADP, (len(packed), 1, 1)))
    group = renderer.active_move_group(0)
    assert group, "this cell should have periodic images to move as a group"

    index, _offset = renderer._adp_follow
    before = np.array(renderer._adp_mesh_obj.points)
    renderer.preview_atom_position(0, packed.positions[0] + np.array([0.0, 0.0, 1.5]))
    after = np.array(renderer._adp_mesh_obj.points)

    for atom in (0, *group):
        assert np.allclose(after[index == atom] - before[index == atom], [0.0, 0.0, 1.5])
    untouched = [a for a in np.unique(index) if a not in (0, *group)]
    for atom in untouched:
        assert np.allclose(after[index == atom], before[index == atom])


def test_a_dragged_ellipsoid_keeps_its_own_shape():
    """Moving an atom doesn't change its displacement tensor, and two atoms with
    different tensors must not end up sharing one."""
    structure = _two_atoms()
    renderer = _adp_renderer(structure=structure)
    renderer.set_adp_tensors(
        np.array([np.diag([0.02, 0.005, 0.005]), np.diag([0.005, 0.02, 0.005])])
    )

    def shapes():
        index, _offset = renderer._adp_follow
        points = np.array(renderer._adp_mesh_obj.points)
        return [np.ptp(points[index == a], axis=0) for a in (0, 1)]

    before = shapes()
    renderer.preview_atom_position(1, structure.positions[1] + np.array([0.0, 0.0, 3.0]))
    after = shapes()

    assert all(np.allclose(b, a) for b, a in zip(before, after))
    assert not np.allclose(after[0], after[1])  # still distinguishable from each other


def test_polyhedra_stand_down_while_ellipsoids_are_shown():
    """A translucent coordination solid swallows a 0.1 Å ellipsoid whole, so the
    two are never drawn together — but the setting is untouched, and the
    polyhedra come back the moment the ellipsoids go."""
    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    renderer = StructureRenderer(
        pv.Plotter(off_screen=True),
        RenderSettings(show_polyhedra=True, show_adp_ellipsoids=True),
    )
    renderer.set_structure(nacl)
    assert renderer._polyhedra_actor is not None  # no ADPs yet, so polyhedra draw

    renderer.set_adp_tensors(np.tile(_TYPICAL_ADP, (len(nacl), 1, 1)))
    assert renderer._adp_actor is not None
    assert renderer._polyhedra_actor is None
    assert renderer._polyhedra_edge_actor is None  # the outline goes with them
    assert renderer.settings.show_polyhedra is True  # the user's setting is intact

    renderer.set_adp_tensors(None)
    assert renderer._polyhedra_actor is not None


def test_hiding_the_ellipsoids_brings_the_polyhedra_back():
    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    renderer = StructureRenderer(
        pv.Plotter(off_screen=True),
        RenderSettings(show_polyhedra=True, show_adp_ellipsoids=True),
    )
    renderer.set_structure(nacl)
    renderer.set_adp_tensors(np.tile(_TYPICAL_ADP, (len(nacl), 1, 1)))
    assert renderer._polyhedra_actor is None

    renderer.set_settings(RenderSettings(show_polyhedra=True, show_adp_ellipsoids=False))

    assert renderer._adp_actor is None
    assert renderer._polyhedra_actor is not None


def test_appearance_only_settings_skip_the_rebuild():
    """The Display dock streams settings on every slider tick, and a rebuild on a
    large cell is not cheap. Opacities, the background and the projection change
    nothing any mesh depends on, so they are pushed onto the actors in place."""
    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(nacl)

    rebuilds = []
    renderer._rebuild = lambda: rebuilds.append(1)

    base = renderer.settings
    for changed in (
        dict(atom_opacity=0.4),
        dict(polyhedra_opacity=0.9),
        dict(background_color="#202020"),
        dict(parallel_projection=not base.parallel_projection),
    ):
        renderer.set_settings(replace(base, **changed))
        assert rebuilds == [], changed

    # ...and the opacity really reached the actor, rather than being dropped.
    renderer.set_settings(replace(base, atom_opacity=0.25))
    assert renderer._atom_actor.GetProperty().GetOpacity() == pytest.approx(0.25)

    # Anything geometric still rebuilds, including a re-push of identical
    # settings (which is how staged ADP tensors get drawn).
    for changed in (dict(atom_scale=0.7), dict(show_bonds=False),
                    dict(polyhedra_min_vertices=6), {}):
        rebuilds.clear()
        renderer.set_settings(replace(base, **changed))
        assert rebuilds == [1], changed


def test_dragging_a_large_cell_does_not_rescan_hydrogen_bonds():
    """The scan is by far the costliest part of drawing hydrogen bonds (~140 ms on
    a 2400-atom hydrated cell), so the live paths cap it at _LIVE_HBOND_MAX_ATOMS.
    The drag path used to skip that cap and pay it on every mouse-move event."""
    from crystalline.core import bonds
    from crystalline.viz.renderer import _LIVE_HBOND_MAX_ATOMS

    def water_box(n_waters):
        positions, symbols = [], []
        side = float(n_waters) ** (1 / 3) * 4.0
        rng = np.random.default_rng(0)
        for _ in range(n_waters):
            o = rng.random(3) * side
            positions += [o, o + [0.96, 0, 0], o + [-0.24, 0.93, 0]]
            symbols += ["O", "H", "H"]
        return Structure.from_ase(
            Atoms(symbols, positions=positions, cell=np.eye(3) * side * 1.3, pbc=True)
        )

    scans = []
    real_pairs = bonds.hydrogen_bond_pairs
    bonds.hydrogen_bond_pairs = lambda *a, **k: (scans.append(1), real_pairs(*a, **k))[1]
    try:
        for n_waters, expect_scan in ((20, True), (_LIVE_HBOND_MAX_ATOMS, False)):
            structure = water_box(n_waters)
            renderer = StructureRenderer(pv.Plotter(off_screen=True))
            renderer.set_structure(structure)
            target = renderer.atom_position(0)
            scans.clear()
            renderer.preview_atom_position(0, target + 0.05)
            assert bool(scans) is expect_scan, (n_waters, scans)
    finally:
        bonds.hydrogen_bond_pairs = real_pairs


def test_the_ellipsoid_mask_is_not_recomputed_for_every_frame():
    """Which atoms get an ellipsoid depends on the tensors and the probability,
    never on where the atoms are — but it is consulted on every re-glyph, i.e.
    once per animation frame and per drag mouse-move."""
    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    renderer = StructureRenderer(
        pv.Plotter(off_screen=True),
        RenderSettings(show_adp_ellipsoids=True, show_polyhedra=False),
    )
    renderer.set_structure(nacl)
    renderer.set_adp_tensors(np.tile(_TYPICAL_ADP, (len(nacl), 1, 1)))

    computed = []
    real_compute = renderer._compute_ellipsoid_atoms
    renderer._compute_ellipsoid_atoms = lambda: (computed.append(1), real_compute())[1]

    positions = np.asarray(nacl.positions, dtype=float)
    for step in range(3):
        renderer.update_positions(positions + 0.01 * step)
    assert computed == []  # served from the cache throughout

    # New tensors decide anew, so the cache must not survive them.
    renderer.set_adp_tensors(np.tile(_TYPICAL_ADP * 4.0, (len(nacl), 1, 1)))
    renderer.update_positions(positions)
    assert computed == [1]


def test_moving_atoms_reuses_the_atom_actor_instead_of_rebuilding_it():
    """A live position change keeps every sphere's size, colour and triangulation
    and moves only its centre, so it writes the mesh's points rather than
    dropping the actor and re-glyphing. That teardown was the largest single
    cost of an animation frame (~10 ms even at 64 atoms, nearly all of it
    pipeline overhead), and it is what made the whole UI sluggish while a phonon
    animation ran."""
    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(nacl)

    actor = renderer._atom_actor
    mesh = renderer._atom_mesh_obj
    assert actor is not None and mesh is not None
    points_before = np.array(mesh.points)
    colours_before = np.array(mesh["rgb"])

    shift = np.array([0.0, 0.0, 0.7])
    renderer.update_positions(np.asarray(nacl.positions, dtype=float) + shift)

    assert renderer._atom_actor is actor          # same actor, not a replacement
    assert renderer._atom_mesh_obj is mesh        # and the same mesh
    assert np.allclose(mesh.points, points_before + shift)  # every vertex followed
    assert np.array_equal(mesh["rgb"], colours_before)      # colours untouched


def test_a_changed_atom_count_still_rebuilds_the_glyph():
    """The follow map is only valid for the atoms it was built from; anything
    that invalidates it has to fall back to a real rebuild rather than index
    out of range."""
    structure = Structure.empty()
    for k in range(4):
        structure.add_atom("C", [k * 1.6, 0.0, 0.0])
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(structure)
    n_before = renderer._atom_mesh_obj.n_points

    structure.add_atom("O", [0.0, 2.0, 0.0])  # -> listener -> refresh -> rebuild
    renderer.refresh()

    assert renderer._atom_mesh_obj.n_points > n_before
    assert renderer.atom_count == 5
    renderer.update_positions(np.asarray(structure.positions, dtype=float))  # must not raise


def test_proportional_arrows_scale_with_the_displacement():
    """The other reading of an arrow field: length is how far the atom moves,
    the longest at the configured scale. Needed away from Γ, where the cells of
    a travelling wave differ only in phase and uniform arrows would draw them
    all as though they were doing the same thing."""
    structure = Structure.empty()
    structure.set_cell(np.eye(3) * 40, periodic=False)
    structure.add_atom("C", [0.0, 0.0, 0.0])
    structure.add_atom("C", [0.0, 0.0, 10.0])
    renderer = _arrow_renderer(
        RenderSettings(
            show_mode_arrows=True, show_polyhedra=False,
            mode_arrow_scale=2.0, mode_arrow_proportional=True,
        ),
        structure,
    )

    # the far atom moves a fifth as far: 10 Å apart, one full arrow + one fifth
    renderer.set_mode_vectors(np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 0.2]]))

    assert _arrow_extent(renderer) == pytest.approx(10.0 + 2.0 + 0.4, abs=0.05)


def test_proportional_arrows_still_ignore_the_eigenvector_norm():
    """Only the *ratios* within a mode are physical — CRYSTALClear renormalises
    the eigenvector, so the absolute size must not reach the screen."""
    renderer = _arrow_renderer(
        RenderSettings(
            show_mode_arrows=True, show_polyhedra=False,
            mode_arrow_scale=2.0, mode_arrow_proportional=True,
        )
    )

    renderer.set_mode_vectors(np.array([[0.0, 0.01, 0.0], [0.0, 0.0, 0.0]]))
    tiny = _arrow_extent(renderer)
    renderer.set_mode_vectors(np.array([[0.0, 100.0, 0.0], [0.0, 0.0, 0.0]]))
    huge = _arrow_extent(renderer)

    assert tiny == pytest.approx(huge, abs=1e-6)
    assert huge == pytest.approx(2.0, abs=0.05)  # the largest arrow is the scale


def test_arrows_are_coloured_by_phase_only_when_it_says_something():
    """Away from Γ the arrows differ from cell to cell only in phase, so that is
    what the colour has to carry. At Γ — one phase for the whole cell — a colour
    ramp would invent a distinction, so the flat arrow colour is kept."""
    settings = RenderSettings(
        show_mode_arrows=True, show_polyhedra=False, mode_arrow_phase_colors=True
    )
    renderer = _arrow_renderer(settings)
    vectors = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    keep = np.array([True, True])

    renderer.set_mode_vectors(vectors, phases=np.array([0.0, np.pi]))
    assert renderer._arrow_actor is not None
    assert np.allclose(renderer._arrow_phases(keep), [0.0, np.pi])

    # wrapped into [0, 2π): the colormap is cyclic, so -π/2 and 3π/2 are one colour
    renderer.set_mode_vectors(vectors, phases=np.array([0.0, -np.pi / 2]))
    assert np.allclose(renderer._arrow_phases(keep), [0.0, 3 * np.pi / 2])

    renderer.set_mode_vectors(vectors, phases=np.zeros(2))  # Γ: one phase
    assert renderer._arrow_phases(keep) is None
    renderer.set_mode_vectors(vectors, phases=None)  # no phase at all
    assert renderer._arrow_phases(keep) is None
    renderer.set_mode_vectors(vectors, phases=np.zeros(5))  # stale, wrong length
    assert renderer._arrow_phases(keep) is None


def test_phase_colours_are_off_unless_asked_for():
    renderer = _arrow_renderer(
        RenderSettings(show_mode_arrows=True, show_polyhedra=False)  # setting off
    )

    renderer.set_mode_vectors(
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]), phases=np.array([0.0, np.pi])
    )

    assert renderer._arrow_actor is not None  # still drawn, just one colour
    assert renderer._arrow_phases(np.array([True, True])) is None


def test_toggling_the_gizmo_off_and_on_leaves_it_visible():
    """The regression: the corner widget must survive a round trip.

    There is one orientation-marker widget per renderer, and pyvista's
    ``add_axes`` replaces it — shrinking the widget it drops to a 0.0001-wide
    viewport on the way out. Re-enabling that discarded widget, which is what
    turning the gizmo back on used to do, put it in a corner too small to see
    while leaving the xyz marker enabled underneath it. So: one widget
    throughout, always at the full gizmo viewport.
    """
    from crystalline.viz.renderer import _GIZMO_VIEWPORT

    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64)))
    widget = renderer._orientation_widget
    assert widget is not None

    renderer.set_settings(RenderSettings(show_lattice_vectors=False,
                                         show_orientation_axes=True))
    renderer.set_settings(RenderSettings(show_lattice_vectors=True))

    assert renderer._orientation_widget is widget  # never replaced
    assert renderer.plotter.renderer.axes_widget is widget  # and none added beside it
    assert tuple(widget.GetViewport()) == _GIZMO_VIEWPORT
    assert widget.GetEnabled()


def test_a_structure_with_no_lattice_still_gets_an_orientation_marker():
    """The two markers share one slot, and the gizmo has first claim — but when
    it has nothing to draw the xyz marker must take the corner rather than the
    corner going empty."""
    from ase.build import molecule

    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_settings(RenderSettings(show_lattice_vectors=True,
                                         show_orientation_axes=True))
    renderer.set_structure(Structure.from_ase(molecule("H2O")))

    assert renderer._lattice_marker() is None  # no cell to draw arrows for
    assert renderer._orientation_widget is not None
    assert renderer._orientation_widget.GetEnabled()


def test_turning_both_markers_off_empties_the_corner():
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64)))
    renderer.set_settings(RenderSettings(show_lattice_vectors=False,
                                         show_orientation_axes=False))
    assert not renderer._orientation_widget.GetEnabled()


# ── coordination polyhedra on a supercell ────────────────────────────────
def _srtio3_supercell(n):
    from ase.spacegroup import crystal

    from crystalline.core.cells import tile_supercell

    unit = Structure.from_ase(crystal(["Sr", "Ti", "O"], [(0, 0, 0), (.5, .5, .5), (.5, .5, 0)],
                                      spacegroup=221, cellpar=[3.905] * 3 + [90] * 3))
    tiled = tile_supercell(unit, (n, n, n))
    return unit, (tiled[0] if isinstance(tiled, tuple) else tiled)


def test_a_supercell_is_analysed_as_one_cell_so_its_size_changes_nothing(monkeypatch):
    """Analysed whole, a 5x5x5 supercell passed CrystalNN's size limit and fell
    to the distance fallback, and the polyhedra changed with the supercell."""
    pytest.importorskip("pymatgen")
    import crystalline.core.bonds as bonds
    from crystalline.viz.renderer import _one_cell

    unit, big = _srtio3_supercell(5)
    analysed = []
    real = bonds.connectivity
    monkeypatch.setattr(bonds, "connectivity",
                        lambda s, *a, **k: analysed.append(len(s)) or real(s, *a, **k))

    folded = _one_cell(big, np.asarray(unit.cell))
    renderer = _renderer_for_polyhedra()
    found = renderer._coordination_polyhedra(folded, 4, bonds.connectivity)

    assert len(folded) == len(unit)
    assert analysed == [len(unit)]
    assert {int(z): len(lig) for z, _c, lig in found} == {38: 12, 22: 6}


def test_an_edited_supercell_is_not_folded():
    """Not periodic any more: one tile differs from the others."""
    from crystalline.viz.renderer import _one_cell

    unit, big = _srtio3_supercell(2)
    atoms = big.to_ase()
    del atoms[0]
    edited = Structure.from_ase(atoms)

    assert len(_one_cell(edited, np.asarray(unit.cell))) == len(edited)


def test_a_strained_cell_is_analysed_again_even_with_the_atoms_in_place():
    """The cache was keyed on the positions alone."""
    pytest.importorskip("pymatgen")
    from crystalline.core.bonds import connectivity

    unit, _big = _srtio3_supercell(1)
    renderer = _renderer_for_polyhedra()
    renderer._coordination_polyhedra(unit, 4, connectivity)
    strained = unit.to_ase()
    strained.set_cell(np.asarray(unit.cell) * 1.4, scale_atoms=False)
    strained = Structure.from_ase(strained)

    found = renderer._coordination_polyhedra(strained, 4, connectivity)

    assert renderer._poly_cache_key[2] == np.round(np.asarray(strained.cell), 4).tobytes()
    assert found is renderer._poly_cache_found


def _renderer_for_polyhedra():
    import pyvista as pv

    from crystalline.viz.renderer import StructureRenderer

    return StructureRenderer(pv.Plotter(off_screen=True))
