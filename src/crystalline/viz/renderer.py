"""Draw a :class:`~crystalline.core.structure.Structure` into a PyVista plotter.

Design notes
------------
* **All atoms are drawn as a single glyphed mesh** (one VTK actor), not one
  actor per atom. VTK's per-frame cost scales with the number of *actors*, so a
  cell with thousands of atoms (a ZIF-8 supercell, say) went from interactive to
  tens of seconds — and eventually a crash — under the old one-actor-per-atom
  scheme. Glyphing collapses that to a single actor whose build/redraw is ~1000×
  faster at those sizes. Per-atom **picking** maps a pick's world position back
  to the nearest atom centre; **position updates** (drag / phonon animation)
  re-glyph the point cloud, which is cheap now that it is one mesh.
* Element colours and radii come from ``ase.data`` (Jmol colours, covalent
  radii) so they match community conventions.
* No Qt here: the plotter is supplied by the caller (a ``pyvistaqt`` interactor
  in the app, or an off-screen ``pyvista.Plotter`` in tests).
"""

from __future__ import annotations

import itertools
from functools import lru_cache
from typing import Optional

import numpy as np
import pyvista as pv
import vtk
from ase.data import chemical_symbols
from scipy.spatial import cKDTree

from crystalline.core import elements
from crystalline.core.structure import Structure
from crystalline.viz.fonts import unicode_font, use_unicode_font
from crystalline.viz.render_settings import RenderSettings

# Fraction of the covalent radius used for the drawn sphere (ball-and-stick).
# Bond radius/tolerance are per-view settings — see :class:`RenderSettings`.
_ATOM_SCALE = 0.5

# Covalent bonds are a single fixed grey (no per-bond colouring). Hydrogen bonds
# are drawn as thin dashed light-blue lines, VESTA-style.
_HBOND_DASH = 0.28   # Å: dash length
_HBOND_GAP = 0.20    # Å: gap between dashes

# Measurement annotations (Geometry panel). Colours are per-view settings now
# (measure_point/line/plane_color); this warm accent is only the fallback default.
_ANNOTATION_COLOR = "#ff7f0e"
_ANNOTATION_LINE_RADIUS = 0.035
_ANNOTATION_POINT_RADIUS = 0.18
_ANNOTATION_FONT_SIZE = 13
_ANNOTATION_PLANE_OPACITY = 0.28
_ANNOTATION_PLANE_MARGIN = 1.25   # patch overhang beyond the fitted atoms
_ANNOTATION_PLANE_MIN_SIZE = 2.0  # Angstrom, so a tight plane is still visible
# Depth bias that lifts annotations in front of the atoms they measure.
_ANNOTATION_DEPTH_OFFSET = -66000.0

# Symmetry elements (Symmetry panel). Drawn with the same three shapes as the
# measurements — a tube for an axis, a patch for a plane, a dot for the centre —
# but thinner and fainter: several of them cross the same structure, and they are
# context for it rather than the thing being looked at.
_SYMMETRY_LINE_RADIUS = 0.025
_SYMMETRY_POINT_RADIUS = 0.12
_SYMMETRY_PLANE_OPACITY = 0.16
_SYMMETRY_FONT_SIZE = 12
# How far past the atoms (and the cell) an element is drawn, in Angstrom.
_SYMMETRY_BOUNDS_MARGIN = 0.5
# How far back from its far end an element's label sits, as a fraction of the
# distance to the centre — see _label_anchor.
_SYMMETRY_LABEL_INSET = 0.08

# Coordination-polyhedra outline: only edges where adjacent faces bend by more
# than this are real polyhedron edges (the rest are the hull's triangulation of
# a flat face), drawn this wide in this fraction of the face colour.
_POLYHEDRA_EDGE_ANGLE = 15.0
_POLYHEDRA_EDGE_SHADE = 0.55

# Material for the atom spheres. The specular highlight is what makes small
# on-screen atoms read as 3D balls rather than flat disks.
_ATOM_MATERIAL = dict(
    smooth_shading=True,
    specular=0.5,
    specular_power=15,
    ambient=0.3,
    diffuse=0.6,
)

# Above these atom counts, bonds are not recomputed on every live position update
# (animation frames / drag) — they refresh on commit instead — to avoid lag.
#
# Covalent bonds are only a KD-tree pass and one tube mesh, which measures well
# under a millisecond per thousand atoms, so the limit is generous: a large cell
# like a MOF must still show its bonds moving with the atoms during a phonon
# animation. Hydrogen bonds cost far more (a per-hydrogen Python scan), so their
# own limit stays low — unless the topology is frozen, in which case the scan
# runs once per animation and each frame is only an array gather.
_LIVE_BOND_MAX_ATOMS = 5000
_LIVE_HBOND_MAX_ATOMS = 400

# Above this atom count, per-atom element labels are suppressed: they'd be an
# unreadable, slow-to-render cloud of text on a large cell.
_ATOM_LABEL_MAX_ATOMS = 400

# Fractional tolerance for deciding two atoms are periodic images (their
# separation is a whole lattice translation). Boundary/supercell images are exact
# lattice shifts, so this can be tight.
_IMAGE_FRAC_TOL = 1e-3

# Lattice-vector indicator: small a/b/c arrows near the structure. Colours follow
# the common convention a=red, b=green, c=blue (as in VESTA).
_LATTICE_COLORS = ("#d62728", "#2ca02c", "#1f77b4")
_LATTICE_LABELS = ("a", "b", "c")
# Fixed arrow length (Å): the gizmo shows only the a/b/c *directions*, so it stays
# the same size regardless of the lattice parameters (editing a/b/c must not
# grow or shrink it). Comparable to a bond length, so it reads well beside atoms.
_LATTICE_ARROW_LENGTH = 2.0
# How far past the arrow tip its label sits, as a multiple of the arrow length.
_LATTICE_LABEL_OFFSET = 1.18
# Corner of the render window the gizmo occupies, as (xmin, ymin, xmax, ymax)
# in normalised viewport coordinates: the lower left, a fifth of each side.
_GIZMO_VIEWPORT = (0.0, 0.0, 0.24, 0.24)
# Label glyph size: a caption scales its text to its own normalised box, so the
# box drives how big the letter comes out next to the arrows.
_LATTICE_LABEL_BOX = 0.05
_LATTICE_LABEL_FONT_SIZE = 20
# The widget frames the marker's 3D bounds, and a 2D caption contributes none —
# so a label anchored at an arrow tip lands on the very edge of the viewport and
# gets clipped. An invisible sphere this many arrow-lengths across pads the
# bounds so the letters have room.
_GIZMO_BOUNDS_PADDING = 1.3

# An atom whose displacement is below this fraction of the mode's largest gets
# no arrow. Arrows are drawn at a uniform length, so without this cut-off an
# atom that barely moves would be flagged as strongly as one that carries the
# mode — and in a big cell most atoms barely move in any given mode.
_MIN_ARROW_FRACTION = 0.05

# Colormap for the per-atom Bloch phase of a mode away from Gamma. Cyclic (its
# two ends are the same colour), because phase is an angle: any other map would
# draw a false discontinuity somewhere in the middle of a perfectly smooth wave.
#
# ``hsv`` over the perceptually better cyclic maps (twilight and friends) for
# one reason: this is read off *arrows*, a few pixels wide. Twilight spends its
# saturated tones in the middle of the cycle and its ends on near-greys, which
# on a thin glyph come out as "grey, black, grey, white" — the phases are all
# there and none of them are legible. The job here is telling a handful of
# discrete phases apart at a glance, and hsv does that.
_PHASE_COLORMAP = "hsv"

# Crystalline-orbital isosurface. The two lobes are the two signs of the
# amplitude — the phase is the whole point of showing an orbital rather than a
# density — in the red/blue every textbook and VESTA use. A modulus field has
# only one sign, and gets the positive colour alone.
_ORBITAL_POSITIVE_COLOR = "#d6453c"
_ORBITAL_NEGATIVE_COLOR = "#3f6fd8"
_ORBITAL_OPACITY = 0.85
# Where the surface is cut, relative to the orbital's strongest point rather than
# as an absolute amplitude: a crystalline orbital is spread over a whole cell, so
# the ~0.02 a.u. that suits a compact molecular orbital encloses nearly the
# entire box here. The dialog exposes it because the right value depends on how
# delocalised the orbital is.
DEFAULT_ORBITAL_ISOVALUE = 0.2


# An ADP ellipsoid whose longest semi-axis is below this (Angstrom) is too
# small to read on screen; that atom keeps its ordinary sphere instead of
# turning into an invisible speck.
_MIN_ELLIPSOID_RADIUS = 0.01


# Settings that change only how the *existing* actors are drawn — nothing about
# the geometry, the connectivity or which atoms get which mesh depends on them.
# A change confined to these is pushed onto the actors in place instead of
# rebuilding the scene, which matters because the Display dock streams settings
# continuously while a slider is dragged: an opacity drag would otherwise pay a
# full teardown-and-redraw per tick.
#
# Anything not listed here (sizes, radii, tolerances, colours that live in a
# mesh's scalar array, every ``show_*`` toggle, the ADP probability and
# temperature) does change geometry, and still rebuilds.
_APPEARANCE_ONLY_SETTINGS = frozenset({
    "atom_opacity",
    "adp_opacity",
    "polyhedra_opacity",
    "mode_arrow_color",
    # Colours and line widths of actors that exist already: set on their
    # properties, never worth a rebuild.
    "bond_color",
    "hydrogen_bond_color",
    "hydrogen_bond_width",
    "cell_color",
    "cell_line_width",
    "atom_label_color",
    "background_color",
    "parallel_projection",
    # Both markers live in the corner widget, which _apply_scene_settings
    # updates — no actor and no geometry depends on either.
    "show_orientation_axes",
    "show_lattice_vectors",
})


def _appearance_only_change(previous: RenderSettings, current: RenderSettings) -> bool:
    """Whether ``current`` differs from ``previous`` *only* in drawn appearance.

    False when nothing changed at all, so a caller that re-pushes identical
    settings to force a redraw — staging ADP tensors with ``redraw=False`` and
    then applying the settings that name them, say — still gets one.
    """
    from dataclasses import fields

    changed = {
        f.name
        for f in fields(RenderSettings)
        if getattr(previous, f.name) != getattr(current, f.name)
    }
    return bool(changed) and changed <= _APPEARANCE_ONLY_SETTINGS


def _sphere_resolution(n_atoms: int) -> int:
    """Fewer triangles per sphere for larger systems, to keep rendering fast."""
    if n_atoms <= 100:
        return 32
    if n_atoms <= 500:
        return 16
    return 8


class StructureRenderer:
    """Renders atoms, bonds and the unit cell for a structure."""

    def __init__(
        self, plotter: pv.BasePlotter, settings: Optional[RenderSettings] = None
    ) -> None:
        self.plotter = plotter
        self._settings = settings if settings is not None else RenderSettings()
        self._structure: Optional[Structure] = None
        # Single glyphed actor for every atom (see module docstring). The live
        # positions/numbers/radii back both re-glyphing and pick→index mapping.
        self._atom_actor = None
        self._atom_mesh_obj = None    # kept so a live update can move its points
        self._atom_follow = None      # (atom index, offset) per sphere vertex
        self._positions: np.ndarray = np.empty((0, 3), dtype=float)
        self._numbers: np.ndarray = np.empty(0, dtype=int)
        self._radii: np.ndarray = np.empty(0, dtype=float)
        # Cached periodic-image group for the atom currently being dragged.
        self._drag_primary: Optional[int] = None
        self._drag_group: list = []
        self._bond_actor = None
        self._hbond_actor = None
        self._cell_actor = None
        self._polyhedra_actor = None
        self._polyhedra_edge_actor = None
        self._polyhedra_mesh_obj = None       # kept so animation can move its points
        self._polyhedra_edge_mesh = None
        self._polyhedra_follow = None         # (atom index, offset) per hull vertex
        self._polyhedra_edge_follow = None
        self._label_actor = None
        self._arrow_actor = None
        # (N, 3) eigenvector of the selected phonon mode, drawn as per-atom arrows.
        self._mode_vectors: Optional[np.ndarray] = None
        # (N,) Bloch phase of each atom's cell, when the mode is away from Gamma.
        self._mode_phases: Optional[np.ndarray] = None
        self._adp_actor = None
        self._adp_mesh_obj = None     # kept so animation can move its points
        self._adp_follow = None       # (atom index, offset) per ellipsoid vertex
        # (N, 3, 3) cartesian ADP tensors in Angstrom^2, drawn as ellipsoids.
        self._adp_tensors: Optional[np.ndarray] = None
        # Which atoms those tensors are drawn on, cached across live position
        # updates — see :meth:`_ellipsoid_atoms`.
        self._ellipsoid_mask: Optional[np.ndarray] = None
        self._annotations: list = []          # measurements drawn over the structure
        self._annotation_actors: list = []
        self._symmetry_elements: list = []    # symmetry elements drawn over the structure
        self._symmetry_actors: list = []
        self._symmetry_labels = False         # write each element's symbol beside it
        self._bond_structure: Optional[Structure] = None  # clean cell for coordination
        # Geometry that decides *which* atoms are bonded while something moves the
        # atoms without changing the chemistry (a phonon animation). None means
        # connectivity is re-derived from the drawn frame, which is what editing wants.
        self._bond_reference: Optional[np.ndarray] = None
        # Connectivity derived from that reference, computed once and reused for
        # every frame of an animation instead of being rebuilt per frame.
        self._frozen_bond_pairs: Optional[tuple] = None
        self._frozen_hbond_pairs: Optional[np.ndarray] = None
        # Cache the (expensive) CrystalNN coordination analysis so a rebuild that
        # only changed display settings — a slider nudge, a colour, a toggle —
        # doesn't re-run it. Keyed on the analysed geometry + min-coordination.
        self._poly_cache_key = None
        self._poly_cache_found: Optional[list] = None
        self._highlight_actors: dict[int, object] = {}  # atom index -> halo actor
        # If set, the cell wireframe outlines this cell (the original unit cell)
        # instead of the structure's own cell — used when a supercell is shown.
        self._reference_cell: Optional[np.ndarray] = None
        # Screen-space a/b/c gizmo. Outlives plotter.clear(), unlike an actor,
        # so it is created once and its marker swapped on each rebuild.
        self._orientation_widget = None
        # Crystalline-orbital isosurface: one actor per sign, kept so they can be
        # cleared without disturbing the rest of the scene.
        self._orbital_actors: list = []
        self._orbital_field = None
        # A scalar field read from a CRYSTAL run — the charge density, the spin
        # density, the potential. Kept the same way, and separate from the
        # orbital: the two can be on screen together.
        self._density_actors: list = []
        self._density_view = None
        self._density_miller_cell = None
        self._slice_frame = None       # (centre, normal, in-plane up) of a drawn slice
        self._slice_extent = None      # (point, u, w, u range, w range) of its rectangle
        self._surface_cache = None     # (key, pieces) of the last contoured field
        self._density_bar = None       # title of the field's colour bar, if one is shown
        self._cutaway = None

    # ── public API ──────────────────────────────────────────────────────
    @property
    def settings(self) -> RenderSettings:
        return self._settings

    def set_settings(self, settings: RenderSettings) -> None:
        """Apply new appearance settings, redrawing only as much as they need.

        A change confined to :data:`_APPEARANCE_ONLY_SETTINGS` is pushed straight
        onto the actors; anything else rebuilds the scene. The Display dock emits
        on every slider tick, so making the cheap changes cheap is what keeps
        dragging an opacity slider from freezing the view on a large cell.
        """
        previous = self._settings
        self._settings = settings
        if _appearance_only_change(previous, settings):
            self._apply_appearance()
            self.plotter.render()
            return
        self._rebuild()

    def _apply_appearance(self) -> None:
        """Push the appearance-only settings onto the live actors.

        An unparsable colour raises, exactly as it does when ``add_mesh`` is
        handed one during a rebuild — ``pv.Color`` is the same parser, so there is
        no failure here that rebuilding would recover from.
        """
        settings = self._settings
        self._apply_scene_settings()  # background, projection, orientation marker
        for actor, opacity in (
            (self._atom_actor, settings.atom_opacity),
            (self._adp_actor, settings.adp_opacity),
            (self._polyhedra_actor, settings.polyhedra_opacity),
        ):
            if actor is not None:
                actor.GetProperty().SetOpacity(opacity)
        if self._arrow_actor is not None:
            self._arrow_actor.GetProperty().SetColor(
                *pv.Color(settings.mode_arrow_color).float_rgb
            )
        for actor, colour in ((self._bond_actor, settings.bond_color),
                              (self._hbond_actor, settings.hydrogen_bond_color),
                              (self._cell_actor, settings.cell_color)):
            if actor is not None:
                actor.GetProperty().SetColor(*pv.Color(colour).float_rgb)
        if self._hbond_actor is not None:
            self._hbond_actor.GetProperty().SetLineWidth(settings.hydrogen_bond_width)
        if self._cell_actor is not None:
            self._cell_actor.GetProperty().SetLineWidth(settings.cell_line_width)
        if self._label_actor is not None:
            # The background is on this path too, and labels that follow it
            # have to change with it.
            try:
                text = self._label_actor.GetMapper().GetInputAlgorithm().GetTextProperty()
                text.SetColor(*pv.Color(self._label_colour()).float_rgb)
            except Exception:  # noqa: BLE001 - labels are redrawn right on a rebuild
                pass

    def set_bond_reference(self, positions: Optional[np.ndarray]) -> None:
        """Fix the bond network to the geometry at ``positions`` (``None`` clears).

        A phonon animation moves atoms far off their sites without changing the
        chemistry, so connectivity must come from the equilibrium geometry rather
        than from each frame — otherwise a stretch that crosses the bond-length
        criterion makes bonds flicker, and the larger the amplitude the worse it
        gets. Editing leaves this ``None`` so moving an atom really does re-bond it.
        """
        self._bond_reference = None if positions is None else np.asarray(positions, dtype=float)
        self._frozen_bond_pairs = None
        self._frozen_hbond_pairs = None

    def set_mode_vectors(self, vectors: Optional[np.ndarray], phases=None) -> None:
        """Draw ``vectors`` as a per-atom arrow field (``None`` clears them).

        Used for the selected phonon mode's eigenvector: an animation shows the
        motion but can't be put in a figure, and a still frame of a vibrating
        structure looks like a distorted one. Arrows say which atoms move, how
        far, and in which direction, all at once.

        ``phases`` is the optional per-atom Bloch phase (radians) of a mode away
        from Gamma — the cell-to-cell lag that *is* the wave. With
        ``mode_arrow_phase_colors`` set it colours the arrows, which is the only
        way a still image can carry it: the amplitude is the same in every cell.

        A vector set that doesn't match the atom count is simply not drawn — the
        displayed structure can be swapped (supercell, cell view, an edit) while
        a mode is still selected.
        """
        self._mode_vectors = None if vectors is None else np.asarray(vectors, dtype=float)
        self._mode_phases = None if phases is None else np.asarray(phases, dtype=float)
        self._redraw_mode_arrows()
        self.plotter.render()

    def set_reference_cell(self, cell: Optional[np.ndarray]) -> None:
        """Choose which cell the wireframe outlines (``None`` = the structure's own).

        Set to the original unit cell to keep its outline while a supercell of
        atoms is displayed. Takes effect on the next rebuild.
        """
        self._reference_cell = None if cell is None else np.asarray(cell, dtype=float)

    def set_structure(self, structure: Structure, bond_structure: Optional[Structure] = None) -> None:
        """Draw ``structure`` from scratch, replacing anything shown.

        ``bond_structure`` (optional) is a clean periodic cell used for the
        chemically-aware coordination analysis behind polyhedra — passed
        separately because the displayed structure may be a boundary-completed
        "packed" cell (with duplicate images) that a near-neighbour algorithm
        can't analyse.
        """
        self._structure = structure
        self._bond_structure = bond_structure
        self._rebuild()

    def refresh(self) -> None:
        """Redraw after the current structure was edited (add/remove atoms)."""
        self._rebuild()

    def update_positions(self, positions: np.ndarray) -> None:
        """Move all atoms without a full rebuild (re-glyphs the atom mesh).

        Used by animation: the atom *count* must be unchanged. Bonds follow the
        atoms for small systems (recomputed here); for large systems bonds are
        left in place during animation to keep it smooth.
        """
        if len(self._positions) != len(positions):
            raise ValueError("position count changed; call refresh() instead")
        self._positions = np.asarray(positions, dtype=float)
        self._reglyph_atoms()
        self._update_live_bonds()
        self._update_polyhedra(self._positions)
        self._update_adp_ellipsoids(self._positions)
        if self._mode_vectors is not None:
            self._redraw_mode_arrows()  # arrows ride along with the atoms
        self.plotter.render()

    @property
    def atom_count(self) -> int:
        """Number of atoms currently drawn (for size-sanity checks)."""
        return len(self._positions)

    def pick_atom_index(self, actor, world_pos) -> Optional[int]:
        """Map a pick (its actor + world position) back to an atom index.

        Only the single atom-glyph actor is pickable, so a hit on it is always an
        atom; the picked surface point is resolved to the nearest atom centre.
        Returns ``None`` for a miss or a hit on any non-atom prop.
        """
        if actor is None or len(self._positions) == 0:
            return None
        if actor is not self._atom_actor and actor is not self._adp_actor:
            return None
        _dist, index = cKDTree(self._positions).query(np.asarray(world_pos, dtype=float))
        return int(index)

    def atom_position(self, index: int) -> np.ndarray:
        """Current cartesian position of atom ``index`` (from the model)."""
        return np.asarray(self._structure.positions[index], dtype=float)

    def rendered_atom_position(self, index: int) -> np.ndarray:
        """Where atom ``index`` is *currently drawn* (moves live during a drag /
        animation, before the model is updated)."""
        return np.asarray(self._positions[index], dtype=float)

    def periodic_image_indices(self, index: int) -> list:
        """Indices of atoms that are periodic images of ``index``.

        Two atoms are periodic images when they are the same element and separated
        by a whole lattice translation of the reference cell (the original unit
        cell when a supercell is shown, otherwise the structure's own cell). These
        are the atoms that must move together with ``index`` during a drag so the
        periodicity is preserved. The dragged atom itself is not included.
        """
        if self._structure is None or len(self._positions) < 2:
            return []
        lattice = self._reference_cell if self._reference_cell is not None else self._structure.cell
        lattice = np.asarray(lattice, dtype=float)
        if np.allclose(lattice, 0.0):
            return []
        base = np.asarray(self._structure.positions, dtype=float)
        diff = base - base[index]
        try:  # fractional coordinates of each atom relative to `index`
            frac = np.linalg.solve(lattice.T, diff.T).T
        except np.linalg.LinAlgError:  # singular (degenerate) cell
            return []
        integral = np.all(np.abs(frac - np.rint(frac)) < _IMAGE_FRAC_TOL, axis=1)
        same_element = self._numbers == self._numbers[index]
        not_self = np.arange(len(base)) != index
        return list(np.nonzero(integral & same_element & not_self)[0])

    def active_move_group(self, index: int) -> list:
        """Indices that should move *with* ``index`` during a drag (excluding it).

        If ``index`` is part of a multi-atom selection (has a halo, and it isn't
        the only one), the whole selection moves together — each selected atom
        also carrying its own periodic images — so an imported/selected fragment
        drags as one piece. Otherwise only ``index``'s periodic images move.
        """
        selected = set(self._highlight_actors.keys())
        if index in selected and len(selected) > 1:
            group: set = set()
            for atom in selected:
                group.add(atom)
                group.update(self.periodic_image_indices(atom))
            group.discard(index)
            return sorted(group)
        return self.periodic_image_indices(index)

    def preview_atom_position(self, index: int, world_pos: np.ndarray) -> None:
        """Move an atom live (with its move group), not the model.

        Used during an interactive drag: the dragged atom follows the cursor and
        every atom in its move group (periodic images, plus the rest of a
        multi-atom selection) moves by the same cartesian shift. The model is
        updated once on drop; bonds redraw with the ensuing ``refresh``. Does not
        call ``render`` — the caller does.

        "The atom" may be a sphere or a thermal ellipsoid — when ADPs are shown
        the ellipsoid *is* the atom, and it is what the user grabbed, so it has
        to be what follows the cursor. Its shape and orientation don't change:
        moving an atom doesn't alter its displacement tensor.
        """
        world_pos = np.asarray(world_pos, dtype=float)
        base = np.asarray(self._structure.positions, dtype=float)
        delta = world_pos - base[index]
        # The move group is fixed for the duration of a drag; cache it per atom.
        if index != self._drag_primary:
            self._drag_primary = index
            self._drag_group = self.active_move_group(index)
        for i in (index, *self._drag_group):
            self._positions[i] = base[i] + delta
            if i in self._highlight_actors:
                self._highlight_actors[i].SetPosition(*delta)
        self._reglyph_atoms()
        self._update_adp_ellipsoids(self._positions)
        self._update_live_bonds()  # keep the bonds attached to the dragged atoms

    def highlight(self, indices) -> None:
        """Draw translucent halos around the selected atoms.

        ``indices`` may be a single atom index, an iterable of indices, or
        ``None`` / empty to clear the selection.
        """
        for actor in self._highlight_actors.values():
            self.plotter.remove_actor(actor, render=False)
        self._highlight_actors = {}

        if indices is None:
            wanted = []
        elif isinstance(indices, (int, np.integer)):
            wanted = [int(indices)]
        else:
            wanted = [int(i) for i in indices]

        if self._structure is not None:
            for index in wanted:
                if not 0 <= index < len(self._structure):
                    continue
                pos = self._structure.positions[index]
                r = _sphere_radius(self._structure.numbers[index], self._settings.atom_scale) * 1.6
                halo = pv.Sphere(radius=r, center=pos)
                actor = self.plotter.add_mesh(
                    halo, color="yellow", opacity=0.35, name=f"__highlight_{index}__"
                )
                # Halos must never intercept picks, or they would block grabbing
                # the selected atoms (and any atom their larger radius overlaps).
                actor.SetPickable(False)
                self._highlight_actors[index] = actor
        self.plotter.render()

    # ── internals ───────────────────────────────────────────────────────
    def _ensure_lights(self) -> None:
        """Re-add a default light kit if the renderer has no lights.

        ``Plotter.clear()`` strips lights; without this, spheres render as flat
        ambient-lit disks instead of shaded 3D balls.
        """
        if not self.plotter.renderer.lights:
            self.plotter.enable_lightkit()

    def _apply_scene_settings(self) -> None:
        """Apply background, projection and the orientation-axes marker.

        These are scene/camera properties (not per-object actors), so they're set
        on every rebuild and independently of whether a structure is loaded.
        Guarded because a bare/off-screen plotter may not support every hook.
        """
        settings = self._settings
        try:
            self.plotter.set_background(settings.background_color)
        except Exception:  # noqa: BLE001 - invalid colour string, etc.
            pass
        try:
            if settings.parallel_projection:
                self.plotter.enable_parallel_projection()
            else:
                self.plotter.disable_parallel_projection()
        except Exception:  # noqa: BLE001
            pass
        self._update_orientation_marker()

    def _update_orientation_marker(self) -> None:
        """Put the right marker in the one corner widget — or empty it.

        A VTK renderer has a *single* orientation-marker widget, so the a/b/c
        gizmo and the stock xyz marker are two candidates for one slot. Both are
        decided here and shown through our own widget, and nothing else may
        touch it. In particular ``plotter.add_axes()`` must not be called: it
        builds a *second* widget and, on the way, shrinks the one it replaces to
        a 0.0001-wide viewport before forgetting it. Turning the gizmo off and
        on again therefore re-enabled that discarded widget — the gizmo came
        back in a corner too small to see, with the xyz marker still drawn
        underneath it.

        The lattice gizmo wins when it is on: it says everything the xyz marker
        does, and says it about the actual cell. When it has nothing to draw —
        a molecule, or a structure with no lattice — the xyz marker takes the
        corner instead, so asking for an orientation marker always gets one.
        """
        settings = self._settings
        marker = self._lattice_marker() if settings.show_lattice_vectors else None
        if marker is None and settings.show_orientation_axes:
            marker = pv.create_axes_marker()
        self._set_orientation_widget(marker)

    def _rebuild(self) -> None:
        # Preserve the current view across the rebuild. clear() drops pyvista's
        # "camera set" flag, so the next add_mesh would auto-fit the camera and
        # zoom the view out on every edit (e.g. dropping a dragged atom). Framing
        # is the viewport's job; a redraw must leave the camera exactly as it was.
        saved_camera = self._current_camera()
        # NB: Plotter.clear() also removes the renderer's lights, which would
        # leave every sphere flat-shaded (ambient only). Restore them after.
        self.plotter.clear()
        self._ensure_lights()
        self._apply_scene_settings()
        self._atom_actor = None
        self._atom_mesh_obj = None
        self._atom_follow = None
        self._positions = np.empty((0, 3), dtype=float)
        self._numbers = np.empty(0, dtype=int)
        self._radii = np.empty(0, dtype=float)
        self._drag_primary = None
        self._drag_group = []
        self._bond_actor = None
        self._hbond_actor = None
        self._cell_actor = None
        self._polyhedra_actor = None
        self._polyhedra_edge_actor = None
        self._polyhedra_mesh_obj = None
        self._polyhedra_edge_mesh = None
        self._polyhedra_follow = None
        self._polyhedra_edge_follow = None
        self._label_actor = None
        self._arrow_actor = None
        self._adp_actor = None
        self._adp_mesh_obj = None
        self._adp_follow = None
        self._ellipsoid_mask = None  # settings/geometry may have changed under it
        self._orbital_actors = []     # likewise: _draw_orbital re-adds them below
        self._annotation_actors = []  # plotter.clear() dropped them; _draw_annotations re-adds
        self._symmetry_actors = []    # likewise: _draw_symmetry_elements re-adds them
        self._density_actors = []     # and _draw_density, last of all
        self._density_bar = None      # plotter.clear() took the colour bar too
        self._highlight_actors = {}
        if self._structure is None or len(self._structure) == 0:
            self._restore_camera(saved_camera)
            self.plotter.render()
            return

        struct = self._structure
        settings = self._settings
        self._positions = np.asarray(struct.positions, dtype=float)
        self._numbers = np.asarray(struct.numbers, dtype=int)
        self._radii = _sphere_radius(self._numbers, settings.atom_scale)
        self._draw_adp_ellipsoids()  # decides which atoms _draw_atoms skips
        self._draw_atoms()

        if settings.show_bonds:
            self._draw_bonds(self._positions, self._numbers)
        if settings.show_hydrogen_bonds:
            self._draw_hydrogen_bonds()
        # A coordination polyhedron is a large translucent solid centred on an
        # atom, and an ADP ellipsoid is a couple of tenths of an Angstrom sitting
        # inside it — the polyhedron swallows it whole. When the ellipsoids are
        # on screen they are the thing being looked at, so the polyhedra stand
        # down; the setting is left alone, so they return when the ellipsoids go.
        if settings.show_polyhedra and self._adp_actor is None:
            self._draw_polyhedra()
        if settings.show_cell:
            self._draw_cell()
        if settings.show_atom_labels:
            self._draw_atom_labels()
        self._draw_mode_arrows()  # the selected mode's eigenvector, if one is set
        self._draw_orbital()      # and so does a shown orbital
        self._draw_annotations()  # measurements survive a rebuild (plotter.clear())
        self._draw_symmetry_elements()  # and so do the shown symmetry elements
        # A drawn field used to vanish on any rebuild — a display setting, an
        # edit — while its actors were still counted as shown. It goes last so
        # a slice's cutaway reaches every actor the rebuild has just made.
        self._draw_density()
        self._restore_camera(saved_camera)
        self.plotter.render()

    def _current_camera(self):
        """The current view — placement *and* zoom — or ``None`` if unreadable.

        ``camera_position`` alone omits the zoom under parallel (orthographic)
        projection: there the zoom is the camera's *parallel scale*, not its
        distance. Capturing only the placement would let pyvista's post-``clear()``
        auto-fit reset the zoom on every edit (visible as a zoom-out when deleting
        an atom). We snapshot the parallel scale (and the perspective view angle)
        too, so a redraw leaves the view exactly where the user left it.
        """
        try:
            camera = self.plotter.camera
            return (self.plotter.camera_position, camera.GetParallelScale(), camera.GetViewAngle())
        except Exception:  # noqa: BLE001 - bare/uninitialised plotter
            return None

    def _restore_camera(self, camera) -> None:
        """Put a previously-captured view back (a redraw must not move or rezoom it)."""
        if camera is None:
            return
        position, parallel_scale, view_angle = camera
        try:
            self.plotter.camera_position = position
            self.plotter.camera.SetParallelScale(parallel_scale)  # zoom under parallel projection
            self.plotter.camera.SetViewAngle(view_angle)
        except Exception:  # noqa: BLE001
            pass

    # ── atoms (single glyphed mesh) ─────────────────────────────────────
    def _draw_atoms(self) -> None:
        """Draw every atom as one glyphed mesh (a single VTK actor)."""
        self._atom_actor = self._build_atom_glyph_actor()

    def _build_atom_glyph_actor(self):
        """Build the atom glyph: a unit sphere replicated at every atom, scaled
        by covalent radius and coloured per element, as one actor.

        Atoms drawn as thermal ellipsoids are left out — the ellipsoid *is* the
        atom there, and a covalent-radius sphere would swallow it whole (an ADP
        ellipsoid is a couple of tenths of an Angstrom across, a drawn atom
        several times that).

        The replication is done here rather than by ``PolyData.glyph`` so that the
        point order is ours by construction: that is what lets
        :meth:`_reglyph_atoms` move the spheres by writing the point array, with
        no filter to re-run and no actor to replace. Same mesh either way — one
        sphere triangulation per atom, scaled by its radius, coloured by element.
        """
        self._atom_mesh_obj = None
        self._atom_follow = None
        shown = self._sphere_atoms()
        if not shown.any():
            return None
        resolution = _sphere_resolution(len(self._numbers))
        template = pv.Sphere(radius=1.0, theta_resolution=resolution, phi_resolution=resolution)
        unit = np.asarray(template.points, dtype=float)
        template_faces = np.asarray(template.faces, dtype=np.int64).reshape(-1, 4)

        index = np.nonzero(shown)[0]
        # (n_shown, n_unit, 3): each atom's sphere, scaled by its own radius. This
        # is also the per-vertex offset from its atom, which is what stays
        # constant while the atoms move.
        offsets = self._radii[index][:, None, None] * unit[None, :, :]
        points = (self._positions[index][:, None, :] + offsets).reshape(-1, 3)

        faces = np.tile(template_faces, (len(index), 1))
        faces[:, 1:] += np.repeat(
            np.arange(len(index)) * len(unit), len(template_faces)
        )[:, None]

        mesh = pv.PolyData(points, faces.ravel())
        mesh["rgb"] = np.repeat(self._rgb_for(self._numbers[index]), len(unit), axis=0)
        self._atom_mesh_obj = mesh
        self._atom_follow = (np.repeat(index, len(unit)), offsets.reshape(-1, 3))
        return self.plotter.add_mesh(
            mesh,
            scalars="rgb",
            rgb=True,
            opacity=self._settings.atom_opacity,
            render=False,
            **_ATOM_MATERIAL,
        )

    def _reglyph_atoms(self) -> None:
        """Move the drawn atoms onto the current positions (no render).

        Writes the mesh's point array — every sphere keeps its size, colour and
        triangulation, only its centre moves — instead of dropping the actor and
        building a fresh glyph. This runs on every animation frame and every drag
        mouse-move, where the teardown-and-rebuild was the single largest cost:
        ~10 ms even for 64 atoms, nearly all of it pyvista/VTK pipeline overhead
        rather than geometry.

        Falls back to a rebuild if the follow map no longer fits the positions,
        so a caller that changed the atom count still gets a correct picture.
        """
        follow = self._atom_follow
        if self._atom_mesh_obj is not None and follow is not None:
            index, offset = follow
            if len(index) == 0 or int(index.max()) < len(self._positions):
                self._atom_mesh_obj.points = self._positions[index] + offset
                return
        if self._atom_actor is not None:
            self.plotter.remove_actor(self._atom_actor, render=False)
        self._atom_actor = None
        if len(self._positions):
            self._atom_actor = self._build_atom_glyph_actor()

    def _rgb_for(self, numbers: np.ndarray) -> np.ndarray:
        """Per-atom uint8 RGB rows (Jmol colours with the settings' overrides applied)."""
        numbers = np.asarray(numbers, dtype=int)
        rgb = (elements.colours(numbers) * 255).astype(np.uint8)
        for z, color in self._settings.atom_colors:
            mask = numbers == int(z)
            if mask.any():
                rgb[mask] = _hex_to_rgb(color)
        return rgb

    def _draw_bonds(self, positions: np.ndarray, numbers: np.ndarray) -> None:
        radius, tol = self._settings.bond_radius, self._settings.bond_tolerance
        mesh = _bond_mesh(positions, numbers, radius, tol, pairs=self._bond_pairs(numbers, tol))
        if mesh is None:
            return
        self._bond_actor = self.plotter.add_mesh(
            mesh, color=self._settings.bond_color, smooth_shading=True, render=False
        )
        self._bond_actor.SetPickable(False)  # only atoms are pick targets

    def _bond_pairs(self, numbers: np.ndarray, tolerance: float) -> Optional[tuple]:
        """Bonded pairs to draw, or ``None`` to derive them from the live frame.

        While a bond reference is set the pairs are computed from it once and
        cached, so an animation pays the neighbour search a single time.
        """
        if self._bond_reference is None or len(self._bond_reference) != len(self._positions):
            return None
        if self._frozen_bond_pairs is None:
            self._frozen_bond_pairs = _bonded_pairs(self._bond_reference, numbers, tolerance)
        return self._frozen_bond_pairs

    def _draw_hydrogen_bonds(self) -> None:
        """Draw D–H···A hydrogen bonds as thin dashed light-blue lines.

        Computed on the atoms as drawn (``self._positions``, no periodicity), like
        the covalent bonds: the displayed cell is already boundary-completed, so
        searching its periodic images too would draw each contact twice (once to a
        visible atom, once to an image out in space). Using the *drawn* positions
        also means the bonds follow a phonon animation, which moves those.
        """
        from crystalline.core.bonds import (
            hydrogen_bond_pairs,
            hydrogen_bond_segments,
            hydrogen_bonds_from_positions,
        )

        if self._bond_reference is not None and len(self._bond_reference) == len(self._positions):
            if self._frozen_hbond_pairs is None:
                self._frozen_hbond_pairs = hydrogen_bond_pairs(self._bond_reference, self._numbers)
            segments = hydrogen_bond_segments(self._positions, self._frozen_hbond_pairs)
        else:
            segments = hydrogen_bonds_from_positions(self._positions, self._numbers)
        if len(segments) == 0:
            return
        mesh = _dashed_lines(segments, _HBOND_DASH, _HBOND_GAP)
        if mesh is None:
            return
        self._hbond_actor = self.plotter.add_mesh(
            mesh, color=self._settings.hydrogen_bond_color,
            line_width=self._settings.hydrogen_bond_width, render=False
        )
        self._hbond_actor.SetPickable(False)  # not a pick target

    def _update_live_bonds(self) -> None:
        """Refresh the bond meshes for a live position change (no render here).

        Shared by both live paths — the animation's ``update_positions`` and the
        drag's ``preview_atom_position`` — so their size limits cannot drift
        apart. They had: the drag ran the hydrogen-bond scan on anything up to
        ``_LIVE_BOND_MAX_ATOMS`` (5000), skipping the far tighter limit that scan
        needs, which cost ~140 ms *per mouse-move event* on a 2400-atom hydrated
        cell and made dragging an atom there unusable.

        Above the covalent limit the bonds are left in place and refresh on
        commit, as documented on ``_LIVE_BOND_MAX_ATOMS``.
        """
        if len(self._positions) > _LIVE_BOND_MAX_ATOMS:
            return
        self._update_bonds(self._positions)
        # The hydrogen-bond scan is the expensive one; a frozen topology makes it
        # a gather, so only the unfrozen path needs the tighter limit.
        frozen = self._frozen_hbond_pairs is not None or self._bond_reference is not None
        if frozen or len(self._positions) <= _LIVE_HBOND_MAX_ATOMS:
            self._update_hydrogen_bonds()

    def _update_hydrogen_bonds(self) -> None:
        """Redraw the hydrogen bonds for the current positions (no render here)."""
        if self._hbond_actor is not None:
            self.plotter.remove_actor(self._hbond_actor, render=False)
            self._hbond_actor = None
        if self._settings.show_hydrogen_bonds:
            self._draw_hydrogen_bonds()

    def _draw_atom_labels(self) -> None:
        """Label each atom with its element symbol (suppressed for large cells)."""
        if len(self._numbers) == 0 or len(self._numbers) > _ATOM_LABEL_MAX_ATOMS:
            return
        symbols = [chemical_symbols[int(z)] for z in self._numbers]
        self._label_actor = self.plotter.add_point_labels(
            np.asarray(self._positions, dtype=float),
            symbols,
            font_size=int(self._settings.atom_label_size),
            text_color=self._label_colour(),
            show_points=False,
            shape=None,
            always_visible=True,
            pickable=False,
            render=False,
        )
        self._label_actor.SetPickable(False)  # text must never intercept picks

    def _label_colour(self) -> str:
        """The labels' colour: the one chosen, or whichever reads on the background."""
        return self._settings.atom_label_color or _readable_on(self._settings.background_color)

    def _draw_polyhedra(self) -> None:
        """Draw coordination polyhedra (convex hull of each cation's ligands).

        VESTA-style. Coordination comes from pymatgen's ``CrystalNN`` on the
        clean unit cell (``bond_structure``) so ionic crystals get the right
        cation–anion polyhedra (Si→tetrahedra, Ca→6–8) with correct geometry
        across the cell boundary. Falls back to a periodic distance search if
        that's unavailable.
        """
        mesh = self._polyhedra_mesh()
        if mesh is None:
            return
        self._polyhedra_actor = self.plotter.add_mesh(
            mesh,
            scalars="colors",
            rgb=True,
            opacity=self._settings.polyhedra_opacity,
            smooth_shading=True,
            render=False,
        )
        self._polyhedra_actor.SetPickable(False)
        # Remember which atom each hull vertex rides on, so the polyhedra can
        # follow a phonon animation without re-running the coordination analysis.
        self._polyhedra_mesh_obj = mesh
        self._polyhedra_follow = _follow_atoms(mesh.points, self._positions, self._cell_or_none())
        self._draw_polyhedra_edges(mesh)

    def _draw_polyhedra_edges(self, mesh: "pv.PolyData") -> None:
        """Outline each polyhedron's edges so its shape reads at low opacity.

        The hull arrives triangulated, so drawing every triangle edge would
        criss-cross each flat face with diagonals; ``extract_feature_edges``
        keeps only the edges where the surface actually bends — the polyhedron's
        real edges. They are drawn opaque, in a darkened shade of the face
        colour, and unlit so the outline stays an even weight from every angle.
        """
        if self._settings.polyhedra_edge_width <= 0:
            return  # no outline asked for
        edges = mesh.extract_feature_edges(
            feature_angle=_POLYHEDRA_EDGE_ANGLE,
            boundary_edges=False,  # closed hulls have none
            non_manifold_edges=False,
            manifold_edges=False,
        )
        if edges is None or edges.n_points == 0:
            return
        if "colors" in edges.point_data:
            shaded = edges["colors"].astype(float) * _POLYHEDRA_EDGE_SHADE
            edges["colors"] = shaded.astype(np.uint8)
        self._polyhedra_edge_actor = self.plotter.add_mesh(
            edges,
            scalars="colors" if "colors" in edges.point_data else None,
            rgb="colors" in edges.point_data,
            line_width=self._settings.polyhedra_edge_width,
            lighting=False,
            render=False,
        )
        self._polyhedra_edge_actor.SetPickable(False)
        self._polyhedra_edge_mesh = edges
        self._polyhedra_edge_follow = _follow_atoms(
            edges.points, self._positions, self._cell_or_none()
        )

    # ── measurement annotations (points / lines / planes) ───────────────
    def set_annotations(self, annotations) -> None:
        """Draw geometry measurements over the structure, replacing any shown.

        ``annotations`` are :class:`~crystalline.core.measure.Measurement`
        objects: a point becomes a marker, a distance/angle/dihedral becomes the
        polyline through its atoms, and a plane becomes a translucent patch.
        They are kept and redrawn on every rebuild, so they survive
        an edit or a settings change rather than blinking out.
        """
        self._annotations = list(annotations)
        self._clear_annotations()
        self._draw_annotations()
        self.plotter.render()

    def _clear_annotations(self) -> None:
        for actor in self._annotation_actors:
            self.plotter.remove_actor(actor, render=False)
        self._annotation_actors = []

    def _draw_annotations(self) -> None:
        """(Re)draw the stored measurements. Never raises into a redraw."""
        from crystalline.core.measure import DIHEDRAL, PLANE, POINT

        self._annotation_actors = []
        if not self._annotations:
            return
        labels: list = []
        label_points: list = []
        for item in self._annotations:
            points = np.asarray(item.points, dtype=float)
            if len(points) == 0:
                continue
            try:
                # A per-item colour (set in the Geometry panel) overrides the
                # type's default from settings.
                if item.kind == POINT:
                    self._add_annotation_actor(
                        pv.Sphere(radius=_ANNOTATION_POINT_RADIUS, center=points[0]),
                        color=item.color or self._settings.measure_point_color,
                    )
                elif item.kind == PLANE:
                    self._draw_annotation_plane(item, points)
                else:  # distance / angle / dihedral: the path through the atoms
                    self._add_annotation_actor(
                        _polyline_tube(points),
                        color=item.color or self._settings.measure_line_color,
                    )
            except Exception:  # noqa: BLE001 - a bad measurement must not kill the redraw
                continue
            if item.kind in (POINT, PLANE):
                labels.append("")  # points and planes carry no floating label
            else:
                labels.append(f"{item.value:.3f} {item.unit}".strip())
            label_points.append(_annotation_anchor(item.kind, points, DIHEDRAL))

        texts = [t for t in labels if t]
        if texts:
            anchors = np.asarray(
                [p for t, p in zip(labels, label_points) if t], dtype=float
            )
            actor = self.plotter.add_point_labels(
                anchors, texts, font_size=_ANNOTATION_FONT_SIZE, show_points=False,
                shape_opacity=0.6, always_visible=True, pickable=False, render=False,
            )
            actor.SetPickable(False)
            self._annotation_actors.append(actor)

    def _draw_annotation_plane(self, item, points: np.ndarray) -> None:
        """A translucent patch spanning the fitted atoms."""
        origin = np.asarray(item.origin, dtype=float)
        normal = np.asarray(item.normal, dtype=float)
        # Size the patch to the atoms it was fitted through, with a little margin.
        spread = float(np.linalg.norm(points - origin, axis=1).max())
        size = max(spread * 2.0 * _ANNOTATION_PLANE_MARGIN, _ANNOTATION_PLANE_MIN_SIZE)
        patch = pv.Plane(center=origin, direction=normal, i_size=size, j_size=size)
        self._add_annotation_actor(
            patch, color=item.color or self._settings.measure_plane_color,
            opacity=_ANNOTATION_PLANE_OPACITY,
            on_top=False,  # a plane reads as a slice *through* the structure
        )

    def _add_annotation_actor(
        self, mesh, color: str = None, opacity: float = 1.0, on_top: bool = True
    ) -> None:
        actor = self.plotter.add_mesh(
            mesh,
            color=color or _ANNOTATION_COLOR,
            opacity=opacity,
            smooth_shading=True,
            render=False,
        )
        actor.SetPickable(False)  # annotations are never a pick target
        if on_top:
            _draw_over_scene(actor)
        self._annotation_actors.append(actor)

    # ── symmetry elements (axes / planes / centres) ─────────────────────
    def set_symmetry_elements(self, elements, labels: bool = False) -> None:
        """Draw symmetry elements over the structure, replacing any shown.

        ``elements`` are :class:`~crystalline.core.symmetry.SymmetryElement`
        objects: a rotation axis becomes a tube across the drawn scene, a mirror
        plane a translucent patch through it, and the inversion centre a dot. All
        of them pass through the one centre the point group acts about.
        ``labels`` writes each element's Hermann–Mauguin symbol beside it.

        They are kept and redrawn on every rebuild, so an edit or a settings
        change doesn't blink them out.
        """
        self._symmetry_elements = list(elements)
        self._symmetry_labels = bool(labels)
        self._clear_symmetry_elements()
        self._draw_symmetry_elements()
        self.plotter.render()

    def _clear_symmetry_elements(self) -> None:
        for actor in self._symmetry_actors:
            self.plotter.remove_actor(actor, render=False)
        self._symmetry_actors = []

    def _draw_symmetry_elements(self) -> None:
        """(Re)draw the stored symmetry elements. Never raises into a redraw."""
        from crystalline.core.symmetry import PLANE, POINT

        self._symmetry_actors = []
        if not self._symmetry_elements:
            return
        bounds = self._scene_bounds(_SYMMETRY_BOUNDS_MARGIN)
        if bounds is None:
            return

        colors = {
            POINT: self._settings.symmetry_point_color,
            PLANE: self._settings.symmetry_plane_color,
        }
        labels: list = []
        anchors: list = []
        for element in self._symmetry_elements:
            try:
                drawn = self._symmetry_mesh(element, bounds)
            except Exception:  # noqa: BLE001 - one bad element must not kill the redraw
                continue
            if drawn is None:
                continue  # this element doesn't reach the drawn box
            mesh, spots = drawn
            self._add_symmetry_actor(
                mesh,
                color=colors.get(element.kind, self._settings.symmetry_axis_color),
                opacity=_SYMMETRY_PLANE_OPACITY if element.kind == PLANE else 1.0,
                # A plane reads as a slice *through* the structure, like a
                # measured plane does; an axis has to stay visible in front of it.
                on_top=element.kind != PLANE,
            )
            if self._symmetry_labels:
                labels.append(element.label)
                anchors.append(_spread_anchor(spots, anchors))

        if labels:
            actor = self.plotter.add_point_labels(
                np.asarray(anchors, dtype=float), labels,
                font_size=_SYMMETRY_FONT_SIZE, show_points=False, shape_opacity=0.6,
                always_visible=True, pickable=False, render=False,
            )
            actor.SetPickable(False)
            use_unicode_font(actor)
            self._symmetry_actors.append(actor)

    def _symmetry_mesh(self, element, bounds):
        """``(mesh, label anchor)`` for ``element`` clipped to the box, or ``None``.

        An axis and a plane are both infinite, so each is cut down to the part
        that crosses the drawn scene: the axis becomes a tube between the two
        faces it leaves through, and the plane the polygon it slices out of the
        box — a hexagon across a cube's diagonal as readily as a square.
        """
        from crystalline.core.symmetry import AXIS, PLANE, segment_in_box

        if element.kind == AXIS:
            segment = segment_in_box(element.origin, element.direction, bounds)
            if segment is None:
                return None
            points = np.asarray(segment, dtype=float)
            return (
                _polyline_tube(points, _SYMMETRY_LINE_RADIUS),
                _label_spots(points, element.origin),
            )
        if element.kind == PLANE:
            patch = _plane_in_box(element.origin, element.direction, bounds)
            if patch is None:
                return None
            return patch, _label_spots(np.asarray(patch.points, dtype=float), element.origin)
        centre = np.asarray(element.origin, dtype=float)
        return pv.Sphere(radius=_SYMMETRY_POINT_RADIUS, center=centre), centre.reshape(1, 3)

    def _add_symmetry_actor(self, mesh, color: str, opacity: float, on_top: bool) -> None:
        actor = self.plotter.add_mesh(
            mesh, color=color, opacity=opacity, smooth_shading=True, render=False
        )
        actor.SetPickable(False)  # symmetry elements are never a pick target
        if on_top:
            _draw_over_scene(actor)
        self._symmetry_actors.append(actor)

    # ── crystalline orbital (isosurface) ────────────────────────────────
    def set_orbital(self, field, isovalue: float = DEFAULT_ORBITAL_ISOVALUE) -> None:
        """Draw a crystalline orbital as a two-lobed isosurface (``None`` clears).

        ``field`` is a :class:`~crystalline.core.orbitals.OrbitalField`.
        ``isovalue`` is a *fraction of the field's peak*, not an absolute
        amplitude: orbitals differ in how peaked they are by orders of magnitude,
        so a fixed absolute value would show a blob for one and nothing at all
        for the next.

        Both signs are drawn, in different colours — the sign structure is what
        distinguishes an orbital from a density, and dropping it would throw away
        the reason for plotting the orbital rather than the charge.

        A field carrying ``phases`` is drawn instead as a single ``|psi|``
        surface coloured by ``arg(psi)`` — see :meth:`_draw_orbital`.
        """
        self._orbital_field = None if field is None else (field, float(isovalue))
        self._clear_orbital()
        self._draw_orbital()
        self.plotter.render()

    def _clear_orbital(self) -> None:
        for actor in self._orbital_actors:
            self.plotter.remove_actor(actor, render=False)
        self._orbital_actors = []

    def _draw_orbital(self) -> None:
        """(Re)draw the stored orbital. Never raises into a redraw."""
        if self._orbital_field is None:
            return
        field, fraction = self._orbital_field
        peak = field.peak
        if peak <= 0.0:
            return
        level = abs(fraction) * peak

        grid = pv.ImageData()
        grid.dimensions = field.shape
        grid.origin = tuple(float(v) for v in field.origin)
        grid.spacing = tuple(float(v) for v in field.spacing)
        # Fortran order: VTK varies x fastest, numpy's reshape varies it slowest.
        grid.point_data["psi"] = np.asarray(field.values, dtype=float).ravel(order="F")

        if field.phases is not None:
            self._draw_phase_coloured_orbital(grid, field, level)
            return

        # A modulus is non-negative, so only the positive surface exists.
        levels = [level] if field.modulus else [-level, level]
        colors = (
            [_ORBITAL_POSITIVE_COLOR] if field.modulus
            else [_ORBITAL_NEGATIVE_COLOR, _ORBITAL_POSITIVE_COLOR]
        )
        for value, color in zip(levels, colors):
            try:
                surface = grid.contour([value], scalars="psi")
            except Exception:  # noqa: BLE001 - a level with no surface is not an error
                continue
            if surface is None or surface.n_points == 0:
                continue  # this level is not crossed anywhere in the box
            surface = _clip_to_cell(surface, field.cell)
            if surface is None or surface.n_points == 0:
                continue
            actor = self.plotter.add_mesh(
                surface, color=color, opacity=_ORBITAL_OPACITY,
                smooth_shading=True, show_scalar_bar=False, render=False,
            )
            actor.SetPickable(False)  # only atoms are pick targets
            self._orbital_actors.append(actor)

    def _draw_phase_coloured_orbital(self, grid, field, level: float) -> None:
        """One ``|psi|`` surface, coloured by the orbital's phase.

        This is what makes the relation between cells legible. ``|psi|`` is
        lattice-periodic, so the *surface* is identical in every cell — and the
        colour, that cell's Bloch phase, is then the only thing that differs,
        changing from cell to cell by exactly ``k.T``. The same reasoning as the
        phonon panel's per-atom phase colours: a relation reads only when the
        thing being related looks the same in each cell.

        The phase is looked up at the nearest grid point of each surface vertex
        rather than being carried through the contour filter as point data. It
        is a step function — one value per cell — and VTK would interpolate it,
        turning every cell boundary that a lobe straddles into a band sweeping
        through the whole colour wheel. Nearest-point lookup keeps the step a
        step, and sidesteps the ±pi seam at the same time.
        """
        try:
            surface = grid.contour([level], scalars="psi")
        except Exception:  # noqa: BLE001 - a level with no surface is not an error
            return
        if surface is None or surface.n_points == 0:
            return
        surface = _clip_to_cell(surface, field.cell)
        if surface is None or surface.n_points == 0:
            return
        surface.point_data["phase"] = _sample_nearest(
            np.asarray(field.phases, dtype=float), field, surface.points
        )
        actor = self.plotter.add_mesh(
            surface, scalars="phase", cmap=_PHASE_COLORMAP,
            clim=(-np.pi, np.pi), opacity=_ORBITAL_OPACITY,
            smooth_shading=True, show_scalar_bar=False, render=False,
        )
        actor.SetPickable(False)
        self._orbital_actors.append(actor)

    # ── a scalar field from a run (charge density, spin density, potential) ──
    def set_density(self, field, options=None, miller_cell=None) -> None:
        """Draw a :class:`~crystalline.crystalio.density.ScalarField` (``None`` clears).

        Unlike an orbital, which this app evaluates on a grid of its own
        choosing, a field comes off a CRYSTAL run on *CRYSTAL's* grid: the steps
        run along the primitive lattice vectors, so for anything but an
        orthogonal cell the sampled box is sheared. That rules out the uniform
        ``ImageData`` the orbital uses — its cells are axis-aligned boxes — and
        calls for a structured grid, whose points are given one by one.

        A density is lattice-periodic, so the drawn surface is the same in every
        cell: it is built once and copied, rather than contoured again.

        ``miller_cell`` is the cell a slice's Miller indices are quoted in —
        the conventional one. Without it the displayed cell is used.
        """
        from crystalline.crystalio import density as density_module

        self._density_view = None if field is None else (
            field, options or density_module.DensityOptions()
        )
        self._density_miller_cell = None if miller_cell is None else np.asarray(
            miller_cell, dtype=float)
        self._surface_cache = None     # a new field, or new options: contour afresh
        self._clear_density()
        self._draw_density()
        self.plotter.render()

    def _clear_density(self) -> None:
        for actor in self._density_actors:
            self.plotter.remove_actor(actor, render=False)
        self._density_actors = []
        if self._density_bar is not None:
            try:
                self.plotter.remove_scalar_bar(self._density_bar, render=False)
            except Exception:  # noqa: BLE001 - already gone with a cleared plotter
                pass
            self._density_bar = None

    def _draw_density(self) -> None:
        """(Re)draw the stored field, and the cutaway that goes with a slice.

        Never raises into a redraw.
        """
        from crystalline.crystalio import density as density_module

        if self._density_view is None:
            self._apply_cutaway(None)
            return
        field, options = self._density_view
        cell = density_module.lattice_of(field, self._cell_or_none())
        if options.view == density_module.SLICE:
            try:
                mesh = self._miller_slice(field, options, cell)
            except Exception:  # noqa: BLE001 - e.g. (000); nothing to draw
                self._apply_cutaway(None)
                return
            self._add_slice_mesh(mesh, field, options)
            if options.cutaway:
                self._mark_atoms_in_plane()
            self._apply_cutaway(self._slice_frame if options.cutaway else None)
            return
        self._apply_cutaway(None)
        try:
            surfaces = self._surface_pieces(field, options, cell)
        except Exception:  # noqa: BLE001 - a level with no surface is not an error
            return
        for colour, pieces, limits in surfaces:
            self._add_density_mesh(pieces, colour, limits, field, options, cell)

    def _surface_pieces(self, field, options, cell):
        """The field's surfaces in the home cell, split into pieces — cached.

        This is all the expensive part — contouring, splitting, deciding which
        cell each piece belongs to — and none of it depends on which atoms are
        on screen. Switching between the conventional and the primitive cell,
        or anything else that rebuilds the scene, used to redo it every time
        and took seconds for MgO; now it is done once per field and isovalue,
        and a rebuild only places the pieces.
        """
        key = (id(field), float(options.isovalue), id(options.colour_by),
               bool(options.clip_to_cell), np.round(np.asarray(cell, dtype=float), 6).tobytes())
        cached = self._surface_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        # Contoured on a grid continued periodically past the cell's faces, so
        # that a surface around an atom on a face or a corner comes out whole.
        padded = _padded_field(field, cell, _SURFACE_PAD)
        grid = _structured_grid(padded)
        surfaces = []
        if grid is not None:
            for mesh, colour in _density_surfaces(grid, padded, options):
                pieces = _home_pieces(mesh, padded, cell, field.origin, options.clip_to_cell)
                if not pieces:
                    continue
                limits = None
                if pieces[0].values is not None:
                    values = np.concatenate([piece.values for piece in pieces])
                    signed = field.signed or (options.colour_by is not None
                                              and options.colour_by.signed)
                    limits = _colour_limits(values, symmetric=signed)
                surfaces.append((colour, pieces, limits))
        self._surface_cache = (key, surfaces)
        return surfaces

    # ── lattice planes ──────────────────────────────────────────────────
    def _miller_slice(self, field, options, lattice):
        """The plane ``(hkl)`` as a flat, rectangular mesh coloured by the field.

        VTK's own cut of the grid can only ever be a piece of the one sampled
        cell — a ragged polygon for any plane not parallel to a cell face. The
        field is periodic, so the plane is laid out as a rectangle spanning
        everything on screen and the field is read at each of its points by
        periodic interpolation instead.
        """
        from crystalline.crystalio import density as density_module

        reference = self._density_miller_cell
        if reference is None:
            reference = self._cell_or_none()
        if reference is None:
            reference = lattice
        point, normal, _spacing = density_module.miller_plane(
            reference, options.miller, options.offset)

        # Two directions in the plane, the first along whichever cell edge lies
        # closest to it, so that a (001) map is drawn square to its a axis.
        edges = np.asarray(reference, dtype=float)
        along = edges[int(np.argmin(np.abs(edges @ normal) / np.linalg.norm(edges, axis=1)))]
        u = along - (along @ normal) * normal
        u = u / np.linalg.norm(u)
        w = np.cross(normal, u)

        corners = self._scene_corners(lattice, field)
        su = (corners - point) @ u
        sw = (corners - point) @ w
        # Wide enough to hide the atoms behind it: a corner atom's lower half
        # otherwise shows past the edge of the map as a crescent.
        pad = 0.2 + (float(self._radii.max()) if len(self._radii) else 0.3)
        u_low, u_high = float(su.min()) - pad, float(su.max()) + pad
        w_low, w_high = float(sw.min()) - pad, float(sw.max()) + pad

        step = max(float(np.linalg.norm(field.steps, axis=1).min()), 0.02)
        nu = int(min(max(np.ceil((u_high - u_low) / step) + 1, 2), _MAX_SLICE_POINTS))
        nw = int(min(max(np.ceil((w_high - w_low) / step) + 1, 2), _MAX_SLICE_POINTS))
        a, b = np.meshgrid(np.linspace(u_low, u_high, nu), np.linspace(w_low, w_high, nw),
                           indexing="ij")
        points = point + a[..., None] * u + b[..., None] * w      # (nu, nw, 3)

        flat = points.reshape(-1, 3)
        if options.colour_by is not None:
            values = density_module.sample_periodic(options.colour_by, flat)
        else:
            values = density_module.sample_periodic(field, flat, lattice)
            if options.logarithmic and not field.signed:
                positive = values[values > 0]
                floor = float(positive.min()) if positive.size else 1e-8
                values = np.log10(np.clip(values, floor, None))

        mesh = pv.StructuredGrid(points[..., 0][..., None], points[..., 1][..., None],
                                 points[..., 2][..., None])
        mesh.point_data["value"] = values.reshape(nu, nw).ravel(order="F")
        centre = point + 0.5 * (u_low + u_high) * u + 0.5 * (w_low + w_high) * w
        self._slice_frame = (centre, normal, u)
        self._slice_extent = (point, u, w, (u_low, u_high), (w_low, w_high))
        return mesh

    def _scene_corners(self, lattice, field) -> np.ndarray:
        """Points bounding what is on screen: the atoms and the cell's corners."""
        points = []
        if len(self._positions):
            points.append(np.asarray(self._positions, dtype=float))
        cell = self._cell_or_none()
        box = cell if cell is not None else lattice
        origin = np.zeros(3) if cell is not None else field.origin
        corners = np.array([origin + i * box[0] + j * box[1] + k * box[2]
                            for i in (0, 1) for j in (0, 1) for k in (0, 1)])
        points.append(corners)
        return np.vstack(points)

    def _add_slice_mesh(self, mesh, field, options) -> None:
        values = np.asarray(mesh.point_data["value"], dtype=float)
        signed = field.signed if options.colour_by is None else options.colour_by.signed
        actor = self.plotter.add_mesh(
            mesh, scalars="value", cmap=options.cmap,
            clim=_colour_limits(values, symmetric=signed),
            opacity=options.opacity, show_scalar_bar=False, render=False,
            # A map is read for its colours: lighting would shade one side of
            # the plane darker than the other and change what it says.
            lighting=False,
        )
        actor.SetPickable(False)
        self._density_actors.append(actor)
        self._add_colour_bar(actor, field, options)

    def _apply_cutaway(self, frame) -> None:
        """Cut away whatever lies in front of a slice, or restore it (``None``).

        Done with clipping planes on the mappers rather than by removing atoms:
        nothing about the structure changes — picking, labels, animation and
        the undo history all still see every atom — only what is drawn does.

        The cut sits a hair *behind* the plane, so that the atoms lying in it go
        too: kept whole, as domes, they covered the map exactly where it is
        densest — an Mg²⁺ drawn at its usual size hides most of rocksalt's
        (001) layer. They are marked by small dots instead; see
        :meth:`_mark_atoms_in_plane`.
        """
        from vtkmodules.vtkCommonDataModel import vtkPlane

        if frame is None and self._cutaway is None:
            return      # nothing was cut, so there is nothing to put back
        self._cutaway = frame
        density = {id(actor) for actor in self._density_actors}
        margin = -_CUTAWAY_BEHIND
        for actor in list(self.plotter.renderer.actors.values()):
            if id(actor) in density:
                continue
            mapper = actor.GetMapper() if hasattr(actor, "GetMapper") else None
            if mapper is None or not hasattr(mapper, "AddClippingPlane"):
                continue
            mapper.RemoveAllClippingPlanes()
            if frame is None:
                continue
            centre, normal, _up = frame
            plane = vtkPlane()
            # VTK keeps what lies on the side the normal points to, so the plane
            # faces back into the crystal: everything beyond it goes.
            plane.SetOrigin(*(centre + margin * normal))
            plane.SetNormal(*(-normal))
            mapper.AddClippingPlane(plane)

    def _mark_atoms_in_plane(self) -> None:
        """Small dots, in each element's colour, wherever an atom lies on the map.

        The map says what the density does; the dots say which nuclei it is
        doing it around. A dot for every atom of the crystal on the map, not
        only for the atoms drawn: a map spans more than the cell box, and a peak
        with a dot next to a peak without one reads as two different things.
        A third of the atom's drawn size, so relative sizes still read and the
        map around each dot stays visible.
        """
        extent = getattr(self, "_slice_extent", None)
        structure = self._structure
        if extent is None or structure is None or len(structure) == 0:
            return
        point, u, w, (u_low, u_high), (w_low, w_high) = extent
        normal = np.cross(u, w)
        positions = np.asarray(structure.positions, dtype=float)
        numbers = np.asarray(structure.numbers, dtype=int)
        cell = self._cell_or_none()
        offsets = [np.zeros(3)]
        if cell is not None:
            corners = np.array([point + a * u + b * w
                                for a in (u_low, u_high) for b in (w_low, w_high)])
            try:
                fractional = corners @ np.linalg.inv(cell)
            except np.linalg.LinAlgError:
                fractional = None
            if fractional is not None:
                low = np.floor(fractional.min(axis=0)).astype(int) - 1
                high = np.ceil(fractional.max(axis=0)).astype(int) + 1
                periodic = np.asarray(structure.pbc, dtype=bool)
                low[~periodic], high[~periodic] = 0, 0
                if int(np.prod(high - low + 1)) <= 4096:
                    offsets = [i * cell[0] + j * cell[1] + k * cell[2]
                               for i in range(low[0], high[0] + 1)
                               for j in range(low[1], high[1] + 1)
                               for k in range(low[2], high[2] + 1)]
        images = (positions[None, :, :] + np.asarray(offsets)[:, None, :]).reshape(-1, 3)
        kinds = np.tile(numbers, len(offsets))
        distance = (images - point) @ normal
        along, across = (images - point) @ u, (images - point) @ w
        keep = ((np.abs(distance) < _IN_PLANE)
                & (along >= u_low) & (along <= u_high)
                & (across >= w_low) & (across <= w_high))
        if not keep.any():
            return
        # A boundary-completed structure already holds some images twice.
        spots, index = np.unique(np.round(images[keep] - np.outer(distance[keep], normal), 3),
                                 axis=0, return_index=True)
        kinds = kinds[keep][index]
        radii = _sphere_radius(kinds, self._settings.atom_scale)
        cloud = pv.PolyData(spots + _CUTAWAY_BEHIND * normal)
        cloud.point_data["radius"] = _IN_PLANE_MARKER * radii
        cloud.point_data["rgb"] = self._rgb_for(kinds)
        dots = cloud.glyph(geom=pv.Sphere(radius=1.0, theta_resolution=24,
                                          phi_resolution=16),
                           scale="radius", orient=False)
        actor = self.plotter.add_mesh(dots, scalars="rgb", rgb=True, smooth_shading=True,
                                      show_scalar_bar=False, render=False)
        actor.SetPickable(False)
        self._density_actors.append(actor)

    def face_density_plane(self) -> None:
        """Turn the camera to look straight at the drawn slice, from the cut side."""
        frame = getattr(self, "_slice_frame", None)
        if frame is None or self._density_view is None:
            return
        centre, normal, up = frame
        try:
            distance = float(self.plotter.camera.GetDistance()) or 20.0
        except Exception:  # noqa: BLE001
            distance = 20.0
        self.plotter.camera_position = [tuple(centre + distance * normal), tuple(centre),
                                        tuple(up)]
        self.plotter.reset_camera()
        self.plotter.render()

    def _add_density_mesh(self, pieces, colour, limits, field, options, cell) -> None:
        """The home cell's pieces, copied onto the cells the atoms on screen occupy.

        The field is sampled over one cell, but the atoms are not always in that
        cell: an output file writes each atom in whichever periodic image it
        came out in, and a supercell spreads them over many. Beryllium is the
        plain case — its two atoms come out one ``b`` and one ``a + c`` from the
        grid's own copies of them, so a field drawn over the home cell alone
        sits a whole translation away from them.

        The field is lattice-periodic, so those cells hold the same pieces. A
        copy is kept only where an atom on screen lies against it, and the kept
        copies go into one mesh, built once.
        """
        offsets = _density_images(field, cell, self._positions)
        mesh = _assemble_pieces(pieces, offsets, self._positions)
        if mesh is None:
            return
        scalars = "value" if limits is not None else None
        actor = self.plotter.add_mesh(
            mesh,
            color=None if scalars else colour,
            scalars=scalars, cmap=options.cmap if scalars else None, clim=limits,
            opacity=options.opacity, smooth_shading=True,
            show_scalar_bar=False, render=False,
        )
        actor.SetPickable(False)   # only atoms are pick targets
        self._density_actors.append(actor)
        if scalars is not None:
            self._add_colour_bar(actor, field, options)

    def _add_colour_bar(self, actor, field, options) -> None:
        """A vertical colour bar at the right edge, keying ``actor``'s colours.

        Written in the same Unicode font as the symmetry labels: VTK's built-in
        fonts stop at Latin-1, and a bar headed "log  (e/bohr)" with its ρ and
        subscript dropped would say nothing. The text takes whichever of black
        or white reads on the chosen background.
        """
        from crystalline.crystalio import density as density_module

        if not options.colour_bar or self._density_bar is not None:
            return
        title = density_module.bar_title(field, options)
        colour = _readable_on(self._settings.background_color)
        try:
            bar = self.plotter.add_scalar_bar(
                title=title, mapper=actor.mapper, vertical=True, n_labels=5, fmt="%.3g",
                position_x=0.86, position_y=0.18, width=0.07, height=0.64,
                title_font_size=14, label_font_size=12, color=colour, render=False,
            )
        except Exception:  # noqa: BLE001 - a bar is a nicety; never break the drawing
            return
        font = unicode_font()
        if font is not None:
            for text in (bar.GetTitleTextProperty(), bar.GetLabelTextProperty()):
                text.SetFontFamily(vtk.VTK_FONT_FILE)
                text.SetFontFile(font)
        self._density_bar = title

    def _scene_bounds(self, margin: float = 0.0) -> Optional[tuple]:
        """The drawn extent as ``(xmin, xmax, …)``, padded by ``margin`` Angstrom.

        Both the atoms and the cell contribute: a symmetry element has to span
        the whole cell even where no atom sits, and has to reach the atoms a
        boundary-completed view puts outside it. Only *periodic* lattice vectors
        count — a slab's formal 500 Å vacuum axis is not part of the scene.
        """
        if self._structure is None or len(self._positions) == 0:
            return None
        points = [self._positions]
        cell = self._cell_or_none()
        if cell is not None:
            vectors = [cell[i] for i, periodic in enumerate(self._structure.pbc) if periodic]
            if vectors:
                points.append(
                    np.asarray(
                        [
                            sum(combination)
                            for combination in itertools.product(
                                *[(np.zeros(3), v) for v in vectors]
                            )
                        ],
                        dtype=float,
                    )
                )
        stacked = np.vstack(points)
        low = stacked.min(axis=0) - margin
        high = stacked.max(axis=0) + margin
        return (low[0], high[0], low[1], high[1], low[2], high[2])

    def _cell_or_none(self) -> Optional[np.ndarray]:
        """The displayed structure's lattice, or ``None`` if it has no usable one."""
        if self._structure is None:
            return None
        cell = np.asarray(self._structure.cell, dtype=float)
        if cell.shape != (3, 3) or abs(np.linalg.det(cell)) < 1e-8:
            return None
        return cell

    def _update_polyhedra(self, positions: np.ndarray) -> None:
        """Move the polyhedra (and their outline) with the atoms they sit on.

        Coordination is fixed for the duration of an animation, so the hulls just
        follow their ligands: each vertex is re-placed on its atom (plus the
        constant lattice offset of the image it belongs to). No re-hulling, no
        near-neighbour analysis — a per-frame array gather.
        """
        for mesh, follow in (
            (self._polyhedra_mesh_obj, self._polyhedra_follow),
            (self._polyhedra_edge_mesh, self._polyhedra_edge_follow),
        ):
            if mesh is None or follow is None:
                continue
            index, offset = follow
            mesh.points = positions[index] + offset

    def _coordination_polyhedra(self, analysis, min_vertices, connectivity) -> list:
        """The coordination polyhedra for ``analysis``, cached across rebuilds.

        CrystalNN is the interactive-redraw bottleneck, and it depends only on the
        analysed geometry and ``min_vertices`` — not on colours, opacity, camera or
        any other display setting. So we key a one-entry cache on the geometry and
        reuse the result whenever only presentation changed.
        """
        positions = np.asarray(analysis.positions, dtype=float)
        numbers = np.asarray(analysis.numbers, dtype=int)
        # bond_tolerance only matters to the distance fallback, but it's cheap to
        # include and keeps that path correct when the tolerance slider moves.
        key = (
            numbers.tobytes(),
            np.round(positions, 4).tobytes(),
            # The cell too: a strained lattice with the atoms left where they
            # were changes every periodic neighbour, and was served from here.
            np.round(np.asarray(analysis.cell, dtype=float), 4).tobytes(),
            int(min_vertices),
            round(float(self._settings.bond_tolerance), 4),
        )
        if key == self._poly_cache_key:
            return self._poly_cache_found

        conn = connectivity(analysis, min_vertices)
        found = list(conn.polyhedra) if conn is not None else []
        if not found:  # periodic distance search on the clean cell (CrystalNN missing/empty)
            found = _fallback_polyhedra(
                analysis.to_ase(), self._settings.bond_tolerance, min_vertices
            )
        self._poly_cache_key = key
        self._poly_cache_found = found
        return found

    def _polyhedra_mesh(self) -> Optional["pv.PolyData"]:
        """Build the polyhedra mesh: CrystalNN on the clean cell, else a fallback.

        Both paths analyse the clean ``bond_structure`` (the periodic cell before
        boundary-completion), which a near-neighbour algorithm can handle — the
        packed cell's duplicate images would confuse it. The distance fallback
        runs whenever CrystalNN is unavailable, too slow, *or* found no polyhedra
        — previously an empty CrystalNN result drew nothing (the coordination
        showed up only on big supercells, where the atom count tipped it into the
        fallback).

        The analysed polyhedra are then replicated onto every *shown* centre, so
        the image atoms that boundary-completion adds are coordinated too rather
        than sitting bare next to their neighbours.
        """
        min_vertices = self._settings.polyhedra_min_vertices
        analysis = self._bond_structure if self._bond_structure is not None else self._structure

        from crystalline.core.bonds import connectivity, replicate_polyhedra

        analysis = _one_cell(analysis, self._reference_cell)
        found = self._coordination_polyhedra(analysis, min_vertices, connectivity)
        if not found:
            return None

        shown = replicate_polyhedra(found, analysis.cell, self._positions, self._numbers)
        overrides = dict(self._settings.atom_colors)  # tint polyhedra by the centre's colour
        return _hull_mesh(shown, overrides)

    def _update_bonds(self, positions: np.ndarray) -> None:
        """Replace the bond mesh for a new set of positions (no render here)."""
        if self._bond_actor is not None:
            self.plotter.remove_actor(self._bond_actor, render=False)
            self._bond_actor = None
        if self._settings.show_bonds:
            self._draw_bonds(positions, self._structure.numbers)

    def _draw_cell(self) -> None:
        if self._structure is None or not self._structure.is_periodic:
            return
        # Outline the reference (original) cell when one is set — e.g. a supercell
        # of atoms is shown but the lines should mark the original unit cell.
        cell = self._reference_cell if self._reference_cell is not None else self._structure.cell
        cell = np.asarray(cell, dtype=float)
        if np.allclose(cell, 0.0):
            return
        # Only the periodic vectors span the drawn cell — a slab/polymer's formal
        # 500 Å vacuum edge is not a real cell edge and must not be outlined.
        vectors = [cell[i] for i, p in enumerate(self._structure.pbc) if p]
        if not vectors:
            return
        edges = _cell_edges(vectors)
        self._cell_actor = self.plotter.add_mesh(
            edges, color=self._settings.cell_color, line_width=self._settings.cell_line_width
        )
        self._cell_actor.SetPickable(False)

    # ── thermal ellipsoids (ADP) ────────────────────────────────────────
    def _ellipsoid_atoms(self) -> np.ndarray:
        """Boolean mask of the atoms that get a thermal ellipsoid.

        Cached: the mask depends on the tensors, the probability and the toggle —
        never on where the atoms currently *are* — but it is consulted on every
        re-glyph, which happens per animation frame and per drag mouse-move.
        Recomputing it there diagonalised every tensor again for nothing. The
        cache is dropped by :meth:`_rebuild` and :meth:`set_adp_tensors`, which
        between them cover every way an input can change.
        """
        if self._ellipsoid_mask is None:
            self._ellipsoid_mask = self._compute_ellipsoid_atoms()
        return self._ellipsoid_mask

    def _compute_ellipsoid_atoms(self) -> np.ndarray:
        """Which atoms qualify for an ellipsoid (see :meth:`_ellipsoid_atoms`).

        An atom qualifies when ellipsoids are switched on, a tensor for it is
        loaded, and that tensor is big enough to draw. A tensor that is all but
        zero — an atom the sampling says doesn't move, or a non-positive-definite
        one clamped flat — is left to its ordinary sphere rather than rendered as
        an invisible speck.
        """
        blank = np.zeros(len(self._positions), dtype=bool)
        if not self._settings.show_adp_ellipsoids or self._adp_tensors is None:
            return blank
        if self._adp_tensors.shape != (len(self._positions), 3, 3):
            return blank  # tensors and displayed geometry out of step
        return self._adp_radii()[:, 2] >= _MIN_ELLIPSOID_RADIUS

    def _sphere_atoms(self) -> np.ndarray:
        """Boolean mask of the atoms drawn as plain spheres (the rest)."""
        return ~self._ellipsoid_atoms()

    def _adp_radii(self) -> np.ndarray:
        """``(natom, 3)`` ascending semi-axis lengths at the set probability."""
        from crystalline.core.adp import ellipsoid_radii

        return ellipsoid_radii(self._adp_tensors, self._settings.adp_probability)

    def set_adp_tensors(self, tensors: Optional[np.ndarray], redraw: bool = True) -> None:
        """Supply ``(natom, 3, 3)`` cartesian ADP tensors (Å²), or ``None``.

        Only stores them; whether they are drawn and at what probability is a
        display setting. A set that doesn't match the displayed atom count is
        kept but not drawn, so switching cell view or supercell doesn't silently
        discard the file's ADPs.

        ``redraw=False`` stages the tensors for a rebuild the caller is about to
        trigger anyway — changing the temperature changes both these tensors and
        the settings, and rebuilding twice makes the view flicker.
        """
        self._adp_tensors = None if tensors is None else np.asarray(tensors, dtype=float)
        self._ellipsoid_mask = None  # a new tensor set decides anew who gets one
        if redraw:
            self._rebuild()

    def _draw_adp_ellipsoids(self) -> None:
        """One merged mesh holding every atom's displacement ellipsoid.

        Each ellipsoid is a unit sphere stretched along the tensor's principal
        axes: a point ``y`` on the sphere maps to ``centre + (y * radii) @ axes``
        because ``axes`` holds the principal directions as rows. The spheres
        share one triangulation, so the whole field is built by transforming
        points and re-offsetting the same face table — one actor, as for atoms.
        """
        from crystalline.core.adp import ellipsoid_axes

        shown = self._ellipsoid_atoms()
        if not shown.any():
            return
        resolution = _sphere_resolution(len(self._numbers))
        template = pv.Sphere(radius=1.0, theta_resolution=resolution, phi_resolution=resolution)
        unit = np.asarray(template.points, dtype=float)
        faces = np.asarray(template.faces, dtype=np.int64).reshape(-1, 4)
        rgb = self._rgb_for(self._numbers)

        points, all_faces, colors = [], [], []
        for offset, index in enumerate(np.nonzero(shown)[0]):
            radii, axes, _pd = ellipsoid_axes(
                self._adp_tensors[index], self._settings.adp_probability
            )
            points.append(self._positions[index] + (unit * radii) @ axes)
            block = faces.copy()
            block[:, 1:] += offset * len(unit)
            all_faces.append(block)
            colors.append(np.repeat(rgb[index][None, :], len(unit), axis=0))

        mesh = pv.PolyData(np.vstack(points), np.vstack(all_faces).ravel())
        mesh["rgb"] = np.vstack(colors)
        # Which atom each vertex belongs to, and where it sits relative to it, so
        # an animation can carry the ellipsoids along without rebuilding them:
        # the tensor is fixed through a vibration, only the centre moves.
        self._adp_mesh_obj = mesh
        self._adp_follow = (
            np.repeat(np.nonzero(shown)[0], len(unit)),
            np.vstack([block - self._positions[i]
                       for i, block in zip(np.nonzero(shown)[0], points)]),
        )
        self._adp_actor = self.plotter.add_mesh(
            mesh,
            scalars="rgb",
            rgb=True,
            opacity=self._settings.adp_opacity,
            render=False,
            **_ATOM_MATERIAL,
        )

    def _update_adp_ellipsoids(self, positions: np.ndarray) -> None:
        """Slide the ellipsoids onto ``positions`` (an animation frame).

        A vibration moves the atoms without changing their displacement tensors,
        so each ellipsoid keeps its shape and orientation and only its centre
        moves — a point translation, not a rebuild.
        """
        if self._adp_mesh_obj is None or self._adp_follow is None:
            return
        index, offset = self._adp_follow
        self._adp_mesh_obj.points = positions[index] + offset

    # ── phonon displacement arrows ──────────────────────────────────────
    def _redraw_mode_arrows(self) -> None:
        """Rebuild the arrow field in place (no render — the caller does that)."""
        if self._arrow_actor is not None:
            self.plotter.remove_actor(self._arrow_actor, render=False)
            self._arrow_actor = None
        self._draw_mode_arrows()

    def _draw_mode_arrows(self) -> None:
        """One glyphed actor holding every displacement arrow.

        Arrows start at the atoms as currently *drawn*, so they travel with the
        atoms through an animation instead of hanging in space.

        By default every arrow is the same length: it shows which way an atom
        moves, not how far. Scaling by displacement would leave the arrows of a
        delocalised mode — where no atom stands out — as a field of specks, and
        in a mode dominated by one hydrogen it would erase every other atom's
        direction entirely; how far each atom moves is readable in two better
        places, the animation itself and the composition line in the Phonons
        panel.

        ``mode_arrow_proportional`` turns that off, scaling each arrow by the
        atom's displacement with the longest at ``mode_arrow_scale``. That is
        the reading a mode away from Gamma needs: its atoms differ from cell to
        cell only in *phase*, so a snapshot distinguishes them only by how far
        each is moving at that instant — uniform arrows draw a travelling wave
        as though every cell were doing the same thing.

        Atoms barely involved in the mode get no arrow at all — below a small
        fraction of the largest displacement, an arrow would claim a motion that
        isn't there (and, drawn uniform, would claim a large one).
        """
        if not self._settings.show_mode_arrows or self._mode_vectors is None:
            return
        vectors = self._mode_vectors
        if vectors.shape != self._positions.shape or len(vectors) == 0:
            return  # modes and displayed geometry out of step
        lengths = np.linalg.norm(vectors, axis=1)
        peak = float(lengths.max()) if len(lengths) else 0.0
        if peak <= 0.0:
            return  # a null mode has nothing to draw
        keep = lengths >= _MIN_ARROW_FRACTION * peak
        if not keep.any():
            return

        cloud = pv.PolyData(self._positions[keep])
        directions = vectors[keep] / lengths[keep, None]
        if self._settings.mode_arrow_proportional:
            # Length carries the displacement, normalised so the most-displaced
            # atom gets the configured length whatever the eigenvector's scale.
            directions = directions * (lengths[keep, None] / peak)
        cloud["vectors"] = directions * self._settings.mode_arrow_scale
        phases = self._arrow_phases(keep)
        if phases is not None:
            cloud["phase"] = phases
        # factor=1: the glyph is already scaled to Angstrom by "vectors" above.
        # Radii are fractions of the arrow's own length, so a short arrow stays
        # proportioned rather than degenerating into a line.
        glyph = cloud.glyph(geom=pv.Arrow(tip_length=0.3, tip_radius=0.13, shaft_radius=0.05),
                            orient="vectors", scale="vectors", factor=1.0)
        if phases is not None:
            # A cyclic colormap, and a full-turn range: phase is an angle, so 0
            # and 2*pi must land on the same colour or the wave shows a seam
            # where there is none.
            self._arrow_actor = self.plotter.add_mesh(
                glyph, scalars="phase", cmap=_PHASE_COLORMAP, clim=(0.0, 2.0 * np.pi),
                smooth_shading=True, show_scalar_bar=False, render=False,
            )
        else:
            self._arrow_actor = self.plotter.add_mesh(
                glyph, color=self._settings.mode_arrow_color, smooth_shading=True, render=False
            )
        self._arrow_actor.SetPickable(False)  # only atoms are pick targets

    def _arrow_phases(self, keep) -> Optional[np.ndarray]:
        """Per-arrow Bloch phase in ``[0, 2*pi)``, or ``None`` to use one colour.

        ``None`` whenever the colouring would say nothing: the setting is off,
        the mode is at Gamma (no phase to show — every cell moves together), or
        the phases don't match the atoms on screen.
        """
        if not self._settings.mode_arrow_phase_colors or self._mode_phases is None:
            return None
        if self._mode_phases.shape != (len(self._positions),):
            return None
        phases = np.mod(self._mode_phases[keep], 2.0 * np.pi)
        if np.ptp(phases) <= 0.0:
            return None  # one phase everywhere: a single colour says it better
        return phases

    def _lattice_marker(self):
        """The a/b/c gizmo, pinned to a fixed corner of the viewport.

        Returns a prop assembly of labelled arrows — one per *periodic*
        direction only, so a slab shows a and b and a polymer just a, never a
        spurious c along CRYSTAL's formal 500 Angstrom vacuum vector — or
        ``None`` when there is no lattice to draw. The arrows share one length,
        so the gizmo shows orientation alone and never rescales with the lattice
        parameters (editing a, b or c must not grow or shrink it).

        It used to be an ordinary set of arrows anchored below the structure's
        lower corner. That put it at a *world* position, so every camera change
        moved it: the three axis-alignment views each showed it somewhere
        different, which is exactly what a gizmo must not do.

        A VTK orientation-marker widget instead lives in screen space — always
        the same corner, always the same size — and only turns to follow the
        camera. It is built from the real lattice vectors rather than a stock
        xyz marker, so a non-orthogonal cell shows its true angles (a hexagonal
        cell's a and b really do come out 120 degrees apart).

        The widget survives ``plotter.clear()``, unlike an actor, so it is
        created once and only its marker is swapped when the cell changes.
        """
        if self._structure is None or not self._structure.is_periodic:
            return None
        cell = self._reference_cell if self._reference_cell is not None else self._structure.cell
        cell = np.asarray(cell, dtype=float)
        if np.allclose(cell, 0.0):
            return None

        # vtkPropAssembly, not vtkAssembly: the latter pokes its own matrix into
        # every child, which overrides anything a child computes for itself. A
        # prop assembly just groups props and lets each render on its own terms,
        # which is what the 2D labels below need.
        assembly = vtk.vtkPropAssembly()
        drawn = 0
        for axis, periodic in enumerate(self._structure.pbc):
            if not periodic:
                continue
            vec = cell[axis]
            length = float(np.linalg.norm(vec))
            if length < 1e-6:
                continue
            direction = vec / length
            arrow = pv.Arrow(
                start=(0.0, 0.0, 0.0), direction=direction, scale=_LATTICE_ARROW_LENGTH
            )
            assembly.AddPart(_unlit_actor(arrow, _LATTICE_COLORS[axis]))
            assembly.AddPart(
                _axis_label_actor(
                    _LATTICE_LABELS[axis],
                    direction * _LATTICE_ARROW_LENGTH * _LATTICE_LABEL_OFFSET,
                    _LATTICE_COLORS[axis],
                )
            )
            drawn += 1
        if not drawn:
            return None
        assembly.AddPart(_bounds_padding_actor(_LATTICE_ARROW_LENGTH * _GIZMO_BOUNDS_PADDING))
        return assembly

    def _set_orientation_widget(self, marker) -> None:
        """Show ``marker`` in the viewport corner (``None`` empties the corner)."""
        if marker is None:
            if self._orientation_widget is not None:
                self._orientation_widget.SetEnabled(0)
            return
        if self._orientation_widget is None:
            try:
                self._orientation_widget = self.plotter.add_orientation_widget(
                    marker, viewport=_GIZMO_VIEWPORT
                )
            except Exception:  # noqa: BLE001 - no interactor (bare off-screen plotter)
                self._orientation_widget = None
            return
        widget = self._orientation_widget
        widget.SetOrientationMarker(marker)
        # The viewport is re-asserted on every swap rather than set once at
        # creation: it is the only thing that says how big the corner is, and a
        # widget that has been through anything that resizes it (see
        # _update_orientation_marker) would otherwise come back invisible.
        widget.SetViewport(*_GIZMO_VIEWPORT)
        widget.SetEnabled(1)


# ── free helpers ─────────────────────────────────────────────────────────
def _sample_nearest(values: np.ndarray, field, points: np.ndarray) -> np.ndarray:
    """``values`` (on ``field``'s grid) at the grid point nearest each of ``points``.

    For a quantity that is constant over a region and jumps between regions —
    a per-cell phase — this is the only correct sampling: interpolating it would
    invent the values in between.
    """
    index = np.rint(
        (np.asarray(points, dtype=float) - field.origin) / field.spacing
    ).astype(int)
    for axis in range(3):
        np.clip(index[:, axis], 0, field.shape[axis] - 1, out=index[:, axis])
    return values[index[:, 0], index[:, 1], index[:, 2]]


def _unlit_actor(mesh, color: str):
    """A flat-shaded actor for ``mesh`` — a gizmo reads best without lighting."""
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*(c / 255.0 for c in _hex_to_rgb(color)))
    actor.GetProperty().SetLighting(False)
    actor.SetPickable(False)
    return actor


def _bounds_padding_actor(radius: float):
    """An invisible sphere that only exists to widen the gizmo's bounds."""
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(pv.Sphere(radius=radius, theta_resolution=8, phi_resolution=8))
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(0.0)
    actor.SetPickable(False)
    return actor


def _axis_label_actor(text: str, position: np.ndarray, color: str):
    """A screen-aligned "a"/"b"/"c" pinned to the 3D point ``position``.

    A ``vtkCaptionActor2D``: the glyph is drawn in the image plane but anchored
    to a world point, so it tracks its arrow tip while staying upright and the
    same size. This is what ``vtkAxesActor`` does for its own labels.

    3D text does not work here. Rendered as geometry it turns edge-on and
    vanishes whenever the axis it labels points at the viewer — precisely the
    situation in each of the a/b/c alignment views — and a ``vtkFollower``,
    which would normally swivel to face the camera, cannot fix it inside an
    assembly whose matrix is imposed on its children.
    """
    caption = vtk.vtkCaptionActor2D()
    caption.SetCaption(text)
    caption.SetAttachmentPoint(*np.asarray(position, dtype=float))
    caption.BorderOff()
    caption.LeaderOff()
    caption.ThreeDimensionalLeaderOff()
    caption.SetPadding(0)
    # A caption sizes its text to its own box, so the box is what sets the glyph
    # size; left at the default it would dwarf the arrows.
    caption.SetWidth(_LATTICE_LABEL_BOX)
    caption.SetHeight(_LATTICE_LABEL_BOX)

    text_property = caption.GetCaptionTextProperty()
    text_property.SetColor(*(c / 255.0 for c in _hex_to_rgb(color)))
    text_property.SetFontSize(_LATTICE_LABEL_FONT_SIZE)
    text_property.BoldOn()
    text_property.ItalicOff()
    text_property.ShadowOff()
    text_property.SetJustificationToCentered()
    text_property.SetVerticalJustificationToCentered()
    caption.SetPickable(False)
    return caption


def _structured_grid(field):
    """A field as a ``pv.StructuredGrid``, points given one by one.

    ``ImageData`` would be cheaper, but it can only hold an axis-aligned box of
    uniform spacing, and CRYSTAL's grid runs along the lattice vectors.
    """
    try:
        # VTK wants the first index varying fastest. A Fortran-order reshape
        # cannot be used here: it would fold the trailing xyz axis into the
        # ordering too, scrambling the points.
        points = field.points().transpose(2, 1, 0, 3).reshape(-1, 3)
        grid = pv.StructuredGrid()
        grid.points = points
        grid.dimensions = field.shape
        grid.point_data["value"] = np.asarray(field.values, dtype=float).ravel(order="F")
    except Exception:  # noqa: BLE001 - a field we cannot lay out is not drawn
        return None
    return grid


# How far past each face of the cell the grid is continued before contouring,
# in Angstrom: more than the radius of any surface worth drawing around an atom.
_SURFACE_PAD = 3.0


def _padded_field(field, lattice, pad: float):
    """The field continued periodically ``pad`` Angstrom past every face.

    Contoured over the one sampled cell, a surface around an atom that sits on
    a face is cut in two by the grid's edge, and one on a corner into eight.
    Reassembling those pieces from copies of the cell works only if every
    neighbour of that corner happens to be copied — and in MgO's primitive
    cell, where every Mg sits on a corner, some were and some were not, so
    equivalent atoms came out with a whole shell, part of one, or none.
    Continued past the faces, the grid holds each of those surfaces whole.
    """
    from dataclasses import replace

    lattice = np.asarray(lattice, dtype=float)
    counts = np.array(field.shape)
    inclusive = np.allclose(lattice, field.steps * (counts - 1)[:, None], atol=1e-6)
    core = field.values[:-1, :-1, :-1] if inclusive else field.values
    period = np.array(core.shape)
    lengths = np.linalg.norm(field.steps, axis=1)
    layers = np.minimum(np.ceil(pad / np.maximum(lengths, 1e-9)).astype(int), period // 2)
    # One more layer at the far end, so the padded grid closes on itself as the
    # original did: it runs from -layers to period + layers inclusive.
    values = np.pad(core, [(int(n), int(n) + 1) for n in layers], mode="wrap")
    origin = field.origin - layers @ field.steps
    return replace(field, values=values, origin=origin)


class _Piece:
    """One connected piece of a surface, as plain arrays.

    Kept as numpy rather than as a VTK mesh because placing pieces is what a
    rebuild does, and doing it on VTK meshes — one threshold filter per piece to
    split them, one merge per copy to join them — was where the seconds went.
    """

    __slots__ = ("points", "triangles", "values", "centre", "radius")

    def __init__(self, points, triangles, values):
        self.points = points
        self.triangles = triangles
        self.values = values
        self.centre = points.mean(axis=0)
        self.radius = float(np.linalg.norm(points - self.centre, axis=1).max())


def _mesh_pieces(mesh):
    """Split a triangle mesh into its connected pieces.

    A sparse connected-components pass over the triangles' edges: the same
    answer as VTK's ``split_bodies``, which runs a threshold filter per piece
    and costs a noticeable fraction of a second on a surface with many.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if mesh is None or mesh.n_points == 0:
        return []
    mesh = mesh.triangulate()
    faces = np.asarray(mesh.faces)
    if faces.size == 0:
        return []
    triangles = faces.reshape(-1, 4)[:, 1:]
    points = np.asarray(mesh.points, dtype=float)
    values = (np.asarray(mesh.point_data["value"], dtype=float)
              if "value" in mesh.point_data else None)
    n = len(points)
    rows = np.concatenate([triangles[:, 0], triangles[:, 1], triangles[:, 2]])
    cols = np.concatenate([triangles[:, 1], triangles[:, 2], triangles[:, 0]])
    graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
    count, labels = connected_components(graph, directed=False)
    face_label = labels[triangles[:, 0]]
    order = np.argsort(face_label, kind="stable")
    bounds = np.searchsorted(face_label[order], np.arange(count + 1))
    pieces = []
    for label in range(count):
        chosen = triangles[order[bounds[label]:bounds[label + 1]]]
        if len(chosen) == 0:
            continue
        used, local = np.unique(chosen, return_inverse=True)
        pieces.append(_Piece(points[used], local.reshape(-1, 3),
                             None if values is None else values[used]))
    return pieces


def _piece_mesh(piece):
    mesh = pv.PolyData.from_regular_faces(piece.points, piece.triangles)
    if piece.values is not None:
        mesh.point_data["value"] = piece.values
    return mesh


def _home_pieces(mesh, field, lattice, origin, clip_to_cell=False):
    """The pieces of a surface belonging to the home cell, each exactly once.

    Contoured on a padded grid, a surface appears once for each copy of its atom
    that the padding reaches: MgO's corner Mg is there eight times. Copied into
    the neighbouring cells, those copies landed on one another and were welded
    into one mesh with the same shell in it two to five times over — which,
    drawn translucent, made equivalent atoms come out darker or lighter than
    each other. So every piece is given to exactly one cell, and the cells are
    then tiled edge to edge:

    * a closed piece — one that does not reach the padding's edge — belongs to
      the cell its centre lies in, and is kept only if that is this one;
    * a small piece that does reach the edge is a scrap of a surface whose atom
      lies outside, whole in that atom's own cell, and is dropped;
    * a large one is a continuous sheet of density at a low isovalue, and is cut
      to this cell so that the copies meet at its faces instead of overlapping.
    """
    lattice = np.asarray(lattice, dtype=float)
    origin = np.asarray(origin, dtype=float)      # the home cell's, not the padding's
    to_index = np.linalg.inv(field.steps)
    to_cell = np.linalg.inv(lattice)
    last = np.array(field.shape, dtype=float) - 1.0
    kept = []
    for piece in _mesh_pieces(mesh):
        index = (piece.points - field.origin) @ to_index
        edge = bool(np.any(index < 0.5) or np.any(index > last - 0.5))
        if not edge:
            home = np.floor((piece.centre - origin) @ to_cell + _ON_FACE)
            if np.all(home == 0):
                kept.append(piece)
            continue
        if float(np.ptp(piece.points, axis=0).max()) < _SURFACE_PAD:
            continue
        kept.extend(_clipped(piece, lattice, origin))
    if clip_to_cell:
        kept = [part for piece in kept for part in _clipped(piece, lattice, origin)]
    return kept


def _clipped(piece, lattice, origin):
    """``piece`` cut to the cell at ``origin``, as pieces again."""
    moved = _piece_mesh(piece).translate(-origin, inplace=False)
    clipped = _clip_to_cell(moved, lattice)
    if clipped is None or clipped.n_points == 0:
        return []
    clipped = clipped.extract_surface().translate(origin, inplace=False)
    return _mesh_pieces(clipped)


def _assemble_pieces(pieces, offsets, positions, cutoff: float = None):
    """One mesh of every piece copied to every offset that an atom lies against.

    A crystal's density fills space, but a view holds only the atoms someone
    asked for, and density around the others reads as the field being in the
    wrong place — so a copy is kept when a drawn atom comes within ``cutoff``
    of it. A k-d tree over the atoms answers that for each copy without the
    full distance matrix, and the kept copies are concatenated as arrays and
    turned into a mesh once, rather than merged one at a time.
    """
    from scipy.spatial import cKDTree

    cutoff = _NEAR_ATOM if cutoff is None else cutoff
    positions = np.asarray(positions, dtype=float) if positions is not None else np.empty((0, 3))
    tree = cKDTree(positions) if len(positions) else None
    points, triangles, values = [], [], []
    count = 0
    for offset in offsets:
        offset = np.asarray(offset, dtype=float)
        for piece in pieces:
            if tree is not None:
                reach = tree.query_ball_point(piece.centre + offset, piece.radius + cutoff)
                if not reach:
                    continue
                gaps, _ = cKDTree(positions[reach]).query(
                    piece.points + offset, k=1, distance_upper_bound=cutoff)
                if not np.isfinite(gaps).any():
                    continue
            points.append(piece.points + offset)
            triangles.append(piece.triangles + count)
            if piece.values is not None:
                values.append(piece.values)
            count += len(piece.points)
    if not points:
        return None
    mesh = pv.PolyData.from_regular_faces(np.vstack(points), np.vstack(triangles))
    if values and len(values) == len(points):
        mesh.point_data["value"] = np.concatenate(values)
    return mesh


# A centre this close to a cell face, as a fraction of the cell, counts as on
# it. A mesh's centroid wobbles by a hundredth of an Angstrom about the nucleus,
# and a surface must not fall between two cells because of that.
_ON_FACE = 1e-3


def _density_surfaces(grid, field, options):
    """Surfaces of constant value: one for a density, two for a signed field.

    Left where the field puts them; the caller repeats and clips them.
    """
    level = abs(float(options.isovalue))
    if level <= 0.0:
        return []
    levels = [(-level, options.negative), (level, options.positive)] if field.signed \
        else [(level, options.positive)]
    out = []
    for value, colour in levels:
        surface = grid.contour([value], scalars="value")
        if surface is None or surface.n_points == 0:
            continue          # this level is not crossed anywhere in the cell
        surface.point_data.clear()
        if options.colour_by is not None:
            surface = _paint(surface, options.colour_by)
        out.append((surface, colour))
    return out


# How close a piece of surface has to come to an atom on screen to be drawn.
# A surface around an atom passes within a fraction of an Angstrom of it; one
# around an atom that is not drawn is a bond length away.
_NEAR_ATOM = 1.0


def _colour_limits(values, symmetric: bool):
    """The colour range for a painted mesh, set by its bulk rather than its extremes.

    A slice through MgO runs from 10⁻² e/bohr³ in the interstices to 10³ on the
    nuclei; even on a logarithmic scale the few points at the nuclei took the
    top of the map to themselves and left the rest of the plane one shade of
    blue. The 2nd and 98th percentiles bound the part worth telling apart, and
    the handful of points beyond them take the end colours. A signed field is
    kept centred on zero, so that white still means zero.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    if symmetric:
        extreme = float(np.percentile(np.abs(values), 98))
        extreme = extreme if extreme > 0 else float(np.abs(values).max()) or 1.0
        return (-extreme, extreme)
    low, high = (float(v) for v in np.percentile(values, [2, 98]))
    if high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        high = low + 1.0
    return (low, high)


def _paint(mesh, other):
    """Replace a mesh's scalars with a second field's values at its vertices.

    This is the electrostatic potential painted onto a density surface: the
    surface says where the molecule ends, the colour says what a charge would
    feel there. Read periodically, since a surface copied into a neighbouring
    cell lies outside the grid the potential was sampled on.
    """
    from crystalline.crystalio import density as density_module

    mesh.point_data["value"] = density_module.sample_periodic(other, np.asarray(mesh.points))
    return mesh


def _density_images(field, cell, positions):
    """The cells to copy the field into: the home cell, plus those the atoms need.

    An atom drawn outside the sampled cell needs its own copy of the field, or
    it is shown bare while its density sits a lattice translation away. But
    "outside" covers two different things, and only one of them wants a copy:

    * A **supercell** — which is how more of the field is shown: it follows the
      atoms, and there is no separate count of cells to set — or a structure
      whose atoms were written in another image
      — beryllium's two atoms come out one ``b`` and one ``a + c`` away, a
      third of a cell clear of the home one. Nothing in the home cell's field
      reaches them.
    * The **stragglers of a boundary-completed view**, where a molecule cut by
      the cell edge is redrawn whole and a few of its atoms poke just past the
      face. The home cell's own surface already reaches those: they are a
      hundredth of a cell out, not a third.

    Every cell holding a drawn atom is copied into. That over-reaches — a cell
    entered by one straggler brings a whole cell of density with it, most of it
    around atoms nobody drew — which is why the copies are then cut back to the
    atoms by :func:`_near_atoms`.
    """
    wanted = {(0, 0, 0)}
    cell = np.asarray(cell, dtype=float)
    if positions is not None and len(positions):
        try:
            inverse = np.linalg.inv(cell)
        except np.linalg.LinAlgError:
            inverse = None
        if inverse is not None:
            fractional = (np.asarray(positions, dtype=float) - field.origin) @ inverse
            # Each surface belongs to the cell its centre is in, and an atom on
            # a face could be given to either side of it: ask for both.
            corners = set()
            for shift in itertools.product((-_ON_FACE, 0.0, _ON_FACE), repeat=3):
                for corner in np.floor(fractional + np.array(shift)).astype(int):
                    corners.add(tuple(int(v) for v in corner))
            wanted |= corners
    if len(wanted) > _MAX_DENSITY_IMAGES:
        return [np.zeros(3)]
    ordered = sorted(wanted, key=lambda t: (t != (0, 0, 0), t))
    return [i * cell[0] + j * cell[1] + k * cell[2] for i, j, k in ordered]


# A slice's resolution along each side; past this a map is finer than the grid
# it is read from and only costs time.
_MAX_SLICE_POINTS = 400

# How far behind a slice the cutaway sits, in Angstrom: enough that nothing
# lying exactly in the plane pokes through it, not enough to be seen.
_CUTAWAY_BEHIND = 0.02
# An atom this close to a slice counts as lying in it, and is marked there.
_IN_PLANE = 0.15
# A mark's size, as a fraction of the atom's drawn radius.
_IN_PLANE_MARKER = 0.35

# How many cells the field is copied into at most. A supercell is how more of
# it is shown, so this has to cover the supercells people build; pieces are
# placed as arrays and a k-d tree, so a thousand cells is still quick.
_MAX_DENSITY_IMAGES = 1000


def _clip_to_cell(surface, cell):
    """Cut an isosurface back to the cell it belongs to.

    The orbital is sampled over the cell's *bounding* box, which for anything but
    an orthogonal cell is appreciably bigger — a rhombohedral cell fills about
    half of it. What is drawn out in the corners is a real part of the Bloch sum,
    but it belongs to neighbouring cells whose atoms are not on screen, so it
    reads as lobes floating in empty space. Clipping keeps the picture to the
    cell that is actually drawn; a supercell is how to see more of the orbital.

    Six half-space clips, one per face. Best-effort: a cell that cannot define
    them leaves the surface whole rather than losing it.
    """
    if cell is None:
        return surface
    cell = np.asarray(cell, dtype=float)
    if cell.shape != (3, 3) or abs(np.linalg.det(cell)) < 1e-8:
        return surface
    try:
        for axis in range(3):
            # The face's outward normal is perpendicular to the other two edges,
            # which is not the edge direction itself unless the cell is orthogonal.
            normal = np.cross(cell[(axis + 1) % 3], cell[(axis + 2) % 3])
            length = np.linalg.norm(normal)
            if length < 1e-12:
                return surface
            normal = normal / length
            if np.dot(normal, cell[axis]) < 0:
                normal = -normal  # point it out of the cell, not into it
            # pyvista's invert=True keeps what lies *below* the plane. The cell
            # is above the face at the origin and below the one at cell[axis].
            for origin, invert in ((np.zeros(3), False), (cell[axis], True)):
                surface = surface.clip(normal=normal, origin=origin, invert=invert)
                if surface.n_points == 0:
                    return surface
    except Exception:  # noqa: BLE001 - purely cosmetic; never lose the orbital
        return surface
    return surface


def _sphere_radius(z, scale: float = _ATOM_SCALE):
    """Drawn sphere radius for atomic number(s) ``z`` (scalar or array in → same out)."""
    scalar = np.isscalar(z) or np.asarray(z).ndim == 0
    drawn = elements.radii(z) * scale
    return float(drawn[0]) if scalar else drawn


def _hex_to_rgb(color: str) -> np.ndarray:
    """Parse ``"#rrggbb"`` (or ``"#rgb"``) into a uint8 RGB triple."""
    text = color.lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    return np.array([int(text[k : k + 2], 16) for k in (0, 2, 4)], dtype=np.uint8)


# Two atoms both below this Pauling electronegativity are treated as cations and
# not bonded to each other — stops covalent radii from spuriously bonding big
# cations (Ca–Ca, Ca–Si) in ionic crystals, while leaving C–C, C–O, Si–O, etc.
_CATION_ELECTRONEGATIVITY = 2.0


@lru_cache(maxsize=None)
def _electronegativity(z: int) -> float:
    try:
        from pymatgen.core import Element

        value = Element.from_Z(int(z)).X
        return float(value) if value and not np.isnan(value) else 0.0
    except Exception:  # noqa: BLE001 - pymatgen missing / element without EN
        return 0.0


def _bonded_pairs(positions: np.ndarray, numbers: np.ndarray, tolerance: float):
    """Return the (i, j) index arrays of bonded atom pairs (KD-tree neighbour search).

    Two atoms bond if their distance < ``tolerance * (r_i + r_j)`` AND they are
    not both cations. The KD-tree finds only pairs within the maximum possible
    bond length, avoiding the O(N^2) all-pairs loop that dominated load/redraw
    time for large structures.
    """
    positions = np.asarray(positions, dtype=float)
    if len(positions) < 2:
        return np.empty(0, int), np.empty(0, int)
    radii = elements.radii(numbers)
    max_bond = tolerance * 2.0 * float(radii.max())
    candidate_pairs = cKDTree(positions).query_pairs(r=max_bond, output_type="ndarray")
    if len(candidate_pairs) == 0:
        return np.empty(0, int), np.empty(0, int)
    i, j = candidate_pairs[:, 0], candidate_pairs[:, 1]
    dists = np.linalg.norm(positions[i] - positions[j], axis=1)
    keep = dists < tolerance * (radii[i] + radii[j])
    i, j = i[keep], j[keep]
    if len(i) == 0:
        return i, j
    en = np.array([_electronegativity(int(z)) for z in numbers])
    both_cations = (en[i] < _CATION_ELECTRONEGATIVITY) & (en[j] < _CATION_ELECTRONEGATIVITY)
    return i[~both_cations], j[~both_cations]


def _bond_mesh(
    positions: np.ndarray,
    numbers: np.ndarray,
    radius: float,
    tolerance: float,
    pairs: Optional[tuple] = None,
) -> Optional[pv.PolyData]:
    """Bond tube mesh for the bonded pairs (radius/tolerance from settings).

    ``pairs`` supplies a ready-made connectivity — from the equilibrium geometry
    during an animation — while the tubes are always drawn between the live
    ``positions``. That keeps the bond network fixed through a vibration instead
    of letting bonds wink out whenever a stretch crosses the distance criterion.
    """
    positions = np.asarray(positions, dtype=float)
    i, j = _bonded_pairs(positions, numbers, tolerance) if pairs is None else pairs
    if len(i) == 0:
        return None
    pts = np.empty((2 * len(i), 3), dtype=float)
    pts[0::2] = positions[i]
    pts[1::2] = positions[j]
    lines = np.empty((len(i), 3), dtype=np.int64)
    lines[:, 0] = 2
    lines[:, 1] = np.arange(0, 2 * len(i), 2)
    lines[:, 2] = np.arange(1, 2 * len(i), 2)
    poly = pv.PolyData(pts)
    poly.lines = lines.ravel()
    return poly.tube(radius=radius)


def _dashed_lines(segments: np.ndarray, dash: float, gap: float) -> Optional[pv.PolyData]:
    """A line mesh that renders ``segments`` (``(M, 2, 3)``) as dashes.

    VTK's line stipple is unreliable across its OpenGL backends, so the dashes are
    real geometry: each segment is chopped into ``dash``-long pieces separated by
    ``gap`` (Å). Cheap here because hydrogen bonds are few.
    """
    pts: list = []
    lines: list = []
    idx = 0
    for start, end in np.asarray(segments, dtype=float):
        vec = end - start
        length = float(np.linalg.norm(vec))
        if length < 1e-6:
            continue
        direction = vec / length
        offset = 0.0
        while offset < length:
            a = start + direction * offset
            b = start + direction * min(offset + dash, length)
            pts.append(a)
            pts.append(b)
            lines.append([2, idx, idx + 1])
            idx += 2
            offset += dash + gap
    if not pts:
        return None
    poly = pv.PolyData(np.asarray(pts, dtype=float))
    poly.lines = np.hstack(lines)
    return poly


def _draw_over_scene(actor) -> None:
    """Make ``actor`` render in front of the structure instead of inside it.

    A measurement between two bonded atoms runs almost entirely *inside* their
    spheres, so an occluded line is invisible exactly when it matters. VTK's
    coincident-topology offset pulls the actor towards the camera in depth only
    — geometry and picking are untouched. Best-effort: if the VTK build doesn't
    expose the knobs, the annotation simply draws normally.
    """
    try:
        mapper = actor.GetMapper()
        mapper.SetResolveCoincidentTopologyToPolygonOffset()
        for setter in (
            "SetRelativeCoincidentTopologyLineOffsetParameters",
            "SetRelativeCoincidentTopologyPolygonOffsetParameters",
            "SetRelativeCoincidentTopologyPointOffsetParameter",
        ):
            method = getattr(mapper, setter, None)
            if method is None:
                continue
            try:
                method(_ANNOTATION_DEPTH_OFFSET, _ANNOTATION_DEPTH_OFFSET)
            except TypeError:  # the point variant takes a single value
                method(_ANNOTATION_DEPTH_OFFSET)
    except Exception:  # noqa: BLE001 - purely cosmetic; never break a redraw
        pass


def _polyline_tube(points: np.ndarray, radius: float = _ANNOTATION_LINE_RADIUS) -> pv.PolyData:
    """A thin tube along the path through ``points`` (2+ vertices)."""
    poly = pv.PolyData()
    poly.points = np.asarray(points, dtype=float)
    segments = len(points) - 1
    poly.lines = np.hstack([[2, k, k + 1] for k in range(segments)]).astype(np.int64)
    return poly.tube(radius=radius)


def _readable_on(background) -> str:
    """Black or white, whichever reads on ``background``."""
    try:
        from matplotlib.colors import to_rgb

        red, green, blue = to_rgb(background)
    except Exception:  # noqa: BLE001 - an unreadable colour: assume a light one
        return "black"
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "black" if luminance > 0.5 else "white"


def _label_spots(points: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Where an element's symbol could go: its own extremities, farthest first.

    Not the middle of the element, which is where every one of them meets: point
    symmetry puts all the axes and all the planes through a single centre, so a
    label at each element's centre stacks the whole lot on one spot and VTK drops
    all but the topmost. The reach of the element itself — the far end of an
    axis, the outer corners of a plane's patch — puts each label out where only
    that element is. Each spot is drawn back from the very edge so the text sits
    on the element rather than off the end of it.
    """
    points = np.asarray(points, dtype=float)
    centre = np.asarray(centre, dtype=float)
    spots = points - (points - centre) * _SYMMETRY_LABEL_INSET
    reach = np.linalg.norm(spots - centre, axis=1)
    # Only the outer half of the element's reach. An axis through a corner of the
    # drawn box leaves it again almost at once on the other side, and that stub is
    # a spot the spreading below would otherwise take to dodge a crowded corner —
    # putting the label back in the pile at the centre it was moved out of.
    spots = spots[reach >= 0.5 * reach.max()] if reach.max() > 0 else spots
    return spots[np.argsort(-np.linalg.norm(spots - centre, axis=1))]


def _spread_anchor(spots: np.ndarray, placed: list) -> np.ndarray:
    """Of an element's possible label spots, the one farthest from those taken.

    Several elements can reach the same corner of the drawn box — nine mirror
    planes through one centre certainly do — so first come, first served would
    still stack their labels. Each label goes to whichever of its own spots is
    most in the clear.
    """
    if not placed:
        return spots[0]
    taken = np.asarray(placed, dtype=float)
    clearance = np.linalg.norm(spots[:, None, :] - taken[None, :, :], axis=2).min(axis=1)
    return spots[int(np.argmax(clearance))]


def _plane_in_box(origin, normal, bounds) -> Optional[pv.PolyData]:
    """The piece of an infinite plane that lies inside an axis-aligned box.

    A plane primitive is a fixed square, which would hang out of the scene at the
    corners; cutting it back against the box's six faces leaves exactly the shape
    the plane makes with the box — a hexagon across a cube's diagonal as readily
    as a square. ``None`` when the plane misses the box entirely.
    """
    limits = np.asarray(bounds, dtype=float)
    size = float(np.linalg.norm(limits[1::2] - limits[0::2])) * 2.0  # over-large, then cut
    # One quad, not the default 10×10 grid: the clip below is what shapes it, and
    # the vertices that survive are then the polygon's own corners — which is
    # what a label looks for a spot among.
    patch = pv.Plane(
        center=origin, direction=normal, i_size=size, j_size=size,
        i_resolution=1, j_resolution=1,
    ).triangulate()
    for axis, name in enumerate("xyz"):
        for index, invert in ((2 * axis, False), (2 * axis + 1, True)):
            face = np.zeros(3)
            face[axis] = limits[index]
            patch = patch.clip(name, origin=face, invert=invert)
            if patch.n_points == 0:
                return None
    return patch


def _annotation_anchor(kind: str, points: np.ndarray, dihedral_kind: str) -> np.ndarray:
    """Where a measurement's value is written: mid-path, or at the angle's vertex."""
    if len(points) == 2:
        return points.mean(axis=0)
    if len(points) == 3:
        return points[1]  # the vertex of the angle
    if kind == dihedral_kind and len(points) == 4:
        return points[1:3].mean(axis=0)  # midpoint of the central bond
    return points.mean(axis=0)


def _follow_atoms(points, positions, cell: Optional[np.ndarray]):
    """Pin each mesh vertex to the atom it sits on: ``points == positions[i] + offset``.

    A polyhedron vertex *is* a ligand atom, but often a periodic image of one —
    coordination is analysed on the unit cell, and hulls are then replicated onto
    the shown images. Matching is therefore done modulo the lattice; the leftover
    offset is that image's lattice translation, which stays constant while the
    atoms vibrate. Positions are keyed by rounded (fractional) coordinates, so
    the whole map is one pass rather than a neighbour search.

    Returns ``(index, offset)`` arrays, or ``None`` if any vertex fails to match
    (the caller then simply leaves the polyhedra where they are).
    """
    points = np.asarray(points, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if len(points) == 0 or len(positions) == 0:
        return None

    def key_of(rows: np.ndarray) -> np.ndarray:
        if cell is None:  # no lattice (a molecule): plain cartesian identity
            return np.round(rows, 4)
        fractional = rows @ np.linalg.inv(cell)
        return np.round(fractional % 1.0, 4) % 1.0  # 0.99996 -> 1.0 -> 0.0

    lookup = {tuple(k): i for i, k in enumerate(key_of(positions))}
    index = np.empty(len(points), dtype=int)
    for n, k in enumerate(key_of(points)):
        found = lookup.get(tuple(k))
        if found is None:
            return None  # an unexpected vertex: don't animate rather than distort
        index[n] = found
    return index, points - positions[index]


def _center_rgb(center_z: int, overrides: Optional[dict]) -> np.ndarray:
    """RGB (0–255 floats) for a polyhedron centred on ``center_z`` — override or Jmol."""
    if overrides and int(center_z) in overrides:
        return _hex_to_rgb(overrides[int(center_z)]).astype(float)
    return elements.colour(int(center_z)) * 255


def _hull_mesh(polyhedra, overrides: Optional[dict] = None) -> Optional[pv.PolyData]:
    """Merge ``[(atomic_number, ligand_positions), …]`` into one coloured mesh.

    Each entry's ligands become a convex-hull polyhedron coloured by the central
    atom (honouring any ``overrides`` colour map); coplanar/degenerate sets are
    skipped. Returns ``None`` if nothing drawable.

    The hulls are concatenated by hand — points stacked, each hull's triangle
    indices shifted by the points already placed — rather than by folding
    ``PolyData.merge`` over them. That fold re-copied the whole accumulated mesh
    once per polyhedron, so the cost grew as the square of their number and
    dominated every redraw: a 1728-atom cell (864 hulls) spent 1.6 s of a 1.66 s
    rebuild in it. Building the arrays directly is the same operation in one pass
    and ~47x faster there, and it is what ``_draw_adp_ellipsoids`` already does
    for the ellipsoid field.

    Points are deliberately *not* welded (the old call passed
    ``merge_points=False``): neighbouring polyhedra share ligand atoms, and
    fusing those vertices would join two independent solids — the shared edge
    turns non-manifold and drops out of the outline, and shading bleeds from one
    polyhedron into the next. Keeping each hull's own copy of its points
    preserves that.
    """
    from scipy.spatial import ConvexHull

    try:  # QhullError moved across scipy versions
        from scipy.spatial import QhullError
    except ImportError:  # pragma: no cover - older scipy
        from scipy.spatial.qhull import QhullError

    points: list = []
    faces: list = []
    colors: list = []
    offset = 0
    for center_z, ligands in polyhedra:
        pts = np.asarray(ligands, dtype=float)
        if len(pts) < 4:
            continue
        try:
            hull = ConvexHull(pts)
        except (QhullError, ValueError):
            continue  # coplanar / too few independent points
        triangles = np.asarray(hull.simplices, dtype=np.int64)
        # VTK face table: each cell is [n_vertices, v0, v1, v2], and this hull's
        # vertices start at `offset` in the concatenated point array.
        block = np.empty((len(triangles), 4), dtype=np.int64)
        block[:, 0] = 3
        block[:, 1:] = triangles + offset
        points.append(pts)
        faces.append(block)
        colors.append(np.tile(_center_rgb(center_z, overrides), (len(pts), 1)))
        offset += len(pts)

    if not points:
        return None
    merged = pv.PolyData(np.vstack(points), np.vstack(faces).ravel())
    merged["colors"] = np.vstack(colors).astype(np.uint8)
    # Qhull doesn't wind its simplices consistently, so adjacent triangles can
    # come out with opposing normals — which shades the facets unevenly and, in
    # ``extract_feature_edges``, makes every edge look like a fold (a cube's 6
    # coplanar face diagonals then get outlined alongside its 12 real edges).
    return merged.compute_normals(consistent_normals=True, auto_orient_normals=True, inplace=False)


def _one_cell(analysis, unit_cell):
    """``analysis`` folded to one unit cell when it is a clean supercell of it.

    Coordination is periodic, so one cell's polyhedra replicated by lattice
    vectors are the supercell's. Analysing the supercell instead cost CrystalNN
    time in proportion to the number of cells and, past its size limit, handed
    the job to the distance fallback — so the polyhedra could change just
    because the supercell was made larger.

    Only an exact supercell is folded: every atom of the fold has to appear once
    per cell. A supercell edited in one tile is not periodic any more, and is
    analysed as it is.
    """
    from crystalline.core.cells import to_analysis_cell

    if unit_cell is None or analysis is None or not analysis.is_periodic:
        return analysis
    cell = np.asarray(unit_cell, dtype=float)
    try:
        ratio = abs(np.linalg.det(np.asarray(analysis.cell, dtype=float)) / np.linalg.det(cell))
    except Exception:  # noqa: BLE001 - a degenerate cell cannot be folded
        return analysis
    cells = int(round(ratio))
    if cells <= 1 or abs(ratio - cells) > 1e-3 or not np.all(analysis.pbc):
        return analysis
    try:
        folded = to_analysis_cell(analysis, cell)
    except Exception:  # noqa: BLE001 - analyse the supercell as it is
        return analysis
    if len(folded) * cells != len(analysis):
        return analysis
    return folded


def _fallback_polyhedra(atoms, tolerance: float, min_vertices: int) -> list:
    """Distance-based coordination polyhedra (used when CrystalNN isn't available).

    Uses ``ase.neighbor_list`` (periodic, so ligands cross the cell boundary
    correctly) for each atom's neighbours, then keeps only the chemically
    sensible ones: a polyhedron is drawn around a **cation** centre using its
    **anion** ligands — see :func:`crystalline.core.bonds.is_ligand`. Without
    that filter, covalent radii spuriously bond big cations to each other (a Ca
    "coordinating" nearby Ca/Si), so the hull swallows other cations; the filter
    reproduces the cation–anion polyhedra CrystalNN would give.

    Returns ``[(atomic_number, centre, ligand_positions), …]`` — the same shape
    :class:`~crystalline.core.bonds.Connectivity` uses, so both paths can be
    replicated onto the shown centres and hulled by the same code.
    """
    from collections import defaultdict

    if len(atoms) == 0:
        return []
    try:
        from ase.neighborlist import natural_cutoffs, neighbor_list

        i, j, offset = neighbor_list("ijD", atoms, natural_cutoffs(atoms, mult=tolerance))
    except Exception:  # noqa: BLE001 - bonding hiccup must not break rendering
        return []
    if len(i) == 0:
        return []

    positions = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()
    en = {int(z): _electronegativity(int(z)) for z in set(int(z) for z in numbers)}

    from crystalline.core.bonds import is_ligand

    # centre index -> list of ligand world positions, by the same rule CrystalNN's
    # path uses, so the two cannot draw different polyhedra for one crystal
    ligands: dict = defaultdict(list)
    for a, b, d in zip(i, j, offset):
        if is_ligand(en[int(numbers[a])], en[int(numbers[b])]):
            ligands[int(a)].append(positions[a] + d)  # neighbour image = positions[a] + d

    polyhedra = []
    for center, pts in ligands.items():
        if len(pts) < min_vertices:
            continue
        polyhedra.append(
            (int(numbers[center]), np.asarray(positions[center], dtype=float),
             np.asarray(pts, dtype=float))
        )
    return polyhedra


def _cell_edges(vectors) -> pv.PolyData:
    """Edges of the cell spanned by ``vectors`` (1–3 lattice vectors).

    Only the *periodic* lattice vectors are passed, so a bulk crystal (3 vectors)
    draws the full 12-edge box, a slab (2 vectors) a 4-edge parallelogram and a
    polymer (1 vector) a single segment — never the parallelepiped stretched along
    CRYSTAL's formal 500 Å vacuum direction.
    """
    vectors = [np.asarray(v, dtype=float) for v in vectors]
    n = len(vectors)
    # Every corner is a 0/1 combination of the spanning vectors; two corners share
    # an edge when their combinations differ in exactly one vector.
    combos = list(itertools.product((0, 1), repeat=n))
    corners = [sum((coeff[k] * vectors[k] for k in range(n)), np.zeros(3)) for coeff in combos]
    pts: list = []
    lines: list = []
    edge = 0
    for i, ci in enumerate(combos):
        for j in range(i + 1, len(combos)):
            if sum(a != b for a, b in zip(ci, combos[j])) == 1:
                pts.append(corners[i])
                pts.append(corners[j])
                lines.append([2, 2 * edge, 2 * edge + 1])
                edge += 1
    poly = pv.PolyData(np.asarray(pts, dtype=float))
    poly.lines = np.hstack(lines)
    return poly


__all__ = ["StructureRenderer"]
