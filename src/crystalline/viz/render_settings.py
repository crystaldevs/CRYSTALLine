"""Tunable appearance settings for :class:`~crystalline.viz.renderer.StructureRenderer`.

A plain, Qt-free dataclass so the renderer stays UI-agnostic and the values are
easy to unit-test and to drive from a settings dialog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class RenderSettings:
    """How the structure is drawn. All fields have sensible ball-and-stick defaults."""

    # atoms
    atom_scale: float = 0.5      # sphere radius as a fraction of the covalent radius
    atom_opacity: float = 1.0    # 1 = opaque, <1 = translucent
    # Per-element colour overrides as an immutable tuple of ``(atomic_number, "#rrggbb")``
    # pairs (kept hashable so the frozen dataclass stays well-behaved); elements
    # absent here keep their default Jmol colour.
    atom_colors: Tuple[Tuple[int, str], ...] = field(default_factory=tuple)

    # bonds — a single fixed colour (see renderer._BOND_COLOR); no per-bond colouring
    show_bonds: bool = True
    bond_radius: float = 0.06    # Angstrom
    bond_tolerance: float = 1.15  # bonded if dist < tolerance * (r_i + r_j)

    # hydrogen bonds — dashed D–H···A contacts, drawn by default
    show_hydrogen_bonds: bool = True

    # cell / gizmo
    show_cell: bool = True
    show_lattice_vectors: bool = True

    # phonon displacement arrows: the selected mode's eigenvector drawn on the
    # atoms, so a mode reads in a still image instead of only while animating.
    # Off by default — the animation is the primary view, and arrows on top of
    # it are clutter until asked for.
    #
    # Arrows are drawn at one length by default: they mark the *direction* each
    # atom moves, and letting them shrink with each atom's share leaves the
    # interesting small contributions invisible. ``mode_arrow_proportional``
    # gives the other reading, where an arrow's length is the atom's share of
    # the motion — the one that matters away from Gamma, where the arrows are
    # all a still picture has to show that the mode is a *wave*: same direction
    # cell after cell, amplitude swelling and reversing along q.
    #
    # ``mode_arrow_phase_colors`` colours each arrow by the Bloch phase of the
    # cell its atom sits in. Away from Gamma that phase is the *only* thing
    # distinguishing one cell from the next — the amplitude is identical
    # everywhere — so it is what a still image of a travelling wave has to show;
    # the colour cycles once per wavelength. It replaces the flat arrow colour,
    # and does nothing at Gamma, where there is one phase for the whole cell.
    show_mode_arrows: bool = False
    mode_arrow_scale: float = 1.2   # Angstrom: arrow length, or the longest one
    mode_arrow_proportional: bool = False
    mode_arrow_phase_colors: bool = False
    mode_arrow_color: str = "#d62728"

    # thermal ellipsoids (ADP): the surface enclosing `adp_probability` of each
    # atom's displacement distribution, drawn at one of the temperatures the
    # output reported. Off unless the loaded file actually carries ADPs.
    # Off here because most files carry no ADPs at all; the Display panel
    # switches it on when a file that has them is opened.
    show_adp_ellipsoids: bool = False
    # ORTEP and structure reports draw 50%; 99% is the default here because
    # computed ADPs are small enough that the conventional surface is hard to
    # see on screen. Same tensor either way — only how far out it is drawn.
    adp_probability: float = 0.99
    adp_opacity: float = 0.85
    adp_temperature_index: int = 0   # which reported temperature to draw

    # atom labels (element symbols drawn at each atom, capped for large cells)
    show_atom_labels: bool = False
    atom_label_size: int = 16  # point size of the element-symbol labels

    # coordination polyhedra (VESTA-style), off until asked for: they hide the
    # atoms they enclose, and a first look at a structure is usually the atoms
    show_polyhedra: bool = False
    polyhedra_opacity: float = 0.3  # translucent enough to see the atoms inside
    polyhedra_min_vertices: int = 4  # only draw around atoms with >= this many bonds

    # geometry-measurement overlays (Geometry panel): dot markers, distance/angle
    # paths, and least-squares plane patches — each independently coloured.
    measure_point_color: str = "#ff7f0e"  # dot markers
    measure_line_color: str = "#ff7f0e"   # distance / angle / dihedral paths
    measure_plane_color: str = "#1f77b4"  # least-squares plane patches

    # point-symmetry elements (Point symmetry panel), drawn with the same three
    # shapes as the measurements but in their own colours, so an axis is never
    # mistaken for a measured distance. Deliberately unlike the palette above.
    symmetry_axis_color: str = "#9467bd"   # rotation and rotoinversion axes
    symmetry_plane_color: str = "#17becf"  # mirror planes
    symmetry_point_color: str = "#e377c2"  # the centre of inversion

    # scene / camera
    background_color: str = "white"
    parallel_projection: bool = True    # orthographic by default; perspective when False
    show_orientation_axes: bool = False  # the little xyz marker in the corner


__all__ = ["RenderSettings"]
