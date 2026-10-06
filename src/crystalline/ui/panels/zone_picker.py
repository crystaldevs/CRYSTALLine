"""Build a band path by clicking points on the first Brillouin zone.

A band path is a dozen lines of integers over a shrinking factor, and the
question it answers — *where in the zone does this go* — is a shape. Typing
``0 0 0  4 0 4`` is a poor way to ask it, and reading one back is worse.

Its own dialog rather than an overlay on the crystal, deliberately: real space
and reciprocal space in one picture is confusing to look at, and the two would
fight over the camera. The zone gets its own small viewport with its own view.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from functools import lru_cache

import numpy as np
from PySide6.QtCore import QEvent
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from crystalline.core.brillouin import (
    CONVENTIONAL,
    PRIMITIVE,
    brillouin_zone,
    display_label,
    is_linear,
    is_planar,
    reciprocal_cell,
    special_points,
    zone_lattice,
)
from crystalline.core.structure import Structure
from crystalline.ui import menus, theme, wheel_zoom
from crystalline.ui.safety import guard
from vtkmodules.vtkCommonCore import VTK_FONT_FILE

from crystalline.viz.fonts import unicode_font

# Special points are halves, thirds, quarters, sixths and eighths, and every
# one of those has a glyph of its own — which reads as the number it is rather
# than as two numbers with a slash between them.
_VULGAR = {
    (1, 2): "½", (1, 3): "⅓", (2, 3): "⅔", (1, 4): "¼", (3, 4): "¾",
    (1, 5): "⅕", (2, 5): "⅖", (3, 5): "⅗", (4, 5): "⅘",
    (1, 6): "⅙", (5, 6): "⅚", (1, 8): "⅛", (3, 8): "⅜", (5, 8): "⅝", (7, 8): "⅞",
}

# How big a special point is drawn, as a fraction of the zone's own extent, so
# the marker is the same visual size whatever the lattice parameters are.
_POINT_RADIUS = 0.024
_PATH_WIDTH = 5

# A subscript, where it has to be drawn as its own actor: how much smaller than
# the base, how far below it (a fraction of the base's font size, scaled by the
# screen's DPI like the glyphs themselves, so it looks the same on a HiDPI
# display and an ordinary one), and a floor so it stays legible.
_SUBSCRIPT_SCALE = 0.7
_SUBSCRIPT_DROP = 0.15
_SUBSCRIPT_GAP = 0.08
_SUBSCRIPT_MIN_SIZE = 8

# The digits Unicode does have, which need no actor of their own.
_SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

# A polymer's zone is drawn as a bar rather than a hairline, and its one axis
# is given room to emerge past the zone edge instead of ending on the point
# marked there — both only apply to the segment case.
_ZONE_BAR_WIDTH = 6
_CHAIN_AXIS_REACH = 1.6

# A label sits just clear of its own marker, as a multiple of the marker's
# radius — both are world lengths, so the gap on screen is the same at every
# zoom. The numbers that used to ride along underneath now go in the corner
# legend, which is what lets the labels sit this close without colliding.
_LABEL_OFFSET = 2.4

# One colour per leg of the path, reused around when a path is longer than the
# list. Chosen to stay apart on a pale ground and in the legend beside them —
# and to avoid the axis red, green and blue, which now belong to k_x, k_y and
# k_z and would otherwise say "axis" in the middle of the zone.
# The same step the structure window's rotate chips use.
_ROTATE_STEP_DEG = 15.0

# Where the corner readout sits, in viewport fractions. Inset from the edge:
# text hard against the frame reads as though it has been cropped.
_LEGEND_X = 0.955
_LEGEND_TOP = 0.915

# How far the mouse may travel between press and release and still count as a
# click rather than a drag of the camera, in pixels.
_CLICK_SLOP = 3

_SEGMENT_COLOURS = [
    "#e8710a", "#a142f4", "#00897b", "#d81b60",
    "#6d4c41", "#5b6bd6", "#8d6e00", "#00838f",
]

# Dash and gap for the guide lines, as fractions of the zone's extent. VTK's
# OpenGL2 backend dropped line stippling, so a dashed line is built out of real
# short segments; these are the pieces.
_DASH = 0.045
_GAP = 0.035

# Labels pymatgen and ASE spell out, which MathText sets as the letter itself.
_GREEK = {"G": r"\Gamma", "Gamma": r"\Gamma", "Sigma": r"\Sigma",
          "Delta": r"\Delta", "Lambda": r"\Lambda"}
# The same letters for the plain-text path, where VTK draws the glyph itself.
_PLAIN_GREEK = {"Gamma": "\u0393", "Sigma": "\u03a3", "Delta": "\u0394",
                "Lambda": "\u039b"}

# The markers, and the one the side panel is pointing at.
_POINT_COLOUR = "#d6453c"
_CURRENT_COLOUR = "#f5a623"


class ZonePickerDialog(QDialog):
    """Click special points in order; the path is the chain between them."""

    def __init__(self, structure: Structure, parent: Optional[QWidget] = None,
                 picking: bool = True) -> None:
        super().__init__(parent)
        self.setWindowTitle("Path builder" if picking else "Brillouin zone")
        # Wider than it was: the side column now carries the coordinates as
        # well as the path, and both want to be readable at once.
        self.resize(1000, 620)
        self._picking = picking
        self._selected: Optional[str] = None
        self._structure = structure
        self._setting = PRIMITIVE
        self._picked: List[str] = []
        self._lattice = structure
        self._points: dict = {}
        self._reciprocal = reciprocal_cell(structure)
        self._actors: dict = {}
        self._path_actors: list = []
        self._legend_names: list = []
        self._flat_normal = None
        self._chain = None     # unit vector along a polymer's zone, else None
        self._extent = 1.0

        self._resolve_lattice()

        outer = QVBoxLayout(self)
        if not self._points:
            message = ("This lattice could not be classified, so it has no labelled "
                       "points — the zone is drawn, but the path has to be typed.")
        elif picking:
            message = ("Click the labelled points in the order the path should visit "
                       "them. The zone is the Wigner–Seitz cell of the reciprocal "
                       "lattice.")
        else:
            message = ("The first Brillouin zone — the Wigner–Seitz cell of the "
                       "reciprocal lattice — with this lattice's high-symmetry "
                       "points. Drag to rotate, scroll to zoom.")
        note = QLabel(message)
        note.setWordWrap(True)
        note.setStyleSheet("color: palette(mid);")
        outer.addWidget(note)

        # Which cell's zone. Two different pictures of the same crystal, and
        # picking one of them wrong is the classic way to draw a Brillouin zone
        # that looks plausible and is not the crystal's.
        chooser = QHBoxLayout()
        chooser.addWidget(QLabel("Cell"))
        self._setting_box = QComboBox()
        self._setting_box.addItem("Primitive (the Bravais lattice's own zone)", PRIMITIVE)
        self._setting_box.addItem("Conventional (the crystallographic cell)", CONVENTIONAL)
        self._setting_box.currentIndexChanged.connect(self._on_setting_changed)
        chooser.addWidget(self._setting_box, 1)
        if picking:
            # CRYSTAL reads a BAND path in the primitive reciprocal basis, so a
            # path picked on the conventional zone would be the wrong numbers.
            self._setting_box.setEnabled(False)
            self._setting_box.setToolTip(
                "A band path is written in the primitive reciprocal basis, which "
                "is what CRYSTAL reads — so a path is always picked on the "
                "primitive zone. Cell ▸ Brillouin zone can show either."
            )
        elif not _is_crystal(self._structure):
            # A slab's symmetry is a layer group and a chain's a rod group;
            # neither has a 3D Bravais lattice to offer a second cell of, and
            # zone_lattice hands both back untouched. The box would be a choice
            # between a picture and itself.
            self._setting_box.setEnabled(False)
            self._setting_box.setToolTip(
                "Only a 3D crystal has two cells to choose between. A slab or a "
                "polymer is drawn in the cell it was computed in."
            )
        chooser.addStretch(1)
        outer.addLayout(chooser)

        # The same controls as the structure window's view toolbar, in the same
        # shapes: solid axis chips to look down an axis, ghost chips to orbit by
        # a step. A second 3D view that invented its own vocabulary would make
        # the user learn the app twice.
        outer.addWidget(self._view_toolbar())

        body = QHBoxLayout()
        self._view = _ZoneView(self)
        body.addWidget(self._view, 1)
        self._list = QListWidget()
        self._total = QLabel()
        self._total.setStyleSheet("color: palette(mid);")
        if picking:
            side = QVBoxLayout()
            side.addWidget(QLabel("Path"))
            self._list.setMaximumWidth(240)
            side.addWidget(self._list, 1)
            undo = QPushButton("Remove last")
            undo.clicked.connect(self._undo)
            clear = QPushButton("Clear")
            clear.clicked.connect(self._clear)
            side.addWidget(undo)
            side.addWidget(clear)
            side.addWidget(self._total)
            body.addLayout(side)
        outer.addLayout(body)
        # Opened from the menu there is nothing to accept: it is a picture, and
        # the only thing to do with it is close it.
        standard = (QDialogButtonBox.Ok | QDialogButtonBox.Cancel if picking
                    else QDialogButtonBox.Close)
        buttons = QDialogButtonBox(standard, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        self._draw()
        self._watch_for_empty_clicks()

    def _draw_and_sync(self) -> None:
        """Redraw for a change of options, leaving the camera where it was."""
        self._draw(keep_camera=self._view.plotter.camera_position)
        self._sync()

    @guard()
    def _look_along(self, direction) -> None:
        """Point the camera down a reciprocal axis, or back to the default.

        Straight down an axis is how a zone is drawn in a paper — the faces
        perpendicular to it show their true shape — and getting there by
        dragging is fiddly and never quite square.
        """
        plotter = self._view.plotter
        if direction is None:
            plotter.view_isometric()
        else:
            eye = np.asarray(direction, dtype=float)
            # Any up vector will do so long as it is not the direction itself.
            up = (0, 0, 1) if abs(eye[2]) < 0.9 else (0, 1, 0)
            plotter.camera_position = [tuple(eye * self._extent * 4),
                                       (0.0, 0.0, 0.0), up]
        self._centre_on_gamma()
        plotter.render()

    @guard()
    def _save_image(self) -> None:
        """Export the view — the same dialog and formats as the structure window."""
        from crystalline.ui.image_export import export_view

        export_view(self, self._view.export_image, "brillouin_zone")

    # ── the view toolbar, mirroring the structure window's ──────────────
    def _view_toolbar(self) -> QWidget:
        bar = QToolBar(self)

        caption = QLabel("VIEW")
        caption.setContentsMargins(6, 0, 6, 0)
        bar.addWidget(caption)
        # Coloured like the structure window's a/b/c chips and like the axes
        # drawn in the view, so a chip and its arrow are plainly the same axis.
        for axis, letter, direction in (("a", "x", (1, 0, 0)),
                                        ("b", "y", (0, 1, 0)),
                                        ("c", "z", (0, 0, 1))):
            button = QToolButton(bar)
            button.setIcon(menus.axis_icon("k", letter, "#ffffff", self.font()))
            button.setToolTip(f"Look down k{letter}")
            menus.style_chip(button, "axis", axis)
            button.clicked.connect(
                lambda _checked=False, d=direction: self._look_along(d))
            bar.addWidget(button)

        self._reset_button = QToolButton(bar)
        self._reset_button.setToolTip("Fit the whole zone, from the default view")
        self._reset_button.setIcon(menus.reset_view_icon())
        menus.style_chip(self._reset_button, "ghost")
        self._reset_button.clicked.connect(
            lambda _checked=False: self._look_along(None))
        bar.addWidget(self._reset_button)

        bar.addWidget(_spacer(10))
        rotate = QLabel("ROTATE")
        rotate.setContentsMargins(2, 0, 6, 0)
        bar.addWidget(rotate)
        for label, tip, azimuth, elevation, roll in menus.ROTATE_CHIPS:
            button = QToolButton(bar)
            button.setText(label)
            button.setToolTip(tip)
            button.setAutoRepeat(True)          # hold to keep turning
            menus.style_chip(button, "ghost")
            # The chip's signs are what the scene does; camera_angles turns
            # them into what the camera has to do, and the step box says how far.
            button.clicked.connect(
                lambda _checked=False, a=azimuth, e=elevation, r=roll:
                self._rotate_view(
                    *menus.camera_angles(a, e, r, self._step.value())))
            bar.addWidget(button)
        self._step = menus.rotate_step_box(bar)
        bar.addWidget(self._step)

        bar.addWidget(_spacer(10))
        self._guides = QCheckBox("Symmetry lines")
        self._guides.setToolTip(
            "The lines Γ→X, Γ→L, Γ→K … that a zone diagram labels Δ, Λ and Σ. "
            "Dashed, because they run through the inside of the zone."
        )
        self._guides.setChecked(True)
        self._guides.toggled.connect(lambda _on: self._draw_and_sync())
        if is_linear(self._structure):
            # Every line from Γ on a segment runs along the segment. A tick box
            # that cannot change the picture is worse than one that is not there.
            self._guides.setEnabled(False)
            self._guides.setToolTip(
                "A polymer's zone is a line, and the symmetry line Γ→X is that "
                "line — there is nothing to draw inside it."
            )
        bar.addWidget(self._guides)

        stretch = QWidget(bar)
        stretch.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bar.addWidget(stretch)
        save = QToolButton(bar)
        save.setText("Export image…")
        save.setToolTip("Save the view — the same formats, resolution and "
                        "transparency as the structure window's export.")
        save.setProperty("chip", "ghost")
        save.clicked.connect(self._save_image)
        bar.addWidget(save)
        return bar

    @guard()
    def _rotate_view(self, azimuth: float = 0.0, elevation: float = 0.0,
                     roll: float = 0.0) -> None:
        """Orbit by a step, exactly as the structure viewport's chips do."""
        camera = self._view.plotter.renderer.GetActiveCamera()
        if azimuth:
            camera.Azimuth(azimuth)
        if elevation:
            camera.Elevation(elevation)
            # Elevation alone skews (and at the poles flips) the up vector.
            camera.OrthogonalizeViewUp()
        if roll:
            # vtkCamera.Roll turns the *up vector*, so the scene appears to go
            # the other way: negate to make a positive roll read as clockwise.
            camera.Roll(-roll)
        self._view.plotter.renderer.ResetCameraClippingRange()
        self._view.plotter.render()

    def _resolve_lattice(self) -> None:
        """Fix the cell once; the zone and its labels then cannot disagree."""
        self._lattice = zone_lattice(self._structure, self._setting)
        self._points = special_points(self._lattice)
        self._reciprocal = reciprocal_cell(self._lattice)

    @guard()
    def _on_setting_changed(self, _index: int = 0) -> None:
        """``currentIndexChanged`` carries the index — take it, or the slot
        raises TypeError into the safety net and the switch does nothing."""
        self._setting = self._setting_box.currentData()
        self._resolve_lattice()
        self._picked = []
        self._draw()
        self._sync()

    # ── the picture ─────────────────────────────────────────────────────
    def _draw(self, keep_camera=None) -> None:
        """Rebuild the scene. ``keep_camera`` restores a saved camera instead of
        refitting, so selecting a point does not also swing the view."""
        plotter = self._view.plotter
        plotter.clear()
        self._actors = {}
        self._path_actors = []
        try:
            vertices, faces = brillouin_zone(self._lattice)
        except Exception as exc:  # noqa: BLE001 - a lattice we cannot build
            plotter.add_text(f"No Brillouin zone: {exc}", font_size=9)
            return
        self._extent = float(np.linalg.norm(vertices, axis=1).max()) or 1.0
        self._flat_normal = _plane_normal(vertices) if len(faces) == 1 else None
        # A polymer's zone is a segment: the one direction it disperses in.
        # Kept because almost everything below is drawn differently for it —
        # a line has no inside to shade, no faces to give it shape, and two
        # directions around it that mean nothing.
        self._chain = _segment_direction(vertices)

        import pyvista as pv

        for face in faces:
            loop = list(face) + [face[0]]
            # The zone is the subject. On a solid or a polygon the edges are
            # the outline of something already shaded; a segment is the whole
            # of it, so it is drawn as a bar rather than as a hairline that
            # the axes out of Γ would then outweigh.
            plotter.add_mesh(pv.lines_from_points(vertices[loop]),
                             color="#7f8c9b" if self._chain is None else "#5b8dee",
                             line_width=2 if self._chain is None else _ZONE_BAR_WIDTH,
                             pickable=False)
        # A translucent body gives the zone its shape; the wireframe alone
        # reads as a tangle from most angles. A slab's zone is a *polygon* —
        # flat — and asking a 3D triangulation for the solid inside it returns
        # nothing at all, so the polygon is filled directly.
        if len(faces) == 1 and len(vertices) >= 3:
            polygon = pv.PolyData(vertices, faces=[len(vertices), *range(len(vertices))])
            plotter.add_mesh(polygon, color="#5b8dee", opacity=0.18,
                             show_edges=False, pickable=False)
        elif len(vertices) >= 4:
            hull = pv.PolyData(vertices).delaunay_3d().extract_surface()
            plotter.add_mesh(hull, color="#5b8dee", opacity=0.10,
                             show_edges=False, pickable=False)

        radius = _POINT_RADIUS * self._extent
        for label, fractional in self._points.items():
            centre = np.asarray(fractional) @ self._reciprocal
            sphere = pv.Sphere(radius=radius, center=centre)
            actor = plotter.add_mesh(
                sphere, color=_CURRENT_COLOUR if label == self._selected
                else _POINT_COLOUR, pickable=True)
            self._actors[label] = (actor, centre)
            # The name, and under it the numbers it stands for. One label with
            # a newline rather than two labels: the second line has to sit
            # under the first on *screen*, and two world-anchored labels would
            # only line up from one direction.
            anchor = centre + self._label_direction(centre) * radius * _LABEL_OFFSET
            _draw_label(plotter, anchor, label, self._label_colour(), 15)
        # On a segment every "symmetry line" from Γ runs along the zone itself,
        # so the dashes would be drawn on top of the bar and say nothing.
        if self._guides.isChecked() and self._chain is None:
            self._draw_guides(plotter)
        self._draw_axes(plotter)
        # pyvista refuses to enable a picker that is already enabled, so the
        # old one comes down first. A redraw is now rare — options and the
        # cell setting, not every click — so this costs nothing.
        plotter.disable_picking()
        plotter.enable_mesh_picking(callback=self._on_pick, show=False,
                                    show_message=False, left_clicking=True)
        if keep_camera is None:
            if self._chain is not None:
                self._view_across_the_chain()
            elif self._flat_normal is not None:
                self._view_face_on()
            self._centre_on_gamma()
        else:
            plotter.camera_position = keep_camera
            plotter.renderer.ResetCameraClippingRange()
        # Placed — say so. ``reset_camera`` leaves pyvista believing nobody has
        # set the camera, and then the next thing added with the default
        # arguments (a leg of the path, on every click) refits it: zoom and pan
        # thrown away, the view recentred on the bounding box instead of Γ.
        plotter.camera_set = True
        plotter.render()

    def _centre_on_gamma(self) -> None:
        """Fit the view, then put Γ back in the middle of it.

        ``reset_camera`` centres on the bounding box of everything drawn, and
        the axes only run in +kx, +ky and +kz — so the box is not centred on
        the zone and the zone drifts off-centre, taking a "look down kx" view
        off-axis with it. The zone is the subject; Γ is its centre.
        """
        plotter = self._view.plotter
        plotter.reset_camera()
        camera = plotter.renderer.GetActiveCamera()
        offset = np.asarray(camera.GetFocalPoint(), dtype=float)
        camera.SetFocalPoint(0.0, 0.0, 0.0)
        camera.SetPosition(*(np.asarray(camera.GetPosition(), dtype=float) - offset))
        plotter.renderer.ResetCameraClippingRange()

    def _label_direction(self, centre: np.ndarray) -> np.ndarray:
        """Which way a point's label is nudged, clear of what it names.

        Outward from Γ, with the two cases where outward is no use. On a
        segment it is along the bar, where the axis arrow and the next label
        already are. And Γ, which has no outward of its own, is given one
        pointing at the camera — invisible on a flat zone, which is now looked
        at square on, so there its label goes up the plane instead.
        """
        if self._chain is not None:
            return _perpendicular_to(self._chain)
        direction = _outward(centre)
        if self._flat_normal is not None and abs(
                float(np.dot(direction, self._flat_normal))) > 0.9:
            # Up and to the left: the two axes drawn out of Γ are the positive
            # halves of the in-plane ones, so straight up lands on one of them.
            up = _upright_in(self._flat_normal)
            away = up - np.cross(up, self._flat_normal)
            return away / float(np.linalg.norm(away))
        return direction

    def _view_across_the_chain(self) -> None:
        """Look at a segment side-on, with the chain across the view.

        The default isometric view is for a solid. A line seen from a corner is
        a foreshortened line — and seen down its own axis, a dot.
        """
        plotter = self._view.plotter
        up = _perpendicular_to(self._chain)
        eye = np.cross(self._chain, up)
        plotter.camera_position = [tuple(eye * self._extent * 4),
                                   (0.0, 0.0, 0.0), tuple(up)]

    def _view_face_on(self) -> None:
        """Look square at a flat zone, the way every zone diagram prints one.

        A slab's zone is a polygon, and the default isometric view shows it as
        a sheared version of itself — the angles between Γ–M and Γ–K are not the
        angles on screen, which is the one thing the picture is for. Seen down
        its normal, the polygon has its true shape.
        """
        plotter = self._view.plotter
        normal = _facing(self._flat_normal)
        plotter.camera_position = [tuple(normal * self._extent * 4),
                                   (0.0, 0.0, 0.0), tuple(_upright_in(normal))]

    def _draw_guides(self, plotter) -> None:
        """Γ to every special point, dashed.

        These are the symmetry lines a zone diagram names Δ, Λ and Σ. They run
        through the inside of the zone, and a dashed line is how a drawing says
        "this is behind the surface you are looking at" — the same convention
        the Bilbao diagrams use for the hidden parts.
        """
        for _label, (_actor, centre) in self._actors.items():
            if float(np.linalg.norm(centre)) < 1e-9:
                continue   # Γ itself: there is no line from a point to itself
            dashes = _dashed_line(np.zeros(3), centre, self._extent)
            if dashes is not None:
                plotter.add_mesh(dashes, color="#8a94a6", line_width=2,
                                 pickable=False)

    def _draw_axes(self, plotter) -> None:
        """The reciprocal axes out of Γ, labelled as a zone diagram labels them."""
        reach = self._extent * 1.35
        # Each axis in its chip's colour, so the button and the arrow it aims
        # down are plainly the same thing — the structure window's a/b/c chips
        # match its lattice gizmo the same way.
        # Through _draw_label like every other label here, which is also what
        # gives them their subscripts. Spelt out as MathText literals, these
        # stayed dollar-signed on a machine whose MathText does not work, long
        # after the k-points had stopped being.
        for index, (direction, name, colour) in enumerate((
            (np.array([1.0, 0, 0]), "k_x", theme.AXIS_COLOURS[0]),
            (np.array([0, 1.0, 0]), "k_y", theme.AXIS_COLOURS[1]),
            (np.array([0, 0, 1.0]), "k_z", theme.AXIS_COLOURS[2]),
        )):
            # A slab disperses in its plane and nowhere else. Drawing the axis
            # across it would promise a direction the bands do not have.
            if self._flat_normal is not None and abs(
                    float(np.dot(direction, self._flat_normal))) > 0.9:
                continue
            # The same rule one dimension down: a polymer disperses along the
            # chain alone, so an axis square to it is dropped. The one left runs
            # *along* the zone, which is the truth about a polymer — the bar is
            # the axis — so it is drawn on from the zone's edge outwards rather
            # than along the bar, which would paint half of it the axis colour.
            start = np.zeros(3)
            if self._chain is not None:
                if abs(float(np.dot(direction, self._chain))) < 0.1:
                    continue
                start = direction * self._extent
                reach = self._extent * _CHAIN_AXIS_REACH
            import pyvista as pv

            tip = direction * reach
            plotter.add_mesh(pv.lines_from_points(np.array([start, tip])),
                             color=colour, line_width=2, pickable=False)
            # A head, so the line reads as an axis rather than as another edge.
            head = self._extent * 0.06
            plotter.add_mesh(pv.Cone(center=tip - direction * head * 0.5,
                                     direction=direction, height=head,
                                     radius=head * 0.32, resolution=16),
                             color=colour, pickable=False)
            _draw_label(plotter, tip * 1.04, name, colour, 14)

    def _label_colour(self) -> str:
        """Text that reads against both the marker and the viewport's ground.

        Red-on-red was the old trouble: the labels were the markers' own colour,
        so wherever the two met the text vanished. The theme's ordinary text
        colour contrasts with the background by construction and with the red
        markers by being nothing like them.
        """
        from PySide6.QtWidgets import QApplication
        from crystalline.ui import theme

        return theme.active_palette(QApplication.instance()).text

    # ── clicking nothing ────────────────────────────────────────────────
    def _watch_for_empty_clicks(self) -> None:
        """A click on nothing clears the selection.

        pyvista's mesh picking only calls back on a *hit*, so a click that
        misses is silent — and the last point stayed marked and read out in the
        corner with nothing to say it was stale. These two observers catch the
        miss. They are installed once: a redraw re-arms pyvista's picker, not
        ours.
        """
        interactor = self._view.plotter.iren
        interactor.add_observer("LeftButtonPressEvent", self._on_press)
        interactor.add_observer("LeftButtonReleaseEvent", self._on_release)

    @guard()
    def _on_press(self, *_args) -> None:
        self._press_at = self._view.plotter.iren.interactor.GetEventPosition()

    @guard()
    def _on_release(self, *_args) -> None:
        """Clear the selection if this was a click on empty space.

        A press and release far apart is a drag of the camera, which must not
        count — rotating the zone by starting on the background is the most
        ordinary thing anyone does here.
        """
        import vtk

        x, y = self._view.plotter.iren.interactor.GetEventPosition()
        start = getattr(self, "_press_at", None)
        if start is None or abs(x - start[0]) > _CLICK_SLOP or abs(y - start[1]) > _CLICK_SLOP:
            return
        picker = vtk.vtkPropPicker()
        picker.PickProp(x, y, self._view.plotter.renderer)
        hit = picker.GetViewProp()
        if any(hit is actor for actor, _centre in self._actors.values()):
            return                                   # a marker: _on_pick has it
        if self._selected is None:
            return
        self._selected = None
        self._recolour_markers()
        self._legend_selection()
        self._view.plotter.render()

    @guard()
    def _on_pick(self, mesh) -> None:
        """Whichever labelled point the clicked mesh is nearest."""
        if mesh is None or mesh.n_points == 0:
            return
        clicked = np.asarray(mesh.center, dtype=float)
        best, best_distance = None, np.inf
        for label, (_actor, centre) in self._actors.items():
            distance = float(np.linalg.norm(centre - clicked))
            if distance < best_distance:
                best, best_distance = label, distance
        if best is None or best_distance > _POINT_RADIUS * self._extent * 2:
            return
        self._selected = best
        if self._picking:
            self._picked.append(best)
        # Only what changed: the marker colours and the legend. Rebuilding the
        # whole scene here re-made fourteen face actors, the hull, every label
        # and the picker on *every click*, which is what made the view slower
        # the more it was used.
        self._recolour_markers()
        self._sync(append_only=True)

    def _append_leg(self) -> None:
        """Draw just the leg the last click added, and its legend line."""
        import pyvista as pv

        index = len(self._picked) - 2
        if index < 0:
            return
        start, end = self._picked[index], self._picked[index + 1]
        if start not in self._actors or end not in self._actors:
            return
        ends = np.array([self._actors[start][1], self._actors[end][1]])
        self._path_actors.append(
            self._view.plotter.add_mesh(pv.lines_from_points(ends),
                                        color=segment_colour(index),
                                        line_width=_PATH_WIDTH, pickable=False,
                                        reset_camera=False)
        )

    def _undo(self) -> None:
        if self._picked:
            self._picked.pop()
            self._selected = None
            self._recolour_markers()
            self._sync()

    def _clear(self) -> None:
        self._picked = []
        self._selected = None
        self._recolour_markers()
        self._sync()

    def _sync(self, append_only: bool = False) -> None:
        """Refresh the path list, and the picture of the path.

        ``append_only`` says the path grew by one leg at the end, which is the
        common case and the one that has to stay cheap.
        """
        self._list.clear()
        for index, (start, end) in enumerate(zip(self._picked, self._picked[1:])):
            item = QListWidgetItem(
                f"{display_label(start)}  →  {display_label(end)}"
                f"     {self._segment_length(start, end):.3f} Å⁻¹")
            # The same colour as the leg it names, so the list and the picture
            # can be read against each other without counting positions.
            item.setForeground(QColor(segment_colour(index)))
            self._list.addItem(item)
        if len(self._picked) == 1:
            self._list.addItem(f"{display_label(self._picked[0])}  → …")
        total = self._path_length()
        self._total.setText(f"total  {total:.3f} Å⁻¹" if total else "")
        if append_only:
            self._append_leg()
            self._legend_selection()
            self._view.plotter.render()
        else:
            self._draw_path()

    def _recolour_markers(self) -> None:
        """Mark the selected point, without touching anything else."""
        for name, (actor, _centre) in self._actors.items():
            actor.prop.color = (_CURRENT_COLOUR if name == self._selected
                                else _POINT_COLOUR)

    def _segment_length(self, start: str, end: str) -> float:
        """|Δk| along one leg, in Å⁻¹.

        A band plot's horizontal axis is this distance, so it is what decides
        how the segments share out the points you ask for — a leg twice as long
        gets half the sampling density for the same NSUB.
        """
        here, there = self._points.get(start), self._points.get(end)
        if here is None or there is None:
            return 0.0
        return float(np.linalg.norm((np.asarray(there) - np.asarray(here))
                                    @ self._reciprocal))

    def _path_length(self) -> float:
        return sum(self._segment_length(a, b)
                   for a, b in zip(self._picked, self._picked[1:]))

    def _draw_path(self) -> None:
        import pyvista as pv

        plotter = self._view.plotter
        for actor in list(self._path_actors):
            plotter.remove_actor(actor, render=False)
        self._path_actors = []
        # A leg at a time, each in its own colour, so the picture and the
        # legend beside it name the same thing twice.
        for index, (start, end) in enumerate(zip(self._picked, self._picked[1:])):
            if start not in self._actors or end not in self._actors:
                continue
            ends = np.array([self._actors[start][1], self._actors[end][1]])
            self._path_actors.append(
                plotter.add_mesh(pv.lines_from_points(ends),
                                 color=segment_colour(index),
                                 line_width=_PATH_WIDTH, pickable=False,
                                 reset_camera=False)
            )
        self._draw_legend()
        plotter.render()

    # ── the corner legend ───────────────────────────────────────────────
    #
    # In the corner rather than under each marker: the points of a cubic
    # lattice all lie within about thirty degrees of each other, so coordinates
    # drawn at the points overlap into a heap however they are placed. A corner
    # reads as a caption and has as much room as it needs.
    #
    # Every line is added under a name, and a name replaces its previous actor
    # in place — so appending a leg costs one text actor, not a rebuild. That
    # matters: rebuilding the whole legend and the whole path on every click
    # made each click cost more than the last, and by two dozen clicks a
    # selection took half a second.
    def _legend_line(self, slot: int, text: str, colour: str) -> str:
        plotter = self._view.plotter
        name = f"zone-legend-{slot}"
        plotter.add_text(text, position=(_LEGEND_X, _LEGEND_TOP - slot * 0.045),
                         viewport=True, font_size=12, color=colour,
                         font_file=unicode_font(), name=name, render=False)
        actor = plotter.renderer.actors.get(name)
        if actor is not None:                     # right-align on the corner
            actor.GetTextProperty().SetJustificationToRight()
        if name not in self._legend_names:
            self._legend_names.append(name)
        return name

    def _legend_selection(self) -> None:
        """The corner reads out the clicked point, and only that.

        It used to list the legs of the path as well — which the Path list
        beside the view already does, in the same colours. Saying it twice made
        the corner grow with the path and told the reader nothing new.
        """
        fractional = (self._points.get(self._selected)
                      if self._selected is not None else None)
        if fractional is None:
            self._view.plotter.remove_actor("zone-legend-0", render=False)
            if "zone-legend-0" in self._legend_names:
                self._legend_names.remove("zone-legend-0")
            return
        self._legend_line(
            0,
            f"{display_label(self._selected)}    "
            + "  ".join(_tidy_number(v) for v in fractional),
            _CURRENT_COLOUR,
        )

    def _clear_legend(self) -> None:
        for name in self._legend_names:
            self._view.plotter.remove_actor(name, render=False)
        self._legend_names = []

    def _draw_legend(self) -> None:
        """Rebuild the legend — one line, for whichever point is selected."""
        self._clear_legend()
        self._legend_selection()

    # ── the result ──────────────────────────────────────────────────────
    def path(self) -> List[Tuple[Tuple[str, str], tuple]]:
        """The picked chain as the band editor's ``(labels, segment)`` rows."""
        rows = []
        for start, end in zip(self._picked, self._picked[1:]):
            rows.append((
                (start, end),
                (tuple(self._points[start]), tuple(self._points[end])),
            ))
        return rows

    @classmethod
    def pick(cls, structure: Structure, parent=None):
        """Show the picker; return the path, or ``None`` if it was cancelled."""
        dialog = cls(structure, parent, picking=True)
        if dialog.exec() != QDialog.Accepted:
            return None
        return dialog.path() or None

    @classmethod
    def visualise(cls, structure: Structure, parent=None) -> None:
        """Show the zone on its own, with nothing to pick and nothing to return."""
        cls(structure, parent, picking=False).exec()



def _math_label(label: str) -> str:
    """A k-point label as VTK will actually draw it.

    Plain Unicode in a font that has Greek, never MathText. VTK can rasterise
    ``$\\Gamma$`` through matplotlib, and where that works it sets the letter
    italic as a band diagram does — but where it does not, the label is drawn
    verbatim, dollar signs and backslash and all, and no probe from this side
    predicts which machine is which. A label that is always right is worth more
    than one that is occasionally prettier: the subscript of ``k_x`` is dropped
    and the letter stands upright, and it renders everywhere.
    """
    return _plain_label(label)


def _plain_label(label: str) -> str:
    """The same label without MathText, as one string: ``Σ_1`` -> ``Σ₁``.

    Drawn in :func:`~crystalline.viz.fonts.unicode_font`, which has the Greek
    that VTK's own font lacks. What :func:`_label_parts` could not fold into a
    single string is simply run together here; the scene draws the two pieces
    separately (see :func:`_draw_label`).
    """
    return "".join(_label_parts(label))


def _label_parts(label: str) -> Tuple[str, str]:
    """``(base, subscript)``: a k-label split where it has to be drawn in two.

    A digit subscript is folded into the base, because Unicode has ₀–₉ and the
    label font draws them: ``Σ_1`` is one string, ``Σ₁``. A *letter* subscript
    has no such glyph — there is a ₓ and no ᵧ or ᵶ — so it comes back on its
    own, to be drawn smaller and lower beside the base. That is k_x, k_y and
    k_z, which are the axes of the zone and were reading as ``kx``.
    """
    name = str(display_label(label))
    for plain, letter in _PLAIN_GREEK.items():
        if name.startswith(plain):
            name = letter + name[len(plain):]
            break
    base, _, subscript = name.partition("_")
    if subscript.isdigit():
        return base + subscript.translate(_SUBSCRIPT_DIGITS), ""
    return base, subscript


def _tidy_number(value: float) -> str:
    """A coordinate as the simple fraction it almost always is.

    Special points are halves, thirds, quarters and eighths; ``0.333`` reads as
    an approximation of something, while ``1/3`` reads as the thing itself.
    """
    from fractions import Fraction

    ratio = Fraction(float(value)).limit_denominator(24)
    if abs(float(ratio) - float(value)) > 1e-6:
        return f"{value:.4g}"
    if ratio.denominator == 1:
        return str(ratio.numerator)
    sign = "-" if ratio.numerator < 0 else ""
    glyph = _VULGAR.get((abs(ratio.numerator), ratio.denominator))
    if glyph:
        return sign + glyph
    return f"{ratio.numerator}/{ratio.denominator}"


def _dashed_line(start, end, extent: float):
    """A line as a run of short segments.

    VTK's OpenGL2 backend dropped line stippling, so a dashed line has to be
    built out of real pieces. The dash and gap scale with the zone so the
    pattern looks the same whatever the lattice parameters are.
    """
    import pyvista as pv

    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    length = float(np.linalg.norm(end - start))
    dash, gap = _DASH * extent, _GAP * extent
    if length < 1e-9 or dash < 1e-12:
        return None
    direction = (end - start) / length
    points, lines, offset = [], [], 0.0
    while offset < length:
        head = start + direction * offset
        tail = start + direction * min(offset + dash, length)
        index = len(points)
        points += [head, tail]
        lines += [2, index, index + 1]
        offset += dash + gap
    if not points:
        return None
    return pv.PolyData(np.asarray(points), lines=np.asarray(lines))


def _draw_label(plotter, position, label: str, colour: str, size: int):
    """Draw ``label`` at ``position``, with a letter subscript as a real one.

    Two actors at the same point rather than one string: VTK sets a text actor
    in a single size, and neither it nor the font has any notion of a subscript.
    The second is placed with ``SetDisplayOffset``, which is in *pixels applied
    after projection* — so the subscript keeps its place beside the base at
    every zoom and from every angle, which a second world-anchored label would
    not (both actors are drawn at a constant size on screen, so a world-space
    gap between them would open and close as the view moved).
    """
    base, subscript = _label_parts(label)
    actor = _billboard(plotter, position, base, colour, size)
    if subscript:
        dpi = _dpi_of(plotter)
        small = max(_SUBSCRIPT_MIN_SIZE, int(round(size * _SUBSCRIPT_SCALE)))
        _billboard(plotter, position, subscript, colour, small,
                   offset=(_text_width(base, size, dpi)
                           + int(round(size * _SUBSCRIPT_GAP * dpi / 72.0)),
                           -int(round(size * _SUBSCRIPT_DROP * dpi / 72.0))))
    return actor


def _dpi_of(plotter) -> int:
    """The render window's DPI, which is what its text actors are drawn at.

    Asked of the window rather than assumed, so the width measured below is
    measured for the same rendering that is about to happen.
    """
    window = getattr(plotter, "ren_win", None) or getattr(plotter, "render_window", None)
    try:
        return int(window.GetDPI())
    except Exception:  # noqa: BLE001 - a plotter with no window yet
        return 72


def _text_width(text: str, size: int, dpi: int) -> int:
    """How wide ``text`` is drawn, in pixels — where the subscript begins.

    Measured with the same font and size VTK will draw it at, through VTK's own
    text renderer. Guessing from the character count puts the subscript inside
    the letter or out in space, differently for every label.
    """
    if not text:
        return 0
    import vtk

    prop = vtk.vtkTextProperty()
    prop.SetFontSize(size)
    font = unicode_font()
    if font is not None:
        prop.SetFontFamily(VTK_FONT_FILE)
        prop.SetFontFile(font)
    box = [0, 0, 0, 0]
    try:
        if vtk.vtkTextRenderer.GetInstance().GetBoundingBox(prop, text, box, dpi):
            return int(box[1] - box[0]) + 1
    except Exception:  # noqa: BLE001 - no text renderer: fall back to an estimate
        pass
    return int(round(size * 0.6 * len(text)))


def _billboard(plotter, position, text: str, colour: str, size: int,
               offset=(0, 0)):
    """A label anchored in the scene, drawn as its own actor.

    Not ``add_point_labels``: that builds a ``vtkLabelPlacementMapper``, which
    decides for itself which labels are worth drawing — by collision, and by
    how much time the current render was allowed. The result is labels that
    flicker in during an interaction and vanish when it ends, with nothing to
    say why. A billboard actor has no such opinion: it faces the camera, sits
    where it is put, and is always drawn.
    """
    import vtk
    from pyvista import Color

    actor = vtk.vtkBillboardTextActor3D()
    actor.SetInput(text)
    actor.SetPosition(*[float(v) for v in position])
    actor.SetDisplayOffset(int(offset[0]), int(offset[1]))
    prop = actor.GetTextProperty()
    prop.SetFontSize(size)
    # VTK's built-in font draws a Greek letter as nothing at all, so the plain
    # text these labels fall back to needs a font that has one. MathText, when
    # it is working, uses matplotlib's own fonts and ignores this.
    font = unicode_font()
    if font is not None:
        prop.SetFontFamily(VTK_FONT_FILE)
        prop.SetFontFile(font)
    prop.SetColor(*Color(colour).float_rgb)
    prop.SetJustificationToLeft()
    prop.SetVerticalJustificationToCentered()
    plotter.add_actor(actor, reset_camera=False, pickable=False, render=False)
    return actor


def _spacer(width: int) -> QWidget:
    """A fixed gap between groups of chips, as the structure toolbar uses."""
    widget = QWidget()
    widget.setFixedWidth(width)
    return widget



@lru_cache(maxsize=1)
def mathtext_works() -> bool:
    """Whether VTK will actually draw ``$\\Gamma$`` as a gamma on this machine.

    ``_enable_mathtext`` registers the backend, but registering it is not the
    same as it working: where matplotlib and VTK disagree about their internals,
    the dispatcher still routes the string to MathText and the raster comes back
    empty — or the string is drawn verbatim, dollar signs and all, which is what
    a user sees. Asking VTK to rasterise one glyph answers it for certain, once.
    """
    try:
        from vtkmodules.vtkCommonDataModel import vtkImageData
        from vtkmodules.vtkRenderingCore import vtkTextProperty, vtkTextRenderer

        renderer = vtkTextRenderer.GetInstance()
        if renderer is None or renderer.DetectBackend(r"$\Gamma$") != 2:  # 2 = MathText
            return False
        image = vtkImageData()
        if not renderer.RenderString(vtkTextProperty(), r"$\Gamma$", image, [0, 0], 72):
            return False
        width, height, _ = image.GetDimensions()
        return width > 1 and height > 1
    except Exception:  # noqa: BLE001 - any doubt at all means use plain text
        return False


def _enable_mathtext() -> None:
    """Let VTK render ``$\\Gamma$`` and ``$k_x$`` instead of nothing.

    VTK's built-in font carries no Greek and cannot set a subscript, so those
    labels came out blank. Its MathText backend can do both — but only if
    ``vtkRenderingMatplotlib`` has been imported, which registers it. Importing
    it is the whole of the fix, and it is not an error if the build lacks it:
    the labels fall back to plain text.
    """
    try:
        import vtkmodules.vtkRenderingMatplotlib as _mathtext  # noqa: F401
        _ = _mathtext  # imported purely to register the backend
    except Exception:  # noqa: BLE001 - a build without it still draws a zone
        pass


def segment_colour(index: int) -> str:
    """The colour of the ``index``-th leg of a path, in the picture and the legend."""
    return _SEGMENT_COLOURS[index % len(_SEGMENT_COLOURS)]


def _plane_normal(vertices: np.ndarray):
    """The normal of a flat zone, or ``None`` if it is not flat."""
    if len(vertices) < 3:
        return None
    normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    length = float(np.linalg.norm(normal))
    if length < 1e-12:
        return None
    return normal / length


def _is_crystal(structure: Structure) -> bool:
    """Whether this is a 3D crystal — the only case with a cell to choose."""
    return not (is_linear(structure) or is_planar(structure)) and bool(structure.is_periodic)


def _segment_direction(vertices: np.ndarray):
    """The unit vector along a two-point zone, or ``None`` if it is not one."""
    if len(vertices) != 2:
        return None
    along = np.asarray(vertices[1], dtype=float) - np.asarray(vertices[0], dtype=float)
    length = float(np.linalg.norm(along))
    if length < 1e-12:
        return None
    return along / length


def _facing(normal: np.ndarray) -> np.ndarray:
    """A plane's normal, turned to point towards the viewer rather than away.

    Which way ``_plane_normal`` points falls out of the order the zone's
    corners happen to be wound in, so without this the same slab could open
    seen from above or from below between one lattice and the next. Towards
    +z where the plane allows it, else +y, else +x.
    """
    normal = np.asarray(normal, dtype=float)
    for axis in (np.array([0.0, 0, 1.0]), np.array([0, 1.0, 0]), np.array([1.0, 0, 0])):
        along = float(np.dot(normal, axis))
        if abs(along) > 1e-6:
            return normal if along > 0 else -normal
    return normal


def _upright_in(normal: np.ndarray) -> np.ndarray:
    """Which way is up for a plane seen face-on.

    The cartesian axis that lies most nearly *in* the plane, preferring z and
    then y — so a slab in the xy plane comes up the familiar way round, k_y up
    and k_x to the right, rather than on its side.
    """
    normal = np.asarray(normal, dtype=float)
    best, best_length = None, 0.0
    for axis in (np.array([0.0, 0, 1.0]), np.array([0, 1.0, 0]), np.array([1.0, 0, 0])):
        inside = axis - normal * float(np.dot(axis, normal))
        length = float(np.linalg.norm(inside))
        if length > best_length + 1e-6:          # a later axis must be clearly better
            best, best_length = inside, length
    if best is None or best_length < 1e-9:
        return _perpendicular_to(normal)
    return best / best_length


def _perpendicular_to(direction: np.ndarray) -> np.ndarray:
    """Some unit vector square to ``direction`` — "up", for a zone with no up.

    Built from whichever cartesian axis is least like it, so the choice is
    stable for a chain along any direction rather than degenerating when the
    chain happens to lie along the one picked.
    """
    direction = np.asarray(direction, dtype=float)
    axis = np.zeros(3)
    axis[int(np.argmin(np.abs(direction)))] = 1.0
    out = np.cross(direction, np.cross(axis, direction))
    length = float(np.linalg.norm(out))
    return out / length if length > 1e-12 else axis


def _outward(point: np.ndarray) -> np.ndarray:
    """A unit vector pointing away from the zone's centre, for label placement.

    Γ sits at the centre and has no outward direction of its own; it gets a
    fixed one, which keeps its label clear of its marker like all the others.
    """
    length = float(np.linalg.norm(point))
    if length < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    return np.asarray(point, dtype=float) / length


class _ZoneView(QWidget):
    """A small VTK viewport of its own, separate from the structure's."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        _enable_mathtext()
        from pyvistaqt import QtInteractor

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # Not redrawn five times a second for as long as the dialog is open,
        # as pyvistaqt's default does: every change here asks for its draw
        # (see Viewport, which turns the same timer off).
        self.plotter = QtInteractor(self, auto_update=False)
        self.plotter.setAcceptDrops(False)  # see Viewport: drops belong to the window
        layout.addWidget(self.plotter)
        from crystalline.ui import theme
        from PySide6.QtWidgets import QApplication

        self.plotter.set_background(theme.active_palette(QApplication.instance()).scene)
        self._smooth_the_zoom()
        self.setMinimumSize(360, 320)

    def export_image(self, path: str, *, scale: int = 1,
                     transparent: bool = False) -> str:
        """Save the view, by the same route the structure viewport uses."""
        from crystalline.viz.export import save_view_image

        return save_view_image(self.plotter, path, scale=scale,
                               transparent=transparent)

    def _smooth_the_zoom(self) -> None:
        """Zoom exactly as the structure viewport does.

        Not "smoothly" by some separate measure — *the same*, from the same
        module, so the two 3D views in the app cannot end up with different
        wheels under the same hand. VTK's own fixed-step dolly is consumed and
        replaced with one proportional to the real scroll delta.

        The parallel projection stays: it is the right one for a polyhedron
        diagram, since every face's edges stay parallel the way a published
        zone figure draws them.
        """
        self.plotter.enable_parallel_projection()
        self.plotter.interactor.installEventFilter(self)

    @guard(default=False)
    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt's name
        if obj is self.plotter.interactor and event.type() == QEvent.Wheel:
            factor = wheel_zoom.zoom_factor(event)
            if factor != 1.0:
                wheel_zoom.apply_zoom(
                    self.plotter.renderer.GetActiveCamera(), factor)
                self.plotter.renderer.ResetCameraClippingRange()
                self.plotter.render()
            return True  # consumed: VTK's own fixed-step dolly must not also run
        return super().eventFilter(obj, event)


__all__ = ["ZonePickerDialog"]
