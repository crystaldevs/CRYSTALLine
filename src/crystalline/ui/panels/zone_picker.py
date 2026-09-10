"""Build a band path by clicking points on the first Brillouin zone.

A band path is a dozen lines of integers over a shrinking factor, and the
question it answers — *where in the zone does this go* — is a shape. Typing
``0 0 0  4 0 4`` is a poor way to ask it, and reading one back is worse.

Its own dialog rather than an overlay on the crystal, deliberately: real space
and reciprocal space in one picture is confusing to look at, and the two would
fight over the camera. The zone gets its own small viewport with its own view.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QEvent, QSize
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon
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
    reciprocal_cell,
    special_points,
    zone_lattice,
)
from crystalline.core.structure import Structure
from crystalline.ui import wheel_zoom
from crystalline.ui.safety import guard

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

# A label sits just clear of its own marker, as a multiple of the marker's
# radius — both are world lengths, so the gap on screen is the same at every
# zoom. The numbers that used to ride along underneath now go in the corner
# legend, which is what lets the labels sit this close without colliding.
_LABEL_OFFSET = 2.4

# One colour per leg of the path, reused around when a path is longer than the
# list. Chosen to stay apart on a pale ground and in the legend beside them.
# The same step the structure window's rotate chips use.
_ROTATE_STEP_DEG = 15.0

# Where the corner readout sits, in viewport fractions. Inset from the edge:
# text hard against the frame reads as though it has been cropped.
_LEGEND_X = 0.955
_LEGEND_TOP = 0.915

_SEGMENT_COLOURS = [
    "#e8710a", "#1a73e8", "#12a150", "#a142f4",
    "#d93025", "#00897b", "#c77700", "#5b6bd6",
]

# Dash and gap for the guide lines, as fractions of the zone's extent. VTK's
# OpenGL2 backend dropped line stippling, so a dashed line is built out of real
# short segments; these are the pieces.
_DASH = 0.045
_GAP = 0.035

# Labels pymatgen and ASE spell out, which MathText sets as the letter itself.
_GREEK = {"G": r"\Gamma", "Gamma": r"\Gamma", "Sigma": r"\Sigma",
          "Delta": r"\Delta", "Lambda": r"\Lambda"}

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
        bar.setIconSize(QSize(22, 18))

        caption = QLabel("VIEW")
        caption.setContentsMargins(6, 0, 6, 0)
        bar.addWidget(caption)
        # Coloured like the structure window's a/b/c chips and like the axes
        # drawn in the view, so a chip and its arrow are plainly the same axis.
        for axis, letter, direction in (("a", "x", (1, 0, 0)),
                                        ("b", "y", (0, 1, 0)),
                                        ("c", "z", (0, 0, 1))):
            button = QToolButton(bar)
            button.setIcon(_subscript_icon("k", letter, "#ffffff", self.font()))
            button.setToolTip(f"Look down k{letter}")
            button.setProperty("chip", "axis")
            button.setProperty("axis", axis)
            button.clicked.connect(
                lambda _checked=False, d=direction: self._look_along(d))
            bar.addWidget(button)

        bar.addWidget(_spacer(10))
        rotate = QLabel("ROTATE")
        rotate.setContentsMargins(2, 0, 6, 0)
        bar.addWidget(rotate)
        for label, tip, azimuth, elevation, roll in (
            ("◀", "Rotate left", -_ROTATE_STEP_DEG, 0.0, 0.0),
            ("▶", "Rotate right", _ROTATE_STEP_DEG, 0.0, 0.0),
            ("▲", "Rotate up", 0.0, _ROTATE_STEP_DEG, 0.0),
            ("▼", "Rotate down", 0.0, -_ROTATE_STEP_DEG, 0.0),
            ("↺", "Rotate anticlockwise in the screen plane", 0.0, 0.0, -_ROTATE_STEP_DEG),
            ("↻", "Rotate clockwise in the screen plane", 0.0, 0.0, _ROTATE_STEP_DEG),
        ):
            button = QToolButton(bar)
            button.setText(label)
            button.setToolTip(f"{tip} ({_ROTATE_STEP_DEG:g}°)")
            button.setAutoRepeat(True)          # hold to keep turning
            button.setProperty("chip", "ghost")
            button.clicked.connect(
                lambda _checked=False, a=azimuth, e=elevation, r=roll:
                self._rotate_view(a, e, r))
            bar.addWidget(button)

        bar.addWidget(_spacer(10))
        home = QToolButton(bar)
        home.setText("⌂")
        home.setToolTip("Back to the default three-quarter view")
        home.setProperty("chip", "ghost")
        home.clicked.connect(lambda _checked=False: self._look_along(None))
        bar.addWidget(home)

        bar.addWidget(_spacer(10))
        self._guides = QCheckBox("Symmetry lines")
        self._guides.setToolTip(
            "The lines Γ→X, Γ→L, Γ→K … that a zone diagram labels Δ, Λ and Σ. "
            "Dashed, because they run through the inside of the zone."
        )
        self._guides.setChecked(True)
        self._guides.toggled.connect(lambda _on: self._draw_and_sync())
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

        import pyvista as pv

        for face in faces:
            loop = list(face) + [face[0]]
            plotter.add_mesh(pv.lines_from_points(vertices[loop]),
                             color="#7f8c9b", line_width=2, pickable=False)
        # A translucent solid gives the zone its shape; the wireframe alone
        # reads as a tangle from most angles.
        if len(vertices) >= 4:
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
            anchor = centre + _outward(centre) * radius * _LABEL_OFFSET
            plotter.add_point_labels([anchor], [_math_label(label)],
                                     font_size=15, shape=None,
                                     always_visible=True, show_points=False,
                                     text_color=self._label_colour(),
                                     pickable=False)
        if self._guides.isChecked():
            self._draw_guides(plotter, radius)
        self._draw_axes(plotter)
        # pyvista refuses to enable a picker that is already enabled, so the
        # old one comes down first. A redraw is now rare — options and the
        # cell setting, not every click — so this costs nothing.
        plotter.disable_picking()
        plotter.enable_mesh_picking(callback=self._on_pick, show=False,
                                    show_message=False, left_clicking=True)
        if keep_camera is None:
            self._centre_on_gamma()
        else:
            plotter.camera_position = keep_camera
            plotter.renderer.ResetCameraClippingRange()
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

    def _draw_guides(self, plotter, radius: float) -> None:
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
        for direction, name in ((np.array([1.0, 0, 0]), r"$k_x$"),
                                (np.array([0, 1.0, 0]), r"$k_y$"),
                                (np.array([0, 0, 1.0]), r"$k_z$")):
            import pyvista as pv

            tip = direction * reach
            plotter.add_mesh(pv.lines_from_points(np.array([np.zeros(3), tip])),
                             color="#8a94a6", line_width=1, pickable=False)
            # A head, so the line reads as an axis rather than as another edge.
            head = self._extent * 0.06
            plotter.add_mesh(pv.Cone(center=tip - direction * head * 0.5,
                                     direction=direction, height=head,
                                     radius=head * 0.32, resolution=16),
                             color="#8a94a6", pickable=False)
            plotter.add_point_labels([tip * 1.04], [name], font_size=14,
                                     shape=None, always_visible=True,
                                     show_points=False,
                                     text_color=self._label_colour(),
                                     pickable=False)

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
                                        line_width=_PATH_WIDTH, pickable=False)
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
                                 line_width=_PATH_WIDTH, pickable=False)
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
                         font_file=_unicode_font(), name=name, render=False)
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

    VTK's built-in font has no Greek, so a bare "Γ" renders as *nothing at all*
    — the point looked unlabelled. Its MathText backend (matplotlib, registered
    by importing vtkRenderingMatplotlib) does have Greek, and italicises the
    letter the way a band diagram sets it, so labels go through that instead.
    Anything MathText might choke on falls back to plain text, which is worse
    looking but never blank.
    """
    name = str(label)
    greek = _GREEK.get(name.split("_")[0])
    if greek:
        name = greek + name[len(name.split("_")[0]):]
    elif not name.replace("_", "").isalnum():
        return display_label(label)
    return f"${name}$"


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


@lru_cache(maxsize=1)
def _unicode_font() -> Optional[str]:
    """A font file that actually has Γ, → and Å in it.

    VTK's built-in font has none of them, and draws a missing glyph as nothing
    at all — so a legend line reading "Γ → X  0.238 Å⁻¹" came out with holes
    where its most important characters were. DejaVu Sans ships with
    matplotlib, which is already a dependency.
    """
    try:
        import matplotlib

        path = (Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
                / "DejaVuSans.ttf")
        return str(path) if path.exists() else None
    except Exception:  # noqa: BLE001 - the labels degrade, nothing breaks
        return None


def _spacer(width: int) -> QWidget:
    """A fixed gap between groups of chips, as the structure toolbar uses."""
    widget = QWidget()
    widget.setFixedWidth(width)
    return widget


def _subscript_icon(base: str, sub: str, colour: str, font) -> QIcon:
    """``k`` with a real subscript, drawn as an icon.

    A QToolButton renders plain text only, and Unicode has a subscript x but no
    subscript y or z — so "k_y" cannot be written as a string at all. Drawing
    the two pieces at two sizes is the only way to label these chips the way a
    reciprocal-space axis is written.
    """
    from PySide6.QtGui import QFont, QFontMetrics, QPainter, QPixmap

    big = QFont(font)
    big.setBold(True)
    big.setPointSizeF(font.pointSizeF() + 1.0)
    small = QFont(big)
    small.setPointSizeF(big.pointSizeF() * 0.68)

    big_metrics, small_metrics = QFontMetrics(big), QFontMetrics(small)
    drop = small_metrics.height() * 0.28          # how far the subscript sits low
    width = big_metrics.horizontalAdvance(base) + small_metrics.horizontalAdvance(sub) + 2
    height = int(big_metrics.height() + drop) + 2

    # Retina: draw at the device ratio so the glyphs are not soft.
    ratio = 2
    pixmap = QPixmap(int(width) * ratio, height * ratio)
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.TextAntialiasing, True)
    painter.setPen(QColor(colour))
    baseline = big_metrics.ascent()
    painter.setFont(big)
    painter.drawText(1, baseline, base)
    painter.setFont(small)
    painter.drawText(1 + big_metrics.horizontalAdvance(base),
                     int(baseline + drop), sub)
    painter.end()
    return QIcon(pixmap)


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
        self.plotter = QtInteractor(self)
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
