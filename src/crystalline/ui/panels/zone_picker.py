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

import numpy as np
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
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

# Every label is pushed out to the same radius, as a multiple of the zone's
# extent — a ring of text around the polyhedron rather than a fringe hugging it.
# The high-symmetry points of an fcc lattice all sit within about thirty degrees
# of each other, so a fixed *offset* leaves the labels as bunched as the points
# are; a common radius spreads them by their angular separation instead, which
# is the only thing that actually differs between them.
_LABEL_RADIUS = 1.18

# Dash and gap for the guide lines, as fractions of the zone's extent. VTK's
# OpenGL2 backend dropped line stippling, so a dashed line is built out of real
# short segments; these are the pieces.
_DASH = 0.045
_GAP = 0.035

# The markers, and the one the side panel is pointing at.
_POINT_COLOUR = "#d6453c"
_CURRENT_COLOUR = "#f5a623"


class ZonePickerDialog(QDialog):
    """Click special points in order; the path is the chain between them."""

    def __init__(self, structure: Structure, parent: Optional[QWidget] = None,
                 picking: bool = True) -> None:
        super().__init__(parent)
        self.setWindowTitle("Pick a path on the Brillouin zone" if picking
                            else "Brillouin zone")
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

        # The tools live in a row above the picture rather than in a column
        # beside it, so nothing competes with the zone for width.
        tools = QHBoxLayout()
        tools.addWidget(QLabel("View"))
        for name, direction in (("kx", (1, 0, 0)), ("ky", (0, 1, 0)),
                                ("kz", (0, 0, 1)), ("⌂", None)):
            button = QPushButton(name)
            button.setFixedWidth(40)
            button.setToolTip("Back to the default three-quarter view"
                              if direction is None else f"Look down {name}")
            button.clicked.connect(
                lambda _checked=False, d=direction: self._look_along(d))
            tools.addWidget(button)
        tools.addSpacing(12)
        self._guides = QCheckBox("Symmetry lines")
        self._guides.setToolTip(
            "The lines Γ→X, Γ→L, Γ→K … that a zone diagram labels Δ, Λ and Σ. "
            "Dashed, because they run through the inside of the zone."
        )
        self._guides.setChecked(True)
        self._guides.toggled.connect(lambda _on: self._draw_and_sync())
        tools.addWidget(self._guides)
        # The coordinates ride with the points they belong to, in the picture
        # itself — a column of numbers beside a diagram makes the reader do the
        # matching. Off is for when the zone is wanted as a clean figure.
        self._coordinates = QCheckBox("All coordinates")
        self._coordinates.setToolTip(
            "Coordinates for every point at once. Off by default because the "
            "high-symmetry points of a cubic lattice all sit within about "
            "thirty degrees of each other, so six two-line labels overlap into "
            "an unreadable heap — which is why printed zone diagrams label the "
            "points and table the vectors.\n\n"
            "Click any point to see its own coordinates whatever this says."
        )
        self._coordinates.setChecked(False)
        self._coordinates.toggled.connect(lambda _on: self._draw_and_sync())
        tools.addWidget(self._coordinates)
        tools.addStretch(1)
        save = QPushButton("Save image…")
        save.setToolTip("Write the view to a PNG at twice the on-screen size, "
                        "which is what a figure wants.")
        save.clicked.connect(self._save_image)
        tools.addWidget(save)
        outer.addLayout(tools)

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
        """Write the view out as a PNG, at figure resolution."""
        from PySide6.QtWidgets import QFileDialog

        name = "brillouin-zone.png"
        path, _filter = QFileDialog.getSaveFileName(
            self, "Save the zone", name, "PNG image (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        self._view.plotter.screenshot(path, scale=2)

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
            caption = display_label(label)
            if self._coordinates.isChecked() or label == self._selected:
                caption += "\n" + "  ".join(_tidy_number(v) for v in fractional)
            anchor = self._label_anchor(centre, radius)
            plotter.add_point_labels([anchor], [caption], font_size=13, bold=True,
                                     shape=None, always_visible=True,
                                     show_points=False,
                                     text_color=self._label_colour(),
                                     pickable=False)
        if self._guides.isChecked():
            self._draw_guides(plotter, radius)
        self._draw_axes(plotter)
        # Every redraw re-arms the picker, and pyvista refuses to enable one
        # that is already enabled — so it is taken down first. Without this the
        # second redraw raises, and the safety net eats it: the picture stops
        # updating and nothing says why.
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

    def _label_anchor(self, centre: np.ndarray, radius: float) -> np.ndarray:
        """Where a point's label goes: out on the ring, or beside Γ.

        Γ is the one point with no direction of its own, and it is alone in the
        middle of the picture — so its label stays with it rather than being
        flung out to a ring position it does not have.
        """
        if float(np.linalg.norm(centre)) < 1e-9:
            return centre + np.array([0.0, 0.0, 1.0]) * radius * 3.0
        return _outward(centre) * self._extent * _LABEL_RADIUS

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
        for direction, name in ((np.array([1.0, 0, 0]), "kx"),
                                (np.array([0, 1.0, 0]), "ky"),
                                (np.array([0, 0, 1.0]), "kz")):
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
            plotter.add_point_labels([tip * 1.04], [name], font_size=12,
                                     shape=None, always_visible=True,
                                     show_points=False, italic=True,
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
        self._draw(keep_camera=self._view.plotter.camera_position)
        self._sync()

    def _undo(self) -> None:
        if self._picked:
            self._picked.pop()
            self._sync()

    def _clear(self) -> None:
        self._picked = []
        self._sync()

    def _sync(self) -> None:
        self._list.clear()
        for start, end in zip(self._picked, self._picked[1:]):
            self._list.addItem(
                f"{display_label(start)}  →  {display_label(end)}"
                f"     {self._segment_length(start, end):.3f} Å⁻¹")
        if len(self._picked) == 1:
            self._list.addItem(f"{display_label(self._picked[0])}  → …")
        total = self._path_length()
        self._total.setText(f"total  {total:.3f} Å⁻¹" if total else "")
        self._draw_path()

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
        if len(self._picked) >= 2:
            points = np.array([self._actors[label][1] for label in self._picked
                               if label in self._actors])
            if len(points) >= 2:
                self._path_actors.append(
                    plotter.add_mesh(pv.lines_from_points(points),
                                     color="#f5a623", line_width=_PATH_WIDTH,
                                     pickable=False)
                )
        plotter.render()

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
