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
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from crystalline.core.brillouin import brillouin_zone, reciprocal_cell, special_points
from crystalline.core.structure import Structure
from crystalline.ui.safety import guard

# How big a special point is drawn, as a fraction of the zone's own extent, so
# the marker is the same visual size whatever the lattice parameters are.
_POINT_RADIUS = 0.035
_PATH_WIDTH = 5


class ZonePickerDialog(QDialog):
    """Click special points in order; the path is the chain between them."""

    def __init__(self, structure: Structure, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Pick a path on the Brillouin zone")
        self.resize(880, 560)
        self._structure = structure
        self._picked: List[str] = []
        self._points = special_points(structure)
        self._reciprocal = reciprocal_cell(structure)
        self._actors: dict = {}
        self._path_actors: list = []
        self._extent = 1.0

        outer = QVBoxLayout(self)
        note = QLabel(
            "Click the labelled points in the order the path should visit them. "
            "The zone is the Wigner–Seitz cell of the reciprocal lattice."
            if self._points else
            "This lattice could not be classified, so it has no labelled points — "
            "the zone is drawn, but the path has to be typed."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: palette(mid);")
        outer.addWidget(note)

        body = QHBoxLayout()
        self._view = _ZoneView(self)
        body.addWidget(self._view, 1)
        side = QVBoxLayout()
        side.addWidget(QLabel("Path"))
        self._list = QListWidget()
        self._list.setMaximumWidth(190)
        side.addWidget(self._list, 1)
        undo = QPushButton("Remove last")
        undo.clicked.connect(self._undo)
        clear = QPushButton("Clear")
        clear.clicked.connect(self._clear)
        side.addWidget(undo)
        side.addWidget(clear)
        body.addLayout(side)
        outer.addLayout(body)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        self._draw()

    # ── the picture ─────────────────────────────────────────────────────
    def _draw(self) -> None:
        plotter = self._view.plotter
        plotter.clear()
        self._actors = {}
        self._path_actors = []
        try:
            vertices, faces = brillouin_zone(self._structure)
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

        for label, fractional in self._points.items():
            centre = np.asarray(fractional) @ self._reciprocal
            sphere = pv.Sphere(radius=_POINT_RADIUS * self._extent, center=centre)
            actor = plotter.add_mesh(sphere, color="#d6453c", pickable=True)
            self._actors[label] = (actor, centre)
            plotter.add_point_labels([centre], [label], font_size=13,
                                     shape=None, always_visible=True,
                                     text_color="#d6453c", pickable=False)
        plotter.enable_mesh_picking(callback=self._on_pick, show=False,
                                    show_message=False, left_clicking=True)
        plotter.reset_camera()

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
        self._picked.append(best)
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
            self._list.addItem(f"{start}  →  {end}")
        if len(self._picked) == 1:
            self._list.addItem(f"{self._picked[0]}  → …")
        self._draw_path()

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
        dialog = cls(structure, parent)
        if dialog.exec() != QDialog.Accepted:
            return None
        return dialog.path() or None


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
        self.setMinimumSize(360, 320)


__all__ = ["ZonePickerDialog"]
