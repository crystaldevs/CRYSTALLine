"""The 3D viewport widget: a pyvistaqt interactor with atom pick + drag.

This is where PyVista/VTK meets Qt. It embeds a ``QtInteractor``, owns a
:class:`StructureRenderer`, and drives an :class:`AtomDragController` that turns
mouse interaction into two Qt signals:

* ``atom_picked`` — a click on an atom (selection),
* ``atom_moved``  — an atom was dragged to a new position (already committed to
  the model).

The viewport keeps a reference to the current :class:`Structure` so drags can be
committed via ``move_atom``; it's refreshed whenever a new structure is shown.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from pyvistaqt import QtInteractor
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout

from crystalline.core.structure import Structure
from crystalline.ui.drag_controller import install_atom_drag
from crystalline.viz.renderer import StructureRenderer

# How far past the auto-framed view the user may zoom out before it's capped.
# Big enough to see context, small enough that the structure stays findable.
_MAX_ZOOM_OUT_FACTOR = 6.0

# Arrow-key nudge step (Å): a precise default, and a coarser one with Shift held.
_NUDGE_STEP = 0.1
_NUDGE_STEP_COARSE = 0.5
_ARROW_KEYS = (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down)


class Viewport(QWidget):
    """3D view of the current structure: click to select, drag to move atoms."""

    atom_picked = Signal(int, bool)  # picked atom index, and whether it's additive (Ctrl/Shift)
    atom_moved = Signal(int)   # emits the index of a dragged atom (post-commit)
    selection_cleared = Signal()
    interaction_started = Signal()  # an atom drag began (e.g. stop animation)
    delete_requested = Signal()  # Del/Backspace pressed while the 3D view has focus
    nudge_requested = Signal(object)  # arrow key: (dx, dy, dz) world vector to move the selection

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.interactor = QtInteractor(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.interactor)

        self.renderer = StructureRenderer(self.interactor)
        self._structure: Optional[Structure] = None
        self._reference_cell: Optional[np.ndarray] = None  # original cell, for axis views
        self._editing = False  # arrow-key nudging is only live in editing mode
        self.interactor.set_background("white")

        self._drag = install_atom_drag(
            self.interactor,
            self.renderer,
            on_commit=self._commit_move,
            on_click=self._on_click_atom,
            on_click_empty=self._on_click_empty,
            on_grab=lambda _index: self.interaction_started.emit(),
            editing=False,
        )
        # A QtInteractor can quietly swap back to its default trackball style on
        # focus changes, which kills atom-dragging. Re-assert our style when the
        # view is (re)entered/refocused. Do NOT do it on plain presses —
        # SetInteractorStyle during the press that starts a drag disrupts it.
        self.interactor.installEventFilter(self)

    def eventFilter(self, obj, event) -> bool:
        if obj is self.interactor:
            etype = event.type()
            if etype in (QEvent.Enter, QEvent.FocusIn):
                self._drag.reactivate()
            elif etype == QEvent.KeyPress:
                key = event.key()
                if key in (Qt.Key_Delete, Qt.Key_Backspace):
                    # The embedded VTK widget swallows plain key presses, so the
                    # Del menu shortcut never fires while the 3D view has focus
                    # (which it does right after picking an atom). Route it out.
                    self.delete_requested.emit()
                    return True  # consume: don't also let VTK act on it
                if self._editing and key in _ARROW_KEYS:
                    self.nudge_requested.emit(self._nudge_vector(key, event.modifiers()))
                    return True  # consume: arrows nudge the selection, not the camera
        return super().eventFilter(obj, event)

    def _nudge_vector(self, key, modifiers) -> np.ndarray:
        """World-space move for an arrow key, in the screen plane of the camera.

        Left/Right track the camera's horizontal axis and Up/Down its vertical
        one, so a nudge always goes the way it looks on screen regardless of how
        the structure is rotated. Shift takes a coarser step.
        """
        step = _NUDGE_STEP_COARSE if (modifiers & Qt.ShiftModifier) else _NUDGE_STEP
        camera = self.interactor.camera
        view_dir = np.asarray(camera.GetDirectionOfProjection(), dtype=float)
        up = np.asarray(camera.GetViewUp(), dtype=float)
        up = up / np.linalg.norm(up)
        right = np.cross(view_dir, up)
        right = right / np.linalg.norm(right)
        direction = {
            Qt.Key_Right: right, Qt.Key_Left: -right,
            Qt.Key_Up: up, Qt.Key_Down: -up,
        }[key]
        return direction * step

    def show_structure(self, structure: Structure, reference_cell=None, bond_structure=None) -> None:
        self._structure = structure
        self._reference_cell = None if reference_cell is None else np.asarray(reference_cell, float)
        # Draw the cell wireframe for ``reference_cell`` (the original unit cell)
        # if given, so a supercell keeps the original cell's outline.
        self.renderer.set_reference_cell(reference_cell)
        self.renderer.set_structure(structure, bond_structure=bond_structure)
        self.interactor.reset_camera()
        self._drag.reactivate()  # keep our atom-drag style active across rebuilds
        self._limit_zoom_out()

    def _limit_zoom_out(self) -> None:
        """Cap zoom-out relative to the framed view so the structure can't be lost."""
        camera = self.interactor.camera
        focal = np.asarray(camera.GetFocalPoint(), dtype=float)
        position = np.asarray(camera.GetPosition(), dtype=float)
        framed = float(np.linalg.norm(position - focal))
        if framed > 0.0:
            self._drag.set_max_distance(framed * _MAX_ZOOM_OUT_FACTOR)

    # ── view alignment ──────────────────────────────────────────────────
    def can_align_axes(self) -> bool:
        """Whether the loaded structure has a cell to align the view to."""
        cell = self._active_cell()
        return cell is not None and not np.allclose(cell, 0.0)

    def align_view_along(self, axis: int) -> None:
        """Look down a lattice axis (0=a, 1=b, 2=c): that vector points into screen.

        The camera is aimed along the chosen lattice vector at the structure's
        centre, with an up direction taken from another lattice vector so the
        crystal sits square-on. Falls back to the world axis if there's no cell.
        """
        if self._structure is None:
            return
        cell = self._active_cell()
        if cell is not None and not np.allclose(cell[axis], 0.0):
            direction = np.asarray(cell[axis], dtype=float)
        else:  # non-periodic (or degenerate vector): use the world axis
            direction = np.eye(3)[axis]
            cell = None

        dir_hat = direction / np.linalg.norm(direction)
        up = self._up_for(dir_hat, cell, axis)
        focal = np.asarray(self._structure.positions, dtype=float).mean(axis=0)

        camera = self.interactor.camera
        camera.SetFocalPoint(*focal)
        camera.SetPosition(*(focal - dir_hat * 10.0))  # distance refit by reset_camera
        camera.SetViewUp(*up)
        self.interactor.reset_camera()  # keeps direction + up, refits the distance
        self._drag.reactivate()
        self._limit_zoom_out()
        self.interactor.render()

    def set_annotations(self, annotations) -> None:
        """Draw the Geometry panel's measurements over the structure."""
        self.renderer.set_annotations(annotations)

    def set_symmetry_elements(self, elements, labels: bool = False) -> None:
        """Draw the Symmetry panel's ticked elements over the structure."""
        self.renderer.set_symmetry_elements(elements, labels=labels)

    def rotate_view(self, azimuth: float = 0.0, elevation: float = 0.0, roll: float = 0.0) -> None:
        """Orbit the camera around the structure by the given angles (degrees).

        ``azimuth`` turns the crystal left/right about the current up direction,
        ``elevation`` tips it up/down, and ``roll`` spins it in the screen plane
        about the axis perpendicular to the screen (positive = the structure
        turns clockwise). The camera keeps its distance, so this is a pure
        rotation of the view — the counterpart of the a/b/c alignment buttons
        for looking at the structure from an arbitrary angle. Works with or
        without a cell.
        """
        if self._structure is None:
            return
        camera = self.interactor.camera
        if azimuth:
            camera.Azimuth(azimuth)
        if elevation:
            camera.Elevation(elevation)
            # Elevation alone skews (and at the poles flips) the up vector.
            camera.OrthogonalizeViewUp()
        if roll:
            # vtkCamera.Roll turns the *up vector* about the view direction, so
            # the scene appears to go the other way: negate to make a positive
            # roll read as the structure turning clockwise on screen.
            camera.Roll(-roll)
        self._drag.reactivate()
        self.interactor.render()

    def _active_cell(self) -> Optional[np.ndarray]:
        """The cell that defines a/b/c for view alignment (reference cell first)."""
        if self._reference_cell is not None:
            return self._reference_cell
        if self._structure is not None and self._structure.is_periodic:
            cell = np.asarray(self._structure.cell, dtype=float)
            if not np.allclose(cell, 0.0):
                return cell
        return None

    @staticmethod
    def _up_for(dir_hat: np.ndarray, cell: Optional[np.ndarray], axis: int) -> np.ndarray:
        """An up vector roughly perpendicular to the view direction.

        Uses the *cyclically next* lattice vector (its component orthogonal to
        the view direction), so the three axis views agree with one another:
        looking down a puts b up and c to the right, down b puts c up and a to
        the right, down c puts a up and b to the right. Taking the other vectors
        in plain index order instead would give the b view a→up, which is the
        previous axis rather than the next — mirroring that one view against the
        other two, and sending the a/b/c gizmo to the opposite side of the screen.

        Falls back to the remaining lattice vector, then to a world axis that
        isn't parallel to the view direction.
        """
        if cell is not None:
            for other in ((axis + 1) % 3, (axis + 2) % 3):
                vec = np.asarray(cell[other], dtype=float)
                perp = vec - dir_hat * float(np.dot(vec, dir_hat))
                if np.linalg.norm(perp) > 1e-3:
                    return perp / np.linalg.norm(perp)
        # world-axis fallback: pick the one least aligned with the view direction
        world = np.eye(3)[int(np.argmin(np.abs(dir_hat)))]
        perp = world - dir_hat * float(np.dot(world, dir_hat))
        return perp / np.linalg.norm(perp)

    def set_editing_enabled(self, enabled: bool) -> None:
        """Enable/disable drag-to-move (selection and camera stay available)."""
        self._editing = bool(enabled)  # gates arrow-key nudging
        self._drag.set_editing_enabled(enabled)
        # Toggling from the menu can leave QtInteractor on its default style;
        # make sure our atom-drag style is (re)active so dragging still works.
        self._drag.reactivate()

    def set_selection(self, indices) -> None:
        """Highlight exactly the given atom indices (the shared selection)."""
        self.renderer.highlight(indices)

    # ── export ──────────────────────────────────────────────────────────
    def export_image(self, path: str, *, scale: int = 1, transparent: bool = False) -> str:
        """Save the current view to an image file (raster or vector by extension).

        ``scale`` supersamples the raster output; ``transparent`` drops the
        background (raster formats only — see :func:`save_view_image`).
        """
        from crystalline.viz.export import save_view_image

        return save_view_image(self.interactor, path, scale=scale, transparent=transparent)

    @property
    def camera_position(self):
        """The current camera placement (to reproduce this view off-screen)."""
        return self.interactor.camera_position

    @property
    def camera_state(self):
        """Full camera snapshot — placement *and* zoom — for off-screen renders.

        ``camera_position`` alone omits the parallel-projection zoom (the camera's
        parallel scale), so an export that used only it would reframe the structure
        and change the zoom. This carries the scale (and perspective view angle) too.
        """
        camera = self.interactor.camera
        return (self.interactor.camera_position, camera.GetParallelScale(), camera.GetViewAngle())

    # ── drag-controller callbacks ───────────────────────────────────────
    def _commit_move(self, index: int, position: np.ndarray) -> None:
        """Persist a drag to the model; the whole move group shifts by the same vector.

        The move group (the dragged atom's periodic images, plus the rest of a
        multi-atom selection) translates together — matching the live preview —
        so a selected fragment moves as one piece and periodicity is preserved.
        """
        if self._structure is None:
            return
        group = self.renderer.active_move_group(index)  # same group the preview used
        delta = np.asarray(position, dtype=float) - self._structure.positions[index]
        self._structure.translate_atoms([index, *group], delta)  # -> refresh via listener
        self.atom_moved.emit(index)

    def _on_click_atom(self, index: int, additive: bool) -> None:
        self.atom_picked.emit(index, additive)

    def _on_click_empty(self) -> None:
        self.selection_cleared.emit()


__all__ = ["Viewport"]
