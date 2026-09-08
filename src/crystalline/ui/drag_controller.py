"""Interactive drag-to-move for atoms, via a custom VTK interactor style.

Why a custom style (and not observers + a temporary style swap): pyvista routes
``LeftButtonReleaseEvent`` observers onto the *interactor style* (releases are
otherwise swallowed by the interactor — see pyvista issue #4976). Swapping the
style mid-drag to freeze the camera therefore throws away the release observer,
so the drag would never "drop". Instead we install ONE persistent style,
:class:`AtomDragStyle`, for the whole session.

The style subclasses the trackball camera and, per VTK's Python recipe, adds
observers that forward to the base handlers (``OnLeftButtonDown`` etc.) only when
we are *not* dragging an atom:

* press on an atom  -> begin drag, skip the base handler (camera stays put),
* press elsewhere   -> forward: normal rotate/pan/zoom,
* move while dragging-> move that atom's actor live (model untouched),
* release           -> commit to the model (one refresh, bonds redraw), or, if
  the pointer barely moved, treat it as a selection click.

Only picking/forwarding is VTK-specific; the projection is plain geometry
(``world_to_depth`` / ``screen_to_world``), unit-tested against a real camera.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import vtk

from crystalline.viz.renderer import StructureRenderer

# Pointer travel (pixels, L1) below which a press→release counts as a click.
_CLICK_SLOP_PX = 3


class AtomDragStyle(vtk.vtkInteractorStyleTrackballCamera):
    """Trackball camera style that also lets you drag atoms."""

    def __init__(
        self,
        plotter,
        renderer: StructureRenderer,
        on_commit: Callable[[int, np.ndarray], None],
        on_click: Optional[Callable[[int, bool], None]] = None,
        on_click_empty: Optional[Callable[[], None]] = None,
        on_grab: Optional[Callable[[int], None]] = None,
        on_camera: Optional[Callable[[bool], None]] = None,
        editing: bool = False,
    ) -> None:
        super().__init__()
        self._plotter = plotter
        self._renderer = renderer
        self._vtk_ren = plotter.renderer  # vtkRenderer subclass (pick + project)
        self._on_commit = on_commit
        self._on_click = on_click
        self._on_click_empty = on_click_empty
        self._on_grab = on_grab
        self._on_camera = on_camera
        self._editing = editing
        self._picker = vtk.vtkPropPicker()

        self._index: Optional[int] = None
        self._grabbed = False  # dragging this atom (edit mode), vs just tracking it
        self._depth = 0.0
        self._start_px = (0, 0)
        self._moved = False
        self._max_distance: Optional[float] = None  # zoom-out cap (camera↔focal)

        self.SetDefaultRenderer(self._vtk_ren)
        self.AddObserver("LeftButtonPressEvent", self._on_press)
        self.AddObserver("MouseMoveEvent", self._on_move)
        self.AddObserver("LeftButtonReleaseEvent", self._on_release)
        # Bracket every camera move (rotate/pan/dolly). ``on_camera`` lets the
        # app stand other continuous work down for the duration — see
        # Viewport.camera_busy.
        self.AddObserver("StartInteractionEvent", self._on_start_interaction)
        # Fires after any camera move (rotate/pan/dolly/wheel) — clamp zoom there.
        self.AddObserver("EndInteractionEvent", self._on_end_interaction)

    def set_editing_enabled(self, enabled: bool) -> None:
        """Turn drag-to-move on/off. When off, the pointer only orbits + selects."""
        self._editing = bool(enabled)

    def set_max_distance(self, distance: Optional[float]) -> None:
        """Cap how far the camera may pull back from the focal point (None = no cap)."""
        self._max_distance = distance

    def zoom_by(self, factor: float) -> None:
        """Zoom by ``factor`` (>1 moves in), continuously rather than in steps.

        VTK's trackball style dollies a fixed ~21% per wheel event whatever the
        scroll was, which is what makes the wheel feel notched — and on a
        trackpad, where a gentle two-finger swipe delivers many small events, it
        makes the view leap. The viewport reads the real scroll delta and calls
        this with a factor proportional to it.

        Handles both projections: under parallel projection the "zoom" is the
        camera's parallel scale, not its distance, so dollying would do nothing.
        """
        if factor <= 0.0 or not np.isfinite(factor):
            return
        camera = self._vtk_ren.GetActiveCamera()
        if camera.GetParallelProjection():
            scale = camera.GetParallelScale() / factor
            if scale > 0.0 and np.isfinite(scale):
                camera.SetParallelScale(scale)
        else:
            camera.Dolly(factor)
        self._clamp_zoom()
        self._vtk_ren.ResetCameraClippingRange()
        self._plotter.render()

    def _on_start_interaction(self, *_args) -> None:
        if self._on_camera is not None:
            self._on_camera(True)

    def _on_end_interaction(self, *_args) -> None:
        self._clamp_zoom()
        if self._on_camera is not None:
            self._on_camera(False)

    def _clamp_zoom(self) -> None:
        """Pull the camera back in if it zoomed out past the cap, so the
        structure can't vanish into the distance and get lost."""
        if not self._max_distance:
            return
        camera = self._vtk_ren.GetActiveCamera()
        focal = np.asarray(camera.GetFocalPoint(), dtype=float)
        position = np.asarray(camera.GetPosition(), dtype=float)
        offset = position - focal
        distance = float(np.linalg.norm(offset))
        if distance > self._max_distance and distance > 0.0:
            camera.SetPosition(*(focal + offset / distance * self._max_distance))
            self._vtk_ren.ResetCameraClippingRange()
            self._plotter.render()

    def reactivate(self) -> None:
        """Re-assert this style as the interactor's active style.

        A ``QtInteractor`` can quietly swap back to its default trackball style
        after focus/menu changes (e.g. toggling editing from the menu). When
        that happens, presses hit the default camera style and atom-dragging
        silently stops working. Re-installing our style restores it; if it is
        already active this is a harmless no-op.
        """
        _install_style(self._plotter, self)

    # ── event handlers (forward to base only when dragging an atom) ─────
    def _on_press(self, *_args) -> None:
        x, y = self.GetInteractor().GetEventPosition()
        self._start_px = (x, y)
        self._moved = False
        self._index = self._pick_atom(x, y)
        # Only *grab* (freeze camera, move the atom) when editing is on and an
        # atom was hit. Otherwise forward so the camera orbits as usual; a
        # no-move release is still a selection click (handled in _on_release).
        self._grabbed = self._editing and self._index is not None
        if self._grabbed:
            self._depth = world_to_depth(self._vtk_ren, self._renderer.atom_position(self._index))
            if self._on_grab is not None:
                self._on_grab(self._index)  # e.g. stop a running phonon animation
        else:
            self.OnLeftButtonDown()

    def _on_move(self, *_args) -> None:
        if not self._grabbed:
            self.OnMouseMove()  # default camera interaction
            return
        x, y = self.GetInteractor().GetEventPosition()
        if abs(x - self._start_px[0]) + abs(y - self._start_px[1]) > _CLICK_SLOP_PX:
            self._moved = True
        world = screen_to_world(self._vtk_ren, x, y, self._depth)
        self._renderer.preview_atom_position(self._index, world)
        self._plotter.render()

    def _on_release(self, *_args) -> None:
        x, y = self.GetInteractor().GetEventPosition()
        index, grabbed = self._index, self._grabbed
        self._index, self._grabbed = None, False

        if grabbed:
            if self._moved:
                self._on_commit(index, screen_to_world(self._vtk_ren, x, y, self._depth))
            elif self._on_click is not None:
                self._on_click(index, self._additive())
            return

        self.OnLeftButtonUp()  # default (finish any camera move)
        if not self._is_click(x, y):
            return  # a camera drag, not a click — leave the selection alone
        if index is not None and self._on_click is not None:
            self._on_click(index, self._additive())
        elif index is None and self._on_click_empty is not None:
            self._on_click_empty()

    def _additive(self) -> bool:
        """Whether Ctrl/Shift is held — i.e. add to the selection, not replace it."""
        iren = self.GetInteractor()
        return bool(iren.GetControlKey()) or bool(iren.GetShiftKey())

    def _is_click(self, x: int, y: int) -> bool:
        return abs(x - self._start_px[0]) + abs(y - self._start_px[1]) <= _CLICK_SLOP_PX

    # ── picking ─────────────────────────────────────────────────────────
    def _pick_atom(self, x: int, y: int) -> Optional[int]:
        # Atoms are one glyphed actor, so a hit is resolved to the nearest atom
        # centre from the picked surface point (see StructureRenderer.pick_atom_index).
        self._picker.Pick(x, y, 0, self._vtk_ren)
        return self._renderer.pick_atom_index(
            self._picker.GetActor(), self._picker.GetPickPosition()
        )


def _install_style(plotter, style) -> None:
    """Make ``style`` the interactor's active style, via pyvista's own tracking.

    Setting pyvista's ``iren.style`` property — rather than calling
    ``SetInteractorStyle`` on the raw VTK interactor — keeps pyvista's bookkeeping
    (``_style_class``) in sync. That matters because pyvista re-applies its
    *tracked* style through ``update_style()`` while handling ordinary mouse
    events (its chart-interaction check). If we install the style directly,
    pyvista's tracked style stays its DEFAULT, so the first mouse interaction
    reverts the interactor to that default and atom picking/dragging silently
    dies. Falls back to the raw call on pyvista versions without ``style``, and
    is a no-op when there's no live interactor (off-screen construction).
    """
    iren = getattr(plotter, "iren", None)
    if iren is None:
        return
    try:
        iren.style = style  # pyvista property: records + applies (update_style)
    except Exception:  # noqa: BLE001 - older pyvista: set the VTK interactor directly
        try:
            iren.interactor.SetInteractorStyle(style)
        except AttributeError:
            pass


def install_atom_drag(
    plotter, renderer, on_commit, on_click=None, on_click_empty=None, on_grab=None,
    on_camera=None, editing=False,
) -> AtomDragStyle:
    """Create and activate the drag style on ``plotter``'s interactor."""
    style = AtomDragStyle(
        plotter, renderer, on_commit, on_click, on_click_empty, on_grab, on_camera, editing
    )
    _install_style(plotter, style)
    return style


# ── projection helpers (plain geometry, unit-tested) ───────────────────────
def world_to_depth(vtk_ren, world: np.ndarray) -> float:
    """Display-space z (depth-buffer value) of a world point."""
    vtk_ren.SetWorldPoint(float(world[0]), float(world[1]), float(world[2]), 1.0)
    vtk_ren.WorldToDisplay()
    return vtk_ren.GetDisplayPoint()[2]


def screen_to_world(vtk_ren, x: float, y: float, depth: float) -> np.ndarray:
    """World point under screen (x, y) at the given display depth.

    Holding ``depth`` fixed at the dragged atom's value makes the cursor move
    the atom within the plane through it parallel to the image plane.
    """
    vtk_ren.SetDisplayPoint(float(x), float(y), float(depth))
    vtk_ren.DisplayToWorld()
    w = vtk_ren.GetWorldPoint()
    if w[3] != 0.0:
        return np.array([w[0] / w[3], w[1] / w[3], w[2] / w[3]])
    return np.array([w[0], w[1], w[2]])


__all__ = ["AtomDragStyle", "install_atom_drag", "world_to_depth", "screen_to_world"]
