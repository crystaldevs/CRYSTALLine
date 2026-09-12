"""How a scroll wheel zooms a 3D view — one answer, for every 3D view.

VTK's trackball style dollies a fixed ~21% per wheel *event*, whatever was
actually scrolled. That is what reads as notching: a mouse notch and the
faintest trackpad nudge move the camera equally far, and a trackpad's stream of
small events makes the view leap. Reading the real delta and raising a
per-notch factor to it makes the zoom continuous and proportional.

The structure viewport worked this way already. This module is where its
answer lives so that every other 3D view — the Brillouin zone, and whatever
comes next — zooms exactly like it rather than like a second opinion.
"""

from __future__ import annotations

import numpy as np

# How far one full notch of a mouse wheel zooms.
ZOOM_PER_NOTCH = 1.15
# What Qt reports for one notch, in eighths of a degree.
WHEEL_UNITS_PER_NOTCH = 120.0
# One gesture should never invert or teleport the view, however large a delta a
# device reports.
MAX_ZOOM_PER_EVENT = 4.0


def zoom_factor(event) -> float:
    """The factor a wheel event should zoom by, or ``1.0`` for no movement.

    ``>1`` moves in. Proportional to the delta, so a trackpad's many small
    events each do proportionally little and add up to the same travel.
    """
    delta = event.angleDelta().y() or event.angleDelta().x()
    if not delta:
        return 1.0
    notches = float(delta) / WHEEL_UNITS_PER_NOTCH
    return float(np.clip(ZOOM_PER_NOTCH ** notches,
                         1.0 / MAX_ZOOM_PER_EVENT, MAX_ZOOM_PER_EVENT))


def apply_zoom(camera, factor: float) -> None:
    """Zoom a vtkCamera by ``factor``, whichever projection it is using.

    Under parallel projection the "zoom" is the camera's parallel scale, not
    its distance, so dollying would do nothing at all.
    """
    if factor <= 0.0 or not np.isfinite(factor):
        return
    if camera.GetParallelProjection():
        scale = camera.GetParallelScale() / factor
        if scale > 0.0 and np.isfinite(scale):
            camera.SetParallelScale(scale)
    else:
        camera.Dolly(factor)


__all__ = ["MAX_ZOOM_PER_EVENT", "WHEEL_UNITS_PER_NOTCH", "ZOOM_PER_NOTCH",
           "apply_zoom", "zoom_factor"]
