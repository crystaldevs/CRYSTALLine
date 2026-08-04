"""Export the 3D view: still images and phonon-mode animations.

* **Stills** are captured from a live plotter (WYSIWYG) — raster via
  ``screenshot`` (PNG/JPEG/…) or vector via ``save_graphic`` (SVG/PDF/EPS/…).
* **Animations** are rendered on a *fresh off-screen* plotter so the live view
  isn't disturbed and the whole thing stays testable headlessly. The same
  :class:`StructureRenderer` + :class:`PhononAnimator` are driven over one full
  vibration cycle; frames are written as an animated GIF (via Pillow, no extra
  deps), an MP4/video (via imageio-ffmpeg, if installed), or a PNG sequence.

Qt-free on purpose: the UI hands in the structure/mode/settings and a file path.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pyvista as pv

from crystalline.core.phonons import PhononMode
from crystalline.viz.phonon_animator import DEFAULT_AMPLITUDE, PhononAnimator
from crystalline.viz.renderer import StructureRenderer
from crystalline.viz.render_settings import RenderSettings

# Formats ``pyvista`` writes as a raster screenshot vs. a vector graphic.
RASTER_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
VECTOR_IMAGE_EXTS = (".svg", ".eps", ".ps", ".pdf", ".tex")
# Animation containers.
GIF_EXTS = (".gif",)
MOVIE_EXTS = (".mp4", ".mov", ".avi", ".webm", ".mkv")

# Defaults for an exported loop: smooth enough, small enough.
DEFAULT_FRAMES = 36
DEFAULT_FPS = 18


# ── still image ─────────────────────────────────────────────────────────────
def save_view_image(
    plotter, path: str, *, scale: int = 1, transparent: bool = False
) -> str:
    """Save the plotter's current view to ``path`` (raster or vector by extension).

    ``scale`` supersamples a raster screenshot (2 → twice the pixels each way,
    the 3D equivalent of a higher DPI); ``transparent`` drops the background so
    the structure sits on an alpha channel. Both apply to raster formats only —
    vector output (``save_graphic``) has neither knob and ignores them.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in VECTOR_IMAGE_EXTS:
        plotter.save_graphic(path)
    elif ext in RASTER_IMAGE_EXTS:
        plotter.screenshot(path, transparent_background=transparent, scale=max(1, int(scale)))
    else:
        raise ValueError(
            f"unsupported image format '{ext}'. Use one of "
            f"{', '.join(RASTER_IMAGE_EXTS + VECTOR_IMAGE_EXTS)}."
        )
    return path


# ── animation ───────────────────────────────────────────────────────────────
def render_animation_frames(
    structure,
    equilibrium: np.ndarray,
    mode: PhononMode,
    settings: Optional[RenderSettings] = None,
    *,
    amplitude: float = DEFAULT_AMPLITUDE,
    n_frames: int = DEFAULT_FRAMES,
    reference_cell=None,
    bond_structure=None,
    camera=None,
    window_size=(800, 600),
) -> List[np.ndarray]:
    """Render one full vibration cycle off-screen; return a list of RGB frames.

    A fresh off-screen plotter/renderer is used so nothing about the live view
    changes. ``camera`` reproduces the on-screen view when supplied; otherwise the
    structure is auto-framed. It may be either a plain pyvista ``camera_position``
    or a ``(camera_position, parallel_scale, view_angle)`` snapshot — the latter is
    needed under parallel (orthographic) projection, where the zoom lives in the
    parallel scale, *not* in ``camera_position`` (so passing only the placement
    would let the off-screen plotter reframe and zoom the view).
    """
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    try:
        renderer = StructureRenderer(plotter, settings)
        renderer.set_reference_cell(reference_cell)
        renderer.set_structure(structure, bond_structure=bond_structure)
        if camera is not None:
            _apply_camera(plotter, camera)
        else:
            plotter.reset_camera()

        animator = PhononAnimator(renderer)
        animator.amplitude = amplitude
        animator.set_mode(np.asarray(equilibrium, dtype=float), mode)

        frames = []
        for phase in PhononAnimator.phase_sequence(n_frames):
            animator.set_frame(phase)
            frames.append(np.asarray(plotter.screenshot(return_img=True)))
        return frames
    finally:
        plotter.close()


def _apply_camera(plotter, camera) -> None:
    """Reproduce ``camera`` on ``plotter`` — placement, and zoom if provided.

    ``camera`` is either a bare ``camera_position`` (placement only) or a
    ``(camera_position, parallel_scale, view_angle)`` snapshot. Applying the
    parallel scale is what keeps a parallel-projection render at the on-screen zoom.
    The two are told apart by the second element: a scalar (the parallel scale) for
    the snapshot, a focal-point vector for a bare ``camera_position``.
    """
    if np.isscalar(camera[1]):
        position, parallel_scale, view_angle = camera
        plotter.camera_position = position
        plotter.camera.SetParallelScale(parallel_scale)
        plotter.camera.SetViewAngle(view_angle)
    else:
        plotter.camera_position = camera


def save_animation(frames: List[np.ndarray], path: str, *, fps: int = DEFAULT_FPS) -> List[str]:
    """Write ``frames`` to ``path``; format chosen by extension.

    ``.gif`` → animated GIF (Pillow); a video extension → MP4/… (imageio-ffmpeg);
    a raster image extension → a numbered PNG-style sequence (``stem_000.ext`` …).
    Returns the list of files written.
    """
    if not frames:
        raise ValueError("no frames to write")
    ext = os.path.splitext(path)[1].lower()
    if ext in GIF_EXTS:
        _save_gif(frames, path, fps)
        return [path]
    if ext in MOVIE_EXTS:
        _save_movie(frames, path, fps)
        return [path]
    if ext in RASTER_IMAGE_EXTS:
        return _save_frame_sequence(frames, path)
    raise ValueError(
        f"unsupported animation format '{ext}'. Use .gif, a video "
        f"({', '.join(MOVIE_EXTS)}), or an image extension for a frame sequence."
    )


def _save_gif(frames, path, fps) -> None:
    from PIL import Image

    images = [Image.fromarray(f) for f in frames]
    duration = max(1, int(round(1000.0 / max(1, fps))))  # ms per frame
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,  # loop forever
        disposal=2,  # restore background between frames (no ghosting)
    )


VIDEO_MISSING_MESSAGE = (
    "Video export needs the 'imageio-ffmpeg' package, which ships the ffmpeg "
    "encoder:\n\n    pip install imageio-ffmpeg\n\n"
    "GIF and frame-sequence export need no extras."
)


def video_export_available() -> bool:
    """Whether an ffmpeg backend is installed, so a video can actually be written.

    Checked *before* a mode is rendered rather than after: an animation is
    dozens of off-screen frames, and discovering the encoder is missing at the
    end means the whole wait was wasted.
    """
    try:
        import imageio_ffmpeg  # noqa: F401 - presence is the whole question
    except Exception:  # noqa: BLE001 - not installed, or installed but broken
        return False
    return True


def _save_movie(frames, path, fps) -> None:
    """Encode ``frames`` to a video container via imageio's ffmpeg plugin.

    The plugin is named explicitly. Left to itself, ``imageio.get_writer`` picks
    whatever plugin claims the extension, and with imageio-ffmpeg absent an
    ``.mp4`` lands on one that cannot encode video at all — which surfaced as
    ``TypeError: write() got an unexpected keyword argument 'fps'`` instead of
    anything a user could act on.
    """
    if not video_export_available():
        raise RuntimeError(VIDEO_MISSING_MESSAGE)
    import imageio

    frames = [_even_sized(np.asarray(frame)) for frame in frames]
    # macro_block_size=1 keeps the resolution that was asked for: the default
    # (16) silently rescales the whole movie up to the next multiple of 16.
    # h264 still needs even dimensions, which _even_sized guarantees.
    writer = imageio.get_writer(
        path, format="FFMPEG", fps=fps, macro_block_size=1, codec=_codec_for(path)
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


# Containers that cannot carry the default H.264 stream. Handing ffmpeg a codec
# its container rejects does not fail: it writes an empty file and reports
# success, so a .webm export came out as a few hundred bytes of header.
_CONTAINER_CODECS = {".webm": "libvpx-vp9"}


def _codec_for(path: str) -> str:
    """The video codec to encode ``path`` with, from its container."""
    return _CONTAINER_CODECS.get(os.path.splitext(path)[1].lower(), "libx264")


def _even_sized(frame: np.ndarray) -> np.ndarray:
    """Trim a frame to even width and height (h264 encodes nothing else).

    A row or column at most, off the bottom/right — invisible, and preferable to
    the alternative of rescaling every frame.
    """
    height, width = frame.shape[:2]
    return frame[: height - (height % 2), : width - (width % 2)]


def _save_frame_sequence(frames, path) -> List[str]:
    from PIL import Image

    directory = os.path.dirname(path) or "."
    stem, ext = os.path.splitext(os.path.basename(path))
    written = []
    for i, frame in enumerate(frames):
        out = os.path.join(directory, f"{stem}_{i:03d}{ext}")
        Image.fromarray(frame).save(out)
        written.append(out)
    return written


__all__ = [
    "RASTER_IMAGE_EXTS",
    "VECTOR_IMAGE_EXTS",
    "GIF_EXTS",
    "MOVIE_EXTS",
    "VIDEO_MISSING_MESSAGE",
    "save_view_image",
    "render_animation_frames",
    "save_animation",
    "video_export_available",
]
