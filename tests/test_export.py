"""Image and phonon-animation export (off-screen PyVista)."""

import os

import numpy as np
import pytest

pytest.importorskip("pyvista")
pytest.importorskip("ase")

import pyvista as pv  # noqa: E402
from ase.build import bulk  # noqa: E402

from crystalline.core.phonons import PhononMode  # noqa: E402
from crystalline.core.structure import Structure  # noqa: E402
from crystalline.viz import export  # noqa: E402
from crystalline.viz.render_settings import RenderSettings  # noqa: E402
from crystalline.viz.renderer import StructureRenderer  # noqa: E402


@pytest.fixture
def nacl():
    return Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64))


def _plotter_with(structure):
    p = pv.Plotter(off_screen=True)
    StructureRenderer(p).set_structure(structure)
    p.reset_camera()
    return p


def test_save_view_image_raster_and_vector(nacl, tmp_path):
    p = _plotter_with(nacl)
    png = export.save_view_image(p, str(tmp_path / "view.png"))
    svg = export.save_view_image(p, str(tmp_path / "view.svg"))
    p.close()
    assert os.path.getsize(png) > 0
    assert os.path.getsize(svg) > 0


def test_save_view_image_rejects_unknown_format(nacl, tmp_path):
    p = _plotter_with(nacl)
    with pytest.raises(ValueError):
        export.save_view_image(p, str(tmp_path / "view.xyz"))
    p.close()


def test_render_animation_frames_shape(nacl):
    mode = PhononMode(120.0, np.array([[1, 0, 0], [-1, 0, 0]], float))
    frames = export.render_animation_frames(
        nacl, nacl.positions, mode, RenderSettings(), amplitude=0.6, n_frames=6
    )
    assert len(frames) == 6
    assert frames[0].ndim == 3 and frames[0].shape[2] in (3, 4)
    assert frames[0].dtype == np.uint8
    # the mode actually moves atoms, so not every frame is identical
    assert not all(np.array_equal(frames[0], f) for f in frames[1:])


def test_apply_camera_preserves_parallel_zoom():
    """The export must reproduce the on-screen zoom. Under parallel projection that
    zoom is the camera's parallel scale, which a bare ``camera_position`` omits —
    so the full snapshot must be applied, or the render zooms in."""
    plotter = pv.Plotter(off_screen=True)
    try:
        plotter.add_mesh(pv.Sphere())
        plotter.enable_parallel_projection()
        snapshot = (plotter.camera_position, 0.37, plotter.camera.GetViewAngle())

        export._apply_camera(plotter, snapshot)
        assert np.isclose(plotter.camera.GetParallelScale(), 0.37)  # zoom applied

        # a bare camera_position must still work (placement only, no crash)
        export._apply_camera(plotter, plotter.camera_position)
    finally:
        plotter.close()


def test_save_animation_gif(tmp_path):
    frames = [np.full((20, 30, 3), i * 20, np.uint8) for i in range(5)]
    out = export.save_animation(frames, str(tmp_path / "anim.gif"), fps=10)
    assert out == [str(tmp_path / "anim.gif")]
    from PIL import Image

    with Image.open(out[0]) as im:
        assert getattr(im, "n_frames", 1) == 5


def test_save_animation_png_sequence(tmp_path):
    frames = [np.zeros((10, 10, 3), np.uint8) for _ in range(4)]
    written = export.save_animation(frames, str(tmp_path / "frame.png"))
    assert len(written) == 4
    assert all(os.path.exists(p) for p in written)
    assert os.path.basename(written[0]) == "frame_000.png"


def test_save_animation_rejects_unknown_and_empty(tmp_path):
    with pytest.raises(ValueError):
        export.save_animation([], str(tmp_path / "x.gif"))
    with pytest.raises(ValueError):
        export.save_animation([np.zeros((4, 4, 3), np.uint8)], str(tmp_path / "x.qqq"))


# ── video export ─────────────────────────────────────────────────────────
# Encoding needs imageio-ffmpeg, which a plain install doesn't pull in. The
# rules below have to hold whether or not it is present, so most of these test
# the decision rather than the encoding.
def test_a_missing_encoder_says_what_to_install(tmp_path, monkeypatch):
    """Without this the failure was `TypeError: write() got an unexpected keyword
    argument 'fps'` — imageio falling through to a plugin that cannot encode
    video at all, reported as a bug in our call rather than a missing package."""
    monkeypatch.setattr(export, "video_export_available", lambda: False)
    frames = [np.zeros((8, 8, 3), np.uint8)] * 2

    with pytest.raises(RuntimeError, match="imageio-ffmpeg"):
        export.save_animation(frames, str(tmp_path / "mode.mp4"))


def test_webm_is_encoded_with_a_codec_its_container_accepts():
    """Handing ffmpeg H.264 for a WebM container does not fail — it writes an
    empty file and reports success, so the export looked like it worked and
    produced 300 bytes."""
    assert export._codec_for("/tmp/mode.webm") == "libvpx-vp9"
    assert export._codec_for("/tmp/mode.mp4") == "libx264"
    assert export._codec_for("/tmp/mode.MOV") == "libx264"


def test_frames_are_trimmed_to_even_dimensions():
    """H.264 encodes even dimensions only. Trimming a row costs nothing visible;
    the alternative (imageio's default) rescales the whole movie to the next
    multiple of 16, quietly changing the resolution that was asked for."""
    assert export._even_sized(np.zeros((101, 103, 3), np.uint8)).shape == (100, 102, 3)
    assert export._even_sized(np.zeros((100, 102, 3), np.uint8)).shape == (100, 102, 3)


@pytest.mark.skipif(not export.video_export_available(), reason="needs imageio-ffmpeg")
@pytest.mark.parametrize("ext", [".mp4", ".webm"])
def test_a_written_video_decodes_back_to_the_frames_given(tmp_path, ext):
    """A file that exists is not a file that plays — this is what caught the
    empty WebM. Read back with imageio-ffmpeg's own reader rather than
    ``imageio.get_reader``: the latter parses the ffmpeg banner and mis-reads
    the one shipped with current imageio-ffmpeg, which is a fault in the check,
    not in the file."""
    import imageio_ffmpeg

    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (64, 65, 3)).astype(np.uint8) for _ in range(8)]
    path = str(tmp_path / f"mode{ext}")

    export.save_animation(frames, path, fps=12)

    stream = imageio_ffmpeg.read_frames(path)
    meta = next(stream)
    decoded = sum(1 for _ in stream)
    assert decoded == len(frames)
    assert meta["size"] == (64, 64)  # odd width trimmed, not rescaled
    assert meta["fps"] == 12.0
