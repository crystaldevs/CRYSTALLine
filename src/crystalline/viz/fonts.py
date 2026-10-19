"""The font the 3D labels are drawn in.

VTK's built-in fonts stop at Latin-1, and a character they do not have is
drawn as nothing at all — which is worst where it matters most. A Brillouin
zone's labels are Γ, → and Å⁻¹; a symmetry element's are σ and the combining
overbar of 1̄ and 4̄, and dropping that bar turns a centre of inversion into
the identity and a rotoinversion axis into a rotation.

DejaVu Sans has all of them and ships with matplotlib, which arrives with
pymatgen, so it is on hand wherever the app runs. Everything here is
best-effort: without the file, or on a VTK build that arranges its text
properties differently, the labels fall back to the built-in font and are
merely uglier.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

# Named modules rather than the ``vtk`` umbrella: importing ``vtk`` pulls in all
# 163 VTK modules, where PyVista needs 57, and costs about 0.6 s of every start.
from vtkmodules.vtkCommonCore import VTK_FONT_FILE

__all__ = ["match_screen_dpi", "unicode_font", "use_unicode_font"]


@lru_cache(maxsize=1)
def unicode_font() -> Optional[str]:
    """Path to a font file holding those symbols, or ``None`` if there is none."""
    try:
        import matplotlib

        font = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        return str(font) if font.is_file() else None
    except Exception:  # noqa: BLE001 - labels fall back to the built-in font
        return None


def use_unicode_font(actor) -> None:
    """Draw a label actor's text in :func:`unicode_font`, if there is one.

    The text property sits on the label *hierarchy* feeding the mapper, not on
    the actor, so it takes a step through the pipeline to reach.
    """
    font = unicode_font()
    if font is None:
        return
    try:
        text = actor.GetMapper().GetInputAlgorithm().GetTextProperty()
        text.SetFontFamily(VTK_FONT_FILE)
        text.SetFontFile(font)
    except Exception:  # noqa: BLE001 - purely cosmetic; never break a redraw
        pass


def match_screen_dpi(window, pixel_ratio: float) -> bool:
    """Draw ``window``'s text at the size its font size says, on this screen.

    A font size is in points, and VTK turns it into pixels as ``size x DPI /
    72`` — with the DPI the render window was *told*, which nothing tells it:
    it stays at 72 on every screen. On a Retina screen the window is two pixels
    to every point of the widget it fills, so a label set at 15 came out 15
    pixels tall there, which is 7.5 points: half the size it is on an ordinary
    screen, and half the size the same number means in the rest of the app.
    Told 72 x the screen's pixel ratio, it draws the label at the same size on
    both. Returns whether that changed anything, since text that changed size
    has to be drawn again.
    """
    dpi = int(round(72 * float(pixel_ratio or 1.0)))
    try:
        if int(window.GetDPI()) == dpi:
            return False
        window.SetDPI(dpi)
        return True
    except Exception:  # noqa: BLE001 - no window yet: text stays as VTK sizes it
        return False
