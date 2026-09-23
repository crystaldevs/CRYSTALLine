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

import vtk

__all__ = ["unicode_font", "use_unicode_font"]


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
        text.SetFontFamily(vtk.VTK_FONT_FILE)
        text.SetFontFile(font)
    except Exception:  # noqa: BLE001 - purely cosmetic; never break a redraw
        pass
