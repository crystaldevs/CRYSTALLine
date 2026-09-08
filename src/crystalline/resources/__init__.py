"""Bundled static assets (the app logo, theme glyphs) and helpers to locate them."""

from __future__ import annotations

from pathlib import Path


def logo_path() -> str:
    """Absolute path to the CRYSTALLine logo (SVG)."""
    return str(Path(__file__).with_name("logo.svg"))


def asset_path(name: str) -> str:
    """Absolute path to a bundled asset, in the form a Qt stylesheet can use.

    Qt's stylesheet ``url()`` resolves through the file and resource systems, not
    through data URIs, so a glyph drawn inline in the sheet simply does not
    appear — which is why these are files. Forward slashes because that is what
    ``url()`` expects on every platform, Windows included.
    """
    return str(Path(__file__).with_name(name)).replace("\\", "/")


__all__ = ["asset_path", "logo_path"]
