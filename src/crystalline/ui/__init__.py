"""PySide6 UI layer. The only package (besides ui-adjacent viewport wiring)
that imports Qt.

``MainWindow`` is re-exported here, but *lazily*: importing it eagerly made
``from crystalline.ui import safety, theme`` — the first thing startup does —
pull in the window, and with it VTK, PyVista and matplotlib's font scan. The
splash screen then appeared after the slow part rather than before it, which is
the whole thing it exists to cover. See :func:`crystalline.app.run`.
"""

__all__ = ["MainWindow"]


def __getattr__(name: str):
    """Import ``MainWindow`` on first use (PEP 562)."""
    if name == "MainWindow":
        from crystalline.ui.main_window import MainWindow

        return MainWindow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
