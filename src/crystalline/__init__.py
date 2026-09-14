"""CRYSTALLine — a desktop GUI for CRYSTAL structures and vibrational modes.

The package is layered so that domain logic stays independent of the Qt UI:

    core/  — domain model (structures, phonon modes). No Qt, no rendering.
    io/    — adapters over CRYSTALClear for loading/saving CRYSTAL files.
    viz/   — PyVista-based 3D rendering and animation.
    ui/    — PySide6 widgets that wire the above together.

Only ``ui`` (and ``viz``'s Qt-embedding helpers) import Qt, so ``core`` and
``io`` remain unit-testable without a display server.
"""

__version__ = "0.2.2"
