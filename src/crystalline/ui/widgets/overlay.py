"""A widget that covers its parent and keeps covering it.

Both things drawn over the 3D view — the busy scrim and the drag-and-drop hint
— are the same widget underneath: a child sized to its parent's rectangle,
hidden until something puts it up. A child does not follow its parent's
geometry on its own, so each watches the parent for resizes through an event
filter. What they do *not* share is how they paint, and whether the pointer
goes through them: a hint must never become the drop target it is describing,
while a scrim exists precisely to swallow clicks. Those stay with each.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from crystalline.ui.safety import guard

__all__ = ["ParentOverlay"]


class ParentOverlay(QWidget):
    """Base for an overlay: sized to ``parent``, hidden until shown."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setVisible(False)
        parent.installEventFilter(self)

    @guard(False)
    def eventFilter(self, obj, event):
        if obj is self.parent() and event.type() in (event.Type.Resize, event.Type.Show):
            self._fit()
        return super().eventFilter(obj, event)

    def _fit(self) -> None:
        parent = self.parentWidget()
        if parent is not None:
            self.setGeometry(parent.rect())

    def _raise_over_parent(self) -> None:
        """Put the overlay up, on top of whatever the parent has drawn."""
        self._fit()
        self.raise_()
        self.setVisible(True)
