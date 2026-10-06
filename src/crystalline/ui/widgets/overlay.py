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

from PySide6.QtCore import QRect, Qt
from PySide6.QtWidgets import QWidget

from crystalline.ui.safety import guard

__all__ = ["ParentOverlay"]


class ParentOverlay(QWidget):
    """Base for an overlay: sized to ``parent``, hidden until shown."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        # A window of its own, asked for rather than left to chance. An overlay
        # has to be one to be seen over the 3D views, which are windows of
        # their own: drawn into its parent's, it would sit underneath them. It
        # used to become one by accident — Qt makes every sibling native when
        # the first VTK view asks for a window.
        self.setAttribute(Qt.WA_NativeWindow, True)
        self.setVisible(False)
        parent.installEventFilter(self)

    @guard(False)
    def eventFilter(self, obj, event):
        # Only while it is up: taken down, it is kept at no size at all (see
        # _take_down), and fitting it to the parent again would undo that.
        if (obj is self.parent() and self.isVisible()
                and event.type() in (event.Type.Resize, event.Type.Show)):
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

    def _take_down(self) -> None:
        """Hide the overlay, and shrink it to nothing.

        Hidden is not gone on macOS. The window keeps the last picture painted
        into it, and that picture reached the screen again, for one frame, each
        time a tab was closed: the drop hint, "Open 13 files", over the 3D view
        long after the files had been dropped. Not reproducible from a script
        — only after a real drag — so the window is left no room to show it in:
        at no size, a picture it keeps has nowhere to appear. ``_raise_over_
        parent`` gives it its size back.

        Shrunk, not destroyed. Destroying the window was the first answer, and
        it crashed the app on every drop: the hint is up under the pointer
        while files are dragged in, so macOS takes its window for the drop
        target, and it is taken down in the middle of the drag — its window
        freed, then reached for by the drag's next message (a segfault in
        ``-[NSDragDestination _setTarget:]``). A window shrunk to nothing is
        still there to be reached for.

        Shrunk first, then hidden: Qt keeps a hidden widget's new size to
        itself until it is shown again, and the window would have stayed the
        size of the page, picture and all. And to a pixel, not to nothing: Qt
        takes an empty size to mean "hide the window" and leaves its frame as
        it was. One pixel of a picture — the faint corner of the hint's scrim —
        is not one anyone can see.
        """
        self.setGeometry(QRect(self.x(), self.y(), 1, 1))
        self.setVisible(False)
