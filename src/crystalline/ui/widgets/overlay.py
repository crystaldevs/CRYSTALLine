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

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QWidget

from crystalline.ui.safety import guard

__all__ = ["ParentOverlay"]


class ParentOverlay(QWidget):
    """Base for an overlay: sized to ``parent``, hidden until shown."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        # A window of its own, asked for rather than inherited. An overlay has
        # to be one to be seen over the 3D views, which are windows of their
        # own: drawn into its parent's, it would sit underneath them. It used
        # to become one by accident — Qt makes every sibling native when the
        # first VTK view asks for a window — and that is not enough once
        # _take_down has let the window go, since it is made again on showing
        # only for a widget that asks.
        self.setAttribute(Qt.WA_NativeWindow, True)
        self.setVisible(False)
        # Lets the window go a turn after the overlay is taken down (see
        # _take_down). A timer of its own, owned by the overlay, rather than a
        # singleShot: the overlay can be destroyed in that turn — a window
        # closing takes its scrim with it — and a loose call would arrive at a
        # widget whose C++ half has gone. This one is destroyed with it.
        self._letting_go = QTimer(self)
        self._letting_go.setSingleShot(True)
        self._letting_go.timeout.connect(self._renew_window)
        parent.installEventFilter(self)

    @guard(False)
    def eventFilter(self, obj, event):
        if obj is self.parent() and event.type() in (event.Type.Resize, event.Type.Show):
            self._fit()
            if not self.internalWinId():
                # Made here, with the window it belongs to, rather than at the
                # moment it is first put up: for the drop hint that moment is a
                # drag, and a window is better not made — or unmade — inside
                # one. See _renew_window, which keeps one ready after that.
                self.create()
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
        """Hide the overlay, and let its window go with it — a turn later.

        Hidden is not gone on macOS. The window keeps the last picture painted
        into it, and that picture reached the screen again, for one frame, each
        time a tab was closed: the drop hint, "Open 13 files", over the 3D view
        long after the files had been dropped. Not reproducible from a script
        — only after a real drag — so the picture is not left anywhere it could
        come back from.

        A turn later because of where this is called from. The drop hint is
        taken down inside ``dragLeaveEvent`` and ``dropEvent``, which run in the
        middle of the platform's drag session, and the session holds the native
        views it is dispatching to: destroying one there takes the process down
        with it. It is the hazard ``MainWindow.dropEvent`` already defers the
        opening of a file around; the window has to wait for the same moment.
        """
        self.setVisible(False)
        self._letting_go.start(0)

    @guard()
    def _renew_window(self) -> None:
        """Replace the hidden window with an empty one, once nothing is dragging.

        Replaced rather than simply dropped: without a window, the next showing
        would make one — and the next showing is the next drag, which puts the
        same work back inside a session. Made here instead, while nothing is in
        flight, and with nothing painted into it there is no picture to come
        back.
        """
        if self.isVisible() or not self.internalWinId():
            return  # shown again in the meantime, or no window to renew
        self.destroy(True, True)
        self.create()
