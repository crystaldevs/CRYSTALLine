"""A word-wrapped label that admits how tall it has to be.

A ``QLabel`` with word wrap reports the height of a *single line* as its
minimum, however many lines it will actually draw: ``QWidget.minimumSizeHint``
knows nothing of the width the label will be given. A dialog whose minimum is
computed from it is therefore a line or two too short, and Qt lets the window be
that size — so every row below the label is squeezed, and a widget with a
minimum height of its own (a list, a table) keeps its height regardless, because
``setGeometry`` clamps to it. It then paints over its neighbour, and a checkbox
is drawn across the bottom of a table.

So the label is asked, after each resize, how tall the text it is now wrapping
really is, and that becomes its minimum. The height travels up through the
layouts the ordinary way, and the dialog can no longer be made shorter than its
own contents.

The width it answers for is the one it currently has on screen — the question is
"is this dialog, as it stands, tall enough for what the label is drawing", not
"how tall would it have to be at the narrowest width it could be dragged to".
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QLabel, QWidget

from crystalline.ui.safety import guard


class WrappedLabel(QLabel):
    """A label that wraps its text and counts the lines towards its minimum."""

    def __init__(self, text: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self.setWordWrap(True)

    @guard(QSize(0, 0))
    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt spelling
        """The single-line minimum, raised to what this width actually needs."""
        hint = super().minimumSizeHint()
        width = self.width()
        # Only once the label is on screen does its width mean anything. Before
        # that it is Qt's default, and answering for it would set the dialog a
        # minimum height computed at a width it will never have — which is how a
        # small dialog opens with a hand's width of empty space under its note.
        if not self.isVisible() or width <= 0:
            return hint
        return QSize(hint.width(), max(hint.height(), self.heightForWidth(width)))

    @guard()
    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt spelling
        """Re-offer the minimum once the width is known.

        Narrower means more lines means taller, and the layout has to be told:
        it asked for the minimum before the label had a width to wrap at.
        """
        super().resizeEvent(event)
        if event.oldSize().width() != event.size().width():
            self.updateGeometry()

    @guard()
    def setText(self, text: str) -> None:  # noqa: N802 - Qt spelling
        """A longer message is a taller label; say so rather than be clipped."""
        super().setText(text)
        self.updateGeometry()


__all__ = ["WrappedLabel"]
