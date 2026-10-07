"""A tab bar for more tabs than fit: readable names, and tabs that slide whole."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QTabBar

from crystalline.ui.safety import guard

_SIDES = (QTabBar.ButtonPosition.LeftSide, QTabBar.ButtonPosition.RightSide)

# How much of a name a crowded tab keeps, in characters. Qt's own minimum is
# three — "t…t" — which says nothing; a dozen tells two CRYSTAL runs apart
# ("thiou…80K.out") while still fitting several tabs across a bar.
READABLE_CHARACTERS = 12


class SlidingTabBar(QTabBar):
    """A :class:`QTabBar` for when the tabs outgrow the bar.

    Two things QTabBar does badly there:

    * It squeezes every name down to three characters before it starts to
      scroll, so the row reads "t…t  t…t  t…t". Here a tab keeps about
      :data:`READABLE_CHARACTERS` of its name, and the bar scrolls sooner.
    * A trackpad swipe across the bar scrolls it by repainting at the new
      offset and nothing more: the names and backgrounds slid along, while the
      close buttons — widgets of their own — stayed where they were until
      something next laid the bar out, so each × ended up beside another file's
      name. Here every button is moved on with its tab, to the place Qt had
      put it.
    """

    @guard(QSize(0, 0))
    def minimumTabSizeHint(self, index: int) -> QSize:  # noqa: N802 - Qt's name
        """Room for :data:`READABLE_CHARACTERS` of the name, not Qt's three."""
        smallest = super().minimumTabSizeHint(index)
        if self.elideMode() == Qt.TextElideMode.ElideNone:
            return smallest
        # Qt's minimum holds three characters and the ellipsis; widen it by the
        # rest, but never past the tab's full size — a short name stays whole.
        wider = smallest.width() + self.fontMetrics().averageCharWidth() * (READABLE_CHARACTERS - 3)
        full = self.tabSizeHint(index)
        width = min(wider, full.width()) if self._horizontal() else smallest.width()
        return QSize(max(width, smallest.width()), smallest.height())

    @guard()
    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt's name
        anchored = self._buttons_on_tabs()
        super().wheelEvent(event)
        for index, button, offset in anchored:
            rect = self.tabRect(index)
            if not rect.isNull():
                button.move(rect.topLeft() + offset)

    def _horizontal(self) -> bool:
        return self.shape() in (QTabBar.Shape.RoundedNorth, QTabBar.Shape.RoundedSouth,
                                QTabBar.Shape.TriangularNorth, QTabBar.Shape.TriangularSouth)

    def _buttons_on_tabs(self) -> list:
        """``(tab index, button, position on the tab)`` for every tab's buttons."""
        found = []
        for index in range(self.count()):
            rect = self.tabRect(index)
            if rect.isNull():
                continue
            for side in _SIDES:
                button = self.tabButton(index, side)
                if button is not None:
                    found.append((index, button, button.pos() - rect.topLeft()))
        return found


__all__ = ["READABLE_CHARACTERS", "SlidingTabBar"]
