"""A tab bar of fixed-width tabs: a row that stays put, and slides whole."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QTabBar

from crystalline.ui.safety import guard

_SIDES = (QTabBar.ButtonPosition.LeftSide, QTabBar.ButtonPosition.RightSide)

# How much of a name every tab has room for, in average characters: a tab is
# this wide whatever its name, as Firefox's are, so the row does not move when
# the bar it sits in gets narrower or wider. A name-sized tab moved with every
# change — the file tabs reflowed each time the Phonons panel came or went.
#
# Narrower than Firefox's ~225 px on purpose. A row that does not fit has to
# shrink, and shrinking is the movement this is here to stop: at 17 (181 px on
# macOS) three files fit the tab row left between both side panels of a
# 1200-point window, and a run name still shows both ends ("brucit…TENS.out").
TAB_CHARACTERS = 17

# How far a crowded tab shrinks before the bar scrolls instead, in characters.
# Qt's own minimum is three — "t…t" — which says nothing; a dozen tells two
# CRYSTAL runs apart ("thiou…80K.out") while still fitting several across.
READABLE_CHARACTERS = 12


class SlidingTabBar(QTabBar):
    """A :class:`QTabBar` of fixed-width tabs, for rows that change width.

    Three things QTabBar does badly for a row of open files:

    * Each tab is as wide as its name, so the row is laid out again whenever
      the bar's width changes, and every tab moves — including the one just
      clicked. Here every tab is :data:`TAB_CHARACTERS` wide, as in Firefox,
      and stays where it is for as long as the row fits.
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
    def tabSizeHint(self, index: int) -> QSize:  # noqa: N802 - Qt's name
        """Room for :data:`TAB_CHARACTERS` of a name, whatever the name is."""
        natural = super().tabSizeHint(index)
        if not self._fixed():
            return natural
        return QSize(self._width_with_room_for(index, natural, TAB_CHARACTERS),
                     natural.height())

    @guard(QSize(0, 0))
    def minimumTabSizeHint(self, index: int) -> QSize:  # noqa: N802 - Qt's name
        """Room for :data:`READABLE_CHARACTERS` of the name, not Qt's three."""
        if not self._fixed():
            return super().minimumTabSizeHint(index)
        # Not from Qt's minimum: Qt measures it by asking tabSizeHint with the
        # name cut to three characters, which is answered above — every tab
        # the same width whatever its name, and so no smaller at all.
        natural = super().tabSizeHint(index)
        return QSize(self._width_with_room_for(index, natural, READABLE_CHARACTERS),
                     natural.height())

    def _fixed(self) -> bool:
        """Whether tabs are given a width rather than their name's.

        Only while names are elided: a bar that does not elide has asked for
        every name whole, and a fixed width would cut them off instead.
        """
        return self.elideMode() != Qt.TextElideMode.ElideNone and self._horizontal()

    def _width_with_room_for(self, index: int, natural: QSize, characters: int) -> int:
        """``index``'s tab with room for ``characters`` in place of its name.

        Qt's own size is the name plus everything around it — the padding, the
        close button — so taking the name back off leaves what every tab
        carries, measured as Qt measured it.
        """
        metrics = self.fontMetrics()
        name = metrics.size(Qt.TextFlag.TextShowMnemonic, self.tabText(index)).width()
        return natural.width() - name + metrics.averageCharWidth() * characters

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


__all__ = ["READABLE_CHARACTERS", "SlidingTabBar", "TAB_CHARACTERS"]
