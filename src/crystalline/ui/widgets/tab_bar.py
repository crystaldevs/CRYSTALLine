"""A tab bar of fixed-width tabs: a row that stays put, and slides whole."""

from __future__ import annotations

from PySide6.QtCore import QPoint, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QInputDevice, QLinearGradient, QPainter, QPalette, QWheelEvent
from PySide6.QtWidgets import QTabBar, QToolButton, QWidget

from crystalline.ui.safety import guard

_SIDES = (QTabBar.ButtonPosition.LeftSide, QTabBar.ButtonPosition.RightSide)

# How much of a name every tab has room for, in average characters: a tab is
# this wide whatever its name, as Firefox's are, so the row does not move when
# the bar it sits in gets narrower or wider. A name-sized tab moves with every
# change — the file tabs would reflow each time the Phonons panel comes or goes.
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

# How far in from an edge a cut-off row fades out, in pixels: two close
# buttons, so a tab half out of view reads as going on past the edge.
FADE_WIDTH = 40
_SLACK = 2


class SlidingTabBar(QTabBar):
    """A :class:`QTabBar` of fixed-width tabs, for rows that change width.

    Four things QTabBar does badly for a row of open files:

    * Each tab is as wide as its name, so the row is laid out again whenever
      the bar's width changes, and every tab moves — including the one just
      clicked. Here every tab is :data:`TAB_CHARACTERS` wide, as in Firefox,
      and stays where it is for as long as the row fits.
    * It squeezes every name down to three characters before it starts to
      scroll, so the row reads "t…t  t…t  t…t". Here a tab keeps about
      :data:`READABLE_CHARACTERS` of its name, and the bar scrolls sooner.
    * A trackpad swipe across the bar scrolls it by repainting at the new
      offset and nothing more: the names and backgrounds slide along, while
      the close buttons — widgets of their own — stay where they are until
      something next lays the bar out, so each × ends up beside another file's
      name. Here every button is moved on with its tab, to the place Qt had
      put it.
    * A row that does not fit gets two arrow buttons at one end. Here the
      theme gives them no width, and each end where tabs are cut off fades out
      instead (:class:`_EdgeFade`). A click on a fade scrolls the row a tab
      that way, as the arrows do, without changing the tab shown; a swipe or
      the wheel, either way, scrolls it too.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._fades = (_EdgeFade(self, -1), _EdgeFade(self, 1))
        # Choosing a tab scrolls it into view: the fades follow at once.
        self.currentChanged.connect(self._place_fades)

    def scroll_by_tab(self, direction: int) -> None:
        """Bring the next tab to the left (``-1``) or right (``1``) into view.

        By pressing Qt's own arrow, which has no width here but scrolls just
        the same: a tab at a time, leaving the current tab as it is.
        """
        name = "ScrollLeftButton" if direction < 0 else "ScrollRightButton"
        arrow = self.findChild(QToolButton, name)
        if arrow is None or not arrow.isEnabled():
            return
        anchored = self._buttons_on_tabs()
        arrow.click()
        self._move_buttons(anchored)
        self._place_fades()

    def tabInserted(self, index: int) -> None:  # noqa: N802 - Qt's name
        super().tabInserted(index)
        self._place_fades()  # over the close button the new tab has just been given

    def tabRemoved(self, index: int) -> None:  # noqa: N802 - Qt's name
        super().tabRemoved(index)
        self._place_fades()

    def tabLayoutChange(self) -> None:  # noqa: N802 - Qt's name
        super().tabLayoutChange()
        self._place_fades()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt's name
        super().resizeEvent(event)
        self._place_fades()

    @guard()
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt's name
        super().paintEvent(event)
        # A repaint follows every scroll, whatever moved the row — choosing a
        # tab, a drag — so this is where a fade that no longer matches is seen.
        fades = getattr(self, "_fades", ())
        if fades and tuple(f.isVisibleTo(self) for f in fades) != self.hidden_ends():
            QTimer.singleShot(0, self, self._place_fades)

    def _place_fades(self) -> None:
        """Show a fade over each end that has tabs past it, and only there."""
        width = min(FADE_WIDTH, self.width() // 3)
        # getattr: QTabBar's own constructor may lay the bar out before the
        # fades exist.
        for fade, hidden in zip(getattr(self, "_fades", ()), self.hidden_ends()):
            x = 0 if fade.side < 0 else self.width() - width
            fade.setGeometry(x, 0, width, self.height())
            fade.setVisible(hidden)
            fade.raise_()  # over the close buttons, which are widgets too

    def hidden_ends(self) -> tuple:
        """``(left, right)``: whether tabs run past each end of the bar."""
        if self.count() == 0 or not self._horizontal():
            return False, False
        rects = [self.tabRect(i) for i in range(self.count())]
        rects = [r for r in rects if not r.isNull()]
        if not rects:
            return False, False
        # A pixel or two over is Qt's rounding, not a tab out of view: scrolled
        # to the end, the last tab overhangs the bar by one.
        return (min(r.left() for r in rects) < -_SLACK,
                max(r.right() for r in rects) >= self.width() + _SLACK)

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
        """Scroll the row, whichever way the wheel or the swipe goes.

        Qt scrolls a horizontal bar only for a sideways swipe, and turns a
        mouse wheel into a change of tab. Here an up-and-down one scrolls it
        sideways, and a mouse wheel scrolls it a tab a notch, as a click on a
        fade does: neither changes the tab shown.
        """
        if not self._horizontal():
            super().wheelEvent(event)
            return
        angle = event.angleDelta()
        along = angle.x() if abs(angle.x()) >= abs(angle.y()) else angle.y()
        device = event.pointingDevice()
        if device is None or not (device.capabilities()
                                  & QInputDevice.Capability.PixelScroll):
            if along:
                self.scroll_by_tab(-1 if along > 0 else 1)
            event.accept()
            return
        if abs(angle.y()) > abs(angle.x()):
            pixel = event.pixelDelta()
            event = QWheelEvent(event.position(), event.globalPosition(),
                                QPoint(pixel.y(), pixel.x()), QPoint(angle.y(), angle.x()),
                                event.buttons(), event.modifiers(), event.phase(),
                                event.inverted(), event.source(), device)
        anchored = self._buttons_on_tabs()
        super().wheelEvent(event)
        self._move_buttons(anchored)
        self._place_fades()

    def _move_buttons(self, anchored: list) -> None:
        """Put each button back on its tab, where :meth:`_buttons_on_tabs` found it."""
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


class _EdgeFade(QWidget):
    """The fade over one end of a :class:`SlidingTabBar` that has tabs past it.

    A widget of its own, not painted by the bar: the close buttons are widgets
    too, drawn after the bar paints, and one at the edge would stand out sharp
    over its faded tab. Over them, it also takes the click a half-hidden ×
    would otherwise get, and scrolls the row instead — a click at the edge of
    the row is a reach for what is past it, not for closing a file.
    """

    def __init__(self, bar: SlidingTabBar, side: int) -> None:
        super().__init__(bar)
        self.side = side  # -1 the left end, 1 the right
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.hide()

    @guard()
    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt's name
        if event.button() == Qt.MouseButton.LeftButton:
            self.parentWidget().scroll_by_tab(self.side)
        event.accept()

    # A quick second click is a second step, not a double click.
    mouseDoubleClickEvent = mousePressEvent

    @guard()
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt's name
        # The bar's ground, what the tabs sit on: the fade is the row going
        # under it, not a shadow laid over it.
        ground = self.parentWidget().palette().color(QPalette.ColorRole.Window)
        clear = QColor(ground)
        clear.setAlpha(0)
        edge, inside = (0, self.width()) if self.side < 0 else (self.width(), 0)
        gradient = QLinearGradient(edge, 0, inside, 0)
        # Nearly solid for the outer third, so what is cut off at the very edge
        # — often a × — is gone rather than half there, then easing to nothing.
        nearly = QColor(ground)
        nearly.setAlpha(225)
        gradient.setColorAt(0.0, ground)
        gradient.setColorAt(0.3, nearly)
        gradient.setColorAt(1.0, clear)
        painter = QPainter(self)
        painter.fillRect(self.rect(), gradient)
        painter.end()


__all__ = ["FADE_WIDTH", "READABLE_CHARACTERS", "SlidingTabBar", "TAB_CHARACTERS"]
