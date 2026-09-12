"""The panel shown over the 3D view while a file is being dragged onto it.

Drag-and-drop has no discoverable affordance of its own: the only feedback the
platform gives is a cursor badge, which says a drop is possible but not what it
would do. Opening a file and importing atoms into one are different enough —
one replaces the structure, the other appends to it and lands in the undo
history — that saying which is about to happen is worth a panel.

It is inert: transparent to the pointer and refusing drops itself, so it never
becomes the drop target it is describing.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QWidget

from crystalline.ui.safety import guard

# The dashed frame sits in from the edge so it reads as a target rather than as
# a border on the viewport.
_INSET = 18
_RADIUS = 14
_BORDER_WIDTH = 2
_DASH = (6.0, 4.0)
# Faint: the structure underneath stays legible, and the panel is only ever up
# for as long as a pointer is held over the window.
_SCRIM_ALPHA = 40
_TINT_ALPHA = 26

# The text sits on a solid card rather than straight on the tint. The 3D
# background is a setting — it can be white, near-black, or anything someone
# picked — so text painted over it directly is legible on some scenes and not
# on others. The card carries its own contrast.
_CARD_ALPHA = 242
_CARD_RADIUS = 10
_CARD_MAX_WIDTH = 460
_CARD_PADDING_X = 22
_CARD_PADDING_Y = 16
_TITLE_POINT_ADD = 2
_LINE_GAP = 8


class DropHint(QWidget):
    """A translucent "drop it here" panel over its parent (hidden by default)."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._title = ""
        self._detail = ""
        # Never a drop target itself: with either of these wrong the panel would
        # appear under the pointer and take the drag away from the widget it is
        # advertising, which reads as the drop silently failing.
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAcceptDrops(False)
        self.setVisible(False)
        parent.installEventFilter(self)

    def show_hint(self, title: str, detail: str = "") -> None:
        """Put the panel up, naming what the drop would do."""
        self._title, self._detail = title, detail
        self._fit()
        self.raise_()
        self.setVisible(True)
        self.update()

    def hide_hint(self) -> None:
        self.setVisible(False)

    @guard(False)
    def eventFilter(self, obj, event):
        if obj is self.parent() and event.type() in (event.Type.Resize, event.Type.Show):
            self._fit()
        return super().eventFilter(obj, event)

    def _fit(self) -> None:
        parent = self.parentWidget()
        if parent is not None:
            self.setGeometry(parent.rect())

    def _fonts(self):
        title = QFont(self.font())
        title.setPointSize(self.font().pointSize() + _TITLE_POINT_ADD)
        title.setBold(True)
        return title, QFont(self.font())

    @guard()
    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        palette = self.palette()
        accent = palette.highlight().color()

        scrim = QColor(palette.window().color())
        scrim.setAlpha(_SCRIM_ALPHA)
        painter.fillRect(self.rect(), scrim)

        frame = QRectF(self.rect()).adjusted(_INSET, _INSET, -_INSET, -_INSET)
        tint = QColor(accent)
        tint.setAlpha(_TINT_ALPHA)
        painter.setBrush(tint)
        pen = QPen(accent, _BORDER_WIDTH)
        pen.setStyle(Qt.CustomDashLine)
        pen.setDashPattern(list(_DASH))
        painter.setPen(pen)
        painter.drawRoundedRect(frame, _RADIUS, _RADIUS)

        title_font, detail_font = self._fonts()
        title_metrics, detail_metrics = QFontMetrics(title_font), QFontMetrics(detail_font)
        # A CRYSTAL output's name is routinely sixty characters of run
        # parameters, which is far wider than any sensible card. Elided in the
        # middle, so the stem and the extension — the two halves that say which
        # file this is — both survive.
        limit = min(_CARD_MAX_WIDTH, max(120, self.width() - 2 * (_INSET + _CARD_PADDING_X)))
        title = title_metrics.elidedText(self._title, Qt.ElideMiddle, limit)
        detail = detail_metrics.elidedText(self._detail, Qt.ElideRight, limit)

        text_width = max(title_metrics.horizontalAdvance(title),
                         detail_metrics.horizontalAdvance(detail))
        text_height = title_metrics.height() + (
            _LINE_GAP + detail_metrics.height() if detail else 0
        )
        card = QRectF(
            0.0, 0.0,
            text_width + 2 * _CARD_PADDING_X,
            text_height + 2 * _CARD_PADDING_Y,
        )
        card.moveCenter(QRectF(self.rect()).center())

        surface = QColor(palette.window().color())
        surface.setAlpha(_CARD_ALPHA)
        painter.setBrush(surface)
        painter.setPen(QPen(accent, 1))
        painter.drawRoundedRect(card, _CARD_RADIUS, _CARD_RADIUS)

        line = card.adjusted(_CARD_PADDING_X, _CARD_PADDING_Y,
                             -_CARD_PADDING_X, -_CARD_PADDING_Y)
        painter.setFont(title_font)
        painter.setPen(palette.windowText().color())
        painter.drawText(
            line.adjusted(0, 0, 0, -(text_height - title_metrics.height())),
            Qt.AlignHCenter | Qt.AlignTop, title,
        )
        if detail:
            painter.setFont(detail_font)
            painter.setPen(palette.mid().color())
            painter.drawText(line, Qt.AlignHCenter | Qt.AlignBottom, detail)
        painter.end()


__all__ = ["DropHint"]
