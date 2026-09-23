"""A slider with two handles: a span, not a point.

Qt has one handle per slider, so a range has to be spelled as two separate
controls — a "from" and a "to" that know nothing about each other, can be put
the wrong way round, and show nothing about how much of the whole they cover.
One track with two handles says all of that at a glance.

Painted rather than styled, for the same reason as the toggle: a stylesheet can
colour a groove but cannot put a second knob on it.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath
from PySide6.QtWidgets import QSizePolicy, QWidget

from crystalline.ui.safety import guard

_TRACK_HEIGHT = 4
_KNOB_RADIUS = 7
_MIN_WIDTH = 90


class RangeSlider(QWidget):
    """Select a low/high span over ``[minimum, maximum]``.

    ``valuesChanged`` carries ``(low, high)`` and fires while dragging, so a
    caller can track it live; ``rangeReleased`` fires once the handle is let go,
    for work too expensive to do per pixel.
    """

    valuesChanged = Signal(float, float)
    rangeReleased = Signal(float, float)

    def __init__(self, minimum: float = 0.0, maximum: float = 1.0, parent=None) -> None:
        super().__init__(parent)
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self._low = self._minimum
        self._high = self._maximum
        self._dragging: int = 0  # 0 none, 1 low, 2 high
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumWidth(_MIN_WIDTH)
        self.setCursor(Qt.PointingHandCursor)

    # ── state ───────────────────────────────────────────────────────────
    @guard(QSize(0, 0))
    def sizeHint(self) -> QSize:
        return QSize(160, 2 * _KNOB_RADIUS + 4)

    @guard(QSize(0, 0))
    def minimumSizeHint(self) -> QSize:
        return QSize(_MIN_WIDTH, 2 * _KNOB_RADIUS + 4)

    def bounds(self) -> tuple:
        return (self._minimum, self._maximum)

    def setBounds(self, minimum: float, maximum: float) -> None:  # noqa: N802 - Qt spelling
        """Change the span the track covers, keeping the selection inside it."""
        if maximum <= minimum:
            return
        self._minimum, self._maximum = float(minimum), float(maximum)
        self.setValues(self._low, self._high)

    def values(self) -> tuple:
        return (self._low, self._high)

    def setValues(self, low: float, high: float) -> None:  # noqa: N802 - Qt spelling
        """Set both handles at once, ordered and clamped to the bounds.

        Ordered rather than rejected: a caller that hands them over the wrong way
        round means the span between them, and swapping is the reading that
        cannot surprise anyone.
        """
        low, high = (float(low), float(high)) if low <= high else (float(high), float(low))
        self._low = min(max(low, self._minimum), self._maximum)
        self._high = min(max(high, self._minimum), self._maximum)
        self.update()

    # ── geometry ────────────────────────────────────────────────────────
    def _span(self) -> float:
        return max(self._maximum - self._minimum, 1e-12)

    def _x_of(self, value: float) -> float:
        usable = max(self.width() - 2 * _KNOB_RADIUS, 1)
        return _KNOB_RADIUS + (value - self._minimum) / self._span() * usable

    def _value_of(self, x: float) -> float:
        usable = max(self.width() - 2 * _KNOB_RADIUS, 1)
        fraction = (x - _KNOB_RADIUS) / usable
        return self._minimum + min(max(fraction, 0.0), 1.0) * self._span()

    # ── interaction ─────────────────────────────────────────────────────
    @guard()
    def mousePressEvent(self, event) -> None:
        x = event.position().x()
        # Grab whichever handle is nearer, so a click anywhere on the track pulls
        # the sensible one rather than always the same end.
        self._dragging = 1 if abs(x - self._x_of(self._low)) <= abs(x - self._x_of(self._high)) else 2
        self._drag_to(x)

    @guard()
    def mouseMoveEvent(self, event) -> None:
        if self._dragging:
            self._drag_to(event.position().x())

    @guard()
    def mouseReleaseEvent(self, _event) -> None:
        if self._dragging:
            self._dragging = 0
            self.rangeReleased.emit(self._low, self._high)

    def _drag_to(self, x: float) -> None:
        value = self._value_of(x)
        if self._dragging == 1:
            self._low = min(value, self._high)
        else:
            self._high = max(value, self._low)
        self.update()
        self.valuesChanged.emit(self._low, self._high)

    # ── painting ────────────────────────────────────────────────────────
    @guard()
    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        palette = self.palette()
        enabled = self.isEnabled()

        middle = self.height() / 2.0
        track = QColor(palette.mid().color())
        selected = QColor(palette.highlight().color())
        if not enabled:
            track.setAlpha(90)
            selected.setAlpha(90)

        path = QPainterPath()
        path.addRoundedRect(
            float(_KNOB_RADIUS), middle - _TRACK_HEIGHT / 2.0,
            float(max(self.width() - 2 * _KNOB_RADIUS, 1)), float(_TRACK_HEIGHT),
            _TRACK_HEIGHT / 2.0, _TRACK_HEIGHT / 2.0,
        )
        painter.fillPath(path, track)

        low_x, high_x = self._x_of(self._low), self._x_of(self._high)
        span = QPainterPath()
        span.addRoundedRect(
            low_x, middle - _TRACK_HEIGHT / 2.0, max(high_x - low_x, 1.0),
            float(_TRACK_HEIGHT), _TRACK_HEIGHT / 2.0, _TRACK_HEIGHT / 2.0,
        )
        painter.fillPath(span, selected)

        knob = QColor(palette.base().color())
        edge = QColor(palette.mid().color()) if not enabled else selected
        painter.setBrush(knob)
        for x in (low_x, high_x):
            painter.setPen(edge)
            painter.drawEllipse(
                x - _KNOB_RADIUS, middle - _KNOB_RADIUS,
                2.0 * _KNOB_RADIUS, 2.0 * _KNOB_RADIUS,
            )
        painter.end()


__all__ = ["RangeSlider"]
