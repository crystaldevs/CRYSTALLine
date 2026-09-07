"""A sliding on/off switch.

Qt has no switch widget, and the usual substitute — a checkable button that
fills with colour when on — has to carry a label saying what "on" means, which
is why it grows into a wide slab. A switch says on and off by its shape, so the
label beside it can be a noun instead of a sentence.

Painted rather than styled: a stylesheet can round a button and colour it, but
it cannot move a knob from one end of a track to the other.
"""

from __future__ import annotations

from PySide6.QtCore import (
    Property,
    QAbstractAnimation,
    QEasingCurve,
    QPropertyAnimation,
    QSize,
    Qt,
)
from PySide6.QtGui import QColor, QPainter, QPainterPath
from PySide6.QtWidgets import QAbstractButton, QSizePolicy

# The track, and the knob that runs along it. Sized to sit on a toolbar row
# without setting the row's height.
_TRACK_WIDTH = 36
_TRACK_HEIGHT = 20
_KNOB_MARGIN = 2
_SLIDE_MS = 130


class ToggleSwitch(QAbstractButton):
    """A checkable switch: a track, and a knob that slides across it.

    Colours come from the widget's palette rather than from a stylesheet, so it
    follows a theme change with everything else — ``Highlight`` when on, the
    window's mid-tones when off.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setFocusPolicy(Qt.TabFocus)
        self._offset = 0.0  # 0 = off (knob left), 1 = on (knob right)
        self._slide = QPropertyAnimation(self, b"offset", self)
        self._slide.setDuration(_SLIDE_MS)
        self._slide.setEasingCurve(QEasingCurve.OutCubic)
        self.toggled.connect(self._animate)

    # ── geometry ────────────────────────────────────────────────────────
    def sizeHint(self) -> QSize:
        return QSize(_TRACK_WIDTH, _TRACK_HEIGHT)

    minimumSizeHint = sizeHint

    # ── the animated property ───────────────────────────────────────────
    def _get_offset(self) -> float:
        return self._offset

    def _set_offset(self, value: float) -> None:
        self._offset = float(value)
        self.update()

    offset = Property(float, _get_offset, _set_offset)

    def _animate(self, checked: bool) -> None:
        target = 1.0 if checked else 0.0
        self._slide.stop()
        if not self.isVisible():
            # Nothing to animate *into*: a switch built or synced before it is on
            # screen should simply start in its state. Sliding there would also
            # mean the first paint catches the knob mid-travel, so a control set
            # on at construction draws as off.
            self._set_offset(target)
            return
        self._slide.setStartValue(self._offset)
        self._slide.setEndValue(target)
        self._slide.start()

    def setChecked(self, checked: bool) -> None:  # noqa: N802 - Qt's spelling
        """Set the state, and put the knob where it belongs.

        Overridden because a programmatic change with signals blocked — how a
        view syncs a control to the state it displays — never reaches the
        animation, and the knob would sit at the wrong end while the switch
        claimed to be on.
        """
        super().setChecked(checked)
        target = 1.0 if checked else 0.0
        # A programmatic change with signals blocked — how a view syncs a control
        # to the state it displays — never reaches _animate, so place the knob.
        #
        # Compared against Running explicitly: a PySide6 enum is an object, so
        # ``not self._slide.state()`` is False even for Stopped, and the knob was
        # left behind on every blocked sync.
        running = self._slide.state() == QAbstractAnimation.State.Running
        if self._offset != target and not running:
            self._set_offset(target)

    # ── painting ────────────────────────────────────────────────────────
    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        palette = self.palette()

        height = _TRACK_HEIGHT
        width = _TRACK_WIDTH
        radius = height / 2.0
        enabled = self.isEnabled()

        on = QColor(palette.highlight().color())
        off = QColor(palette.mid().color())
        if not enabled:
            on.setAlpha(90)
            off.setAlpha(90)
        # Blend between off and on with the knob, so the track fills as it slides
        # rather than snapping colour at the end of the travel.
        track = QColor(
            int(off.red() + (on.red() - off.red()) * self._offset),
            int(off.green() + (on.green() - off.green()) * self._offset),
            int(off.blue() + (on.blue() - off.blue()) * self._offset),
            int(off.alpha() + (on.alpha() - off.alpha()) * self._offset),
        )

        path = QPainterPath()
        path.addRoundedRect(0.0, 0.0, float(width), float(height), radius, radius)
        painter.fillPath(path, track)

        knob_size = height - 2 * _KNOB_MARGIN
        travel = width - knob_size - 2 * _KNOB_MARGIN
        x = _KNOB_MARGIN + travel * self._offset
        knob = QColor(palette.brightText().color() if False else Qt.white)
        if not enabled:
            knob.setAlpha(150)
        painter.setPen(Qt.NoPen)
        painter.setBrush(knob)
        painter.drawEllipse(x, float(_KNOB_MARGIN), float(knob_size), float(knob_size))
        painter.end()


__all__ = ["ToggleSwitch"]
