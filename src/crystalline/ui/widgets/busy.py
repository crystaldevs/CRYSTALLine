"""A modern busy indicator, and a way to run work without freezing behind it.

The app used to mark long operations with ``Qt.WaitCursor`` — the platform's
oldest idiom, a spinning disc or an hourglass on the pointer, that says only
"wait" and says it in the corner of the eye.

The two halves here go together on purpose. An animated indicator over work that
blocks the main thread is *worse* than a cursor: the animation freezes with
everything else, and a stopped spinner reads as a crashed app. So the overlay
comes with :class:`Worker`, which puts the work on a thread and leaves the event
loop free to actually draw it.
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import (
    QObject,
    QRectF,
    QThread,
    QTimer,
    Qt,
    Signal,
)
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from crystalline.ui.safety import guard

# The arc sweeps rather than a set of fading dots: one moving shape reads as
# progress at any size, and costs one repaint of a small rectangle.
_SPINNER_SIZE = 34
_SPINNER_WIDTH = 3
_ARC_SPAN = 100          # degrees of the circle the arc covers
_DEGREES_PER_TICK = 9
_TICK_MS = 16            # ~60 fps: the indicator is the one thing that must be smooth
_SCRIM_ALPHA = 150


class BusyOverlay(QWidget):
    """A translucent scrim over its parent, with a spinner and a line of text.

    Sizes itself to the parent and follows it, so a caller only has to
    :meth:`start` and :meth:`stop` it.
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._angle = 0
        self._message = ""
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)  # swallow clicks
        self.setVisible(False)
        self._timer = QTimer(self)
        self._timer.setInterval(_TICK_MS)
        self._timer.timeout.connect(self._advance)
        parent.installEventFilter(self)

    # ── lifecycle ───────────────────────────────────────────────────────
    def start(self, message: str = "") -> None:
        self._message = message
        self._angle = 0
        self._fit()
        self.raise_()
        self.setVisible(True)
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
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

    def _advance(self) -> None:
        self._angle = (self._angle + _DEGREES_PER_TICK) % 360
        self.update()

    # ── painting ────────────────────────────────────────────────────────
    @guard()
    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        palette = self.palette()

        scrim = QColor(palette.window().color())
        scrim.setAlpha(_SCRIM_ALPHA)
        painter.fillRect(self.rect(), scrim)

        centre = self.rect().center()
        box = QRectF(
            centre.x() - _SPINNER_SIZE / 2.0,
            centre.y() - _SPINNER_SIZE / 2.0 - (10 if self._message else 0),
            float(_SPINNER_SIZE), float(_SPINNER_SIZE),
        )

        track = QColor(palette.mid().color())
        track.setAlpha(90)
        painter.setPen(QPen(track, _SPINNER_WIDTH, Qt.SolidLine, Qt.RoundCap))
        painter.drawArc(box, 0, 360 * 16)

        painter.setPen(QPen(palette.highlight().color(), _SPINNER_WIDTH,
                            Qt.SolidLine, Qt.RoundCap))
        # Qt measures arcs in sixteenths of a degree, anticlockwise from 3 o'clock.
        painter.drawArc(box, -self._angle * 16, -_ARC_SPAN * 16)

        if self._message:
            painter.setPen(palette.windowText().color())
            painter.drawText(
                self.rect().adjusted(0, _SPINNER_SIZE + 18, 0, 0),
                Qt.AlignHCenter | Qt.AlignVCenter,
                self._message,
            )
        painter.end()


class _Runner(QObject):
    """The callable, living on the worker thread."""

    done = Signal(object)
    error = Signal(object)

    def __init__(self, work: Callable[[], object]) -> None:
        super().__init__()
        self._work = work

    def run(self) -> None:
        try:
            result = self._work()
        except Exception as exc:  # noqa: BLE001 - reported to the caller, not swallowed
            self.error.emit(exc)
        else:
            self.done.emit(result)


class Worker(QObject):
    """Run one callable on a thread and deliver its result on the UI thread.

    Only what is safe off-thread should be given to it: parsing, numpy, pymatgen.
    Rendering is not — VTK and Qt widgets belong to the thread that made them —
    so a caller does the drawing in the ``finished`` handler, back on the main
    thread.

    The split between this and :class:`_Runner` is what makes that true. Moving
    *this* object to the thread would put its signals there too, and a handler
    connected as a bare callable has no receiver whose thread Qt can queue to —
    so it would run on the worker thread, and the first Qt or VTK call in it
    would be a crash waiting for a busy machine. The runner moves; this stays,
    and Qt queues the runner's signals to it because it is a QObject that lives
    here.
    """

    finished = Signal(object)
    failed = Signal(object)

    def __init__(self, work: Callable[[], object], parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._thread = QThread()
        self._runner = _Runner(work)
        self._runner.moveToThread(self._thread)
        self._thread.started.connect(self._runner.run)
        # Auto-connected to slots on *this* object, which lives on the UI thread,
        # so Qt queues them there.
        self._runner.done.connect(self._on_done)
        self._runner.error.connect(self._on_error)

    def start(self) -> None:
        self._thread.start()

    def _on_done(self, result) -> None:
        self._thread.quit()
        self._thread.wait()
        self.finished.emit(result)

    def _on_error(self, exc) -> None:
        self._thread.quit()
        self._thread.wait()
        self.failed.emit(exc)


__all__ = ["BusyOverlay", "Worker"]
