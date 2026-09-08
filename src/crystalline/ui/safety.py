"""Keep one failure from taking the whole window down with it.

Every crash report this app has produced is a ``SIGSEGV`` in native code, and
most of them share a frame: ``Shiboken::Errors::storePythonOverrideErrorOrPrint``,
under ``eventFilter`` or ``styleHint``. That is shiboken crashing *while
reporting* a Python exception that escaped a Qt virtual override — so the
segfault is not the disease. An ordinary Python error in an override is, and
the process dies for it with nothing on screen and nothing in the log that
names the actual mistake.

Hence three layers, for the three ways this app can die:

* :func:`guard` wraps a virtual override so an exception never reaches
  shiboken. This is the one that prevents crashes rather than merely recording
  them: Qt gets a valid return value, the window survives, and the user is told
  what failed instead of watching the app vanish.
* :func:`install` puts the same reporting behind ``sys.excepthook``, which
  covers slots, timers and queued calls. PySide already survives those, but it
  prints the traceback to a stderr nobody reads and leaves the app half-updated
  and silent.
* ``faulthandler`` writes a native traceback for the crashes that are genuinely
  native — VTK's Cocoa view outliving its window, say. Those cannot be caught
  from Python at all; the most that can be done is make the next one
  diagnosable instead of anonymous.

Reports are rate-limited by traceback signature. A ``paintEvent`` that raises
raises on every repaint, and a dialog per repaint would be worse than the crash
it replaced.
"""

from __future__ import annotations

import faulthandler
import os
import sys
import traceback
from functools import wraps
from pathlib import Path
from typing import Callable, Optional

# Past this the log is truncated on startup. The launcher's own stderr log
# reached 8.6 MB of repeated VTK warnings, which is a log nobody will open.
_MAX_LOG_BYTES = 2_000_000

# Failures already surfaced, by traceback signature, and how often each has
# fired since. A repeat is counted, not shown again.
_seen: dict = {}
# Raise instead of swallowing. Tests turn this on: a guard that quietly ate a
# real bug would make the suite pass over exactly the faults it exists to
# survive.
_strict = bool(os.environ.get("CRYSTALLINE_STRICT"))
_installed = False


def log_path() -> Path:
    """Where errors are written — the platform's own place for logs."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Logs" / "CRYSTALLine"
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "CRYSTALLine" / "Logs"
    else:
        base = Path(
            os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state"
        ) / "CRYSTALLine"
    return base / "errors.log"


def strict(enabled: bool) -> None:
    """Make :func:`guard` re-raise instead of swallowing (used by the tests)."""
    global _strict
    _strict = bool(enabled)


def install(app=None) -> None:
    """Put the net up: the excepthook, the native handler and the log file.

    Safe to call twice, and safe to call with no ``app`` — everything degrades
    to writing the log, which is what a headless or test run wants anyway.
    """
    global _installed
    if _installed:
        return
    _installed = True

    path = log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > _MAX_LOG_BYTES:
            path.unlink()
        # Held open for the life of the process: faulthandler writes to this
        # descriptor from a signal handler, where reopening a file is not an
        # option. Deliberately never closed.
        handle = open(path, "a", buffering=1)
        faulthandler.enable(handle)
    except OSError:
        pass  # an unwritable log must not stop the app starting

    previous = sys.excepthook

    def hook(kind, value, tb) -> None:
        report("an unexpected error", exc=(kind, value, tb))
        if previous is not None:  # keep whatever was there (pytest, an IDE)
            try:
                previous(kind, value, tb)
            except Exception:  # noqa: BLE001 - reporting must not raise
                pass

    sys.excepthook = hook


def guard(default=None, what: Optional[str] = None) -> Callable:
    """Wrap a Qt virtual override so an exception cannot escape into shiboken.

    ``default`` is what Qt gets if the body fails, and it has to be a value Qt
    can use: ``False`` for an event filter, a real ``QSize`` for a size hint.
    ``None`` suits the handlers that return nothing.

    The window carries on with one thing not done — an unpainted widget, an
    ignored drop — which is a far better outcome than the process disappearing
    mid-session with unsaved edits in it.
    """

    def decorate(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            try:
                return fn(self, *args, **kwargs)
            except Exception:  # noqa: BLE001 - the whole point
                if _strict:
                    raise
                report(what or f"{type(self).__name__}.{fn.__name__}")
                return default

        return wrapper

    return decorate


def report(context: str, exc=None) -> bool:
    """Log a failure and, the first time each one happens, show it.

    Returns whether it was shown. Later repeats of the same traceback are
    counted in the log and nowhere else.
    """
    kind, value, tb = exc or sys.exc_info()
    if value is None:
        return False
    text = "".join(traceback.format_exception(kind, value, tb))
    signature = (context, text.splitlines()[-1] if text else "")
    first = signature not in _seen
    _seen[signature] = _seen.get(signature, 0) + 1

    _write(context, text, _seen[signature])
    if first:
        _show(context, value, text)
    return first


def _write(context: str, text: str, count: int) -> None:
    from datetime import datetime

    try:
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as handle:
            stamp = datetime.now().isoformat(timespec="seconds")
            repeat = "" if count == 1 else f"  (repeat {count})"
            handle.write(f"\n=== {stamp}  {context}{repeat} ===\n{text}")
    except OSError:
        pass


def _show(context: str, value: BaseException, text: str) -> None:
    """Offer the failure to the user, on the next turn of the event loop.

    Never immediately: this is routinely called from inside a paint or an event
    filter, and opening a dialog there re-enters Qt at the worst possible
    moment — which is the same mistake that made a dropped file segfault.
    """
    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication
    except Exception:  # noqa: BLE001 - no Qt: the log is the whole report
        return
    if QApplication.instance() is None:
        return
    QTimer.singleShot(0, lambda: _dialog(context, value, text))


def _dialog(context: str, value: BaseException, text: str) -> None:
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox

        box = QMessageBox(QApplication.activeWindow())
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Something went wrong")
        box.setText(f"CRYSTALLine hit a problem in {context}.")
        box.setInformativeText(
            f"{type(value).__name__}: {value}\n\n"
            "The window is still usable and nothing has been lost, but whatever "
            "was being done just then did not finish.\n\n"
            f"Details were written to\n{log_path()}"
        )
        box.setDetailedText(text)
        box.setStandardButtons(QMessageBox.Ok)
        box.setModal(False)  # the app stays usable behind it
        box.show()
    except Exception:  # noqa: BLE001 - a failed report must not start a loop
        pass


__all__ = ["guard", "install", "log_path", "report", "strict"]
