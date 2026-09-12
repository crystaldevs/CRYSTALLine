"""The safety net: a failure in a Qt override must not take the window down.

Every crash report this app has produced is a SIGSEGV, and most share the frame
``Shiboken::Errors::storePythonOverrideErrorOrPrint`` under ``eventFilter`` or
``styleHint`` — shiboken crashing while *reporting* a Python exception that
escaped an override. So the exception is the fault and the segfault is the
symptom, and catching the former is what stops the latter.
"""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSize  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from crystalline.ui import safety  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _lenient():
    """These tests are about the net, so here the guards must actually catch.

    conftest puts the suite in strict mode, where a guard re-raises; that is
    right everywhere else and would defeat every assertion below.
    """
    safety.strict(False)
    safety._seen.clear()
    yield
    safety.strict(True)


# ── guard ─────────────────────────────────────────────────────────────────
def test_a_failing_override_returns_a_usable_value_instead_of_raising():
    """Qt gets a valid answer and carries on. Letting the exception through is
    what segfaults, so what matters is that nothing escapes."""

    class Widget:
        @safety.guard(False)
        def eventFilter(self, obj, event):
            raise ValueError("boom")

        @safety.guard(QSize(0, 0))
        def sizeHint(self):
            raise ValueError("boom")

        @safety.guard()
        def paintEvent(self, event):
            raise ValueError("boom")

    widget = Widget()
    assert widget.eventFilter(None, None) is False
    assert widget.sizeHint() == QSize(0, 0)
    assert widget.paintEvent(None) is None


def test_a_working_override_is_untouched():
    class Widget:
        @safety.guard(False)
        def eventFilter(self, obj, event):
            return event == "mine"

    assert Widget().eventFilter(None, "mine") is True
    assert Widget().eventFilter(None, "theirs") is False


def test_the_guard_keeps_the_wrapped_name():
    """Qt dispatches virtual overrides by name; a decorator that renamed them
    would silently unhook every one of them."""

    class Widget:
        @safety.guard(False)
        def eventFilter(self, obj, event):
            return False

    assert Widget.eventFilter.__name__ == "eventFilter"


def test_strict_mode_re_raises():
    """What the suite runs in, so a guard cannot hide a genuine bug."""

    class Widget:
        @safety.guard(False)
        def paintEvent(self, event):
            raise ValueError("boom")

    safety.strict(True)
    try:
        with pytest.raises(ValueError):
            Widget().paintEvent(None)
    finally:
        safety.strict(False)


# ── reporting ─────────────────────────────────────────────────────────────
def test_the_same_failure_is_shown_once_and_counted_after(tmp_path, monkeypatch):
    """A paintEvent that raises raises on every repaint. One dialog per repaint
    would be worse than the crash it replaced."""
    log = tmp_path / "errors.log"
    monkeypatch.setattr(safety, "log_path", lambda: log)
    shown = []
    monkeypatch.setattr(safety, "_show", lambda *a: shown.append(a))

    class Widget:
        @safety.guard()
        def paintEvent(self, event):
            raise ValueError("boom")

    widget = Widget()
    for _ in range(5):
        widget.paintEvent(None)

    assert len(shown) == 1                      # surfaced once
    text = log.read_text()
    assert text.count("Widget.paintEvent") == 5  # logged every time
    assert "repeat 5" in text                    # and the repeats are counted


def test_two_different_failures_are_both_shown(tmp_path, monkeypatch):
    monkeypatch.setattr(safety, "log_path", lambda: tmp_path / "errors.log")
    shown = []
    monkeypatch.setattr(safety, "_show", lambda *a: shown.append(a))

    class Widget:
        @safety.guard()
        def paintEvent(self, event):
            raise ValueError("one")

        @safety.guard()
        def resizeEvent(self, event):
            raise TypeError("two")

    widget = Widget()
    widget.paintEvent(None)
    widget.resizeEvent(None)
    assert len(shown) == 2


def test_reporting_survives_an_unwritable_log(tmp_path, monkeypatch):
    """The report is the fallback; it must not become the failure."""
    monkeypatch.setattr(safety, "log_path", lambda: tmp_path / "nope" / "x" / "errors.log")
    monkeypatch.setattr(safety, "_show", lambda *a: None)
    monkeypatch.setattr(
        safety.Path, "mkdir", lambda *a, **k: (_ for _ in ()).throw(OSError("read-only"))
    )

    class Widget:
        @safety.guard()
        def paintEvent(self, event):
            raise ValueError("boom")

    Widget().paintEvent(None)  # must not raise


def test_report_with_no_exception_in_flight_does_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(safety, "log_path", lambda: tmp_path / "errors.log")
    assert safety.report("nothing happened") is False


# ── installation ──────────────────────────────────────────────────────────
def test_install_is_idempotent_and_hooks_the_excepthook(monkeypatch, tmp_path):
    monkeypatch.setattr(safety, "log_path", lambda: tmp_path / "errors.log")
    monkeypatch.setattr(safety, "_installed", False)
    original = __import__("sys").excepthook

    safety.install()
    hooked = __import__("sys").excepthook
    assert hooked is not original

    safety.install()  # a second call must not stack another layer
    assert __import__("sys").excepthook is hooked

    __import__("sys").excepthook = original


def test_the_excepthook_reports_and_still_chains(monkeypatch, tmp_path):
    """Whatever was there before — pytest's, an IDE's — keeps working."""
    import sys

    monkeypatch.setattr(safety, "log_path", lambda: tmp_path / "errors.log")
    monkeypatch.setattr(safety, "_installed", False)
    monkeypatch.setattr(safety, "_show", lambda *a: None)
    chained = []
    original = sys.excepthook
    sys.excepthook = lambda *a: chained.append(a)
    try:
        safety.install()
        sys.excepthook(ValueError, ValueError("boom"), None)
        assert chained, "the previous excepthook was dropped"
        assert (tmp_path / "errors.log").read_text().count("ValueError") >= 1
    finally:
        sys.excepthook = original


def test_the_log_lives_somewhere_the_platform_expects():
    path = safety.log_path()
    assert path.name == "errors.log"
    assert "CRYSTALLine" in str(path)


# ── the real widgets are guarded ──────────────────────────────────────────
def test_the_app_s_own_overrides_are_guarded(qapp):
    """The net is only up where it is applied. These are the overrides that
    appear in the crash reports, or would if they failed."""
    from crystalline.ui.main_window import MainWindow
    from crystalline.ui.viewport import Viewport
    from crystalline.ui.widgets import BusyOverlay, DropHint, RangeSlider, ToggleSwitch

    expected = {
        Viewport: ["eventFilter"],
        MainWindow: ["dragEnterEvent", "dragMoveEvent", "dragLeaveEvent", "dropEvent"],
        BusyOverlay: ["eventFilter", "paintEvent"],
        DropHint: ["eventFilter", "paintEvent"],
        ToggleSwitch: ["sizeHint", "paintEvent"],
        RangeSlider: ["sizeHint", "minimumSizeHint", "paintEvent",
                      "mousePressEvent", "mouseMoveEvent", "mouseReleaseEvent"],
    }
    for owner, names in expected.items():
        for name in names:
            fn = owner.__dict__.get(name)
            assert fn is not None, f"{owner.__name__}.{name} is not defined here"
            assert getattr(fn, "__wrapped__", None) is not None, (
                f"{owner.__name__}.{name} is not guarded — an exception there "
                f"segfaults the app rather than raising"
            )


def test_a_broken_paint_does_not_kill_the_widget(qapp, monkeypatch, tmp_path):
    """End to end on a real widget. This is the shape of the crash: a paint that
    raises, on a widget Qt is driving. Guarded, the widget stays alive and
    repaintable and the failure is reported instead."""
    from crystalline.ui.widgets import ToggleSwitch

    log = tmp_path / "errors.log"
    monkeypatch.setattr(safety, "log_path", lambda: log)
    shown = []
    monkeypatch.setattr(safety, "_show", lambda *a: shown.append(a))

    host = QWidget()
    switch = ToggleSwitch(host)
    switch.resize(60, 30)
    assert switch.grab().size().isValid()      # paints cleanly to begin with

    # break something paintEvent actually reaches, and check it really fired
    monkeypatch.setattr(
        type(switch), "isEnabled",
        lambda self: (_ for _ in ()).throw(ZeroDivisionError("boom")),
    )
    switch.grab()                              # must not raise
    switch.grab()                              # ...and the widget is still usable
    assert log.read_text().count("ZeroDivisionError") >= 2, "the sabotage never fired"
    assert len(shown) == 1                     # reported once, not once per repaint
