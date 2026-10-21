"""``crystalline --check``: the one thing a stuck user can run that talks back.

The installed command is a ``gui_scripts`` entry, which on Windows runs under
``pythonw`` with no console — a startup failure there prints to nothing at all.
"""

import sys

import pytest

from crystalline import diagnose
from crystalline.__main__ import main


def test_every_check_reports_a_label_a_verdict_and_a_reason():
    results = diagnose.checks()
    assert results
    for label, ok, detail in results:
        assert label and isinstance(label, str)
        assert ok in (True, False, None)
        assert detail, f"{label} gave a verdict with nothing to act on"


def test_an_unsupported_python_is_named_with_what_to_do(monkeypatch):
    monkeypatch.setattr(diagnose.sys, "version_info", (3, 9, 23, "final", 0))
    label, ok, detail = diagnose._python()
    assert ok is False
    assert "3.11" in detail and "environment" in detail


def test_a_supported_python_passes():
    label, ok, detail = diagnose._python()
    assert ok is (sys.version_info[:2] >= diagnose.MINIMUM_PYTHON)


def test_a_dependency_that_will_not_import_says_which_and_why(monkeypatch):
    import importlib

    real = importlib.import_module

    def fail(name, *a, **k):
        if name == "spglib":
            raise ImportError("libsymspg.so: cannot open shared object file")
        return real(name, *a, **k)

    monkeypatch.setattr(importlib, "import_module", fail)
    results = {label: (ok, detail) for label, ok, detail in diagnose._packages()}
    ok, detail = results["spglib"]
    assert ok is False
    assert "libsymspg" in detail          # the real reason, not a generic one
    assert "space groups" in detail       # ...and what it costs the user


def test_a_crystalclear_without_the_optional_apis_is_a_note_not_a_failure(monkeypatch):
    """Stock 0.2.15 has no get_ADP. The app runs; thermal ellipsoids just never
    appear, which looks like a CRYSTALLine bug unless something says otherwise."""
    pytest.importorskip("CRYSTALClear")
    from CRYSTALClear.crystal_io import Crystal_output

    monkeypatch.delattr(Crystal_output, "get_ADP", raising=False)
    label, ok, detail = diagnose._crystalclear_features()
    assert ok is None                      # a note: not fatal
    assert "get_ADP" in detail
    assert "pip install -U CRYSTALClear" in detail


def test_a_headless_linux_session_is_told_what_it_actually_needs(monkeypatch):
    """Qt's own message tells the user to reinstall the application, which does
    not help on a machine with no display."""
    monkeypatch.setattr(diagnose.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    label, ok, detail = diagnose._display()
    assert ok is False
    assert "ssh -X" in detail and "reinstall" not in detail.lower()


def test_report_returns_non_zero_when_something_essential_failed(monkeypatch, capsys):
    monkeypatch.setattr(diagnose, "checks",
                        lambda: [("Python", True, "fine"), ("spglib", False, "broken")])
    assert diagnose.report() == 1
    out = capsys.readouterr().out
    assert "[FAIL]" in out and "1 check(s) failed" in out


def test_report_returns_zero_when_only_notes(monkeypatch, capsys):
    monkeypatch.setattr(diagnose, "checks",
                        lambda: [("Python", True, "fine"), ("x", None, "optional")])
    assert diagnose.report() == 0
    assert "All good" in capsys.readouterr().out


# ── the command line ──────────────────────────────────────────────────────
def test_version_prints_and_exits_without_starting_the_gui(capsys):
    from crystalline import __version__

    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


def test_help_explains_check(capsys):
    assert main(["--help"]) == 0
    assert "--check" in capsys.readouterr().out


def test_check_runs_the_report_rather_than_the_app(monkeypatch):
    called = []
    monkeypatch.setattr(diagnose, "report", lambda *a, **k: called.append(1) or 0)
    assert main(["--check"]) == 0
    assert called, "--check must not fall through to launching the window"
