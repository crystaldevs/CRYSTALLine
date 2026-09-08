"""Platform-plugin selection at startup.

Regression (Ubuntu/Wayland): the GUI died before showing a window with

    vtkXOpenGLRenderWindow ERR| The result is out of range, failed to get the
    converted tmp.
    X Error of failed request: BadWindow (invalid Window parameter)
      Major opcode of failed request: 12 (X_ConfigureWindow)

VTK renders into an X11 window and gets its id from Qt's ``winId()``. On a
native Wayland Qt plugin that is a 64-bit surface pointer, not an X window id,
so VTK's parse overflows and it then configures a bogus window. Running Qt on
XWayland avoids it — no display needed to test the decision itself.

``os.environ`` is swapped for a throwaway copy: the function under test writes
``QT_QPA_PLATFORM`` itself, and a leak of that into the real environment breaks
every Qt test that runs afterwards.
"""

import os
import sys

import pytest

from crystalline.app import _prefer_x11_on_wayland

_SESSION_VARS = ("QT_QPA_PLATFORM", "XDG_SESSION_TYPE", "WAYLAND_DISPLAY", "DISPLAY")


@pytest.fixture
def env(monkeypatch):
    """A throwaway environment, Linux-looking and with no session vars set."""
    fake = {k: v for k, v in os.environ.items() if k not in _SESSION_VARS}
    monkeypatch.setattr(os, "environ", fake)
    monkeypatch.setattr(sys, "platform", "linux")
    return fake


def test_wayland_session_falls_back_to_xcb(env):
    env["XDG_SESSION_TYPE"] = "wayland"
    env["WAYLAND_DISPLAY"] = "wayland-0"
    env["DISPLAY"] = ":0"  # XWayland is there to fall back to

    assert _prefer_x11_on_wayland() is True
    assert env["QT_QPA_PLATFORM"] == "xcb"


def test_an_x11_session_is_left_alone(env):
    env["XDG_SESSION_TYPE"] = "x11"
    env["DISPLAY"] = ":0"

    assert _prefer_x11_on_wayland() is False
    assert "QT_QPA_PLATFORM" not in env


def test_wayland_without_xwayland_is_left_alone(env):
    """No DISPLAY means no X server to fall back to — forcing xcb would only
    swap one startup failure for another."""
    env["XDG_SESSION_TYPE"] = "wayland"
    env["WAYLAND_DISPLAY"] = "wayland-0"

    assert _prefer_x11_on_wayland() is False
    assert "QT_QPA_PLATFORM" not in env


def test_an_explicit_platform_choice_wins(env):
    env["XDG_SESSION_TYPE"] = "wayland"
    env["WAYLAND_DISPLAY"] = "wayland-0"
    env["DISPLAY"] = ":0"
    env["QT_QPA_PLATFORM"] = "wayland"

    assert _prefer_x11_on_wayland() is False
    assert env["QT_QPA_PLATFORM"] == "wayland"  # untouched


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_other_platforms_are_untouched(env, monkeypatch, platform):
    monkeypatch.setattr(sys, "platform", platform)
    env["XDG_SESSION_TYPE"] = "wayland"
    env["DISPLAY"] = ":0"

    assert _prefer_x11_on_wayland() is False
    assert "QT_QPA_PLATFORM" not in env
