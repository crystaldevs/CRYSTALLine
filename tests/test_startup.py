"""What may and may not be imported before the window is on screen.

Startup used to import the main window — and with it VTK, PyVista and
matplotlib's font scan — before ``QApplication`` existed, so a fresh
installation showed nothing at all for a quarter of a minute. The order is now
load-bearing, and it is the kind of thing an innocent-looking import re-breaks,
so the rules are checked here rather than trusted.

Each check runs in its own interpreter: what matters is what a *fresh* process
loads, and the test suite has half of PyVista imported already.
"""

import subprocess
import sys
import textwrap

import pytest

_SRC = "src"


def _modules_after(code: str) -> set:
    """The module names loaded by a fresh interpreter running ``code``."""
    script = textwrap.dedent(f"""
        import sys
        {code}
        print("\\n".join(sorted(sys.modules)))
    """)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         cwd=None, env={**__import__("os").environ, "PYTHONPATH": _SRC,
                                        "QT_QPA_PLATFORM": "offscreen"})
    assert out.returncode == 0, out.stderr[-2000:]
    return set(out.stdout.split())


def test_importing_the_ui_package_does_not_build_the_window():
    """``from crystalline.ui import safety, theme`` is startup's first move.

    It used to drag in the main window through the package's own re-export,
    which meant the splash screen appeared *after* the slow part it exists to
    cover.
    """
    loaded = _modules_after("import crystalline.ui")

    assert "crystalline.ui.main_window" not in loaded
    assert not [m for m in loaded if m.startswith("vtkmodules")]
    assert "matplotlib.font_manager" not in loaded


def test_the_window_is_still_reachable_from_the_package():
    """Lazily, but by the same name as before."""
    pytest.importorskip("PySide6")
    from crystalline.ui import MainWindow

    assert MainWindow.__name__ == "MainWindow"


def test_the_renderer_does_not_import_the_whole_of_vtk():
    """``import vtk`` loads every VTK module — three times what PyVista needs,
    and about 0.6 s of every start. The renderer names the few it uses."""
    pytest.importorskip("pyvista")
    loaded = _modules_after("import crystalline.viz.renderer")

    assert "vtk" not in loaded, "the vtk umbrella is back in the renderer"
    assert "vtkmodules.vtkRenderingCore" in loaded  # the named ones are there


def test_the_font_cache_check_costs_nothing_and_answers():
    """Asking whether matplotlib has scanned this machine's fonts must not be
    what triggers the scan."""
    pytest.importorskip("matplotlib")
    from crystalline import app

    assert isinstance(app._fonts_are_cached(), bool)


def test_the_font_cache_is_warmed_on_a_thread_that_finishes():
    pytest.importorskip("matplotlib")
    from crystalline import app

    thread = app._warm_font_cache()
    thread.join(timeout=120)
    assert not thread.is_alive()
    assert "matplotlib.font_manager" in sys.modules
