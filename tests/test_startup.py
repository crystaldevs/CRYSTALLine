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


def test_nothing_in_the_app_imports_the_whole_of_vtk():
    """Not only for speed. ``import vtk`` and ``vtkmodules.all`` load every
    library in the wheel, including ones the app never touches, such as
    ``vtkRenderingWebXR``. Windows' Smart App Control blocks a DLL it does not
    trust, and a blocked library fails the whole import — so one module nobody
    needs takes the 3D view down with it. Each import names its own classes."""
    import ast
    from pathlib import Path

    umbrella = {"vtk", "vtkmodules.all"}
    found = []
    for path in sorted(Path(_SRC, "crystalline").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            else:
                continue
            found += [f"{path}:{node.lineno} {n}" for n in names if n in umbrella]
    assert not found, found


def test_the_brillouin_zone_labels_do_not_import_the_whole_of_vtk():
    """Measuring a label and drawing it — the zone picker's text — with the
    classes named, and the text renderer still the real one."""
    pytest.importorskip("pyvista")
    loaded = _modules_after(
        "import pyvista\n"
        "        from crystalline.ui.panels import zone_picker\n"
        "        assert zone_picker._text_width('X', 12, 72) > 0\n"
        "        zone_picker._billboard(pyvista.Plotter(off_screen=True), (0, 0, 0), 'X', 'black', 12)"
    )

    assert "vtk" not in loaded and "vtkmodules.all" not in loaded
    assert "vtkmodules.vtkRenderingFreeType" in loaded


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


def test_the_window_opens_without_scipy_s_spatial_index():
    """A quarter of a second of every start, for a window that opens with
    nothing in it. The 3D view imports it when it first finds bonds."""
    pytest.importorskip("pyvista")
    loaded = _modules_after("import crystalline.ui.main_window")

    assert "scipy.spatial" not in loaded, "scipy.spatial is imported at startup again"


def test_what_the_first_file_needs_is_imported_once_the_window_is_up():
    """So that the first file opened does not pay for the imports instead:
    half a second of pymatgen, sympy, spglib and ASE, plus the k-d tree."""
    pytest.importorskip("scipy")
    import inspect

    from crystalline import app

    thread = app._warm_for_the_first_file()
    thread.join(timeout=120)
    assert not thread.is_alive()
    assert "scipy.spatial" in sys.modules
    # ...started after the window is shown, not alongside its import, where
    # the two would only take turns.
    source = inspect.getsource(app)
    assert source.index("window.show()") < source.index("_warm_for_the_first_file)")


def test_the_warm_up_list_names_modules_that_exist():
    """A name that has moved would be skipped without a word, and the first
    file would quietly pay for it again."""
    import importlib.util

    from crystalline import app

    missing = [name for name in app._FIRST_FILE_MODULES
               if importlib.util.find_spec(name) is None]
    assert missing == []
