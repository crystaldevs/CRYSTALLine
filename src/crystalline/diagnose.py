"""``crystalline --check``: does this machine have what the app needs?

A pip install of CRYSTALLine pulls in Qt, VTK, an OpenGL stack and the
scientific stack, and when one of them is missing or mismatched the failure
arrives as somebody else's error message. Qt's is actively misleading — it
tells a user on a headless box to reinstall the application, which will not
help — and on Windows the ``crystalline`` launcher is a ``gui_scripts`` entry
running under ``pythonw``, so a startup failure prints to a console that does
not exist and the command simply appears to do nothing.

So this runs the checks in the order they break, says what each one means, and
returns a non-zero exit code if anything essential failed. It is deliberately
importable without a display: each stage is attempted separately, and a stage
that cannot even be imported is reported rather than raised.
"""

from __future__ import annotations

import os
import platform
import sys
from typing import List, Tuple

# (label, ok, detail). ok is None for "not applicable / could not tell", which
# is not a failure — a machine with no ADP-capable CRYSTALClear still runs.
Result = Tuple[str, object, str]

MINIMUM_PYTHON = (3, 11)


def _python() -> Result:
    version = ".".join(str(n) for n in sys.version_info[:3])
    if sys.version_info[:2] < MINIMUM_PYTHON:
        need = ".".join(str(n) for n in MINIMUM_PYTHON)
        return ("Python", False,
                f"{version} — CRYSTALLine needs {need} or newer. "
                f"Make a new environment on a supported Python and install there.")
    return ("Python", True, f"{version} ({platform.python_implementation()})")


def _packages() -> List[Result]:
    """Every dependency the app imports, and what version arrived."""
    import importlib

    wanted = [
        ("PySide6", "the GUI toolkit"),
        ("vtkmodules.all", "the 3D renderer"),
        ("pyvista", "the rendering layer"),
        ("pyvistaqt", "the Qt/VTK bridge"),
        ("numpy", ""),
        ("scipy", ""),
        ("matplotlib", "property plots"),
        ("ase", ""),
        ("pymatgen", "symmetry and CIF"),
        ("spglib", "space groups"),
        ("PIL", "animation export"),
        ("CRYSTALClear", "reading CRYSTAL output"),
    ]
    results: List[Result] = []
    for name, why in wanted:
        label = name.split(".")[0]
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - a broken install is the point
            results.append((label, False,
                            f"could not import ({type(exc).__name__}: {exc})"
                            + (f" — needed for {why}" if why else "")))
            continue
        top = sys.modules.get(label, module)
        version = getattr(top, "__version__", None) or _metadata_version(label)
        results.append((label, True, version or "installed"))
    return results


def _metadata_version(name: str) -> str:
    import importlib.metadata as md

    for candidate in (name, name.lower(), {"PIL": "Pillow", "vtkmodules": "vtk"}.get(name, name)):
        try:
            return md.version(candidate)
        except Exception:  # noqa: BLE001
            continue
    return ""


def _crystalclear_features() -> Result:
    """Whether this CRYSTALClear exposes what the optional panels need.

    Stock 0.2.15 has no ``get_ADP`` and no ``get_calculation_info``; the app
    handles their absence, but silently — thermal ellipsoids and some Info rows
    just never appear, which looks like a bug in CRYSTALLine rather than a
    missing feature upstream.
    """
    try:
        from CRYSTALClear.crystal_io import Crystal_output
    except Exception as exc:  # noqa: BLE001
        return ("CRYSTALClear features", False, f"unavailable ({exc})")
    missing = [name for name in ("get_ADP", "get_calculation_info")
               if not hasattr(Crystal_output, name)]
    if missing:
        return ("CRYSTALClear features", None,
                f"no {', '.join(missing)} — thermal ellipsoids and some Info rows "
                f"will be missing. Upgrade with: pip install -U CRYSTALClear")
    return ("CRYSTALClear features", True, "ADP and calculation info available")


def _display() -> Result:
    """Can Qt actually open a window here?"""
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        return ("Display", False,
                "no DISPLAY or WAYLAND_DISPLAY — this looks like a headless "
                "session. CRYSTALLine is a desktop app: run it locally, or over "
                "SSH with X forwarding (ssh -X), or in a remote desktop.")
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # noqa: BLE001
        return ("Display", False, f"PySide6 unavailable ({exc})")
    try:
        app = QApplication.instance() or QApplication([])
        platform_name = app.platformName()
    except Exception as exc:  # noqa: BLE001
        return ("Display", False,
                f"Qt could not start ({exc}). On Linux this is usually missing "
                f"system libraries — libGL and libxkbcommon are the usual two.")
    return ("Display", True, f"Qt platform plugin '{platform_name}'")


def _opengl() -> Result:
    """The check that matters most, because it is the one that fails silently.

    A VTK render window is the app's whole centre. Software-only or missing
    OpenGL shows up here as a clear failure rather than as a window that opens
    black and then dies.
    """
    try:
        import pyvista as pv
    except Exception as exc:  # noqa: BLE001
        return ("OpenGL / VTK", False, f"pyvista unavailable ({exc})")
    try:
        plotter = pv.Plotter(off_screen=True, window_size=(64, 64))
        plotter.add_mesh(pv.Sphere())
        # render(), not show(): show() on an off-screen plotter waits on a
        # window that will never be interacted with, and the check hangs.
        plotter.render()
        image = plotter.screenshot(return_img=True)
        info = plotter.render_window.ReportCapabilities()
        plotter.close()
        if image is None or not getattr(image, "size", 0):
            return ("OpenGL / VTK", False, "the render window produced no pixels")
    except Exception as exc:  # noqa: BLE001
        return ("OpenGL / VTK", False,
                f"could not render ({type(exc).__name__}: {exc}). On Linux, "
                f"install libgl1 and libglx-mesa0 (or your distribution's "
                f"equivalent); in a container, add a software GL driver.")
    renderer = ""
    for line in (info or "").splitlines():
        if "OpenGL renderer string" in line:
            renderer = line.split(":", 1)[-1].strip()
    return ("OpenGL / VTK", True, renderer or "render window created")


def checks() -> List[Result]:
    """Run everything, cheapest and most fundamental first."""
    results = [_python()]
    results.extend(_packages())
    results.append(_crystalclear_features())
    results.append(_display())
    results.append(_opengl())
    return results


def report(stream=None) -> int:
    """Print the checks. Returns 0 if nothing essential failed, else 1."""
    stream = stream or sys.stdout
    from crystalline import __version__

    print(f"CRYSTALLine {__version__} on {platform.platform()}", file=stream)
    print(file=stream)

    results = checks()
    width = max(len(label) for label, _, _ in results)
    failed = 0
    for label, ok, detail in results:
        mark = {True: "ok  ", False: "FAIL", None: "note"}[ok]
        print(f"  [{mark}] {label:<{width}}  {detail}", file=stream)
        failed += ok is False

    print(file=stream)
    from crystalline.ui.safety import log_path

    print(f"  Errors are logged to {log_path()}", file=stream)
    if failed:
        # Deliberately not "the app will not start": a below-floor Python often
        # starts and then fails somewhere specific and confusing instead, which
        # is the situation this command exists to name.
        print(f"\n{failed} check(s) failed. CRYSTALLine is unlikely to work "
              f"properly until they are fixed.", file=stream)
    else:
        print("\nAll good. Start the app with:  crystalline", file=stream)
    return 1 if failed else 0


__all__ = ["checks", "report"]
