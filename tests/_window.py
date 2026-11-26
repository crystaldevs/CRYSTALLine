"""A real ``MainWindow`` for tests, with only its 3D views stood in for.

A ``QtInteractor`` segfaults under the offscreen platform the tests run on, so
the window's plumbing has mostly been checked by reading its source. This
builds the real window instead — its tabs, docks, panels and menus, and the
real ``_show_read_file``, ``_fill_tab`` and ``_apply_cell_view`` — around a
stand-in for each tab's view: a plain widget with ``Viewport``'s signals,
handing what it is given to the real ``StructureRenderer`` over an off-screen
PyVista plotter. What a test checks is then what the window does with a file,
tab by tab, not what its source says it does.

The user's settings are left alone: the window reads the cell choice and the
appearance from ``QSettings``, and both are pointed at a throwaway file. And a
message box would wait for a click that never comes, so one fails the test
with its text instead.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings, Signal  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QWidget  # noqa: E402


class _Interactor:
    """The VTK widget's part the window touches: asking for a draw, closing."""

    def __init__(self, view):
        self._view = view

    def render(self):
        self._view.draws += 1

    def close(self):
        self._view.plotter.close()


class StandInViewport(QWidget):
    """What a tab's ``Viewport`` is to the window, without a ``QtInteractor``.

    The same signals, and every method the window calls on a view, each doing
    what the real one does to the renderer — so measurements, lattice planes and
    symmetry elements are drawn by the real ``StructureRenderer`` and a fault
    there still fails. A method the window starts calling that is not here is
    an ``AttributeError``, on purpose: add it, as for any stand-in.
    """

    atom_picked = Signal(int, bool)
    atom_moved = Signal(int)
    selection_cleared = Signal()
    interaction_started = Signal()
    camera_busy = Signal(bool)
    delete_requested = Signal()
    nudge_requested = Signal(object)

    def __init__(self, parent=None):
        import pyvista as pv

        from crystalline.viz.renderer import StructureRenderer

        super().__init__(parent)
        self.plotter = pv.Plotter(off_screen=True)
        self.renderer = StructureRenderer(self.plotter)
        self.interactor = _Interactor(self)
        self.draws = 0
        self.annotations = []
        self.selection = []

    def show_structure(self, structure, reference_cell=None, bond_structure=None):
        self.renderer.set_reference_cell(reference_cell)
        self.renderer.set_structure(structure, bond_structure=bond_structure)

    def draw_now(self):
        self.draws += 1

    def can_align_axes(self):
        return True

    def align_view_along(self, axis, reciprocal=False):
        pass

    def reset_view(self):
        pass

    def rotate_view(self, azimuth=0.0, elevation=0.0, roll=0.0):
        pass

    def set_annotations(self, annotations):
        self.annotations = list(annotations)
        self.renderer.set_annotations(annotations)

    def set_lattice_planes(self, planes, miller_cell=None):
        self.renderer.set_lattice_planes(planes, miller_cell)

    def set_symmetry_elements(self, elements, labels=False):
        self.renderer.set_symmetry_elements(elements, labels=labels)

    def set_editing_enabled(self, enabled):
        pass

    def set_selection(self, indices):
        self.selection = list(indices)
        self.renderer.highlight(indices)


def make_window(monkeypatch, tmp_path):
    """A real ``MainWindow`` on stand-in views, its settings in ``tmp_path``."""
    from crystalline.ui import main_window, preferences, theme

    QApplication.instance() or QApplication([])
    settings = str(tmp_path / "settings.ini")
    # Both modules make their QSettings in a function of their own; on macOS
    # the native store ignores QSettings.setPath, so an ini file it is.
    throwaway = lambda: QSettings(settings, QSettings.IniFormat)  # noqa: E731
    monkeypatch.setattr(preferences, "_settings", throwaway)
    monkeypatch.setattr(theme, "_settings", throwaway)
    monkeypatch.setattr(main_window, "Viewport", StandInViewport)

    def no_dialog(_parent, title, text, *args, **kwargs):
        raise AssertionError(f"a message box would have waited for a click: {title}: {text}")

    for name in ("warning", "critical", "information", "question"):
        monkeypatch.setattr(QMessageBox, name, no_dialog)
    return main_window.MainWindow()


def close_window(window):
    """Close the window and delete it, then the plotters behind its tabs' views.

    Deleted, not only closed: a closed window is still one of the
    application's top-level widgets, and an app-wide change reaches it — the
    theme restyles every top-level window, down to each view's corner marker.
    Left closed with its plotters gone, a later test switching the theme
    segfaulted in VTK (``vtkOrientationMarkerWidget::SetEnabled``). Qt deletes
    what is handed to ``deleteLater`` only when the event loop comes round,
    which ``processEvents`` does not count, so the posted deletes are sent here;
    and the plotters go last, once nothing is left to draw into them.
    """
    from PySide6.QtCore import QEvent

    from crystalline.ui.main_window import MainWindow

    plotters = [tab.viewport.plotter for tab in window._tabs if tab.built()]
    window.close()
    window.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    for plotter in plotters:
        plotter.close()
    assert not any(isinstance(w, MainWindow) for w in QApplication.topLevelWidgets())


def open_file(window, path, read=None, front=True):
    """Open ``path`` as File ▸ Open does once the file is read.

    ``read`` is what the reader made of it — :func:`main_window._read_file`
    unless given — and the rest is the window's own: a tab taken or added, and
    filled, at the front or behind the tab on screen; then, as the reading
    queue does once nothing else is waiting, the output's details — the Info
    panel's rows about the run — if they are still to read. Returns the tab.
    """
    from crystalline.ui import main_window

    read = read if read is not None else main_window._read_file(path)
    tab = window._show_read_file(path, read, front=front)
    if read.output_props is None and read.output_path:
        window._deliver_details(tab, path, main_window._read_details(read.output_path))
    QApplication.processEvents()
    return tab


def show_tab(window, tab):
    """Click on ``tab`` in the tab bar."""
    window._file_tabs.setCurrentWidget(tab.page)
    QApplication.processEvents()


def visible_geometry_panel(window):
    """The Geometry panel the dock shows: the one the user's clicks reach."""
    return window._panel_stacks["geometry_panel"].currentWidget()


def measure(window, atoms):
    """Click ``atoms`` in the tab on screen, in that order, then press Measure selection.

    Clicked as the 3D view reports a pick: the first on its own, each next one
    added with Ctrl/Shift. The order matters — an angle is measured at the
    second atom picked.
    """
    for picked, atom in enumerate(atoms):
        window.viewport.atom_picked.emit(int(atom), picked > 0)
    QApplication.processEvents()
    assert window.geometry_panel._selection == [int(a) for a in atoms]
    visible_geometry_panel(window)._measure_btn.click()
    return window.geometry_panel.measurements()[-1]


def export(window, monkeypatch, path):
    """Press Export to CSV… in the dock, and choose ``path`` in the dialog.

    Returns the name the dialog offered, and the CSV as rows.
    """
    import csv

    offered = []

    def ask(_parent, _title, suggested, _filter):
        offered.append(suggested)
        return str(path), "CSV files (*.csv)"

    monkeypatch.setattr(QFileDialog, "getSaveFileName", ask)
    visible_geometry_panel(window)._export_btn.click()
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return offered[0], list(csv.reader(handle))


def measure_a_bond(window):
    """Click the two closest atoms on screen and measure them: a bond, on any file."""
    from _panels import closest_pair

    return measure(window, closest_pair(window.structure.positions))
