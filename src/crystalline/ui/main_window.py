"""Main application window: viewport in the centre, dockable panels around it.

Wiring lives here and nowhere else — panels and the viewport expose signals,
and ``MainWindow`` connects them. Adding a new property (DOS, bands, elastic)
is: build a panel, dock it, connect its signals here.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSpinBox,
)

from crystalline.core.cells import (
    CellView,
    as_view,
    complete_boundary,
    expand_modes_to_conventional,
    tile_supercell,
    to_analysis_cell,
)
from crystalline.core.adp import ADPSet
from crystalline.core.phonons import PhononModes
from crystalline.core.structure import Structure
from crystalline.core.undo import UndoHistory
from crystalline.viz.phonon_animator import PhononAnimator
from crystalline.ui import menus
from crystalline.ui.viewport import Viewport
from crystalline.ui.panels.structure_panel import StructurePanel
from crystalline.ui.panels.phonon_panel import PhononPanel
from crystalline.ui.panels.info_panel import InfoPanel
from crystalline.ui.panels.display_settings import DisplayPanel
from crystalline.ui.panels.geometry_panel import GeometryPanel
from crystalline.ui.panels.symmetry_panel import SymmetryPanel
from crystalline.ui.panels.plot_view import PlotPanel

# Geometry of the floating Plots window on the first plot: a fraction of the main
# window's width (never below the minimum), 4:3, inset from its lower-right corner.
_PLOT_FLOAT_WIDTH_FRACTION = 0.55
_PLOT_FLOAT_MIN_WIDTH = 640
_PLOT_FLOAT_MARGIN = 24

# How far (cm⁻¹) a click may land from a mode and still be taken to mean it.
# Wide enough to forgive aiming at a broadened peak's flank, narrow enough that
# clicking empty baseline selects nothing.
_PEAK_PICK_TOLERANCE = 20.0


class MainWindow(QMainWindow):
    def __init__(self, structure: Optional[Structure] = None) -> None:
        super().__init__()
        self.setWindowTitle("CRYSTALLine — CRYSTAL structure & phonon viewer")
        from crystalline.resources import logo_path

        self.setWindowIcon(QIcon(logo_path()))
        self.resize(1200, 800)

        # The structure as loaded (CRYSTAL's primitive cell) is the pristine
        # source; ``self.structure`` is the cell view derived from it and shown.
        # By default we show the crystallographic (conventional) cell.
        self._source = structure if structure is not None else Structure.empty()
        self._cell_view = CellView.CRYSTALLOGRAPHIC
        self._supercell = (1, 1, 1)
        self._show_boundary = True  # show partially-belonging molecules by default
        self._editing = False
        self._modes: Optional[PhononModes] = None
        # Every q-point the run sampled (``[Gamma]`` for a plain FREQCALC, one
        # entry per commensurate q for a SCELPHONO run) and which of them
        # ``self._modes`` currently holds — the panel offers the choice, the
        # window rebuilds the view around it.
        self._qmodes: list = []
        self._qindex = 0
        # Supercell to restore when the phonon panel's Untile is pressed, or
        # ``None`` when the tiling on screen is the user's own doing.
        self._tile_restore: Optional[tuple] = None
        self._adps: Optional[ADPSet] = None  # thermal ellipsoids, if the run has them
        # source-atom index of each displayed atom, so a per-atom quantity can be
        # laid onto the (expanded, tiled, boundary-completed) cell on screen.
        self._adp_index: Optional[np.ndarray] = None
        # Undo history for structure edits (snapshots of the shown cell). A view
        # change (supercell/boundary/lattice/load) re-derives from source and
        # resets it — those aren't part of the per-edit undo timeline.
        self._history = UndoHistory()
        self._suppress_undo = False
        self._output_path: Optional[str] = None  # last-loaded CRYSTAL .out, for plots
        self._output_props: dict = {}  # parsed CRYSTAL-output rows for the Info panel
        self._axis_actions: list = []  # a/b/c view-alignment actions (menu + toolbar)
        (self.structure, _, self._unit_cell, self._bond_structure,
         self._adp_index) = self._compose_view(self._cell_view, self._supercell, None)

        # right dock: phonon modes (built before the viewport listener, which
        # references it). Animator drives the renderer.
        self.viewport = Viewport(self)
        self.animator = PhononAnimator(self.viewport.renderer)
        self.phonon_panel = PhononPanel(self.animator, self)

        # centre: 3D viewport
        self.setCentralWidget(self.viewport)
        self.viewport.show_structure(
            self.structure, reference_cell=self._unit_cell, bond_structure=self._bond_structure
        )
        self.structure.add_listener(self._on_structure_changed)

        # The Structure panel is kept as the selection/edit model (it owns the
        # shared selection and backs the Edit-menu tools) but is no longer shown
        # as a dock — 3D picking/drag and the Edit menu drive editing instead.
        self.structure_panel = StructurePanel(self.structure, self)
        self.structure_panel.hide()
        self._phonon_dock = self._dock("Phonons", self.phonon_panel, Qt.RightDockWidgetArea)

        # left dock: crystallographic info of the loaded system
        self.info_panel = InfoPanel(self)
        info_dock = self._info_dock = self._dock("Info", self.info_panel, Qt.LeftDockWidgetArea)
        self.info_panel.show_structure(self._source)

        # left dock (tabbed behind Info): live display settings.
        self.display_panel = DisplayPanel(
            self.viewport.renderer.settings, self._apply_render_settings, self
        )
        self._display_dock = self._dock("Display", self.display_panel, Qt.LeftDockWidgetArea)
        self.tabifyDockWidget(info_dock, self._display_dock)

        # left dock (tabbed alongside): measurements + atom tools for the selection.
        self.geometry_panel = GeometryPanel(self.structure, self)
        self._geometry_dock = self._dock("Geometry", self.geometry_panel, Qt.LeftDockWidgetArea)
        self.tabifyDockWidget(info_dock, self._geometry_dock)

        info_dock.raise_()  # show Info on top by default
        self.display_panel.set_elements(self.structure.numbers)  # initial element swatches

        # right dock, tabbed with Phonons: the structure's point-symmetry elements.
        # Hidden until Cell ▸ Point symmetry analysis asks for it — it is an
        # analysis someone goes looking for, not something every session needs on
        # screen (and the search itself only runs once the panel is opened).
        self.symmetry_panel = SymmetryPanel(self._symmetry_source(), self)
        self._symmetry_dock = self._dock(
            "Point symmetry", self.symmetry_panel, Qt.RightDockWidgetArea
        )
        self.tabifyDockWidget(self._phonon_dock, self._symmetry_dock)
        self._symmetry_dock.hide()

        # bottom dock: property plots (IR/Raman/bands/DOS…), one tab each.
        # Hidden until the first plot is built so it doesn't take up space.
        self.plot_panel = PlotPanel(self)
        self._plot_dock = self._dock("Plots", self.plot_panel, Qt.BottomDockWidgetArea)
        self._plot_dock.hide()
        self._plot_dock_floated = False  # floated once, on the first plot built

        # Bottom-right status indicators. Order matters: permanent widgets stack
        # left-to-right in call order, so counts sit left of the editing badge.
        self._count_status = QLabel()
        self._count_status.setStyleSheet("color: palette(mid); padding: 0 8px;")
        self.statusBar().addPermanentWidget(self._count_status)

        self._cell_status = QLabel()
        self._cell_status.setStyleSheet("color: palette(mid); padding: 0 8px;")
        self.statusBar().addPermanentWidget(self._cell_status)

        # Shown only while editing is on.
        self._editing_status = QLabel("● Editing mode")
        self._editing_status.setStyleSheet("color: #d9822b; font-weight: bold; padding: 0 8px;")
        self.statusBar().addPermanentWidget(self._editing_status)
        self._editing_status.hide()

        self._connect_signals()
        menus.build_menus(self)
        self._reset_undo()
        self._update_export_actions()
        self._update_status()
        self._update_import_action()  # match the (possibly non-empty) initial structure

    # ── wiring ──────────────────────────────────────────────────────────
    def _on_structure_changed(self, s: Structure) -> None:
        """Model edited: record undo, redraw, and reconcile a running animation."""
        self._capture_undo(s)
        self._update_status()  # atom count may have changed (add/remove)
        self._update_import_action()  # importing needs a non-empty structure
        self._refresh_info()  # symmetry/point group may have changed with the edit
        self.symmetry_panel.invalidate(self._symmetry_source())  # and so may its elements
        self.viewport.renderer.refresh()
        # Editing the geometry: stop any animation and re-anchor it to the edited
        # geometry (drop the modes if the atom count changed). Passing the new
        # positions avoids resetting atoms back to the stale equilibrium, which
        # would undo the edit that was just made.
        self.phonon_panel.invalidate_on_edit(s.positions)

    # ── undo ────────────────────────────────────────────────────────────
    def _capture_undo(self, s: Structure) -> None:
        """Push the pre-edit snapshot so this change can be undone.

        Each edit fires exactly one change notification; the baseline held from
        the previous notification is the state *before* this edit, so it's what
        an undo restores. Skipped while an undo is itself being applied.
        """
        if self._suppress_undo:
            return
        self._history.record(s.to_ase())
        self._update_undo_action()

    def _reset_undo(self) -> None:
        """Clear the undo history and re-baseline to the current structure.

        Used when the shown structure is replaced wholesale (file load, cell
        view, supercell, boundary, lattice parameters): edits don't carry across
        those, so neither does their undo timeline.
        """
        self._history.reset(self.structure.to_ase() if self.structure is not None else None)
        self._update_undo_action()

    def _undo(self) -> None:
        """Revert the most recent structure edit."""
        self._apply_history(self._history.undo())

    def _redo(self) -> None:
        """Re-apply the most recently undone structure edit."""
        self._apply_history(self._history.redo())

    def _apply_history(self, atoms) -> None:
        """Restore a snapshot returned by undo()/redo() (shared plumbing)."""
        if atoms is None:
            return
        self._suppress_undo = True
        try:
            self.structure.restore(atoms)  # listeners redraw; capture suppressed
        finally:
            self._suppress_undo = False
        self.structure_panel.clear_selection()  # indices may no longer be valid
        self._update_undo_action()

    def _update_undo_action(self) -> None:
        undo = getattr(self, "_undo_action", None)
        if undo is not None:
            undo.setEnabled(self._history.can_undo())
        redo = getattr(self, "_redo_action", None)
        if redo is not None:
            redo.setEnabled(self._history.can_redo())

    def _connect_signals(self) -> None:
        # viewport pick -> update selection (additive with Ctrl/Shift)
        self.viewport.atom_picked.connect(self.structure_panel.select_atom)
        # dragging an atom in 3D -> update selection (keep a group drag intact)
        self.viewport.atom_moved.connect(self._on_atom_moved)
        # clicking empty space in the 3D view -> clear the panel selection
        self.viewport.selection_cleared.connect(self.structure_panel.clear_selection)
        # Del/Backspace over the 3D view -> delete the selection (the menu's Del
        # shortcut can't fire while the VTK widget holds keyboard focus)
        self.viewport.delete_requested.connect(self._delete_selected)
        # arrow keys over the 3D view (editing mode) -> nudge the selection
        self.viewport.nudge_requested.connect(self._nudge_selection)
        # starting to drag an atom -> stop any running phonon animation
        self.viewport.interaction_started.connect(self.phonon_panel.stop)
        # the panel owns the selection -> highlight it + refresh the Edit menu
        self.structure_panel.selection_changed.connect(self._on_selection_changed)
        # a phonon mode was (de)selected -> refresh the animation-export action
        self.phonon_panel.mode_selected.connect(lambda _row: self._update_export_actions())
        # another q-point was picked -> show that q's modes on a rebuilt view
        self.phonon_panel.qpoint_selected.connect(self._set_qpoint)
        # "Tile n×n×n" next to the q-point -> the supercell one period needs
        self.phonon_panel.tile_requested.connect(self._tile_to_qpoint)

        # Geometry panel: the same edit operations as the Edit menu (so undo and
        # the selection model behave identically), plus measurement overlays.
        self.geometry_panel.editing_toggled.connect(self._toggle_editing)
        self.geometry_panel.delete_requested.connect(self._delete_selected)
        self.geometry_panel.duplicate_requested.connect(self._duplicate_selected)
        self.geometry_panel.translate_requested.connect(self._translate_selected)
        self.geometry_panel.set_element_requested.connect(self._set_element_of_selection)
        self.geometry_panel.add_atom_requested.connect(self._add_atom)
        self.geometry_panel.annotations_changed.connect(self.viewport.set_annotations)
        # Symmetry panel: the ticked elements are drawn over the structure.
        self.symmetry_panel.elements_changed.connect(self.viewport.set_symmetry_elements)

    def _symmetry_source(self) -> Structure:
        """The structure the symmetry search runs on: one clean unit cell.

        Not the one on screen — a supercell has the wrong box and a
        boundary-completed view holds two copies of every atom that straddles the
        edge, and a symmetry finder can read neither. The same fold the Info
        panel's analysis goes through gives a single cell, edits included; the
        elements it finds repeat with that cell's lattice, so they still fill the
        drawn box however many cells of it are on screen.
        """
        try:
            return to_analysis_cell(self.structure, self._unit_cell)
        except Exception:  # noqa: BLE001 - fall back to the shown cell rather than nothing
            return self.structure

    def _dock(self, title: str, widget, area) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _show_about(self) -> None:
        menus.show_about(self)

    def _update_view_actions(self) -> None:
        """Enable a/b/c view alignment only when there's a cell to align to."""
        enabled = self.viewport.can_align_axes()
        for widget in (*self._axis_actions, *getattr(self, "_axis_buttons", [])):
            widget.setEnabled(enabled)

    def _update_plot_actions(self) -> None:
        """Enable each plot action according to the loaded file.

        Output plots (IR/Raman/elastic/EOS) are enabled only when the loaded
        CRYSTAL output actually contains their data; data plots (bands/DOS/XRD)
        stay enabled — they read a separate file the user chooses.
        """
        from crystalline.crystalio import output_availability

        available = output_availability(self._output_path)
        for kind in self._plot_kinds:
            enabled = True if kind.source == "data" else (kind.key in available)
            self._plot_actions[kind.key].setEnabled(enabled)

    def _open_plot(self, kind) -> None:
        """Build ``kind``'s figure and add it as a tab in the Plots dock.

        Output-file plots reuse the loaded CRYSTAL output (prompting only if none
        is loaded); data-file plots always prompt for their PROPERTIES file.
        """
        if kind.source == "output" and self._output_path:
            path = self._output_path
        else:
            path, _ = QFileDialog.getOpenFileName(self, kind.caption, "", kind.file_filter)
            if not path:
                return
        try:
            figure = kind.build(path)
        except Exception as exc:  # noqa: BLE001 - surface any read/plot error to the user
            QMessageBox.critical(self, "Plot failed", f"Could not create the plot:\n{exc}")
            return
        # No pick handler: none of these plots is drawn against a wavenumber
        # axis. The spectra dialog's figures are, and pass one themselves.
        self.plot_panel.add_figure(figure, kind.label.rstrip("… "))
        self._reveal_plot_dock()

    # ── thermal ellipsoids (ADP) ────────────────────────────────────────
    def _load_adps(self, path: str) -> Optional[ADPSet]:
        """The ADPs of the file just opened, or ``None`` if it carries none.

        Most frequency runs don't: ADPs need the ``ADP`` keyword, so their
        absence is the normal case and not worth reporting to the user.
        """
        try:
            from crystalline.crystalio import load_adp

            return load_adp(path)
        except Exception:  # noqa: BLE001 - never let this break loading a file
            return None

    def _displayed_adp(self, temperature_index: Optional[int] = None) -> Optional[np.ndarray]:
        """``(natom, 3, 3)`` tensors for the *source* cell at one temperature.

        Defaults to the temperature the renderer's settings currently name;
        ``temperature_index`` overrides it, which is what lets a settings change
        be applied before the renderer has been told about it.
        """
        if self._adps is None or len(self._adps) == 0:
            return None
        if temperature_index is None:
            temperature_index = self.viewport.renderer.settings.adp_temperature_index
        return self._adps.at(temperature_index)

    def _adp_tensors_for_view(self, temperature_index: Optional[int] = None) -> Optional[np.ndarray]:
        """The chosen temperature's tensors laid onto the *displayed* atoms.

        The displayed cell can be expanded, tiled and boundary-completed, so each
        drawn atom takes the tensor of the source atom it images — that is what
        ``self._adp_index`` records. Cheap enough to redo whenever the
        temperature changes, which is the point of keeping the index around.
        """
        tensors = self._displayed_adp(temperature_index)
        if tensors is None or self._adp_index is None or len(self._adp_index) == 0:
            return None
        if int(np.max(self._adp_index)) >= len(tensors):
            return None  # ADPs don't span the source cell (a mismatched pairing)
        return tensors[self._adp_index]

    def _refresh_adp_tensors(self, temperature_index=None, redraw: bool = True) -> None:
        """Push the ellipsoid tensors matching the current view and temperature."""
        self.viewport.renderer.set_adp_tensors(
            self._adp_tensors_for_view(temperature_index), redraw=redraw
        )

    def _update_adp_controls(self, autoshow: bool = False) -> None:
        """Offer the ellipsoid controls only for a file that has ADPs.

        ``autoshow`` is passed when a file has just been opened, so a run that
        computed ADPs displays them without being asked. A view change re-lists
        the same temperatures and leaves the toggle alone.
        """
        labels = [] if self._adps is None else [
            self._adps.label(i) for i in range(len(self._adps))
        ]
        self.display_panel.set_adp_temperatures(labels, autoshow=autoshow)

    # ── vibrational spectra ─────────────────────────────────────────────
    def _open_spectra(self) -> None:
        """Offer every spectrum the loaded output holds, and plot the chosen ones.

        Raman polarisations and anharmonic levels between them run to dozens of
        curves, so they live behind one dialog rather than one menu entry each,
        and several can be overlaid at once — which is how the components are
        actually read.
        """
        from crystalline.crystalio import available_spectra, load_spectra, plot_spectra
        from crystalline.ui.panels.spectra_dialog import SpectraDialog

        path = self._output_path
        if not path:
            QMessageBox.information(
                self, "No output loaded",
                "Load a CRYSTAL .out from a frequency calculation to plot its spectra.",
            )
            return
        kinds = available_spectra(path)
        if not kinds:
            QMessageBox.information(
                self, "No spectra in this output",
                "This output carries no IR or Raman intensities.\n\n"
                "IR needs a FREQCALC with INTENS; Raman additionally needs INTRAMAN "
                "with a CPHF step.",
            )
            return

        dialog = SpectraDialog(kinds, self)
        if dialog.exec() != QDialog.Accepted:
            return
        chosen = dialog.selected_kinds()
        try:
            data = load_spectra(path)
            figure = plot_spectra(
                [(kind.label, data[kind]) for kind in chosen],
                title=chosen[0].label if len(chosen) == 1 else None,
                **dialog.options(),
            )
        except Exception as exc:  # noqa: BLE001 - surface any read/plot error
            QMessageBox.critical(self, "Plot failed", f"Could not create the plot:\n{exc}")
            return

        title = chosen[0].label if len(chosen) == 1 else f"Spectra ({len(chosen)})"
        # The x axis is a wavenumber, so clicking a peak still selects its mode.
        self.plot_panel.add_figure(figure, title, on_pick=self._select_mode_near)
        self._reveal_plot_dock()

    def _update_spectra_action(self) -> None:
        """Enable the spectra entry only for an output that has any."""
        action = getattr(self, "_spectra_action", None)
        if action is not None:
            action.setEnabled(bool(self._output_path))

    # ── VCI wavefunctions ───────────────────────────────────────────────
    def _open_vci(self) -> None:
        """Draw the VCI coefficients of the loaded output, the chosen way.

        A VCI run carries one state per configuration, so there is no single
        figure to show: the dialog is where which states — and how much mixing
        to keep — get decided.
        """
        from crystalline.crystalio import plot_vci, vci_run
        from crystalline.ui.panels.vci_dialog import VCIDialog

        path = self._output_path
        if not path:
            QMessageBox.information(
                self, "No output loaded",
                "Load a CRYSTAL .out from an anharmonic calculation to plot "
                "its VCI states.",
            )
            return

        # Parsing a large run takes a moment, and it happens before any dialog
        # appears, so say so with the cursor rather than looking hung.
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            run, out = vci_run(path)
        finally:
            QApplication.restoreOverrideCursor()
        if run is None:
            QMessageBox.information(
                self, "No VCI states in this output",
                "This output carries no VCI wavefunctions.\n\n"
                "They need an ANHARM run with a VCI step, on either the "
                "harmonic (VCI@HO) or the VSCF (VCI@VSCF) basis.",
            )
            return

        dialog = VCIDialog(run, self)
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            figure = plot_vci(out, **dialog.options())
        except Exception as exc:  # noqa: BLE001 - surface any plot error
            QMessageBox.critical(self, "Plot failed", f"Could not create the plot:\n{exc}")
            return

        # No pick handler: neither representation is drawn against a wavenumber
        # axis — the states sit on a categorical axis, one column each.
        self.plot_panel.add_figure(figure, dialog.title())
        self._reveal_plot_dock()

    # ── anharmonic scan ─────────────────────────────────────────────────
    def _open_anscan(self) -> None:
        """Draw the anharmonic scan of the loaded output.

        The wavefunction coefficients are not in the ``.out`` but in
        ANSCANWF.DAT, so the file next to it is used when there is one and the
        user is asked only when there is not.
        """
        import os

        from crystalline.crystalio import (
            WF_FILTER,
            anscan_run,
            find_wavefunctions,
            plot_anscan,
        )
        from crystalline.ui.panels.anscan_dialog import AnscanDialog

        path = self._output_path
        if not path:
            QMessageBox.information(
                self, "No output loaded",
                "Load a CRYSTAL .out from an ANSCAN run to plot its scan.",
            )
            return

        wavefunctions = find_wavefunctions(path)
        if wavefunctions is None:
            wavefunctions, _ = QFileDialog.getOpenFileName(
                self, "Open ANSCAN wavefunctions (ANSCANWF.DAT)",
                os.path.dirname(path), WF_FILTER,
            )
            if not wavefunctions:
                return

        # get_anscan re-reads the phonon block on its way through, which is not
        # instant on a large run, and it happens before any dialog appears.
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            run, out = anscan_run(path, wavefunctions)
        finally:
            QApplication.restoreOverrideCursor()
        if run is None:
            QMessageBox.information(
                self, "No anharmonic scan in this output",
                "This output and its wavefunction file could not be read as an "
                "ANSCAN run.\n\n"
                "It needs a FREQCALC with ANHARM/ANSCAN, and the ANSCANWF.DAT "
                "written by that same run.",
            )
            return

        dialog = AnscanDialog(run, self)
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            figure = plot_anscan(out, **dialog.options())
        except Exception as exc:  # noqa: BLE001 - surface any plot error
            QMessageBox.critical(self, "Plot failed", f"Could not create the plot:\n{exc}")
            return

        # No pick handler: the abscissa is the displacement the mode was scanned
        # along, not a wavenumber, so there is no mode under the cursor.
        self.plot_panel.add_figure(figure, dialog.title())
        self._reveal_plot_dock()

    def _update_anscan_action(self) -> None:
        """Enable the scan entry only for an output that carries a solved one.

        The cheap text probe, like the VCI entry: this runs on every load, and
        the real parse walks the phonon block as well.
        """
        from crystalline.crystalio import has_anscan

        action = getattr(self, "_anscan_action", None)
        if action is not None:
            action.setEnabled(has_anscan(self._output_path))

    # ── anharmonic PES ──────────────────────────────────────────────────
    def _open_pes(self) -> None:
        """Draw a cut through the PES an ANHAPES run differentiated.

        The surface spans every mode the run was given, so there is no single
        figure: which mode, or which pair of them, is the dialog's job.
        """
        from crystalline.crystalio import pes_run, plot_pes
        from crystalline.ui.panels.pes_dialog import PESDialog

        path = self._output_path
        if not path:
            QMessageBox.information(
                self, "No output loaded",
                "Load a CRYSTAL .out from an anharmonic calculation to plot "
                "its potential-energy surface.",
            )
            return

        # Two parses — the PES constants and the harmonic frequencies — before
        # any dialog appears, and neither is instant on a large run.
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            run, out = pes_run(path)
        finally:
            QApplication.restoreOverrideCursor()
        if run is None:
            QMessageBox.information(
                self, "No PES constants in this output",
                "This output carries no cubic and quartic energy derivatives.\n\n"
                "They need a FREQCALC with ANHARM/ANHAPES, whose harmonic "
                "frequencies have to be in the same file.",
            )
            return

        dialog = PESDialog(run, self)
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            figure = plot_pes(out, **dialog.options())
        except Exception as exc:  # noqa: BLE001 - surface any plot error
            QMessageBox.critical(self, "Plot failed", f"Could not create the plot:\n{exc}")
            return

        # No pick handler: both axes are normal coordinates, not wavenumbers.
        self.plot_panel.add_figure(figure, dialog.title())
        self._reveal_plot_dock()

    def _update_pes_action(self) -> None:
        """Enable the PES entry only for an output that carries the constants."""
        from crystalline.crystalio import has_pes

        action = getattr(self, "_pes_action", None)
        if action is not None:
            action.setEnabled(has_pes(self._output_path))

    # ── plot typography ─────────────────────────────────────────────────
    def _open_plot_font(self) -> None:
        """Set the font of the plots built from now on.

        Remembered on the window so reopening the dialog shows the current
        choice rather than resetting to the default.
        """
        from crystalline.crystalio.plotting import (
            DEFAULT_FONT_FAMILY,
            DEFAULT_FONT_SIZE,
            apply_font,
        )
        from crystalline.ui.panels.font_dialog import PlotFontDialog

        family = getattr(self, "_plot_font_family", DEFAULT_FONT_FAMILY)
        size = getattr(self, "_plot_font_size", DEFAULT_FONT_SIZE)

        dialog = PlotFontDialog(family, size, self)
        if dialog.exec() != QDialog.Accepted:
            return
        chosen = dialog.options()
        apply_font(**chosen)
        self._plot_font_family = chosen["family"]
        self._plot_font_size = chosen["size"]

    def _update_vci_action(self) -> None:
        """Enable the VCI entry only for an output that carries VCI states.

        Uses the cheap text probe, not the real parse: this runs every time a
        file is loaded, and parsing a large VCI block takes a second or two.
        """
        from crystalline.crystalio import has_vci

        action = getattr(self, "_vci_action", None)
        if action is not None:
            action.setEnabled(has_vci(self._output_path))

    # ── spectrum ↔ mode linking ─────────────────────────────────────────
    def _select_mode_near(self, frequency: float) -> None:
        """Select the mode nearest ``frequency`` (cm⁻¹) and reveal the Phonons dock.

        Clicking a peak is how people ask "what is this band?", so the answer has
        to be the structure moving, not a number. A click far from every mode is
        ignored rather than snapping to a distant one: on a broadened spectrum
        most of the x axis is baseline, and jumping to whatever happens to be
        closest would be noise, not an answer.

        An IR or Raman band is a zone-centre quantity, so a click always answers
        with a Gamma mode: if another q-point is on show, the panel is sent back
        to Gamma first rather than matching the frequency against modes that
        cannot have produced the peak.
        """
        if self._qindex != 0:
            self.phonon_panel.select_qpoint(0)
        frequencies = self.phonon_panel.frequencies()
        if frequencies is None or len(frequencies) == 0:
            return
        index = int(np.argmin(np.abs(frequencies - frequency)))
        if abs(frequencies[index] - frequency) > _PEAK_PICK_TOLERANCE:
            return
        if self.phonon_panel.select_mode(index):
            self._phonon_dock.show()
            self._phonon_dock.raise_()

    def _reveal_plot_dock(self) -> None:
        """Show the Plots dock — floating, the first time a plot is built.

        Docked at the bottom of the main window, a plot gets a wide, short strip:
        the worst shape for a figure that wants to be roughly square (polar
        elastic sections, elastic surfaces, XRD). So the first plot pops the dock
        out into a free-floating window sized 4:3 beside the main window.

        It stays a dock, not a separate window class: drag it back to re-attach.
        Whatever the user settles on is then left alone — the float happens once,
        not on every plot.
        """
        if not self._plot_dock_floated:
            self._plot_dock_floated = True
            self._plot_dock.setFloating(True)
            self._plot_dock.show()
            screen = self.screen()
            where = _floating_plot_geometry(
                self.frameGeometry(), screen.availableGeometry() if screen else None
            )
            self._plot_dock.setGeometry(where)
        self._plot_dock.show()
        self._plot_dock.raise_()

    def _show_display_panel(self) -> None:
        """Reveal and raise the (dockable) display-settings panel."""
        self._display_dock.show()
        self._display_dock.raise_()

    def _show_symmetry_panel(self) -> None:
        """Open the point-symmetry panel (a tab beside Phonons) and analyse."""
        self._symmetry_dock.show()
        self._symmetry_dock.raise_()
        self.symmetry_panel.show_analysis()

    # ── panels ──────────────────────────────────────────────────────────
    def _panel_docks(self) -> list:
        """``(title, dock)`` for every panel, in the order the View menu lists them.

        Closing a dock with its × is easy to do by accident and, without this,
        impossible to undo — the panel is simply gone for the rest of the
        session. Every dock is registered here so it can always be brought back.
        """
        return [
            ("Info", self._info_dock),
            ("Display", self._display_dock),
            ("Geometry", self._geometry_dock),
            ("Point symmetry", self._symmetry_dock),
            ("Phonons", self._phonon_dock),
            ("Plots", self._plot_dock),
        ]

    def _restore_all_panels(self) -> None:
        """Bring every closed panel back, and float none of them.

        A dock the user dragged out and then closed would otherwise reappear
        floating off-screen if the window has since moved, so anything being
        restored is re-docked on the way.
        """
        for _title, dock in self._panel_docks():
            if dock.isHidden():
                dock.setFloating(False)
                dock.show()
        self._info_dock.raise_()

    def _apply_render_settings(self, settings) -> None:
        """Apply new display settings, re-selecting the ADP temperature if it moved.

        The temperature is a *setting*, but the tensors it names live on the
        renderer — so moving the picker has to push the new ones as well, or the
        ellipsoids keep the shape they were given when the file was loaded. They
        are staged rather than drawn, so the settings change below is the single
        rebuild the user sees.
        """
        renderer = self.viewport.renderer
        if settings.adp_temperature_index != renderer.settings.adp_temperature_index:
            self._refresh_adp_tensors(settings.adp_temperature_index, redraw=False)
        renderer.set_settings(settings)

    def _restore_geometry(self) -> None:
        """Discard in-app edits and rebuild the originally-loaded geometry."""
        self.structure_panel.clear_selection()
        self._apply_cell_view()  # re-derives from the pristine source

    # ── editing: mode, selection, tools ─────────────────────────────────
    def _toggle_editing(self, enabled: bool) -> None:
        """Route the Geometry panel's checkbox through the Edit-menu action.

        Going via the action keeps the menu's tick, the status badge and the
        panel in step whichever one the user clicked. Guarded with ``getattr``
        because signals are connected before the menus are built, and skipped
        when already in the requested state so the two can't ping-pong.
        """
        action = getattr(self, "_edit_mode_action", None)
        if action is None:
            self._set_editing(enabled)  # no menu yet: apply it directly
        elif action.isChecked() != bool(enabled):
            action.setChecked(bool(enabled))  # -> toggled -> _set_editing

    def _set_editing(self, enabled: bool) -> None:
        self._editing = bool(enabled)
        self.viewport.set_editing_enabled(self._editing)
        self.structure_panel.set_editing_enabled(self._editing)
        self.geometry_panel.set_editing_enabled(self._editing)
        self._editing_status.setVisible(self._editing)  # bottom-right indicator
        self._update_edit_actions()

    def _on_atom_moved(self, index: int) -> None:
        """After a drag: keep a multi-atom (group) selection so it can be dragged
        again; for a lone atom, make it the selection so the editor tracks it."""
        if index not in self.structure_panel.selected_indices():
            self.structure_panel.select_atom(index, False)

    def _on_selection_changed(self, indices) -> None:
        self.viewport.set_selection(indices)
        self.geometry_panel.set_selection(indices)  # drives what can be measured
        self._update_edit_actions()
        self._update_status()

    def _update_import_action(self) -> None:
        """Enable 'Import atoms' only once there's a structure to import into."""
        action = getattr(self, "_import_action", None)
        if action is not None:
            action.setEnabled(len(self.structure) > 0)

    def _update_status(self) -> None:
        """Refresh the bottom-right indicators: atom count, selection, cell view."""
        count = getattr(self, "_count_status", None)
        if count is None:
            return
        n = len(self.structure)
        text = f"{n} atom{'s' if n != 1 else ''}"
        selected = len(self.structure_panel.selected_indices())
        if selected:
            text += f"  ·  {selected} selected"
        count.setText(text)

        na, nb, nc = self._supercell
        cell = "" if self._supercell == (1, 1, 1) else f"supercell {na}×{nb}×{nc}"
        self._cell_status.setText(cell)

    def _update_edit_actions(self) -> None:
        """Edit tools need editing on and at least one atom selected."""
        active = self._editing and bool(self.structure_panel.selected_indices())
        for action in self._edit_tool_actions:
            action.setEnabled(active)

    def _select_all(self) -> None:
        self.structure_panel.set_selection(range(len(self.structure)))

    def _clear_selection(self) -> None:
        self.structure_panel.clear_selection()

    def _invert_selection(self) -> None:
        current = set(self.structure_panel.selected_indices())
        self.structure_panel.set_selection(set(range(len(self.structure))) - current)

    def _delete_selected(self) -> None:
        indices = self.structure_panel.selected_indices()
        if not self._editing or not indices:
            return
        self.structure_panel.clear_selection()  # deletion shifts indices; drop them first
        self.structure.remove_atoms(indices)

    def _duplicate_selected(self) -> None:
        indices = self.structure_panel.selected_indices()
        if not self._editing or not indices:
            return
        # Offset the copies so they don't sit exactly on the originals.
        new = self.structure.duplicate_atoms(indices, offset=(1.5, 0.0, 0.0))
        self.structure_panel.set_selection(new)

    def _translate_selected(self) -> None:
        indices = self.structure_panel.selected_indices()
        if not self._editing or not indices:
            return
        vector = self._ask_vector()
        if vector is not None:
            self.structure.translate_atoms(indices, vector)

    def _nudge_selection(self, vector) -> None:
        """Move the selection by ``vector`` (an arrow-key step in the view plane)."""
        indices = self.structure_panel.selected_indices()
        if not self._editing or not indices:
            return
        self.structure.translate_atoms(indices, vector)  # -> undo + redraw
        # The redraw rebuilds the scene and drops the selection halos; the
        # selection itself is unchanged, so re-apply them to keep it visible.
        self.viewport.set_selection(indices)

    def _set_element_selected(self) -> None:
        """Edit-menu route: ask for the symbol, then apply it to the selection."""
        if not self._editing or not self.structure_panel.selected_indices():
            return
        symbol, ok = QInputDialog.getText(self, "Set element", "Element symbol:")
        if ok and symbol.strip():
            self._set_element_of_selection(symbol)

    def _set_element_of_selection(self, symbol: str) -> None:
        """Re-element the selection (the Geometry panel supplies the symbol)."""
        indices = self.structure_panel.selected_indices()
        if not self._editing or not indices or not symbol.strip():
            return
        try:
            self.structure.set_symbols(indices, symbol.strip())
        except ValueError:
            QMessageBox.warning(self, "Unknown element", f"'{symbol}' is not a known element.")

    def _add_atom(self, symbol: str) -> None:
        """Add one atom of ``symbol`` at the centre of the cell, and select it.

        The centre is the obvious place to drop an atom you are about to drag
        into position: inside the cell and away from the boundary. For a
        non-periodic system it lands on the structure's centroid.
        """
        if not self._editing or not symbol.strip():
            return
        try:
            index = self.structure.add_atom(symbol.strip(), self._new_atom_position())
        except ValueError:
            QMessageBox.warning(self, "Unknown element", f"'{symbol}' is not a known element.")
            return
        self.structure_panel.set_selection([index])  # ready to drag / translate

    def _new_atom_position(self) -> list:
        cell = np.asarray(self.structure.cell, dtype=float)
        if self.structure.is_periodic and not np.allclose(cell, 0.0):
            return list(0.5 * cell.sum(axis=0))
        if len(self.structure):
            return list(np.asarray(self.structure.positions, dtype=float).mean(axis=0))
        return [0.0, 0.0, 0.0]

    def _ask_vector(self):
        """Small dialog returning a cartesian (dx, dy, dz) shift, or None."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Translate selected")
        form = QFormLayout(dialog)
        boxes = []
        for axis in ("x", "y", "z"):
            box = QDoubleSpinBox()
            box.setRange(-1000.0, 1000.0)
            box.setDecimals(3)
            box.setSingleStep(0.1)
            form.addRow(f"Δ{axis} (Å)", box)
            boxes.append(box)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.Accepted:
            return None
        return [box.value() for box in boxes]

    def _on_boundary_toggled(self, enabled: bool) -> None:
        self._show_boundary = bool(enabled)
        self._apply_cell_view()

    def _open_lattice_dialog(self) -> None:
        """Edit the crystal's lattice parameters (a, b, c, α, β, γ).

        Applied to the pristine source and re-derived through the same pipeline
        as the other Cell actions, so the current view (conventional cell,
        supercell, boundary completion) is rebuilt on the new lattice. Atoms keep
        their fractional coordinates (the cell is reshaped around them).
        """
        if not self._source.is_periodic or np.allclose(self._source.cell, 0.0):
            QMessageBox.information(
                self,
                "Lattice parameters",
                "The loaded system is not periodic — there is no cell to edit.",
            )
            return

        a, b, c, alpha, beta, gamma = (float(v) for v in self._source.cellpar)
        dialog = QDialog(self)
        dialog.setWindowTitle("Lattice parameters")
        form = QFormLayout(dialog)
        # (label, value, min, max, decimals) — lengths in Å, angles in degrees
        specs = [
            ("a (Å)", a, 0.1, 1000.0, 4),
            ("b (Å)", b, 0.1, 1000.0, 4),
            ("c (Å)", c, 0.1, 1000.0, 4),
            ("α (°)", alpha, 1.0, 179.0, 3),
            ("β (°)", beta, 1.0, 179.0, 3),
            ("γ (°)", gamma, 1.0, 179.0, 3),
        ]
        boxes = []
        for label, value, lo, hi, decimals in specs:
            box = QDoubleSpinBox()
            box.setRange(lo, hi)
            box.setDecimals(decimals)
            box.setSingleStep(0.1)
            box.setValue(value)
            form.addRow(label, box)
            boxes.append(box)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() != QDialog.Accepted:
            return
        self._source.set_lattice_parameters(*(box.value() for box in boxes))
        self._apply_cell_view()
        self.info_panel.show_structure(self._source, self._output_props)

    def _open_supercell_dialog(self) -> None:
        """Prompt for the na × nb × nc repetitions and rebuild the view.

        The cell wireframe keeps outlining the original unit cell (not the
        enlarged supercell box) — see ``_compose_view``/``Viewport.show_structure``.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("Supercell")
        form = QFormLayout(dialog)
        boxes = []
        for axis, value in zip(("a", "b", "c"), self._supercell):
            box = QSpinBox()
            box.setRange(1, 12)
            box.setValue(value)
            form.addRow(f"Repeat along {axis}", box)
            boxes.append(box)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() != QDialog.Accepted:
            return
        # A supercell chosen here is the user's own: the phonon panel's Untile
        # must not offer to throw it away for a cell they never asked for.
        self._tile_restore = None
        self._set_supercell(tuple(box.value() for box in boxes))
        self._apply_cell_view()

    def _set_supercell(self, reps) -> None:
        """Record the supercell tiling and keep the menu action's label in step."""
        self._supercell = tuple(reps)
        action = getattr(self, "_supercell_action", None)
        if action is not None:
            na, nb, nc = self._supercell
            suffix = "" if self._supercell == (1, 1, 1) else f"  ({na}×{nb}×{nc})"
            action.setText(f"Supercell…{suffix}")
        panel = getattr(self, "phonon_panel", None)
        if panel is not None:
            # ...and what its Untile would go back to, if it did the tiling.
            panel.set_supercell(self._supercell, self._tile_restore)

    # ── phonon q-points (a SCELPHONO run samples more than Gamma) ────────
    def _set_qmodes(self, qmodes) -> None:
        """Adopt a file's phonon modes: every sampled q-point, Gamma first.

        The window keeps them all so switching q-point is a re-tiling rather
        than a re-read of the output, and shows the Gamma set — what a
        frequency calculation is normally opened for.
        """
        self._qmodes = list(qmodes or [])
        self._qindex = 0
        self._modes = self._qmodes[0] if self._qmodes else None
        self.phonon_panel.set_qpoints([m.qpoint for m in self._qmodes], 0)

    def _set_qpoint(self, index: int) -> None:
        """Show the modes of sampled q-point ``index``.

        Changing q changes the *modes*, not the structure: the atoms sit exactly
        where they did. So only the modes are re-derived — they still have to go
        through the cell pipeline, since away from Gamma each image atom's
        displacement carries the phase of the cell it sits in (see
        ``core.cells``) — and the scene on screen is left alone. Rebuilding it
        would cost an order of magnitude more (a VTK teardown and redraw of
        every atom) for a picture identical to the one already there, which is
        felt as a lag on what should be an instant choice.
        """
        if not 0 <= index < len(self._qmodes) or index == self._qindex:
            return
        self._qindex = index
        self._modes = self._qmodes[index]
        if not self._reload_modes():
            self._apply_cell_view()  # couldn't reuse the view: rebuild it

    def _reload_modes(self) -> bool:
        """Put the current q's modes on the displayed structure, as it stands.

        Returns whether that was possible. It isn't when the composed modes no
        longer fit what is on screen — the structure may have been edited since
        it was built — in which case the caller falls back to a full rebuild,
        which re-derives both together.
        """
        try:
            _shown, modes, _cell, _analysis, _index = self._compose_view(
                self._cell_view, self._supercell, self._modes
            )
        except Exception:  # noqa: BLE001 - symmetry analysis can fail; rebuild instead
            return False
        if modes is None or len(modes) == 0:
            return False
        if modes[0].n_atoms != len(self.structure):
            return False  # edited since: the modes and the atoms no longer line up
        self.phonon_panel.set_modes(
            self.structure.positions, modes, self.structure.numbers
        )
        self._update_export_actions()
        return True

    def _tile_to_qpoint(self, reps) -> None:
        """Tile the cell to one whole period of the selected q — or undo that.

        The cell the view had before the panel first tiled it is remembered, so
        the button can put it back: tiling for one q and then moving to another
        (or to Gamma) must not strand the user in an enlarged cell whose only way
        out is the Supercell dialog. The restore point survives re-tiling for a
        different q — it is where the *panel* took over, not the last stop.
        """
        reps = tuple(int(r) for r in reps)
        if reps == self._tile_restore:
            self._tile_restore = None  # this is the way back; the panel is done
        elif self._tile_restore is None:
            self._tile_restore = self._supercell
        self._set_supercell(reps)
        self._apply_cell_view()

    # ── file actions ────────────────────────────────────────────────────
    def _open_file(self) -> None:
        """Single open: loads geometry, and phonon modes too if the file has them."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open structure file", "",
            "Structure files (*.out *.gui *.34 *.cif);;CRYSTAL files (*.out *.gui *.34);;"
            "CIF files (*.cif);;All files (*)",
        )
        if not path:
            return
        try:
            from crystalline.crystalio import load

            result = load(path)
        except Exception as exc:  # noqa: BLE001 - surface any parse error to the user
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        self._source = result.structure
        self._set_qmodes(result.qpoints if result.has_phonons else [])
        self._adps = self._load_adps(path)
        self._tile_restore = None  # nothing of the old file's tiling to go back to
        self._set_supercell((1, 1, 1))  # a fresh file starts at its own unit cell
        # Remember the output file so property plots (IR/Raman/elastic/EOS) can
        # read it directly; geometry-only files (.gui/.34/.cif) carry no such data.
        self._output_path = None if path.lower().endswith((".gui", ".34", ".cif")) else path
        self._apply_cell_view()
        # After the view exists (and its ADP tensors have been pushed), so
        # switching the ellipsoids on draws them straight away.
        self._update_adp_controls(autoshow=True)
        self._update_info(path)
        self._update_plot_actions()  # enable only the plots this file supports
        self._update_spectra_action()
        self._update_vci_action()
        self._update_anscan_action()
        self._update_pes_action()
        self._update_import_action()  # a structure is now loaded — allow importing

    def _import_atoms(self) -> None:
        """Read atoms from an .xyz/.pdb/.cif file and add them to the current structure.

        The atoms are appended at their cartesian coordinates (the source cell is
        ignored). It goes through the normal edit path, so it lands in the undo
        history and the view/panels refresh.
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Import atoms", "", "Atom files (*.xyz *.pdb *.cif);;All files (*)"
        )
        if not path:
            return
        try:
            from crystalline.crystalio import read_atoms

            atoms = read_atoms(path)
        except Exception as exc:  # noqa: BLE001 - surface any read/parse error
            QMessageBox.critical(self, "Import failed", f"Could not read atoms:\n{exc}")
            return
        symbols = list(atoms.get_chemical_symbols())
        if not symbols:
            QMessageBox.information(self, "Nothing imported", "No atoms found in that file.")
            return
        try:
            new = self.structure.add_atoms(symbols, atoms.get_positions())  # -> undo + redraw
        except Exception as exc:  # noqa: BLE001 - e.g. an unknown element symbol
            QMessageBox.critical(self, "Import failed", f"Could not add the atoms:\n{exc}")
            return
        self.display_panel.set_elements(self.structure.numbers)  # new elements may appear
        # Turn editing on and select the imported atoms so they can be dragged
        # into place as a whole straight away.
        if not self._editing:
            self._edit_mode_action.setChecked(True)  # toggles _set_editing(True)
        self.structure_panel.set_selection(new)

    def _update_info(self, path: str) -> None:
        """Refresh the crystallographic info panel for the loaded system."""
        props = {}
        try:
            from crystalline.crystalio import output_properties

            props = output_properties(path)
        except Exception:  # noqa: BLE001 - never let output parsing break loading
            props = {}
        self._output_props = props or {}
        self.info_panel.show_structure(self._source, self._output_props)

    def _refresh_info(self) -> None:
        """Re-analyse the shown structure so the Info panel tracks geometry edits.

        The displayed structure may be a supercell or boundary-completed, neither
        of which a symmetry finder can read directly, so it's folded back into the
        original unit cell first (see :func:`to_analysis_cell`). The CRYSTAL-output
        rows are carried over unchanged — they describe the loaded file, not the
        live geometry. Never let this abort an edit: it runs before the viewport
        redraws, so a symmetry-analysis hiccup must not swallow the redraw.
        """
        try:
            analysis = to_analysis_cell(self.structure, self._unit_cell)
            self.info_panel.show_structure(analysis, self._output_props)
        except Exception:  # noqa: BLE001 - info is advisory; never break the edit
            pass

    def _save_gui(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CRYSTAL .gui", "structure.gui", "CRYSTAL gui (*.gui)"
        )
        if not path:
            return
        try:
            from crystalline.crystalio import save_structure_gui

            save_structure_gui(self.structure, path)
        except Exception as exc:  # noqa: BLE001 - surface any write error to the user
            self._report_save_error(".gui file", exc)

    def _save_cif(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save CIF", "structure.cif", "CIF (*.cif)")
        if not path:
            return
        try:
            from crystalline.crystalio import save_structure_cif

            save_structure_cif(self.structure, path)
        except Exception as exc:  # noqa: BLE001 - surface any write error to the user
            self._report_save_error("CIF", exc)

    def _build_crystal_input(self) -> None:
        """Open the CRYSTAL ``.d12`` input builder for the current unit cell.

        Uses ``_source`` (the pristine loaded cell), not the displayed structure,
        so symmetry reduction sees the real unit cell rather than a supercell
        tiling or boundary-completed duplicates.
        """
        if len(self._source) == 0:
            QMessageBox.information(
                self, "Build CRYSTAL input", "Open or build a structure first."
            )
            return
        from crystalline.ui.panels.input_builder import InputBuilderDialog

        InputBuilderDialog(self._source, self).exec()

    def _report_save_error(self, what: str, exc: BaseException) -> None:
        """Tell the user a save failed — never with an empty dialog.

        pymatgen's symmetry errors (``SymmetryUndeterminedError``) carry no
        message at all, which used to render as a blank message box; fall back
        to the exception's type name so there is always something to report.
        """
        QMessageBox.critical(
            self, "Save failed", f"Could not save the {what}:\n{exc or type(exc).__name__}"
        )

    # ── export (image / animation) ──────────────────────────────────────
    def _export_image(self) -> None:
        """Save the current 3D view as an image, choosing format/scale/transparency."""
        options = self._ask_image_options()
        if options is None:
            return
        ext, label, scale, transparent = options
        path, _ = QFileDialog.getSaveFileName(
            self, "Export image", f"crystal_view.{ext}", f"{label} (*.{ext})"
        )
        if not path:
            return
        try:
            self.viewport.export_image(path, scale=scale, transparent=transparent)
        except Exception as exc:  # noqa: BLE001 - surface any render/write error
            QMessageBox.critical(self, "Export failed", f"Could not save the image:\n{exc}")

    def _ask_image_options(self):
        """Prompt for ``(ext, filter_label, scale, transparent)``; ``None`` if cancelled.

        Scale supersamples raster output (the 3D analogue of DPI); transparency
        needs an alpha channel, so both are greyed out for the formats that can't
        use them (vector, and opaque rasters like JPEG/BMP).
        """
        # (ext, menu label, is_vector, has_alpha)
        formats = [
            ("png", "PNG image", False, True),
            ("jpg", "JPEG image", False, False),
            ("tif", "TIFF image", False, True),
            ("svg", "SVG vector", True, False),
            ("pdf", "PDF vector", True, False),
            ("eps", "EPS vector", True, False),
        ]
        dialog = QDialog(self)
        dialog.setWindowTitle("Export image")
        form = QFormLayout(dialog)

        fmt_box = QComboBox(dialog)
        for ext, label, is_vector, has_alpha in formats:
            fmt_box.addItem(label, (ext, label, is_vector, has_alpha))
        form.addRow("Format:", fmt_box)

        scale_box = QSpinBox(dialog)
        scale_box.setRange(1, 8)
        scale_box.setValue(2)
        scale_box.setPrefix("×")
        scale_box.setToolTip("Supersampling: ×2 renders at twice the on-screen pixels each way.")
        form.addRow("Resolution:", scale_box)

        transparent = QCheckBox("Transparent background", dialog)
        form.addRow("", transparent)

        def sync_enabled() -> None:
            _ext, _label, is_vector, has_alpha = fmt_box.currentData()
            scale_box.setEnabled(not is_vector)
            transparent.setEnabled(not is_vector and has_alpha)

        fmt_box.currentIndexChanged.connect(sync_enabled)
        sync_enabled()

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() != QDialog.Accepted:
            return None
        ext, label, is_vector, _has_alpha = fmt_box.currentData()
        scale = 1 if is_vector else scale_box.value()
        want_transparent = transparent.isChecked() and transparent.isEnabled()
        return ext, label, scale, want_transparent

    def _export_animation(self) -> None:
        """Render the selected phonon mode over one cycle and save it.

        GIF needs no extra packages; a PNG filename writes a numbered frame
        sequence; MP4 needs imageio-ffmpeg (a clear message says so if missing).
        Rendered off-screen so the live view is untouched, reusing the current
        appearance and camera.
        """
        selection = self.phonon_panel.current_selection()
        if selection is None:
            QMessageBox.information(
                self, "No mode selected", "Select a vibrational mode to export its animation."
            )
            return
        options = self._ask_animation_options()
        if options is None:
            return
        ext, label, window_size, n_frames, fps = options
        path, _ = QFileDialog.getSaveFileName(
            self, "Export phonon animation", f"phonon_mode.{ext}", f"{label} (*.{ext})"
        )
        if not path:
            return
        equilibrium, mode = selection
        from crystalline.viz.export import (
            MOVIE_EXTS,
            VIDEO_MISSING_MESSAGE,
            render_animation_frames,
            save_animation,
            video_export_available,
        )

        # Checked before rendering, not after: the frames are the slow part, and
        # a missing encoder found at the end wastes the whole wait. The typed
        # filename decides the format, so this catches a video extension typed
        # by hand as well as one chosen in the dialog.
        if os.path.splitext(path)[1].lower() in MOVIE_EXTS and not video_export_available():
            QMessageBox.warning(self, "Video export unavailable", VIDEO_MISSING_MESSAGE)
            return

        try:
            frames = render_animation_frames(
                self.structure,
                equilibrium,
                mode,
                self.viewport.renderer.settings,
                amplitude=self.animator.amplitude,
                n_frames=n_frames,
                reference_cell=self._unit_cell,
                bond_structure=self._bond_structure,
                camera=self.viewport.camera_state,  # placement + zoom (parallel scale)
                window_size=window_size,
            )
            written = save_animation(frames, path, fps=fps)
        except Exception as exc:  # noqa: BLE001 - surface encode/write errors clearly
            QMessageBox.critical(self, "Export failed", f"Could not save the animation:\n{exc}")
            return
        if len(written) > 1:
            QMessageBox.information(
                self, "Animation exported", f"Wrote {len(written)} frames next to\n{written[0]}"
            )

    def _ask_animation_options(self):
        """Prompt for ``(ext, filter_label, window_size, n_frames, fps)`` or ``None``.

        Resolution sets the off-screen render size; frames control the smoothness
        of one vibration cycle; FPS the playback speed. A PNG target writes a
        numbered frame sequence rather than a single file.
        """
        from crystalline.viz.export import (
            DEFAULT_FPS,
            DEFAULT_FRAMES,
            VIDEO_MISSING_MESSAGE,
            video_export_available,
        )

        # Video needs an ffmpeg encoder that isn't part of a plain install. The
        # entries stay listed when it is missing — removing them would leave
        # someone hunting for MP4 with no explanation — but are disabled, and
        # say what to install.
        video = video_export_available()
        formats = [
            ("gif", "Animated GIF", True),
            ("mp4", "MP4 video", video),
            ("mov", "QuickTime video", video),
            ("webm", "WebM video", video),
            ("png", "PNG frame sequence", True),
            ("jpg", "JPEG frame sequence", True),
        ]
        resolutions = [
            ("640 × 480", (640, 480)),
            ("800 × 600", (800, 600)),
            ("1280 × 960", (1280, 960)),
            ("1920 × 1440", (1920, 1440)),
        ]
        dialog = QDialog(self)
        dialog.setWindowTitle("Export phonon animation")
        form = QFormLayout(dialog)

        fmt_box = QComboBox(dialog)
        for row, (ext, label, enabled) in enumerate(formats):
            fmt_box.addItem(label if enabled else f"{label}  — needs imageio-ffmpeg",
                            (ext, label))
            if not enabled:
                item = fmt_box.model().item(row)
                item.setEnabled(False)
                item.setToolTip(VIDEO_MISSING_MESSAGE)
        form.addRow("Format:", fmt_box)
        if not video:
            hint = QLabel("Video formats need <code>pip install imageio-ffmpeg</code>", dialog)
            hint.setStyleSheet("color: palette(mid);")
            form.addRow("", hint)

        res_box = QComboBox(dialog)
        for label, size in resolutions:
            res_box.addItem(label, size)
        res_box.setCurrentIndex(1)  # 800 × 600
        form.addRow("Resolution:", res_box)

        frames_box = QSpinBox(dialog)
        frames_box.setRange(8, 240)
        frames_box.setValue(DEFAULT_FRAMES)
        frames_box.setToolTip("Frames per vibration cycle — more is smoother but larger.")
        form.addRow("Frames:", frames_box)

        fps_box = QSpinBox(dialog)
        fps_box.setRange(1, 60)
        fps_box.setValue(DEFAULT_FPS)
        fps_box.setSuffix(" fps")
        form.addRow("Frame rate:", fps_box)

        # FPS is meaningless for a still-frame sequence; grey it out for those.
        def sync_enabled() -> None:
            fps_box.setEnabled(fmt_box.currentData()[0] not in ("png", "jpg"))

        fmt_box.currentIndexChanged.connect(sync_enabled)
        sync_enabled()

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() != QDialog.Accepted:
            return None
        ext, label = fmt_box.currentData()
        return ext, label, res_box.currentData(), frames_box.value(), fps_box.value()

    def _update_export_actions(self) -> None:
        """Enable animation export only when a phonon mode is selected."""
        action = getattr(self, "_export_anim_action", None)
        if action is not None:
            action.setEnabled(self.phonon_panel.has_mode())

    # ── cell view (always the crystallographic cell) ────────────────────
    def _apply_cell_view(self) -> None:
        """Rebuild the shown structure (and phonon modes) from the pristine source.

        Regenerating from the source (rather than transforming the currently
        shown cell) keeps the conversion well-defined and doubles as
        "restore geometry": any in-app edits to the shown structure are dropped.
        """
        try:
            shown, modes, unit_cell, analysis, source_index = self._compose_view(
                self._cell_view, self._supercell, self._modes
            )
        except Exception as exc:  # noqa: BLE001 - symmetry analysis can fail
            QMessageBox.warning(
                self, "Cell view unavailable", f"Could not build the crystallographic cell:\n{exc}"
            )
            # Fall back to the loaded cell as-is, which needs no analysis.
            shown, modes, unit_cell, analysis, source_index = self._compose_view(
                CellView.PRIMITIVE, self._supercell, self._modes
            )

        self._replace_structure(shown, unit_cell, analysis)
        self._adp_index = source_index
        self._refresh_adp_tensors()
        self._update_adp_controls()
        if modes is not None:
            self.phonon_panel.set_modes(
                self.structure.positions, modes, self.structure.numbers
            )
        else:
            self.phonon_panel.clear()
        self._update_export_actions()  # modes may have appeared/disappeared

    def _compose_view(self, view: CellView, supercell, modes):
        """Return ``(structure, modes, unit_cell, analysis, adp)`` for the view.

        Pipeline: crystallographic cell → supercell tiling → (optional) boundary
        completion. ``analysis`` is the clean periodic cell *before* boundary
        completion — it drives the bond/polyhedra coordination analysis, since
        the packed cell has duplicate images CrystalNN can't handle. ``unit_cell``
        is the pre-tiling cell, so the viewport keeps outlining the original cell.
        ``modes`` is ``None`` when the file has no phonons. ``source_index[i]``
        is the atom of ``self._source`` that displayed atom ``i`` is an image of;
        it rides the same replication as the modes, and is what lets a per-atom
        quantity (the ADP tensors) be laid onto the displayed geometry at any
        time without recomposing the view.
        """
        # Carrying the source atom *indices* through the pipeline, rather than
        # the ADP tensors themselves, keeps the cell operations ignorant of ADPs
        # and makes changing the temperature a re-index instead of a re-tile.
        base_index = np.arange(len(self._source))
        if view is CellView.CRYSTALLOGRAPHIC:
            # One expansion produces a consistent structure + modes + index triple.
            base, expanded, base_index = expand_modes_to_conventional(
                self._source, modes if modes is not None else PhononModes([]),
                per_atom=base_index,
            )
            base_modes = expanded if modes is not None else None
        else:
            base, base_modes = as_view(self._source, view), modes
        unit_cell = base.cell.copy()
        # A clean periodic cell (before boundary completion) drives the
        # chemically-aware bond/polyhedra analysis — the packed cell has
        # duplicate images a near-neighbour algorithm can't handle.
        analysis, analysis_modes, analysis_index = tile_supercell(
            base, supercell, base_modes, per_atom=base_index
        )
        if self._show_boundary:
            shown, shown_modes, shown_index = complete_boundary(
                analysis, analysis_modes, per_atom=analysis_index
            )
        else:
            shown, shown_modes, shown_index = analysis, analysis_modes, analysis_index
        return shown, shown_modes, unit_cell, analysis, shown_index

    def _replace_structure(self, new: Structure, unit_cell=None, bond_structure=None) -> None:
        """Swap in a structure to display, rebinding the existing panels.

        We rebind rather than recreate widgets so loading a file (or a view
        change) doesn't spawn duplicate docks (and keeps signals intact).
        ``unit_cell`` outlines the original cell; ``bond_structure`` is the clean
        cell used for the bond/polyhedra coordination analysis. Both are also kept
        on the window for later re-derives (animation export, symmetry re-analysis
        of edits), so they must track the structure being shown.
        """
        self.structure = new
        if unit_cell is not None:
            self._unit_cell = unit_cell
        if bond_structure is not None:
            self._bond_structure = bond_structure
        self.structure.add_listener(self._on_structure_changed)
        self.viewport.show_structure(
            self.structure, reference_cell=unit_cell, bond_structure=bond_structure
        )
        self.structure_panel.set_structure(self.structure)
        if hasattr(self, "geometry_panel"):
            self.geometry_panel.set_structure(self.structure)
        if hasattr(self, "symmetry_panel"):
            self.symmetry_panel.set_structure(self._symmetry_source())
        self._reset_undo()  # edits (and their undo history) don't cross a re-derive
        self._update_view_actions()  # a/b/c alignment depends on the cell just shown
        if hasattr(self, "display_panel"):
            self.display_panel.set_elements(self.structure.numbers)  # refresh element swatches


def _floating_plot_geometry(window: QRect, screen: Optional[QRect]) -> QRect:
    """Where the floating Plots window should sit, given the main window's frame.

    A 4:3 panel — matplotlib's own default figure ratio, so axes get room in both
    directions instead of the letterbox a bottom dock imposes — sized against the
    main window and tucked towards its lower right so the 3D view stays visible.
    Clamped to ``screen`` (when known) so it can't open partly off-display.
    """
    width = max(_PLOT_FLOAT_MIN_WIDTH, int(window.width() * _PLOT_FLOAT_WIDTH_FRACTION))
    height = int(round(width * 3 / 4))
    if screen is not None and not screen.isEmpty():
        width = min(width, screen.width())
        height = min(height, screen.height())

    margin = _PLOT_FLOAT_MARGIN
    x = window.x() + max(margin, window.width() - width - margin)
    y = window.y() + max(margin, window.height() - height - margin)
    if screen is not None and not screen.isEmpty():
        x = min(max(x, screen.x()), screen.right() - width + 1)
        y = min(max(y, screen.y()), screen.bottom() - height + 1)
    return QRect(x, y, width, height)


__all__ = ["MainWindow"]
