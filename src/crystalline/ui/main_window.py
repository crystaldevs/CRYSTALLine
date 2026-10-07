"""Main application window: a tab per open file in the centre, panels around it.

Wiring lives here and nowhere else — panels and the viewport expose signals,
and ``MainWindow`` connects them. Adding a new property (DOS, bands, elastic)
is: build a panel, dock it, connect its signals here.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import Optional, Sequence

import numpy as np
from PySide6.QtCore import QEvent, QRect, Qt, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSpinBox,
    QStackedWidget,
    QTabBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from dataclasses import dataclass

from crystalline.core.cells import (
    CellView,
    as_view,
    complete_boundary,
    expand_modes_to_conventional,
    tile_supercell,
    to_analysis_cell,
    to_conventional,
)
from crystalline.core.adp import ADPSet
from crystalline.core.phonons import PhononModes
from crystalline.core.structure import Structure
from crystalline.core.undo import UndoHistory

# Orbital energies are stored in Hartree and shown in the unit bands are quoted in.
_HARTREE_TO_EV = 27.211386245988


@dataclass(frozen=True)
class _Snapshot:
    """Everything undo has to put back to restore what was on screen.

    An ordinary edit — a drag, an added atom — changes the shown structure, and
    a snapshot of its atoms is enough to reverse it. Three Cell actions are not
    like that. A supercell and boundary completion change how the view is
    *derived* from the source; a cell-view switch changes it too; and a
    lattice-parameter edit changes the source itself. None of them touch the
    shown atoms in a way that can be replayed onto a differently-derived cell —
    an atom index in a 2x2x1 supercell does not mean what it meant in the unit
    cell — so each one used to clear the history instead, taking every edit
    before it down as well.

    Carrying the derivation alongside the atoms is what lets undo step back
    through the lot: restore the source, put the three view settings back,
    re-derive, and lay the shown atoms (edits included) over the result.
    """

    source: object        # ase.Atoms: the pristine cell the view derives from
    shown: object         # ase.Atoms: what is on screen, edits included
    view: CellView
    supercell: tuple
    boundary: bool
    # Whether ``shown`` carried anything the source did not, at the moment this
    # was taken. Restoring it has to put that back too: the restore itself looks
    # like an edit from the outside (it goes through the structure's listeners),
    # and a window left believing in edits that are not there treats the next
    # cell-setting switch as destructive and records a step for it.
    edited: bool = False
from crystalline.viz.phonon_animator import PhononAnimator
from crystalline.ui import menus
from crystalline.ui.file_tabs import (
    PER_TAB_PANELS,
    UNTITLED,
    FileTab,
    openable,
    route_to_active_tab,
    tab_labels,
    window_title,
)
from crystalline.ui.viewport import Viewport
from crystalline.ui.safety import guard
from crystalline.ui.widgets import BusyOverlay, DropHint, SlidingTabBar, Worker
from crystalline.ui.panels.structure_panel import StructurePanel
from crystalline.ui.panels.phonon_panel import PhononPanel
from crystalline.ui.panels.info_panel import InfoPanel
from crystalline.ui.panels.display_settings import DisplayPanel
from crystalline.ui.panels.geometry_panel import GeometryPanel
from crystalline.ui.panels.symmetry_panel import SymmetryPanel
from crystalline.ui.panels.plot_view import PlotPanel

_APPLICATION_TITLE = "CRYSTALLine — CRYSTAL structure & phonon viewer"

# Geometry of the floating Plots window on the first plot: a fraction of the main
# window's width (never below the minimum), 4:3, inset from its lower-right corner.
_PLOT_FLOAT_WIDTH_FRACTION = 0.55
_PLOT_FLOAT_MIN_WIDTH = 640
_PLOT_FLOAT_MARGIN = 24

# How far (cm⁻¹) a click may land from a mode and still be taken to mean it.
# Wide enough to forgive aiming at a broadened peak's flank, narrow enough that
# clicking empty baseline selects nothing.
_PEAK_PICK_TOLERANCE = 20.0


# Per-axis ceiling in the Supercell dialog. Generous rather than tuned: what
# actually costs is the total atom count, which is checked separately, and a
# slab or a polymer legitimately wants a big number down one axis.
_MAX_SUPERCELL_REPEAT = 99
# Above this many atoms the view is slow enough to be worth confirming first.
# Rotation is not the problem — it is display-locked and independent of size —
# but composing the view and the neighbour analyses behind bonds and polyhedra
# both grow with it.
_SLOW_SUPERCELL_ATOMS = 20_000


@route_to_active_tab
class MainWindow(QMainWindow):
    """The window: one tab per open file, docks around the tab shown.

    Each tab (:class:`~crystalline.ui.file_tabs.FileTab`) holds a file's state
    and its own 3D view and panels; the per-file names below — ``structure``,
    ``_source``, ``viewport``, ``phonon_panel`` and the rest listed in
    :data:`~crystalline.ui.file_tabs.PER_TAB` — are properties that reach the
    tab on screen, so everything here acts on the file being looked at.
    """

    def __init__(self, structure: Optional[Structure] = None) -> None:
        super().__init__()
        self.setWindowTitle(_APPLICATION_TITLE)
        from crystalline.resources import logo_path

        self.setWindowIcon(QIcon(logo_path()))
        self.resize(1200, 800)

        # Shared by every tab.
        self._tab: Optional[FileTab] = None
        self._tabs: list = []
        # Set while the window itself moves between tabs, so the Plots window
        # being hidden or shown on the way is not taken for the user's doing.
        self._switching_tabs = False
        self._editing = False
        # Workers in flight. Held because a dropped one is collected mid-run and
        # takes its QThread down with it.
        self._workers: list = []
        # Files waiting to be read, as (path, comes to the front), and the one
        # being read now: read one at a time, off the main thread (see _load_path).
        self._pending_reads: list = []
        self._reading = False
        self._reading_path: Optional[str] = None
        self._reader: Optional[Worker] = None
        # What each plot dialog was last set to, keyed by dialog. These are tuned
        # rather than answered — a broadening width, a frequency window — so
        # reopening one starts from the last accepted settings, not the defaults.
        self._plot_dialog_state: dict = {}
        self._axis_actions: list = []  # a/b/c view-alignment actions (menu + toolbar)

        # centre: one page per open file, each its own 3D view. Movable, so
        # files can be put side by side in the bar; closable, one at a time.
        self._file_tabs = QTabWidget(self)
        # Before any setting below, which the tab widget hands to its bar: a
        # swipe across the tabs moves their close buttons with them.
        self._file_tabs.setTabBar(SlidingTabBar(self._file_tabs))
        self._file_tabs.setObjectName("fileTabs")  # centred by the theme, over the view
        self._file_tabs.setTabsClosable(True)
        self._file_tabs.setMovable(True)
        self._file_tabs.setElideMode(Qt.ElideMiddle)  # keep both ends of a long run name
        self._file_tabs.setUsesScrollButtons(True)
        self.setCentralWidget(self._file_tabs)

        # The panels of every tab live in the docks, in a stack per dock that
        # shows the current tab's.
        self._panel_stacks = {name: QStackedWidget(self) for name in PER_TAB_PANELS}
        self._phonon_dock = self._dock(
            "Phonons", self._panel_stacks["phonon_panel"], Qt.RightDockWidgetArea
        )

        # left dock: crystallographic info of the loaded system
        info_dock = self._info_dock = self._dock(
            "Info", self._panel_stacks["info_panel"], Qt.LeftDockWidgetArea
        )

        # left dock (tabbed behind Info): live display settings.
        self._display_dock = self._dock(
            "Display", self._panel_stacks["display_panel"], Qt.LeftDockWidgetArea
        )
        self.tabifyDockWidget(info_dock, self._display_dock)

        # left dock (tabbed alongside): measurements + atom tools for the selection.
        self._geometry_dock = self._dock(
            "Geometry", self._panel_stacks["geometry_panel"], Qt.LeftDockWidgetArea
        )
        self.tabifyDockWidget(info_dock, self._geometry_dock)

        info_dock.raise_()  # show Info on top by default

        # right dock, tabbed with Phonons: the structure's point-symmetry elements.
        # Hidden until Cell ▸ Point symmetry analysis asks for it — it is an
        # analysis someone goes looking for, not something every session needs on
        # screen (and the search itself only runs once the panel is opened).
        self._symmetry_dock = self._dock(
            "Point symmetry", self._panel_stacks["symmetry_panel"], Qt.RightDockWidgetArea
        )
        self.tabifyDockWidget(self._phonon_dock, self._symmetry_dock)
        self._symmetry_dock.hide()

        # bottom dock: property plots (IR/Raman/bands/DOS…), one tab each.
        # Hidden until the first plot is built so it doesn't take up space.
        self._plot_dock = self._dock(
            "Plots", self._panel_stacks["plot_panel"], Qt.BottomDockWidgetArea
        )
        # The Plots window is the exception to the fixed layout. A figure is
        # looked at beside the structure, moved around, and shut when it has been
        # read — so it can be closed and dragged, and it is *not* allowed to dock:
        # without that it snaps back into the main window whenever it drifts near
        # an edge, which is maddening when the whole point is to place it.
        self._plot_dock.setAllowedAreas(Qt.NoDockWidgetArea)
        self._plot_dock.setFeatures(
            QDockWidget.DockWidgetClosable
            | QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
        )
        self._plot_dock.setTitleBarWidget(None)  # it keeps its own bar, and its ×
        self._plot_dock.hide()
        self._plot_dock_floated = False  # floated once, on the first plot built
        self._plot_dock.visibilityChanged.connect(self._on_plot_dock_visibility)

        # Bottom-right status indicators. Order matters: permanent widgets stack
        # left-to-right in call order, so counts sit left of the editing badge.
        self._count_status = QLabel()
        self._count_status.setStyleSheet("color: palette(mid); padding: 0 8px;")
        self.statusBar().addPermanentWidget(self._count_status)

        self._cell_status = QLabel()
        self._cell_status.setStyleSheet("color: palette(mid); padding: 0 8px;")
        self.statusBar().addPermanentWidget(self._cell_status)

        # Which file is being read, shown only while one is — where a loading
        # screen would have stood between the user and the tab already open.
        self._reading_status = QLabel()
        self._reading_status.setStyleSheet("color: palette(mid); padding: 0 8px;")
        self.statusBar().addPermanentWidget(self._reading_status)
        self._reading_status.hide()

        # Shown only while editing is on.
        self._editing_status = QLabel("● Editing mode")
        self._editing_status.setStyleSheet("color: #d9822b; font-weight: bold; padding: 0 8px;")
        self.statusBar().addPermanentWidget(self._editing_status)
        self._editing_status.hide()

        self._settle_docks()
        # After _settle_docks, which pins every tab bar it finds: this one is
        # the user's to reorder and to close tabs from.
        self._file_tabs.tabBar().setMovable(True)
        self._file_tabs.tabBar().setTabsClosable(True)
        self._file_tabs.currentChanged.connect(self._on_file_tab_changed)
        self._file_tabs.tabCloseRequested.connect(self._close_tab_at)
        # Long work shows this rather than a wait cursor, and runs off the main
        # thread so it can actually animate — see crystalline.ui.widgets.busy.
        self._busy = BusyOverlay(self)
        # Drag-and-drop: the window takes the drop, the hint sits over the 3D
        # views — the part of the window a file is aimed at.
        self.setAcceptDrops(True)
        self._drop_hint = DropHint(self._file_tabs)

        # The first tab: the structure handed in, or an empty one that the first
        # file opened takes over.
        self._add_tab(structure)
        # The theme is painted on the application before this window exists, so
        # the change that normally carries the 3D ground along with it has
        # already happened by the time there is a viewport to carry it to:
        # without this, launching in dark mode gave a dark app around a white
        # 3D view until the theme was next switched.
        self._follow_theme_background()
        menus.build_menus(self)
        self._reset_undo()
        self._refresh_chrome()  # menus, toolbar and status bar, for the tab on screen

    # ── file tabs ───────────────────────────────────────────────────────
    def _create_tab(self, structure: Optional[Structure] = None,
                    settings=None) -> FileTab:
        """A new tab holding ``structure`` (or nothing): its state and its page.

        Not its widgets. The page is an empty container the bar can show, name
        and close; the 3D view and the panels are built by :meth:`_realise_tab`
        the first time the tab is actually shown. A view is a VTK render window
        and costs ~100 ms to make, so opening several files at once would spend
        that on tabs nobody has looked at.

        Built *as* the current tab — the window's per-file names reach whatever
        ``self._tab`` is — and handed back without being shown; the caller adds
        it to the bar and switches to it. ``settings`` are the display settings
        it starts from: the tab open when it was made, so a new file comes up
        looking like the last one, and each goes its own way from there.
        """
        tab = FileTab()
        tab.page = QWidget(self._file_tabs)
        page_layout = QVBoxLayout(tab.page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        tab.start_settings = settings
        previous = self._tab
        self._tab = tab
        try:
            # The structure as loaded (CRYSTAL's primitive cell) is the pristine
            # source; ``self.structure`` is the cell view derived from it and
            # shown. By default we show the crystallographic (conventional) cell.
            self._source = structure if structure is not None else Structure.empty()
            self._cell_view = CellView.CRYSTALLOGRAPHIC
            self._supercell = (1, 1, 1)
            self._show_boundary = True  # show partially-belonging molecules by default
            # Whether the shown structure carries edits the source does not, so
            # that a re-derive can tell "nothing to lose" from "this discards work".
            self._shown_edited = False
            self._modes = None
            # Every q-point the run sampled (``[Gamma]`` for a plain FREQCALC,
            # one entry per commensurate q for a SCELPHONO run) and which of
            # them ``self._modes`` currently holds — the panel offers the
            # choice, the window rebuilds the view around it.
            self._qmodes = []
            self._qindex = 0
            # How many atoms the modes the panel was last given describe, so an
            # edit that puts that geometry back can have them back too. ``None``
            # when the panel has never been given any.
            self._modes_natom = None
            # Supercell to restore when the phonon panel's Untile is pressed, or
            # ``None`` when the tiling on screen is the user's own doing.
            self._tile_restore = None
            self._adps = None  # thermal ellipsoids, if the run has them
            # Undo history: snapshots of the shown cell and of how it was
            # derived (see _Snapshot), so Cell actions are undoable along with
            # edits. Only a new file starts it over.
            self._history = UndoHistory()
            self._suppress_undo = False
            self._output_path = None  # the loaded CRYSTAL .out, for plots
            self._output_props = {}  # parsed CRYSTAL-output rows for the Info panel
            # The orbital currently shown, if any: what it takes to rebuild it
            # when the displayed cell changes (a supercell shows more of one).
            self._orbital = None
            # Whether a scalar field (density, spin density, potential) is drawn.
            self._density_shown = False
            # ``_adp_index`` is the source atom each displayed atom is an image
            # of, so a per-atom quantity (the ADP tensors) can be laid onto the
            # expanded, tiled, boundary-completed cell on screen.
            (self.structure, _, self._unit_cell, self._bond_structure,
             self._adp_index) = self._compose_view(self._cell_view, self._supercell, None)

            self.structure.add_listener(self._routed(tab, self._note_edited))
            self.structure.add_listener(self._routed(tab, self._on_structure_changed))
        finally:
            self._tab = previous
        return tab

    def _realise_tab(self, tab: FileTab) -> bool:
        """Build ``tab``'s widgets, and put into them any file waiting to be shown.

        Called the first time a tab is shown, and only then: a tab opened
        alongside others and never looked at costs nothing but its page and the
        file it has read. Everything is built as the current tab, the same way
        and in the same order as when there was only ever one.

        Returns whether the widgets were built just now, for
        :meth:`_activate_tab` to settle them on screen (:meth:`_settle_tab`).
        """
        new = not tab.built()
        if new:
            with self._acting_on(tab):
                self._build_tab_widgets(tab)
        pending = tab.pending
        if pending is not None:
            tab.pending = None
            with self._acting_on(tab):
                self._fill_tab(tab, *pending)
        return new

    def _settle_tab(self, tab: FileTab) -> None:
        """Lay out and draw a tab on screen, before the window next reaches it.

        For a tab just built, or just given its file. Left to Qt, both happen a
        turn of the event loop later, and the window is shown in between: the
        Info panel with rows not laid out yet — scrollbars they will not need,
        light bars down and across a dark dock, or the Crystallography rows
        missing and the CRYSTAL output rows sitting where they should be — and
        the 3D view with nothing drawn in it, which macOS shows white. Each for
        one frame: a flash every time a tab is first looked at, whether by a
        click on it, by closing the tab in front of it or by opening its file.

        The layouts first, since the docks they settle can change the size of
        the view; then the view, with the file in it (it is on screen already,
        see :meth:`_build_tab_widgets`).
        """
        QApplication.sendPostedEvents(None, QEvent.LayoutRequest)
        tab.viewport.draw_now()

    def _build_tab_widgets(self, tab: FileTab) -> None:
        """One tab's 3D view and panels, wired to each other and to the window."""
        # The 3D view, inside the tab's page, and the phonon panel driving it.
        self.viewport = Viewport(tab.page)
        tab.page.layout().addWidget(self.viewport)
        if tab.page.isVisible():
            # The tab is being looked at, and Qt would show the view on the next
            # turn of the event loop: the window would reach the screen once
            # with a bare page where the view goes. And everything below would
            # be framed in a view that is not on screen yet, at a default size
            # it never has — a tall view got the framing of a wide one, and the
            # cell ran off both sides. Shown now, the page's layout gives it its
            # real size on the way; _settle_tab draws it once its file is in.
            self.viewport.show()
        if tab.start_settings is not None:
            self.viewport.renderer.set_settings(tab.start_settings)
        self.animator = PhononAnimator(self.viewport.renderer)
        self.phonon_panel = PhononPanel(self.animator, self)
        self.viewport.show_structure(
            self.structure, reference_cell=self._unit_cell,
            bond_structure=self._bond_structure,
        )

        # The Structure panel is kept as the selection/edit model (it owns
        # the shared selection and backs the Edit-menu tools) but is not
        # shown — 3D picking/drag and the Edit menu drive editing instead.
        self.structure_panel = StructurePanel(self.structure, self)
        self.structure_panel.hide()

        # Which cell the Info panel describes — the one computed, or
        # pymatgen's standard one — is remembered between sessions, and it is
        # also where the input builder starts.
        from crystalline.ui import preferences

        self.info_panel = InfoPanel(self)
        self.info_panel.set_cell_choice(preferences.cell_choice())
        self.info_panel.cell_choice_changed.connect(self._on_cell_choice_changed)
        self.info_panel.show_structure(self._analysis_cell())

        self.display_panel = DisplayPanel(
            self.viewport.renderer.settings,
            self._routed(tab, self._apply_render_settings), self,
        )
        self.display_panel.set_elements(self.structure.numbers)  # element swatches
        self.geometry_panel = GeometryPanel(self.structure, self)
        self.symmetry_panel = SymmetryPanel(self._analysis_cell(), self)
        self.plot_panel = PlotPanel(self)
        self._connect_tab_signals(tab)
        for name, stack in self._panel_stacks.items():
            stack.addWidget(getattr(tab, name))
        if tab.start_settings is None:
            self._follow_theme_background()  # a fresh ground matches the theme

    def _add_tab(self, structure: Optional[Structure] = None,
                 show: bool = True) -> FileTab:
        """Open a new tab (holding ``structure``, or empty), and switch to it.

        ``show=False`` sets it behind the tab on screen instead — a file read in
        while another is being worked in — and such a tab is never built, which
        is what makes opening ten files at once cost one 3D view rather than
        ten. (The window's first tab is shown whatever is asked: there is
        nothing else to show.)
        """
        # From the tab on screen, if it has a view to take them from: one that
        # has not been looked at yet has nothing of its own either.
        settings = None
        if self._tab is not None and self._tab.built():
            settings = self.viewport.renderer.settings
        tab = self._create_tab(structure, settings)
        self._tabs.append(tab)
        index = self._file_tabs.addTab(tab.page, UNTITLED)
        self._update_tab_labels()
        if show and self._file_tabs.currentIndex() != index:
            self._file_tabs.setCurrentIndex(index)  # -> _on_file_tab_changed
        if self._tab is None or (show and self._tab is not tab):
            self._activate_tab(tab)  # the first tab: Qt made it current on adding
        return tab

    def _tab_for_page(self, widget) -> Optional[FileTab]:
        for tab in self._tabs:
            if tab.page is widget:
                return tab
        return None

    def _on_file_tab_changed(self, index: int) -> None:
        tab = self._tab_for_page(self._file_tabs.widget(index)) if index >= 0 else None
        if tab is not None:
            self._activate_tab(tab)

    def _activate_tab(self, tab: FileTab) -> None:
        """Show ``tab``: its view, its panels, and the menus and status bar for it."""
        old = self._tab
        if old is not None and old is not tab and old in self._tabs and old.built():
            # A mode left playing in a tab nobody can see is work for nothing,
            # and would be found still running on the way back.
            old.phonon_panel.stop()
        self._tab = tab
        if self._file_tabs.currentWidget() is not tab.page:
            blocked = self._file_tabs.blockSignals(True)
            self._file_tabs.setCurrentWidget(tab.page)
            self._file_tabs.blockSignals(blocked)
        # Its page is on screen; now it needs something in it. A tab opened
        # beside others is built here, the first time it is looked at — before
        # the panels below are reached for, since this is what makes them.
        new = self._realise_tab(tab)
        for name, stack in self._panel_stacks.items():
            stack.setCurrentWidget(getattr(tab, name))
        # Editing is a mode of the window, not of a file: the tab arrived at
        # follows it.
        self.viewport.set_editing_enabled(self._editing)
        self.structure_panel.set_editing_enabled(self._editing)
        self.geometry_panel.set_editing_enabled(self._editing)
        # Each tab's plots come and go with it.
        self._switching_tabs = True
        try:
            if tab.plots_open and tab.plot_panel.count() > 0:
                self._plot_dock.show()
            elif self._plot_dock.isVisible():
                self._plot_dock.hide()
        finally:
            self._switching_tabs = False
        self._refresh_chrome()
        if new:
            self._settle_tab(tab)  # after everything above has had its say
        self._show_notice(tab)  # read in behind: what it could not read is said now

    def _take_tab_for_file(self) -> FileTab:
        """The tab a file coming to the front goes into: this one if it is empty,
        else a new one in front. The window opens on an empty tab, and the first
        file takes it over rather than leaving an empty page beside its own."""
        if self._tab is not None and self._tab.is_blank():
            return self._tab
        return self._add_tab()

    def _close_tab_at(self, index: int) -> None:
        tab = self._tab_for_page(self._file_tabs.widget(index))
        if tab is not None:
            self._close_tab(tab)

    def _close_current_tab(self) -> None:
        if self._tab is not None:
            self._close_tab(self._tab)

    def _close_tab(self, tab: FileTab) -> None:
        """Close ``tab`` and let its file go; an empty tab takes the last one's place."""
        if self._workers:
            # Something is being built for a tab — possibly this one — and its
            # result is on its way to that tab's widgets.
            self.statusBar().showMessage("Wait for the current task to finish.", 4000)
            return
        if len(self._tabs) == 1 and tab.is_blank():
            return  # nothing to close: the window always has a tab
        if tab.built():
            tab.phonon_panel.stop()
            tab.plot_panel.clear()  # releases the figures, and their pick handlers
        self._tabs.remove(tab)
        index = self._file_tabs.indexOf(tab.page)
        if index >= 0:
            self._file_tabs.removeTab(index)  # -> the neighbour becomes current
        if tab.built():
            for name, stack in self._panel_stacks.items():
                stack.removeWidget(getattr(tab, name))
            try:
                tab.viewport.interactor.close()  # releases the VTK render window
            except Exception:  # noqa: BLE001 - it is going away either way
                pass
        tab.pending = None  # a file read for a tab nobody looked at
        for widget in tab.widgets() + [tab.page]:
            if hasattr(widget, "deleteLater"):
                widget.deleteLater()
        if not self._tabs:
            self._tab = None
            self._add_tab()
        self._update_tab_labels()

    def _next_tab(self) -> None:
        self._step_tab(+1)

    def _previous_tab(self) -> None:
        self._step_tab(-1)

    def _step_tab(self, step: int) -> None:
        count = self._file_tabs.count()
        if count > 1 and not self._workers:
            self._file_tabs.setCurrentIndex((self._file_tabs.currentIndex() + step) % count)

    def _update_tab_labels(self) -> None:
        """Name every tab after its file — with the folder, where two share a name."""
        labels = tab_labels([tab.path for tab in self._tabs])
        # The window always keeps a tab, so the empty one it is left with has
        # nothing to close: no × on it that would do nothing.
        lone_blank = len(self._tabs) == 1 and self._tabs[0].is_blank()
        bar = self._file_tabs.tabBar()
        for tab, label in zip(self._tabs, labels):
            tab.label = label
            index = self._file_tabs.indexOf(tab.page)
            if index >= 0:
                self._file_tabs.setTabText(index, label)
                self._file_tabs.setTabToolTip(index, tab.path or "No file open")
                for side in (QTabBar.LeftSide, QTabBar.RightSide):  # left on macOS
                    button = bar.tabButton(index, side)
                    if button is not None:
                        button.setVisible(not lone_blank)
        self._update_window_title()

    def _update_window_title(self) -> None:
        """``run.out — CRYSTALLine``: the file on screen, named as its tab is."""
        tab = self._tab
        name = tab.label if tab is not None and tab.path else None
        self.setWindowTitle(window_title(name, _APPLICATION_TITLE))

    def _routed(self, tab: FileTab, slot):
        """``slot``, run against ``tab`` whichever tab is on screen when it fires.

        Almost everything a tab's widgets emit comes from the user working in
        the tab on screen. What does not — a display change applied after a
        short delay, the camera settling after a wheel zoom — belongs to the tab
        it came from all the same, not to whichever one has been switched to in
        between; and nothing at all is done for a tab that has been closed.
        """
        def call(*args):
            if tab is self._tab:
                return slot(*args)
            if tab not in self._tabs:
                return None
            with self._acting_on(tab):
                result = slot(*args)
            self._refresh_chrome()  # the menus are the tab on screen's, again
            return result

        return call

    @contextmanager
    def _acting_on(self, tab: FileTab):
        """Point the window's per-file names at ``tab`` for the duration."""
        previous = self._tab
        self._tab = tab
        try:
            yield tab
        finally:
            self._tab = previous

    def _refresh_chrome(self) -> None:
        """Set the menus, the toolbar and the status bar to the tab on screen."""
        if self._tab is None:
            return
        self._update_window_title()
        self._update_undo_action()
        self._update_view_actions()  # a/b/c alignment, and the cell-view controls
        self._update_boundary_control()
        self._update_supercell_action()
        if getattr(self, "_plot_actions", None) is not None:
            self._update_plot_actions()
        self._update_orbital_actions()
        self._update_density_actions()
        self._update_spectra_action()
        self._update_vci_action()
        self._update_anscan_action()
        self._update_pes_action()
        self._update_import_action()
        self._update_export_actions()
        if getattr(self, "_edit_tool_actions", None) is not None:
            self._update_edit_actions()
        self._update_status()

    def _capability(self, key: str):
        """What the open output offers for ``key`` (see :func:`_probe`), once per tab and file.

        Keyed by the output as well: a tab's file can change — an empty tab
        takes over the first file opened — and what was found for the one
        before must not answer for the next. A file opened through
        :meth:`_load_path` arrives with these already found, off the main thread.
        """
        cache = self._tab.capabilities
        entry = (key, self._output_path)
        if entry not in cache:
            cache[entry] = _probe(key, self._output_path)
        return cache[entry]

    def _on_cell_choice_changed(self, choice: str) -> None:
        """One Cell choice for every tab's Info panel, remembered for next time."""
        from crystalline.ui import preferences

        preferences.set_cell_choice(choice)
        for tab in self._tabs:
            if tab is not self._tab and tab.built():
                tab.info_panel.set_cell_choice(choice)

    def _on_plot_dock_visibility(self, visible: bool) -> None:
        """The user opening or shutting the Plots window, for the tab on screen."""
        if not self._switching_tabs and self._tab is not None:
            self._tab.plots_open = bool(visible)

    # ── wiring ──────────────────────────────────────────────────────────
    def _on_structure_changed(self, s: Structure) -> None:
        """Model edited: record undo, redraw, and reconcile a running animation."""
        self._capture_undo(s)
        if len(self._tabs) == 1:
            self._update_tab_labels()  # atoms built in the empty tab make it closable
        self._update_status()  # atom count may have changed (add/remove)
        self._update_import_action()  # importing needs a non-empty structure
        self._refresh_info()  # symmetry/point group may have changed with the edit
        self.symmetry_panel.invalidate(self._analysis_cell())  # and so may its elements
        # The Geometry panel's position boxes show a live atom, so they go stale
        # the moment anything moves it — a drag, an arrow-key nudge, an undo.
        self.geometry_panel.sync_position()
        self.geometry_panel.refresh_planes()  # atoms may have moved onto or off a plane
        self.viewport.renderer.refresh()
        # Editing the geometry: stop any animation and re-anchor it to the edited
        # geometry (drop the modes if the atom count changed). Passing the new
        # positions avoids resetting atoms back to the stale equilibrium, which
        # would undo the edit that was just made.
        self.phonon_panel.invalidate_on_edit(s.positions)
        self._reconcile_modes_with_edit(s)

    # ── undo ────────────────────────────────────────────────────────────
    def _snapshot(self) -> _Snapshot:
        """The shown state and the derivation that produced it — see :class:`_Snapshot`."""
        return _Snapshot(
            source=self._source.to_ase(),
            shown=self.structure.to_ase() if self.structure is not None else None,
            view=self._cell_view,
            supercell=tuple(self._supercell),
            boundary=bool(self._show_boundary),
            edited=bool(self._shown_edited),
        )

    def _capture_undo(self, _structure: Optional[Structure] = None) -> None:
        """Push the pre-change snapshot so this change can be undone.

        Each edit fires exactly one change notification; the baseline held from
        the previous one is the state *before* this change, so it's what an undo
        restores. Every re-derive of the view records here too, which is what
        makes a supercell or a lattice-parameter change undoable rather than the
        end of the timeline.

        Skipped while an undo is itself being applied, and when the state is the
        one already held: a rebuild that changed nothing — switching q-point
        falls back to one — would otherwise leave an undo step that puts nothing
        back.
        """
        if self._suppress_undo:
            return
        snapshot = self._snapshot()
        if snapshot == self._history.baseline:
            return
        self._history.record(snapshot)
        self._update_undo_action()

    def _reset_undo(self) -> None:
        """Clear the undo history and re-baseline to what is on screen.

        For a new file only: nothing about the last one is worth stepping back
        into. Everything else — edits and the Cell actions alike — records a
        snapshot instead.
        """
        self._history.reset(self._snapshot() if self.structure is not None else None)
        self._update_undo_action()

    def _undo(self) -> None:
        """Revert the most recent structure edit."""
        self._apply_history(self._history.undo())

    def _redo(self) -> None:
        """Re-apply the most recently undone structure edit."""
        self._apply_history(self._history.redo())

    def _apply_history(self, snapshot) -> None:
        """Restore a snapshot returned by undo()/redo() (shared plumbing).

        An edit is put back by restoring the shown atoms, which is cheap and
        leaves the view alone. When the derivation itself is what changed — a
        supercell, a cell view, boundary completion, or a lattice edit that
        moved the source under it — the source and the three view settings go
        back first and the view is rebuilt from them; the shown atoms are then
        laid over the result, since they may carry edits made after that
        derivation.
        """
        if snapshot is None:
            return
        self._suppress_undo = True
        try:
            if self._derivation_matches(snapshot):
                self.structure.restore(snapshot.shown)  # listeners redraw
            else:
                self._source.restore(snapshot.source)
                self._cell_view = snapshot.view
                self._set_supercell(snapshot.supercell)
                self._show_boundary = snapshot.boundary
                self._apply_cell_view()
                self.structure.restore(snapshot.shown)
                self._update_cell_view_controls()
                self._update_boundary_control()
            # After the restores, not before: each of them runs the structure's
            # listeners, and ``_note_edited`` is one of them.
            self._shown_edited = snapshot.edited
        finally:
            self._suppress_undo = False
        self.structure_panel.clear_selection()  # indices may no longer be valid
        self._update_undo_action()

    def _note_edited(self, _structure: Structure) -> None:
        """Remember that the shown cell now carries something the source does not.

        A listener of its own rather than a line in
        :meth:`_on_structure_changed`, so it holds for every change however it
        arrives, and so a caller that rebuilds the view cannot forget it.
        """
        self._shown_edited = True

    def _derivation_matches(self, snapshot) -> bool:
        """Whether the view is already derived the way ``snapshot`` was."""
        return (
            snapshot.view is self._cell_view
            and tuple(snapshot.supercell) == tuple(self._supercell)
            and bool(snapshot.boundary) == bool(self._show_boundary)
            and snapshot.source == self._source.to_ase()
        )

    def _update_boundary_control(self) -> None:
        """Tick the boundary entry for the view actually shown, without re-deriving it."""
        action = getattr(self, "_boundary_action", None)
        if action is not None:
            blocked = action.blockSignals(True)
            action.setChecked(self._show_boundary)
            action.blockSignals(blocked)

    def _update_undo_action(self) -> None:
        undo = getattr(self, "_undo_action", None)
        if undo is not None:
            undo.setEnabled(self._history.can_undo())
        redo = getattr(self, "_redo_action", None)
        if redo is not None:
            redo.setEnabled(self._history.can_redo())

    def _connect_tab_signals(self, tab: FileTab) -> None:
        """Wire one tab's view and panels: to each other, and to the window.

        What a tab's widgets ask of the window goes through :meth:`_routed`, so
        it is done to *that* tab even if it arrives once another is on screen.
        """
        on = lambda slot: self._routed(tab, slot)  # noqa: E731
        viewport, selection = tab.viewport, tab.structure_panel
        phonons, geometry = tab.phonon_panel, tab.geometry_panel
        display = tab.display_panel
        # viewport pick -> update selection (additive with Ctrl/Shift)
        viewport.atom_picked.connect(selection.select_atom)
        # dragging an atom in 3D -> update selection (keep a group drag intact)
        viewport.atom_moved.connect(on(self._on_atom_moved))
        # clicking empty space in the 3D view -> clear the panel selection
        viewport.selection_cleared.connect(selection.clear_selection)
        # Del/Backspace over the 3D view -> delete the selection (the menu's Del
        # shortcut can't fire while the VTK widget holds keyboard focus)
        viewport.delete_requested.connect(on(self._delete_selected))
        # arrow keys over the 3D view (editing mode) -> nudge the selection
        viewport.nudge_requested.connect(on(self._nudge_selection))
        # starting to drag an atom -> stop any running phonon animation
        viewport.interaction_started.connect(phonons.stop)
        # moving the camera -> suspend the animation for the duration, so the
        # drag gets the whole event loop and the view keeps up with the pointer
        viewport.camera_busy.connect(phonons.hold)
        # the panel owns the selection -> highlight it + refresh the Edit menu
        selection.selection_changed.connect(on(self._on_selection_changed))
        # a phonon mode was (de)selected -> refresh the animation-export action
        phonons.mode_selected.connect(on(lambda _row: self._update_export_actions()))
        # another q-point was picked -> show that q's modes on a rebuilt view
        phonons.qpoint_selected.connect(on(self._set_qpoint))
        # "Tile n×n×n" next to the q-point -> the supercell one period needs
        phonons.tile_requested.connect(on(self._tile_to_qpoint))
        # modes loaded, dropped or restored -> the arrow controls follow them,
        # as the ellipsoid controls follow a file's ADPs
        phonons.modes_changed.connect(on(display.set_phonons_available))

        # Geometry panel: the same edit operations as the Edit menu (so undo and
        # the selection model behave identically), plus measurement overlays.
        geometry.editing_toggled.connect(on(self._toggle_editing))
        geometry.delete_requested.connect(on(self._delete_selected))
        geometry.duplicate_requested.connect(on(self._duplicate_selected))
        geometry.translate_requested.connect(on(self._translate_selected))
        geometry.set_element_requested.connect(on(self._set_element_of_selection))
        geometry.set_position_requested.connect(on(self._set_position_of_selection))
        geometry.add_atom_requested.connect(on(self._add_atom))
        geometry.annotations_changed.connect(viewport.set_annotations)
        # Lattice planes (hkl), indices in the conventional cell; "Select atoms"
        # on a plane goes through the shared selection like a pick would.
        geometry.lattice_planes_changed.connect(viewport.set_lattice_planes)
        geometry.select_atoms_requested.connect(selection.set_selection)
        with self._acting_on(tab):
            geometry.set_miller_cell(self._miller_cell())
        # Symmetry panel: the ticked elements are drawn over the structure.
        tab.symmetry_panel.elements_changed.connect(viewport.set_symmetry_elements)
        tab.symmetry_panel.reduction_changed.connect(on(self._apply_symmetry_reduction))

    def _analysis_cell(self) -> Structure:
        """The shown structure folded back into one clean unit cell, edits included.

        Not the structure as displayed — a supercell has the wrong box and a
        boundary-completed view holds two copies of every atom that straddles the
        edge, and neither a symmetry finder nor a CRYSTAL deck can be built from
        that. Not the pristine loaded cell either: that has the right box but
        none of the user's edits. The fold gives both.

        Used by the point-symmetry search (the elements it finds repeat with the
        cell's lattice, so they still fill the drawn box however many cells are
        on screen) and by the input builder, which needs one cell to symmetrise.
        """
        try:
            return to_analysis_cell(self.structure, self._unit_cell)
        except Exception:  # noqa: BLE001 - fall back to the shown cell rather than nothing
            return self.structure

    def _dock(self, title: str, widget, area) -> QDockWidget:
        """Dock a panel: fixed in place, named by its tab rather than a title bar.

        Panels are not movable, floatable or closable by hand. The layout is part
        of the design — where the Display panel is relative to the view is not a
        preference — and a dock dragged out or shut by accident was work to get
        back. Visibility stays available, deliberately, in View ▸ Panels.
        """
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(area, dock)
        return dock

    def _settle_docks(self) -> None:
        """Put the tabs at the foot of every dock area, and stop them being dragged.

        Called once the tabifying is done. A tabbed dock is named twice — a title
        bar above it and a tab below — so the title bar comes off and the tab is
        the name. A dock that ends up alone in its area has no tab, so it keeps
        its title bar or it would have no label at all.
        """
        from PySide6.QtWidgets import QTabBar, QTabWidget

        for area in (
            Qt.LeftDockWidgetArea, Qt.RightDockWidgetArea,
            Qt.TopDockWidgetArea, Qt.BottomDockWidgetArea,
        ):
            self.setTabPosition(area, QTabWidget.South)

        for _title, dock in self._panel_docks():
            if self.tabifiedDockWidgets(dock):
                dock.setTitleBarWidget(_no_title_bar(dock))

        for bar in self.findChildren(QTabBar):
            bar.setMovable(False)      # the tab order is not the user's to shuffle
            bar.setTabsClosable(False)

    def _show_about(self) -> None:
        menus.show_about(self)

    def _set_cell_view(self, view: CellView) -> None:
        """Draw the crystallographic (conventional) cell, or the primitive one.

        Re-derived from the pristine source through the ordinary pipeline, so a
        shown orbital is rebuilt over whichever cell this makes — the two must
        never drift apart. Editing the shown structure and then switching is a
        re-derive, so those edits are dropped, as they are for every other Cell
        action.
        """
        if view is self._cell_view:
            return
        self._cell_view = view
        # Which of the two settings a crystal is drawn in is a way of looking at
        # it, not a change to it, so switching is no undo step of its own and
        # Ctrl+Z goes on belonging to the edits. The baseline still moves with
        # it, or the next edit would push the *old* setting onto the stack and
        # undoing that edit would flip the cell as well.
        #
        # Unless there were edits to lose. This re-derives from the source, so
        # anything done to the shown structure since it was built is dropped
        # here — that is a change, it is the only copy of that work, and it is
        # recorded so undo can step back to it, setting and all.
        if self._shown_edited:
            self._apply_cell_view()
        else:
            self._suppress_undo = True
            try:
                self._apply_cell_view()
            finally:
                self._suppress_undo = False
            self._history.rebase(self._snapshot())
            self._update_undo_action()
        self._update_cell_view_controls()

    def _update_cell_view_controls(self) -> None:
        """Point the menu entries and toolbar chips at the cell actually shown.

        Set without re-emitting: they are a display of the current view, and
        echoing a change back would re-derive it a second time — and for the
        toolbar switch, would fight the user's click.
        """
        for view, action in getattr(self, "_cell_view_actions", {}).items():
            blocked = action.blockSignals(True)
            action.setChecked(view is self._cell_view)
            action.blockSignals(blocked)
        switch = getattr(self, "_conventional_switch", None)
        if switch is not None:
            blocked = switch.blockSignals(True)
            switch.setChecked(self._cell_view is CellView.CRYSTALLOGRAPHIC)
            switch.blockSignals(blocked)

    # ── appearance ──────────────────────────────────────────────────────
    def _set_appearance(self, mode: str) -> None:
        """Switch the app between the system, light and dark looks, and remember it."""
        from PySide6.QtWidgets import QApplication

        from crystalline.ui import theme

        theme.set_mode(QApplication.instance(), mode)
        for name, action in getattr(self, "_appearance_actions", {}).items():
            blocked = action.blockSignals(True)
            action.setChecked(name == mode)
            action.blockSignals(blocked)

    def _toggle_appearance(self) -> None:
        """Flip between the light and dark themes from the toolbar lamp.

        An explicit choice, not a nudge to the system one: pressing it while
        following the desktop settles on whichever of the two is *not* showing,
        which is what someone reaching for the lamp is asking for. "Match system"
        stays available in View ▸ Appearance.
        """
        from PySide6.QtWidgets import QApplication

        from crystalline.ui import theme

        showing_dark = theme.palette_for(QApplication.instance()) is theme.DARK
        self._set_appearance("light" if showing_dark else "dark")

    # kept for the menu route; the toolbar drives _set_appearance directly

    def refresh_theme(self) -> None:
        """Redraw anything that baked a theme colour in.

        The theme calls this on every window after a change. A stylesheet
        restyles itself, but the toolbar's glyphs are rendered once in the text
        colour — left alone they would stay dark on a dark toolbar — and the
        lamp's tooltip names the theme it would switch *to*.
        """
        from crystalline.ui.menus import refresh_appearance_button, refresh_history_icons

        refresh_history_icons(self)
        refresh_appearance_button(self)
        for tab in getattr(self, "_tabs", []):
            if not tab.built():
                continue  # nothing to restyle yet; it is built in the new theme
            for panel in (tab.phonon_panel, tab.geometry_panel):
                refresh = getattr(panel, "refresh_theme_icons", None)
                if callable(refresh):
                    refresh()
            with self._acting_on(tab):
                self._follow_theme_background()

    def _follow_theme_background(self) -> None:
        """Move the 3D ground to match the app theme — unless it is the user's.

        A dark chrome around a white viewport reads as a bug rather than a
        choice. But the background is a real setting in Display ▸ Scene, so a
        colour someone chose deliberately must survive a theme change: only a
        ground that is still one of the themes' own is moved.
        """
        from PySide6.QtWidgets import QApplication

        from crystalline.ui import theme

        from PySide6.QtGui import QColor

        panel = getattr(self, "display_panel", None)
        if panel is None:
            return

        def same(a: str, b: str) -> bool:
            """Compare as colours, not as strings.

            The default ground is spelled ``"white"`` and a theme's is
            ``"#ffffff"`` — the same colour written two ways, and comparing the
            text would take a fresh app's untouched background for a deliberate
            choice and never move it.
            """
            first, second = QColor(a), QColor(b)
            return first.isValid() and second.isValid() and first.rgb() == second.rgb()

        if not any(same(panel.background(), ground) for ground in theme.scene_backgrounds()):
            return  # chosen by the user; leave it alone
        panel.set_background(theme.active_palette(QApplication.instance()).scene)

    def _update_view_actions(self) -> None:
        """Enable a/b/c view alignment only when there's a cell to align to."""
        enabled = self.viewport.can_align_axes()
        for widget in (*self._axis_actions, *getattr(self, "_axis_buttons", [])):
            widget.setEnabled(enabled)
        self._update_cell_view_controls()

    def _update_plot_actions(self) -> None:
        """Enable each plot action according to the loaded file.

        Output plots (IR/Raman/elastic/EOS) are enabled only when the loaded
        CRYSTAL output actually contains their data; data plots (bands/DOS/XRD)
        stay enabled — they read a separate file the user chooses.
        """
        available = self._capability("plots")
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
            # Start in the folder of the loaded output: a properties run leaves
            # its data files beside the output it was run from, so that is where
            # the file being asked for almost always is.
            start = os.path.dirname(self._output_path) if self._output_path else ""
            path, _ = QFileDialog.getOpenFileName(
                self, kind.caption, start, kind.file_filter)
            if not path:
                return
        # Some of these take seconds — an elastic surface is evaluated over a
        # grid of directions — so the build goes to a thread and the window shows
        # the busy overlay instead of locking up.
        #
        # Only the *figure* is built there. CRYSTALClear draws on matplotlib's
        # Agg backend (forced in crystalio.plotting), which needs no display and
        # no Qt; the canvas that wraps it is created here, on the UI thread,
        # where every Qt object belongs.
        def build():
            return kind.build(path)

        def show(figure) -> None:
            # No pick handler: none of these plots is drawn against a wavenumber
            # axis. The spectra dialog's figures are, and pass one themselves.
            self.plot_panel.add_figure(figure, kind.label.rstrip("… "))
            self._reveal_plot_dock()

        self._run_busy(build, f"Building {kind.label.rstrip('… ').lower()}…",
                       show, "Plot failed")

    # ── thermal ellipsoids (ADP) ────────────────────────────────────────
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
        self._restore_dialog(dialog, "spectra")
        if dialog.exec() != QDialog.Accepted:
            return
        self._remember_dialog(dialog, "spectra")
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

    # ── electronic bands and DOS ────────────────────────────────────────
    def _open_electronic(self) -> None:
        """A band structure, a DOS or both, with every option in view.

        They used to be two one-click entries drawn at CRYSTALClear's defaults —
        an x axis of k-distances, energies only relative, no way to put the two
        side by side. They read files of their own, from a .d3 run, so the
        dialog looks beside the loaded output for them first.
        """
        from pathlib import Path

        from crystalline.crystalio import plot_electronic
        from crystalline.ui.panels.electronic_dialog import ElectronicDialog

        structure = None
        current = getattr(self, "structure", None)
        if current is not None and len(current):
            try:
                # The analysis cell, so the path's corners are recognised in
                # the same cell the band path was written for.
                structure = self._analysis_cell()
            except Exception:  # noqa: BLE001 - naming the corners is a nicety
                structure = None
        # Beside the file this tab was opened from, whatever kind it is. Taking
        # it from the *output* path left a geometry (.cif, .gui, fort.34) with
        # nowhere to look — the dialog opened on no folder and offered nothing,
        # with the grids sitting next to the very file on screen.
        opened = self._output_path or (self._tab.path if self._tab is not None else None)
        folder = str(Path(opened).parent) if opened else ""
        # The run's own files are told apart by the Fermi level they record,
        # not by being named after the output.
        try:
            efermi = float(getattr(self, "_output_props", {}).get("Fermi energy (eV)"))
        except (TypeError, ValueError):
            efermi = None
        dialog = ElectronicDialog(structure=structure, folder=folder,
                                  parent=self, efermi=efermi)
        self._restore_dialog(dialog, "electronic")
        if dialog.exec() != QDialog.Accepted:
            return
        self._remember_dialog(dialog, "electronic")
        bands_path, dos_path, options = dialog.request()
        title = dialog.plot_title()

        def build():
            return plot_electronic(bands_path, dos_path, options)

        def show(figure) -> None:
            self.plot_panel.add_figure(figure, title)
            self._reveal_plot_dock()

        self._run_busy(build, "Building the electronic structure plot…", show, "Plot failed")

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
        self._restore_dialog(dialog, "vci")
        if dialog.exec() != QDialog.Accepted:
            return
        self._remember_dialog(dialog, "vci")
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
        self._restore_dialog(dialog, "anscan")
        if dialog.exec() != QDialog.Accepted:
            return
        self._remember_dialog(dialog, "anscan")
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
        action = getattr(self, "_anscan_action", None)
        if action is not None:
            action.setEnabled(self._capability("anscan"))

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
        self._restore_dialog(dialog, "pes")
        if dialog.exec() != QDialog.Accepted:
            return
        self._remember_dialog(dialog, "pes")
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
        action = getattr(self, "_pes_action", None)
        if action is not None:
            action.setEnabled(self._capability("pes"))

    # ── plot typography ─────────────────────────────────────────────────
    def _open_orbitals(self) -> None:
        """Draw a crystalline orbital from an ORBITALS run over the structure.

        The Molden files a ``PROPERTIES``/``ORBITALS`` run writes sit beside the
        loaded output, so they are found rather than asked for — as the ANSCAN
        wavefunctions already are.

        The chosen file's own cell replaces what is displayed. The orbital is
        expanded on *that* cell's atoms, so drawing it over anything else (a
        supercell, a boundary-completed view, a different file) would put the
        lobes on the wrong atoms.

        Away from Γ the dialog can also tile the view to one whole period of the
        orbital's k, which is where the phase relation between cells becomes
        visible — the same thing the phonon panel's Tile does for a mode at that
        q, for the same reason.
        """
        from crystalline.crystalio import molden
        from crystalline.ui.panels.orbital_dialog import OrbitalDialog

        found = molden.find_orbital_files(self._output_path) if self._output_path else []
        if not found:
            QMessageBox.information(
                self, "No orbitals found",
                "No ORBITALS Molden files were found beside the loaded output.\n\n"
                "They are written by a PROPERTIES run with the ORBITALS keyword, "
                "one per sampled k-point, and are expected in the same folder as "
                "the .out file.",
            )
            return

        dialog = OrbitalDialog(found, self, view_cell=self._orbital_view_cell(found))
        self._restore_dialog(dialog, "orbitals")
        if dialog.exec() != QDialog.Accepted:
            return
        self._remember_dialog(dialog, "orbitals")
        choice = dialog.selection()

        try:
            data = molden.load(choice["path"])
            imaginary = (
                molden.load(choice["imaginary_path"]) if choice["imaginary_path"] else None
            )
        except Exception as exc:  # noqa: BLE001 - surface any read error
            QMessageBox.critical(
                self, "Orbital unavailable", f"Could not read the orbital file:\n{exc}"
            )
            return

        # The previously shown orbital goes first, and the view is settled
        # second: every step below re-derives the displayed cell, and each one
        # would otherwise kick off a full rebuild of an orbital that is about to
        # be replaced anyway — which is no longer cheap now that a tiled orbital
        # is sampled as finely per cell as a single one.
        self._clear_orbital()
        self._show_orbital_structure(data)
        tiling = self._tile_to_period(choice["kpoint"], data.cell) if choice["tile"] else None
        self._orbital = {
            "data": data,
            "imaginary": imaginary,
            "index": choice["index"],
            "isovalue": choice["isovalue"],
            "samples": choice["samples"],
            "kpoint": choice["kpoint"],
            "draw": choice["draw"],
            "phase": choice["phase"],
        }
        if not self._refresh_orbital():
            self._orbital = None
            return
        energy = data.orbitals[choice["index"]].energy * _HARTREE_TO_EV
        drawn = "amplitude" if imaginary is None else {
            "phase": "|ψ|, coloured by cell phase",
            "modulus": "|ψ|",
        }.get(choice["draw"], "amplitude")
        self.statusBar().showMessage(
            f"Orbital {choice['index'] + 1} of {len(data.orbitals)}"
            f"  ·  {energy:.3f} eV"
            f"  ·  {drawn}"
            f"  ·  surface at {choice['isovalue']:.0%}"
            + ("" if tiling is None else "  ·  tiled {}×{}×{}".format(*tiling)),
            10000,
        )

    def _orbital_view_cell(self, paths) -> Optional[np.ndarray]:
        """The unit cell an orbital from ``paths`` will be displayed on.

        The dialog needs it to say how many cells one period of k spans, and it
        cannot know: the Molden file carries the primitive cell, and the view
        shows the crystallographic one unless told otherwise. Derived the same
        way :meth:`_compose_view` derives it, from the same two calls, so the
        number named in the dialog is the number that will be tiled.

        ``None`` if the file can't be read — the dialog then offers no tiling
        rather than offering a wrong one.
        """
        from ase import Atoms

        from crystalline.crystalio import molden

        for path in paths:
            try:
                data = molden.load(path)
            except Exception:  # noqa: BLE001 - try the next file, then give up
                continue
            if data.cell is None:
                return None
            primitive = Structure.from_ase(
                Atoms(numbers=data.numbers, positions=data.positions,
                      cell=np.asarray(data.cell, dtype=float), pbc=True)
            )
            try:
                if self._cell_view is CellView.CRYSTALLOGRAPHIC:
                    return to_conventional(primitive).cell.copy()
                return as_view(primitive, self._cell_view).cell.copy()
            except Exception:  # noqa: BLE001 - no symmetry found: the file's own cell
                return np.asarray(data.cell, dtype=float)
        return None

    def _tile_to_period(self, kpoint, cell) -> Optional[tuple]:
        """Put one whole period of ``kpoint`` on screen.

        The same offer the phonon panel's Tile makes for a mode at that q, and
        the tiling is worked out here rather than in the dialog because it
        depends on the cell being shown: k is quoted against the Molden file's
        (primitive) lattice, and the crystallographic cell of a rhombohedral
        crystal already holds three of those — one hexagonal cell of corundum
        spans a whole period of k = (1/3, 1/3, 1/3), where three primitive cells
        along each axis would be asked for otherwise.

        ``kpoint`` is fractional against ``cell``, the lattice it was quoted on.
        Returns the tiling applied, or ``None`` if one cell already shows a
        period; the view is left alone in that case.
        """
        from crystalline.core.orbitals import commensurate_repeats_in

        reps = commensurate_repeats_in(kpoint, cell, self._unit_cell)
        if reps == (1, 1, 1):
            return None
        self._tile_restore = None  # this tiling is the user's, not the panel's
        self._set_supercell(reps)
        self._apply_cell_view()
        return reps

    def _run_busy(self, work, message: str, done, failed_title: str) -> None:
        """Run ``work`` off the main thread behind the busy overlay.

        ``done`` receives the result back on the UI thread, which is where any
        drawing has to happen: VTK and Qt widgets belong to the thread that made
        them, so only the computation moves.

        It receives it for the tab the work was started in, whichever tab is on
        screen by then (:meth:`_routed`). The tab bar is disabled meanwhile, but
        that only stops the *user* moving: a file read in behind the work brings
        its own tab to the front when it is ready, and without this the orbital
        or the plot would be drawn into that newly opened file instead.
        """
        self._busy.start(message)
        started_in = self._tab
        worker = Worker(work)
        self._workers.append(worker)  # held: a dropped worker takes its thread down
        self._file_tabs.tabBar().setEnabled(False)

        def cleanup() -> None:
            self._busy.stop()
            if worker in self._workers:
                self._workers.remove(worker)
            if not self._workers:
                self._file_tabs.tabBar().setEnabled(True)

        def on_failed(exc) -> None:
            cleanup()
            QMessageBox.critical(self, failed_title, str(exc))

        deliver = self._routed(started_in, done)
        worker.finished.connect(lambda result: (cleanup(), deliver(result)))
        worker.failed.connect(on_failed)
        worker.start()

    def _refresh_orbital(self) -> bool:
        """(Re)build the shown orbital over the displayed cell. False if it failed.

        Called whenever the displayed cell changes — a supercell, or a switch
        between the primitive and crystallographic views — so the orbital always
        covers exactly the cell being drawn — which is the way to see more of one than
        a single cell holds. The orbital is re-evaluated over the larger box
        rather than the single-cell result being tiled: away from Gamma it picks
        up the phase e^(2πi k·T) from cell to cell, and only its modulus repeats.
        """
        if self._orbital is None:
            return False
        from crystalline.core import orbitals as orbital_maths

        chosen = dict(self._orbital)
        box_cell, repeat = self._unit_cell, self._supercell

        def build():
            # Pure numpy on data already parsed — safe away from the UI thread.
            return orbital_maths.evaluate(
                chosen["data"], chosen["index"], samples=chosen["samples"],
                # The cell actually on screen — the crystallographic one unless
                # the view says otherwise — and the supercell of it being shown.
                box_cell=box_cell, repeat=repeat, imaginary=chosen["imaginary"],
                # k is what makes the images add with a phase instead of in
                # step; without it this would draw the Γ orbital of these
                # coefficients, which is a different orbital entirely.
                kpoint=chosen.get("kpoint"),
                draw=chosen.get("draw", orbital_maths.AMPLITUDE),
                phase=chosen.get("phase", 0.0),
            )

        def draw(field) -> None:
            self.viewport.renderer.set_orbital(field, chosen["isovalue"])
            self._update_orbital_actions()

        self._run_busy(build, "Building the orbital…", draw, "Orbital unavailable")
        return True

    def _show_orbital_structure(self, data) -> None:
        """Display the cell the orbital is defined on, replacing the current view.

        The Molden file carries its own geometry, so that becomes the source the
        view is derived from — but the *view* is left alone, crystallographic by
        default. The orbital is then sampled over whichever cell is on screen
        (see :meth:`_refresh_orbital`): the conventional cell is a different
        region of the same crystal, not a different crystal, and the Bloch sum
        runs over the file's own lattice either way.

        What must not happen is the two drifting apart — the orbital sampled on
        the file's rhombohedral cell while the view shows the hexagonal
        conventional one, three times the volume, which put lobes where no atom
        was and atoms where no lobe was.

        The supercell is deliberately *kept*: it is how you ask to see more of an
        orbital than one cell holds, and the orbital is rebuilt across it.
        """
        from ase import Atoms

        self._source = Structure.from_ase(
            Atoms(
                numbers=data.numbers, positions=data.positions,
                cell=np.asarray(data.cell, dtype=float), pbc=True,
            )
        )
        self._modes = None
        self._qmodes = []
        self._qindex = 0
        self._adps = None
        self.phonon_panel.clear()
        self._apply_cell_view()
        self._update_view_actions()

    def _open_density(self) -> None:
        """Draw a charge density, a spin density or a potential from an ECH3 run.

        The grids sit beside the SCF output — ``DENS_CUBE.DAT``, ``POT_CUBE.DAT``,
        ``fort.31`` — so the dialog looks there first, as the orbital and band
        entries do. What comes back is CRYSTAL's own field on CRYSTAL's own
        grid: it is drawn over the structure as it is, with no resampling, so
        what is on screen is what the run computed.
        """
        from pathlib import Path

        from crystalline.crystalio import density
        from crystalline.ui.panels.density_dialog import DensityDialog

        # Beside the file this tab was opened from, whatever kind it is. Taking
        # it from the *output* path left a geometry (.cif, .gui, fort.34) with
        # nowhere to look — the dialog opened on no folder and offered nothing,
        # with the grids sitting next to the very file on screen.
        opened = self._output_path or (self._tab.path if self._tab is not None else None)
        folder = str(Path(opened).parent) if opened else ""
        miller_cell = self._conventional_cell()
        source = getattr(self, "_source", None)
        cell = (np.asarray(source.cell, dtype=float)
                if source is not None and len(source) and source.is_periodic else None)
        dialog = DensityDialog(folder=folder, parent=self,
                               miller_cell=miller_cell, cell=cell)
        self._restore_dialog(dialog, "density")
        if dialog.exec() != QDialog.Accepted:
            return
        self._remember_dialog(dialog, "density")

        field, options = dialog.request()
        if field is None:
            return
        try:
            self.viewport.renderer.set_density(field, options, miller_cell=miller_cell)
            if options.view == density.SLICE:
                # A plane is read face-on; seen edge-on it is a line.
                self.viewport.renderer.face_density_plane()
        except Exception as exc:  # noqa: BLE001 - surface any drawing failure
            QMessageBox.critical(
                self, "Field unavailable", f"Could not draw the field:\n{exc}"
            )
            return
        self._density_shown = True
        self._update_density_actions()
        if options.view == density.SLICE:
            from crystalline.core.lattice_planes import label

            told = f"{field.name} · {label(options.miller, miller_cell)} plane"
        else:
            told = f"{field.name} · isovalue {options.isovalue:.4g} {field.unit}"
        self.statusBar().showMessage(told, 6000)

    def _conventional_cell(self):
        """The conventional cell of the loaded structure, or ``None``.

        Miller indices are quoted in it. Taken from the structure as loaded,
        not from the view: a supercell or a primitive view must not change what
        (001) means. It is the same construction the conventional view is built
        with, so the plane lands in the frame the atoms are drawn in. Best
        effort: without it the renderer falls back to the displayed cell.
        """
        structure = getattr(self, "_source", None)
        if structure is None or not len(structure) or not structure.is_periodic:
            return None
        try:
            from crystalline.core.cells import to_conventional

            return np.asarray(to_conventional(structure).cell, dtype=float)
        except Exception:  # noqa: BLE001 - symmetry analysis is a nicety here
            return None

    def _miller_cell(self):
        """The cell the Geometry panel's lattice planes are indexed in, or ``None``.

        The conventional cell, as for the density slice — so (hkl) means the
        same plane there and in the view — falling back to the unit cell on
        screen when that cannot be built. A molecule has none.
        """
        structure = getattr(self, "_source", None)
        if structure is None or not len(structure) or not structure.is_periodic:
            return None
        cell = self._conventional_cell()
        if cell is None:
            unit = getattr(self, "_unit_cell", None)
            cell = None if unit is None else np.asarray(unit, dtype=float)
        if cell is None or cell.shape != (3, 3) or abs(np.linalg.det(cell)) < 1e-8:
            return None
        return cell

    def _clear_density(self) -> None:
        """Take a shown field off the view."""
        self.viewport.renderer.set_density(None)
        self._density_shown = False
        self._update_density_actions()

    def _update_density_actions(self) -> None:
        """Enable the field entry once the tab holds a file, Clear while one is drawn.

        The grids sit beside the file the tab was opened from, so without one
        there is nowhere to look: the dialog opened on no folder and listed
        nothing. That was only reachable while the window was reading — which it
        now does in the background, leaving every menu live over a tab whose
        file has not arrived yet, where opening a file used to hold the whole
        window still. Every other entry that needs the file is already enabled
        from what was read; this one is enabled by there being a file at all,
        so a geometry (.cif, .gui) keeps it and its Browse button.
        """
        tab = self._tab
        field = getattr(self, "_density_action", None)
        if field is not None:
            ready = tab is not None and bool(tab.path)
            field.setEnabled(ready)
            field.setToolTip(
                "Draw a charge density, a spin density or an electrostatic potential "
                "from a PROPERTIES run with ECH3 or POT3" if ready else
                "Open a file first: the grids are looked for beside it"
            )
        clear = getattr(self, "_clear_density_action", None)
        if clear is not None:
            clear.setEnabled(bool(getattr(self, "_density_shown", False)))

    def _clear_orbital(self) -> None:
        """Take a shown orbital off the view."""
        self.viewport.renderer.set_orbital(None)
        self._orbital = None
        self._update_orbital_actions()

    def _update_orbital_actions(self) -> None:
        """Enable the orbital entries according to what is loaded and shown."""
        action = getattr(self, "_orbitals_action", None)
        if action is not None:
            available = self._capability("orbitals")
            action.setEnabled(available)
            action.setToolTip(
                "" if available
                else "No ORBITALS Molden files beside the loaded output"
            )
        clear = getattr(self, "_clear_orbital_action", None)
        if clear is not None:
            clear.setEnabled(self._orbital is not None)

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
        action = getattr(self, "_vci_action", None)
        if action is not None:
            action.setEnabled(self._capability("vci"))

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

        It never re-docks: it is the one panel whose place is the user's, and a
        window that snaps back into the frame every time it nears an edge cannot
        be placed. The float happens once, not on every plot.
        """
        tab = getattr(self, "_tab", None)
        if tab is not None:
            tab.plots_open = True  # this file's plots are open: bring them back with it
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

    @guard()
    def _apply_symmetry_reduction(self, rotations: tuple) -> None:
        """Treat the crystal as belonging to a lower group from now on.

        Applied to the structure the app holds, not to the analysis cell the
        panel was handed: that one is a fold of this one, remade on every edit,
        and a choice recorded there would vanish with it. From here the
        reduction rides in ``atoms.info``, so every derived copy — the analysis
        cell, the builder's structure, an undo snapshot — carries it along.
        """
        self.structure.set_reduced_symmetry(rotations)
        # Nothing else to do by hand: setting it notifies, and the ordinary
        # change path records the undo, refreshes the Info panel and re-hands
        # the symmetry panel an analysis cell — which now carries the choice.
        self.symmetry_panel.set_structure(self._analysis_cell())

    def _show_symmetry_panel(self) -> None:
        """Open the point-symmetry panel (a tab beside Phonons) and analyse."""
        self._symmetry_dock.show()
        self._symmetry_dock.raise_()
        self.symmetry_panel.show_analysis()

    def _show_brillouin_zone(self) -> None:
        """Draw this lattice's first Brillouin zone, on its own.

        On the clean analysis cell, like the band path in the input builders: a
        supercell's zone is a folded fraction of the real one, and drawing that
        under the same name would be a lie.
        """
        structure = self._analysis_cell()
        if len(structure) == 0:
            QMessageBox.information(
                self, "Brillouin zone", "Open or build a structure first."
            )
            return
        from crystalline.ui.panels.zone_picker import ZonePickerDialog

        ZonePickerDialog.visualise(structure, self)

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

    def _set_position_of_selection(self, position) -> None:
        """Place the one selected atom at the cartesian coordinates typed in the
        Geometry panel.

        Routed through the viewport's :meth:`~crystalline.ui.viewport.Viewport.move_atom_to`
        — the same call a finished drag makes — so a typed move and a dragged one
        are the same operation: one undo step, and the atom's periodic images
        travel with it instead of being left behind.
        """
        indices = self.structure_panel.selected_indices()
        if not self._editing or len(indices) != 1:
            return
        self.viewport.move_atom_to(indices[0], position)

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
        """Add one atom of ``symbol`` in the middle of the structure, and select it.

        The middle is the obvious place to drop an atom you are about to drag
        into position: among the atoms already there and away from the cell
        boundary. What counts as the middle of a slab or a polymer is
        :meth:`Structure.centre`, not the centre of the cell.
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
        """The middle of the structure — see :meth:`Structure.centre`.

        The centre of the whole cell was used here, which for a slab or a polymer
        is 250 Å out in the vacuum CRYSTAL writes across the aperiodic
        directions: the atom was added, and selected, somewhere off screen.
        """
        return [float(x) for x in self.structure.centre()]

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
        self.info_panel.show_structure(self._analysis_cell(), self._output_props)

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
            # The old cap of 12 per axis was arbitrary and got in the way: what
            # costs anything is the total atom count, not the repetition along
            # any one direction, and a slab or a chain wants a large number down
            # one axis and one along the others. So the number is free and the
            # size is shown instead, with a confirmation past the point where it
            # is genuinely slow.
            box.setRange(1, _MAX_SUPERCELL_REPEAT)
            box.setValue(value)
            form.addRow(f"Repeat along {axis}", box)
            boxes.append(box)

        size = QLabel()
        size.setStyleSheet("color: palette(mid);")
        form.addRow("", size)

        def show_size() -> None:
            cells = boxes[0].value() * boxes[1].value() * boxes[2].value()
            atoms = cells * len(self._source)
            note = f"{cells} cell{'' if cells == 1 else 's'}, {atoms:,} atoms"
            if atoms > _SLOW_SUPERCELL_ATOMS:
                note += " — large; building the view will take a moment"
            size.setText(note)

        for box in boxes:
            box.valueChanged.connect(show_size)
        show_size()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() != QDialog.Accepted:
            return
        reps = tuple(box.value() for box in boxes)
        atoms = reps[0] * reps[1] * reps[2] * len(self._source)
        if atoms > _SLOW_SUPERCELL_ATOMS:
            # Asked rather than refused: the number is sometimes what someone
            # actually wants, and a cap they cannot pass is worse than a wait
            # they agreed to.
            confirm = QMessageBox.question(
                self, "Large supercell",
                f"{reps[0]}×{reps[1]}×{reps[2]} is {atoms:,} atoms.\n\n"
                f"Building and drawing that will take a while, and analyses that "
                f"run over neighbours — bonds, polyhedra — will be slower still.\n\n"
                f"Go ahead?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if confirm != QMessageBox.Yes:
                return
        # A supercell chosen here is the user's own: the phonon panel's Untile
        # must not offer to throw it away for a cell they never asked for.
        self._tile_restore = None
        self._set_supercell(tuple(box.value() for box in boxes))
        self._apply_cell_view()

    def _set_supercell(self, reps) -> None:
        """Record the supercell tiling and keep the menu action's label in step."""
        self._supercell = tuple(reps)
        self._update_supercell_action()
        panel = getattr(self, "phonon_panel", None)
        if panel is not None:
            # ...and what its Untile would go back to, if it did the tiling.
            panel.set_supercell(self._supercell, self._tile_restore)

    def _update_supercell_action(self) -> None:
        """Name the tiling on screen in the Cell menu's Supercell entry."""
        action = getattr(self, "_supercell_action", None)
        if action is not None:
            na, nb, nc = self._supercell
            suffix = "" if self._supercell == (1, 1, 1) else f"  ({na}×{nb}×{nc})"
            action.setText(f"Supercell…{suffix}")

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
        self._show_modes(modes)
        self._update_export_actions()
        return True

    def _show_modes(self, modes) -> None:
        """Hand composed modes to the panel, and remember the geometry they fit.

        The atom count is what :meth:`_reconcile_modes_with_edit` needs later: a
        mode carries one displacement per atom, so it applies to a structure of
        that size and to no other.
        """
        self.phonon_panel.set_modes(
            self.structure.positions, modes, self.structure.numbers
        )
        self._modes_natom = len(self.structure)

    def _reconcile_modes_with_edit(self, s: Structure) -> None:
        """Offer the file's modes back when an edit leaves them applicable again.

        Adding or deleting an atom makes them inapplicable — there is no
        displacement for an atom the calculation never saw — so the panel drops
        them. Undoing that edit brings the geometry back, and the modes should
        come back with it: the panel cannot know, having thrown them away, so it
        used to stay empty and grey for the rest of the session however the
        structure was put back. Moving an atom or changing an element keeps the
        count, so the panel keeps the modes and this does nothing.

        When they genuinely do not apply, the panel says so. Going quiet and
        greying out says only that something happened.
        """
        if self.phonon_panel.has_modes() or self._modes is None:
            return
        if self._modes_natom == len(s) and self._reload_modes():
            return
        if self._modes_natom is not None:
            self.phonon_panel.set_note(
                f"The modes read from this file describe {self._modes_natom} atoms "
                f"and the structure now has {len(s)}, so they cannot be shown on "
                f"it. Undo the edit to animate them again."
            )

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

    # ── drag and drop ───────────────────────────────────────────────────
    @guard()
    def dragEnterEvent(self, event) -> None:
        """Accept dragged files the app can do something with, and say what.

        Every structure among them opens, each in a tab of its own; atoms to
        import are added to the structure on screen when nothing is being
        opened alongside them.
        """
        to_open, to_import = self._dropped_files(event)
        if not to_open and not to_import:
            event.ignore()
            return
        event.acceptProposedAction()
        if len(to_open) > 1:
            self._drop_hint.show_hint(f"Open {len(to_open)} files", "each in a new tab")
        elif to_open:
            self._drop_hint.show_hint(f"Open {os.path.basename(to_open[0])}", "in a new tab")
        else:
            self._drop_hint.show_hint(
                f"Add the atoms in {os.path.basename(to_import[0])}",
                "appends to the current structure — undoable",
            )

    @guard()
    def dragMoveEvent(self, event) -> None:
        # Qt asks again on every move; without this the drop is refused whatever
        # dragEnterEvent said.
        if not any(self._dropped_files(event)):
            event.ignore()
        else:
            event.acceptProposedAction()

    @guard()
    def dragLeaveEvent(self, event) -> None:
        self._drop_hint.hide_hint()
        super().dragLeaveEvent(event)

    @guard()
    def dropEvent(self, event) -> None:
        """Take the file, and open it on the *next* turn of the event loop.

        Not inside this handler. A drop arrives in the middle of the platform's
        drag session, and opening a file tears the VTK scene down and builds a
        new one — re-entering VTK from inside a drag it is itself dispatching
        segfaults outright when the drop lands on the 3D view. Accepting first
        and deferring also lets the application the file came from see the drag
        finish immediately, instead of holding its pointer for the second or so
        a large output takes to parse.
        """
        self._drop_hint.hide_hint()
        to_open, to_import = self._dropped_files(event)
        if not to_open and not to_import:
            event.ignore()
            return
        event.acceptProposedAction()
        ignored = len(self._dropped_paths(event)) - len(to_open) - len(to_import)
        QTimer.singleShot(0, lambda: self._handle_drop(to_open, to_import, ignored))

    def _handle_drop(self, to_open: list, to_import: list, ignored: int) -> None:
        """Open or import the dropped files, once the drag itself is over.

        Every structure dropped opens, each in its own tab. Atoms to import go
        into the structure on screen, and only the first of them — see
        :func:`~crystalline.ui.file_tabs.openable`.
        """
        done = self._load_paths(to_open)
        if not to_open:
            done = [path for path in to_import if self._import_path(path)]
        if done and ignored:
            # Said rather than silently dropped: a multiple selection dragged in
            # one gesture looks like it should all arrive. "Opening": the files
            # are still being read when this is said.
            verb = "Opening" if to_open else "Imported"
            what = (os.path.basename(done[0]) if len(done) == 1
                    else f"{len(done)} files")
            self.statusBar().showMessage(
                f"{verb} {what} — {ignored} other dropped "
                f"file{'' if ignored == 1 else 's'} ignored",
                8000,
            )

    def _dropped_files(self, event):
        """``(to_open, to_import)``: the dropped files the window will act on."""
        from crystalline.crystalio import file_action

        to_open, to_import, _ignored = openable(self._dropped_paths(event), file_action)
        return to_open, to_import

    @staticmethod
    def _dropped_paths(event) -> list:
        """The local files a drag carries, in order. Non-file drags give none."""
        data = event.mimeData()
        if data is None or not data.hasUrls():
            return []
        return [
            url.toLocalFile() for url in data.urls()
            if url.isLocalFile() and url.toLocalFile()
        ]

    # ── file actions ────────────────────────────────────────────────────
    def _open_file(self) -> None:
        """Open one or more files, each in a tab of its own.

        Geometry, and phonon modes too where the file has them.
        """
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open structure files", self._open_folder(),
            "Structure files (*.out *.gui *.f34 *.cif);;CRYSTAL files (*.out *.gui *.f34);;"
            "CIF files (*.cif);;All files (*)",
        )
        self._load_paths(paths)

    def _open_folder(self) -> str:
        """Where the Open dialog starts: beside the file on screen, if there is one."""
        path = self._tab.path if self._tab is not None else None
        return os.path.dirname(path) if path else ""

    def _load_paths(self, paths: Sequence[str]) -> list:
        """Open each of ``paths`` in a tab of its own, the first to the front.

        The rest are read behind it, in order, so the file the user asked for
        first is the one they land in rather than the last to finish reading.
        """
        for index, path in enumerate(paths):
            self._load_path(path, front=index == 0)
        return list(paths)

    def _load_path(self, path: str, front: bool = True) -> None:
        """Open ``path`` in a tab of its own, reading it off the main thread.

        Reading is the slow part — CRYSTALClear walks the whole output, and
        finding out which plots, orbitals and anharmonic data it holds walks it
        again — so :func:`_read_file` does all of it on a worker, and nothing
        stands between the user and the window meanwhile: no overlay, only the
        status bar saying what is being read. Files are read one at a time, in
        the order given.

        ``front`` is whether the file's tab comes to the front when it is ready.
        Of several opened at once only the first does, so that it can be worked
        in while the rest are read and set behind it, one by one, without
        taking the screen from it, and a tab set behind this way is not built
        until it is looked at (:meth:`_realise_tab`).

        Split out of :meth:`_open_file` so a file arriving any other way — dropped
        on the window — goes through exactly the same sequence (see
        :meth:`_show_read_file` for what that is).
        """
        self._pending_reads.append((path, front))
        if not self._reading:
            self._read_next()
        else:
            self._update_reading_status()

    def _read_next(self) -> None:
        """Start reading the next queued file, if there is one."""
        if not self._pending_reads:
            self._reading = False
            self._reader = None
            self._update_reading_status()
            return
        self._reading = True
        path, front = self._pending_reads.pop(0)
        self._reading_path = path
        self._update_reading_status()

        def shown(read) -> None:
            try:
                self._show_read_file(path, read, front)
            finally:
                self._read_next()

        def failed(exc) -> None:
            # A file that will not read opens no tab, and stops nothing queued
            # behind it.
            QMessageBox.critical(self, "Load failed", f"{os.path.basename(path)}:\n{exc}")
            self._read_next()

        # Not through _run_busy: no overlay, and the tab bar stays live — the
        # result goes to a tab of its own, not to whichever one is on screen.
        # Held here until the next is started: a dropped worker takes its
        # thread down with it.
        self._reader = Worker(lambda: _read_file(path))
        self._reader.finished.connect(shown)
        self._reader.failed.connect(failed)
        self._reader.start()

    def _update_reading_status(self) -> None:
        """Say in the status bar which file is being read, and how many are to come."""
        label = getattr(self, "_reading_status", None)
        if label is None:
            return
        if not self._reading:
            label.hide()
            return
        text = f"Reading {os.path.basename(self._reading_path)}…"
        if self._pending_reads:
            text += f"  {len(self._pending_reads)} more to come"
        label.setText(text)
        label.show()

    def _show_read_file(self, path: str, read: "_ReadFile", front: bool = True) -> None:
        """Put a file that has been read into a tab.

        In front — the tab on screen, if that one is empty (the window opens on
        an empty tab, and the first file takes it over), else a new tab brought
        to the front — or, when not ``front``, in a tab set behind the one being
        worked in, where it waits: that tab is named and holds its file, and
        nothing is built or drawn for it until it is looked at
        (:meth:`_realise_tab`). Putting a file into widgets is the part that
        has to happen on this thread, and doing it for a tab nobody has turned
        to yet is the one cost reading in the background does not remove.
        """
        if front or self._tab.is_blank():
            tab = self._take_tab_for_file()
            self._fill_tab(tab, path, read)
            self._settle_tab(tab)  # filled on screen: shown finished, not a frame early
            self._show_notice(tab)
            return
        tab = self._add_tab(show=False)
        tab.path = path
        tab.pending = (path, read)
        self._update_tab_labels()

    def _show_notice(self, tab: FileTab) -> None:
        """Say what opening ``tab``'s file had to leave out — once, when it is seen.

        An output whose modes could not be read opens with its geometry alone,
        and nothing else says so: the Phonons panel of a file with no modes at
        all looks exactly the same, so the file would appear to have been a
        geometry all along. For a tab read in behind the one being worked in,
        that is said when it is first brought to the front, not in the middle
        of the work.
        """
        notice, tab.notice = tab.notice, None
        if notice:
            QMessageBox.warning(self, "Modes not read",
                                f"{os.path.basename(tab.path or '')}:\n{notice}")

    def _fill_tab(self, tab: FileTab, path: str, read: "_ReadFile") -> None:
        """Load what was read into ``tab``, the tab the window is acting on.

        There is a lot to do, and all of it on the UI thread: whatever the tab
        showed before has to be let go, the supercell and the tiling reset, and
        eight menu sections re-enabled against what this file turns out to
        contain — from what :func:`_read_file` found, not by reading the file
        again.
        """
        result = read.loaded
        tab.path = path
        tab.notice = result.note
        tab.capabilities.update(
            {(key, read.output_path): found for key, found in read.capabilities.items()}
        )
        # The previous file's plots belong to the previous file: their figures
        # stay live otherwise, and a spectrum's peak-pick handler would select
        # modes in a structure it knows nothing about.
        self.plot_panel.clear()
        # And the orbital. Forgetting our record of it was not enough: the
        # renderer keeps the field it was given, so the rebuild for the new
        # structure drew the *previous* file's orbital over it — lobes from one
        # crystal on the atoms of another.
        self._clear_orbital()
        # The same for a charge density or a potential, which a rebuild redraws
        # just as faithfully — and a slice's cutaway with it, hiding half of the
        # new structure behind a plane of the old one.
        self._clear_density()
        # Lattice planes outlive a change of view, not a change of crystal.
        self.geometry_panel.clear_lattice_planes()
        self._source = result.structure
        self._set_qmodes(result.qpoints if result.has_phonons else [])
        self._adps = read.adps
        self._tile_restore = None  # nothing of the old file's tiling to go back to
        self._set_supercell((1, 1, 1))  # a fresh file starts at its own unit cell
        # Remember the output file so property plots (IR/Raman/elastic/EOS) can
        # read it directly; geometry-only files (.gui/.f34/.cif) carry no such data.
        self._output_path = read.output_path
        self._apply_cell_view()
        # After the view exists (and its ADP tensors have been pushed), so
        # switching the ellipsoids on draws them straight away.
        self._update_adp_controls(autoshow=True)
        self._update_info(read.output_props)
        self._update_plot_actions()  # enable only the plots this file supports
        self._update_density_actions()  # the grids are looked for beside this file
        self._update_orbital_actions()  # and the orbitals, if the run wrote any
        self._update_spectra_action()
        self._update_vci_action()
        self._update_anscan_action()
        self._update_pes_action()
        self._update_import_action()  # a structure is now loaded — allow importing
        self._update_tab_labels()  # the tab is named after its file, and the window too
        # A new file starts its own timeline: nothing about the last one is
        # worth stepping back into.
        self._reset_undo()
        # A note on what could not be read (modes, say) is kept on the tab and
        # said when the tab is seen — see _show_notice.

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
        self._import_path(path)

    def _import_path(self, path: str) -> bool:
        """Append the atoms in ``path`` to the current structure. False if it failed.

        Split out of :meth:`_import_atoms` for the same reason as
        :meth:`_load_path`: a dropped file must land in the undo history and pick
        up the selection the same way a chosen one does.
        """
        try:
            from crystalline.crystalio import read_atoms

            atoms = read_atoms(path)
        except Exception as exc:  # noqa: BLE001 - surface any read/parse error
            QMessageBox.critical(self, "Import failed", f"Could not read atoms:\n{exc}")
            return False
        symbols = list(atoms.get_chemical_symbols())
        if not symbols:
            QMessageBox.information(self, "Nothing imported", "No atoms found in that file.")
            return False
        try:
            new = self.structure.add_atoms(symbols, atoms.get_positions())  # -> undo + redraw
        except Exception as exc:  # noqa: BLE001 - e.g. an unknown element symbol
            QMessageBox.critical(self, "Import failed", f"Could not add the atoms:\n{exc}")
            return False
        self.display_panel.set_elements(self.structure.numbers)  # new elements may appear
        # Turn editing on and select the imported atoms so they can be dragged
        # into place as a whole straight away.
        if not self._editing:
            self._edit_mode_action.setChecked(True)  # toggles _set_editing(True)
        self.structure_panel.set_selection(new)
        return True

    def _update_info(self, output_props: dict) -> None:
        """Refresh the crystallographic info panel for the loaded system.

        ``output_props`` are the rows the CRYSTAL output gives about its own run
        (read by :func:`_read_file`). The crystallography is described on the
        cell on screen, folded to one cell — not on the file's primitive cell —
        so that "as in the file" means the cell the 3D view draws, and so that
        the panel says the same thing before an edit as after one.
        """
        self._output_props = output_props or {}
        self.info_panel.show_structure(self._analysis_cell(), self._output_props)

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
        """Open the CRYSTAL ``.d12`` input builder for the structure as edited.

        Built from :meth:`_analysis_cell` — the shown structure folded back into
        a single unit cell — so symmetry reduction sees one clean cell rather
        than a supercell tiling or boundary-completed duplicates, *and* the deck
        describes the geometry currently on screen.

        This used to pass ``_source``, the pristine loaded cell. That got the
        cell right but silently discarded every edit made since the file was
        opened: a dragged atom, an added or deleted one, a changed element all
        came out as the original geometry, with nothing on screen to say so.
        (Lattice-parameter edits were the exception — those are applied to
        ``_source`` itself, so they came through either way.)
        """
        structure = self._analysis_cell()
        if len(structure) == 0:
            QMessageBox.information(
                self, "Build CRYSTAL input", "Open or build a structure first."
            )
            return
        from crystalline.ui.panels.input_builder import InputBuilderDialog

        # The builder starts on the cell the Info panel is showing; changing it
        # there is for that one deck and is not remembered.
        panel = getattr(self, "info_panel", None)
        if panel is not None:
            choice = panel.cell_choice()
        else:
            from crystalline.ui import preferences

            choice = preferences.cell_choice()
        InputBuilderDialog(structure, self, cell_choice=choice).exec()

    def _build_properties_input(self) -> None:
        """Open the PROPERTIES (``.d3``) builder for the structure as edited.

        The same clean single cell the ``.d12`` builder uses: a ``.d3`` carries
        no geometry, but the band path is derived from the lattice, and deriving
        it from a supercell or a boundary-completed view would give the path of
        a different Brillouin zone.
        """
        structure = self._analysis_cell()
        if len(structure) == 0:
            QMessageBox.information(
                self, "Build properties input", "Open or build a structure first."
            )
            return
        from crystalline.ui.panels.properties_builder import PropertiesBuilderDialog

        PropertiesBuilderDialog(structure, self).exec()

    # ── remembered plot-dialog settings ─────────────────────────────────
    def _restore_dialog(self, dialog, key: str) -> None:
        """Reopen a plot dialog on the settings it was last accepted with.

        Also tidies its forms. Every plot dialog is opened through here, so it is
        the one place that reaches all of them — see :func:`theme.tidy_forms` for
        why they need it.
        """
        from crystalline.ui import theme
        from crystalline.ui.panels.dialog_state import restore

        theme.tidy_forms(dialog)
        restore(dialog, self._plot_dialog_state.get(key))

    def _remember_dialog(self, dialog, key: str) -> None:
        """Keep an accepted dialog's settings for the next time it is opened.

        Only on accept: a cancelled dialog is the user backing out, and having it
        still change what comes up next time would make Cancel do something.
        """
        from crystalline.ui.panels.dialog_state import capture

        self._plot_dialog_state[key] = capture(dialog)

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
        from crystalline.ui.image_export import export_view

        export_view(self, self.viewport.export_image, "crystal_view")


    def _export_animation(self) -> None:
        """Render the selected phonon mode over one cycle and save it.

        GIF needs no extra packages; a PNG filename writes a numbered frame
        sequence; MP4 needs imageio-ffmpeg (a clear message says so if missing).

        Drawn by the view on screen, supersampled, rather than by an off-screen
        plotter of its own. A second VTK render window beside the live one is
        what killed the app: every frame came out, the file was written, and the
        first draw afterwards faulted in C++ — no traceback, no crash report,
        because by then the damage was in the OpenGL state the two had shared.
        An image export, which does everything else an export does and only this
        differently, has never done it.

        So the view really does animate while this runs, and is put back where
        it stood when it is over. The frames are the view's own shape, which is
        also what the picture on screen promises.
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
        from crystalline.viz.export import (
            MOVIE_EXTS,
            VIDEO_MISSING_MESSAGE,
            frames_from_view,
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

        # Rendered straight into the writer, one frame at a time, with the event
        # loop pumped between them. Three things have to be true at once and
        # only this arrangement has all three.
        #
        # Nothing may be held: 240 frames at 1920x1440 is two gigabytes, and the
        # GIF writer then copied every one — six and a half gigabytes, which the
        # system answers by killing the process, leaving nothing behind to say
        # so. That was the crash.
        #
        # Rendering is VTK's work and VTK's is the main thread's, so it cannot
        # be moved off; and staging the frames to files to free the thread is
        # slower than the export it was meant to unblock — writing two gigabytes
        # costs 243 ms a frame where encoding one costs 150.
        #
        # So the loop stays here and gives the window its turns, excluding user
        # input: the spinner turns, the message counts the frames, the window
        # repaints and the system sees an app that is answering — but nothing
        # the user clicks can re-enter an export that is halfway through.
        # Nothing else may be driving the live view while this runs. The loop
        # below gives the window its turns, and a mode left playing would take
        # one of them to redraw the view it is animating — a second VTK render
        # window drawing in the middle of the off-screen one's frame.
        self.phonon_panel.stop()
        self._trace(f"export: {n_frames} frames at {window_size} -> {path}")
        standing_at = self.phonon_panel.current_phase()
        self._busy.start(f"Exporting {n_frames} frames…")
        try:
            written = save_animation(
                frames_from_view(
                    self.viewport.interactor, self.animator,
                    n_frames=n_frames, size=window_size, on_frame=self._exporting_frame,
                ),
                path, fps=fps,
            )
        except Exception as exc:  # noqa: BLE001 - surface encode/write errors clearly
            self._trace(f"export: failed: {exc}")
            QMessageBox.critical(self, "Export failed", f"Could not save the animation:\n{exc}")
            return
        finally:
            self._trace("export: written; putting the view back")
            self.phonon_panel.show_phase(standing_at)   # where the mode was standing
            self._busy.stop()
        self._trace("export: done, back to the event loop")
        if len(written) > 1:
            QMessageBox.information(
                self, "Animation exported", f"Wrote {len(written)} frames next to\n{written[0]}"
            )

    @staticmethod
    def _trace(message: str) -> None:
        """Say where the app has got to, when it is asked to.

        Set ``CRYSTALLINE_TRACE=1`` and these go to stderr, which the launcher
        keeps in ~/Library/Logs/CRYSTALLine.log. For a crash that takes the
        process down without raising — a fault in VTK or Qt, which leaves no
        traceback and no crash report — the last line printed is the evidence:
        it says which step was running when the process died. That is how the
        export crash was found, after everything else had come back empty.
        """
        import os
        import sys

        if os.environ.get("CRYSTALLINE_TRACE"):
            print(f"[trace] {message}", file=sys.stderr, flush=True)

    def _exporting_frame(self, done: int, total: int) -> None:
        """Show how far an export has got, and let the window answer for itself.

        ``ExcludeUserInputEvents`` is the whole safety of it: paints, timers and
        the window server's own messages are delivered — so the app repaints and
        is not marked unresponsive — while clicks and keys wait in the queue
        until the export is over and cannot start a second one on top of it.
        """
        from PySide6.QtCore import QEventLoop
        from PySide6.QtWidgets import QApplication

        if done in (1, total):
            self._trace(f"export: frame {done} of {total}")
        self._busy.pulse(f"Frame {done} of {total}…")
        QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)

    def _ask_animation_options(self):
        """Prompt for ``(ext, filter_label, size, n_frames, fps)`` or ``None``.

        The size is the animation's, which the frames are fitted to; frames
        control the smoothness of one vibration cycle; FPS the playback speed.
        A PNG target writes a numbered frame sequence rather than a single file.
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
        # Sizes, as before — the frames are drawn by the view on screen, but they
        # are fitted to whichever of these is chosen rather than coming out the
        # shape the window happens to be. A view with docks either side of it is
        # taller than it is wide, and a portrait animation is letterboxed in
        # black by every viewer there is.
        from crystalline.viz.export import _window_size

        view = _window_size(self.viewport.interactor)
        resolutions = [
            ("640 × 480", (640, 480)),
            ("800 × 600", (800, 600)),
            ("1280 × 960", (1280, 960)),
            ("1920 × 1440", (1920, 1440)),
            (f"The view, {view[0]} × {view[1]}", view),
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
            self._show_modes(modes)
        else:
            self.phonon_panel.clear()
            self._modes_natom = None
        self._update_export_actions()  # modes may have appeared/disappeared
        # A shown orbital belongs to the cell on screen, so it is rebuilt across
        # whatever that now is — taking a supercell redraws it over the supercell.
        if self._orbital is not None:
            self._refresh_orbital()
        # Every Cell action funnels through here, so this one line is what makes
        # a supercell, a cell-view switch, boundary completion and a lattice
        # edit undoable. A load resets the history afterwards; an undo rebuilding
        # the view has capture suppressed.
        self._capture_undo()

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
        self._shown_edited = False  # freshly derived: nothing on it the source lacks
        if unit_cell is not None:
            self._unit_cell = unit_cell
        if bond_structure is not None:
            self._bond_structure = bond_structure
        # _note_edited before the heavier one.
        self.structure.add_listener(self._routed(self._tab, self._note_edited))
        self.structure.add_listener(self._routed(self._tab, self._on_structure_changed))
        self.viewport.show_structure(
            self.structure, reference_cell=unit_cell, bond_structure=bond_structure
        )
        self.structure_panel.set_structure(self.structure)
        if hasattr(self, "geometry_panel"):
            self.geometry_panel.set_structure(self.structure)
            # The source may be a new crystal, or a new lattice: re-derive the
            # cell the lattice planes are indexed in (which redraws them).
            self.geometry_panel.set_miller_cell(self._miller_cell())
        if hasattr(self, "symmetry_panel"):
            self.symmetry_panel.set_structure(self._analysis_cell())
        self._update_view_actions()  # a/b/c alignment depends on the cell just shown
        if hasattr(self, "display_panel"):
            self.display_panel.set_elements(self.structure.numbers)  # refresh element swatches


# ── reading a file, off the main thread ────────────────────────────────────
# Everything below is Qt-free: it runs on a worker (MainWindow._read_next), so it
# may parse, but never touch a widget.

# What an open output is probed for, to enable the menus that need it.
_PROBES = ("plots", "orbitals", "vci", "anscan", "pes")


def _probe(key: str, output_path: Optional[str]):
    """What the output at ``output_path`` offers for ``key`` — one of :data:`_PROBES`.

    The one definition, used by :meth:`MainWindow._capability` and by
    :func:`_read_file` alike, so a probe answered off the main thread means
    what it means when the window asks.
    """
    from crystalline import crystalio
    from crystalline.crystalio import molden

    if key == "plots":
        return crystalio.output_availability(output_path)
    if key == "orbitals":
        return bool(output_path and molden.find_orbital_files(output_path))
    if key == "vci":
        return crystalio.has_vci(output_path)
    if key == "anscan":
        return crystalio.has_anscan(output_path)
    if key == "pes":
        return crystalio.has_pes(output_path)
    raise KeyError(f"no probe {key!r}")


@dataclass
class _ReadFile:
    """Everything a new tab takes from its file, read in one go."""

    loaded: object                  # crystalio.LoadedFile: the structure, and modes if any
    output_path: Optional[str]      # the CRYSTAL output; None for a geometry-only file
    adps: Optional[ADPSet]          # thermal ellipsoids, when the run has them
    output_props: dict              # the run's own rows for the Info panel
    capabilities: dict              # probe key → what _probe found


def _read_file(path: str) -> _ReadFile:
    """Read ``path`` for a tab: the structure, then everything else the window asks of it.

    Only the structure is required — a file that will not read raises, and opens
    no tab. The rest is best effort, as it was when the window read it itself:
    most runs carry no ADPs, not every output has the rows the Info panel shows,
    and a probe that fails here is simply left for the window to ask again.
    """
    from crystalline.crystalio import load, load_adp, output_properties

    loaded = load(path)
    # Geometry-only files (.gui/.f34/.cif) carry no results to plot.
    output_path = None if path.lower().endswith((".gui", ".f34", ".cif")) else path
    try:
        adps = load_adp(path)
    except Exception:  # noqa: BLE001 - never let this break loading a file
        adps = None
    try:
        output_props = output_properties(path) or {}
    except Exception:  # noqa: BLE001 - never let output parsing break loading
        output_props = {}
    capabilities = {}
    for key in _PROBES:
        try:
            capabilities[key] = _probe(key, output_path)
        except Exception:  # noqa: BLE001 - asked again, and reported, by the window
            pass
    return _ReadFile(loaded, output_path, adps, output_props, capabilities)


def _no_title_bar(dock: QDockWidget) -> QWidget:
    """A title bar for ``dock`` that takes up no room at all.

    Not a bare ``QWidget``: that measures (-1, -1) — "no size hint" — and a
    dock adds its title bar's height to its content's. A dock whose content
    had no minimum height yet — each dock holds a stack of the tabs' panels,
    empty until the first tab is made — then asked for a minimum height of
    -1, which Qt reports on macOS as "Negative sizes (0,-1) are not possible".
    An empty layout with no margins measures (0, 0), as the bar should.
    """
    bar = QWidget(dock)
    layout = QHBoxLayout(bar)
    layout.setContentsMargins(0, 0, 0, 0)
    return bar


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
