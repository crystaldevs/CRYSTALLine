"""Menu-bar, toolbar and About-dialog construction for :class:`MainWindow`.

Split out so ``main_window`` stays about *behaviour* — what happens when an
action fires — while everything here is declarative: create actions, connect
them to the window's slots, and stash on the window the ones it later enables
or disables (``_undo_action``, ``_edit_tool_actions``, ``_plot_actions``, …).

Every function takes the window it builds for and keeps no state of its own,
so the ordering constraints stay visible in one place: the Edit menu must be
built before the toolbar (which reuses the undo/redo actions), and the View
menu before the toolbar's ``_update_view_actions`` call.
"""

from __future__ import annotations

from PySide6.QtGui import QAction, QColor, QIcon
from PySide6.QtWidgets import QLabel, QToolBar, QToolButton, QWidget

# How far one click of the toolbar's rotate buttons orbits the view, and the
# chip colour that sets them apart from the coloured a/b/c alignment buttons.
_ROTATE_STEP_DEG = 15.0
_ROTATE_CHIP_COLOR = "#6c757d"


def build_menus(window) -> None:
    """Build the whole menu bar and the toolbars, in dependency order."""
    _build_file_menu(window)
    _build_cell_menu(window)
    _build_edit_menu(window)
    _build_view_menu(window)
    _build_plot_menu(window)
    _build_help_menu(window)
    _build_toolbars(window)


# ── File ──────────────────────────────────────────────────────────────────
def _build_file_menu(window) -> None:
    file_menu = window.menuBar().addMenu("&File")

    open_action = QAction("Open…", window)
    open_action.setShortcut("Ctrl+O")
    open_action.triggered.connect(window._open_file)
    file_menu.addAction(open_action)

    # Importing atoms only makes sense once there's a structure to add them to —
    # enabled by ``_update_import_action`` after a file is opened.
    window._import_action = QAction("Import atoms into structure…", window)
    window._import_action.setEnabled(False)
    window._import_action.triggered.connect(window._import_atoms)
    file_menu.addAction(window._import_action)

    file_menu.addSeparator()
    save_gui = QAction("Save structure as .gui…", window)
    save_gui.triggered.connect(window._save_gui)
    file_menu.addAction(save_gui)

    save_cif = QAction("Save structure as .cif…", window)
    save_cif.triggered.connect(window._save_cif)
    file_menu.addAction(save_cif)

    build_input = QAction("Build CRYSTAL input (.d12)…", window)
    build_input.triggered.connect(window._build_crystal_input)
    file_menu.addAction(build_input)

    file_menu.addSeparator()
    export_image = QAction("Export image…", window)
    export_image.triggered.connect(window._export_image)
    file_menu.addAction(export_image)

    window._export_anim_action = QAction("Export phonon animation…", window)
    window._export_anim_action.triggered.connect(window._export_animation)
    file_menu.addAction(window._export_anim_action)


# ── Cell ──────────────────────────────────────────────────────────────────
def _build_cell_menu(window) -> None:
    """A 'Cell' menu: supercell and boundary-completion of the crystallographic cell."""
    cell_menu = window.menuBar().addMenu("&Cell")

    window._lattice_action = QAction("Lattice parameters…", window)
    window._lattice_action.triggered.connect(window._open_lattice_dialog)
    cell_menu.addAction(window._lattice_action)

    window._supercell_action = QAction("Supercell…", window)
    window._supercell_action.triggered.connect(window._open_supercell_dialog)
    cell_menu.addAction(window._supercell_action)

    # Show whole molecules/atoms that only partially belong to the cell
    # (their periodic images poke in) — the "packed" view, on by default.
    # Unchecking restricts the view to the cell's own atoms.
    window._boundary_action = QAction("Complete molecules at cell boundary", window, checkable=True)
    window._boundary_action.setChecked(window._show_boundary)
    window._boundary_action.toggled.connect(window._on_boundary_toggled)
    cell_menu.addAction(window._boundary_action)

    # The panel this opens is not one of the docks shown at startup: it is asked
    # for, from here, when someone wants the analysis.
    cell_menu.addSeparator()
    symmetry_action = QAction("Point symmetry analysis", window)
    symmetry_action.setToolTip(
        "Find the rotation axes, mirror planes and centre of inversion that hold "
        "a point of the structure fixed, and draw them over it"
    )
    symmetry_action.triggered.connect(window._show_symmetry_panel)
    cell_menu.addAction(symmetry_action)


# ── Edit ──────────────────────────────────────────────────────────────────
def _build_edit_menu(window) -> None:
    """An 'Edit' menu: turn editing on, select atoms, and run edit tools."""
    edit_menu = window.menuBar().addMenu("&Edit")

    window._undo_action = QAction(_history_icon(window, "edit-undo", "SP_ArrowBack"), "Undo", window)
    window._undo_action.setShortcut("Ctrl+Z")
    window._undo_action.setToolTip("Undo (Ctrl+Z)")
    window._undo_action.triggered.connect(window._undo)
    window._undo_action.setEnabled(False)
    edit_menu.addAction(window._undo_action)

    window._redo_action = QAction(
        _history_icon(window, "edit-redo", "SP_ArrowForward"), "Redo", window
    )
    window._redo_action.setShortcuts(["Ctrl+Shift+Z", "Ctrl+Y"])
    window._redo_action.setToolTip("Redo (Ctrl+Shift+Z)")
    window._redo_action.triggered.connect(window._redo)
    window._redo_action.setEnabled(False)
    edit_menu.addAction(window._redo_action)
    edit_menu.addSeparator()

    window._edit_mode_action = QAction("Editing mode", window, checkable=True)
    window._edit_mode_action.setShortcut("Ctrl+E")
    window._edit_mode_action.toggled.connect(window._set_editing)
    edit_menu.addAction(window._edit_mode_action)

    edit_menu.addSeparator()
    for text, slot, shortcut in (
        ("Select all", window._select_all, "Ctrl+A"),
        ("Clear selection", window._clear_selection, None),
        ("Invert selection", window._invert_selection, None),
    ):
        action = QAction(text, window)
        if shortcut:
            action.setShortcut(shortcut)
        action.triggered.connect(slot)
        edit_menu.addAction(action)

    edit_menu.addSeparator()
    # These act on the current selection and only while editing is on.
    window._edit_tool_actions = []
    for text, slot, shortcut in (
        ("Delete selected", window._delete_selected, "Del"),
        ("Duplicate selected", window._duplicate_selected, "Ctrl+D"),
        ("Translate selected…", window._translate_selected, None),
        ("Set element of selected…", window._set_element_selected, None),
    ):
        action = QAction(text, window)
        if shortcut:
            action.setShortcut(shortcut)
        action.triggered.connect(slot)
        edit_menu.addAction(action)
        window._edit_tool_actions.append(action)

    edit_menu.addSeparator()
    restore_action = QAction("Restore geometry", window)
    restore_action.setShortcut("Ctrl+R")
    restore_action.triggered.connect(window._restore_geometry)
    edit_menu.addAction(restore_action)
    window._update_edit_actions()


def _history_icon(window, theme_name: str, standard_pixmap: str) -> QIcon:
    """An undo/redo icon: the desktop theme's if present, else a Qt fallback."""
    from PySide6.QtWidgets import QStyle

    icon = QIcon.fromTheme(theme_name)
    if icon.isNull():
        icon = window.style().standardIcon(getattr(QStyle.StandardPixmap, standard_pixmap))
    return icon


# ── View ──────────────────────────────────────────────────────────────────
def _build_view_menu(window) -> None:
    """A 'View' menu: show the display panel and align the view to an axis."""
    view_menu = window.menuBar().addMenu("&View")
    display_action = QAction("Display settings", window)
    display_action.triggered.connect(window._show_display_panel)
    view_menu.addAction(display_action)

    # Docks close with a × and, without these, stay closed for good. Qt's own
    # toggleViewAction shows/hides and stays checked in step with the dock, even
    # when it is closed by its own button rather than from here.
    panels_menu = view_menu.addMenu("Panels")
    window._panel_actions = {}
    for title, dock in window._panel_docks():
        action = dock.toggleViewAction()
        action.setText(title)
        panels_menu.addAction(action)
        window._panel_actions[title] = action
    panels_menu.addSeparator()
    restore_all = QAction("Restore all panels", window)
    restore_all.triggered.connect(window._restore_all_panels)
    panels_menu.addAction(restore_all)

    view_menu.addSeparator()
    for label, axis in (("Along a axis", 0), ("Along b axis", 1), ("Along c axis", 2)):
        action = QAction(label, window)
        action.triggered.connect(
            lambda _checked=False, a=axis: window.viewport.align_view_along(a)
        )
        view_menu.addAction(action)
        window._axis_actions.append(action)


# ── Plot ──────────────────────────────────────────────────────────────────
def _build_plot_menu(window) -> None:
    """A 'Plot' menu routing CRYSTAL results through CRYSTALClear.plot.

    Output-file plots (IR/Raman/elastic/EOS) read the loaded ``.out``
    directly; data-file plots (bands/DOS/XRD) open a file dialog. Related
    entries (the elastic surfaces) go into a submenu. The whole menu is
    disabled if CRYSTALClear is missing.
    """
    from crystalline.crystalio import available_plots, crystalclear_available
    from crystalline.crystalio.anscan import plottable as anscan_plottable
    from crystalline.crystalio.pes import plottable as pes_plottable
    from crystalline.crystalio.vci import plottable as vci_plottable

    plot_menu = window.menuBar().addMenu("&Plot")
    window._plot_kinds = available_plots()
    window._plot_actions: dict = {}
    submenus: dict = {}
    for kind in window._plot_kinds:
        target = plot_menu
        if kind.group:
            target = submenus.get(kind.group)
            if target is None:
                target = plot_menu.addMenu(kind.group)
                submenus[kind.group] = target
        action = QAction(kind.label, window)
        action.triggered.connect(lambda _checked=False, k=kind: window._open_plot(k))
        target.addAction(action)
        window._plot_actions[kind.key] = action
    # Everything the vibrational spectra of the loaded output can offer —
    # Raman polarisations and anharmonic levels included — behind one entry,
    # because there are far too many curves for one action each.
    plot_menu.addSeparator()
    window._spectra_action = QAction("Vibrational spectra…", window)
    window._spectra_action.triggered.connect(window._open_spectra)
    plot_menu.addAction(window._spectra_action)
    # Likewise for the VCI wavefunctions: a run holds one state per
    # configuration, so which states and how much mixing to keep are choices
    # only the user can make (see crystalline.crystalio.vci). Listed only when
    # the installed CRYSTALClear can actually draw them.
    if vci_plottable():
        window._vci_action = QAction("VCI states…", window)
        window._vci_action.triggered.connect(window._open_vci)
        plot_menu.addAction(window._vci_action)
    # And for the anharmonic scan, where what goes on top of the potential —
    # wavefunctions, densities, how tall — is the choice, and the coefficients
    # come from a second file (see crystalline.crystalio.anscan).
    if anscan_plottable():
        window._anscan_action = QAction("Anharmonic scan…", window)
        window._anscan_action.triggered.connect(window._open_anscan)
        plot_menu.addAction(window._anscan_action)
    # And for the PES an ANHAPES run differentiates: a quartic surface over
    # every mode it was given, of which the figure is always a one- or two-mode
    # cut (see crystalline.crystalio.pes).
    if pes_plottable():
        window._pes_action = QAction("Anharmonic PES…", window)
        window._pes_action.triggered.connect(window._open_pes)
        plot_menu.addAction(window._pes_action)
    # Typography applies to every figure, not to one kind, so it sits on its own
    # at the foot of the menu rather than inside any of the plot entries.
    plot_menu.addSeparator()
    window._font_action = QAction("Plot font…", window)
    window._font_action.triggered.connect(window._open_plot_font)
    plot_menu.addAction(window._font_action)

    if not crystalclear_available():
        plot_menu.setEnabled(False)
        plot_menu.setTitle("&Plot (CRYSTALClear not installed)")
    window._update_plot_actions()


# ── Help ──────────────────────────────────────────────────────────────────
def _build_help_menu(window) -> None:
    help_menu = window.menuBar().addMenu("&Help")
    about_action = QAction("About CRYSTALLine", window)
    about_action.triggered.connect(window._show_about)
    help_menu.addAction(about_action)


def show_about(parent) -> None:
    """A small About dialog showing the logo and version."""
    from PySide6.QtCore import Qt as _Qt
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout

    from crystalline.resources import logo_path

    dialog = QDialog(parent)
    dialog.setWindowTitle("About CRYSTALLine")
    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(24, 20, 24, 16)
    layout.setSpacing(12)

    logo = QLabel()
    logo.setPixmap(QPixmap(logo_path()).scaledToWidth(320, _Qt.SmoothTransformation))
    logo.setAlignment(_Qt.AlignCenter)
    layout.addWidget(logo)

    caption = QLabel(
        "<div style='text-align:center'>"
        "<b>CRYSTALLine</b><br>"
        "A desktop viewer &amp; editor for CRYSTAL structures and phonons.<br>"
        "<span style='color:gray'>Built on CRYSTALClear.<br>"
    )
    caption.setTextFormat(_Qt.RichText)
    caption.setAlignment(_Qt.AlignCenter)
    layout.addWidget(caption)

    buttons = QDialogButtonBox(QDialogButtonBox.Close)
    buttons.rejected.connect(dialog.reject)
    buttons.accepted.connect(dialog.accept)
    layout.addWidget(buttons)
    dialog.exec()


# ── toolbars ──────────────────────────────────────────────────────────────
def _build_toolbars(window) -> None:
    """Toolbars for the most-used actions: Undo, and a/b/c view alignment."""
    edit_toolbar = QToolBar("Edit", window)
    edit_toolbar.addAction(window._undo_action)
    edit_toolbar.addAction(window._redo_action)
    window.addToolBar(edit_toolbar)

    view_toolbar = QToolBar("View", window)
    caption = QLabel("View along")
    caption.setContentsMargins(8, 0, 6, 0)
    view_toolbar.addWidget(caption)
    # Colour the a/b/c chips to match the lattice gizmo (a=red, b=green,
    # c=blue), so the button and the on-screen axis arrow read as the same.
    window._axis_buttons: list = []
    window._rotate_buttons: list = []
    for label, axis, color in (("a", 0, "#d62728"), ("b", 1, "#2ca02c"), ("c", 2, "#1f77b4")):
        button = QToolButton(window)
        button.setText(label)
        button.setToolTip(f"Look down the {label} axis")
        button.setStyleSheet(_axis_chip_style(color))
        button.clicked.connect(lambda _checked=False, a=axis: window.viewport.align_view_along(a))
        view_toolbar.addWidget(button)
        window._axis_buttons.append(button)

    # Orbit the view by a fixed step. Unlike a/b/c alignment these need no cell,
    # so they stay enabled for molecules too.
    view_toolbar.addWidget(_toolbar_spacer(10))
    rotate_caption = QLabel("Rotate")
    rotate_caption.setContentsMargins(2, 0, 6, 0)
    view_toolbar.addWidget(rotate_caption)
    # The last two spin the structure in the screen plane (about the axis
    # perpendicular to the screen) rather than orbiting the camera around it.
    for label, tooltip, azimuth, elevation, roll in (
        ("◀", "Rotate left", -_ROTATE_STEP_DEG, 0.0, 0.0),
        ("▶", "Rotate right", _ROTATE_STEP_DEG, 0.0, 0.0),
        ("▲", "Rotate up", 0.0, _ROTATE_STEP_DEG, 0.0),
        ("▼", "Rotate down", 0.0, -_ROTATE_STEP_DEG, 0.0),
        ("↺", "Rotate anticlockwise in the screen plane", 0.0, 0.0, -_ROTATE_STEP_DEG),
        ("↻", "Rotate clockwise in the screen plane", 0.0, 0.0, _ROTATE_STEP_DEG),
    ):
        button = QToolButton(window)
        button.setText(label)
        button.setToolTip(f"{tooltip} ({_ROTATE_STEP_DEG:g}°)")
        button.setAutoRepeat(True)  # hold to keep turning
        button.setStyleSheet(_axis_chip_style(_ROTATE_CHIP_COLOR))
        button.clicked.connect(
            lambda _checked=False, a=azimuth, e=elevation, r=roll: window.viewport.rotate_view(
                a, e, r
            )
        )
        view_toolbar.addWidget(button)
        window._rotate_buttons.append(button)

    view_toolbar.addWidget(_toolbar_spacer(6))
    window.addToolBar(view_toolbar)
    window._update_view_actions()


def _toolbar_spacer(width: int) -> QWidget:
    spacer = QWidget()
    spacer.setFixedWidth(width)
    return spacer


def _axis_chip_style(color: str) -> str:
    """Qt stylesheet for a rounded, coloured a/b/c axis button (with states)."""
    hover = QColor(color).lighter(115).name()
    pressed = QColor(color).darker(110).name()
    return f"""
        QToolButton {{
            background-color: {color};
            color: white;
            font-weight: bold;
            border: none;
            border-radius: 4px;
            padding: 4px 11px;
            margin: 2px 1px;
        }}
        QToolButton:hover {{ background-color: {hover}; }}
        QToolButton:pressed {{ background-color: {pressed}; }}
        QToolButton:disabled {{ background-color: #cccccc; color: #f0f0f0; }}
    """


__all__ = ["build_menus", "show_about"]
