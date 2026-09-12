"""The application's visual style: one stylesheet, light and dark.

Qt's default look is serviceable and generic. This gives the app a deliberate
one — flat surfaces, restrained borders, consistent radii and spacing — without
reaching for a whole widget toolkit.

Three rules kept it from going wrong:

* **Nothing styles bare ``QWidget``.** A rule on ``QWidget`` inherits into every
  child, including the VTK render window, and painting a background behind an
  OpenGL surface is at best wasted and at worst a flicker. Every rule here names
  the class it means.
* **Colours come from tokens, not from literals scattered through the sheet**,
  so the dark variant is the same sheet with a different table — and so a colour
  cannot drift between two controls that ought to match.
* **The system's own accent is not overridden** where Qt already provides one;
  selections and focus rings use the palette's highlight, so the app still looks
  like it belongs on the desktop it is running on.

The 3D view itself is not styled here: its background is a render setting the
user owns (Display ▸ Scene), not part of the chrome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from crystalline.resources import asset_path


@dataclass(frozen=True)
class Palette:
    """The colours one variant of the theme is built from."""

    window: str        # the app's ground
    surface: str       # panels and cards sitting on it
    raised: str        # controls: fields, buttons
    border: str        # ordinary separations
    border_strong: str # emphasis, focus, hover
    text: str
    text_muted: str    # secondary labels, hints, units
    accent: str        # the app's own colour, for checked and active states
    accent_soft: str   # a tint of it, for selected-but-quiet surfaces
    track: str         # slider grooves, scrollbar channels
    chrome: str        # toolbar and dock headers: the frame around the work
    scene: str         # the 3D viewport's ground


LIGHT = Palette(
    window="#eef0f4",
    surface="#ffffff",
    raised="#ffffff",
    border="#d8dbe2",
    border_strong="#aeb4bf",
    text="#1b1e24",
    text_muted="#69707c",
    accent="#3d6ff0",
    accent_soft="#e4ebfd",
    track="#e0e3e9",
    chrome="#e6e9ef",
    scene="#ffffff",
)

DARK = Palette(
    window="#1c1f24",
    surface="#25292f",
    raised="#2c3138",
    border="#373c44",
    border_strong="#4e555f",
    text="#e7eaef",
    text_muted="#98a0ad",
    accent="#5b8dee",
    accent_soft="#2b3a5c",
    track="#343941",
    chrome="#191c20",
    # Not pure black: a structure lit against #000 loses its darkest faces
    # entirely, and the cell wireframe with them.
    scene="#14161a",
)

# The check, radio and mixed glyphs. Styling an indicator's background replaces
# whatever the platform style drew inside it, so without these a ticked box is a
# plain blue square — distinguishable from an unticked one only by colour, which
# is a weak signal and no signal at all to someone who cannot tell the two apart.
_CHECK_GLYPH = asset_path("check.svg")
_RADIO_GLYPH = asset_path("radio.svg")
_MIXED_GLYPH = asset_path("mixed.svg")

# The lattice-axis colours, shared with the 3D gizmo so a toolbar chip and the
# arrow it aims at are recognisably the same axis (a=red, b=green, c=blue, as in
# VESTA). Slightly deeper than the arrows' own so they hold up as a filled chip.
# Mirrors ``renderer._LATTICE_COLORS`` — the chips and the 3D gizmo have to be
# recognisably the same axis. Copied rather than imported so styling the app does
# not drag VTK in at startup; a test asserts the two stay in step.
_AXIS_A = "#d62728"
_AXIS_B = "#2ca02c"
_AXIS_C = "#1f77b4"

# The same three, for anything outside this module that has to match a chip —
# the Brillouin zone's k_x/k_y/k_z axes, drawn in the colours of the buttons
# that aim down them.
AXIS_COLOURS = (_AXIS_A, _AXIS_B, _AXIS_C)

# Geometry shared by every control, so nothing is a one-off.
_RADIUS = 5
_RADIUS_SMALL = 4
_FIELD_PADDING = "4px 7px"


def stylesheet(palette: Palette) -> str:
    """The application stylesheet for one colour table."""
    p = palette
    # A spin box's arrows are drawn by the style, and Fusion's are two or three
    # pixels of near-invisible triangle — smaller still once the field has
    # comfortable padding. These are drawn chevrons at a readable size, in the
    # one shade that has to be picked per theme: a stylesheet cannot recolour an
    # image, so each variant ships its own pair.
    tone = "light" if palette is DARK else "dark"
    chevron_up = asset_path(f"chevron-up-{tone}.svg")
    chevron_down = asset_path(f"chevron-down-{tone}.svg")
    return f"""
    /* ── surfaces ─────────────────────────────────────────────────────── */
    QMainWindow, QDialog {{
        background-color: {p.window};
    }}
    QDockWidget {{
        color: {p.text};
        titlebar-close-icon: none;
        titlebar-normal-icon: none;
    }}
    QDockWidget::title {{
        background-color: {p.chrome};
        color: {p.text_muted};
        padding: 7px 12px;
        border-bottom: 1px solid {p.border};
        font-weight: 700;
        text-align: left;
    }}

    /* ── sections ─────────────────────────────────────────────────────────
       Flat, not cards. A stack of bordered boxes down one panel reads as a form
       rather than a tool, and the nested borders eat the width the rows need.
       A small header and honest whitespace group just as well. */
    QLabel[role="section"] {{
        color: {p.text_muted};
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.09em;
        padding: 0 0 2px 0;
    }}
    /* A collapsible section's header is a button, so that it can be clicked —
       but it must read as a heading, not as a control: no frame, no fill, and
       the same weight as the labels beside it. */
    QToolButton[role="section"] {{
        color: {p.text_muted};
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.09em;
        background: transparent;
        border: none;
        padding: 2px 0;
        text-align: left;
    }}
    QToolButton[role="section"]:hover {{ color: {p.text}; }}
    QToolButton[role="section"]:pressed {{ background: transparent; }}
    QGroupBox {{
        background: transparent;
        border: none;
        margin-top: 16px;
        padding: 6px 0 0 0;
        color: {p.text};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 0px;
        padding: 0;
        color: {p.text_muted};
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.09em;
    }}
    /* A *checkable* group box draws its own indicator, and it is a separate
       sub-control from QCheckBox::indicator. Styling only the latter left the
       group box's tick with no box at all — invisible when unchecked, and a
       bare floating tick when checked. These mirror the checkbox rules. */
    QGroupBox::indicator {{
        width: 15px; height: 15px;
        border: 1px solid {p.border_strong};
        border-radius: 3px;
        background-color: {p.raised};
    }}
    QGroupBox::indicator:checked {{
        background-color: {p.accent};
        border-color: {p.accent};
        image: url("{_CHECK_GLYPH}");
    }}
    QGroupBox::indicator:hover {{ border-color: {p.accent}; }}
    QGroupBox::indicator:disabled {{
        border-color: {p.border};
        background-color: {p.window};
    }}
    /* Disabled *and* checked keeps a filled box, or the white tick is drawn on
       the light theme's pale ground and vanishes — the state reads as
       unchecked, which is a different thing entirely. */
    QGroupBox::indicator:checked:disabled,
    QCheckBox::indicator:checked:disabled,
    QRadioButton::indicator:checked:disabled {{
        background-color: {p.border_strong};
        border-color: {p.border_strong};
    }}

    /* ── text and labels ──────────────────────────────────────────────── */
    QLabel {{ color: {p.text}; background: transparent; }}
    QLabel:disabled {{ color: {p.text_muted}; }}

    /* ── buttons ──────────────────────────────────────────────────────── */
    QPushButton {{
        background-color: {p.raised};
        border: 1px solid {p.border};
        border-radius: {_RADIUS_SMALL}px;
        padding: 5px 12px;
        color: {p.text};
    }}
    QPushButton:hover {{ border-color: {p.border_strong}; }}
    QPushButton:pressed {{ background-color: {p.track}; }}
    QPushButton:disabled {{ color: {p.text_muted}; border-color: {p.border}; }}
    QPushButton:default {{ border-color: {p.accent}; }}

    /* ── fields ───────────────────────────────────────────────────────── */
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
        background-color: {p.raised};
        border: 1px solid {p.border};
        border-radius: {_RADIUS_SMALL}px;
        padding: {_FIELD_PADDING};
        color: {p.text};
        selection-background-color: {p.accent};
        selection-color: #ffffff;
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
    QPlainTextEdit:focus, QTextEdit:focus {{
        border-color: {p.accent};
    }}
    QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
    QComboBox:disabled {{
        color: {p.text_muted};
        background-color: {p.window};
    }}
    QComboBox::drop-down {{ border: none; width: 20px; }}
    QComboBox::down-arrow {{ image: url("{chevron_down}"); width: 9px; height: 7px; }}

    QSpinBox::up-button, QDoubleSpinBox::up-button,
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        subcontrol-origin: border;
        background: transparent;
        border: none;
        width: 18px;
        margin: 1px 2px 1px 0;
    }}
    QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-position: top right; }}
    QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-position: bottom right; }}
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
        background: {p.track};
        border-radius: 3px;
    }}
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
        image: url("{chevron_up}"); width: 9px; height: 7px;
    }}
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
        image: url("{chevron_down}"); width: 9px; height: 7px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {p.raised};
        border: 1px solid {p.border};
        selection-background-color: {p.accent};
        selection-color: #ffffff;
        outline: none;
    }}

    /* ── checkboxes ───────────────────────────────────────────────────── */
    QCheckBox, QRadioButton {{ color: {p.text}; spacing: 7px; background: transparent; }}
    QCheckBox:disabled, QRadioButton:disabled {{ color: {p.text_muted}; }}
    QCheckBox::indicator, QRadioButton::indicator {{
        width: 15px; height: 15px;
        border: 1px solid {p.border_strong};
        background-color: {p.raised};
    }}
    QCheckBox::indicator {{ border-radius: 3px; }}
    QRadioButton::indicator {{ border-radius: 8px; }}
    QCheckBox::indicator:checked {{
        background-color: {p.accent};
        border-color: {p.accent};
        image: url("{_CHECK_GLYPH}");
    }}
    QCheckBox::indicator:indeterminate {{
        background-color: {p.accent};
        border-color: {p.accent};
        image: url("{_MIXED_GLYPH}");
    }}
    QRadioButton::indicator:checked {{
        background-color: {p.accent};
        border-color: {p.accent};
        image: url("{_RADIO_GLYPH}");
    }}
    QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
        border-color: {p.accent};
    }}
    QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
        border-color: {p.border};
        background-color: {p.window};
    }}

    /* ── sliders ──────────────────────────────────────────────────────── */
    QSlider::groove:horizontal {{
        height: 4px;
        background: {p.track};
        border-radius: 2px;
    }}
    QSlider::sub-page:horizontal {{
        background: {p.accent};
        border-radius: 2px;
    }}
    QSlider::groove:horizontal {{
        margin: 0 1px;
    }}
    QSlider::handle:horizontal {{
        width: 13px;
        height: 13px;
        margin: -5px 0;
        border-radius: 7px;
        background: {p.surface};
        border: 1px solid {p.border_strong};
    }}
    /* A handle drawn taller than the groove needs the row to be that tall, or
       Qt clips its top against the widget's own rectangle. */
    QSlider:horizontal {{ min-height: 20px; }}
    QSlider::handle:horizontal:hover {{ border-color: {p.accent}; }}
    QSlider::groove:horizontal:disabled {{ background: {p.border}; }}
    QSlider::sub-page:horizontal:disabled {{ background: {p.border_strong}; }}

    /* ── lists, trees, tables ─────────────────────────────────────────── */
    QListWidget, QTreeWidget, QTableWidget, QListView, QTreeView {{
        background-color: {p.raised};
        border: 1px solid {p.border};
        border-radius: {_RADIUS_SMALL}px;
        outline: none;
        color: {p.text};
    }}
    QListWidget::item, QTreeWidget::item {{
        padding: 3px 4px;
        border-radius: 3px;
    }}
    QListWidget::item:selected, QTreeWidget::item:selected {{
        background-color: {p.accent};
        color: #ffffff;
    }}
    /* Not on the selected row. Without the guard, hovering a selected item
       repaints it in the plain hover colour — the selection appears to vanish
       under the pointer and come back when it leaves, which is what made the
       mode list feel unpredictable to move over. */
    QListWidget::item:hover:!selected, QTreeWidget::item:hover:!selected {{
        background-color: {p.track};
        color: {p.text};
    }}
    QHeaderView::section {{
        background-color: {p.window};
        color: {p.text_muted};
        border: none;
        border-bottom: 1px solid {p.border};
        padding: 4px 6px;
        font-weight: 600;
    }}

    /* ── tabs ─────────────────────────────────────────────────────────────
       Two different things share this class and must not share a style:

       * a QTabWidget's tabs sit *above* their pane, so they lean on it — top
         corners rounded, bottom edge merged into the page below;
       * a QMainWindow's dock tabs sit *below* their dock. Given the same style
         they come out upside down: rounded at the top and merging downwards
         into a status bar they have nothing to do with, which is exactly what
         made them look broken.

       So the dock tabs get a shape with no up or down about it — a filled pill,
       tinted with the accent when selected. It reads the same wherever Qt
       decides to put the bar. */
    QTabWidget::pane {{
        background: {p.surface};
        border: 1px solid {p.border};
        border-radius: {_RADIUS_SMALL}px;
        top: -1px;
    }}
    QTabWidget > QTabBar::tab {{
        background: transparent;
        color: {p.text_muted};
        padding: 6px 14px;
        margin-right: 2px;
        border: 1px solid transparent;
        border-top-left-radius: {_RADIUS_SMALL}px;
        border-top-right-radius: {_RADIUS_SMALL}px;
    }}
    QTabWidget > QTabBar::tab:selected {{
        background: {p.surface};
        color: {p.text};
        border-color: {p.border};
        border-bottom-color: {p.surface};
    }}
    QTabWidget > QTabBar::tab:hover:!selected {{ color: {p.text}; }}

    /* dock tabs */
    QMainWindow > QTabBar {{
        background: {p.chrome};
        qproperty-drawBase: 0;
    }}
    QMainWindow > QTabBar::tab {{
        background: transparent;
        color: {p.text_muted};
        border: none;
        border-radius: {_RADIUS_SMALL}px;
        padding: 6px 14px;
        margin: 4px 2px;
        font-weight: 600;
    }}
    QMainWindow > QTabBar::tab:selected {{
        background: {p.accent_soft};
        color: {p.accent};
    }}
    QMainWindow > QTabBar::tab:hover:!selected {{
        background: {p.track};
        color: {p.text};
    }}

    /* ── toolbars, menus, status ──────────────────────────────────────── */
    QToolBar {{
        background-color: {p.chrome};
        border: none;
        border-bottom: 1px solid {p.border};
        spacing: 3px;
        padding: 5px 8px;
    }}
    QToolBar::separator {{ background: {p.border}; width: 1px; margin: 5px 8px; }}
    /* Text actions (Undo/Redo) sat as bare words next to the coloured chips,
       which read as an unfinished row. Give them the same chip shape so the
       toolbar is one kind of thing throughout. */
    QToolBar QToolButton {{
        background: transparent;
        border: 1px solid transparent;
        border-radius: {_RADIUS_SMALL}px;
        padding: 4px 9px;
        color: {p.text};
    }}
    QToolBar QToolButton:hover {{
        background: {p.track};
        border-color: {p.border};
    }}
    QToolBar QToolButton:pressed {{ background: {p.border}; }}
    QToolBar QToolButton:disabled {{ color: {p.text_muted}; }}
    /* The small captions that group the row ("View along", "Rotate", "Cell"). */
    QToolBar QLabel {{
        color: {p.text_muted};
        font-size: 11px;
        font-weight: 700;
    }}
    QMenuBar {{ background-color: {p.window}; color: {p.text}; }}
    QMenuBar::item {{ padding: 5px 10px; background: transparent; }}
    QMenuBar::item:selected {{ background: {p.track}; border-radius: {_RADIUS_SMALL}px; }}
    QMenu {{
        background-color: {p.surface};
        border: 1px solid {p.border};
        border-radius: {_RADIUS_SMALL}px;
        padding: 4px;
    }}
    QMenu::item {{ padding: 5px 22px 5px 20px; border-radius: 3px; }}
    QMenu::item:selected {{ background-color: {p.accent}; color: #ffffff; }}
    QMenu::item:disabled {{ color: {p.text_muted}; }}
    QMenu::separator {{ height: 1px; background: {p.border}; margin: 4px 8px; }}
    QStatusBar {{
        background-color: {p.window};
        border-top: 1px solid {p.border};
        color: {p.text_muted};
    }}
    QStatusBar::item {{ border: none; }}

    /* ── scrollbars: slim, and out of the way ─────────────────────────── */
    QScrollBar:vertical {{
        background: transparent; width: 10px; margin: 0;
    }}
    QScrollBar:horizontal {{
        background: transparent; height: 10px; margin: 0;
    }}
    QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
        background: {p.border_strong};
        border-radius: 5px;
        min-height: 24px;
        min-width: 24px;
    }}
    QScrollBar::handle:hover {{ background: {p.text_muted}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

    /* ── toolbar chips ────────────────────────────────────────────────────
       Declared here rather than on the buttons so a theme change restyles them
       with everything else — a chip that painted itself would keep its light
       colours on a dark toolbar.

       Three weights, by how much each one has to say. The a/b/c chips are solid
       because their colour *is* the information: each matches the axis arrow of
       the same colour in the 3D view. Everything else is quiet by default and
       only fills in when it is on. */
    QToolBar QToolButton[chip="axis"] {{
        color: #ffffff;
        font-weight: 700;
        border: none;
        border-radius: 5px;
        padding: 4px 11px;
    }}
    QToolBar QToolButton[chip="axis"][axis="a"] {{ background-color: {_AXIS_A}; }}
    QToolBar QToolButton[chip="axis"][axis="b"] {{ background-color: {_AXIS_B}; }}
    QToolBar QToolButton[chip="axis"][axis="c"] {{ background-color: {_AXIS_C}; }}
    QToolBar QToolButton[chip="axis"]:hover {{ border: none; }}
    QToolBar QToolButton[chip="axis"]:disabled {{
        background-color: {p.track};
        color: {p.text_muted};
    }}

    QToolBar QToolButton[chip="ghost"] {{
        background: transparent;
        color: {p.text};
        border: 1px solid {p.border_strong};
        border-radius: 5px;
        padding: 4px 9px;
        font-weight: 700;
    }}
    QToolBar QToolButton[chip="ghost"]:hover {{
        background: {p.track};
        color: {p.text};
        border-color: {p.border_strong};
    }}
    QToolBar QToolButton[chip="ghost"]:pressed {{ background: {p.border}; }}

    QToolBar QToolButton[chip="toggle"] {{
        background: transparent;
        color: {p.text_muted};
        border: 1px solid {p.border};
        border-radius: 5px;
        padding: 4px 11px;
        font-weight: 700;
    }}
    QToolBar QToolButton[chip="toggle"]:hover {{ background: {p.track}; color: {p.text}; }}
    QToolBar QToolButton[chip="toggle"]:checked {{
        background: {p.accent};
        color: #ffffff;
        border-color: {p.accent};
    }}

    /* ── misc ─────────────────────────────────────────────────────────── */
    QToolTip {{
        background-color: {p.surface};
        color: {p.text};
        border: 1px solid {p.border};
        border-radius: {_RADIUS_SMALL}px;
        padding: 4px 7px;
    }}
    QSplitter::handle {{ background: {p.border}; }}
    QProgressBar {{
        border: 1px solid {p.border};
        border-radius: {_RADIUS_SMALL}px;
        background: {p.track};
        text-align: center;
        color: {p.text};
    }}
    QProgressBar::chunk {{ background-color: {p.accent}; border-radius: 3px; }}
    """


def monochrome_icon(name: str, color: str, size: int = 18):
    """A bundled single-colour SVG glyph, drawn in ``color``, as a ``QIcon``.

    The undo/redo icons used to come from ``QIcon.fromTheme`` with a
    ``QStyle.standardIcon`` fallback. That is null on macOS (there is no
    freedesktop icon theme) and the fallback is drawn by the *style* — so
    switching the app to Fusion, which a stylesheet needs to be honoured
    consistently, silently replaced them with nothing and the toolbar fell back
    to showing the words. Drawing our own removes the platform from the question
    entirely.

    The colour is substituted at load, so one file serves both themes: a glyph
    baked dark would vanish into the dark toolbar.
    """
    from PySide6.QtCore import QByteArray, Qt
    from PySide6.QtGui import QIcon, QPainter, QPixmap
    from PySide6.QtSvg import QSvgRenderer
    from PySide6.QtWidgets import QApplication

    from crystalline.resources import asset_path

    try:
        with open(asset_path(name), "r") as handle:
            svg = handle.read().replace("GLYPH", color)
        renderer = QSvgRenderer(QByteArray(svg.encode()))

        # Drawn at the screen's real pixel density, not at logical size. A pixmap
        # made 18x18 on a 2x display is 18 physical pixels blown up to 36 — which
        # is exactly the soft, low-resolution look the glyphs had. Rendering at
        # 36 and declaring the ratio lets Qt place it at 18 points and paint
        # every pixel.
        app = QApplication.instance()
        ratio = float(app.devicePixelRatio()) if app is not None else 1.0
        ratio = ratio if ratio > 0 else 1.0
        pixmap = QPixmap(int(round(size * ratio)), int(round(size * ratio)))
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        try:
            renderer.render(painter)
        finally:
            painter.end()
        pixmap.setDevicePixelRatio(ratio)
        return QIcon(pixmap)
    except Exception:  # noqa: BLE001 - no QtSvg, or a missing file: no icon
        from PySide6.QtGui import QIcon as _QIcon

        return _QIcon()


def tidy_forms(widget) -> None:
    """Left-align the labels of every ``QFormLayout`` under ``widget``.

    The panels were rebuilt on a grid with a fixed label column so their rows
    line up. The dialogs are still forms, and Qt's default form right-aligns
    each label against its own field — which is exactly what made the panels
    look unfinished: every control starts wherever its label happens to end, so
    no two rows agree.

    Done by walking the widget rather than by overriding the style's layout
    hints. That looked tidier — one override for the whole app — but a
    ``QProxyStyle`` with a Python ``styleHint`` puts a Python call on every hint
    query during painting, and marshalling the hint's ``returnData`` back out
    segfaults. This is duller and it survives a repaint.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QFormLayout

    for form in widget.findChildren(QFormLayout):
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(7)


def qpalette(palette: Palette):
    """A ``QPalette`` matching the sheet, for everything the sheet does not name.

    Deliberately nothing here styles bare ``QWidget`` (a rule there inherits into
    the VTK render window), so plain container widgets — a panel's own body, a
    scroll area's viewport, a dock's ground — take their colour from the palette
    instead. If the two disagree the result is a dark sheet on light containers,
    which is worse than either on its own. Setting both from one table is what
    keeps them together.
    """
    from PySide6.QtGui import QColor, QPalette

    p = palette
    out = QPalette()
    window, base, text = QColor(p.window), QColor(p.raised), QColor(p.text)
    muted, accent = QColor(p.text_muted), QColor(p.accent)

    out.setColor(QPalette.Window, window)
    out.setColor(QPalette.WindowText, text)
    out.setColor(QPalette.Base, base)
    out.setColor(QPalette.AlternateBase, QColor(p.surface))
    out.setColor(QPalette.Text, text)
    out.setColor(QPalette.Button, QColor(p.raised))
    out.setColor(QPalette.ButtonText, text)
    out.setColor(QPalette.ToolTipBase, QColor(p.surface))
    out.setColor(QPalette.ToolTipText, text)
    out.setColor(QPalette.Highlight, accent)
    out.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    out.setColor(QPalette.PlaceholderText, muted)
    # The sliding switches paint their off-state track from Mid, so it has to be
    # a real colour here rather than whatever Fusion derives from Window.
    out.setColor(QPalette.Mid, QColor(p.border_strong))
    out.setColor(QPalette.Midlight, QColor(p.track))
    out.setColor(QPalette.Link, accent)
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        out.setColor(QPalette.Disabled, role, muted)
    return out


# How the app decides which variant to wear. "system" follows the desktop and is
# the default; the other two are a deliberate override, which matters for a 3D
# viewer — people run one on a dark desktop and still want a light figure, or the
# reverse — and because a desktop that reports no preference at all would
# otherwise leave no way to ask for dark.
MODES = ("system", "light", "dark")
# Light by default. Dark is a click away (the toolbar lamp, or View ▸
# Appearance), and so is following the desktop.
DEFAULT_MODE = "light"

_SETTINGS_ORGANISATION = "CRYSTALLine"
_SETTINGS_APPLICATION = "CRYSTALLine"
_SETTINGS_KEY = "appearance/mode"


def _settings():
    from PySide6.QtCore import QSettings

    return QSettings(_SETTINGS_ORGANISATION, _SETTINGS_APPLICATION)


def current_mode() -> str:
    """The stored appearance choice, or ``"system"`` if none has been made."""
    try:
        mode = str(_settings().value(_SETTINGS_KEY, DEFAULT_MODE))
    except Exception:  # noqa: BLE001 - unreadable settings: the default still works
        return DEFAULT_MODE
    return mode if mode in MODES else DEFAULT_MODE


def set_mode(app, mode: str) -> None:
    """Choose the appearance and remember it for next time."""
    if mode not in MODES:
        raise ValueError(f"unknown appearance mode {mode!r}; expected one of {MODES}")
    try:
        _settings().setValue(_SETTINGS_KEY, mode)
    except Exception:  # noqa: BLE001 - can't persist; still apply it for this session
        pass
    _paint(app, palette_for(app, mode))


def is_dark(app) -> bool:
    """Whether the desktop is asking for a dark UI.

    Qt 6.5 states it directly; older builds only have the palette, where a dark
    window colour is the usable signal.
    """
    try:
        from PySide6.QtCore import Qt as _Qt

        scheme = app.styleHints().colorScheme()
        if scheme == _Qt.ColorScheme.Dark:
            return True
        if scheme == _Qt.ColorScheme.Light:
            return False
    except Exception:  # noqa: BLE001 - older Qt, or no style hints: fall back
        pass
    try:
        from PySide6.QtGui import QPalette

        return app.palette().color(QPalette.Window).lightness() < 128
    except Exception:  # noqa: BLE001
        return False


# The table actually painted, which is not always the one the stored mode names:
# a caller can paint a palette directly. Anything that bakes a colour in — the
# toolbar glyphs — has to read *this*, or it draws for a theme that isn't on.
_active: Optional[Palette] = None


def scene_backgrounds() -> tuple:
    """Every ground the themes supply, so a caller can tell one from a user's own."""
    return (LIGHT.scene, DARK.scene)


def active_palette(app) -> Palette:
    """The colour table currently painted on ``app``."""
    return _active if _active is not None else palette_for(app)


def palette_for(app, mode: Optional[str] = None) -> Palette:
    """The colour table for ``mode`` (default: the stored choice)."""
    mode = mode or current_mode()
    if mode == "dark":
        return DARK
    if mode == "light":
        return LIGHT
    return DARK if is_dark(app) else LIGHT


def apply(app) -> None:
    """Style ``app``, and keep it in step if the desktop theme changes.

    The style is set to Fusion first: it is the one Qt style that honours a
    stylesheet consistently across platforms, where the native macOS and Windows
    styles paint some controls themselves and ignore half of what is asked of
    them — which is how a stylesheet ends up looking right on one machine and
    half-applied on another.
    """
    try:
        app.setStyle("Fusion")
    except Exception:  # noqa: BLE001 - an unusual Qt build; the sheet still applies
        pass
    _paint(app, palette_for(app))

    try:  # follow the desktop, but only while the choice is to follow it
        app.styleHints().colorSchemeChanged.connect(
            lambda *_: current_mode() == "system" and _paint(app, palette_for(app))
        )
    except Exception:  # noqa: BLE001 - older Qt has no such signal
        pass


def _paint(app, palette: Palette) -> None:
    """Put one colour table on the app, through both palette and stylesheet.

    Anything that baked a colour in — the toolbar's undo/redo glyphs, which are
    rendered in the text colour — has to be redrawn, so every top-level window is
    asked to refresh what it owns.
    """
    global _active
    _active = palette
    try:
        app.setPalette(qpalette(palette))
    except Exception:  # noqa: BLE001 - the sheet alone still covers most of it
        pass
    app.setStyleSheet(stylesheet(palette))
    for window in app.topLevelWidgets():
        refresh = getattr(window, "refresh_theme", None)
        if callable(refresh):
            try:
                refresh()
            except Exception:  # noqa: BLE001 - a window that can't restyle is not fatal
                pass


__all__ = [
    "DARK",
    "DEFAULT_MODE",
    "LIGHT",
    "MODES",
    "Palette",
    "apply",
    "is_dark",
    "current_mode",
    "monochrome_icon",
    "palette_for",
    "qpalette",
    "scene_backgrounds",
    "set_mode",
    "stylesheet",
    "tidy_forms",
]
