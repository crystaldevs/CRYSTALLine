"""An embeddable panel that holds property plots as closable tabs.

Used for the plots produced through :mod:`crystalline.crystalio.plotting`
(band structures, DOS, spectra…). Rather than spawning a separate top-level
window per plot, :class:`PlotPanel` lives inside the main window (docked, like
the Info/Phonons panels) and stacks each new plot as its own tab, so several
plots can be built up and compared without a scatter of floating windows. The
dock can still be dragged out and floated by the user if they want one on its
own.

The matplotlib Qt backend is imported lazily so importing the UI package doesn't
pull matplotlib in for users who never open a plot.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from PySide6.QtCore import QEvent, QObject, QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSizePolicy,
    QScrollArea,
    QSpinBox,
    QStackedLayout,
    QStyle,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from crystalline.ui.panels.controls import PANEL_MARGINS
from crystalline.ui.widgets.tab_bar import SlidingTabBar


@dataclass
class PlotRecipe:
    """How a plot was made, kept with it so that it can be made again.

    A figure arriving here is otherwise a dead end: the data it was drawn from,
    the function that drew it and the options it was given all live in the
    method that built it, and are gone when that returns. Nothing in the panel
    can then offer to *change* a plot — only the menu can build another one,
    in another tab, from a dialog that starts where the last one was left.

    ``draw`` takes options and returns a new figure. ``options`` are the ones
    this plot was drawn with, and they belong to the tab rather than to the
    dialog: two spectra tabs can be broadened differently, which the one
    remembered setting per dialog could never express. ``edit`` asks for new
    options, starting from these, and returns them — or ``None`` if the user
    changed their mind. A recipe without an ``edit`` can still be redrawn by
    whatever else learns to ask (a control beside the figure, say).
    """

    draw: Callable[..., Any]
    options: Dict[str, Any] = field(default_factory=dict)
    edit: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = None
    # What to call the tab for a given set of options. A spectrum named after
    # its one curve is named after the count once a second is ticked, so the
    # name follows the options rather than being fixed when the tab is made.
    title: Optional[Callable[[Dict[str, Any]], str]] = None
    # The controls that belong beside the plot: a widget with ``options()``,
    # ``set_options()`` and a ``changed`` signal. Given one, the tab shows it
    # and redraws as it is turned, which is the whole point of keeping a recipe
    # — a width is judged against the peaks it merges, and the only way to know
    # is to look while it moves. Only for a plot that is cheap to draw again.
    controls: Optional[Callable[[], Any]] = None
    # The shape the figure wants, width over height, for given options: the
    # Plots window opens at it, and widens to it for a plot that needs more
    # than the window has. None is the 4:3 every plot gets by default. A band
    # structure beside its DOS is two panels in one figure, and at 4:3 the
    # corners of the path ran into each other along the bottom.
    aspect: Optional[Callable[[Dict[str, Any]], float]] = None

    def redrawn(self, options: Optional[Dict[str, Any]] = None):
        """``(figure, options)`` for ``options``, or for the ones in hand.

        One at a time, whichever thread asks: the plotters go through pyplot,
        whose register of open figures is not safe to use from two threads at
        once, and a redraw asked for beside one plot can now be under way
        while another plot is being built.
        """
        chosen = dict(self.options if options is None else options)
        with _DRAWING:
            return self.draw(**chosen), chosen


# Held while any recipe draws. Re-entrant, so that a draw which builds another
# figure on the way — none does yet — would not wait for itself.
_DRAWING = threading.RLock()


class LiveRedraws(QObject):
    """Redraws asked for by the controls beside plots, drawn off the UI thread.

    A plotter can take seconds — CRYSTALClear broadens a spectrum one grid
    point at a time — and drawn on the UI thread, every knob that came to rest
    froze the whole window for as long as that took. Here the figure is made
    on a worker and only put on screen back on the UI thread, which is the only
    thread a canvas may be touched from.

    The newest request wins. While one figure is being drawn, further turns of
    a knob only replace what is waiting, so a long drag costs one draw more
    than it would have if the hand had been slower, never one per step. One
    figure is drawn at a time, across every plot.

    A figure goes back to the plot tab that asked for it, wherever that tab is
    by then — behind another file's tab, say — and is dropped if the tab has
    been closed. :meth:`withdraw` is for an edit: what the dialog just set
    outranks a redraw still under way from before it.
    """

    #: How long a redraw may take before the status bar says it is under way:
    #: long enough that a quick one does not flash a message, short enough
    #: that a slow one does not look like a knob that did nothing.
    SLOW_MS = 300

    def __init__(self, status_bar, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._bar = status_bar
        self._waiting: Dict[Any, dict] = {}     # tab -> the newest options asked of it
        self._drawing: Optional[tuple] = None   # (tab, options) under way
        self._worker = None
        self._withdrawn: Dict[Any, int] = {}    # tab -> how many edits have overtaken it
        self._slow = QTimer(self)
        self._slow.setSingleShot(True)
        self._slow.setInterval(self.SLOW_MS)
        self._slow.timeout.connect(lambda: self._bar.showMessage(_DRAWING_MESSAGE))

    # ── asked ───────────────────────────────────────────────────────────
    def request(self, tab, options: dict) -> None:
        """Draw ``tab``'s plot again with ``options`` over the ones it has."""
        if getattr(tab, "recipe", None) is None:
            return
        self._waiting[tab] = dict(options)
        self._next()

    def withdraw(self, tab) -> None:
        """Forget what is waiting for ``tab``, and drop what is being drawn for it."""
        self._waiting.pop(tab, None)
        self._withdrawn[tab] = self._withdrawn.get(tab, 0) + 1

    def wanted(self, tab) -> dict:
        """The newest options asked of ``tab`` and not on screen yet, if any.

        For an edit started in the meantime: the dialog should open on what the
        controls beside the plot say, not on the figure they are about to
        replace.
        """
        if tab in self._waiting:
            return dict(self._waiting[tab])
        if self._drawing is not None and self._drawing[0] is tab:
            return dict(self._drawing[1])
        return {}

    def idle(self) -> bool:
        return self._worker is None and not self._waiting

    # ── drawing ─────────────────────────────────────────────────────────
    def _next(self) -> None:
        if self._worker is not None:
            return
        while self._waiting:
            tab, options = next(iter(self._waiting.items()))
            del self._waiting[tab]
            if _alive(tab):
                break
        else:
            return
        from crystalline.ui.widgets.busy import Worker

        recipe = tab.recipe
        merged = {**recipe.options, **options}
        overtaken = self._withdrawn.get(tab, 0)
        self._drawing = (tab, options)
        self._worker = Worker(lambda: recipe.redrawn(merged))
        self._worker.finished.connect(lambda drawn: self._done(tab, overtaken, drawn))
        self._worker.failed.connect(self._failed)
        self._slow.start()
        self._worker.start()

    def _done(self, tab, overtaken: int, drawn) -> None:
        figure, chosen = drawn
        panel = tab._panel() if _alive(tab) else None
        index = panel.index_of(tab) if panel is not None else -1
        if index < 0 or self._withdrawn.get(tab, 0) != overtaken:
            import matplotlib.pyplot as plt

            plt.close(figure)    # closed meanwhile, or an edit has been made since
        else:
            recipe = tab.recipe
            recipe.options = chosen
            title = recipe.title(chosen) if recipe.title is not None else None
            panel.replace_figure(index, figure, title)
        self._finish()

    def _failed(self, exc) -> None:
        # The status bar rather than a dialog: one of several, perhaps, as a
        # slider passes through a range that cannot be drawn, and three modal
        # apologies in a row is not how to say so.
        self._finish()
        self._bar.showMessage(f"Could not draw the plot: {exc}", 6000)

    def _finish(self) -> None:
        self._slow.stop()
        if self._bar.currentMessage() == _DRAWING_MESSAGE:
            self._bar.clearMessage()
        self._worker = None
        self._drawing = None
        self._next()


_DRAWING_MESSAGE = "Drawing the plot again…"


def _alive(widget) -> bool:
    """Whether ``widget``'s Qt side still exists — a closed tab's has gone."""
    import shiboken6

    return shiboken6.isValid(widget)


# Vector formats ignore raster DPI; JPEG has no alpha channel. Used to grey out
# the controls that don't apply to the chosen format.
# The room the controls beside a figure are given, whichever plot it is: the
# widest of the app's own, the band structure's energy window — a range slider
# between two value boxes — so that every plot gets the same share of the
# window. Each took its own width before, and the figure in a window of one
# size came out a different size for every kind of plot.
_CONTROLS_WIDTH = 400

# The widest column made so far, which every column is given: one whose
# controls need more than _CONTROLS_WIDTH (larger fonts, another platform)
# widens the rest to match rather than standing out from them.
_COLUMN = {"width": 0}
#: How far the figure's sheet stands in from the tab's edges.
_PAPER_INSET = 14

# How long the controls must be still before the plot is drawn again. Long
# enough that a drag is not a slideshow, short enough that letting go feels
# like the plot answering rather than catching up.
_REDRAW_REST_MS = 180

_VECTOR_FORMATS = frozenset({"svg", "pdf", "eps", "ps"})
_OPAQUE_FORMATS = frozenset({"jpg", "jpeg"})


def _finish(figure) -> None:
    """The house style's finishing pass, on every figure that reaches a tab.

    Here because every plot arrives here, whichever code drew it — a new tab or
    a tab redrawn with new options — and the pass is idempotent.
    """
    try:
        from crystalline.crystalio import figure_style
    except Exception:  # noqa: BLE001 - a plot is still worth showing unstyled
        return
    try:
        figure_style.finish(figure)
    except Exception:  # noqa: BLE001 - likewise
        pass


def _unfolded_width(controls: QWidget) -> int:
    """The least width ``controls`` need with every section open.

    A folded section's rows are hidden, and a hidden widget asks for no room:
    measured as they open, the controls were as wide as their open sections,
    and unfolding a wider one — a range slider with its two boxes — squeezed
    its rows into a column already fixed.
    """
    folded = [header._section.body for header in controls.findChildren(QToolButton)
              if getattr(header, "_section", None) is not None
              and header._section.body.isHidden()]
    for body in folded:
        body.setVisible(True)     # not on screen: the controls are not shown yet
    try:
        # A minimum set outright counts as much as one the layout asks for.
        return max(controls.minimumSizeHint().width(), controls.minimumWidth())
    finally:
        for body in folded:
            body.setVisible(False)


class _PlotTab(QWidget):
    """One matplotlib ``Figure`` embedded with a slim pan/zoom/save button row.

    The stock matplotlib ``NavigationToolbar2QT`` is kept but hidden — it's the
    engine that actually performs home/pan/zoom/save on the canvas — while a row
    of plain :class:`QToolButton`\\ s drives it, so the panel gets a lighter,
    native-feeling control strip instead of matplotlib's own toolbar.
    """

    def __init__(
        self,
        figure,
        title: str = "plot",
        parent: Optional[QWidget] = None,
        on_pick=None,
        recipe: Optional[PlotRecipe] = None,
    ) -> None:
        super().__init__(parent)
        self.recipe = recipe

        # Imported here (not at module load) so the UI package stays importable
        # without matplotlib, and to bind to whatever Qt binding is running.
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg,
            NavigationToolbar2QT,
        )

        _finish(figure)
        self.figure = figure
        self._title = title
        self._on_pick = on_pick
        # Let the figure re-flow its axes to fill the canvas on every resize,
        # so the plot grows with the panel instead of sitting small inside wide
        # default margins. Guarded: some figures (3D/colorbar) reject an engine.
        try:
            figure.set_layout_engine("tight")
        except Exception:  # noqa: BLE001 - keep the figure's own layout if so
            pass

        self.canvas = FigureCanvasQTAgg(figure)
        # The canvas defaults to a Preferred policy that clings to the figure's
        # 640×480 hint; make it greedily fill the tab.
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.setMinimumSize(200, 150)

        # Hidden matplotlib toolbar: not shown, only used for its home/back/
        # forward/pan/zoom/save behaviour, which our own buttons call.
        self._nav = NavigationToolbar2QT(self.canvas, self)
        self._nav.hide()
        # Route matplotlib's coordinate string (built from format_coord) to our
        # own readout label instead of the hidden toolbar's message widget.
        self._nav.set_message = self._set_coords

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # Toolbar under the figure, not over it. Above, it sits between the tab
        # that names the plot and the plot itself, so the eye crosses a row of
        # controls to reach the thing it opened; the figure should be the first
        # thing in the panel and the controls the last.
        #
        # The live controls, where there are any, go down the right-hand side
        # rather than along the bottom: the dock is wide and short, so a column
        # costs width the figure has to spare and a row costs height it does
        # not. The toolbar stays under both, as it is about the whole tab.
        self.controls = self._build_controls()
        self.controls_column: Optional[QScrollArea] = None
        # The figure on a sheet of its own, inset and edged, rather than a
        # white rectangle running into the tab bar and the buttons below it:
        # it is a page the window shows, as it will be on paper.
        self._paper = QFrame(self)
        self._paper.setObjectName("plotPaper")
        sheet = QVBoxLayout(self._paper)
        sheet.setContentsMargins(1, 1, 1, 1)
        sheet.addWidget(self.canvas)
        across = QHBoxLayout()
        across.setContentsMargins(_PAPER_INSET, _PAPER_INSET, 0, 0)
        across.setSpacing(0)
        across.addWidget(self._paper, 1)
        if self.controls is None:
            across.addSpacing(_PAPER_INSET)
        else:
            self.controls_column = self._controls_column(self.controls)
            across.addWidget(self.controls_column, 0)
        layout.addLayout(across, 1)
        layout.addWidget(self._build_toolbar())
        if self._on_pick is not None:
            self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
            # Connected *after* the toolbar's own cursor handler (built above),
            # so this one runs last and wins. Setting the cursor once at
            # construction does not: matplotlib resets it on every mouse move.
            self.canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)
            self.canvas.setToolTip("Click a peak to animate the mode that produced it")
        self.canvas.draw_idle()

    def _on_canvas_motion(self, event) -> None:
        """A pointing hand over the plot area, to advertise that it is clickable.

        Only inside the axes, and only when neither pan nor zoom is armed —
        those modes own both the drag and the cursor, and showing a hand while
        the click would pan instead would be a lie.
        """
        if str(self._nav.mode):
            return
        self.canvas.setCursor(
            Qt.PointingHandCursor if event.inaxes is not None else Qt.ArrowCursor
        )

    def _on_canvas_click(self, event) -> None:
        """Report the clicked x position to the tab's pick handler.

        Only plain clicks inside the axes count: while pan or zoom is armed the
        drag belongs to matplotlib, and a click that lands on the figure margin
        has no data coordinate to report.
        """
        if event.inaxes is None or event.xdata is None:
            return
        if str(self._nav.mode):  # '' when neither pan nor zoom is active
            return
        self._on_pick(float(event.xdata))

    def _build_toolbar(self) -> QWidget:
        """A slim row of icon buttons driving the hidden matplotlib toolbar."""
        bar = QWidget(self)
        row = QHBoxLayout(bar)
        # Inset like the figure, less the buttons' own padding, so the first
        # glyph lines up with the sheet's edge.
        row.setContentsMargins(_PAPER_INSET - 3, 4, _PAPER_INSET - 3, 4)
        row.setSpacing(2)

        self._btn_reset = self._tool_button(
            "plot-home.svg", "Reset", "Reset the view", self._nav.home
        )
        self._btn_back = self._tool_button(
            "plot-back.svg", "Back", "Back to previous view", self._nav.back
        )
        self._btn_forward = self._tool_button(
            "plot-forward.svg", "Forward", "Forward to next view", self._nav.forward
        )
        # Pan and zoom toggle matplotlib's interaction mode, so they're checkable
        # and mutually exclusive (the mode itself lives in the matplotlib toolbar).
        self._btn_pan = self._tool_button(
            "plot-pan.svg", "Pan", "Drag to pan", self._toggle_pan, checkable=True
        )
        self._btn_zoom = self._tool_button(
            "plot-zoom.svg", "Zoom", "Drag a box to zoom", self._toggle_zoom, checkable=True
        )

        for btn in (self._btn_reset, self._btn_back, self._btn_forward):
            row.addWidget(btn)
        row.addWidget(self._separator())
        row.addWidget(self._btn_pan)
        row.addWidget(self._btn_zoom)
        row.addStretch(1)

        # Live coordinate readout of the point under the cursor.
        self._coords = QLabel("", bar)
        self._coords.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._coords.setProperty("role", "plotCoords")
        row.addWidget(self._coords)

        row.addWidget(self._separator())
        # Only for a plot that knows how it was made and how to ask for new
        # options: everything else in this row acts on the figure as drawn,
        # while this one draws it again.
        if self.recipe is not None and self.recipe.edit is not None:
            row.addWidget(
                self._tool_button("plot-edit.svg", "Edit…",
                                  "Change this plot's settings and draw it again",
                                  self._edit_requested)
            )
        row.addWidget(
            self._tool_button("plot-save.svg", "Save…", "Save the plot to a file",
                              self._save_figure)
        )
        return bar

    def _build_controls(self):
        """The recipe's own controls, wired to redraw as they are turned.

        Debounced: a slider reports every pixel it passes under the hand, and a
        spectrum redrawn per pixel would be a spectrum nobody can drag. A short
        rest after the last movement is the difference between following the
        hand and fighting it. The timer belongs to the tab, so it cannot fire
        into a tab that has been closed.
        """
        if self.recipe is None or self.recipe.controls is None:
            return None
        widget = self.recipe.controls()
        if widget is None:
            return None
        widget.setParent(self)
        # The figure is the thing; the controls sit beside it and take no more
        # room than they need — which is their own minimum, asked rather than
        # chosen. Capped below it, a row is squeezed past the point where Qt
        # can lay it out and the range slider is drawn over its value boxes.
        widget.setMaximumWidth(max(_unfolded_width(widget), _CONTROLS_WIDTH))
        widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        widget.set_options(self.recipe.options)
        self._settle = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(_REDRAW_REST_MS)
        self._settle.timeout.connect(self._controls_settled)
        widget.changed.connect(self._settle.start)
        return widget

    def _controls_column(self, controls: QWidget) -> "QScrollArea":
        """The controls down the right-hand side, at one width, scrolled if short.

        Scrolled rather than stretching the window: a section unfolded beside a
        short plot made the controls taller than the window, and the window
        grew to fit them — a click on "Appearance" changed the figure's shape.
        One width, the most the controls may take, so a section unfolding or a
        redraw replacing the canvas cannot move the boundary with the figure.
        """
        holder = QWidget()
        column = QVBoxLayout(holder)
        # The side panels' margins: without them the boxes ran into the edge
        # of the window, and the first of them into the figure.
        left, _top, right, bottom = PANEL_MARGINS
        # No margin above: a section opens with space of its own, which puts
        # the first heading level with the top of the figure beside it.
        column.setContentsMargins(left, 0, right, bottom)
        column.addWidget(controls, 0)
        column.addStretch(1)      # the rows sit at the top, beside the axes
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(holder)
        self._column_extra = left + right
        if not scroll.style().styleHint(QStyle.SH_ScrollBar_Transient, None, scroll):
            self._column_extra += scroll.verticalScrollBar().sizeHint().width()  # one that takes room
        width = controls.maximumWidth() + self._column_extra
        _COLUMN["width"] = max(_COLUMN["width"], width)
        scroll.setFixedWidth(_COLUMN["width"])
        controls.setMaximumWidth(_COLUMN["width"] - self._column_extra)
        return scroll

    def _match_column(self) -> None:
        """Take the width every column now has, if another plot has widened it."""
        column = self.controls_column
        if column is not None and column.width() != _COLUMN["width"]:
            self.controls.setMaximumWidth(_COLUMN["width"] - self._column_extra)
            column.setFixedWidth(_COLUMN["width"])

    def showEvent(self, event) -> None:  # noqa: N802 - Qt's name
        self._match_column()
        super().showEvent(event)

    def _controls_settled(self) -> None:
        """Ask for the plot to be drawn again, as the controls now describe it."""
        panel = self._panel()
        if panel is not None and self.controls is not None:
            panel.redraw_requested.emit(self, dict(self.controls.options()))

    def _panel(self):
        """The PlotPanel this tab is in, whatever it is nested inside."""
        parent = self.parent()
        while parent is not None and not isinstance(parent, PlotPanel):
            parent = parent.parent()
        return parent

    def _edit_requested(self) -> None:
        """Ask whoever owns this tab to put new options to the user.

        Not done here: the dialogs belong to the window, as does the thread a
        slow plot is drawn on, and a panel that reached for either would have
        to know every kind of plot there is.
        """
        panel = self._panel()
        if panel is not None:
            panel.edit_requested.emit(self)

    def _separator(self) -> QFrame:
        line = QFrame(self)
        line.setObjectName("plotToolSeparator")
        line.setFixedSize(1, 16)
        return line

    def _tool_button(self, icon_name, text, tooltip, slot, checkable: bool = False) -> QToolButton:
        btn = QToolButton(self)
        btn.setToolTip(tooltip)
        btn.setAccessibleName(text.rstrip("…"))
        btn.setCheckable(checkable)
        btn.setAutoRaise(True)
        btn.setProperty("role", "plotTool")
        btn.setProperty("glyph", icon_name)
        btn.setIcon(self._icon(icon_name))
        btn.setIconSize(QSize(18, 18))
        btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
        btn.clicked.connect(slot)
        return btn

    @staticmethod
    def _icon(name):
        """One of the app's own glyphs, in the theme's text colour.

        Not matplotlib's: those are heavy filled shapes from another family,
        and the row under every plot was the one place in the app they showed.
        """
        from PySide6.QtWidgets import QApplication

        from crystalline.ui import theme

        return theme.monochrome_icon(name, theme.active_palette(QApplication.instance()).text)

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt's name
        super().changeEvent(event)
        if event.type() == QEvent.PaletteChange:
            # Drawn in the theme's colour, so drawn again when it changes: a
            # dark glyph left over from the light theme vanishes in the dark.
            for button in self.findChildren(QToolButton):
                glyph = button.property("glyph")
                if glyph:
                    button.setIcon(self._icon(glyph))

    def _set_coords(self, message: str) -> None:
        self._coords.setText(message or "")

    # ── saving ──────────────────────────────────────────────────────────
    def _save_figure(self) -> None:
        """Save the figure, letting the user pick format, DPI and transparency."""
        filetypes = self.canvas.get_supported_filetypes()  # {ext: description}
        default_ext = self.canvas.get_default_filetype()

        options = self._ask_save_options(filetypes, default_ext)
        if options is None:
            return
        ext, dpi, transparent = options

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save plot",
            f"{_slug(self._title)}.{ext}",
            f"{filetypes[ext]} (*.{ext})",
        )
        if not path:
            return
        try:
            self.figure.savefig(path, format=ext, dpi=dpi, transparent=transparent)
        except Exception as exc:  # noqa: BLE001 - surface any write/encode error
            QMessageBox.critical(self, "Save failed", f"Could not save the plot:\n{exc}")

    def _ask_save_options(self, filetypes: dict, default_ext: str):
        """Prompt for ``(ext, dpi, transparent)``; ``None`` if cancelled.

        DPI is greyed out for vector formats (they don't rasterise) and
        transparency for JPEG (no alpha channel), so the offered options always
        match the chosen format.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("Save plot")
        form = QFormLayout(dialog)

        fmt_box = QComboBox(dialog)
        for ext in sorted(filetypes):
            fmt_box.addItem(f"{ext.upper()} — {filetypes[ext]}", ext)
        idx = fmt_box.findData(default_ext)
        fmt_box.setCurrentIndex(idx if idx >= 0 else 0)
        form.addRow("Format:", fmt_box)

        dpi_box = QSpinBox(dialog)
        dpi_box.setRange(10, 2400)
        dpi_box.setSingleStep(50)
        dpi_box.setValue(300)
        dpi_box.setSuffix(" dpi")
        form.addRow("Resolution:", dpi_box)

        transparent = QCheckBox("Transparent background", dialog)
        form.addRow("", transparent)

        def sync_enabled() -> None:
            ext = fmt_box.currentData()
            dpi_box.setEnabled(ext not in _VECTOR_FORMATS)
            transparent.setEnabled(ext not in _OPAQUE_FORMATS)

        fmt_box.currentIndexChanged.connect(sync_enabled)
        sync_enabled()

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() != QDialog.Accepted:
            return None
        ext = fmt_box.currentData()
        return ext, dpi_box.value(), transparent.isChecked() and transparent.isEnabled()

    def _toggle_pan(self) -> None:
        self._nav.pan()
        self._sync_mode_buttons()

    def _toggle_zoom(self) -> None:
        self._nav.zoom()
        self._sync_mode_buttons()

    def _sync_mode_buttons(self) -> None:
        """Reflect matplotlib's active mode on the pan/zoom buttons.

        Arming pan or zoom also hands the cursor back to matplotlib: a hand left
        over from before would promise a mode selection where the click now pans.
        """
        mode = str(self._nav.mode)  # '' / 'pan/zoom' / 'zoom rect'
        self._btn_pan.setChecked(mode == "pan/zoom")
        self._btn_zoom.setChecked(mode == "zoom rect")
        if mode and self._on_pick is not None:
            self.canvas.unsetCursor()

    def set_figure(self, figure, title: Optional[str] = None) -> None:
        """Show ``figure`` here instead, and let the old one go.

        A canvas belongs to one figure, so a changed plot means a new canvas in
        the same tab — which is the point: the plot the user is looking at
        changes, rather than a second tab appearing beside it with one setting
        different. Everything bound to the old canvas is rebuilt with it: the
        hidden navigation toolbar that performs pan and zoom, the coordinate
        readout it reports through, and the click handler that ties a peak back
        to its mode.
        """
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg,
            NavigationToolbar2QT,
        )

        old_canvas, old_figure, old_nav = self.canvas, self.figure, self._nav
        _finish(figure)
        self.figure = figure
        if title is not None:
            self._title = title
        try:
            figure.set_layout_engine("tight")
        except Exception:  # noqa: BLE001 - keep the figure's own layout if so
            pass

        self.canvas = FigureCanvasQTAgg(figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.setMinimumSize(200, 150)
        self._nav = NavigationToolbar2QT(self.canvas, self)
        self._nav.hide()
        self._nav.set_message = self._set_coords
        if self._on_pick is not None:
            self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
            self.canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)
            self.canvas.setToolTip("Click a peak to animate the mode that produced it")

        self._paper.layout().replaceWidget(old_canvas, self.canvas)
        self._release(old_canvas, old_figure)
        # The old toolbar goes too. It is a child of this tab, so nothing else
        # would ever take it away, and it holds the old canvas, which holds the
        # old figure: every redraw from the controls beside a plot left one
        # behind, about 8 MB each, for as long as the tab was open.
        old_nav.setParent(None)
        old_nav.deleteLater()
        # Pan and zoom belong to the canvas that has gone; the buttons that
        # showed them pressed are now about a canvas in its resting state.
        self._sync_mode_buttons()
        self.canvas.draw_idle()

    @staticmethod
    def _release(canvas, figure) -> None:
        """Take a canvas out of the tree and close its figure.

        A canvas draws lazily: a resize or a change queues the draw on a
        zero-timer, and that timer fires on the next turn of the event loop,
        by which time this canvas is gone — "Internal C++ object already
        deleted", raised inside the backend, nowhere near the close that caused
        it. The queued draw is called off first; matplotlib's handler asks
        whether one is still wanted before it touches the widget.
        """
        import matplotlib.pyplot as plt

        try:
            canvas._draw_pending = False  # noqa: SLF001 - what the backend checks
            canvas.setParent(None)
            canvas.close()
            # Deleted, not only closed: a closed widget with no parent is
            # still a widget, and this one holds its figure.
            canvas.deleteLater()
        except Exception:  # noqa: BLE001 - already torn down; the close below still runs
            pass
        plt.close(figure)

    def close_figure(self) -> None:
        """Release the matplotlib figure when this tab goes away.

        The canvas is taken out of the widget tree *first*. matplotlib draws
        lazily — a resize or a data change queues an idle draw — and that draw
        fires on the next turn of the event loop, by which time the tab's C++
        side may be gone. It then raises ``Internal C++ object already deleted``
        from inside the backend, nowhere near the close that caused it.
        """
        self._release(self.canvas, self.figure)


class PlotPanel(QWidget):
    """A tabbed container for property plots, one closable tab per figure.

    Add a plot with :meth:`add_figure`; the panel takes care of building the
    canvas/toolbar, giving the tab a close button, and freeing the figure when
    a tab is closed. When no plots are open it shows a short hint instead.
    """

    #: A tab whose Edit button was pressed. The window answers it: it owns the
    #: dialogs, and the thread a slow plot is drawn on.
    edit_requested = Signal(object)
    #: ``(tab, options)``: its controls have come to rest somewhere new, and the
    #: plot should be drawn again that way.
    redraw_requested = Signal(object, dict)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._tabs = QTabWidget(self)
        # Closable tabs, so the close buttons must travel with a swipe too.
        self._tabs.setTabBar(SlidingTabBar(self._tabs))
        self._tabs.setDocumentMode(True)
        self._tabs.setTabsClosable(True)
        self._tabs.setMovable(True)
        self._tabs.tabCloseRequested.connect(self._close_tab)

        self._placeholder = QLabel(
            "No plots yet — build one from the Plot menu.", self
        )
        self._placeholder.setEnabled(False)
        self._placeholder.setMargin(16)

        # Swap between the hint (when empty) and the tabs (when populated).
        self._stack = QStackedLayout(self)
        self._stack.setContentsMargins(0, 0, 0, 0)
        self._stack.addWidget(self._placeholder)
        self._stack.addWidget(self._tabs)
        self._show_placeholder()

    # ── public API ──────────────────────────────────────────────────────
    def add_figure(self, figure, title: str, on_pick=None,
                   recipe: Optional[PlotRecipe] = None) -> None:
        """Add ``figure`` as a new tab titled ``title`` and bring it to front.

        ``on_pick`` — when given, a callable receiving the x data coordinate of
        each plain click inside the axes. Used to tie a spectrum's peaks back to
        the modes that produced them.

        ``recipe`` — how this plot was made, so that it can be made again with
        other options (see :class:`PlotRecipe`). A tab without one can only be
        looked at, saved and closed, which is what every tab could do before.
        """
        tab = _PlotTab(figure, title, self, on_pick=on_pick, recipe=recipe)
        index = self._tabs.addTab(tab, title)
        self._tabs.setCurrentIndex(index)
        self._stack.setCurrentWidget(self._tabs)

    def controls_width(self) -> int:
        """How much width the plots here want for their own controls, if any.

        Asked when the dock is first floated, so that the 4:3 it opens at is
        the figure's and not the whole window's: a column beside the plot came
        out of the plot's width, which is what made it letterboxed.
        """
        widths = [tab.controls_column.width()
                  for tab in (self._tabs.widget(i) for i in range(self._tabs.count()))
                  if tab is not None and tab.controls_column is not None]
        return max(widths) if widths else 0

    def index_of(self, tab) -> int:
        """Where ``tab`` is now — tabs are movable, so this is asked, not kept."""
        return self._tabs.indexOf(tab)

    def current_tab(self):
        """The plot tab on show, or ``None`` when there are no plots."""
        return self._tabs.currentWidget() if self._tabs.count() else None

    def widget_at(self, index: int):
        """The tab at ``index`` — its figure, its canvas and its recipe."""
        return self._tabs.widget(index)

    def tab_text(self, index: int) -> str:
        """What the tab at ``index`` is called."""
        return self._tabs.tabText(index)

    def replace_figure(self, index: int, figure, title: Optional[str] = None) -> None:
        """Put ``figure`` in the tab at ``index``, in place of the one there.

        What makes a plot changeable rather than merely re-buildable: the tab
        keeps its place in the bar, and the figure in front of the user becomes
        the new one.
        """
        tab = self._tabs.widget(index)
        if tab is None:
            return
        tab.set_figure(figure, title)
        if title is not None:
            self._tabs.setTabText(index, title)

    def count(self) -> int:
        """How many plots are open."""
        return self._tabs.count()

    def clear(self) -> None:
        """Close every plot, releasing its figure.

        Called when a new file is loaded. Nothing used to close them, so the
        figures of every file opened in a session stayed live — each one a
        matplotlib ``Figure`` and a Qt canvas that Qt still repaints — and the
        window grew heavier with every plot ever drawn.

        Correctness, not only weight: a spectrum carries a pick handler wired to
        *its* file's modes, so a plot left behind from the previous structure
        would select modes in the new one.
        """
        while self._tabs.count():
            self._close_tab(0)

    # ── internals ───────────────────────────────────────────────────────
    def _close_tab(self, index: int) -> None:
        widget = self._tabs.widget(index)
        self._tabs.removeTab(index)
        if isinstance(widget, _PlotTab):
            widget.close_figure()
        widget.deleteLater()
        if self._tabs.count() == 0:
            self._show_placeholder()

    def _show_placeholder(self) -> None:
        self._stack.setCurrentWidget(self._placeholder)


def _slug(text: str) -> str:
    """A filesystem-friendly default filename stem from a plot title."""
    cleaned = "".join(c if c.isalnum() else "_" for c in text).strip("_")
    return cleaned.lower() or "plot"


__all__ = ["LiveRedraws", "PlotPanel", "PlotRecipe"]
