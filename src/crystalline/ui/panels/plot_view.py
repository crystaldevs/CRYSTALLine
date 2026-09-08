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

from typing import Optional

from PySide6.QtCore import QSize, Qt
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
    QSpinBox,
    QStackedLayout,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


# Vector formats ignore raster DPI; JPEG has no alpha channel. Used to grey out
# the controls that don't apply to the chosen format.
_VECTOR_FORMATS = frozenset({"svg", "pdf", "eps", "ps"})
_OPAQUE_FORMATS = frozenset({"jpg", "jpeg"})


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
    ) -> None:
        super().__init__(parent)

        # Imported here (not at module load) so the UI package stays importable
        # without matplotlib, and to bind to whatever Qt binding is running.
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg,
            NavigationToolbar2QT,
        )

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
        layout.addWidget(self.canvas, 1)  # stretch: the canvas takes the space
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
        row.setContentsMargins(4, 2, 4, 2)
        row.setSpacing(2)

        self._btn_reset = self._tool_button(
            "home_large.png", "Reset", "Reset the view", self._nav.home
        )
        self._btn_back = self._tool_button(
            "back_large.png", "Back", "Back to previous view", self._nav.back
        )
        self._btn_forward = self._tool_button(
            "forward_large.png", "Forward", "Forward to next view", self._nav.forward
        )
        # Pan and zoom toggle matplotlib's interaction mode, so they're checkable
        # and mutually exclusive (the mode itself lives in the matplotlib toolbar).
        self._btn_pan = self._tool_button(
            "move_large.png", "Pan", "Drag to pan", self._toggle_pan, checkable=True
        )
        self._btn_zoom = self._tool_button(
            "zoom_to_rect_large.png", "Zoom", "Drag a box to zoom", self._toggle_zoom, checkable=True
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
        self._coords.setStyleSheet("color: palette(mid);")
        row.addWidget(self._coords)

        row.addWidget(self._separator())
        row.addWidget(
            self._tool_button("filesave_large.png", "Save…", "Save the plot to a file",
                              self._save_figure)
        )
        return bar

    def _separator(self) -> QFrame:
        line = QFrame(self)
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        return line

    def _tool_button(self, icon_name, text, tooltip, slot, checkable: bool = False) -> QToolButton:
        btn = QToolButton(self)
        btn.setToolTip(tooltip)
        btn.setCheckable(checkable)
        btn.setAutoRaise(True)
        icon = self._icon(icon_name)
        if icon is not None:
            btn.setIcon(icon)
            btn.setIconSize(QSize(18, 18))
            btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
        else:  # matplotlib icons unavailable — fall back to a text button
            btn.setText(text)
            btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        btn.clicked.connect(slot)
        return btn

    def _icon(self, name):
        """A themed matplotlib toolbar icon, or ``None`` if unavailable."""
        try:
            return self._nav._icon(name)  # handles light/dark theming for us
        except Exception:  # noqa: BLE001 - private API; degrade to text buttons
            return None

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

    def close_figure(self) -> None:
        """Release the matplotlib figure when this tab goes away.

        The canvas is taken out of the widget tree *first*. matplotlib draws
        lazily — a resize or a data change queues an idle draw — and that draw
        fires on the next turn of the event loop, by which time the tab's C++
        side may be gone. It then raises ``Internal C++ object already deleted``
        from inside the backend, nowhere near the close that caused it.
        """
        import matplotlib.pyplot as plt

        try:
            self.canvas.setParent(None)
            self.canvas.close()
        except Exception:  # noqa: BLE001 - already torn down; the close below still runs
            pass
        plt.close(self.figure)


class PlotPanel(QWidget):
    """A tabbed container for property plots, one closable tab per figure.

    Add a plot with :meth:`add_figure`; the panel takes care of building the
    canvas/toolbar, giving the tab a close button, and freeing the figure when
    a tab is closed. When no plots are open it shows a short hint instead.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._tabs = QTabWidget(self)
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
    def add_figure(self, figure, title: str, on_pick=None) -> None:
        """Add ``figure`` as a new tab titled ``title`` and bring it to front.

        ``on_pick`` — when given, a callable receiving the x data coordinate of
        each plain click inside the axes. Used to tie a spectrum's peaks back to
        the modes that produced them.
        """
        tab = _PlotTab(figure, title, self, on_pick=on_pick)
        index = self._tabs.addTab(tab, title)
        self._tabs.setCurrentIndex(index)
        self._stack.setCurrentWidget(self._tabs)

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


__all__ = ["PlotPanel"]
