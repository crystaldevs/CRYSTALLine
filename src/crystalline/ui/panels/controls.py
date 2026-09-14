"""Shared building blocks for the settings panels.

One grid, one set of metrics, one kind of row — so a slider in the Phonons panel
sits exactly where a slider in the Display panel does. They were built
separately, with ``QFormLayout`` and right-aligned labels, and the result was
that no two panels (and inside a panel, no two groups) put their controls at the
same place.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QColorDialog,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStyledItemDelegate,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


# One grid for the whole panel. Rows used to be QFormLayout rows with
# right-aligned labels, so a control started wherever its own label happened to
# end — "Show bonds", "Radius (Å)" and "Show hydrogen bonds" each began at a
# different x, and the ragged edge was the loudest thing on the panel. A fixed
# label column puts every control on one line down the whole panel.
_LABEL_COLUMN = 112
_VALUE_WIDTH = 66      # spin boxes, so a column of numbers is a column
_ROW_SPACING = 7
_SECTION_SPACING = 16


class _Section:
    """A collapsible block of settings: a header you can click, then rows.

    Not a ``QGroupBox``. Six bordered cards stacked down one panel read as a form
    rather than a tool, and the nested borders eat the width that makes the rows
    legible. A small header and honest whitespace group just as well.

    Collapsible because the Display panel has eleven of these: everything the
    renderer can be told, all at once, is a wall. Folding the ones a given
    session does not need turns it into a short list of headings that can be
    scanned — and the state is per-section, so what someone opens stays open.
    """

    def __init__(self, parent: QVBoxLayout, title: str, collapsed: bool = False,
                 switch: Optional[str] = None) -> None:
        """``switch``, when given, puts an on/off checkbox at the right of the
        header, with that text as its tooltip.

        For a section that is about one thing in the view — the bonds, the cell —
        whether it is shown is the first thing anyone wants and the only thing
        most people change. Inside the section it was a click to unfold away,
        and with every section folded, out of sight entirely.
        """
        self._parent = parent
        self.switch: Optional[QCheckBox] = None
        self.header = QToolButton()
        # "&" in a button's text is a mnemonic, so "Cell & axes" would draw as
        # "Cell _axes" — the same trap the group-box title fell into. Doubling it
        # is the escape; the section keeps the readable title it was given.
        self.header.setText(title.upper().replace("&", "&&"))
        self.header.setCheckable(True)
        self.header.setChecked(not collapsed)
        self.header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.header.setArrowType(Qt.DownArrow if not collapsed else Qt.RightArrow)
        self.header.setProperty("role", "section")
        self.header.setCursor(Qt.PointingHandCursor)
        self.header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.header.toggled.connect(self._on_toggled)

        parent.addSpacing(_SECTION_SPACING)
        if switch is None:
            parent.addWidget(self.header)
        else:
            self.switch = QCheckBox()
            self.switch.setToolTip(switch)
            self.switch.setAccessibleName(switch)
            self.switch.setCursor(Qt.PointingHandCursor)
            row = QWidget()
            line = QHBoxLayout(row)
            line.setContentsMargins(0, 0, 0, 0)
            line.setSpacing(6)
            line.addWidget(self.header, 1)
            line.addWidget(self.switch, 0, Qt.AlignRight | Qt.AlignVCenter)
            parent.addWidget(row)

        # The rows live in a container so the whole section can be hidden at
        # once — a layout cannot be hidden, only the widgets in it.
        self.body = QWidget()
        body_layout = QVBoxLayout(self.body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        self.grid = QGridLayout()
        self.grid.setContentsMargins(0, 4, 0, 0)
        self.grid.setHorizontalSpacing(10)
        self.grid.setVerticalSpacing(_ROW_SPACING)
        self.grid.setColumnMinimumWidth(0, _LABEL_COLUMN)
        self.grid.setColumnStretch(0, 0)
        self.grid.setColumnStretch(1, 1)
        body_layout.addLayout(self.grid)
        self._body_layout = body_layout
        parent.addWidget(self.body)
        self.body.setVisible(not collapsed)

        self._row = 0
        self._rows: dict = {}
        # Keep this object alive for as long as its header exists. A caller
        # builds sections and drops them — the widgets survive because Qt owns
        # them, but the Python Section would be collected, taking the ``toggled``
        # handler with it, and the headers would stop folding the moment
        # construction finished.
        self.header._section = self

    def _on_toggled(self, open_: bool) -> None:
        self.header.setArrowType(Qt.DownArrow if open_ else Qt.RightArrow)
        self.body.setVisible(open_)

    def add(self, label: str, widget: QWidget) -> None:
        """A labelled row: the name in the fixed column, the control in the rest."""
        caption = QLabel(label)
        self.grid.addWidget(caption, self._row, 0, Qt.AlignLeft | Qt.AlignVCenter)
        self.grid.addWidget(widget, self._row, 1)
        self._rows[label] = (caption, widget)
        self._row += 1

    def row_widgets(self, label: str) -> tuple:
        """The ``(label, control)`` pair of a row, for showing or hiding it.

        A row that does not apply — the Gaussian width of a pure Lorentzian —
        should go away entirely rather than sit there greyed, and hiding it means
        hiding both halves.
        """
        return self._rows[label]

    def add_wide(self, widget: QWidget) -> None:
        """A row with no separate label — a checkbox carries its own text, so it
        spans both columns and every tick lines up at the panel's left edge."""
        self.grid.addWidget(widget, self._row, 0, 1, 2)
        self._row += 1

    def append(self, widget: QWidget) -> None:
        """Add a widget *below* the section's rows.

        The grid is filled top to bottom as rows are added, so anything that has
        to stay at the foot — a hint, a reset button — cannot live in it: a later
        row would be placed under it, not after it.
        """
        self._body_layout.addSpacing(6)
        self._body_layout.addWidget(widget)

    def set_enabled(self, enabled: bool) -> None:
        """Grey the whole section — for settings a given file has no data for."""
        for index in range(self.grid.count()):
            widget = self.grid.itemAt(index).widget()
            if widget is not None:
                widget.setEnabled(enabled)


class ColourButton(QPushButton):
    """A colour swatch that opens the picker — the Display panel's disc, reusable.

    Dialogs remember it like any other setting: :mod:`dialog_state` stores
    whatever :meth:`colour` returns and hands it back to :meth:`setColour`.
    """

    colourChanged = Signal(str)

    def __init__(self, colour: str, title: str = "Choose colour",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._colour = QColor(colour).name()
        self._title = title
        self.clicked.connect(self._choose)
        self._paint()

    def colour(self) -> str:
        return self._colour

    def setColour(self, colour: str) -> None:  # noqa: N802 - Qt spelling
        chosen = QColor(colour)
        if not chosen.isValid() or chosen.name() == self._colour:
            return
        self._colour = chosen.name()
        self._paint()
        self.colourChanged.emit(self._colour)

    def _choose(self) -> None:
        chosen = QColorDialog.getColor(QColor(self._colour), self, self._title)
        if chosen.isValid():
            self.setColour(chosen.name())

    def _paint(self) -> None:
        # One look for every swatch in the app, painted where it was designed.
        from crystalline.ui.panels.display_settings import DisplayPanel

        DisplayPanel._paint_swatch(self, self._colour)


class RichTextDelegate(QStyledItemDelegate):
    """Draw item text as rich text, so that a subscript is drawn as one.

    Qt's item views take plain strings. Most of the symbols this app shows can
    be written with the Unicode subscripts — C₂, S₄, σₕ — but σd cannot, since
    Unicode has no subscript d, and a row reading "σd" beside a row reading
    "σₕ" looks like a mistake. The rows are given HTML instead, and this draws
    it: the style paints the row itself (selection, checkbox, focus) and only
    the text is replaced.
    """

    def __init__(self, parent=None, markup=None) -> None:
        """``markup`` turns the item's plain text into HTML.

        The items keep plain text — that is what a tooltip, a copy and a test
        see — and the markup is applied here, at the moment of drawing.
        """
        super().__init__(parent)
        self._markup = markup or (lambda text: text)

    def paint(self, painter, option, index) -> None:
        from PySide6.QtGui import QAbstractTextDocumentLayout, QFontMetrics
        from PySide6.QtWidgets import QApplication, QStyle, QStyleOptionViewItem

        styled = QStyleOptionViewItem(option)
        self.initStyleOption(styled, index)
        style = styled.widget.style() if styled.widget else QApplication.style()
        area = style.subElementRect(QStyle.SE_ItemViewItemText, styled, styled.widget)

        # Elide rather than clip: a row too narrow for its text ends in an
        # ellipsis, as it would in any other list.
        text = styled.text
        metrics = QFontMetrics(styled.font)
        if metrics.horizontalAdvance(text) > area.width():
            text = metrics.elidedText(text, Qt.ElideRight, area.width())

        document = self._document(styled.font, text)
        styled.text = ""                      # the style draws everything but this
        style.drawControl(QStyle.CE_ItemViewItem, styled, painter, styled.widget)

        context = QAbstractTextDocumentLayout.PaintContext()
        if styled.state & QStyle.State_Selected:
            context.palette.setColor(
                context.palette.ColorRole.Text,
                styled.palette.color(styled.palette.ColorGroup.Normal,
                                     styled.palette.ColorRole.HighlightedText),
            )
        painter.save()
        painter.setClipRect(area)
        # Centre the line in its row rather than hanging it from the top: a
        # subscript makes the document a little taller than a plain line, and
        # the excess would otherwise fall out of the bottom of the row.
        spare = max(0.0, (area.height() - document.size().height()) / 2)
        painter.translate(area.left(), area.top() + spare)
        document.documentLayout().draw(painter, context)
        painter.restore()

    def sizeHint(self, option, index):  # noqa: N802 - Qt's name
        from PySide6.QtCore import QSize
        from PySide6.QtWidgets import QStyleOptionViewItem

        styled = QStyleOptionViewItem(option)
        self.initStyleOption(styled, index)
        document = self._document(styled.font, styled.text)
        size = super().sizeHint(option, index)
        # The row must hold the marked-up line, which a subscript makes taller
        # than the plain one the style measured.
        return QSize(
            int(document.idealWidth()) + 8,
            max(size.height(), int(document.size().height()) + 2),
        )

    def _document(self, font, text):
        """The row's text as a laid-out document, with no margin of its own.

        A QTextDocument keeps a 4 px margin unless told otherwise, which would
        push the line down inside the row and let its own bottom be cut off.
        """
        from PySide6.QtGui import QTextDocument

        document = QTextDocument()
        document.setDocumentMargin(0)
        document.setDefaultFont(font)
        document.setHtml(self._markup(text))
        return document


def _left(widget: QWidget) -> QWidget:
    """Pin a fixed-width control to the left of its column.

    A grid stretches its cell, so a spin box or a swatch dropped straight in
    would grow to the panel's width and the column of values would stop being a
    column.
    """
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(widget, 0)
    row.addStretch(1)
    return holder




def slider_row(
    section: "_Section",
    label: str,
    value: float,
    low: float,
    high: float,
    step: float,
    decimals: int = 2,
    suffix: str = "",
    logarithmic: bool = False,
    on_change: Optional[Callable[[float], None]] = None,
) -> QDoubleSpinBox:
    """A labelled slider bound to a spin box, on the shared grid.

    The slider is the coarse control and the box the exact one; they track each
    other through a guard, since each drives the other and a naive pair would
    ring. Returns the box, which is the value's home.

    ``logarithmic`` spaces the slider by ratio rather than by difference, for a
    value that is read as a multiple — a playback speed, a displacement
    amplitude. On a linear scale a 0.1–10× speed puts 1× nine percent along, so
    the entire useful range is crushed into the first centimetre of travel and
    the default sits against the left stop. Geometrically, 1× is the midpoint.
    """
    if logarithmic and (low <= 0.0 or high <= 0.0):
        raise ValueError("a logarithmic slider needs strictly positive bounds")
    box = QDoubleSpinBox()
    box.setRange(low, high)
    box.setSingleStep(step)
    box.setDecimals(decimals)
    box.setSuffix(suffix)
    box.setValue(value)
    box.setFixedWidth(_VALUE_WIDTH)

    slider = QSlider(Qt.Horizontal)
    slider.setRange(0, 1000)

    if logarithmic:
        log_low, log_high = math.log(low), math.log(high)

        def to_slider(v: float) -> int:
            v = min(max(v, low), high)
            return int(round((math.log(v) - log_low) / (log_high - log_low) * 1000))

        def from_slider(s: int) -> float:
            return math.exp(log_low + (s / 1000.0) * (log_high - log_low))
    else:

        def to_slider(v: float) -> int:
            return int(round((v - low) / (high - low) * 1000))

        def from_slider(s: int) -> float:
            return low + (s / 1000.0) * (high - low)

    slider.setValue(to_slider(value))
    guard = {"lock": False}

    def on_box(v: float) -> None:
        if guard["lock"]:
            return
        guard["lock"] = True
        slider.setValue(to_slider(v))
        guard["lock"] = False
        if on_change is not None:
            on_change(v)

    def on_slider(s: int) -> None:
        if guard["lock"]:
            return
        guard["lock"] = True
        box.setValue(from_slider(s))
        guard["lock"] = False
        if on_change is not None:
            on_change(box.value())

    box.valueChanged.connect(on_box)
    slider.valueChanged.connect(on_slider)

    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    layout.addWidget(slider, 1)
    layout.addWidget(box, 0)
    section.add(label, row)
    return box


def range_row(
    section: "_Section",
    label: str,
    low: float,
    high: float,
    minimum: float,
    maximum: float,
    decimals: int = 0,
    step: float = 100.0,
    on_change: Optional[Callable[[float, float], None]] = None,
):
    """A two-handled span, with a spin box at each end.

    A range spelled as a separate "from" and "to" says nothing about how much of
    the whole it covers, and lets the two be put the wrong way round. One track
    shows the span; the boxes are there for an exact edge.

    Returns ``(slider, low box, high box)``.
    """
    from crystalline.ui.widgets import RangeSlider

    slider = RangeSlider(minimum, maximum)
    slider.setValues(low, high)

    boxes = []
    for value in (low, high):
        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setDecimals(decimals)
        box.setSingleStep(step)
        box.setValue(value)
        box.setFixedWidth(_VALUE_WIDTH)
        boxes.append(box)
    low_box, high_box = boxes

    guard = {"lock": False}

    def from_slider(a: float, b: float) -> None:
        if guard["lock"]:
            return
        guard["lock"] = True
        low_box.setValue(a)
        high_box.setValue(b)
        guard["lock"] = False
        if on_change is not None:
            on_change(a, b)

    def from_boxes() -> None:
        if guard["lock"]:
            return
        guard["lock"] = True
        slider.setValues(low_box.value(), high_box.value())
        guard["lock"] = False
        if on_change is not None:
            on_change(*slider.values())

    slider.valuesChanged.connect(from_slider)
    low_box.valueChanged.connect(lambda _v: from_boxes())
    high_box.valueChanged.connect(lambda _v: from_boxes())

    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    layout.addWidget(low_box, 0)
    layout.addWidget(slider, 1)
    layout.addWidget(high_box, 0)
    section.add(label, row)
    return slider, low_box, high_box


Section = _Section
left = _left
LABEL_COLUMN = _LABEL_COLUMN
VALUE_WIDTH = _VALUE_WIDTH

__all__ = [
    "LABEL_COLUMN",
    "Section",
    "VALUE_WIDTH",
    "left",
    "range_row",
    "slider_row",
]
