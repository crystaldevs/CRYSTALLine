"""What an empty tab shows: a way to open a file, and the files opened lately.

The window opens on an empty tab, and an empty 3D view is a dark rectangle
with nothing to say what to do with it. This sits over the view instead, for as
long as the tab has nothing in it, and goes the moment it has: a file opened, or
atoms built.

An overlay, like the drop hint and the busy scrim (see
:class:`~crystalline.ui.widgets.overlay.ParentOverlay`), rather than a page put
in the view's place: the view stays where it is, at the size it will have, so a
structure is framed in it exactly as it would be without this.
"""

from __future__ import annotations

import os
from typing import Iterable, List

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QPainter
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QLabel, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

from crystalline.ui.safety import guard
from crystalline.ui.widgets.overlay import ParentOverlay

DOCS_URL = "https://crystaldevs.github.io/CRYSTALLine/"
RELEASE_NOTES_URL = DOCS_URL + "release-notes.html"
RECENT_SHOWN = 8

_STYLE = """
QWidget#welcomeBody { background: transparent; }
QLabel#welcomeTitle { font-size: 30px; font-weight: 600; }
QLabel[role="welcomeMuted"] { color: palette(placeholder-text); }
QLabel[role="welcomeHeading"] {
    color: palette(placeholder-text); font-size: 12px; font-weight: 600;
    letter-spacing: 1.5px;
}
QPushButton#welcomeOpen {
    border: 1.5px dashed palette(mid); border-radius: 14px;
    background: palette(base); padding: 28px 24px; text-align: left;
}
QPushButton#welcomeOpen:hover, QPushButton#welcomeOpen:focus {
    border-color: palette(highlight);
}
QPushButton[role="welcomeRecent"] {
    border: none; border-radius: 8px; background: transparent;
    padding: 6px 10px; text-align: left;
}
QPushButton[role="welcomeRecent"]:hover, QPushButton[role="welcomeRecent"]:focus {
    background: palette(alternate-base);
}
QPushButton[role="welcomeLink"] {
    border: none; background: transparent; padding: 2px 0;
    color: palette(link); text-align: left;
}
QPushButton[role="welcomeLink"]:hover { text-decoration: underline; }
"""


def _label(text: str = "", role: str = "", parent=None) -> QLabel:
    label = QLabel(text, parent)
    if role:
        label.setProperty("role", role)
    # Clicks go to the button the label sits in.
    label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
    return label


def folder_label(path: str) -> str:
    """Where a file is, with the home folder written as ``~``."""
    folder = os.path.dirname(path)
    home = os.path.expanduser("~")
    if folder == home or folder.startswith(home + os.sep):
        folder = "~" + folder[len(home):]
    return folder


class WelcomeOverlay(ParentOverlay):
    """The empty tab's page: open a file, or one of the last few."""

    #: Open… was pressed.
    open_requested = Signal()
    #: A recent file was chosen.
    path_chosen = Signal(str)
    #: "Clear" under the recent files.
    clear_requested = Signal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setStyleSheet(_STYLE)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(32, 32, 32, 32)
        outer.addStretch(2)

        body = QWidget(self)
        body.setObjectName("welcomeBody")
        body.setMaximumWidth(860)
        body.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        column = QVBoxLayout(body)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(28)

        header = QHBoxLayout()
        header.setSpacing(12)
        title = QLabel("CRYSTALLine", body)
        title.setObjectName("welcomeTitle")
        header.addWidget(title, 0, Qt.AlignBaseline)
        from crystalline import __version__

        self.version = _label(__version__, "welcomeMuted", body)
        header.addWidget(self.version, 0, Qt.AlignBaseline)
        header.addStretch(1)
        column.addLayout(header)

        grid = QGridLayout()
        grid.setHorizontalSpacing(40)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(0, 5)
        grid.setColumnStretch(1, 6)

        grid.addWidget(_label("START", "welcomeHeading", body), 0, 0)
        self.open_button = QPushButton(body)
        self.open_button.setObjectName("welcomeOpen")
        self.open_button.setCursor(Qt.PointingHandCursor)
        self.open_button.setAccessibleName("Open a file")
        self.open_button.clicked.connect(self.open_requested)
        inside = QVBoxLayout(self.open_button)
        inside.setContentsMargins(24, 24, 24, 24)
        inside.setSpacing(6)
        inside.addStretch(1)
        first = _label("Open a file…")
        font = first.font()
        font.setPointSizeF(font.pointSizeF() * 1.25)
        font.setWeight(font.Weight.DemiBold)
        first.setFont(font)
        inside.addWidget(first)
        shortcut = QKeySequence(QKeySequence.Open).toString(QKeySequence.NativeText)
        inside.addWidget(_label(f"or drop it here · {shortcut}", "welcomeMuted"))
        inside.addStretch(1)
        self.open_button.setMinimumHeight(150)
        grid.addWidget(self.open_button, 1, 0, Qt.AlignTop)

        self.recent_heading = _label("RECENT", "welcomeHeading", body)
        grid.addWidget(self.recent_heading, 0, 1)
        self.recent_box = QWidget(body)
        self._recent_layout = QVBoxLayout(self.recent_box)
        self._recent_layout.setContentsMargins(0, 0, 0, 0)
        self._recent_layout.setSpacing(2)
        grid.addWidget(self.recent_box, 1, 1, Qt.AlignTop)
        column.addLayout(grid)

        links = QHBoxLayout()
        links.setSpacing(24)
        for text, url in (("Documentation", DOCS_URL), ("Release notes", RELEASE_NOTES_URL)):
            link = QPushButton(text, body)
            link.setProperty("role", "welcomeLink")
            link.setCursor(Qt.PointingHandCursor)
            link.clicked.connect(lambda _checked=False, u=url: self._open_url(u))
            links.addWidget(link)
        links.addStretch(1)
        column.addLayout(links)

        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(body, 100)
        row.addStretch(1)
        outer.addLayout(row)
        outer.addStretch(3)
        self._recent: List[str] = []
        self.set_recent([])
        self.recent_heading.setVisible(False)
        self.recent_box.setVisible(False)

    # ── content ─────────────────────────────────────────────────────────
    def set_recent(self, paths: Iterable[str]) -> None:
        """Show these files, newest first; the column goes if there are none."""
        paths = list(paths)[:RECENT_SHOWN]
        if paths == self._recent and self._recent_layout.count():
            return               # asked on every refresh of the window; same list
        self._recent = paths
        while self._recent_layout.count():
            item = self._recent_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # Hidden now, deleted later: deleteLater waits for the event
                # loop, and until then the old rows would draw on under the new ones.
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
        for path in self._recent:
            self._recent_layout.addWidget(self._recent_row(path))
        if self._recent:
            clear = QPushButton("Clear", self.recent_box)
            clear.setProperty("role", "welcomeLink")
            clear.setCursor(Qt.PointingHandCursor)
            clear.clicked.connect(self.clear_requested)
            self._recent_layout.addSpacing(6)
            self._recent_layout.addWidget(clear, 0, Qt.AlignLeft)
        self.recent_heading.setVisible(bool(self._recent))
        self.recent_box.setVisible(bool(self._recent))

    def recent(self) -> List[str]:
        return list(self._recent)

    def _recent_row(self, path: str) -> QPushButton:
        row = QPushButton(self.recent_box)
        row.setProperty("role", "welcomeRecent")
        row.setCursor(Qt.PointingHandCursor)
        row.setToolTip(path)
        row.setAccessibleName(f"Open {os.path.basename(path)}")
        row.clicked.connect(lambda _checked=False, p=path: self.path_chosen.emit(p))
        lines = QVBoxLayout(row)
        lines.setContentsMargins(10, 6, 10, 6)
        lines.setSpacing(1)
        name = _label("")
        font = name.font()
        font.setWeight(font.Weight.DemiBold)
        name.setFont(font)
        # Elided, as the folder is: a long output name was cut off mid-word
        # by the edge of the row, with nothing to say there was more.
        name.setText(name.fontMetrics().elidedText(os.path.basename(path), Qt.ElideMiddle, 380))
        lines.addWidget(name)
        folder = _label("", "welcomeMuted")
        metrics = folder.fontMetrics()
        folder.setText(metrics.elidedText(folder_label(path), Qt.ElideMiddle, 380))
        lines.addWidget(folder)
        row.setMinimumHeight(name.sizeHint().height() + folder.sizeHint().height() + 16)
        return row

    # ── showing ─────────────────────────────────────────────────────────
    def present(self, shown: bool) -> None:
        """Put the page up over the empty tab, or take it down."""
        if shown and not self.isVisible():
            self._raise_over_parent()
        elif not shown and self.isVisible():
            self._take_down()

    @staticmethod
    def _open_url(url: str) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl(url))

    @guard()
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt's name
        # Opaque, in the window's own colour: an overlay is a window of its own
        # (see ParentOverlay), and this one hides the empty view entirely.
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().window())
        painter.end()


__all__ = ["DOCS_URL", "RELEASE_NOTES_URL", "WelcomeOverlay", "folder_label"]
