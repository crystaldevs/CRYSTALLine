"""Help ▸ About CRYSTALLine: the logo, the version, and where to go next.

Also the versions of what the program runs on, since that is the first thing a
bug report needs — shown small, and copied in one click as plain text.
"""

from __future__ import annotations

import platform
from typing import List, Tuple

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QGuiApplication, QPalette, QPixmap
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QGridLayout, QHBoxLayout,
                               QLabel, QPushButton, QVBoxLayout)

from crystalline.resources import asset_path
from crystalline.ui.widgets.welcome import DOCS_URL, RELEASE_NOTES_URL

SOURCE_URL = "https://github.com/crystaldevs/CRYSTALLine"
LICENCE = "GPL-3.0-or-later"
DESCRIPTION = (
    "A program to build, display and manipulate the structures used by the "
    "CRYSTAL quantum-chemistry code, to animate the vibrational modes computed "
    "from them, and to plot the properties a calculation returns."
)
LOGO_WIDTH = 380

_STYLE = """
QLabel[role="aboutMuted"] { color: palette(placeholder-text); }
QPushButton[role="aboutLink"] {
    border: none; background: transparent; padding: 2px 0; color: palette(link);
}
QPushButton[role="aboutLink"]:hover { text-decoration: underline; }
"""


def _version(module_name: str, distribution: str = "") -> str:
    """A module's own ``__version__``, else its installed metadata, else "—"."""
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", None)
        if version:
            return str(version)
    except Exception:  # noqa: BLE001 - not installed, or broken: say so, don't fail
        return "—"
    try:
        from importlib.metadata import version as _metadata_version

        return _metadata_version(distribution or module_name)
    except Exception:  # noqa: BLE001
        return "—"


def system_info() -> List[Tuple[str, str]]:
    """What this copy of CRYSTALLine is, and what it runs on."""
    import crystalline

    try:
        from PySide6 import __version__ as pyside
        from PySide6.QtCore import qVersion

        qt = f"{qVersion()} (PySide6 {pyside})"
    except Exception:  # noqa: BLE001
        qt = "—"
    return [
        ("CRYSTALLine", crystalline.__version__),
        ("CRYSTALClear", _version("CRYSTALClear")),
        ("Python", platform.python_version()),
        ("Qt", qt),
        ("VTK", _version("vtk")),
        ("PyVista", _version("pyvista")),
        ("Matplotlib", _version("matplotlib")),
        ("NumPy", _version("numpy")),
        ("System", _system()),
    ]


def _system() -> str:
    """The operating system as people know it: macOS 15.5, not Darwin 24.5.0."""
    mac = platform.mac_ver()[0]
    name = f"macOS {mac}" if mac else f"{platform.system()} {platform.release()}"
    return f"{name} ({platform.machine()})"


def system_info_text() -> str:
    """:func:`system_info` as lines to paste into an issue."""
    return "\n".join(f"{name}: {value}" for name, value in system_info())


class AboutDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        import crystalline

        self.setWindowTitle("About CRYSTALLine")
        self.setStyleSheet(_STYLE)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 18)
        layout.setSpacing(0)

        self.logo = QLabel(self)
        self.logo.setAlignment(Qt.AlignCenter)
        self._paint_logo()
        layout.addWidget(self.logo)
        layout.addSpacing(14)

        version = QLabel(f"Version {crystalline.__version__}", self)
        version.setAlignment(Qt.AlignCenter)
        version.setProperty("role", "aboutMuted")
        layout.addWidget(version)
        layout.addSpacing(14)

        description = QLabel(DESCRIPTION, self)
        description.setWordWrap(True)
        description.setAlignment(Qt.AlignCenter)
        description.setFixedWidth(LOGO_WIDTH)
        layout.addWidget(description, 0, Qt.AlignHCenter)
        layout.addSpacing(12)

        links = QHBoxLayout()
        links.setSpacing(20)
        links.addStretch(1)
        for text, url in (("Documentation", DOCS_URL), ("Release notes", RELEASE_NOTES_URL),
                          ("Source code", SOURCE_URL)):
            link = QPushButton(text, self)
            link.setProperty("role", "aboutLink")
            link.setCursor(Qt.PointingHandCursor)
            link.setAutoDefault(False)
            link.clicked.connect(lambda _checked=False, u=url: QDesktopServices.openUrl(QUrl(u)))
            links.addWidget(link)
        links.addStretch(1)
        layout.addLayout(links)
        layout.addSpacing(18)

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(2)
        for row, (name, value) in enumerate(system_info()):
            key = QLabel(name, self)
            key.setProperty("role", "aboutMuted")
            key.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            shown = QLabel(value, self)
            shown.setTextInteractionFlags(Qt.TextSelectableByMouse)
            for label in (key, shown):
                font = label.font()
                font.setPointSizeF(font.pointSizeF() * 0.9)
                label.setFont(font)
            grid.addWidget(key, row, 0)
            grid.addWidget(shown, row, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)
        layout.addSpacing(18)

        credit = QLabel(f"Built on CRYSTALClear · Released under the {LICENCE} licence", self)
        credit.setAlignment(Qt.AlignCenter)
        credit.setProperty("role", "aboutMuted")
        layout.addWidget(credit)
        layout.addSpacing(16)

        buttons = QDialogButtonBox(QDialogButtonBox.Close, self)
        self.copy_button = buttons.addButton("Copy version info", QDialogButtonBox.ActionRole)
        self.copy_button.setAutoDefault(False)
        self.copy_button.clicked.connect(self._copy)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _paint_logo(self) -> None:
        # The wordmark's frame and lattice are grey on white; on a dark window
        # the light-on-dark version is the one that reads.
        dark = self.palette().color(QPalette.Window).lightness() < 128
        pixmap = QPixmap(asset_path("wordmark-dark.png" if dark else "wordmark-light.png"))
        ratio = self.devicePixelRatioF() or 1.0
        pixmap = pixmap.scaledToWidth(round(LOGO_WIDTH * ratio), Qt.SmoothTransformation)
        pixmap.setDevicePixelRatio(ratio)
        self.logo.setPixmap(pixmap)

    def _copy(self) -> None:
        QGuiApplication.clipboard().setText(system_info_text())
        self.copy_button.setText("Copied")

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt's name
        super().changeEvent(event)
        if event.type() == event.Type.PaletteChange and hasattr(self, "logo"):
            self._paint_logo()


__all__ = ["AboutDialog", "SOURCE_URL", "system_info", "system_info_text"]
