"""Help ▸ About: it builds in either theme, and copies what a bug report needs."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QGuiApplication, QPalette  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import crystalline  # noqa: E402
from crystalline.ui.panels import about_dialog  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_the_dialog_shows_the_logo_and_the_version(qapp):
    dialog = about_dialog.AboutDialog()
    assert not dialog.logo.pixmap().isNull()
    texts = [label.text() for label in dialog.findChildren(type(dialog.logo))]
    assert f"Version {crystalline.__version__}" in texts


def test_the_logo_follows_a_dark_palette(qapp):
    dialog = about_dialog.AboutDialog()
    light = dialog.logo.pixmap().toImage()
    palette = dialog.palette()
    palette.setColor(QPalette.Window, palette.color(QPalette.Window).darker(400))
    dialog.setPalette(palette)
    assert dialog.logo.pixmap().toImage() != light


def test_version_info_is_copied_as_plain_lines(qapp):
    dialog = about_dialog.AboutDialog()
    dialog.copy_button.click()
    copied = QGuiApplication.clipboard().text().splitlines()
    assert copied[0] == f"CRYSTALLine: {crystalline.__version__}"
    assert any(line.startswith("Python: ") for line in copied)
    assert dialog.copy_button.text() == "Copied"


def test_a_missing_package_reads_as_a_dash():
    assert about_dialog._version("no_such_module_here") == "—"
