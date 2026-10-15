"""The page an empty tab shows, and the recent files it offers."""

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QPushButton, QWidget  # noqa: E402

from crystalline.ui import preferences  # noqa: E402
from crystalline.ui.widgets.welcome import WelcomeOverlay, folder_label  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def no_recent():
    preferences.forget_recent_files()
    yield
    preferences.forget_recent_files()


def test_files_are_remembered_newest_first_once_each_and_only_while_there(tmp_path, no_recent):
    first, second = tmp_path / "a.out", tmp_path / "b.cif"
    first.write_text("x")
    second.write_text("y")

    preferences.remember_file(str(first))
    preferences.remember_file(str(second))
    preferences.remember_file(str(tmp_path / "." / "a.out"))   # the same file again

    assert preferences.recent_files() == [os.path.realpath(first), os.path.realpath(second)]
    second.unlink()
    assert preferences.recent_files() == [os.path.realpath(first)]


def test_the_list_is_kept_short(tmp_path, no_recent):
    for n in range(preferences.RECENT_LIMIT + 4):
        path = tmp_path / f"run{n}.out"
        path.write_text("x")
        preferences.remember_file(str(path))
    assert len(preferences.recent_files()) == preferences.RECENT_LIMIT


def test_the_page_offers_the_recent_files_and_hides_the_column_without_any(qapp, tmp_path):
    host = QWidget()
    page = WelcomeOverlay(host)
    chosen = []
    page.path_chosen.connect(chosen.append)

    page.set_recent([])
    assert page.recent_box.isHidden() and page.recent_heading.isHidden()

    path = str(tmp_path / "quartz.out")
    page.set_recent([path])
    rows = [b for b in page.recent_box.findChildren(QPushButton)
            if b.property("role") == "welcomeRecent"]
    assert len(rows) == 1 and not page.recent_box.isHidden()
    rows[0].click()
    assert chosen == [path]


def test_opening_is_one_click_away(qapp):
    host = QWidget()                     # kept: the page is its child
    page = WelcomeOverlay(host)
    asked = []
    page.open_requested.connect(lambda: asked.append(True))
    page.open_button.click()
    assert asked


def test_a_folder_under_home_is_written_from_the_tilde():
    home = os.path.expanduser("~")
    assert folder_label(os.path.join(home, "runs", "mgo.out")) == os.path.join("~", "runs")
    assert folder_label("/scratch/mgo.out") == "/scratch"


def test_the_window_shows_it_over_an_empty_tab_and_not_over_a_file():
    import inspect

    from crystalline.ui.main_window import MainWindow

    assert "self._sync_welcome()" in inspect.getsource(MainWindow._refresh_chrome)
    assert "self._sync_welcome()" in inspect.getsource(MainWindow._on_structure_changed)
    source = inspect.getsource(MainWindow._show_read_file)
    assert source.index("_sync_welcome") < source.index("_settle_tab")   # down, then drawn
    assert "remember_file(path)" in source


def test_the_suite_never_writes_to_the_real_apps_settings(qapp):
    """The guard in conftest only works if the app's settings honour it."""
    path = preferences._settings().fileName()
    assert "Library/Preferences" not in path and path.endswith(".ini")
