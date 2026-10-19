"""Test-wide setup.

The safety net (``crystalline.ui.safety``) swallows exceptions escaping Qt
virtual overrides, which is exactly right in the app and exactly wrong here: a
guard that quietly ate a real bug would let the suite pass over the very faults
it exists to survive. Strict mode makes every guard re-raise, so a broken
``paintEvent`` fails a test instead of being reported and shrugged off.

The tests that exercise the net itself turn strict mode off around what they
are testing.
"""

import pytest


@pytest.fixture(autouse=True, scope="session")
def _guards_re_raise():
    safety = pytest.importorskip("crystalline.ui.safety")
    safety.strict(True)
    yield
    safety.strict(False)


@pytest.fixture(autouse=True, scope="session")
def _settings_of_their_own(tmp_path_factory):
    """Keep the suite's settings away from the person running it.

    The app remembers things between sessions — which sections are folded, the
    files opened lately — through QSettings, and a test that opens a file or
    folds a section would otherwise write them into the real app's settings:
    test outputs at the top of File ▸ Open recent. Every QSettings made here
    is an INI file in a folder of the session's own.
    """
    QtCore = pytest.importorskip("PySide6.QtCore")
    QSettings = QtCore.QSettings
    folder = tmp_path_factory.mktemp("settings")
    before = QSettings.defaultFormat()
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(folder))
    yield folder
    QSettings.setDefaultFormat(before)
