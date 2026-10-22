"""A 3D view built inside a window not on screen yet must not hide its page.

From pyvistaqt 0.13 the view is a ``QOpenGLWidget``. Put into a native parent
that draws as raster, Qt re-creates that parent's window for OpenGL and hands it
back the visibility ``isVisible()`` had: False, in a window not yet shown. The
window's first tab is built in the constructor, so its page came out hidden and
the first file opened showed nothing, on every platform with a fresh install.

The view is stood in for here by a plain ``QOpenGLWidget``, which is what makes
Qt do it, so the tests do not depend on which pyvistaqt is installed.
"""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QStackedWidget, QWidget  # noqa: E402

QOpenGLWidget = pytest.importorskip("PySide6.QtOpenGLWidgets").QOpenGLWidget


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _window_with_a_native_page():
    """A window not shown yet, holding a page that is native as the tab pages are.

    The tab pages are native because the overlays are (``ParentOverlay``): Qt
    gives a native widget's siblings native windows of their own.
    """
    window = QWidget()
    stack = QStackedWidget(window)
    page = QWidget()
    stack.addWidget(page)
    overlay = QWidget(stack)
    overlay.setAttribute(Qt.WA_NativeWindow, True)
    overlay.setVisible(False)
    overlay.create()
    page.winId()
    return window, page


def test_qt_hides_the_native_parent_of_an_opengl_widget_made_before_showing(qapp):
    """The premise. If a Qt release stops doing this, the guard is harmless."""
    window, page = _window_with_a_native_page()
    QOpenGLWidget(page)
    if not page.isHidden():
        pytest.skip("this Qt does not re-create the parent here")
    assert page.isHidden()
    window.close()


def test_a_view_made_before_showing_leaves_its_page_shown(qapp, monkeypatch):
    from crystalline.ui import viewport

    monkeypatch.setattr(viewport, "QtInteractor", lambda parent, **_options: QOpenGLWidget(parent))
    window, page = _window_with_a_native_page()

    view = viewport.make_interactor(page, auto_update=False)

    assert view.parentWidget() is page
    assert not page.isHidden()
    window.show()
    QApplication.processEvents()
    assert page.isVisible() and view.isVisible()
    window.close()


def test_a_page_hidden_on_purpose_stays_hidden(qapp, monkeypatch):
    """Put back as it was, not forced visible: a tab behind the one on screen."""
    from crystalline.ui import viewport

    monkeypatch.setattr(viewport, "QtInteractor", lambda parent, **_options: QOpenGLWidget(parent))
    window, page = _window_with_a_native_page()
    page.hide()

    viewport.make_interactor(page)

    assert page.isHidden()
    window.close()


def test_both_3d_views_are_built_through_it():
    import inspect

    from crystalline.ui import viewport
    from crystalline.ui.panels import zone_picker

    assert "make_interactor(self" in inspect.getsource(viewport.Viewport.__init__)
    assert "make_interactor(self" in inspect.getsource(zone_picker._ZoneView.__init__)
