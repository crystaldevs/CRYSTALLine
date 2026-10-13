"""The Plots dock: it floats on the first plot, at a figure-friendly aspect.

``MainWindow`` can't be constructed headless (VTK interactor), so the reveal
logic is exercised unbound against a stub window that provides just the bits it
touches — and the geometry maths, which is pure, is tested directly.
"""


import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QRect, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QDockWidget, QMainWindow, QWidget  # noqa: E402

from crystalline.ui.main_window import MainWindow, _floating_plot_geometry  # noqa: E402

_SCREEN = QRect(0, 0, 2560, 1440)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_floating_plot_geometry_is_four_by_three_and_on_screen():
    for window in (QRect(100, 80, 1200, 800), QRect(0, 0, 2560, 1400), QRect(50, 50, 700, 500)):
        rect = _floating_plot_geometry(window, _SCREEN)
        assert rect.width() / rect.height() == pytest.approx(4 / 3, abs=0.01)
        assert _SCREEN.contains(rect), "the plot window must open fully on screen"
        assert rect.width() >= 640  # never squeezed below a usable width


def test_floating_plot_geometry_scales_with_the_main_window():
    small = _floating_plot_geometry(QRect(0, 0, 1200, 800), _SCREEN)
    large = _floating_plot_geometry(QRect(0, 0, 2400, 1400), _SCREEN)
    assert large.width() > small.width()


def test_floating_plot_geometry_is_clamped_to_a_small_screen():
    screen = QRect(0, 0, 800, 600)
    rect = _floating_plot_geometry(QRect(0, 0, 1200, 800), screen)
    assert screen.contains(rect)


class _StubWindow(QMainWindow):
    """Only what ``_reveal_plot_dock`` touches, so no VTK is needed."""

    def __init__(self, controls: int = 0) -> None:
        super().__init__()
        self.resize(1200, 800)
        self._plot_dock = QDockWidget("Plots", self)
        self.addDockWidget(Qt.BottomDockWidgetArea, self._plot_dock)
        self._plot_dock.hide()
        self._plot_dock_floated = False
        self.plot_panel = type("_Panel", (), {"controls_width": lambda _s: controls,
                                              "current_tab": lambda _s: None})()

    _figure_aspect = MainWindow._figure_aspect
    _widen_plot_window = MainWindow._widen_plot_window


def test_the_four_by_three_is_the_figure_s_not_the_window_s():
    """A column of controls beside the plot came out of the plot's own width,
    which left the figure letterboxed in a window that measured 4:3."""
    frame = QRect(0, 0, 1200, 800)
    plain = _floating_plot_geometry(frame, _SCREEN)
    with_controls = _floating_plot_geometry(frame, _SCREEN, beside=378)

    assert with_controls.width() - plain.width() == 378   # added, not taken out
    assert with_controls.height() == plain.height()       # the figure keeps its shape
    assert plain.width() / plain.height() == pytest.approx(4 / 3, abs=0.01)


def test_the_window_asks_the_panel_how_much_its_controls_want(qapp):
    plain, with_controls = _StubWindow(), _StubWindow(controls=200)

    MainWindow._reveal_plot_dock(plain)
    MainWindow._reveal_plot_dock(with_controls)

    assert with_controls._plot_dock.width() > plain._plot_dock.width()


def test_first_plot_floats_the_dock_then_leaves_it_alone(qapp):
    window = _StubWindow()
    assert not window._plot_dock.isFloating()  # starts docked at the bottom

    MainWindow._reveal_plot_dock(window)
    assert window._plot_dock.isFloating()  # first plot pops it out
    assert not window._plot_dock.isHidden()
    ratio = window._plot_dock.width() / max(window._plot_dock.height(), 1)
    assert ratio == pytest.approx(4 / 3, abs=0.3)  # roughly square, not a strip

    # The user drags it back; building more plots must not float it again.
    window._plot_dock.setFloating(False)
    MainWindow._reveal_plot_dock(window)
    assert not window._plot_dock.isFloating()
    assert not window._plot_dock.isHidden()  # still revealed and raised


# ── spectrum ↔ mode linking ───────────────────────────────────────────────
class _StubPhononPanel:
    """The two calls ``_select_mode_near`` makes on the real panel."""

    def __init__(self, freqs) -> None:
        self._freqs = freqs
        self.selected = None
        self.qpoint = None  # which q-point the panel was sent to, if any

    def frequencies(self):
        return None if self._freqs is None else np.asarray(self._freqs, dtype=float)

    def select_mode(self, index: int) -> bool:
        self.selected = index
        return True

    def select_qpoint(self, index: int) -> bool:
        self.qpoint = index
        return True


class _StubModeWindow(QMainWindow):
    """Only what ``_select_mode_near`` touches, so no VTK is needed."""

    def __init__(self, freqs=(100.0, 1650.0, 3400.0), qindex=0) -> None:
        super().__init__()
        self.phonon_panel = _StubPhononPanel(freqs)
        self._qindex = qindex  # which sampled q-point is on show
        self._phonon_dock = QDockWidget("Phonons", self)
        self.addDockWidget(Qt.RightDockWidgetArea, self._phonon_dock)
        self._phonon_dock.hide()
        # The spectra dialog hands this method out as its pick callback, so the
        # stub needs it bound the way the real window has it.
        self._select_mode_near = lambda freq: MainWindow._select_mode_near(self, freq)




def test_clicking_a_peak_selects_that_mode_and_reveals_the_dock(qapp):
    window = _StubModeWindow()

    MainWindow._select_mode_near(window, 1655.0)  # aimed at the 1650 cm⁻¹ band

    assert window.phonon_panel.selected == 1
    assert not window._phonon_dock.isHidden()


def test_clicking_a_peak_answers_with_a_gamma_mode(qapp):
    """IR and Raman bands are zone-centre quantities. With another q-point of a
    SCELPHONO run on show, matching the click against *its* frequencies would
    name a mode that cannot have produced the peak."""
    window = _StubModeWindow(qindex=3)

    MainWindow._select_mode_near(window, 1655.0)

    assert window.phonon_panel.qpoint == 0  # sent back to Gamma first
    assert window.phonon_panel.selected == 1


def test_clicking_the_baseline_selects_nothing(qapp):
    """Most of a broadened spectrum is empty; snapping to the nearest mode from
    hundreds of cm⁻¹ away would be noise, not an answer."""
    window = _StubModeWindow()

    MainWindow._select_mode_near(window, 900.0)  # nowhere near 100/1650/3400

    assert window.phonon_panel.selected is None
    assert window._phonon_dock.isHidden()


def test_clicking_a_spectrum_with_no_modes_loaded_is_harmless(qapp):
    window = _StubModeWindow(freqs=None)

    MainWindow._select_mode_near(window, 1650.0)  # must not raise

    assert window.phonon_panel.selected is None


class _StubEvent:
    def __init__(self, xdata, inaxes=True) -> None:
        self.xdata = xdata
        self.inaxes = object() if inaxes else None


def _tab_with_pick(qapp):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from crystalline.ui.panels.plot_view import _PlotTab

    figure = Figure()
    figure.add_subplot(111).plot([0, 1], [0, 1])
    clicks = []
    return _PlotTab(figure, "spectrum", None, on_pick=clicks.append), clicks


def test_a_plain_click_inside_the_axes_reports_its_frequency(qapp):
    tab, clicks = _tab_with_pick(qapp)

    tab._on_canvas_click(_StubEvent(1655.0))

    assert clicks == [1655.0]


def test_clicks_outside_the_axes_or_while_zooming_are_ignored(qapp):
    tab, clicks = _tab_with_pick(qapp)

    tab._on_canvas_click(_StubEvent(1655.0, inaxes=False))  # on the figure margin
    tab._on_canvas_click(_StubEvent(None))                  # no data coordinate
    assert clicks == []

    tab._toggle_zoom()  # zoom armed: the drag belongs to matplotlib
    tab._on_canvas_click(_StubEvent(1655.0))
    assert clicks == []

    tab._toggle_zoom()  # disarmed again
    tab._on_canvas_click(_StubEvent(1655.0))
    assert clicks == [1655.0]


def test_a_clickable_plot_shows_a_pointing_hand_over_the_axes(qapp):
    """The cursor is what advertises that the peaks do something. Setting it once
    at construction does not survive: matplotlib resets it on every mouse move,
    so the tab has to answer that event itself."""
    tab, _clicks = _tab_with_pick(qapp)

    tab._on_canvas_motion(_StubEvent(1500.0))
    assert tab.canvas.cursor().shape() == Qt.PointingHandCursor

    tab._on_canvas_motion(_StubEvent(None, inaxes=False))  # on the figure margin
    assert tab.canvas.cursor().shape() == Qt.ArrowCursor


def test_pan_and_zoom_keep_their_own_cursor(qapp):
    """While a mode is armed the click pans or zooms; a hand would promise
    something else."""
    tab, _clicks = _tab_with_pick(qapp)
    tab._on_canvas_motion(_StubEvent(1500.0))
    assert tab.canvas.cursor().shape() == Qt.PointingHandCursor

    tab._toggle_pan()
    tab._on_canvas_motion(_StubEvent(1500.0))
    assert tab.canvas.cursor().shape() != Qt.PointingHandCursor


def test_a_plot_that_is_not_clickable_gets_no_hand(qapp):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from crystalline.ui.panels.plot_view import _PlotTab

    figure = Figure()
    figure.add_subplot(111).plot([0, 1], [0, 1])
    tab = _PlotTab(figure, "elastic surface", None)  # no on_pick

    assert tab.canvas.cursor().shape() != Qt.PointingHandCursor
    assert not tab.canvas.toolTip()



def test_the_toolbar_sits_below_the_figure(qapp):
    """Above, it sits between the tab that names the plot and the plot itself, so
    the eye crosses a row of controls to reach the thing it opened."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from crystalline.ui.panels.plot_view import PlotPanel

    figure = plt.figure()
    figure.add_subplot(111).plot([0, 1], [0, 1])
    panel = PlotPanel()
    panel.add_figure(figure, "a plot")

    tab = panel._tabs.widget(0)
    layout = tab.layout()
    widgets = [layout.itemAt(i).widget() for i in range(layout.count())]
    canvas_at = next(i for i, w in enumerate(widgets) if w is tab.canvas)

    assert canvas_at < len(widgets) - 1, "the toolbar should follow the canvas"

    panel.clear()   # and closing it must not leave a draw queued on a dead canvas


# ── a figure that wants another shape ────────────────────────────────────
def test_a_wider_figure_keeps_the_height_and_takes_the_width():
    """Bands beside a DOS: two panels in one figure, whose path labels ran into
    each other at 4:3. The window opens as tall as ever, and wider."""
    frame = QRect(0, 0, 1200, 800)
    plain = _floating_plot_geometry(frame, _SCREEN, beside=300)
    wide = _floating_plot_geometry(frame, _SCREEN, beside=300, aspect=2.0)

    assert wide.height() == plain.height()
    assert (wide.width() - 300) / wide.height() == pytest.approx(2.0, abs=0.01)
    assert _SCREEN.contains(wide)


class _Tab:
    """A plot tab: its canvas, and the recipe that says what shape it wants."""

    def __init__(self, parent, aspect=None):
        from crystalline.ui.panels.plot_view import PlotRecipe

        self.canvas = QWidget(parent)
        self.canvas.setGeometry(0, 0, 300, 225)          # a 4:3 figure
        self.recipe = PlotRecipe(draw=lambda **o: None,
                                 aspect=(lambda _o: aspect) if aspect else None)


def _floating(window):
    dock = window._plot_dock
    dock.setFloating(True)
    dock.setGeometry(0, 50, 500, 400)
    dock.show()
    return dock


def test_a_plot_that_wants_more_width_widens_the_open_window(qapp):
    window = _StubWindow()
    window.show()
    dock = _floating(window)
    before = dock.geometry()

    window._widen_plot_window(_Tab(dock, aspect=2.0))

    assert dock.width() == before.width() + (450 - 300)  # the figure's 300 -> 2 x 225
    assert dock.height() == before.height()
    window.close()


def test_widening_stops_at_the_edge_of_the_screen(qapp):
    window = _StubWindow()
    window.show()
    dock = _floating(window)
    screen = dock.screen().availableGeometry()

    window._widen_plot_window(_Tab(dock, aspect=8.0))    # far more than there is

    assert dock.width() == screen.width()
    assert screen.contains(dock.geometry())
    window.close()


def test_the_window_is_never_narrowed_under_the_user(qapp):
    """Its size may be the user's own, and one that shrank whenever a 4:3 plot
    followed a wide one would jump about."""
    window = _StubWindow()
    window.show()
    dock = _floating(window)
    before = dock.geometry()

    window._widen_plot_window(_Tab(dock))                # 4:3, and it already is

    assert dock.geometry() == before
    window.close()
