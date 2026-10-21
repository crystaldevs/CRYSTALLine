"""A plot tab that can be redrawn: the recipe, and swapping the figure."""

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Signal  # noqa: E402
from PySide6.QtWidgets import QApplication, QCheckBox, QWidget  # noqa: E402

from crystalline.ui.panels.plot_view import PlotPanel, PlotRecipe  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel(qapp):
    """A panel that is emptied afterwards, as the window empties it on a new
    file: a canvas dropped with a draw still queued fires it into a widget
    whose C++ half has gone."""
    widget = PlotPanel()
    yield widget
    widget.clear()
    qapp.processEvents()


def _figure(label="first"):
    from matplotlib.figure import Figure

    figure = Figure()
    figure.add_subplot(111).plot([0, 1], [0, 1], label=label)
    return figure


def test_a_recipe_draws_again_with_the_options_it_is_given():
    """The options belong to the plot, not to the dialog that was last used:
    two spectra tabs can be broadened differently."""
    asked = []

    def draw(**options):
        asked.append(options)
        return _figure()

    recipe = PlotRecipe(draw=draw, options={"hwhm": 5.0})

    figure, options = recipe.redrawn()
    assert asked[-1] == {"hwhm": 5.0} and options == {"hwhm": 5.0}

    figure, options = recipe.redrawn({"hwhm": 12.0})
    assert asked[-1] == {"hwhm": 12.0}
    assert recipe.options == {"hwhm": 5.0}       # the recipe is not changed by a redraw
    assert options == {"hwhm": 12.0}             # the caller decides what to keep


def test_a_changed_plot_stays_in_the_same_tab(panel):
    """The point of the whole thing: the plot being looked at changes, instead
    of a second tab appearing beside it with one setting different."""
    import matplotlib.pyplot as plt

    first = _figure("first")
    panel.add_figure(first, "Raman")
    tab = panel.widget_at(0)
    old_canvas = tab.canvas

    second = _figure("second")
    panel.replace_figure(0, second, title="Raman (12 cm⁻¹)")

    assert panel.count() == 1                     # not a new tab
    assert tab.figure is second
    assert tab.canvas is not old_canvas           # a canvas belongs to one figure
    assert panel.tab_text(0) == "Raman (12 cm⁻¹)"
    assert not plt.fignum_exists(getattr(first, "number", -1))   # the old one let go


def test_the_pick_handler_survives_a_redraw(panel):
    """A spectrum's peaks stay clickable after its broadening is changed."""
    picked = []

    def on_pick(x):
        picked.append(x)

    panel.add_figure(_figure(), "IR", on_pick=on_pick)
    tab = panel.widget_at(0)

    panel.replace_figure(0, _figure())

    assert tab._on_pick is on_pick
    assert tab.canvas.toolTip() != ""             # still says a peak can be clicked


def test_a_tab_keeps_the_recipe_it_was_given(panel):
    recipe = PlotRecipe(draw=lambda **o: _figure(), options={"hwhm": 5.0})

    panel.add_figure(_figure(), "IR", recipe=recipe)

    assert panel.widget_at(0).recipe is recipe
    assert panel.widget_at(0).recipe.options == {"hwhm": 5.0}


# ── the Edit button ───────────────────────────────────────────────────────
def _edit_button(tab):
    """The row is icon-only wherever matplotlib's icons load, so it is the
    tooltip that names a button."""
    from PySide6.QtWidgets import QToolButton

    return next((b for b in tab.findChildren(QToolButton)
                 if b.toolTip().startswith("Change this plot")), None)


def test_only_a_plot_that_can_be_changed_offers_to_be(panel):
    """Every other button in the row acts on the figure as drawn; this one
    draws it again, which a plot that does not know how it was made cannot."""
    panel.add_figure(_figure(), "Elastic surface")
    assert _edit_button(panel.widget_at(0)) is None

    panel.add_figure(_figure(), "Raman",
                     recipe=PlotRecipe(draw=lambda **o: _figure(), options={}))
    assert _edit_button(panel.widget_at(1)) is None   # nothing to ask the user

    panel.add_figure(_figure(), "IR", recipe=PlotRecipe(
        draw=lambda **o: _figure(), options={}, edit=lambda o: o))
    assert _edit_button(panel.widget_at(2)) is not None


def test_pressing_edit_asks_whoever_owns_the_panel(panel, qapp):
    """The panel does not open the dialog itself: the dialogs belong to the
    window, as does the thread a slow plot is drawn on."""
    asked = []
    panel.edit_requested.connect(asked.append)
    panel.add_figure(_figure(), "IR", recipe=PlotRecipe(
        draw=lambda **o: _figure(), options={}, edit=lambda o: o))
    tab = panel.widget_at(0)

    _edit_button(tab).click()
    qapp.processEvents()

    assert asked == [tab]


def _editing_window(panel):
    """The window's edit handler, on a stand-in with the bits it touches."""
    from crystalline.ui.main_window import MainWindow

    class _Window:
        _edit_plot = MainWindow._edit_plot

        _redraw_plot = MainWindow._redraw_plot
        _live_redraws = MainWindow._live_redraws
        _widen_plot_window = lambda self, tab: None   # noqa: E731 - no floating window here

        def __init__(self):
            self.plot_panel = panel
            self.busy = []
            self.said = []
            outer = self

            class _Bar:
                """The status bar, as far as the redraws use it."""

                def __init__(self):
                    self.message = ""

                def showMessage(self, text, _timeout=0):   # noqa: N802 - Qt's name
                    self.message = text
                    outer.said.append(text)

                def currentMessage(self):                  # noqa: N802 - Qt's name
                    return self.message

                def clearMessage(self):                    # noqa: N802 - Qt's name
                    self.message = ""

            self._bar = _Bar()

        def _run_busy(self, build, message, show, _failure):
            self.busy.append(message)
            show(build())            # the thread, without the thread

        def statusBar(self):
            return self._bar

    return _Window()


def _drained(window, qapp, limit=10.0):
    """Let the redraws asked of ``window`` finish, as the event loop would."""
    import time

    end = time.time() + limit
    while not window._live_redraws().idle() and time.time() < end:
        qapp.processEvents()
        time.sleep(0.005)
    qapp.processEvents()
    assert window._live_redraws().idle(), "a redraw never finished"


def test_editing_a_plot_redraws_it_in_the_same_tab(panel):
    """What the whole thing is for: the plot being looked at changes."""
    drawn = []

    def draw(**options):
        drawn.append(options)
        return _figure()

    recipe = PlotRecipe(
        draw=draw, options={"hwhm": 5.0}, title=lambda o: f"IR ({o['hwhm']:.0f})",
        edit=lambda options: {**options, "hwhm": 12.0},
    )
    panel.add_figure(_figure(), "IR (5)", recipe=recipe)
    tab = panel.widget_at(0)

    _editing_window(panel)._edit_plot(tab)

    assert drawn == [{"hwhm": 12.0}]
    assert panel.count() == 1 and panel.widget_at(0) is tab
    assert panel.tab_text(0) == "IR (12)"
    assert recipe.options == {"hwhm": 12.0}   # the tab's own settings, kept


def test_a_cancelled_edit_changes_nothing(panel):
    recipe = PlotRecipe(draw=lambda **o: _figure(), options={"hwhm": 5.0},
                        edit=lambda _options: None)
    panel.add_figure(_figure(), "IR", recipe=recipe)
    window = _editing_window(panel)

    window._edit_plot(panel.widget_at(0))

    assert window.busy == []                  # nothing was drawn at all
    assert recipe.options == {"hwhm": 5.0}


def test_a_tab_closed_while_it_was_being_drawn_is_not_resurrected(panel, qapp, monkeypatch):
    """Plots are drawn off the main thread, so the tab can go in the meantime.
    The finished figure is then let go rather than left in memory with nothing
    to show it."""
    import matplotlib.pyplot as plt

    made = []

    def draw(**_options):
        made.append(_figure())
        return made[-1]

    closed = []
    monkeypatch.setattr(plt, "close", closed.append)

    recipe = PlotRecipe(draw=draw, options={}, edit=lambda options: dict(options))
    panel.add_figure(_figure(), "IR", recipe=recipe)
    tab = panel.widget_at(0)
    window = _editing_window(panel)
    panel.clear()                              # closed before the draw lands
    qapp.processEvents()

    window._edit_plot(tab)

    assert panel.count() == 0                  # no tab comes back
    assert made[-1] in closed                  # and what was drawn is released


# ── controls beside the plot ──────────────────────────────────────────────
class _Knobs(QWidget):
    """Stands in for a plot's own controls: a widget with options and a signal."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._options = {}

    def options(self):
        return dict(self._options)

    def set_options(self, options):
        self._options = dict(options or {})

    def turn(self, **options):
        """A knob moved, as a slider does under a hand."""
        self._options.update(options)
        self.changed.emit()


def test_a_plot_without_controls_is_laid_out_as_before(panel):
    panel.add_figure(_figure(), "Elastic surface")
    assert panel.widget_at(0).controls is None


def test_the_controls_open_on_the_plot_s_own_settings(panel):
    """They describe the plot they sit beside, so they start where it is."""
    knobs = _Knobs()
    panel.add_figure(_figure(), "IR", recipe=PlotRecipe(
        draw=lambda **o: _figure(), options={"hwhm": 7.0}, controls=lambda: knobs))

    assert panel.widget_at(0).controls is knobs
    assert knobs.options() == {"hwhm": 7.0}


def test_turning_a_knob_asks_for_a_redraw_once_it_comes_to_rest(panel, qapp):
    """A slider reports every pixel under the hand; a spectrum redrawn per
    pixel is a spectrum nobody can drag."""
    import time

    asked = []
    knobs = _Knobs()
    panel.redraw_requested.connect(lambda tab, options: asked.append(options))
    panel.add_figure(_figure(), "IR", recipe=PlotRecipe(
        draw=lambda **o: _figure(), options={"hwhm": 5.0}, controls=lambda: knobs))

    for width in (6.0, 7.0, 8.0):          # a drag across three values
        knobs.turn(hwhm=width)
        qapp.processEvents()
    assert asked == []                      # nothing yet: the hand is still moving

    deadline = time.time() + 2
    while not asked and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.02)

    assert asked == [{"hwhm": 8.0}]         # once, where it came to rest


def test_a_live_redraw_replaces_the_figure_without_a_dialog(panel, qapp):
    """The point of the whole exercise: turn a knob, see the plot."""
    drawn = []

    def draw(**options):
        drawn.append(options)
        return _figure()

    recipe = PlotRecipe(draw=draw, options={"hwhm": 5.0, "curves": ("IR",)},
                        title=lambda o: f"IR ({o['hwhm']:.0f})", controls=_Knobs)
    panel.add_figure(_figure(), "IR (5)", recipe=recipe)
    tab = panel.widget_at(0)
    window = _editing_window(panel)

    window._redraw_plot(tab, {"hwhm": 9.0})
    _drained(window, qapp)

    assert drawn == [{"hwhm": 9.0, "curves": ("IR",)}]   # merged over what it had
    assert panel.count() == 1 and panel.widget_at(0) is tab
    assert panel.tab_text(0) == "IR (9)"
    assert window.busy == []                             # no scrim for a knob


def test_a_plot_that_cannot_be_drawn_says_so_quietly(panel, qapp):
    """One of several, as a slider passes through a range that cannot be drawn:
    three modal apologies in a row is not how to say so."""
    def draw(**_options):
        raise ValueError("no curve in that range")

    recipe = PlotRecipe(draw=draw, options={}, controls=_Knobs)
    panel.add_figure(_figure(), "IR", recipe=recipe)
    window = _editing_window(panel)

    window._redraw_plot(panel.widget_at(0), {"hwhm": 9.0})
    _drained(window, qapp)

    assert "no curve in that range" in window.said[-1]
    assert panel.count() == 1


# ── the dialogs that are all the same shape ───────────────────────────────
def test_one_helper_serves_every_dialog_driven_plot(panel, qapp):
    """PES, VCI and the anharmonic scan differ only in how their dialog is
    built and how the figure is drawn; everything else — restore, ask,
    remember, name the tab — is the same, so it is written once."""
    from crystalline.ui.main_window import MainWindow

    class _Dialog:
        """The part of a plot dialog the helper touches."""

        def __init__(self, answer):
            self._answer = answer
            self.span = 2.0          # a widget-ish attribute for dialog_state

        def exec(self):
            return self._answer

        def options(self):
            return {"mode": 7, "span": self.span}

        def title(self):
            return f"Mode 7 — span {self.span:.0f}"

    from PySide6.QtWidgets import QDialog

    drawn = []
    accepted = _Dialog(QDialog.Accepted)

    class _Window:
        _dialog_recipe = MainWindow._dialog_recipe
        _restore_dialog = lambda self, dialog, key, state=None: None   # noqa: E731
        _remember_dialog = lambda self, dialog, key: None              # noqa: E731

    later = _Dialog(QDialog.Accepted)
    later.span = 5.0
    recipe = _Window()._dialog_recipe(
        "pes", lambda: later, lambda **o: drawn.append(o) or _figure(), accepted)

    assert recipe.options == {"mode": 7, "span": 2.0}
    assert recipe.title(recipe.options) == "Mode 7 — span 2"

    options = recipe.edit(dict(recipe.options))          # the user changes the span

    assert options == {"mode": 7, "span": 5.0}
    assert recipe.title(options) == "Mode 7 — span 5"    # the dialog names it, not us

    # ...unless what the name says can be changed beside the plot, as a VCI
    # tab's representation can: then it is worked out from what is drawn.
    named = _Window()._dialog_recipe(
        "vci", lambda: later, lambda **o: _figure(), accepted,
        name=lambda o: f"span {o['span']:.0f}")
    assert named.title({"mode": 7, "span": 9.0}) == "span 9"


def test_a_cancelled_dialog_leaves_the_plot_as_it_was(panel):
    from PySide6.QtWidgets import QDialog

    from crystalline.ui.main_window import MainWindow

    class _Dialog:
        def __init__(self, answer):
            self._answer = answer

        def exec(self):
            return self._answer

        def options(self):
            return {"mode": 7}

        def title(self):
            return "Mode 7"

    class _Window:
        _dialog_recipe = MainWindow._dialog_recipe
        _restore_dialog = lambda self, dialog, key, state=None: None   # noqa: E731
        _remember_dialog = lambda self, dialog, key: None              # noqa: E731

    recipe = _Window()._dialog_recipe(
        "vci", lambda: _Dialog(QDialog.Rejected), lambda **o: _figure(),
        _Dialog(QDialog.Accepted))

    assert recipe.edit(dict(recipe.options)) is None
    assert recipe.title(recipe.options) == "Mode 7"


def test_an_edit_moves_the_knobs_beside_the_plot_too(panel):
    """They describe the plot; after the dialog changes it they would
    otherwise assert the settings of a figure no longer on screen."""
    knobs = _Knobs()
    recipe = PlotRecipe(
        draw=lambda **o: _figure(), options={"hwhm": 5.0}, controls=lambda: knobs,
        edit=lambda options: {**options, "hwhm": 20.0},
    )
    panel.add_figure(_figure(), "IR", recipe=recipe)
    tab = panel.widget_at(0)
    assert knobs.options() == {"hwhm": 5.0}

    _editing_window(panel)._edit_plot(tab)

    assert knobs.options() == {"hwhm": 20.0}


def test_the_controls_keep_clear_of_the_window_s_edge(panel):
    """With no margin the boxes ran into the edge of the window, and the first
    of them into the figure. The side panels' own margins, so that a column of
    controls looks the same wherever it stands."""
    from crystalline.ui.panels.controls import PANEL_MARGINS

    knobs = _Knobs()
    knobs.setFixedWidth(250)
    panel.add_figure(_figure(), "IR", recipe=PlotRecipe(
        draw=lambda **o: _figure(), controls=lambda: knobs))
    from crystalline.ui.panels.plot_view import _CONTROLS_WIDTH

    column = panel.widget_at(0).controls_column.widget().layout()

    margins = column.contentsMargins()
    assert (margins.left(), margins.right(), margins.bottom()) == (
        PANEL_MARGINS[0], PANEL_MARGINS[2], PANEL_MARGINS[3])
    # ...and the window that opens for them leaves the figure its width
    assert panel.controls_width() >= _CONTROLS_WIDTH + PANEL_MARGINS[0] + PANEL_MARGINS[2]


def test_unfolding_a_section_beside_a_plot_leaves_the_window_its_size(panel, qapp):
    """Unfolding Appearance beside a spectrum made the controls taller than the
    Plots window, and the window grew to hold them: a click changed the
    figure's shape. They scroll instead, in a column of one width."""
    from PySide6.QtWidgets import QVBoxLayout as _Column

    from crystalline.ui.panels.controls import Section

    class _Tall(_Knobs):
        def __init__(self):
            super().__init__()
            layout = self.layout() or _Column(self)
            self.section = Section(layout, "Appearance", collapsed=True)
            for n in range(30):
                self.section.add(f"Row {n}", QCheckBox())

    knobs = _Tall()
    panel.add_figure(_figure(), "IR", recipe=PlotRecipe(
        draw=lambda **o: _figure(), controls=lambda: knobs))
    panel.resize(700, 320)
    panel.show()
    qapp.processEvents()
    column = panel.widget_at(0).controls_column
    width = column.width()

    knobs.section.header.click()
    qapp.processEvents()
    assert panel.minimumSizeHint().height() < 320
    assert panel.size().height() == 320
    assert column.width() == width
    panel.hide()


def test_a_redrawn_plot_lets_the_old_figure_go(panel, qapp):
    """Every redraw from the controls beside a plot used to leave its figure
    behind, about 8 MB each for as long as the tab was open: the hidden
    toolbar that does pan and zoom was made afresh for each new canvas and the
    old one kept, a child of the tab, holding its canvas and through it its
    figure. Tuning a spectrum for a while left the app hundreds of MB heavier.
    """
    import gc
    import weakref

    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
    from PySide6.QtCore import QEvent

    def settle():
        qapp.processEvents()
        # What the running event loop does with deleteLater, and
        # processEvents() alone does not.
        QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        gc.collect()

    panel.add_figure(_figure(), "IR", recipe=PlotRecipe(draw=lambda **o: _figure()))
    old = []
    for _ in range(5):
        figure = _figure()
        old.append(weakref.ref(figure))
        panel.replace_figure(0, figure, "IR")
        del figure
        settle()

    assert [ref() is None for ref in old[:-1]] == [True] * 4      # all but the one on screen
    assert len(panel.widget_at(0).findChildren(NavigationToolbar2QT)) == 1



# ── redrawn off the UI thread ─────────────────────────────────────────────
def _slow_recipe(drawn, seconds=0.15):
    """A plot that takes a while to draw, as a CRYSTALClear spectrum does."""
    import threading
    import time

    def draw(**options):
        drawn.append((dict(options), threading.current_thread() is threading.main_thread()))
        time.sleep(seconds)
        return _figure()

    return PlotRecipe(draw=draw, options={"hwhm": 5.0},
                      title=lambda o: f"IR ({o['hwhm']:.0f})", controls=_Knobs)


def test_a_live_redraw_is_drawn_off_the_ui_thread(panel, qapp):
    """A plotter can take seconds, and on the UI thread every knob that came to
    rest froze the whole window for as long."""
    drawn = []
    panel.add_figure(_figure(), "IR (5)", recipe=_slow_recipe(drawn))
    window = _editing_window(panel)

    window._redraw_plot(panel.widget_at(0), {"hwhm": 8.0})
    assert panel.tab_text(0) == "IR (5)"             # returned at once: nothing drawn yet
    _drained(window, qapp)

    assert [on_ui for _options, on_ui in drawn] == [False]
    assert panel.tab_text(0) == "IR (8)"


def test_the_newest_request_wins(panel, qapp):
    """A drag sends a request per resting point; those that arrive while a
    figure is being drawn only replace each other, and the last is drawn."""
    drawn = []
    panel.add_figure(_figure(), "IR (5)", recipe=_slow_recipe(drawn))
    tab = panel.widget_at(0)
    window = _editing_window(panel)

    for width in (6.0, 7.0, 8.0, 9.0):
        window._redraw_plot(tab, {"hwhm": width})
    _drained(window, qapp)

    assert [options["hwhm"] for options, _on_ui in drawn] == [6.0, 9.0]
    assert tab.recipe.options["hwhm"] == 9.0
    assert panel.tab_text(0) == "IR (9)"


def test_an_edit_outranks_a_redraw_still_under_way(panel, qapp):
    """What the dialog sets is newer than what the controls asked for before
    it was opened: a figure from before must not land on top of it."""
    drawn = []
    panel.add_figure(_figure(), "IR (5)", recipe=_slow_recipe(drawn))
    tab = panel.widget_at(0)
    window = _editing_window(panel)
    live = window._live_redraws()

    window._redraw_plot(tab, {"hwhm": 8.0})
    assert live.wanted(tab) == {"hwhm": 8.0}         # what the dialog would open on
    live.withdraw(tab)                               # as an accepted edit does
    _drained(window, qapp)

    assert panel.tab_text(0) == "IR (5)"
    assert tab.recipe.options["hwhm"] == 5.0


def test_a_plot_closed_while_it_was_being_redrawn_is_left_closed(panel, qapp):
    drawn = []
    panel.add_figure(_figure(), "IR (5)", recipe=_slow_recipe(drawn))
    window = _editing_window(panel)

    window._redraw_plot(panel.widget_at(0), {"hwhm": 8.0})
    panel.clear()                                    # the tab goes, the figure is on its way
    _drained(window, qapp)

    assert panel.count() == 0


def test_a_slow_redraw_says_it_is_under_way(panel, qapp):
    """A plot that takes longer than a moment would look like a knob that did
    nothing, so the status bar says it is being drawn — and stops saying so."""
    from crystalline.ui.panels.plot_view import LiveRedraws

    drawn = []
    panel.add_figure(_figure(), "IR (5)", recipe=_slow_recipe(drawn, LiveRedraws.SLOW_MS / 1000 + 0.2))
    window = _editing_window(panel)

    window._redraw_plot(panel.widget_at(0), {"hwhm": 8.0})
    _drained(window, qapp)

    assert any("Drawing the plot again" in text for text in window.said)
    assert window.statusBar().currentMessage() == ""


def test_every_plot_gives_its_controls_the_same_share_of_the_window(panel, qapp):
    """Each column took its controls' own width, so in a window of one size the
    figure came out a different size for every kind of plot."""
    from crystalline.ui.panels import plot_view

    shared = dict(plot_view._COLUMN)
    narrow, wide = _Knobs(), _Knobs()
    narrow.setMinimumWidth(120)
    wide.setMinimumWidth(520)          # wider than the shared width itself
    for title, knobs in (("narrow", narrow), ("wide", wide)):
        panel.add_figure(_figure(), title, recipe=PlotRecipe(
            draw=lambda **o: _figure(), controls=lambda k=knobs: k))
    panel.resize(1100, 500)
    panel.show()
    widths = []
    for index in range(panel.count()):
        panel._tabs.setCurrentIndex(index)
        qapp.processEvents()
        widths.append(panel.widget_at(index).controls_column.width())
    panel.hide()
    plot_view._COLUMN.update(shared)    # the next test's columns are its own

    assert widths[0] == widths[1]
    assert widths[0] >= 520


def test_the_controls_fold_away_for_every_plot_and_stay_folded(panel, qapp):
    """A button folds the column away, for the figure to have the window. It
    is one choice for every plot — one folded and the next not would put the
    figure at another size again — and it is remembered."""
    from crystalline.ui import preferences
    from crystalline.ui.panels import plot_view

    before = dict(plot_view._SHOWN)
    try:
        for title in ("first", "second"):
            panel.add_figure(_figure(), title, recipe=PlotRecipe(
                draw=lambda **o: _figure(), controls=lambda: _Knobs()))
        panel.resize(900, 400)
        panel.show()
        panel._tabs.setCurrentIndex(0)
        qapp.processEvents()
        first = panel.widget_at(0)
        assert not first.controls_column.isHidden()

        first._btn_controls.click()
        qapp.processEvents()
        assert first.controls_column.isHidden()
        assert first._btn_controls.toolTip() == "Show the controls"

        panel._tabs.setCurrentIndex(1)
        qapp.processEvents()
        assert panel.widget_at(1).controls_column.isHidden()
        assert preferences.section_open("plots/controls", True) is False
        assert panel.controls_width() == 0       # a new window opens for the figure alone
    finally:
        panel.hide()
        plot_view._SHOWN.update(before)
        preferences.set_section_open("plots/controls", True)
