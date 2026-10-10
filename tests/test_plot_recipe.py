"""A plot tab that can be redrawn: the recipe, and swapping the figure."""

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

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
