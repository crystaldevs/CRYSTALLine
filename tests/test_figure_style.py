"""The house style every figure is drawn in, and the pass that finishes one."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402

from crystalline.crystalio import figure_style  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_rcparams():
    saved = dict(matplotlib.rcParams)
    yield
    matplotlib.rcParams.update(saved)


def test_curves_are_drawn_in_the_house_colours_with_the_house_ticks():
    figure_style.apply()
    figure, ax = plt.subplots()
    first, second = ax.plot([0, 1], [0, 1]), ax.plot([0, 1], [1, 0])

    assert to_hex(first[0].get_color()) == figure_style.CYCLE[0]   # CRYSTAL cyan
    assert to_hex(second[0].get_color()) == figure_style.CYCLE[1]  # then orange
    assert matplotlib.rcParams["xtick.direction"] == "in"
    assert matplotlib.rcParams["xtick.top"] and matplotlib.rcParams["ytick.right"]
    plt.close(figure)


def test_the_font_is_left_to_the_plot_font_setting():
    """Plot ▸ Plot font owns the font; the style never touches it."""
    assert not any(key.startswith(("font.", "mathtext.")) for key in figure_style.STYLE)


def test_stock_colours_become_house_ones_legend_included():
    """CRYSTALClear names its colours outright — the reach of rcParams ends
    there — so a finished figure has them swapped, swatches and all."""
    figure, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], color="tab:blue", label="E(V)")
    ax.plot([0, 1], [1, 0], "r--", label="harmonic")
    ax.plot([0, 1], [0.5, 0.5], color="black", label="reference")
    ax.plot([0, 1], [0.2, 0.2], color="#123456", label="chosen")
    ax.legend()

    figure_style.finish(figure)
    figure_style.finish(figure)                       # idempotent

    colours = [to_hex(line.get_color()) for line in ax.get_lines()]
    assert colours == [figure_style.CYCLE[0], figure_style.CYCLE[4], "#000000", "#123456"]
    legend = ax.get_legend()
    handles = getattr(legend, "legend_handles", None) or legend.legendHandles
    assert [to_hex(h.get_color()) for h in handles] == colours
    assert ax.get_lines()[1].get_linestyle() == "--"  # only the colour changes
    plt.close(figure)


def test_a_heatmap_keeps_only_its_labelled_ticks():
    figure_style.apply()
    figure, ax = plt.subplots()
    image = ax.imshow(np.eye(4))
    figure.colorbar(image)

    figure_style.finish(figure)

    for axes in figure.axes:
        assert len(axes.xaxis.get_minorticklocs()) == 0
        assert len(axes.yaxis.get_minorticklocs()) == 0
    plt.close(figure)


def test_a_band_path_has_ticks_only_at_its_corners():
    """Between Γ and X a minor tick marks nothing."""
    from crystalline.crystalio import electronic as el

    figure_style.apply()
    figure, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 0])
    ax.set_xticks([0, 1, 2], ["Γ", "X", "Γ"])

    el._tidy_k_axis(ax)

    assert len(ax.xaxis.get_minorticklocs()) == 0
    assert not ax.xaxis._major_tick_kw.get("tick2On", True)   # nothing along the top
    plt.close(figure)


def test_every_figure_reaching_a_plot_tab_is_finished():
    import inspect

    from crystalline.ui.panels import plot_view

    assert "_finish(figure)" in inspect.getsource(plot_view._PlotTab.__init__)
    assert "_finish(figure)" in inspect.getsource(plot_view._PlotTab.set_figure)


def test_an_elastic_surface_says_what_it_measures_in_an_even_colour_map():
    """Not CRYSTALClear's jet surface: viridis, and a colour bar that names the
    property and its unit. The values are still CRYSTALClear's."""
    pytest.importorskip("CRYSTALClear")
    from crystalline.crystalio import elastic

    # An isotropic solid: E = 200 GPa, ν = 0.25, so every direction alike.
    E, nu = 200.0, 0.25
    lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[np.arange(3), np.arange(3)] = lam + 2 * mu
    C[np.arange(3, 6), np.arange(3, 6)] = mu

    figure = elastic.elastic_surface(C, "young", ndeg=24)
    surface_axes, bar_axes = figure.axes

    assert surface_axes.name == "3d"
    assert bar_axes.get_ylabel() == "Young's modulus (GPa)"
    low, high = bar_axes.get_ylim()
    assert low == pytest.approx(E, rel=1e-6) and high == pytest.approx(E, rel=1e-6)
    compressibility = elastic.elastic_surface(C, "comp", ndeg=24)
    assert compressibility.axes[1].get_ylabel() == "Linear compressibility (TPa$^{-1}$)"


def test_a_3d_box_loses_its_walls_and_its_grid_line_per_minor_tick():
    figure_style.apply()
    figure = plt.figure()
    ax = figure.add_subplot(projection="3d")
    ax.plot([0, 1], [0, 1], [0, 1])

    figure_style.finish(figure)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        assert len(axis.get_minorticklocs()) == 0
        assert axis.pane.get_facecolor()[3] == 0.0         # no grey wall
    plt.close(figure)


def test_a_contour_maps_bar_is_ticked_at_round_numbers():
    figure, ax = plt.subplots()
    x = np.linspace(-1, 1, 30)
    filled = ax.contourf(x, x, 12532.0 * np.outer(x, x), levels=21)
    figure.colorbar(filled)

    figure_style.finish(figure)
    figure.canvas.draw()

    ticks = figure.axes[1].get_yticks()
    assert all(float(t).is_integer() and int(t) % 1000 == 0 for t in ticks)
    assert len(ax.xaxis.get_minorticklocs()) == 0          # a map: no minor ticks
    plt.close(figure)


def test_a_level_spans_only_the_well_and_a_wavefunction_stays_whole():
    from crystalline.crystalio.pes import _levels_inside_the_well

    figure, ax = plt.subplots()
    x = np.linspace(-3, 3, 601)
    ax.plot(x, x ** 2, label="cubic + quartic")
    ax.hlines(y=1.0, xmin=-3, xmax=3)                       # E = 1: inside |x| <= 1
    psi = np.exp(-8 * x ** 2)
    ax.plot(x, 1.0 + 0.5 * psi)                             # the state, on its level

    _levels_inside_the_well(ax)

    (x0, e0), (x1, e1) = ax.collections[0].get_segments()[0]
    assert e0 == e1 == 1.0 and x0 == pytest.approx(-1.0, abs=0.02)
    assert x1 == pytest.approx(1.0, abs=0.02)
    wave = ax.get_lines()[1].get_xdata()
    assert wave[0] == -3.0 and wave[-1] == 3.0             # tails and all
    plt.close(figure)


def test_a_figure_carries_no_title_unless_one_was_asked_for():
    """The tab names a plot; CRYSTALClear's own titles ("VCI@…", the elastic
    sections', an XRD file's) went over it all the same."""
    from matplotlib.figure import Figure

    from crystalline.crystalio import figure_style

    figure = Figure()
    ax = figure.add_subplot()
    ax.set_title("VCI@6")
    figure.suptitle("Seebeck at different T")
    figure_style.finish(figure)
    assert ax.get_title() == "" and figure._suptitle.get_text() == ""

    wanted = Figure()
    ax = wanted.add_subplot()
    ax.set_title("MgO, my way")
    figure_style.finish(figure_style.keep_title(wanted))
    assert ax.get_title() == "MgO, my way"


def _dark(figure):
    from crystalline.crystalio import figure_style

    figure_style.to_screen(figure, background="#14161a", ink="#e7eaef", rule="#373c44")


def _drawn():
    from matplotlib.figure import Figure

    from crystalline.crystalio import figure_style

    figure_style.apply()
    figure = Figure()
    ax = figure.add_subplot()
    ax.plot([0, 1], [0, 1], color=figure_style.INK, label="bands")
    ax.plot([0, 1], [1, 0], color="#3f9e4d", label="Fermi level")
    ax.axvline(0.5, color="k")
    ax.set_xlabel("k")
    ax.legend()
    return figure, ax


def test_on_screen_a_figure_takes_the_window_s_colours_and_keeps_its_data_s():
    """A white figure sat in the dark window like a sheet left on the desk."""
    figure, ax = _drawn()
    _dark(figure)

    assert to_hex(figure.patch.get_facecolor()) == "#14161a"
    assert to_hex(ax.lines[0].get_color()) == "#e7eaef"       # the ink
    assert to_hex(ax.lines[2].get_color()) == "#e7eaef"       # black, too
    assert to_hex(ax.lines[1].get_color()) == "#3f9e4d"       # a curve's own colour stays
    assert to_hex(ax.xaxis.label.get_color()) == "#e7eaef"
    assert to_hex(ax.get_legend().get_frame().get_facecolor()) == "#14161a"


def test_back_on_paper_it_is_exactly_as_drawn_however_often_it_was_shown():
    from crystalline.crystalio import figure_style

    figure, ax = _drawn()
    before = [to_hex(figure.patch.get_facecolor()), to_hex(ax.lines[0].get_color()),
              to_hex(ax.xaxis.label.get_color())]
    _dark(figure)
    _dark(figure)                     # the theme changed: no colour swapped twice
    figure_style.to_paper(figure)

    assert [to_hex(figure.patch.get_facecolor()), to_hex(ax.lines[0].get_color()),
            to_hex(ax.xaxis.label.get_color())] == before


def test_a_window_already_white_and_inked_leaves_the_figure_alone():
    from crystalline.crystalio import figure_style

    figure, ax = _drawn()
    figure_style.to_screen(figure, background="#ffffff", ink=figure_style.INK,
                           rule="#dde3e8")
    assert getattr(figure, "_crystalline_on_paper", None) in (None, [])
    assert to_hex(ax.lines[0].get_color()) == figure_style.INK
