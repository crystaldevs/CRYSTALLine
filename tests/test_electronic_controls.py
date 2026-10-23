"""Changing a band structure or a DOS beside the plot, and editing it again.

What the plot *is* — the files, the mode, the projections and their names —
stays with the dialog; what is judged by looking, the energy window first of
all, stands next to the figure. Editing reopens the dialog on the plot's own
choices rather than on the folder's best guess.
"""

import dataclasses
import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("CRYSTALClear")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from crystalline.crystalio import electronic as el  # noqa: E402
from crystalline.ui.panels import electronic_controls as ec  # noqa: E402
from crystalline.ui.panels.electronic_dialog import ElectronicDialog  # noqa: E402

_MGO = os.path.join(os.path.dirname(__file__), "data", "mgo")


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _answer(dialog) -> dict:
    """What the window keeps as a plot's options: the dialog's whole answer."""
    bands_path, dos_path, options = dialog.request()
    return {"bands_path": bands_path, "dos_path": dos_path,
            **dataclasses.asdict(options), **dialog.extent()}


def _dialog(qapp, mode=el.BANDS_AND_DOS):
    dialog = ElectronicDialog(folder=_MGO)
    dialog.mode.setCurrentIndex(dialog.mode.findData(mode))
    dialog.show()              # the window is fitted to the files on the first show
    qapp.processEvents()
    dialog.hide()
    return dialog


def _knobs(options):
    knobs = ec.ElectronicControls()
    knobs.set_options(options)
    moves = []
    knobs.changed.connect(lambda: moves.append(knobs.options()))
    return knobs, moves


def _draw(options):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from crystalline.crystalio import plot_electronic

    figure = plot_electronic(*ec.plot_arguments(options))
    limits = figure.axes[0].get_ylim()
    plt.close(figure)
    return limits


def test_the_controls_open_on_the_plot_s_window_and_its_track(qapp):
    answer = _answer(_dialog(qapp))
    knobs, moves = _knobs(answer)

    assert knobs.options()["energy_range"] == answer["energy_range"]
    assert tuple(knobs._window[0].bounds()) == answer["energy_bounds"]
    assert moves == []                       # opened on it, not announced


def test_a_dragged_window_is_the_window_drawn(qapp):
    """The reason it is beside the plot: pulled until the bands that matter
    fill the axis, and seen doing it."""
    answer = _answer(_dialog(qapp))
    knobs, moves = _knobs(answer)

    knobs._window[1].setValue(-6.0)

    assert moves[-1]["energy_range"][0] == -6.0
    assert _draw({**answer, **moves[-1]}) == pytest.approx((-6.0, answer["energy_range"][1]))


def test_what_the_tracks_were_is_not_handed_to_the_plotter(qapp):
    """They travel with a plot's options for the controls; plot_electronic
    would refuse them."""
    answer = _answer(_dialog(qapp))

    bands_path, dos_path, options = ec.plot_arguments(answer)

    assert (bands_path, dos_path) == (answer["bands_path"], answer["dos_path"])
    assert isinstance(options, el.ElectronicOptions)


@pytest.mark.parametrize("mode, dos, overlay", [
    (el.BANDS, False, False),            # a band structure has no DOS axis
    (el.DOS, True, True),                # a DOS alone can split into panels
    (el.BANDS_AND_DOS, True, False),     # the combined view always overlays
])
def test_what_a_plot_has_no_use_for_is_not_offered(qapp, mode, dos, overlay):
    knobs, _ = _knobs({"mode": mode, "dos_polarised": False})

    assert knobs._dos_section.header.isVisibleTo(knobs) is dos
    assert knobs.overlay.isVisibleTo(knobs) is overlay
    assert not knobs.spin_down.isVisibleTo(knobs)         # spin-restricted


def test_spin_down_is_offered_for_a_spin_polarised_dos(qapp):
    knobs, _ = _knobs({"mode": el.DOS, "dos_polarised": True})

    assert knobs.spin_down.isVisibleTo(knobs)


def test_a_dos_range_is_set_by_hand_once_it_is_no_longer_fitted(qapp):
    answer = _answer(_dialog(qapp, el.DOS))
    knobs, moves = _knobs(answer)
    assert knobs.options()["dos_range"] is None
    assert not knobs._dos_span[1].isEnabled()

    knobs.dos_auto.setChecked(False)

    assert moves[-1]["dos_range"] == answer["dos_handles"]
    assert knobs._dos_span[1].isEnabled()


def test_editing_reopens_the_dialog_on_the_plot_s_own_choices(qapp):
    """Not the folder's best guess: the projections ticked and renamed, the
    title, and the window dragged beside the plot all come back."""
    first = _dialog(qapp)
    first._table.item(0, 0).setCheckState(Qt.Unchecked)
    first._table.item(1, 0).setText("Mg s")
    first._title.setText("MgO, my way")
    plotted = _answer(first)
    knobs, moves = _knobs(plotted)
    knobs._window[1].setValue(-6.0)
    plotted = {**plotted, **moves[-1]}

    again = ElectronicDialog(folder=_MGO)
    again.set_request(*ec.plot_arguments(plotted))
    again.show()
    qapp.processEvents()
    again.hide()

    assert _answer(again) == {**plotted, "dos_handles": _answer(again)["dos_handles"]}
    assert again._table.item(0, 0).checkState() == Qt.Unchecked
    assert again._table.item(1, 0).text() == "Mg s"
    assert again.window_values() == (-6.0, plotted["energy_range"][1])


def test_the_tab_is_named_by_what_is_drawn():
    assert ec.title({"mode": el.BANDS}) == "Band structure"
    assert ec.title({"mode": el.BANDS_AND_DOS}) == "Bands + DOS"


# ── a legend only where it tells something apart ─────────────────────────
@pytest.mark.parametrize("mode, projections, bands_spin, dos_spin, overlay, useful", [
    (el.BANDS, 0, False, False, True, False),        # one band structure
    (el.BANDS, 0, True, False, True, True),          # α and β
    (el.DOS, 1, False, False, True, False),          # one projection
    (el.DOS, 3, False, False, True, True),           # three
    (el.DOS, 1, False, True, True, False),           # on one axis α and β are named alike
    (el.DOS, 1, False, True, False, True),           # split into panels they are not
    (el.BANDS_AND_DOS, 1, False, False, True, False),
    (el.BANDS_AND_DOS, 1, True, False, True, True),
    (el.BANDS_AND_DOS, 2, False, False, True, True),
])
def test_a_legend_is_drawn_only_when_it_names_more_than_one_thing(
        mode, projections, bands_spin, dos_spin, overlay, useful):
    assert el.legend_names_anything(mode, projections, bands_spin, dos_spin, overlay) is useful


def test_one_projection_draws_no_legend_whatever_is_asked():
    """It would name the one curve on the plot, which its tab already does."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from crystalline.crystalio import plot_electronic

    figure = plot_electronic(None, os.path.join(_MGO, "mgo_doss_partialao.DOSS"),
                             el.ElectronicOptions(mode=el.DOS, projections=[1], legend=True))
    named = [ax for ax in figure.axes if ax.get_legend() is not None]
    plt.close(figure)

    assert named == []


@pytest.mark.parametrize("plot, offered", [
    ({"mode": el.BANDS, "bands_polarised": False}, False),
    ({"mode": el.DOS, "projections": [1], "dos_projections": 9}, False),
    ({"mode": el.DOS, "projections": None, "dos_projections": 9}, True),
    ({"mode": el.BANDS_AND_DOS, "projections": [2, 3], "dos_projections": 9}, True),
])
def test_the_tick_beside_the_plot_is_offered_only_then(qapp, plot, offered):
    knobs, _ = _knobs(plot)

    assert (not knobs.legend.isHidden()) is offered


def test_splitting_a_spin_polarised_dos_into_panels_brings_the_tick_back(qapp):
    knobs, _ = _knobs({"mode": el.DOS, "projections": [1], "dos_projections": 5,
                       "dos_polarised": True, "overlay_projections": True})
    assert knobs.legend.isHidden()

    knobs.overlay.setChecked(False)

    assert not knobs.legend.isHidden()


def test_bands_beside_a_dos_ask_for_a_wider_window():
    """Measured at the size the window opens at: the path's corners collide
    along the bottom until 2:1."""
    assert ec.aspect({"mode": el.BANDS_AND_DOS}) == ec.BANDS_AND_DOS_ASPECT == 2.0
    assert ec.aspect({"mode": el.BANDS}) is None             # the window's own 4:3


def test_the_colours_beside_the_plot_are_the_plots_and_show_where_they_apply():
    """The dialog's colours, so a plot can be recoloured where it is drawn;
    a band colour means nothing to a DOS, nor a spin-down one to a
    spin-restricted run. Folded until wanted, as in the dialog."""
    knobs, moves = _knobs({"mode": el.BANDS, "bands_polarised": True,
                           "band_colour": "#112233", "fermi_colour": "#445566"})

    assert moves == []                                    # taken quietly
    options = knobs.options()
    assert (options["band_colour"], options["fermi_colour"]) == ("#112233", "#445566")
    assert not knobs.band_colour.isVisibleTo(knobs)       # folded
    assert not knobs.band_colour.isHidden() and not knobs.beta_colour.isHidden()

    knobs.line_colour.setColour("#000000")
    assert moves and moves[-1]["fermi_colour"] == "#000000"

    dos, _moves = _knobs({"mode": el.DOS})
    assert dos.band_colour.isHidden() and dos.beta_colour.isHidden()
    restricted, _moves = _knobs({"mode": el.BANDS, "bands_polarised": False})
    assert not restricted.band_colour.isHidden() and restricted.beta_colour.isHidden()


def test_the_colours_reach_the_plotter():
    _bands, _dos, options = ec.plot_arguments(
        {"mode": el.BANDS, "band_colour": "#112233", "beta_colour": "#223344",
         "fermi_colour": "#445566"})

    assert (options.band_colour, options.beta_colour, options.fermi_colour) == (
        "#112233", "#223344", "#445566")
