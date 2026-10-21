"""The PES cut picker.

The options it returns feed ``plot_pes`` as keyword arguments, so their names
and types are the contract between the two. The behaviour worth pinning is what
happens at the edges: a run with nothing to couple must not offer a two-mode
map, and the pair filter has to actually narrow the list — it is what makes 630
pairs usable.
"""

import os

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDialogButtonBox  # noqa: E402

from crystalline.crystalio.pes import PESMode, PESPair, PESRun  # noqa: E402
from crystalline.ui.panels.pes_dialog import PESDialog  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _pair(modei, modej, iijj=0.0):
    return PESPair(modei=modei, modej=modej,
                   frequencyi=1000.0 + modei, frequencyj=1000.0 + modej,
                   iij=0.0, ijj=0.0, iiij=0.0, ijjj=0.0, iijj=iijj)


def _run(pairs=None):
    modes = tuple(PESMode(mode=m, frequency=1000.0 + m, eta3=-1.0, eta4=2.0)
                  for m in (7, 8, 12))
    if pairs is None:
        # Deliberately not in strength order, as pes_run would deliver them.
        pairs = (_pair(7, 12, iijj=300.0), _pair(7, 8, iijj=40.0),
                 _pair(8, 12, iijj=4.0))
    return PESRun(modes=modes, pairs=pairs, ntriplets=0)


def test_it_opens_on_a_one_mode_cut_of_the_first_mode(qapp):
    options = PESDialog(_run()).options()

    assert options["dimension"] == "1D"
    assert options["mode"] == 7
    assert options["span"] == pytest.approx(2.0)
    assert options["harmonic"] is True
    assert options["levels"] is False       # solving is asked for, not assumed


def test_a_two_mode_cut_maps_the_anharmonic_part_by_default(qapp):
    """Contours of the total surface are ellipses — the harmonic bowl is orders
    of magnitude deeper than anything the constants add."""
    dialog = PESDialog(_run())
    dialog.dimension.setCurrentIndex(1)

    options = dialog.options()
    assert options["dimension"] == "2D"
    assert options["quantity"] == "anharmonic"
    assert options["representation"] == "map"   # the one that can be read off
    assert (options["modei"], options["modej"]) == (7, 12)   # strongest first


def test_a_pair_can_be_asked_for_as_a_surface(qapp):
    dialog = PESDialog(_run())
    dialog.dimension.setCurrentIndex(1)
    dialog.representation.setCurrentIndex(
        dialog.representation.findData("surface"))

    assert dialog.options()["representation"] == "surface"
    # The tab has to say which of the two it is, or a map and a surface of the
    # same pair land as two tabs with one name.
    assert dialog.title() == "PES 7 × 12 3D"


def test_a_run_with_nothing_coupled_cannot_ask_for_a_map(qapp):
    """A single-mode scan has no pairs; the entry has to be unreachable rather
    than fail once it is picked."""
    dialog = PESDialog(_run(pairs=()))

    assert not dialog.dimension.model().item(1).isEnabled()


def test_the_filter_narrows_the_pairs_to_one_mode(qapp):
    dialog = PESDialog(_run())
    dialog.dimension.setCurrentIndex(1)
    assert dialog.pairs.count() == 3

    dialog.filter.setCurrentIndex(dialog.filter.findData(12))
    assert dialog.pairs.count() == 2
    dialog.pairs.setCurrentRow(1)
    modei, modej = dialog.options()["modei"], dialog.options()["modej"]
    assert 12 in (modei, modej)


def test_a_filter_that_leaves_no_pair_blocks_the_plot(qapp):
    dialog = PESDialog(_run(pairs=(_pair(7, 8),)))
    dialog.dimension.setCurrentIndex(1)
    dialog.filter.setCurrentIndex(dialog.filter.findData(12))

    assert dialog.pairs.count() == 0
    assert not dialog._buttons.button(QDialogButtonBox.Ok).isEnabled()


def test_the_state_count_is_only_editable_when_states_are_drawn(qapp):
    dialog = PESDialog(_run())

    assert not dialog.nstates.isEnabled()
    dialog.levels.setChecked(True)
    assert dialog.nstates.isEnabled()
    assert dialog.options()["levels"] is True


def test_the_title_names_the_cut(qapp):
    dialog = PESDialog(_run())
    assert dialog.title() == "PES mode 7"

    dialog.dimension.setCurrentIndex(1)
    assert dialog.title() == "PES 7 × 12"


# ── the knobs that sit beside the plot ────────────────────────────────────
def test_the_dialog_and_the_plot_carry_the_same_controls(qapp):
    """One widget, hosted twice: tuned in the dialog or tuned beside the plot,
    it is the same control and cannot drift."""
    from crystalline.ui.panels.pes_options import PESOptions

    dialog = PESDialog(_run())

    assert isinstance(dialog.options_widget, PESOptions)
    assert dialog.span is dialog.options_widget.span
    assert dialog.levels is dialog.options_widget.levels


def test_a_cut_puts_away_the_knobs_it_has_no_use_for(qapp):
    """A one-mode cut has overlays and no surface; a two-mode map the reverse.
    Hidden rather than greyed: a section that cannot apply at all still asks
    to be read before it can be dismissed."""
    from crystalline.ui.panels.pes_options import PESOptions

    knobs = PESOptions()
    knobs.show()

    knobs.set_options({"dimension": "1D", "levels": True})
    assert knobs.harmonic.isVisible() and knobs.nstates.isVisible()
    assert not knobs.representation.isVisible()

    knobs.set_options({"dimension": "2D", "levels": True})
    assert knobs.representation.isVisible() and knobs.quantity.isVisible()
    assert not knobs.harmonic.isVisible() and not knobs.nstates.isVisible()
    knobs.hide()


def test_turning_a_knob_says_so_once(qapp):
    from crystalline.ui.panels.pes_options import PESOptions

    knobs = PESOptions()
    said = []
    knobs.changed.connect(lambda: said.append(knobs.options()))

    knobs.span.setValue(4.0)

    assert [o["span"] for o in said] == [4.0]


def test_the_widget_answers_with_plot_pes_keywords(qapp):
    from crystalline.ui.panels.pes_options import PESOptions

    options = PESOptions().options()

    assert set(options) == {"span", "harmonic", "levels", "nstates", "scale_wf",
                            "legend", "linewidth", "quantity", "representation"}


def test_the_wavefunction_height_is_offered_only_while_states_are_solved(qapp):
    """They are drawn on the potential's own axes, in cm⁻¹: left to itself a
    wavefunction is either a flat line or runs off the top."""
    from crystalline.ui.panels.pes_options import PESOptions

    knobs = PESOptions()
    knobs.show()
    knobs.set_options({"dimension": "1D", "levels": False})
    assert not knobs.scale_wf.isEnabled()

    knobs.levels.setChecked(True)
    assert knobs.scale_wf.isEnabled()

    knobs.scale_wf.setValue(400.0)
    assert knobs.options()["scale_wf"] == 400.0
    knobs.hide()


def test_the_scale_opens_at_the_run_s_own_level_spacing(qapp):
    """A wavefunction is read against the spacing of the levels it belongs to,
    so that is where the slider starts — and it spans a decade either side,
    since a height is judged by ratio."""
    from crystalline.ui.panels.pes_options import PESOptions

    run = _run()
    assert run.scale_wf == 1008.0            # the middle mode of 1007/1008/1012

    knobs = PESOptions(run.scale_wf)

    assert knobs.scale_wf.value() == pytest.approx(1008.0)
    assert knobs.scale_wf.minimum() == pytest.approx(100.8)
    assert knobs.scale_wf.maximum() == pytest.approx(10080.0)


def test_a_run_that_suggests_nothing_still_opens_somewhere_usable(qapp):
    from crystalline.crystalio.pes import PESRun
    from crystalline.ui.panels.pes_options import PESOptions

    knobs = PESOptions(PESRun(modes=(), pairs=(), ntriplets=0).scale_wf)

    assert knobs.scale_wf.value() > 0


def test_the_chosen_cut_is_remembered_by_its_modes(qapp):
    """Reopening the dialog to find mode 7 selected again, not row 1 of
    whatever this run happens to hold."""
    from crystalline.ui.panels.dialog_state import capture, restore

    first = PESDialog(_run())
    first.modes.setCurrentRow(2)                    # mode 12
    first.filter.setCurrentIndex(1)
    chosen = first.selection_state()
    assert chosen["mode"] == 12

    again = PESDialog(_run())
    restore(again, capture(first))

    assert again.selection_state()["mode"] == 12
    assert again.options()["mode"] == 12


def test_a_mode_this_run_does_not_have_is_skipped(qapp):
    """Another run scanned other modes; asking for one it lacks must leave the
    dialog usable rather than empty or raising."""
    from crystalline.crystalio.pes import PESMode, PESRun

    other = PESRun(modes=(PESMode(mode=3, frequency=900.0, eta3=-1.0, eta4=2.0),),
                   pairs=(), ntriplets=0)
    dialog = PESDialog(other)

    dialog.set_selection_state({"mode": 12, "pair": [7, 12], "filter": None})

    assert dialog.options()["mode"] == 3            # the row it already had


def test_the_dialog_opens_wide_enough_for_the_rows_it_must_show(qapp):
    """A row reads "Mode 12 — 888.5 cm⁻¹ (η₃ -377.9, η₄ +838.1 cm⁻¹)", and how
    wide that is depends on the run: sized to the widest of them, it fits on
    one line instead of under a horizontal scrollbar."""
    from crystalline.crystalio.pes import PESMode, PESRun
    from crystalline.ui.panels import controls as module

    long_labels = PESRun(
        modes=tuple(PESMode(mode=m, frequency=f, eta3=a, eta4=b) for m, f, a, b in
                    ((12, 888.5, -377.9, 838.1), (14, 3767.0, -1871.7, 787.8))),
        pairs=(), ntriplets=0)
    dialog = PESDialog(long_labels)
    dialog.show()

    assert module.CHOOSER_MIN_WIDTH <= dialog.width() <= module.CHOOSER_MAX_WIDTH
    assert dialog.width() >= dialog.modes.sizeHintForColumn(0)   # no sideways scrolling
    assert dialog.height() <= module.CHOOSER_MAX_HEIGHT
    assert dialog.modes.height() >= 180
    dialog.hide()


def test_switching_the_cut_does_not_resize_the_window(qapp):
    """It used to call adjustSize() on every change, which threw away whatever
    size the window had — the default among them."""
    dialog = PESDialog(_run())
    dialog.show()
    was = dialog.size()

    dialog.dimension.setCurrentIndex(1)      # one mode -> a pair
    qapp.processEvents()

    # Not shrunk back to its hint, which is what adjustSize() did. It may grow
    # by a few pixels: the pair picker's own minimum is a little taller than
    # the mode list's, and a layout minimum is not something to fight.
    assert dialog.width() == was.width()
    assert dialog.height() >= was.height()
    dialog.hide()


def test_the_legend_tick_sits_under_appearance_as_on_every_plot(qapp):
    """On by default, as on a spectrum, and with the line width under
    Appearance, as on a spectrum. Both are about lines drawn over the
    potential, so the section goes for a pair cut — a contour map is read
    against its colourbar, and its plotter takes no legend at all."""
    from crystalline.ui.panels.pes_options import PESOptions

    knobs = PESOptions(1000.0)
    knobs.show()
    knobs._look.header.click()          # unfolded: it opens folded
    assert knobs.options()["legend"] is True

    knobs.set_options({"dimension": "2D"})
    assert not knobs.legend.isVisible() and not knobs._look.header.isVisible()

    knobs.set_options({"dimension": "1D", "legend": False, "linewidth": 3.0})
    assert knobs.legend.isVisible() and knobs.options()["legend"] is False
    assert knobs.options()["linewidth"] == 3.0
    knobs.hide()


def test_the_line_width_sets_the_potential_and_keeps_the_states_finer():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure

    from crystalline.crystalio.pes import scale_line_widths

    figure = Figure()
    ax = figure.add_subplot()
    potential, = ax.plot([0, 1], [0, 1], linewidth=2.0)
    state, = ax.plot([0, 1], [1, 0], linewidth=1.0)
    level = ax.add_collection(LineCollection([[(0, 0.5), (1, 0.5)]], linewidths=0.3))

    scale_line_widths(ax, 3.0)
    assert potential.get_linewidth() == 3.0
    assert state.get_linewidth() == 1.5
    assert level.get_linewidths()[0] == pytest.approx(0.45)
