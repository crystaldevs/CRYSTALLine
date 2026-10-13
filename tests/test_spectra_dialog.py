"""The vibrational-spectra picker.

The dialog is where "plot the Raman components" actually happens: several curves
are ticked and overlaid on one axes. It is built from whatever the loaded output
turned out to contain, so it must cope with one curve as gracefully as sixty.
"""

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialogButtonBox  # noqa: E402

from crystalline.crystalio.spectra import _parse  # noqa: E402
from crystalline.ui.panels.spectra_dialog import SpectraDialog  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


_RAMAN_COMPONENTS = ["Ram_HO_0K_tot", "Ram_HO_0K_par", "Ram_HO_0K_per"] + [
    f"Ram_HO_0K_comp_{c}" for c in ("xx", "xy", "xz", "yy", "yz", "zz")
]


def _kinds(names):
    return sorted((_parse(n) for n in names), key=lambda k: k.sort_key)


def _dialog(names):
    return SpectraDialog(_kinds(names))


def test_every_ir_curve_sits_in_one_section(qapp):
    """IR has at most one curve per level and temperature, so splitting it the
    way Raman is split would give a column of single-entry branches."""
    dialog = _dialog(
        ["IR_HO_0K", "IR_HO_T", "IR_VSCF_0K", "IR_VCI_T", *_RAMAN_COMPONENTS,
         "Ram_VCI_T_comp_zz"]
    )

    groups = [dialog.tree.topLevelItem(i).text(0) for i in range(dialog.tree.topLevelItemCount())]
    counts = [dialog.tree.topLevelItem(i).childCount() for i in range(len(groups))]

    assert groups == ["IR", "Raman (harmonic, 0 K)", "Raman (VCI, T)"]
    assert counts == [4, 9, 1]


def test_curves_are_named_by_what_distinguishes_them_in_their_section(qapp):
    dialog = _dialog(["IR_HO_0K", "IR_VCI_T", *_RAMAN_COMPONENTS])

    leaves = [leaf.text(0) for leaf in dialog._leaves()]

    # IR leaves say which level and temperature; Raman leaves say which polarisation
    assert leaves[:2] == ["harmonic, 0 K", "VCI, T"]
    assert leaves[2:5] == ["total", "parallel", "perpendicular"]


def test_nothing_can_be_plotted_until_something_is_ticked(qapp):
    dialog = _dialog(["IR_HO_0K", *_RAMAN_COMPONENTS])
    ok = dialog._buttons.button(QDialogButtonBox.Ok)

    assert dialog.selected_kinds() == []
    assert not ok.isEnabled()

    next(dialog._leaves()).setCheckState(0, Qt.Checked)
    assert ok.isEnabled()


def test_several_components_can_be_selected_at_once(qapp):
    """The whole point of the dialog: xx/yy/zz overlaid on shared axes."""
    dialog = _dialog(_RAMAN_COMPONENTS)

    for leaf in dialog._leaves():
        if leaf.data(0, Qt.UserRole).polarisation in ("xx", "yy", "zz"):
            leaf.setCheckState(0, Qt.Checked)

    assert [k.polarisation for k in dialog.selected_kinds()] == ["xx", "yy", "zz"]


def test_ticking_a_group_takes_all_of_its_curves(qapp):
    dialog = _dialog(["IR_HO_0K", *_RAMAN_COMPONENTS])

    dialog.tree.topLevelItem(1).setCheckState(0, Qt.Checked)  # the Raman branch

    assert len(dialog.selected_kinds()) == 9
    assert all(k.kind == "Raman" for k in dialog.selected_kinds())


def test_a_single_curve_file_arrives_ready_to_plot(qapp):
    """A plain harmonic IR run offers one curve; ticking it to get the only
    thing there is would be busywork."""
    dialog = _dialog(["IR_HO_0K"])

    assert [k.attribute for k in dialog.selected_kinds()] == ["IR_HO_0K"]
    assert dialog._buttons.button(QDialogButtonBox.Ok).isEnabled()


def test_select_all_and_clear(qapp):
    dialog = _dialog(["IR_HO_0K", *_RAMAN_COMPONENTS])

    dialog._select_all()
    assert len(dialog.selected_kinds()) == 10

    dialog._select_none()
    assert dialog.selected_kinds() == []


@pytest.mark.parametrize(
    "lineshape, enabled",
    [
        ("lorentz", {"hwhm"}),
        ("gauss", {"stdev"}),
        ("pvoigt", {"hwhm", "stdev", "eta"}),   # the sum of the other two, plus the mix
        ("bars", set()),
    ],
)
def test_only_the_parameters_a_lineshape_uses_are_offered(qapp, lineshape, enabled):
    """A Lorentzian ignores the Gaussian width and vice versa; only a
    pseudo-Voigt needs both, plus eta. Controls that would do nothing are greyed
    rather than left to mislead."""
    dialog = _dialog(["IR_HO_0K"])
    names = [dialog.lineshape.itemData(i) for i in range(dialog.lineshape.count())]

    dialog.lineshape.setCurrentIndex(names.index(lineshape))

    for name, (label, widget) in dialog.options_widget._rows.items():
        assert widget.isEnabled() is (name in enabled), name
        assert label.isEnabled() is (name in enabled), name
    assert dialog.options()["lineshape"] == lineshape


def test_eta_is_a_fraction_and_defaults_to_an_even_mix(qapp):
    from crystalline.crystalio.spectra import DEFAULT_ETA, DEFAULT_HWHM, DEFAULT_STDEV

    dialog = _dialog(["IR_HO_0K"])

    assert (dialog.eta.minimum(), dialog.eta.maximum()) == (0.0, 1.0)
    assert dialog.eta.value() == DEFAULT_ETA
    # the widths start where plot_cry_spec's own defaults are, and differ
    assert dialog.hwhm.value() == DEFAULT_HWHM
    assert dialog.stdev.value() == DEFAULT_STDEV
    assert DEFAULT_HWHM != DEFAULT_STDEV


def test_options_are_ready_to_pass_to_the_plotter(qapp):
    dialog = _dialog(["IR_HO_0K"])
    dialog.hwhm.setValue(12.0)
    dialog.stdev.setValue(4.0)
    dialog.eta.setValue(0.25)
    dialog.fmin.setValue(400.0)
    dialog.fmax.setValue(1800.0)

    assert dialog.options() == {
        "lineshape": "lorentz",
        "hwhm": 12.0,
        "stdev": 4.0,
        "eta": 0.25,
        "offset": 0.0,                  # flat by default: curves share a baseline
        "legend": True,                 # and named, until that gets in the way
        "linewidth": 2.0,               # plot_cry_spec_multi's own
        "frequency_range": (400.0, 1800.0),
    }

    # an inverted range is read as the user meaning the span between them
    dialog.fmin.setValue(1800.0)
    dialog.fmax.setValue(400.0)
    assert dialog.options()["frequency_range"] == (400.0, 1800.0)

    # ...and a zero-width one as "don't clip"
    dialog.fmax.setValue(1800.0)
    assert dialog.options()["frequency_range"] is None


def test_the_dialog_output_feeds_the_plotter(qapp):
    """The seam between dialog and figure: labels and options go straight through."""
    pytest.importorskip("CRYSTALClear")
    import matplotlib

    matplotlib.use("Agg")
    from crystalline.crystalio.spectra import plot_spectra

    dialog = _dialog(_RAMAN_COMPONENTS)
    dialog._select_all()
    curve = np.array([[500.0, 1.0], [1500.0, 2.0]])

    figure = plot_spectra(
        [(k.label, curve) for k in dialog.selected_kinds()], **dialog.options()
    )

    assert len(figure.axes[0].get_legend().get_texts()) == 9


# ── what was ticked, next time ────────────────────────────────────────────
def test_the_ticked_curves_are_named_and_can_be_put_back(qapp):
    """Reopening the dialog used to come up blank however many curves were
    ticked a moment earlier, which left rebuilding the plot from the menu as
    the only way to change it."""
    names = ["IR_HO_0K", "Ram_HO_0K_tot", "Ram_HO_0K_par"]
    dialog = _dialog(names)
    dialog.set_selection_state(["Ram_HO_0K_tot"])

    assert dialog.selection_state() == ["Ram_HO_0K_tot"]
    assert [k.attribute for k in dialog.selected_kinds()] == ["Ram_HO_0K_tot"]

    again = _dialog(names)
    again.set_selection_state(dialog.selection_state())
    assert again.selection_state() == ["Ram_HO_0K_tot"]


def test_a_curve_this_output_does_not_hold_is_skipped(qapp):
    """The objection that kept ticks out of the remembered state: another run
    holds other curves. Answered by naming them, as a combo's choice is."""
    dialog = _dialog(["IR_HO_0K"])

    dialog.set_selection_state(["Ram_VCI_T_comp_zz", "IR_HO_0K"])

    assert dialog.selection_state() == ["IR_HO_0K"]


def test_ticking_nothing_leaves_ok_disabled(qapp):
    from PySide6.QtWidgets import QDialogButtonBox

    dialog = _dialog(["IR_HO_0K", "Ram_HO_0K_tot"])
    dialog.set_selection_state(["IR_HO_0K"])
    assert dialog._buttons.button(QDialogButtonBox.Ok).isEnabled()

    dialog.set_selection_state([])

    assert not dialog._buttons.button(QDialogButtonBox.Ok).isEnabled()


def test_the_offset_is_a_fraction_so_one_slider_suits_every_spectrum(qapp):
    """What an intensity means differs between IR and Raman, so the control is
    a fraction of the tallest peak rather than an intensity of its own."""
    dialog = _dialog(["Ram_HO_0K_tot", "Ram_HO_0K_par"])

    dialog.options_widget.offset.setValue(0.4)

    assert dialog.options()["offset"] == 0.4
    assert dialog.options_widget.offset.minimum() == 0.0   # flat is reachable
    # ...and at the other end each curve sits one tallest peak above the last,
    # clear of it: past that, the slider would only push them off the figure.
    assert dialog.options_widget.offset.maximum() == 1.0


def test_the_legend_is_on_until_it_is_turned_off(qapp):
    """It names the curves, which is what you want until the figure is crowded
    enough that the box hides the peaks it is naming."""
    dialog = _dialog(["Ram_HO_0K_tot", "Ram_HO_0K_par"])
    assert dialog.options()["legend"] is True

    dialog.options_widget.legend.setChecked(False)

    assert dialog.options()["legend"] is False


def test_it_opens_at_the_same_size_as_the_other_chooser_dialogs(qapp):
    """They are the same kind of thing — a list of what this run holds, with
    the controls that say how to draw it underneath — so they are sized by one
    rule and open alike unless their rows genuinely differ."""
    from crystalline.crystalio.pes import PESMode, PESRun
    from crystalline.ui.panels import controls
    from crystalline.ui.panels.pes_dialog import PESDialog

    spectra = _dialog(["IR_HO_0K", "Ram_HO_0K_tot", "Ram_HO_0K_par"])
    pes = PESDialog(PESRun(
        modes=(PESMode(mode=7, frequency=1007.0, eta3=-1.0, eta4=2.0),),
        pairs=(), ntriplets=0))

    assert spectra.width() == pes.width() == controls.CHOOSER_MIN_WIDTH
    # The heights differ, and should: each is its own controls' needs plus the
    # same ten rows of its own list, and the controls differ by design — the
    # spectra have an Appearance section the PES does not. What must not
    # differ is the rule, so that is what is checked.
    pes_list = max((w for w in vars(pes).values() if hasattr(w, "sizeHintForRow")),
                   key=lambda w: w.sizeHintForRow(0))
    for dialog, listing in ((spectra, spectra.tree), (pes, pes_list)):
        rows = controls.CHOOSER_ROWS * listing.sizeHintForRow(0)
        assert dialog.height() == dialog.sizeHint().height() + rows
    assert spectra.height() <= controls.CHOOSER_MAX_HEIGHT
    assert spectra.tree.height() >= 180                       # never below its floor


# ── stacking and naming, only for several curves ──────────────────────────
def test_curves_are_offered_only_once_two_are_ticked(qapp):
    """One curve has nothing to be raised above and nothing to be told apart
    from. Hidden rather than greyed, as a PES cut's Surface section is."""
    dialog = _dialog(["Ram_HO_0K_tot", "Ram_HO_0K_par", "Ram_HO_0K_per"])
    knobs = dialog.options_widget
    dialog.show()
    leaves = list(dialog._leaves())

    assert not knobs.offset.isVisible() and not knobs.legend.isVisible()   # nothing ticked

    leaves[0].setCheckState(0, Qt.Checked)
    assert not knobs.offset.isVisible()                                     # one curve

    leaves[1].setCheckState(0, Qt.Checked)
    assert knobs.offset.isVisible() and knobs.legend.isVisible()           # two

    leaves[1].setCheckState(0, Qt.Unchecked)
    assert not knobs.offset.isVisible()                                     # one again
    dialog.hide()


def test_a_file_with_one_curve_never_offers_them(qapp):
    """Ticked from the start, since there is nothing else; still one curve."""
    dialog = _dialog(["IR_HO_0K"])
    dialog.show()

    assert not dialog.options_widget.offset.isVisible()
    dialog.hide()


def test_beside_a_plot_they_follow_the_curves_it_was_drawn_with(qapp):
    """The same controls stand next to the plot, told what it shows by the
    options it was drawn with."""
    from crystalline.ui.panels.spectra_options import SpectraOptions

    kinds = _kinds(["Ram_HO_0K_tot", "Ram_HO_0K_par"])
    knobs = SpectraOptions()
    knobs.show()

    knobs.set_options({"curves": kinds[:1], "lineshape": "lorentz"})
    assert not knobs.offset.isVisible()

    knobs.set_options({"curves": kinds, "lineshape": "lorentz"})
    assert knobs.offset.isVisible() and knobs.legend.isVisible()
    knobs.hide()


# ── appearance ────────────────────────────────────────────────────────────
def test_the_line_width_reaches_the_curves(qapp):
    """Under Appearance, as on a band structure, and for one curve as much as
    for several: a single spectrum still has a line."""
    pytest.importorskip("CRYSTALClear")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from crystalline.crystalio.spectra import plot_spectra

    dialog = _dialog(["IR_HO_0K"])
    dialog.show()
    assert dialog.options_widget.linewidth.isVisible()
    assert not dialog.options_widget.legend.isVisible()     # one curve: nothing to name
    dialog.hide()

    dialog.options_widget.linewidth.setValue(0.8)
    figure = plot_spectra([("IR", np.array([[500.0, 1.0]]))], **dialog.options())
    widths = {line.get_linewidth() for line in figure.axes[0].lines}
    plt.close(figure)

    assert widths == {0.8}


def test_reopening_the_dialog_keeps_the_stacking_the_legend_and_the_width(qapp):
    """They were not among what the dialog remembers, so every reopening —
    an Edit included — put them back to their defaults."""
    from crystalline.ui.panels.dialog_state import capture, restore

    names = ["Ram_HO_0K_tot", "Ram_HO_0K_par"]
    first = _dialog(names)
    first.options_widget.offset.setValue(0.6)
    first.options_widget.legend.setChecked(False)
    first.options_widget.linewidth.setValue(1.2)

    again = _dialog(names)
    restore(again, capture(first))

    assert again.options()["offset"] == pytest.approx(0.6)
    assert again.options()["legend"] is False
    assert again.options()["linewidth"] == pytest.approx(1.2)



def test_opening_the_spectra_reads_the_output_once():
    """The list came from one read and the plot from a second after OK: the
    whole output parsed twice, with the window frozen for both — 560 ms on a
    3.5 MB anharmonic run."""
    import inspect

    from crystalline.ui.main_window import MainWindow

    source = inspect.getsource(MainWindow._open_spectra)
    assert source.count("load_spectra(") == 1
    assert "available_spectra(" not in source
