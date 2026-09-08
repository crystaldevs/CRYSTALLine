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

    for name, (label, widget) in dialog._rows.items():
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
