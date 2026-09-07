"""The anharmonic-scan options.

What makes this plot usable is that nothing in it has to be guessed: the
wavefunction height arrives from the level spacing of the run, and the options
come back named exactly as ``plot_anscan`` takes them, so the dialog is the
contract between the two.
"""

import os

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from crystalline.crystalio.anscan import AnscanRun  # noqa: E402
from crystalline.ui.panels.anscan_dialog import AnscanDialog  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _run(**kwargs):
    """A CO2 stretch: ten drawable states about 555 cm⁻¹ apart."""
    defaults = dict(mode=7, frequency=487.11, nstates=101, nwf=10,
                    rangescan=(-9.0, 9.0), spacing=555.0, double_well=False)
    defaults.update(kwargs)
    return AnscanRun(**defaults)


def test_it_opens_on_the_wavefunctions_at_the_level_spacing(qapp):
    dialog = AnscanDialog(_run())

    options = dialog.options()
    assert options["scale_wf"] == pytest.approx(555.0)
    assert options["scale_prob"] is None      # one curve per level is enough
    assert options["nstates"] == 10
    assert options["scanpot"] is True         # the data the fit is a fit to
    assert options["harmpot"] is False        # asked for, not drawn by default


def test_a_soft_mode_opens_an_order_of_magnitude_lower(qapp):
    """The same defaults on a shallow double well; a height fixed in the code
    would draw its states either invisible or off the top of the figure."""
    dialog = AnscanDialog(_run(mode=1, frequency=-30.03, spacing=39.4,
                               double_well=True))

    assert dialog.options()["scale_wf"] == pytest.approx(39.4)


def test_unchecking_a_curve_drops_it_rather_than_zeroing_it(qapp):
    """``plot_anscan`` reads None as "leave it out"; a height of zero would draw
    a flat line on every level instead."""
    dialog = AnscanDialog(_run())
    dialog.wavefunctions.setChecked(False)
    dialog.densities.setChecked(True)

    options = dialog.options()
    assert options["scale_wf"] is None
    assert options["scale_prob"] == pytest.approx(1110.0)


def test_a_height_is_only_editable_for_a_curve_being_drawn(qapp):
    dialog = AnscanDialog(_run())

    assert dialog.scale_wf.isEnabled()
    assert not dialog.scale_prob.isEnabled()
    dialog.densities.setChecked(True)
    assert dialog.scale_prob.isEnabled()


def test_no_more_states_can_be_asked_for_than_were_written(qapp):
    """Only the states CRYSTAL wrote coefficients for can carry a curve."""
    dialog = AnscanDialog(_run(nwf=4))

    assert dialog.nstates.maximum() == 4
    dialog.nstates.setValue(99)
    assert dialog.options()["nstates"] == 4


def test_the_numbers_are_named_as_the_scale_factors_they_are(qapp):
    """The numbers multiply a normalised curve; naming them after the curve alone
    reads as if they were a property of the state.

    The naming moved: the checkbox now says only what it draws, and the factor
    has its own labelled row beneath it, which is what a slider needs anyway.
    """
    from PySide6.QtWidgets import QLabel

    dialog = AnscanDialog(_run())
    labels = {label.text() for label in dialog.findChildren(QLabel)}

    assert dialog.wavefunctions.text() == "Wavefunctions"
    assert dialog.densities.text() == "Probability densities"
    assert "ψ scale" in labels
    assert "|ψ|² scale" in labels


def test_a_scale_slider_opens_on_the_run_s_own_suggestion(qapp):
    """The scale that suits one mode is nothing like the one that suits another —
    it comes from the level spacing — so the range is anchored on the run's
    suggestion rather than fixed, or most runs would open against a stop."""
    from PySide6.QtWidgets import QSlider

    dialog = AnscanDialog(_run())
    sliders = dialog.findChildren(QSlider)

    assert sliders, "the scale factors should be sliders"
    for slider in sliders[:2]:
        assert 400 < slider.value() < 600   # the suggestion sits mid-track


def test_the_title_names_the_scanned_mode(qapp):
    assert AnscanDialog(_run()).title() == "ANSCAN mode 7"
    assert AnscanDialog(_run(mode=0)).title() == "ANSCAN"
