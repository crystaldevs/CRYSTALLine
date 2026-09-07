"""Plot dialogs reopen on the settings they were last accepted with."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QSpinBox,
)

from crystalline.ui.panels.dialog_state import capture, restore  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _Dialog(QDialog):
    """A stand-in with the widget kinds the real plot dialogs are built from."""

    def __init__(self, options=(("a", 1), ("b", 2)), extra=True):
        super().__init__()
        self.mode = QComboBox()
        for label, data in options:
            self.mode.addItem(label, data)
        self.shape = QComboBox()          # no user data: remembered by text
        self.shape.addItems(["lorentz", "gauss"])
        self.width = QDoubleSpinBox()
        self.width.setRange(0.0, 100.0)
        self.states = QSpinBox()
        self.states.setRange(0, 50)
        self.harmonic = QCheckBox()
        if extra:
            self.only_here = QDoubleSpinBox()


def test_settings_survive_a_new_dialog(qapp):
    first = _Dialog()
    first.mode.setCurrentIndex(1)
    first.shape.setCurrentText("gauss")
    first.width.setValue(12.5)
    first.states.setValue(7)
    first.harmonic.setChecked(True)

    second = _Dialog()
    restore(second, capture(first))

    assert second.mode.currentData() == 2
    assert second.shape.currentText() == "gauss"
    assert second.width.value() == pytest.approx(12.5)
    assert second.states.value() == 7
    assert second.harmonic.isChecked()


def test_nothing_remembered_yet_leaves_the_defaults(qapp):
    dialog = _Dialog()
    before = dialog.width.value()
    restore(dialog, None)
    restore(dialog, {})
    assert dialog.width.value() == before


def test_a_combo_is_remembered_by_data_not_by_index(qapp):
    """What a file offers differs between runs, so index 2 of one output is not
    index 2 of the next — restoring by position would silently plot the wrong
    mode."""
    first = _Dialog(options=(("mode 4", 4), ("mode 7", 7), ("mode 9", 9)))
    first.mode.setCurrentIndex(2)  # mode 9
    state = capture(first)

    # the next output offers the same mode in a different position
    second = _Dialog(options=(("mode 9", 9), ("mode 11", 11)))
    restore(second, state)

    assert second.mode.currentData() == 9
    assert second.mode.currentIndex() == 0


def test_an_option_this_file_does_not_offer_is_skipped(qapp):
    """A remembered choice that is gone must leave the dialog on its own default,
    not raise and not blank the control."""
    first = _Dialog(options=(("mode 9", 9),))
    state = capture(first)

    second = _Dialog(options=(("mode 1", 1), ("mode 2", 2)))
    restore(second, state)

    assert second.mode.currentData() == 1  # its own default, unharmed


def test_a_control_that_no_longer_exists_is_skipped(qapp):
    """Dialogs gain and lose controls with the file; a stale key must not raise."""
    state = capture(_Dialog(extra=True))

    lean = _Dialog(extra=False)
    restore(lean, state)  # must not raise

    assert not hasattr(lean, "only_here")


def test_capture_ignores_everything_that_is_not_a_setting(qapp):
    """Only the settings widgets are carried; a tree of what *this* file contains
    is the file's contents, not a preference."""
    state = capture(_Dialog())
    assert set(state) == {"mode", "shape", "width", "states", "harmonic", "only_here"}


def test_the_real_spectra_dialog_round_trips(qapp):
    """Against the actual dialog, through its own options() output."""
    from crystalline.ui.panels.spectra_dialog import SpectraDialog

    first = SpectraDialog([])
    first.hwhm.setValue(12.5)
    first.eta.setValue(0.42)
    first.fmin.setValue(400.0)
    first.fmax.setValue(1800.0)
    first.lineshape.setCurrentIndex(first.lineshape.count() - 1)
    wanted = first.options()

    second = SpectraDialog([])
    assert second.options() != wanted  # a fresh one really does start at defaults
    restore(second, capture(first))

    assert second.options() == wanted
