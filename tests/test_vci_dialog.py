"""The VCI representation picker.

The dialog is what makes the two VCI plots usable: a run holds one state per
configuration, so the defaults it hands over — and which options it shows for
which representation — are what decide whether the figure says anything. The
options it returns feed ``plot_vci`` as keyword arguments, so their names and
types are part of the contract.
"""

import os

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QDialogButtonBox,
    QLabel,
)

from crystalline.crystalio.vci import MAX_STATES, VCIRun  # noqa: E402
from crystalline.ui.panels.vci_dialog import VCIDialog  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _blocked(nstates=200):
    """A symmetry-blocked VCI@VSCF run, like an ice or CO2 crystal.

    States every 20 cm⁻¹, cycling through four irreps, so a window of a known
    width holds a known number of them.
    """
    levels = tuple((20.0 * i, 1 + i % 4) for i in range(nstates))
    return VCIRun(basis="VSCF", nstates=nstates, nconfs=nstates, nmodes=33,
                  irreps=(1, 2, 3, 4), modes=tuple(range(4, 37)), zpe=21801.15,
                  levels=levels)


def _molecular(nstates=121, modes=(6, 7, 8, 9)):
    """An unblocked VCI@HO run, like an isolated molecule."""
    levels = tuple((30.0 * i, 0) for i in range(nstates))
    return VCIRun(basis="HO", nstates=nstates, nconfs=nstates, nmodes=4,
                  irreps=(), modes=modes, zpe=2549.67, levels=levels)


def test_options_are_plot_vci_keyword_arguments(qapp):
    """What comes back is splatted straight into plot_vci, so the two have to
    agree on every name."""
    import inspect

    from crystalline.crystalio.vci import plot_vci

    dialog = VCIDialog(_molecular())
    accepted = set(inspect.signature(plot_vci).parameters) - {"out"}

    assert set(dialog.options()) <= accepted


def test_the_symmetry_block_choice_is_offered_only_when_there_are_blocks(qapp):
    """States of different irreps do not mix, so one block at a time is a real
    reading — but only for a run whose matrix was actually blocked."""
    blocked = VCIDialog(_blocked())
    molecular = VCIDialog(_molecular())

    assert blocked.irrep.isEnabled()
    assert [blocked.irrep.itemData(i) for i in range(blocked.irrep.count())] == [
        None, 1, 2, 3, 4
    ]
    assert not molecular.irrep.isEnabled()
    # "All" is still the default: the low-lying states across the blocks are the
    # usual first look at a run.
    assert blocked.options()["irrep"] is None


def test_the_window_opens_on_the_bottom_of_the_spectrum(qapp):
    """A run spans thousands of states; opening on all of them would ask for a
    figure nobody wants."""
    run = _molecular()
    dialog = VCIDialog(run)

    assert dialog.window() == run.default_window()
    assert 0 < run.count_in(*dialog.window()) <= MAX_STATES


def test_a_window_given_backwards_still_reads_low_to_high(qapp):
    dialog = VCIDialog(_molecular())
    dialog.fmin.setValue(900.0)
    dialog.fmax.setValue(300.0)

    assert dialog.window() == (300.0, 900.0)
    assert dialog.options()["frange"] == (300.0, 900.0)


def test_an_empty_window_cannot_be_accepted(qapp):
    """Drawing nothing is not a useful answer, so OK goes away and the label
    says why."""
    run = _molecular()
    dialog = VCIDialog(run)
    beyond = run.energies[-1] + 500.0
    dialog.fmin.setValue(beyond)
    dialog.fmax.setValue(beyond + 100.0)

    assert not dialog._buttons.button(QDialogButtonBox.Ok).isEnabled()
    assert "no states" in dialog._count.text()


def test_too_wide_a_window_cannot_be_accepted_either(qapp):
    dialog = VCIDialog(_blocked())
    dialog.fmin.setValue(0.0)
    dialog.fmax.setValue(4000.0)  # every state of the run

    assert not dialog._buttons.button(QDialogButtonBox.Ok).isEnabled()
    assert "narrow the window" in dialog._count.text()


def test_the_count_follows_the_symmetry_block(qapp):
    """It is the number of columns that will be drawn, so restricting to one
    irrep has to bring it down."""
    dialog = VCIDialog(_blocked())
    dialog.fmin.setValue(0.0)
    dialog.fmax.setValue(400.0)  # 21 states, spread over four irreps

    across = dialog._count.text()
    dialog.irrep.setCurrentIndex(1)  # irrep 1
    within = dialog._count.text()

    assert across != within
    assert dialog._buttons.button(QDialogButtonBox.Ok).isEnabled()
    assert "6 states" in within  # states 0, 80, 160, 240, 320, 400


def test_only_the_chosen_representation_shows_its_options(qapp):
    """The sign of a coefficient means nothing for a ribbon width, and a ribbon
    width means nothing on a heatmap; leaving both on screen would imply they
    apply to whatever is selected."""
    dialog = VCIDialog(_molecular())

    dialog.representation.setCurrentIndex(
        [dialog.representation.itemData(i) for i in
         range(dialog.representation.count())].index("map")
    )
    assert dialog.signed.isVisibleTo(dialog) and dialog.annotate.isVisibleTo(dialog)
    assert not dialog.weight.isVisibleTo(dialog)

    dialog.representation.setCurrentIndex(
        [dialog.representation.itemData(i) for i in
         range(dialog.representation.count())].index("sankey")
    )
    assert dialog.weight.isVisibleTo(dialog)
    assert not dialog.signed.isVisibleTo(dialog) and not dialog.annotate.isVisibleTo(dialog)


def test_the_recovered_modes_are_passed_through(qapp):
    """The plots relabel configurations with these, so they have to survive the
    trip through the dialog untouched."""
    dialog = VCIDialog(_molecular(modes=(6, 7, 8, 9)))

    assert dialog.options()["modes"] == (6, 7, 8, 9)


def test_a_run_whose_modes_are_unknown_says_so(qapp):
    """Falling back to numbering from 1 is fine, but silently showing mode
    numbers that don't match the Phonons dock would not be."""
    warned = VCIDialog(_molecular(modes=()))
    quiet = VCIDialog(_molecular(modes=(6, 7, 8, 9)))

    def labels(dialog):
        return [label.text() for label in dialog.findChildren(QLabel)]

    assert any("labelled from 1" in text for text in labels(warned))
    assert not any("labelled from 1" in text for text in labels(quiet))
    assert warned.options()["modes"] == ()


def test_the_title_names_the_flavour_and_the_representation(qapp):
    """It becomes the tab label, and both VCI@HO and VCI@VSCF figures can be
    open at once."""
    dialog = VCIDialog(_blocked())

    title = dialog.title()
    assert "VCI@VSCF" in title
    assert dialog.representation.currentText().lower() in title


def test_the_defaults_are_the_documented_ones(qapp):
    from crystalline.crystalio.vci import DEFAULT_THRESHOLD

    run = _blocked()
    options = VCIDialog(run).options()

    assert options["frange"] == run.default_window()
    assert options["threshold"] == pytest.approx(DEFAULT_THRESHOLD)
    assert options["representation"] == "map"
    assert options["signed"] is False
    assert options["weight"] == "square"


# ── beside the plot ───────────────────────────────────────────────────────
def _knobs(run):
    from crystalline.ui.panels.vci_options import VCIOptions

    knobs = VCIOptions(run)
    moves = []
    knobs.changed.connect(lambda: moves.append(knobs.options()))
    return knobs, moves


def test_beside_a_plot_the_window_is_pulled_and_drawn(qapp):
    """Which states are drawn is found by looking, so the window is a live
    control like any other: each move that can be drawn asks for a redraw."""
    knobs, moves = _knobs(_molecular())

    assert knobs.window() == (0.0, 300.0)       # where it opens: 11 states
    knobs.fmax.setValue(450.0)                  # 16

    assert moves and moves[-1]["frange"] == (0.0, 450.0)
    assert "16 states" in knobs.count.text()


def test_a_window_that_cannot_be_drawn_holds_the_plot_back(qapp):
    """Said in the count line and the plot left as it was, rather than redrawn
    into an error at every step of the slider on its way back."""
    knobs, moves = _knobs(_molecular())
    knobs.fmin.setValue(0.0)
    moves.clear()

    knobs.fmax.setValue(30.0 * (MAX_STATES + 10))      # far too many
    knobs.threshold.setValue(0.1)                       # still too many: nothing
    assert moves == []
    assert "narrow the window" in knobs.count.text()

    knobs.fmax.setValue(300.0)                          # back in range
    assert moves and moves[-1]["threshold"] == pytest.approx(0.1)


def test_the_knobs_take_a_plot_s_options_without_announcing_them(qapp):
    """The plot already has them: echoing them back would redraw it for
    nothing, every time it was opened or edited."""
    knobs, moves = _knobs(_blocked())
    wanted = dict(knobs.options(), representation="sankey", frange=(100.0, 300.0),
                  irrep=2, threshold=0.05, weight="abs")

    knobs.set_options(wanted)

    assert moves == []
    assert knobs.options() == wanted
    assert knobs.weight.isVisibleTo(knobs) and not knobs.signed.isVisibleTo(knobs)


def test_the_tab_is_named_by_what_is_drawn(qapp):
    """The representation can be switched beside the plot, so the tab's name
    is worked out from the plot's options, not kept from the dialog."""
    from crystalline.ui.panels.vci_options import title

    run = _molecular()

    assert title(run, {"representation": "map"}).endswith("coefficient map")
    assert title(run, {"representation": "sankey"}).endswith("state composition (sankey)")
