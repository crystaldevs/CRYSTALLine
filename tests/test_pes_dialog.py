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
