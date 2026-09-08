"""Reading a run's VCI wavefunctions, and routing them to the right plot.

Two things are worth pinning here. The cheap probe that enables the menu entry
has to agree with the real parse, or the entry lies about what a file holds. And
the VCI-active modes are recovered from the PES scan rather than assumed: the
configuration list is written over a dense 1..nmodes range, so getting this wrong
silently mislabels every configuration with a plausible-looking mode number.
"""

import glob
import os

import pytest

from crystalline.crystalio.vci import (
    PLOT_FUNCTIONS,
    REPRESENTATIONS,
    VCIRun,
    _vci_modes,
    has_vci,
    load_vci,
    vci_run,
)

_BANNER = "VIBRATIONAL CONFIGURATION INTERACTION (VCI)"


# ── the cheap probe ─────────────────────────────────────────────────────
def test_a_file_with_the_banner_and_a_state_carries_vci(tmp_path):
    path = tmp_path / "run.out"
    path.write_text(f" * {_BANNER} *\n VCI STATE (     1)  ENE - ZPE:  0.00 CM-1\n")

    assert has_vci(str(path))


def test_a_run_that_set_vci_up_but_printed_no_state_does_not_count(tmp_path):
    """A calculation can reach the VCI step and be cut short; offering the plot
    for it would open a dialog with nothing behind it."""
    path = tmp_path / "run.out"
    path.write_text(f" * {_BANNER} *\n LIST OF CONFIGURATIONS CONSIDERED:\n")

    assert not has_vci(str(path))


def test_a_harmonic_run_does_not_count(tmp_path):
    path = tmp_path / "run.out"
    path.write_text(" FREQUENCIES COMPUTED ON A FRAGMENT\n MODES  1-  3\n")

    assert not has_vci(str(path))


@pytest.mark.parametrize("path", [None, "", "/nonexistent/run.out"])
def test_no_path_and_no_file_are_not_errors(path):
    assert not has_vci(path)


def test_a_missing_file_yields_no_run():
    assert vci_run("/nonexistent/run.out") == (None, None)


# ── recovering the VCI-active modes ─────────────────────────────────────
def test_modes_come_from_the_pes_scan(tmp_path):
    """The scan names its modes by CRYSTAL index, which is what the plots need
    to label configurations the way the Phonons dock numbers them."""
    path = tmp_path / "run.out"
    path.write_text(
        " COUPLE OF MODES:     4    5  (FREQs:     58.50 CM-1,     61.80 CM-1)\n"
        " COUPLE OF MODES:     4    6  (FREQs:     58.50 CM-1,    210.13 CM-1)\n"
        " COUPLE OF MODES:     5    6  (FREQs:     61.80 CM-1,    210.13 CM-1)\n"
    )

    assert _vci_modes(str(path), nmodes=3) == (4, 5, 6)


def test_modes_are_refused_when_they_do_not_span_the_configurations(tmp_path):
    """Fewer scanned modes than the configurations are wide means the scan was
    restricted; labelling from a partial list would be worse than not at all."""
    path = tmp_path / "run.out"
    path.write_text(" COUPLE OF MODES:     4    5  (FREQs: 58.50 CM-1, 61.80 CM-1)\n")

    assert _vci_modes(str(path), nmodes=12) == ()


def test_a_run_with_no_couples_yields_no_modes(tmp_path):
    """A single-mode VCI couples nothing, so there is nothing to recover."""
    path = tmp_path / "run.out"
    path.write_text(" MODE:   1  LEVELS:\n")

    assert _vci_modes(str(path), nmodes=1) == ()


# ── the registry the dialog is built from ───────────────────────────────
def test_every_representation_names_a_plot_function():
    keys = {key for _label, key in REPRESENTATIONS}

    assert keys == set(PLOT_FUNCTIONS)


def test_the_summary_reads_as_a_description_of_the_run():
    run = VCIRun(basis="VSCF", nstates=25125, nconfs=25125, nmodes=33,
                 irreps=(1, 2, 3, 4), modes=tuple(range(4, 37)), zpe=21801.15)

    assert run.label == "VCI@VSCF"
    summary = run.summary
    assert "VCI@VSCF" in summary and "25125 states" in summary
    assert "4 symmetry blocks" in summary and "21801.2 cm⁻¹" in summary


def test_an_unblocked_run_does_not_mention_symmetry():
    run = VCIRun(basis="HO", nstates=121, nconfs=121, nmodes=4,
                 irreps=(), modes=(6, 7, 8, 9), zpe=2549.67)

    assert "symmetry" not in run.summary
    assert run.label == "VCI@HO"


# ── the frequency window ────────────────────────────────────────────────
def _run(levels):
    return VCIRun(basis="VSCF", nstates=len(levels), nconfs=len(levels), nmodes=3,
                  irreps=tuple(sorted({b for _e, b in levels if b})), modes=(),
                  zpe=None, levels=tuple(levels))


def test_a_window_counts_the_states_between_its_edges_inclusive():
    run = _run([(0.0, 1), (100.0, 2), (200.0, 1), (300.0, 2)])

    assert run.count_in(0.0, 200.0) == 3      # both edges are included
    assert run.count_in(100.0, 100.0) == 1
    assert run.count_in(400.0, 500.0) == 0
    assert run.count_in(200.0, 0.0) == 3      # given the wrong way round


def test_a_window_counts_within_one_symmetry_block():
    """The count is what the dialog shows, so it has to match the number of
    columns the plot will actually draw — irrep filter included."""
    run = _run([(0.0, 1), (100.0, 2), (200.0, 1), (300.0, 2)])

    assert run.count_in(0.0, 300.0, irrep=1) == 2
    assert run.count_in(0.0, 300.0, irrep=2) == 2
    assert run.count_in(0.0, 150.0, irrep=1) == 1


def test_the_default_window_opens_on_the_bottom_of_the_spectrum():
    run = _run([(0.0, 0), (92.2, 0), (302.9, 0), (1500.0, 0)])

    low, high = run.default_window(states=3)

    assert low == 0.0
    # rounded up to a round number, so the third state is inside it
    assert high >= 302.9 and high % 50 == 0
    assert run.count_in(low, high) == 3


def test_the_default_window_of_a_run_with_no_levels_is_still_usable():
    """vci_run always fills the levels in, but a VCIRun built by hand (as the
    dialog tests do) need not, and must not divide by zero."""
    low, high = VCIRun("HO", 0, 0, 0, (), (), None).default_window()

    assert high > low


# ── against real outputs, when this machine has any ─────────────────────
_VCI_OUTS = [
    p for p in sorted(
        glob.glob(os.path.expanduser(
            "~/Desktop/PyCrystal/anharmonic_freq/CO2_molecule/*/co2_anh_*.out"))
    )
    if has_vci(p)
]


@pytest.mark.skipif(not _VCI_OUTS, reason="no sample VCI .out available")
def test_the_probe_agrees_with_the_real_parse():
    pytest.importorskip("CRYSTALClear")

    run, out = vci_run(_VCI_OUTS[0])

    assert run is not None and out is not None
    assert run.nstates == len(out.VCI_energy)
    assert run.basis in ("HO", "VSCF")
    # The configurations are as wide as there are VCI-active modes, and the
    # recovered mode list — when there is one — matches that width.
    assert run.nmodes == out.VCI_list_conf.shape[1]
    assert run.modes == () or len(run.modes) == run.nmodes
    # One level per state, ascending, so a window can be counted against them
    assert len(run.levels) == run.nstates
    assert list(run.energies) == sorted(run.energies)


@pytest.mark.skipif(not _VCI_OUTS, reason="no sample VCI .out available")
def test_a_window_over_the_whole_run_holds_every_state():
    pytest.importorskip("CRYSTALClear")

    run, _out = vci_run(_VCI_OUTS[0])

    assert run.count_in(run.energies[0], run.energies[-1]) == run.nstates


@pytest.mark.skipif(not _VCI_OUTS, reason="no sample VCI .out available")
def test_the_recovered_modes_are_real_phonon_indices():
    """A CO2 molecule's VCI runs on its internal modes, so the indices have to
    sit above the three translations rather than starting at 1."""
    pytest.importorskip("CRYSTALClear")

    run, _out = vci_run(_VCI_OUTS[0])

    assert run.modes, "the PES scan should name the coupled modes"
    assert min(run.modes) > 3


@pytest.mark.skipif(not _VCI_OUTS, reason="no sample VCI .out available")
def test_both_representations_draw_a_figure():
    pytest.importorskip("CRYSTALClear")
    from matplotlib.figure import Figure

    from crystalline.crystalio.vci import plot_vci, plottable

    if not plottable():
        pytest.skip("the installed CRYSTALClear cannot draw VCI states")
    run, out = vci_run(_VCI_OUTS[0])

    for _label, key in REPRESENTATIONS:
        figure = plot_vci(out, representation=key,
                          frange=run.default_window(4), threshold=0.05)
        assert isinstance(figure, Figure)


@pytest.mark.skipif(not _VCI_OUTS, reason="no sample VCI .out available")
def test_a_window_holding_no_state_is_refused():
    """Rather than drawing an empty figure — the window is simply wrong."""
    pytest.importorskip("CRYSTALClear")

    from crystalline.crystalio.vci import plot_vci, plottable

    if not plottable():
        pytest.skip("the installed CRYSTALClear cannot draw VCI states")
    run, out = vci_run(_VCI_OUTS[0])
    above_everything = run.energies[-1] + 1000.0

    with pytest.raises(ValueError):
        plot_vci(out, frange=(above_everything, above_everything + 100.0))


@pytest.mark.skipif(not _VCI_OUTS, reason="no sample VCI .out available")
def test_too_wide_a_window_is_refused_rather_than_drawn():
    """Hundreds of columns is not a plot of anything, so the guard fires before
    the figure is built."""
    pytest.importorskip("CRYSTALClear")

    from crystalline.crystalio.vci import plot_vci, plottable

    if not plottable():
        pytest.skip("the installed CRYSTALClear cannot draw VCI states")
    run, out = vci_run(_VCI_OUTS[0])

    with pytest.raises(ValueError, match="max_states"):
        plot_vci(out, frange=(run.energies[0], run.energies[-1]), max_states=3)


@pytest.mark.skipif(not _VCI_OUTS, reason="no sample VCI .out available")
def test_an_unknown_representation_is_refused():
    pytest.importorskip("CRYSTALClear")
    from crystalline.crystalio.vci import plot_vci

    _run, out = vci_run(_VCI_OUTS[0])

    with pytest.raises(ValueError):
        plot_vci(out, representation="ribbons")


def test_a_file_without_vci_yields_nothing(tmp_path):
    """load_vci swallows the parse failure rather than raising: an output with
    no VCI block is the normal case, not an error."""
    path = tmp_path / "run.out"
    path.write_text(" SOME UNRELATED CRYSTAL OUTPUT\n")

    assert load_vci(str(path)) is None
