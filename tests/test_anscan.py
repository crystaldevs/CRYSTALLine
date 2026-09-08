"""Reading an anharmonic scan, and routing it to the plot.

Three things are worth pinning. The cheap probe that enables the menu entry has
to agree with the real parse, or the entry lies about what a file holds. The
wavefunction file is found beside the output rather than asked for, and getting
that wrong means drawing one run's states on another run's potential — nothing
downstream can catch it. And the wavefunction height is derived from the level
spacing, which is the whole reason the plot needs no hand-tuning: the two runs
at hand want factors of 500 and 30, and neither is guessable.
"""

import os

import pytest

from crystalline.crystalio.anscan import (
    AnscanRun,
    anscan_run,
    find_wavefunctions,
    has_anscan,
    _mode_line,
)

_BANNER = "SCAN ALONG NORMAL MODES"
_STATES = "ANHARMONIC VIBRATIONAL STATES"


def _run(**kwargs):
    defaults = dict(mode=7, frequency=487.11, nstates=101, nwf=10,
                    rangescan=(-9.0, 9.0), spacing=530.0, double_well=False)
    defaults.update(kwargs)
    return AnscanRun(**defaults)


# ── the cheap probe ─────────────────────────────────────────────────────
def test_a_file_with_the_banner_and_the_states_carries_a_scan(tmp_path):
    path = tmp_path / "run.out"
    path.write_text(f" {_BANNER}\n STARTING POINT: -10 ENDING POINT: 10 STEP: 0.9\n"
                    f" {_STATES}\n   ZPE   0.258E+03  257.75\n")

    assert has_anscan(str(path))


def test_a_run_that_scanned_but_solved_nothing_does_not_count(tmp_path):
    """A calculation can walk the mode and be cut short before it diagonalises;
    offering the plot for it would open a dialog with no levels behind it."""
    path = tmp_path / "run.out"
    path.write_text(f" {_BANNER}\n [DISPLAC] [ SCAN POTENTIAL ]\n -9.0  0.16E+00  36837.92\n")

    assert not has_anscan(str(path))


def test_a_harmonic_run_does_not_count(tmp_path):
    path = tmp_path / "run.out"
    path.write_text(" FREQUENCIES COMPUTED ON A FRAGMENT\n MODES  1-  3\n")

    assert not has_anscan(str(path))


@pytest.mark.parametrize("path", [None, "", "/nonexistent/run.out"])
def test_no_path_and_no_file_are_not_errors(path):
    assert not has_anscan(path)


def test_a_missing_file_yields_no_run():
    assert anscan_run("/nonexistent/run.out", "/nonexistent/ANSCANWF.DAT") == (None, None)


def test_a_run_without_wavefunctions_yields_nothing(tmp_path):
    """The coefficients are the point of the plot, so an output on its own is
    not enough however complete it is."""
    assert anscan_run(str(tmp_path / "run.out"), None) == (None, None)


# ── finding ANSCANWF.DAT ────────────────────────────────────────────────
def test_the_wavefunctions_are_found_by_the_output_stem(tmp_path):
    (tmp_path / "co2.out").write_text("")
    wf = tmp_path / "co2.anscanwf"
    wf.write_text("")

    assert find_wavefunctions(str(tmp_path / "co2.out")) == str(wf)


def test_crystals_own_name_is_found_too(tmp_path):
    """A run left as CRYSTAL wrote it has ANSCANWF.DAT sitting next to the output."""
    (tmp_path / "co2.out").write_text("")
    wf = tmp_path / "ANSCANWF.DAT"
    wf.write_text("")

    assert find_wavefunctions(str(tmp_path / "co2.out")) == str(wf)


def test_the_stem_wins_over_crystals_name(tmp_path):
    """Both present means a renamed run alongside a fresh one; the file named
    after this output is the one that belongs to it."""
    (tmp_path / "co2.out").write_text("")
    (tmp_path / "ANSCANWF.DAT").write_text("")
    stem = tmp_path / "co2.anscanwf"
    stem.write_text("")

    assert find_wavefunctions(str(tmp_path / "co2.out")) == str(stem)


def test_a_lone_wavefunction_file_is_taken(tmp_path):
    (tmp_path / "co2.out").write_text("")
    wf = tmp_path / "something_else.anscanwf"
    wf.write_text("")

    assert find_wavefunctions(str(tmp_path / "co2.out")) == str(wf)


def test_two_candidates_are_refused_rather_than_guessed(tmp_path):
    """Two runs side by side: picking either would draw one run's states in the
    other's potential, and the figure would look perfectly plausible."""
    (tmp_path / "co2.out").write_text("")
    (tmp_path / "run_a.anscanwf").write_text("")
    (tmp_path / "run_b.anscanwf").write_text("")

    assert find_wavefunctions(str(tmp_path / "co2.out")) is None


def test_nothing_beside_the_output_yields_nothing(tmp_path):
    (tmp_path / "co2.out").write_text("")

    assert find_wavefunctions(str(tmp_path / "co2.out")) is None


# ── what the output says about the scan ─────────────────────────────────
def test_the_mode_and_its_frequency_come_off_the_header(tmp_path):
    path = tmp_path / "run.out"
    path.write_text(" MODE(CM**-1)     7( 487.1)\n"
                    " [DISPLAC] [       SCAN POTENTIAL       ]\n")

    assert _mode_line(str(path)) == (7, 487.1, False)


def test_an_imaginary_mode_keeps_its_sign(tmp_path):
    """CRYSTAL prints an imaginary frequency as negative, and the harmonic
    parabola drawn from it has to open downwards."""
    path = tmp_path / "run.out"
    path.write_text(" MODE(CM**-1)     1( -30.0)\n")

    mode, frequency, _dwell = _mode_line(str(path))
    assert (mode, frequency) == (1, -30.0)
    assert _run(frequency=frequency).imaginary


def test_a_double_well_is_noticed(tmp_path):
    path = tmp_path / "run.out"
    path.write_text(" MODE(CM**-1)     1( -30.0)\n"
                    " INFORMATION **** init_dwell **** DOUBLE-WELL POTENTIAL DETECTED\n")

    assert _mode_line(str(path))[2] is True


def test_a_file_that_cannot_be_read_is_not_an_error():
    assert _mode_line("/nonexistent/run.out") == (0, 0.0, False)


# ── the description the dialog is built from ────────────────────────────
def test_the_summary_reads_as_a_description_of_the_run():
    summary = _run().summary

    assert "Mode 7" in summary and "487.1 cm⁻¹" in summary
    assert "101 levels" in summary and "10 with a wavefunction" in summary
    assert "double well" not in summary


def test_an_imaginary_mode_is_written_as_such():
    summary = _run(mode=1, frequency=-30.03, double_well=True).summary

    assert "30.0i cm⁻¹" in summary
    assert "double well" in summary


def test_the_wavefunction_height_follows_the_level_spacing():
    """The states are normalised, so the only scale that means anything is the
    gap between the levels they are drawn on. A stiff mode and a soft one want
    heights an order of magnitude apart, which is why this is not a constant."""
    stiff = _run(spacing=530.0)
    soft = _run(spacing=39.4, frequency=-30.03)

    assert stiff.scale_wf == pytest.approx(530.0)
    assert soft.scale_wf == pytest.approx(39.4)
    assert stiff.scale_prob == pytest.approx(2 * stiff.scale_wf)


def test_a_run_with_a_single_level_still_gives_a_usable_height():
    """No gaps to average means no spacing; a zero height would draw nothing."""
    assert _run(nstates=1, nwf=1, spacing=0.0).scale_wf > 0


# ── against a real run, when one is at hand ─────────────────────────────
_REAL = os.path.expanduser(
    "~/Desktop/PyCrystal/anharmonic_freq/newanscan/new_anscan_test/"
    "co2_STO3G_newanscan_mode7"
)


@pytest.mark.skipif(not os.path.isfile(_REAL + ".out"),
                    reason="the reference ANSCAN run is not on this machine")
def test_a_real_run_is_described_and_drawn():
    pytest.importorskip("CRYSTALClear")
    from crystalline.crystalio.anscan import plot_anscan

    assert has_anscan(_REAL + ".out")
    assert find_wavefunctions(_REAL + ".out") == _REAL + ".anscanwf"

    run, out = anscan_run(_REAL + ".out", _REAL + ".anscanwf")
    assert run.mode == 7
    assert run.frequency == pytest.approx(487.11, abs=0.05)
    assert run.nwf == 10
    assert run.rangescan == (-9.0, 9.0)
    # The fundamental of this mode, which is what the height should be worth.
    assert run.scale_wf == pytest.approx(555.0, rel=0.1)

    figure = plot_anscan(out, scale_wf=run.scale_wf, harmpot=True, nstates=4)
    assert figure.axes and figure.axes[0].get_xlabel()
